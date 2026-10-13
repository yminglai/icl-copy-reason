"""Scoring variants and post-hoc checks of the two readings: three margin rules, the single-token subset, float32 re-scoring,
pooled summaries.

  rescore            the SAME items, spans and forward passes as the attention ablation, keeping per-token log-probabilities so the
                     correct-vs-copied margin (hence A_null and the balance D = R + B) is computed under three rules: sum (summed
                     sequence log-probability, the paper's measure), len (length-normalised) and div (the first token at which the two
                     forms differ). Conditions: none, competitor_marker, wordB. Schema of the released variants_<model>_s<seed>_rescored[_fp32].json.
  single_token_subset  re-reads ablation files, regenerates the items, and reports A_null / D on all items and on the subset where both
                     second endings are single tokens (schema of singletok_finals.json / singletok_traj_<model>.json).
  summarise          pooled A_null / D under each rule, item-level correlations between rules, sign-change intervals on checkpoint grids
                     (schema of variants_summary.json).
  fp32_compare       float32 vs bfloat16 scoring (schema of fp32_check.json).
  fp32_body          the pooled float32 held-out-item values used by the body (schema of fp32_body.json).
  pooled_delta / pooled_anull / mean_ci   seed-pooled paired effects with item-bootstrap intervals (default_rng(0), 2,000 resamples).
  qwen05_summary     the Qwen2.5-0.5B class-vs-family check (schema of qwen05_confound.json; the table is written by analysis/).

Inputs : result files of the attention ablation / rescoring runs. Outputs: JSON summaries at the given paths.
"""
from __future__ import annotations

import gc
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from ..io import read_json, write_json
from ..tasks.prompts import Lang, build, four_forms, slots_of
from .ablation import line_sets
from .copy_share import load_model, revision_key
from .lm import LM

CONDS = ["none", "competitor_marker", "wordB"]


RECEIVERS = "query"     # "query": every query-line + continuation token stops attending to the span (the paper's convention);
                        # "post_span": only tokens that START AFTER the span's end do, so a knocked-out query word still attends to itself
                        # (source-preserving variant; identical to "query" for demonstration-line spans)


def token_logprobs(lm, prompts, conts, spans_list, qstarts):
    """Per-item list of (token_logprob, token_char_start) for the continuation tokens, under the knockout mask (same mask as morph_knockout)."""
    full = [p + " " + c for p, c in zip(prompts, conts)]
    enc = lm.tok(full, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    input_ids = enc["input_ids"].to(lm.device)
    am = enc["attention_mask"].to(lm.device)
    B, T = input_ids.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=lm.device))
    mask = causal[None].repeat(B, 1, 1) & am[:, None, :].bool()
    for i in range(B):
        if not spans_list[i]:
            continue
        toks = offs[i]
        row_from = qstarts[i] if RECEIVERS == "query" else max(qstarts[i], max(e for _, e in spans_list[i]))
        q_rows = [t for t in range(T) if int(toks[t, 1]) > 0 and int(toks[t, 0]) >= row_from]
        k_cols = [t for t in range(T) if int(toks[t, 1]) > 0 and any(int(toks[t, 0]) < e and int(toks[t, 1]) > s for s, e in spans_list[i])]
        if q_rows and k_cols:
            rows = torch.tensor(q_rows, device=lm.device)
            cols = torch.tensor(k_cols, device=lm.device)
            mask[i][rows[:, None], cols[None, :]] = False
    dt = next(lm.model.parameters()).dtype
    add = torch.zeros(B, 1, T, T, dtype=dt, device=lm.device)
    add.masked_fill_(~mask[:, None], torch.finfo(dt).min)
    with torch.no_grad():
        logits = lm.model(input_ids=input_ids, attention_mask=add).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(2, input_ids[:, 1:, None]).squeeze(-1).cpu()
    out = []
    for i, p in enumerate(prompts):
        start = len(p)
        toks = [(float(lp[i, t]), int(offs[i, t + 1, 0]), int(input_ids[i, t + 1])) for t in range(lp.shape[1]) if int(offs[i, t + 1, 0]) >= start and int(offs[i, t + 1, 1]) > 0]
        out.append(toks)
    return out


def scores(tok_corr, tok_comp):
    """Three margins (correct minus competitor) from two per-token lists [(logprob, char_start, token_id), ...]."""
    s_sum = sum(t[0] for t in tok_corr) - sum(t[0] for t in tok_comp)
    s_len = sum(t[0] for t in tok_corr) / max(1, len(tok_corr)) - sum(t[0] for t in tok_comp) / max(1, len(tok_comp))
    k = 0
    while k < min(len(tok_corr), len(tok_comp)) and tok_corr[k][2] == tok_comp[k][2]:
        k += 1
    if k < min(len(tok_corr), len(tok_comp)):
        s_div = tok_corr[k][0] - tok_comp[k][0]
    else:                                   # one form is a prefix of the other (should not happen for distinct markers)
        s_div = s_sum
    return {"sum": s_sum, "len": s_len, "div": s_div, "n_corr": len(tok_corr), "n_comp": len(tok_comp), "k_div": k}


def rescore_revision(lm: LM, items, bs: int = 8, receivers: str = "query") -> dict:
    """Three-rule margins for every item under the three conditions; returns the per-revision record."""
    global RECEIVERS
    RECEIVERS = receivers
    lm.model.config._attn_implementation = "eager"
    lm.text_cfg._attn_implementation = "eager"
    fingerprint = lm.fingerprint()  # taken from the loaded weights before any forward pass
    per = {c: [] for c in CONDS}
    for b in range(0, len(items), bs):
        bp = items[b:b + bs]
        spans_all = [line_sets(it) for it in bp]
        for c in CONDS:
            prompts, conts, tags, spans_l, qs = [], [], [], [], []
            for it, (spans, qstart) in zip(bp, spans_all):
                first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
                comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
                for bundle, form in four_forms(it):
                    if bundle in ((it.ia, it.ib), comp):
                        prompts.append(it.prompt)
                        conts.append(form)
                        tags.append(bundle)
                        spans_l.append(None if c == "none" else spans[c])
                        qs.append(qstart)
            tl = token_logprobs(lm, prompts, conts, spans_l, qs)
            for j, it in enumerate(bp):
                first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
                comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
                cand = dict(zip(tags[2 * j:2 * j + 2], tl[2 * j:2 * j + 2]))
                per[c].append(scores(cand[(it.ia, it.ib)], cand[comp]))
    rec = {"n": len(items), "receivers": RECEIVERS, "fingerprint": fingerprint}
    for c in CONDS:
        rec[c] = {"items": per[c]}
        for s in ("sum", "len", "div"):
            m = np.array([x[s] for x in per[c]])
            rec[c][f"margin_{s}"] = round(float(m.mean()), 4)
            rec[c][f"a_null_{s}"] = round(float((1 / (1 + np.exp(m))).mean()), 4)
    for s in ("sum", "len", "div"):
        D = np.array([x[s] for x in per["competitor_marker"]]) + np.array([x[s] for x in per["wordB"]]) - 2 * np.array([x[s] for x in per["none"]])
        rec[f"D_{s}"] = round(float(D.mean()), 4)
    rec["share_equal_length"] = round(float(np.mean([x["n_corr"] == x["n_comp"] for x in per["none"]])), 3)
    return rec


def rescore(model: str, steps: str = "main", n: int = 160, seed: int = 2, tag: str = "_rescored", bs: int = 8, fp32: bool = False,
            receivers: str = "query", out_dir: str | Path = ".", device: str | None = None) -> Path:
    """Three-rule re-scoring of `model` at every revision in `steps` (items = build("compose_h", n, seed + 11)); resumable.
    Returns <out_dir>/variants_<model>_s<seed><tag>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mtag = model.split("/")[-1]
    outp = out_dir / f"variants_{mtag}_s{seed}{tag}.json"
    res = read_json(outp) if outp.exists() else {}
    for step in steps.split(","):
        rev = revision_key(step)
        key = rev or "main"
        if key in res:
            continue
        t0 = time.time()
        lm = load_model(model, revision=rev, dtype="float32" if fp32 else "bfloat16", device=device)
        items = build("compose_h", n, seed=seed + 11)
        rec = rescore_revision(lm, items, bs, receivers)
        rec["dtype"] = "float32" if fp32 else "bfloat16"
        res[key] = rec
        print(f"{mtag} {key}: A_null sum {rec['none']['a_null_sum']:.3f} len {rec['none']['a_null_len']:.3f} div {rec['none']['a_null_div']:.3f} | "
              f"D sum {rec['D_sum']:+.2f} len {rec['D_len']:+.2f} div {rec['D_div']:+.2f} | equal-length share {rec['share_equal_length']:.2f} [{time.time() - t0:.0f}s]", flush=True)
        write_json(res, outp)
        lm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outp


# ---------------------------------------------------------------- pooled effects (shared by the summaries and by analysis/)
_RNG = np.random.default_rng(0)


def mean_ci(d, n_boot: int = 2000, rng=None) -> list[float]:
    """Mean and 95% item-bootstrap interval of a vector of paired effects (numpy default_rng(0) unless given)."""
    rng = _RNG if rng is None else rng
    d = np.asarray(d, float)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def pooled_delta(recs, cond: str, n_boot: int = 2000):
    """Pooled (over seeds) mean paired delta-margin for `cond` against the unmasked pass and its item-bootstrap 95% CI."""
    d = []
    for r in recs:
        if cond not in r or "items" not in r[cond]:
            continue
        base = [it["margin"] for it in r["none"]["items"]]
        ko = [it["margin"] for it in r[cond]["items"]]
        d += [a - b for a, b in zip(ko, base)]
    if not d:
        return None
    return tuple(mean_ci(np.array(d), n_boot))


def pooled_anull(recs) -> float:
    v = [r["none"]["a_null"] for r in recs if "none" in r]
    return float(np.mean(v)) if v else float("nan")


def _tokens_of(key: str) -> int | None:
    m = re.search(r"tokens(\d+)B", key)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- single-token subset
HF_NAMES = {"OLMo-2-0425-1B": "allenai/OLMo-2-0425-1B", "OLMo-2-1124-7B": "allenai/OLMo-2-1124-7B", "OLMo-2-1124-13B": "allenai/OLMo-2-1124-13B",
            "Qwen3-0.6B-Base": "Qwen/Qwen3-0.6B-Base", "Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B-Base", "Qwen3-4B-Base": "Qwen/Qwen3-4B-Base",
            "Qwen3-8B-Base": "Qwen/Qwen3-8B-Base", "Qwen2.5-7B": "Qwen/Qwen2.5-7B", "pythia-1.4b": "EleutherAI/pythia-1.4b",
            "pythia-2.8b": "EleutherAI/pythia-2.8b", "pythia-12b": "EleutherAI/pythia-12b", "gemma-3-4b-pt": "unsloth/gemma-3-4b-pt"}
SINGLE_TOKEN_FINALS = ["OLMo-2-0425-1B", "OLMo-2-1124-7B", "OLMo-2-1124-13B", "Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-8B-Base", "Qwen2.5-7B",
                       "pythia-1.4b", "pythia-2.8b", "pythia-12b", "gemma-3-4b-pt"]


def second_marker_single(tok, it) -> bool:
    """True if, in the full prompt + form context, the SECOND ending of both the correct and the copied form is exactly one token."""
    lang = Lang(**it.lang)
    sl = slots_of(it.lang, it.ia, it.ib)
    first_f = sl[0][0]
    comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
    out = []
    for (a, b) in [(it.ia, it.ib), comp]:
        form = lang.form(it.stem, a, b)
        text = it.prompt + " " + form
        second = lang.slots(a, b)[1][1]
        start, end = len(text) - len(second), len(text)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        n = sum(1 for s, e in enc["offset_mapping"] if s < end and e > start and e > 0)
        out.append(n == 1)
    return all(out)


def _per_item(rec):
    r = rec["main"] if "main" in rec else rec
    an = np.array([i["a_null"] for i in r["none"]["items"]])
    base = np.array([i["margin"] for i in r["none"]["items"]])
    D = np.array([i["margin"] for i in r["competitor_marker"]["items"]]) + np.array([i["margin"] for i in r["wordB"]["items"]]) - 2 * base
    return an, D


def _analyse_single(model: str, files_by_seed: dict, keys: list[str]) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_NAMES.get(model, model))
    res = {}
    for key in keys:
        an_all, D_all, mask_all = [], [], []
        for seed, f in files_by_seed.items():
            d = read_json(f)
            if key not in d:
                continue
            an, D = _per_item(d[key])
            items = build("compose_h", len(an), seed=seed + 11)
            m = np.array([second_marker_single(tok, it) for it in items])
            an_all.append(an)
            D_all.append(D)
            mask_all.append(m)
        if not an_all:
            continue
        an, D, m = np.concatenate(an_all), np.concatenate(D_all), np.concatenate(mask_all)
        res[key] = {"n": int(len(an)), "share_single": float(m.mean()), "a_null_all": mean_ci(an), "D_all": mean_ci(D),
                    "a_null_single": mean_ci(an[m]) if m.sum() >= 20 else None, "D_single": mean_ci(D[m]) if m.sum() >= 20 else None,
                    "a_null_split": mean_ci(an[~m]) if (~m).sum() >= 20 else None, "D_split": mean_ci(D[~m]) if (~m).sum() >= 20 else None}
    return res


def single_token_subset(results_dir: str | Path, out_path: str | Path, mode: str = "finals", traj_model: str = "OLMo-2-1124-7B",
                        traj_tag: str = "_conftraj", finals_tag: str = "_conf", seeds=(2, 3)) -> Path:
    """mode 'finals': the released models' ablation files knockout_<model>_s<seed><finals_tag>.json; mode 'traj': the checkpoint grid of traj_model."""
    results_dir = Path(results_dir)
    if mode == "finals":
        out = {}
        for model in SINGLE_TOKEN_FINALS:
            files = {s: results_dir / f"knockout_{model}_s{s}{finals_tag}.json" for s in seeds if (results_dir / f"knockout_{model}_s{s}{finals_tag}.json").exists()}
            if not files:
                continue
            key = "step143000" if model.startswith("pythia") else "main"
            r = _analyse_single(model, files, [key])
            if r:
                out[model] = r[key]
        return write_json(out, out_path)
    files = {s: results_dir / f"knockout_{traj_model}_s{s}{traj_tag}.json" for s in seeds}
    keys = sorted({k for f in files.values() for k in read_json(f) if "tokens" in k}, key=_tokens_of)
    return write_json(_analyse_single(traj_model, files, keys), out_path)


# ---------------------------------------------------------------- pooled summary of the three rules
RULES = ("sum", "len", "div")
SUMMARY_FINALS = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base", "Qwen3-8B-Base", "Qwen2.5-7B", "OLMo-2-0425-1B", "OLMo-2-1124-7B", "OLMo-2-1124-13B"]


def _pooled_rules(ds, key):
    out = {}
    recs = [d[key] for d in ds if key in d]
    if len(recs) < 2:
        return None
    for s in RULES:
        base = np.concatenate([[x[s] for x in r["none"]["items"]] for r in recs])
        D = np.concatenate([[a[s] + b[s] - 2 * c[s] for a, b, c in zip(r["competitor_marker"]["items"], r["wordB"]["items"], r["none"]["items"])] for r in recs])
        rng = np.random.default_rng(0)
        bs = np.array([D[rng.integers(0, len(D), len(D))].mean() for _ in range(2000)])
        out[s] = {"a_null": float((1 / (1 + np.exp(base))).mean()), "margin": float(base.mean()), "D": float(D.mean()),
                  "D_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "_base": base, "_D": D}
    for s in ("len", "div"):
        out[f"r_margin_sum_{s}"] = float(np.corrcoef(out["sum"]["_base"], out[s]["_base"])[0, 1])
        out[f"r_D_sum_{s}"] = float(np.corrcoef(out["sum"]["_D"], out[s]["_D"])[0, 1])
        out[f"item_sign_agreement_D_sum_{s}"] = float(np.mean(np.sign(out["sum"]["_D"]) == np.sign(out[s]["_D"])))
    for s in RULES:
        out[s].pop("_base")
        out[s].pop("_D")
    out["share_equal_length"] = float(np.mean([r["share_equal_length"] for r in recs]))
    return out


def summarise(results_dir: str | Path, out_path: str | Path, finals=None, traj_models=("OLMo-2-1124-7B", "OLMo-2-1124-13B"), tag: str = "_rescored") -> Path:
    """Pooled summary of variants_<model>_s[23]<tag>.json files (schema of variants_summary.json)."""
    results_dir = Path(results_dir)

    def load(model):
        return [read_json(f) for f in sorted(glob.glob(str(results_dir / f"variants_{model}_s[23]{tag}.json")))]

    res = {"finals": {}, "traj": {}}
    for m in finals or SUMMARY_FINALS:
        ds = load(m)
        if ds:
            p = _pooled_rules(ds, "main")
            if p:
                res["finals"][m] = p
    for m in traj_models:
        ds = load(m)
        if not ds:
            continue
        keys = sorted({k for d in ds for k in d if _tokens_of(k)}, key=_tokens_of)
        rows = []
        for k in keys:
            p = _pooled_rules(ds, k)
            if p:
                rows.append({"tokens": _tokens_of(k), **{s: {"a_null": p[s]["a_null"], "D": p[s]["D"], "D_ci": p[s]["D_ci"]} for s in RULES}})
        if rows:
            sc = {}
            for s in RULES:
                D = [r[s]["D"] for r in rows]
                lo = [r[s]["D_ci"][0] for r in rows]
                hi = [r[s]["D_ci"][1] for r in rows]
                first_neg = next((i for i, v in enumerate(D) if v < 0), None)
                last_ci_pos = max([i for i, v in enumerate(lo) if v > 0], default=None)
                first_ci_neg = next((i for i, v in enumerate(hi) if v < 0), None)
                sc[s] = {"first_sign_change": [rows[first_neg - 1]["tokens"], rows[first_neg]["tokens"]] if first_neg else None,
                         "last_ci_above_zero": rows[last_ci_pos]["tokens"] if last_ci_pos is not None else None,
                         "first_ci_below_zero": rows[first_ci_neg]["tokens"] if first_ci_neg is not None else None,
                         "n_sign_changes": int(sum(1 for i in range(1, len(D)) if (D[i] < 0) != (D[i - 1] < 0)))}
            res["traj"][m] = {"rows": rows, "sign_change": sc}
    return write_json(res, out_path)


# ---------------------------------------------------------------- float32 vs bfloat16
def _items(rec, cond):
    return np.array([x["sum"] if "sum" in x else x["margin"] for x in rec[cond]["items"]])


def _D_items(rec):
    return _items(rec, "competitor_marker") + _items(rec, "wordB") - 2 * _items(rec, "none")


def _bf16_ref(results_dir: Path, model, seed, key, bf16_tag="_rescored", ablation_tag="_conf"):
    p = results_dir / f"variants_{model}_s{seed}{bf16_tag}.json"
    if p.exists() and key in read_json(p):
        return read_json(p)[key]
    cand = results_dir / f"knockout_{model}_s{seed}{ablation_tag}.json"
    if cand.exists():
        d = read_json(cand)
        for k in ([key, "main"] if key == "step143000" else ([key, "step143000"] if key == "main" else [key])) + ([list(d)[0]] if len(d) == 1 else []):
            if k in d and all(c in d[k] for c in ("none", "competitor_marker", "wordB")) and len(d[k]["none"]["items"]) == 160:
                return d[k]
    return None


def fp32_compare(results_dir: str | Path, out_path: str | Path, fp32_tag: str = "_rescored_fp32") -> Path:
    """float32 vs bfloat16 scoring per model x seed x revision (schema of fp32_check.json)."""
    results_dir = Path(results_dir)
    out = {"note": "post-hoc precision check: float32 re-scoring against the bfloat16 run on the same items and masks", "rows": []}
    for f in sorted(glob.glob(str(results_dir / f"variants_*{fp32_tag}.json"))):
        base = os.path.basename(f)
        model = base[len("variants_"):].rsplit("_s", 1)[0]
        seed = int(base.rsplit("_s", 1)[1][0])
        for key, rec in read_json(f).items():
            ref = _bf16_ref(results_dir, model, seed, key)
            if ref is None:
                out["rows"].append({"model": model, "seed": seed, "key": key, "bf16": None})
                continue
            m32, m16, D32, D16 = _items(rec, "none"), _items(ref, "none"), _D_items(rec), _D_items(ref)
            out["rows"].append({"model": model, "seed": seed, "key": "main" if key == "step143000" else key,
                                "a_null_bf16": float((1 / (1 + np.exp(m16))).mean()), "a_null_fp32": float((1 / (1 + np.exp(m32))).mean()),
                                "D_bf16": float(D16.mean()), "D_fp32": float(D32.mean()),
                                "comp_bf16": float((_items(ref, "competitor_marker") - m16).mean()), "comp_fp32": float((_items(rec, "competitor_marker") - m32).mean()),
                                "word_bf16": float((_items(ref, "wordB") - m16).mean()), "word_fp32": float((_items(rec, "wordB") - m32).mean()),
                                "margin_absdiff_mean": float(np.abs(m32 - m16).mean()), "margin_absdiff_max": float(np.abs(m32 - m16).max()),
                                "D_absdiff_mean": float(np.abs(D32 - D16).mean()), "D_absdiff_max": float(np.abs(D32 - D16).max()),
                                "D_sign_agreement": float(np.mean(np.sign(D32) == np.sign(D16)))})
    return write_json(out, out_path)


# ---------------------------------------------------------------- the pooled float32 values of the body
FP32_MODELS = ["pythia-1.4b", "pythia-2.8b", "pythia-12b", "OLMo-2-0425-1B", "Qwen3-0.6B-Base", "OLMo-2-1124-7B", "OLMo-2-1124-13B",
               "Qwen3-1.7B-Base", "gemma-3-4b-pt", "Qwen2.5-7B", "Qwen3-4B-Base", "Qwen3-8B-Base"]


def fp32_records(results_dir: Path, model: str, seeds=("2", "3"), tag: str = "_rescored_fp32") -> list[dict]:
    out = []
    for s in seeds:
        f = results_dir / f"variants_{model}_s{s}{tag}.json"
        if f.exists():
            out.append(read_json(f))
    return out


def _fp32_items(recs, key, cond):
    d = []
    for r in recs:
        if key not in r or cond not in r[key]:
            continue
        base = np.array([it["sum"] for it in r[key]["none"]["items"]])
        ko = np.array([it["sum"] for it in r[key][cond]["items"]])
        d.append(ko - base)
    return np.concatenate(d) if d else None


def fp32_finals(results_dir: str | Path, models=None) -> list[dict]:
    """Per released model: A_null, R (copied candidate's endings), the queried-word effect B and D = R + B from the float32 re-scoring,
    pooled over item seeds 2 and 3 (the 'finals' block of fp32_body.json)."""
    results_dir = Path(results_dir)
    rows = []
    for m in models or FP32_MODELS:
        recs = fp32_records(results_dir, m)
        key = "step143000" if m.startswith("pythia") else "main"
        recs = [r if key in r else {key: r["main"]} for r in recs if key in r or "main" in r]
        an = [np.mean([1 / (1 + np.exp(it["sum"])) for it in r[key]["none"]["items"]]) for r in recs]
        an_ko = [np.mean([1 / (1 + np.exp(it["sum"])) for it in r[key]["competitor_marker"]["items"]]) for r in recs if "competitor_marker" in r[key]]
        R = _fp32_items(recs, key, "competitor_marker")
        W = _fp32_items(recs, key, "wordB")
        rows.append(dict(model=m, nseeds=len(an), a_null=float(np.mean(an)) if an else float("nan"), a_null_under_marker_knockout=float(np.mean(an_ko)) if an_ko else None,
                         competitor_marker=None if R is None else tuple(mean_ci(R)), wordB=None if W is None else tuple(mean_ci(W)),
                         D=None if (R is None or W is None) else tuple(mean_ci(R + W))))
    return rows


def fp32_trajectory(results_dir: str | Path, model: str) -> list[dict]:
    """Checkpoint series from the float32 re-scoring: tokens (B), A_null, R, B (sign-flipped word effect), D (the 'trajectories' block)."""
    recs = fp32_records(Path(results_dir), model)
    keys = sorted([k for k in recs[0] if _tokens_of(k) and all(k in r for r in recs)], key=_tokens_of)

    def series(f):
        return np.array([float(np.mean(np.concatenate([f(r[k]) for r in recs]))) for k in keys])

    an = series(lambda v: np.array([1 / (1 + np.exp(it["sum"])) for it in v["none"]["items"]]))
    comp = series(lambda v: np.array([a["sum"] - b["sum"] for a, b in zip(v["competitor_marker"]["items"], v["none"]["items"])]))
    word = series(lambda v: np.array([a["sum"] - b["sum"] for a, b in zip(v["wordB"]["items"], v["none"]["items"])]))
    return [dict(tokens_B=int(k_), a_null=float(a), R=float(c), B=float(-w), D=float(c + w)) for k_, a, c, w in zip([_tokens_of(k) for k in keys], an, comp, word)]


def fp32_body(results_dir: str | Path, out_path: str | Path, traj_models=("OLMo-2-1124-7B",)) -> Path:
    out = {"note": "pooled over item seeds 2 and 3 (320 items) from variants_*_rescored_fp32.json; mean and 95% item-bootstrap interval",
           "finals": fp32_finals(results_dir), "trajectories": {m: fp32_trajectory(results_dir, m) for m in traj_models}}
    return write_json(out, out_path)


# ---------------------------------------------------------------- Qwen2.5-0.5B class-vs-family check
QWEN05_SPANS = ["competitor_marker", "wordB", "wordA", "sham", "competitor", "single2", "single2_marker", "otherbundles"]


def qwen05_summary(results_dir: str | Path, out_path: str | Path, model: str = "Qwen2.5-0.5B", seeds=(2, 3)) -> Path:
    """Pooled fresh-item ablations in float32 (_fp32) and bfloat16 (_conf) plus the layer-4 word patch (seeds 0/1); schema of qwen05_confound.json."""
    results_dir = Path(results_dir)
    out = {"model": f"Qwen/{model}", "item_seeds": list(seeds), "generator_seeds": [s + 11 for s in seeds], "n_items": 160 * len(seeds), "per_dtype": {}}
    for tag, dt in (("_fp32", "float32"), ("_conf", "bfloat16")):
        recs = [read_json(results_dir / f"knockout_{model}_s{s}{tag}.json")["main"] for s in seeds]
        row = {"a_null": pooled_anull(recs), "base_margin": float(np.mean([r["none"]["margin"] for r in recs])), "four_way_acc": float(np.mean([r["none"]["four_way_acc"] for r in recs]))}
        for sp in QWEN05_SPANS:
            row[sp] = pooled_delta(recs, sp)
        d = np.concatenate([np.array([it["margin"] for it in r["competitor_marker"]["items"]]) + np.array([it["margin"] for it in r["wordB"]["items"]]) - 2 * np.array([it["margin"] for it in r["none"]["items"]]) for r in recs])
        row["D"] = mean_ci(d, rng=np.random.default_rng(0))
        row["D_item_sign_pos"] = float((d > 0).mean())
        row["a_null_under_marker_knockout"] = float(np.mean([r["competitor_marker"]["a_null"] for r in recs]))
        out["per_dtype"][dt] = row
    out["patch_l4"] = {f"s{s}": read_json(results_dir / f"patch_sweep_{model}_s{s}_l4strict.json") for s in (0, 1) if (results_dir / f"patch_sweep_{model}_s{s}_l4strict.json").exists()}
    return write_json(out, out_path)

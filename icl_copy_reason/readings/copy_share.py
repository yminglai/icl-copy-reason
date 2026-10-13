"""Copy-probability share and the other behavioural measures of one model at one revision (the first reading).

Per model x revision: greedy generation (max 8 new tokens) on bind_a / bind_b / compose_s / compose_h items
scores copy (stem
reproduced), per-ending correctness, exact form, demonstration-stem reuse and the generation taxonomy; a forced-choice ending score
with the stem teacher-forced
for the two compose tasks the four-form sequence likelihood (four-way accuracy, copied-candidate rate,
log-odds margin correct vs copied); a behavioural induction score; the weight fingerprint.

Inputs : an LM (readings.lm), task items (tasks.prompts.build) and batch size. Outputs: one record per revision with the same
schema as the released files `triage_<model>_s<seed><tag>.json` (aggregates + per-item records), written by `run`.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

from ..io import read_json, write_json
from ..tasks.prompts import build, four_forms, score, set_template, slots_of
from .lm import LM, load_model

TASKS = ["bind_a", "bind_b", "compose_s", "compose_h"]


def generate(lm, prompts, max_new=8):
    enc = lm.tok(prompts, return_tensors="pt", padding=True).to(lm.device)
    with torch.no_grad():
        out = lm.model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=lm.tok.pad_token_id)
    return [lm.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]


def fc_prefix_pair(lm, base, c, w):
    """Tokenizer-aware forced choice: if "-aff" is a single token (Qwen), compare "-c" vs "-w" after the bare stem; if "-" is its
    own token (Pythia/Gemma), compare "c" vs "w" after "stem-". Mixed cases are skipped."""
    mc, mw = len(lm.tok.encode("-" + c, add_special_tokens=False)) == 1, len(lm.tok.encode("-" + w, add_special_tokens=False)) == 1
    if mc and mw:
        return base, ("-" + c, "-" + w)
    if not mc and not mw:
        return base + "-", (c, w)
    return None, None


def fc_prefix(it, feature):
    """teacher-forced prefix up to the slot of `feature` in surface order (stem + preceding realised affixes)."""
    sl = slots_of(it.lang, it.ia, it.ib)
    k = [i for i, (f, _) in enumerate(sl) if f == feature][0]
    return it.stem + "".join("-" + a for _, a in sl[:k])


def forced_choice(lm, items, bs):
    """P(correct affix) vs the other value of the same feature, with the stem (and preceding affix) teacher-forced."""
    accs = {"a": [], "b": []}
    for b in range(0, len(items), bs):
        bp = items[b:b + bs]
        prompts, keys, pair = [], [], []
        for it in bp:
            lang = it.lang
            for feature, v, affs in (("a", it.ia, lang["affA"]), ("b", it.ib, lang["affB"])):
                if v is None:
                    continue
                pre, pr = fc_prefix_pair(lm, fc_prefix(it, feature), affs[v], affs[1 - v])
                if pr is not None:
                    prompts.append(it.prompt + " " + pre)
                    keys.append(feature)
                    pair.append(pr)
        enc, _ = lm.encode(prompts, [[len(p)] for p in prompts])
        lg, _ = lm.forward(enc)
        for k, (key, (c, w)) in enumerate(zip(keys, pair)):
            accs[key].append(float(lg[k, lm.first_token_id(c)] > lg[k, lm.first_token_id(w)]))
    return {k: (round(float(np.mean(v)), 3) if v else None) for k, v in accs.items()}


def seq_logprob(lm, prompts, conts):
    """sum log P(cont | prompt) for each pair (teacher forced), tokenizer-agnostic (continuation tokenised in context)."""
    full = [p + " " + c for p, c in zip(prompts, conts)]
    enc = lm.tok(full, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    enc = {k: v.to(lm.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = lm.model(**enc).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(2, enc["input_ids"][:, 1:, None]).squeeze(-1)  # [B, T-1]
    out = []
    for i, p in enumerate(prompts):
        start = len(p)  # continuation chars begin at len(p) (the space)
        mask = torch.tensor([int(offs[i, t + 1, 0]) >= start and int(offs[i, t + 1, 1]) > 0 for t in range(lp.shape[1])], device=lm.device)
        out.append(float((lp[i] * mask).sum()))
    return out


def seq_logprob_tokens(lm, prompts, conts):
    """per-pair: (sum log-prob, n continuation tokens, log-prob of the FIRST continuation token that differs across candidates is
    computed by the caller from per-token lists) — returns list of per-token log-prob lists for the continuation."""
    full = [p + " " + c for p, c in zip(prompts, conts)]
    enc = lm.tok(full, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    enc = {k: v.to(lm.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = lm.model(**enc).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(2, enc["input_ids"][:, 1:, None]).squeeze(-1)
    out = []
    for i, p in enumerate(prompts):
        start = len(p)
        toks = [(int(enc["input_ids"][i, t + 1]), float(lp[i, t])) for t in range(lp.shape[1]) if int(offs[i, t + 1, 0]) >= start and int(offs[i, t + 1, 1]) > 0]
        out.append(toks)
    return out


def four_way(lm, items, bs):
    """bare-'=' four-form sequence likelihood for compose items: argmax over the four bundle forms (sum log-prob); also reports the
    length-normalised (mean per token) and first-divergent-token variants
    competitor = attested bundle sharing the first realised affix."""
    acc, comp, margin, acc_norm, acc_first = [], [], [], [], []
    per_item = []
    for b in range(0, len(items), bs // 4 or 1):
        bp = items[b:b + (bs // 4 or 1)]
        prompts, conts, tags = [], [], []
        for it in bp:
            for bundle, form in four_forms(it):
                prompts.append(it.prompt)
                conts.append(form)
                tags.append(bundle)
        tl = seq_logprob_tokens(lm, prompts, conts)
        for j, it in enumerate(bp):
            cand = tags[4 * j:4 * j + 4]
            toks = tl[4 * j:4 * j + 4]
            sums = [sum(lp for _, lp in t) for t in toks]
            norms = [sum(lp for _, lp in t) / max(1, len(t)) for t in toks]
            # first divergent token: position where the four token sequences first differ; score = log-prob of that token
            ids = [[i for i, _ in t] for t in toks]
            k = 0
            while all(len(x) > k for x in ids) and len(set(x[k] for x in ids)) == 1:
                k += 1
            firsts = [t[k][1] if len(t) > k else -1e9 for t in toks]
            tgt = (it.ia, it.ib)
            first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
            competitor = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
            best = cand[int(np.argmax(sums))]
            acc.append(float(best == tgt))
            comp.append(float(best == competitor))
            acc_norm.append(float(cand[int(np.argmax(norms))] == tgt))
            acc_first.append(float(cand[int(np.argmax(firsts))] == tgt))
            lpd = dict(zip(cand, sums))
            margin.append(lpd[tgt] - lpd[competitor])
            per_item.append({"acc": int(best == tgt), "competitor": int(best == competitor), "margin": round(margin[-1], 4),
                             "acc_norm": int(acc_norm[-1]), "acc_first": int(acc_first[-1])})
    four_way.last_items = per_item
    four_way.last_extra = {"four_way_acc_lennorm": round(float(np.mean(acc_norm)), 3), "four_way_acc_firsttok": round(float(np.mean(acc_first)), 3)}
    return round(float(np.mean(acc)), 3), round(float(np.mean(comp)), 3), round(float(np.mean(margin)), 3)


def induction_score(lm, n=64, length=40, seed=0):
    """Behavioural induction / in-context copying score: accuracy of next-token prediction on the second repeat of a random
    token sequence (Olsson et al. 2022 style; first repeat excluded)."""
    g = torch.Generator().manual_seed(seed)
    vocab = lm.tok.vocab_size
    ids = torch.randint(1000, vocab - 1000, (n, length), generator=g)
    seq = torch.cat([ids, ids], dim=1).to(lm.device)
    with torch.no_grad():
        logits = lm.model(input_ids=seq).logits[:, length - 1:-1, :]
    pred = logits.argmax(-1)
    return round(float((pred == ids.to(lm.device)).float().mean()), 3)


def revision_key(step: str) -> str | None:
    """'main' / '' -> None (the main branch); digits -> 'step<N>' (Pythia); anything else is a branch or tag name."""
    if step in ("main", ""):
        return None
    return f"step{step}" if step.isdigit() else step


def score_revision(lm: LM, n: int, seed: int, bs: int = 16, tasks: list[str] | None = None, n_single: int = 2, n_combo: int = 2,
                   order: str = "ab", scheme: str = "morph") -> dict:
    """All measures for one loaded model; returns the per-revision record."""
    rec = {}
    for task in tasks or TASKS:
        items = build(task, n, seed=seed, n_single=n_single, n_combo=n_combo, order=order, scheme=scheme)
        sc = []
        gens = []
        for b in range(0, len(items), bs):
            bp = items[b:b + bs]
            gens = generate(lm, [it.prompt for it in bp])
            sc += [score(it, g) for it, g in zip(bp, gens)]
        agg = {k: round(float(np.mean([s[k] for s in sc if s[k] is not None])), 3) for k in sc[0] if k != "category" and any(s[k] is not None for s in sc)}
        agg["items"] = [{k: (int(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in s.items()} for s in sc]  # per-item outcomes (paired tests)
        cats = [s_["category"] for s_ in sc]
        agg["taxonomy"] = {c: round(cats.count(c) / len(cats), 3) for c in sorted(set(cats))}
        if scheme == "morph":
            agg.update({"fc_" + k: v for k, v in forced_choice(lm, items, bs).items() if v is not None})
        if task.startswith("compose"):
            agg["four_way_acc"], agg["four_way_competitor"], agg["four_way_margin"] = four_way(lm, items, bs)
            agg["four_way_items"] = four_way.last_items
            agg.update(four_way.last_extra)
        agg["n"] = len(items)
        agg["examples"] = [(it.answer, g) for it, g in list(zip(bp, gens))[:3]]
        rec[task] = agg
    rec["induction"] = induction_score(lm)
    rec["fingerprint"] = lm.fingerprint()
    return rec


def run(model: str, steps: str = "main", n: int = 160, seed: int = 0, bs: int = 16, tasks: str | None = None, n_single: int = 2,
        n_combo: int = 2, order: str = "ab", template: str = "default", scheme: str = "morph", tag: str = "", out_dir: str | Path = ".",
        dtype: str = "bfloat16", device: str | None = None) -> Path:
    """Score `model` at every revision in `steps` (comma-separated); resumable per revision. Returns the output path
    <out_dir>/triage_<model>_s<seed><tag>.json."""
    if template != "default":
        set_template(template)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mtag = model.split("/")[-1]
    outp = out_dir / f"triage_{mtag}_s{seed}{tag}.json"
    res = read_json(outp) if outp.exists() else {}
    for step in steps.split(","):
        rev = revision_key(step)
        key = rev or "main"
        if key in res:
            continue
        t0 = time.time()
        lm = load_model(model, revision=rev, dtype=dtype, device=device)
        rec = score_revision(lm, n, seed, bs, tasks.split(",") if tasks else None, n_single, n_combo, order, scheme)
        res[key] = rec
        print(f"{mtag} {key}: fingerprint {rec['fingerprint']} induction {rec['induction']:.2f} | "
              + " | ".join(f"{t}: copy {v['copy']:.2f} exact {v['exact']:.2f} 4way {v.get('four_way_acc', float('nan')):.2f}/{v.get('four_way_competitor', float('nan')):.2f}"
                           for t, v in rec.items() if t not in ("induction", "fingerprint")) + f"  [{time.time() - t0:.0f}s]", flush=True)
        write_json(res, outp)
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outp

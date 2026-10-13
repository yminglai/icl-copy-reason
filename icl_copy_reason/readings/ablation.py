"""Attention ablation of demonstration-line spans (the second reading): which input does the answer depend on?

For each withheld-combination item the four candidate forms are scored by summed log-probability under a 4-D attention mask in
which every token from the query line onward (query + candidate continuation) is forbidden from attending to the token span of a
chosen set of demonstration lines. Sets:
  competitor          the shown-combination lines sharing the first realised ending (the source of the copied candidate)
  competitor_marker   only the second-ending token of those lines (token-count-matched control)
  single2             the single-feature lines carrying the correct SECOND ending (the evidence for composition)
  single2_marker      only the ending token of those lines
  sham                single-feature lines carrying the first feature's OTHER value (length-matched, carries neither)
  otherbundles        the two shown combinations that do not share the first ending
  wordB               the query's second feature word (the queried word)
  wordA               the query's first feature word (same line, does not decide the second slot)
Outcome per item: margin (log-odds correct vs copied candidate), four-way arg-max accuracy, copied-candidate rate and
A_null = P(copied) / (P(copied) + P(correct))
paired differences vs the unmasked baseline with a bootstrap interval.

Inputs : an LM, items, the set names, dtype (bfloat16 default; float32 optional). Outputs: one record per revision with the schema
of the released `knockout_<model>_s<seed><tag>.json` files (per set: aggregates, per-item records, delta_margin + CI), via `run`.
"""
from __future__ import annotations

import gc
import itertools
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..io import read_json, write_json
from ..tasks.prompts import Lang, build, four_forms, set_template, slots_of
from .copy_share import revision_key
from .lm import LM, load_model

SETS = ["competitor", "single2", "sham", "otherbundles", "wordB", "competitor_marker", "wordA", "single2_marker"]


def split_line(line):
    for sep in (" = ", " -> ", " => "):
        if sep in line:
            return line.split(sep)
    return line, ""


def line_sets(it):
    """char spans (start, end) in it.prompt of each knockout set"""
    lang = Lang(**it.lang)
    lines = it.prompt.split("\n")
    header, demos, query = lines[0], lines[1:-1], lines[-1]
    sl = slots_of(it.lang, it.ia, it.ib)
    first_f, first_aff = sl[0]
    second_f, second_aff = sl[1]
    competitor = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
    comp_suffix = lang.form(it.stem, *competitor)
    comp_suffix = comp_suffix[len(it.stem):].lstrip("-") if lang.scheme != "code" else comp_suffix
    first_other = (lang.affA[1 - it.ia] if first_f == "a" else lang.affB[1 - it.ib])

    def rhs_markers(line):
        rhs = split_line(line)[1]
        if lang.scheme == "code":
            return rhs.split(" ")
        parts = rhs.split("-")
        return parts[1:]

    def is_bundle(line):
        return len(rhs_markers(line)) == 2

    def is_competitor(line):
        return is_bundle(line) and rhs_markers(line) == [a for _, a in lang.slots(*competitor)]

    def is_single_second(line):
        m = rhs_markers(line)
        return len(m) == 1 and m[0] == second_aff

    def is_sham(line):
        m = rhs_markers(line)
        return len(m) == 1 and m[0] == first_other
    spans = {k: [] for k in SETS}
    pos = len(header) + 1
    for d in demos:
        span = (pos, pos + len(d))
        if is_competitor(d):
            spans["competitor"].append(span)
        elif is_bundle(d):
            spans["otherbundles"].append(span)
        if is_single_second(d):
            spans["single2"].append(span)
        if is_sham(d):
            spans["sham"].append(span)
        pos += len(d) + 1
    # second-feature word in the query line
    qstart = pos
    words = query.split(" ")
    idx = 1 if (it.ia is not None and it.ib is not None) else 0
    wstart = qstart + len(" ".join(words[:idx])) + (1 if idx > 0 else 0)
    spans["wordB"].append((wstart, wstart + len(words[idx])))
    # token-count-matched controls: (a) only the SECOND-marker token of each competitor line (the minimal span carrying the competitor's
    # second marker); (b) the query's FIRST feature word (single token, same line, does not decide the second slot); (c) only the marker
    # token of each single-feature line carrying the correct second marker
    pos = len(header) + 1
    for d in demos:
        rhs_start = pos + len(split_line(d)[0]) + len(" = " if " = " in d else (" -> " if " -> " in d else " => "))
        rhs = split_line(d)[1]
        if is_competitor(d):
            m = rhs_markers(d)[1]
            off = rhs.rfind(m)
            spans["competitor_marker"].append((rhs_start + off, rhs_start + off + len(m)))
        if is_single_second(d):
            m = rhs_markers(d)[0]
            off = rhs.rfind(m)
            spans["single2_marker"].append((rhs_start + off, rhs_start + off + len(m)))
        pos += len(d) + 1
    if it.ia is not None and it.ib is not None:
        spans["wordA"].append((qstart, qstart + len(words[0])))
    return spans, qstart


def scored_logprobs(lm, prompts, conts, spans_list, qstarts):
    """summed log P(cont | prompt) with the knockout mask; spans_list[i] = list of (s,e) char spans to mask for item i (None = no mask)"""
    full = [p + " " + c for p, c in zip(prompts, conts)]
    enc = lm.tok(full, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    input_ids = enc["input_ids"].to(lm.device)
    am = enc["attention_mask"].to(lm.device)
    B, T = input_ids.shape
    # base causal + padding mask (bool, True = may attend)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=lm.device))
    mask = causal[None].repeat(B, 1, 1) & am[:, None, :].bool()
    for i in range(B):
        if not spans_list[i]:
            continue
        toks = offs[i]
        q_rows = [t for t in range(T) if int(toks[t, 1]) > 0 and int(toks[t, 0]) >= qstarts[i]]
        k_cols = [t for t in range(T) if int(toks[t, 1]) > 0 and any(int(toks[t, 0]) < e and int(toks[t, 1]) > s for s, e in spans_list[i])]
        if q_rows and k_cols:
            rows = torch.tensor(q_rows, device=lm.device)
            cols = torch.tensor(k_cols, device=lm.device)
            mask[i][rows[:, None], cols[None, :]] = False
    # 4-D additive mask
    dt = next(lm.model.parameters()).dtype
    add = torch.zeros(B, 1, T, T, dtype=dt, device=lm.device)
    add.masked_fill_(~mask[:, None], torch.finfo(dt).min)
    with torch.no_grad():
        logits = lm.model(input_ids=input_ids, attention_mask=add).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(2, input_ids[:, 1:, None]).squeeze(-1)
    out = []
    for i, p in enumerate(prompts):
        start = len(p)
        m = torch.tensor([int(offs[i, t + 1, 0]) >= start and int(offs[i, t + 1, 1]) > 0 for t in range(lp.shape[1])], device=lm.device)
        out.append(float((lp[i] * m).sum()))
    return out


def ablate_revision(lm: LM, items, sets: list[str], bs: int = 8, n_boot: int = 2000) -> dict:
    """Score every item under no mask and under each set; returns the per-revision record."""
    lm.model.config._attn_implementation = "eager"
    lm.text_cfg._attn_implementation = "eager"
    conds = ["none"] + list(sets)
    per = {c: [] for c in conds}
    for b in range(0, len(items), bs):
        bp = items[b:b + bs]
        spans_all = [line_sets(it) for it in bp]
        for c in conds:
            prompts, conts, tags, spans_l, qs = [], [], [], [], []
            for it, (spans, qstart) in zip(bp, spans_all):
                for bundle, form in four_forms(it):
                    prompts.append(it.prompt)
                    conts.append(form)
                    tags.append(bundle)
                    spans_l.append(None if c == "none" else spans[c])
                    qs.append(qstart)
            lps = scored_logprobs(lm, prompts, conts, spans_l, qs)
            for j, it in enumerate(bp):
                cand = dict(zip(tags[4 * j:4 * j + 4], lps[4 * j:4 * j + 4]))
                tgt = (it.ia, it.ib)
                first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
                comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
                best = max(cand, key=cand.get)
                margin = cand[tgt] - cand[comp]
                a_null = 1.0 / (1.0 + np.exp(margin))  # P(copied) / (P(copied) + P(correct))
                per[c].append({"acc": int(best == tgt), "competitor": int(best == comp), "margin": round(margin, 4), "a_null": round(float(a_null), 4)})
    rec = {"fingerprint": lm.fingerprint(), "n": len(items)}
    for c in conds:
        rec[c] = {"four_way_acc": round(float(np.mean([x["acc"] for x in per[c]])), 3), "competitor": round(float(np.mean([x["competitor"] for x in per[c]])), 3),
                  "margin": round(float(np.mean([x["margin"] for x in per[c]])), 3), "a_null": round(float(np.mean([x["a_null"] for x in per[c]])), 3), "items": per[c]}
    for c in conds[1:]:
        d = np.array([x["margin"] for x in per[c]]) - np.array([x["margin"] for x in per["none"]])
        rec[c]["delta_margin"] = round(float(d.mean()), 3)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(d), (n_boot, len(d)))
        bs_means = d[idx].mean(1)
        rec[c]["delta_margin_ci"] = [round(float(np.percentile(bs_means, 2.5)), 3), round(float(np.percentile(bs_means, 97.5)), 3)]
    return rec


def run(model: str, steps: str = "main", n: int = 160, seed: int = 0, bs: int = 8, scheme: str = "morph", template: str = "default",
        sets: str | None = None, tag: str = "", fp32: bool = False, out_dir: str | Path = ".", device: str | None = None) -> Path:
    """Ablation for `model` at every revision in `steps`; items = build("compose_h", n, seed + 11). Resumable per revision.
    Returns <out_dir>/knockout_<model>_s<seed><tag>.json."""
    if template != "default":
        set_template(template)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mtag = model.split("/")[-1]
    outp = out_dir / f"knockout_{mtag}_s{seed}{tag}.json"
    res = read_json(outp) if outp.exists() else {}
    set_list = sets.split(",") if sets else SETS
    for step in steps.split(","):
        rev = revision_key(step)
        key = rev or "main"
        if key in res:
            continue
        t0 = time.time()
        lm = load_model(model, revision=rev, dtype="float32" if fp32 else "bfloat16", device=device)
        items = build("compose_h", n, seed=seed + 11, scheme=scheme)
        rec = ablate_revision(lm, items, set_list, bs)
        rec.update({"scheme": scheme, "dtype": "float32" if fp32 else "bfloat16"})
        res[key] = rec
        print(f"{mtag} {key}: base margin {rec['none']['margin']:+.2f} A_null {rec['none']['a_null']:.2f} 4way {rec['none']['four_way_acc']:.2f} | "
              + " | ".join(f"{c}: d-margin {rec[c]['delta_margin']:+.2f} {rec[c]['delta_margin_ci']} A_null {rec[c]['a_null']:.2f}" for c in set_list)
              + f" [{time.time() - t0:.0f}s]", flush=True)
        write_json(res, outp)
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outp


# ---------------------------------------------------------------- sham-control variants (matched sham, value mean-ablation)
VARIANT_CONDS = ["none", "competitor_marker", "sham", "matched_sham", "competitor_marker_meanabl", "matched_sham_meanabl"]


def marker_spans(it):
    """per demonstration line: {"span": (s, e) of the line, "markers": [(s, e, marker)], "is_bundle", "is_competitor"} in line order"""
    lang = Lang(**it.lang)
    lines = it.prompt.split("\n")
    header, demos = lines[0], lines[1:-1]
    sl = slots_of(it.lang, it.ia, it.ib)
    first_f = sl[0][0]
    competitor = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
    comp_markers = [a for _, a in lang.slots(*competitor)]
    out = []
    pos = len(header) + 1
    for d in demos:
        lhs, rhs = split_line(d)
        sep = d[len(lhs):len(d) - len(rhs)]
        rhs_start = pos + len(lhs) + len(sep)
        parts = rhs.split("-")          # stem, marker(s)
        ms, off = [], len(parts[0]) + 1
        for m in parts[1:]:
            ms.append((rhs_start + off, rhs_start + off + len(m), m))
            off += len(m) + 1
        out.append({"span": (pos, pos + len(d)), "markers": ms, "is_bundle": len(ms) == 2, "is_competitor": len(ms) == 2 and [m for _, _, m in ms] == comp_markers})
        pos += len(d) + 1
    return out


def candidates(it, pool):
    """list of (char spans, line indices, markers carried) for the matched-sham search"""
    info = marker_spans(it)
    if pool == "second_markers":
        toks = [(i, ln_["markers"][1]) for i, ln_ in enumerate(info) if ln_["is_bundle"] and not ln_["is_competitor"]]
    else:
        toks = [(i, ln_["markers"][-1]) for i, ln_ in enumerate(info) if not ln_["is_competitor"]]
    return [([(a[1][0], a[1][1]), (b[1][0], b[1][1])], [a[0], b[0]], [a[1][2], b[1][2]]) for a, b in itertools.combinations(toks, 2)]


def all_marker_spans(it):
    return [(s, e) for ln_ in marker_spans(it) for s, e, _ in ln_["markers"]]


def cols_for(offs_row, spans):
    return [t for t in range(len(offs_row)) if int(offs_row[t][1]) > 0 and any(int(offs_row[t][0]) < e and int(offs_row[t][1]) > s for s, e in spans)]


def attention_mass(lm, prompts, qstarts, span_lists):
    """baseline (prompt-only, unmasked) attention from the query-line rows to each span, averaged over layers and heads; returns [B][n_spans]"""
    enc = lm.tok(prompts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    enc = enc.to(lm.device)
    with torch.no_grad():
        out = lm.model(**enc, output_attentions=True)
    A = torch.stack([a.float().mean(1) for a in out.attentions], 0).mean(0)            # [B, T, T] mean over layers and heads
    res = []
    for i in range(len(prompts)):
        rows = [t for t in range(A.shape[1]) if int(offs[i][t][1]) > 0 and int(offs[i][t][0]) >= qstarts[i]]
        res.append([float(A[i][rows][:, cols_for(offs[i], sp)].sum()) if cols_for(offs[i], sp) else 0.0 for sp in span_lists[i]])
    del out
    return res


class ValueMeanAblator:
    """replace the value vectors at `span_cols[r]` with the mean value vector over `marker_cols[r]` (same row), at every layer"""

    def __init__(self, lm):
        self.lm = lm
        self.span_cols = None
        self.marker_cols = None
        self.handles = []
        self.kind = None
        for blk in lm.layers:
            attn = getattr(blk, "attention", None) or blk.self_attn
            if hasattr(attn, "v_proj"):
                self.kind = "v_proj"
                self.handles.append(attn.v_proj.register_forward_hook(self._hook_vproj))
            elif hasattr(attn, "query_key_value"):
                self.kind = "fused"
                self.hd = attn.head_size
                self.handles.append(attn.query_key_value.register_forward_hook(self._hook_fused))
            else:
                raise ValueError("unsupported attention module for value mean-ablation")

    def _apply(self, v):
        """v: [B, T, ...] value tensor (any trailing shape); returns a modified clone"""
        if self.span_cols is None:
            return v
        v = v.clone()
        for r, (sc, mc) in enumerate(zip(self.span_cols, self.marker_cols)):
            if sc and mc:
                v[r, sc] = v[r, mc].mean(0, keepdim=True).to(v.dtype)
        return v

    def _hook_vproj(self, mod, inp, out):
        return self._apply(out)

    def _hook_fused(self, mod, inp, out):
        if self.span_cols is None:
            return out
        B, T, _ = out.shape
        o = out.clone().view(B, T, -1, 3 * self.hd)
        o[..., 2 * self.hd:] = self._apply(o[..., 2 * self.hd:])
        return o.view(B, T, -1)

    def remove(self):
        for h in self.handles:
            h.remove()


def scored_logprobs_vmean(lm, vabl, prompts, conts, span_chars, marker_chars):
    """summed log P(cont | prompt) with the value mean-ablation at the span (char spans per row); ordinary causal + padding mask"""
    full = [p + " " + c for p, c in zip(prompts, conts)]
    enc = lm.tok(full, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")
    vabl.span_cols = [cols_for(offs[i], span_chars[i]) for i in range(len(full))]
    vabl.marker_cols = [cols_for(offs[i], marker_chars[i]) for i in range(len(full))]
    input_ids = enc["input_ids"].to(lm.device)
    am = enc["attention_mask"].to(lm.device)
    try:
        with torch.no_grad():
            logits = lm.model(input_ids=input_ids, attention_mask=am).logits.float()
    finally:
        vabl.span_cols = None
        vabl.marker_cols = None
    lp = torch.log_softmax(logits[:, :-1], -1).gather(2, input_ids[:, 1:, None]).squeeze(-1)
    out = []
    for i, p in enumerate(prompts):
        start = len(p)
        m = torch.tensor([int(offs[i, t + 1, 0]) >= start and int(offs[i, t + 1, 1]) > 0 for t in range(lp.shape[1])], device=lm.device)
        out.append(float((lp[i] * m).sum()))
    return out


def sham_variants(model: str, seed: int, steps: str = "143000", n: int = 160, bs: int = 8, pool: str = "second_markers", tag: str = "_shamfix",
                  overwrite: bool = False, fp32: bool = False, out_dir: str | Path = ".", device: str | None = None) -> Path:
    """The sham-control variants: the standard sham, an attention-mass-matched sham, and value mean-ablation of the competitor-marker and
    matched-sham spans (conditions VARIANT_CONDS). Returns <out_dir>/knockout_variants_<model>_s<seed><tag>.json."""
    args = SimpleNamespace(model=model, steps=steps, seed=seed, n=n, bs=bs, pool=pool, tag=tag, overwrite=overwrite, fp32=fp32)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mtag = args.model.split("/")[-1]
    rev = None if args.steps in ("main", "") else (args.steps if args.steps.startswith("stage") else f"step{args.steps}")
    outp = Path(out_dir) / f"knockout_variants_{mtag}_s{args.seed}{args.tag}.json"
    if outp.exists() and not args.overwrite:
        print("exists", outp)
        return outp
    t0 = time.time()
    lm = load_model(args.model, revision=rev, dtype="float32" if args.fp32 else "bfloat16", device=device)
    lm.model.set_attn_implementation("eager")
    lm.model.config._attn_implementation = "eager"
    lm.text_cfg._attn_implementation = "eager"
    vabl = ValueMeanAblator(lm)
    items = build("compose_h", args.n, seed=args.seed + 11)
    rec = {"model": args.model, "revision": rev or "main", "seed": args.seed, "n": len(items), "pool": args.pool, "fingerprint": lm.fingerprint(), "value_hook": vabl.kind, "dtype": "float32" if args.fp32 else "bfloat16",
           **{c: {"items": []} for c in VARIANT_CONDS}, "matching": []}
    for b0 in range(0, len(items), args.bs):
        bp = items[b0:b0 + args.bs]
        spans_all = [line_sets(it) for it in bp]
        cands_all = [candidates(it, args.pool) for it in bp]
        markers_all = [all_marker_spans(it) for it in bp]
        # (a) matching on baseline attention mass
        span_lists = [[sp["competitor_marker"], sp["sham"]] + [c[0] for c in cands] for (sp, _), cands in zip(spans_all, cands_all)]
        masses = attention_mass(lm, [it.prompt for it in bp], [q for _, q in spans_all], span_lists)
        matched = []
        for it, (_sp, _q0), cands, ms in zip(bp, spans_all, cands_all, masses):
            m_comp, m_sham, m_c = ms[0], ms[1], ms[2:]
            j = int(np.argmin([abs(m - m_comp) for m in m_c])) if m_c else None
            if j is None:
                raise RuntimeError("no matched-sham candidate for an item")
            lang = Lang(**it.lang)
            sl = slots_of(it.lang, it.ia, it.ib)
            first_f = sl[0][0]
            comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
            corr2 = sl[1][1]
            comp2 = [a for _, a in lang.slots(*comp)][1]
            carries = ["correct_second" if m == corr2 else ("competitor_second" if m == comp2 else "other") for m in cands[j][2]]
            matched.append(cands[j][0])
            rec["matching"].append({"mass_competitor": m_comp, "mass_matched": m_c[j], "mass_sham_std": m_sham, "matched_lines": cands[j][1], "matched_markers": cands[j][2], "matched_carries": carries, "n_candidates": len(m_c),
                                    "all_candidate_masses": m_c})
        # scoring of the four forms under each condition
        for c in VARIANT_CONDS:
            prompts, conts, tags, spans_l, qs, mk = [], [], [], [], [], []
            for j, (it, (sp, q0)) in enumerate(zip(bp, spans_all)):
                for bundle, form in four_forms(it):
                    prompts.append(it.prompt)
                    conts.append(form)
                    tags.append(bundle)
                    qs.append(q0)
                    mk.append(markers_all[j])
                    spans_l.append({"none": None, "competitor_marker": sp["competitor_marker"], "sham": sp["sham"], "matched_sham": matched[j],
                                    "competitor_marker_meanabl": sp["competitor_marker"], "matched_sham_meanabl": matched[j]}[c])
            if c.endswith("_meanabl"):
                lps = scored_logprobs_vmean(lm, vabl, prompts, conts, spans_l, mk)
            else:
                lps = scored_logprobs(lm, prompts, conts, spans_l, qs)
            for j, it in enumerate(bp):
                cand = dict(zip(tags[4 * j:4 * j + 4], lps[4 * j:4 * j + 4]))
                tgt = (it.ia, it.ib)
                first_f = slots_of(it.lang, it.ia, it.ib)[0][0]
                comp = (it.ia, 1 - it.ib) if first_f == "a" else (1 - it.ia, it.ib)
                best = max(cand, key=cand.get)
                margin = cand[tgt] - cand[comp]
                a_null = 1.0 / (1.0 + np.exp(margin))
                rec[c]["items"].append({"acc": int(best == tgt), "competitor": int(best == comp), "margin": round(margin, 4), "a_null": round(float(a_null), 4)})
        print(f"  batch {b0 // args.bs + 1}/{(len(items) + args.bs - 1) // args.bs} [{time.time() - t0:.0f}s]", flush=True)
    for c in VARIANT_CONDS:
        per = rec[c]["items"]
        rec[c].update({"four_way_acc": round(float(np.mean([x["acc"] for x in per])), 3), "competitor": round(float(np.mean([x["competitor"] for x in per])), 3),
                       "margin": round(float(np.mean([x["margin"] for x in per])), 3), "a_null": round(float(np.mean([x["a_null"] for x in per])), 3)})
        if c != "none":
            d = np.array([x["margin"] for x in per]) - np.array([x["margin"] for x in rec["none"]["items"]])
            g = np.random.default_rng(0)
            bs_ = d[g.integers(0, len(d), (2000, len(d)))].mean(1)
            rec[c]["delta_margin"] = round(float(d.mean()), 3)
            rec[c]["delta_margin_ci"] = [round(float(np.percentile(bs_, 2.5)), 3), round(float(np.percentile(bs_, 97.5)), 3)]
    bal = np.array([[m["mass_competitor"], m["mass_matched"], m["mass_sham_std"]] for m in rec["matching"]])
    rec["balance"] = {"mean_mass_competitor": float(bal[:, 0].mean()), "mean_mass_matched": float(bal[:, 1].mean()), "mean_mass_sham_std": float(bal[:, 2].mean()),
                      "mean_abs_diff_matched": float(np.abs(bal[:, 1] - bal[:, 0]).mean()), "mean_abs_diff_sham_std": float(np.abs(bal[:, 2] - bal[:, 0]).mean())}
    print(f"{mtag} {rev or 'main'} s{args.seed}: base margin {rec['none']['margin']:+.2f} | " + " | ".join(f"Δ {c} {rec[c]['delta_margin']:+.2f} {rec[c]['delta_margin_ci']}" for c in VARIANT_CONDS[1:]) +
          f" | mass comp {rec['balance']['mean_mass_competitor']:.3f} matched {rec['balance']['mean_mass_matched']:.3f} (|diff| {rec['balance']['mean_abs_diff_matched']:.3f}) sham_std {rec['balance']['mean_mass_sham_std']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    write_json(rec, outp, indent=0)
    print("wrote", outp)
    vabl.remove()
    lm = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outp

"""The two extra exercise shapes tried in the exercise-variant loop and their mechanical checks.

Inputs : the variant prompt table data/exercises/variant_prompts.json (type k and type h2 specifications and generator
         templates; the other fields are those of data/exercises/prompts.json), verified records written by
         `exercises.verify`, and the document file of the topped-up set.
Outputs: `select_documents` (the documents a variant run rewrites: the first n records with >= 1,500 characters), the
         variant prompt builder used by `exercises.generate` when the prompt table carries a variant type, and
         `post_verify`, which recomputes the variant checks on a verified file and rewrites `pass` / `reasons`.

type k   the diagnostic's own structure on real text: two binary features with nonce markers, demonstrations show the
         single-feature forms and three of the four combined cells, the query asks for the fourth on a new term;
         checks: 1k (components attested, composed answer unshown; replaces check 1) and K0, K2-K9 (declared structure).
type h2  type h with four shape changes (8-10 demonstrations, one "Note:" rule line after the last demonstration, a real-text
         prefix <= 400 characters, single query); checks 7 (the Note line) and 8 (prefix / demonstration count) added.
"""
from __future__ import annotations

import collections
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..io import read_jsonl, sha256_file, sha256_text, write_json
from .verification import fill, norm_answer, parse_demos

MIN_CHARS = 1500
ARROWS = ("→", "->", "=>", "⇒", "⟶", "⟹")
ARROW_RE = re.compile(r"(?:→|->|=>|⇒|⟶|⟹)")
MARKER_RE = re.compile(r"^[a-z]{2,5}$")
NOTE_RE = re.compile(r"^\s*note\s*:", re.I)
EX_KEYS_K = ("context", "query", "answer", "justification", "type", "prefix", "features", "demos", "query_term", "query_values", "competitor")
M1K_REASON = {"m1k_components_attested": "CHECK1k_component_missing", "m1k_answer_unshown": "CHECK1k_answer_attested"}
MK_REASON = {"k0_struct": "CHECKK0_struct", "k2_cells": "CHECKK2_cells", "k3_singles": "CHECKK3_singles", "k4_one_query": "CHECKK4_one_query", "k5_query_term_new": "CHECKK5_query_term_new",
             "k6_compose": "CHECKK6_compose", "k7_demos_attested": "CHECKK7_demos_attested", "k8_competitor": "CHECKK8_competitor", "k9_shape": "CHECKK9_shape"}
M78_REASON = {"m7_exactly_one_note": "CHECK7_note_count", "m7_after_last_demo": "CHECK7_note_position", "m7_no_answer_copy_keyterm": "CHECK7_note_leaks", "m8_prefix_le_400": "CHECK8_prefix_long", "m8_demos_8_to_10": "CHECK8_demo_count"}


def gen_prompts_variant(P: dict, doc: dict) -> tuple[str, str]:
    """Generator prompt for a variant type: the variant template with the base system prompt and the topped-up ban appended to the spec."""
    t = doc["type"]
    spec = P["types"][t]
    tpl = P[f"gen_user_{t}"]
    return P["gen_system"], fill(tpl, type=t, name=spec["name"], spec=spec["spec"] + "\n" + P["gen_ban_topup"], document=doc["text"][: P["document_chars"]["prompt_truncate"]])


def select_documents(docs_src: str | Path, n: int, out: str | Path, variant: str, sort_pool_file_row: bool) -> tuple[list[dict], dict]:
    """The first n documents with >= MIN_CHARS characters (file order for the pilot; (pool, file, row) order for the full run), retyped to the variant."""
    docs = [d for d in read_jsonl(docs_src) if d["chars"] >= MIN_CHARS]
    if sort_pool_file_row:
        docs.sort(key=lambda d: (d["pool"], d["file"], d["row"]))
    sel = [{**{k: v for k, v in d.items() if k != "type_before"}, "type": variant, "type_before": d["type"]} for d in docs[:n]]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for d in sel:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    info = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": str(docs_src), "source_sha256": sha256_file(docs_src), "variant": variant,
            "rule": f"first {n} records with chars >= {MIN_CHARS} in " + ("(pool, file, row) order" if sort_pool_file_row else "file order"), "n_selected": len(sel), "n_eligible": len(docs),
            "selection_sha256": sha256_text("\n".join(d["sha256"] for d in sel)), "pools": dict(collections.Counter(d["pool"] for d in sel)), "types_before": dict(collections.Counter(d["type_before"] for d in sel))}
    write_json(info, str(out).replace(".jsonl", "") + "_selection.json")
    return sel, info


# ---------------------------------------------------------------- type k
def _toks(s: Any) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", norm_answer(s or "")) if t]


def _has_seq(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def field(ex: dict, k: str) -> Any:
    """decode a declared field (the generator stores non-string JSON values as json.dumps strings)"""
    v = ex.get(k)
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{":
            try:
                return json.loads(s)
            except Exception:  # noqa: BLE001
                return v
    return v


def _str(x: Any) -> str:
    return x.strip() if isinstance(x, str) else ("" if x is None else str(x))


def _nz(x: Any) -> Any:
    return None if x is None or (isinstance(x, str) and not x.strip()) else x


def compose(term: Any, m1: str | None, m2: str | None) -> str:
    return "-".join([_str(term)] + [m for m in (m1, m2) if m])


def check1k(ex: dict, decl: dict) -> dict:
    """Check 1k (type k; replaces check 1): every token of the query term and both answer markers occurs in the context (components
    attested) AND the answer is not a contiguous token sequence of the context and equals no declared demonstration form (unshown)."""
    ctx = _toks(ex["context"])
    ctxset = set(ctx)
    ans = _toks(ex["answer"])
    qt = _toks(decl.get("query_term") or "")
    comps = qt + [m for m in (decl.get("m1"), decl.get("m2")) if m]
    r: dict[str, Any] = {"components": comps, "missing": [c for c in comps if c not in ctxset], "answer_seq_in_context": _has_seq(ctx, ans), "answer_is_demo_form": norm_answer(ex["answer"]) in decl.get("form_norms", set())}
    r["m1k_components_attested"] = bool(comps) and bool(qt) and decl.get("m1") is not None and decl.get("m2") is not None and not r["missing"]
    r["m1k_answer_unshown"] = bool(ans) and not r["answer_seq_in_context"] and not r["answer_is_demo_form"]
    r["enforced"] = ["m1k_components_attested", "m1k_answer_unshown"]
    r["pass"] = all(r[k] for k in r["enforced"])
    return r


def decode(ex: dict) -> dict:
    """declared structure -> normalised index tables (ok=False with reasons if the structure does not parse)"""
    feats = field(ex, "features")
    demos = field(ex, "demos")
    qv = field(ex, "query_values")
    qt = _str(ex.get("query_term"))
    d: dict[str, Any] = {"ok": False, "why": [], "query_term": qt, "form_norms": set()}
    if not (isinstance(feats, list) and len(feats) == 2 and all(isinstance(f, dict) for f in feats)):
        d["why"].append("features_not_2_objects")
        return d
    vals, marks = [], []
    for f in feats:
        v = f.get("values")
        m = f.get("markers")
        if not (isinstance(v, list) and len(v) == 2 and isinstance(m, list) and len(m) == 2 and all(isinstance(x, str) and x.strip() for x in v + m)):
            d["why"].append("feature_values_or_markers_malformed")
            return d
        vals.append([norm_answer(x) for x in v])
        marks.append([norm_answer(x) for x in m])
    flat = marks[0] + marks[1]
    if len(set(flat)) != 4:
        d["why"].append("markers_not_distinct")
    if not all(MARKER_RE.match(m) for m in flat):
        d["why"].append("marker_not_2_to_5_letters")
    if any(m == v or (len(v) >= 2 and (m in v or v in m)) for m in flat for v in vals[0] + vals[1]):
        d["why"].append("marker_overlaps_value_word")
    if not (isinstance(demos, list) and demos and all(isinstance(x, dict) and _str(x.get("term")) and _str(x.get("form")) and isinstance(x.get("values"), list) and len(x["values"]) == 2 for x in demos)):
        d["why"].append("demos_malformed")
        return d

    def vidx(fi: int, v: Any):
        v = _nz(v)
        if v is None:
            return None
        vn = norm_answer(v)
        return vals[fi].index(vn) if vn in vals[fi] else -1

    dd = []
    for x in demos:
        i1, i2 = vidx(0, x["values"][0]), vidx(1, x["values"][1])
        dd.append({"term": _str(x["term"]), "term_norm": norm_answer(x["term"]), "i1": i1, "i2": i2, "form": _str(x["form"]), "form_norm": norm_answer(x["form"]), "cell": (i1, i2),
                   "expected_norm": norm_answer(compose(x["term"], marks[0][i1] if i1 not in (None, -1) else None, marks[1][i2] if i2 not in (None, -1) else None)) if -1 not in (i1, i2) else None})
    if any(-1 in (x["i1"], x["i2"]) for x in dd):
        d["why"].append("demo_value_not_declared")
    if not (isinstance(qv, list) and len(qv) == 2):
        d["why"].append("query_values_malformed")
        return d
    q1, q2 = vidx(0, qv[0]), vidx(1, qv[1])
    if q1 in (None, -1) or q2 in (None, -1):
        d["why"].append("query_values_not_both_declared")
        q1 = q2 = None
    d.update({"ok": not d["why"], "values": vals, "markers": marks, "demos": dd, "q1": q1, "q2": q2, "m1": marks[0][q1] if q1 is not None else None, "m2": marks[1][q2] if q2 is not None else None,
              "form_norms": {x["form_norm"] for x in dd}, "competitor": _str(ex.get("competitor")), "prefix": _str(ex.get("prefix"))})
    return d


def checkk(ex: dict) -> dict:
    """Check K (type k), computed from the declared fields and the context: K0 structure, K2 three shown cells + query cell unshown,
    K3 single-feature forms and both answer markers attested, K4 exactly one open query, K5 query term new, K6 composition rule,
    K7 demonstration forms attested in the context, K8 competitor is a shown combined form sharing a marker, K9 shape."""
    decl = decode(ex)
    ctx = ex["context"]
    lines = ctx.split("\n")
    ctx_toks = _toks(ctx)
    ans_n = norm_answer(ex["answer"])
    q = ex["query"]
    arrow_lines, _ = parse_demos(ctx)
    r: dict[str, Any] = {"struct_why": decl["why"], "n_declared_demos": len(decl.get("demos", [])), "n_arrow_lines": len(arrow_lines)}
    r["k0_struct"] = decl["ok"]
    qa = ARROW_RE.findall(q)
    open_ctx_lines = [ln.strip() for ln in lines if ln.strip().endswith(ARROWS)]
    r["k4_one_query"] = len(qa) == 1 and q.strip().endswith(ARROWS) and not open_ctx_lines
    r["open_context_arrow_lines"] = open_ctx_lines[:3]
    enforced = ["k0_struct", "k2_cells", "k3_singles", "k4_one_query", "k5_query_term_new", "k6_compose", "k7_demos_attested", "k8_competitor", "k9_shape"]
    if not decl["ok"]:
        for k in enforced[1:]:
            if k != "k4_one_query":
                r[k] = False
        r["enforced"] = enforced
        r["pass"] = False
        r["decl"] = None
        return r
    dd = decl["demos"]
    combined = [x for x in dd if x["i1"] is not None and x["i2"] is not None]
    singles = [x for x in dd if (x["i1"] is None) != (x["i2"] is None)]
    cells = {x["cell"] for x in combined}
    qcell = (decl["q1"], decl["q2"])
    r["shown_cells"] = sorted(cells)
    r["query_cell"] = list(qcell)
    r["n_combined_demos"] = len(combined)
    r["n_single_demos"] = len(singles)
    r["k2_cells"] = len(cells) == 3 and qcell not in cells and len(combined) == 3
    single_vals = {(0, x["i1"]) for x in singles if x["i1"] is not None} | {(1, x["i2"]) for x in singles if x["i2"] is not None}
    r["all_four_singles_report_only"] = len(single_vals) == 4
    forms_toks = [_toks(x["form"]) for x in dd]
    r["k3_singles"] = any(f == 0 for f, _ in single_vals) and any(f == 1 for f, _ in single_vals) and all(any(m in ft for ft in forms_toks) for m in (decl["m1"], decl["m2"]))
    qt_n = norm_answer(decl["query_term"])
    qt_t = _toks(decl["query_term"])
    demo_lines = [ln for ln in lines if any(_has_seq(_toks(ln), _toks(x["form"])) for x in dd)]
    r["k5_query_term_new"] = bool(qt_t) and qt_n not in {x["term_norm"] for x in dd} and not any(_has_seq(_toks(ln), qt_t) for ln in demo_lines) and _has_seq(_toks(q), qt_t)
    exp = norm_answer(compose(decl["query_term"], decl["m1"], decl["m2"]))
    r["expected_answer_norm"] = exp
    r["demo_inconsistent"] = [x["form"] for x in dd if x["expected_norm"] != x["form_norm"]]
    r["k6_compose"] = ans_n == exp and not r["demo_inconsistent"]
    r["demo_forms_unattested"] = [x["form"] for x in dd if not _has_seq(ctx_toks, _toks(x["form"]))]
    r["k7_demos_attested"] = not r["demo_forms_unattested"]
    r["arrow_count_matches_report_only"] = len(arrow_lines) == len(dd)
    comp_n = norm_answer(decl["competitor"])
    comp_t = _toks(decl["competitor"])
    shared = [m for m in (decl["m1"], decl["m2"]) if m in comp_t]
    r["k8_competitor"] = bool(comp_n) and comp_n in decl["form_norms"] and comp_n != ans_n and bool(shared)
    r["competitor_is_first_marker_bundle_report_only"] = bool(comp_n) and comp_n in {x["form_norm"] for x in combined} and decl["m1"] in comp_t and decl["m2"] not in comp_t
    pre = decl["prefix"]
    r["prefix_chars"] = len(pre.strip())
    pre_t = _toks(pre)[:12]
    r["k9_shape"] = 0 < r["prefix_chars"] <= 400 and _has_seq(ctx_toks, pre_t) and 6 <= len(dd) <= 10
    r["query_mentions_values_report_only"] = all(_has_seq(_toks(q), _toks(decl["values"][i][qi])) for i, qi in ((0, decl["q1"]), (1, decl["q2"])))
    r["decl"] = {"values": decl["values"], "markers": decl["markers"], "query_term": decl["query_term"], "competitor": decl["competitor"], "demos": [{"term": x["term"], "cell": list(x["cell"]), "form": x["form"]} for x in dd]}
    r["enforced"] = enforced
    r["pass"] = all(r[k] for k in enforced)
    return r


# ---------------------------------------------------------------- type h2
def _demo_indices(lines: list[str], demos: list[dict]) -> list[int]:
    dl = {d["line"] for d in demos}
    return [i for i, ln in enumerate(lines) if ln.strip() and ln.strip() in dl]


def check7(ex: dict) -> dict:
    """Check 7 (type h2): exactly one 'Note:' line, after the last demonstration, not containing the answer, the copy candidate or the key term."""
    lines = ex["context"].split("\n")
    demos, _ = parse_demos(ex["context"])
    di = _demo_indices(lines, demos)
    ni = [i for i, ln in enumerate(lines) if NOTE_RE.match(ln)]
    note = lines[ni[0]] if ni else ""
    hay = _toks(note)
    hits = [k for k in ("answer", "copy_candidate", "key_term") if _has_seq(hay, _toks(ex.get(k)))]
    r: dict[str, Any] = {"n_note_lines": len(ni), "note_line": note.strip() or None, "note_idx": ni[0] if ni else None, "last_demo_idx": max(di) if di else None, "note_hits": hits}
    r["m7_exactly_one_note"] = len(ni) == 1
    r["m7_after_last_demo"] = bool(ni) and bool(di) and ni[0] > max(di)
    r["m7_no_answer_copy_keyterm"] = bool(ni) and not hits
    r["enforced"] = ["m7_exactly_one_note", "m7_after_last_demo", "m7_no_answer_copy_keyterm"]
    r["pass"] = all(r[k] for k in r["enforced"])
    return r


def check8(ex: dict) -> dict:
    """Check 8 (type h2): the prose before the first demonstration is <= 400 characters and there are 8-10 demonstrations."""
    lines = ex["context"].split("\n")
    demos, _ = parse_demos(ex["context"])
    di = _demo_indices(lines, demos)
    prefix = "\n".join(lines[: min(di)]) if di else ex["context"]
    r: dict[str, Any] = {"prefix_chars": len(prefix.strip()), "n_demos": len(demos), "first_demo_idx": min(di) if di else None}
    r["m8_prefix_le_400"] = r["prefix_chars"] <= 400
    r["m8_demos_8_to_10"] = 8 <= len(demos) <= 10
    r["enforced"] = ["m8_prefix_le_400", "m8_demos_8_to_10"]
    r["pass"] = all(r[k] for k in r["enforced"])
    return r


# ---------------------------------------------------------------- post-verification pass
def post_verify(path: str | Path, docs_path: str | Path, variant: str) -> dict:
    """Idempotent: recompute the variant checks for every record of a verified file, rewrite `pass` and `reasons`
    (type k: check 1 is replaced by check 1k and check K is added; type h2: checks 7 and 8 are added), write <stem>_post_summary.json."""
    path = str(path)
    recs = list(read_jsonl(path))
    t0 = time.time()
    for r in recs:
        ex = r["exercise"]
        if variant == "k":
            mk = checkk(ex)
            m1k = check1k(ex, decode(ex))
            base = [x for x in r["reasons"] if not x.startswith(("CHECK1_", "MECH1_", "CHECK1k", "MECH1k", "CHECKK", "MECHK"))]
            new = [M1K_REASON[k] for k in m1k["enforced"] if not m1k[k]] + [MK_REASON[k] for k in mk["enforced"] if not mk[k]]
            r["mech1k"], r["mechk"], r["pass_base_rule"] = m1k, mk, not base
        else:
            m7, m8 = check7(ex), check8(ex)
            base = [x for x in r["reasons"] if not x.startswith(("CHECK7", "CHECK8", "MECH7", "MECH8"))]
            new = [M78_REASON[k] for k in m7["enforced"] if not m7[k]] + [M78_REASON[k] for k in m8["enforced"] if not m8[k]]
            r["mech7"], r["mech8"], r["pass_base_rule"] = m7, m8, not base
        r["reasons"] = base + new
        r["pass"] = not r["reasons"]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    summ = summarise(recs, docs_path, variant)
    summ["post_pass_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summ["seconds"] = round(time.time() - t0, 1)
    write_json(summ, path.replace(".jsonl", "") + "_post_summary.json")
    return summ


def summarise(recs: list[dict], docs_path: str | Path, variant: str) -> dict:
    n_docs = sum(1 for _ in read_jsonl(docs_path))
    first = [r for r in recs if not r.get("retry")]
    rt = [r for r in recs if r.get("retry")]
    docs_pass = {r["doc_sha256"] for r in recs if r["pass"]}
    out = {"variant": variant, "n_docs": n_docs, "n_records": len(recs), "n_first": len(first), "n_retry": len(rt), "n_first_pass": sum(r["pass"] for r in first), "n_retry_pass": sum(r["pass"] for r in rt),
           "n_docs_pass_any": len(docs_pass), "yield_first_pass_per_doc": sum(r["pass"] for r in first) / max(1, n_docs), "yield_after_retry_per_doc": len(docs_pass) / max(1, n_docs),
           "retry_pass_rate": sum(r["pass"] for r in rt) / max(1, len(rt)), "unique_verified_answers": len({r["answer_norm"] for r in recs if r["pass"]}),
           "reasons": {"first": dict(collections.Counter(x for r in first for x in r["reasons"]).most_common()), "retry": dict(collections.Counter(x for r in rt for x in r["reasons"]).most_common())},
           "n_tokens": {"min": min((r["n_tokens"] for r in recs), default=None), "median": int(np.median([r["n_tokens"] for r in recs])) if recs else None, "max": max((r["n_tokens"] for r in recs), default=None)}}
    if variant == "k":
        out["struct_why"] = dict(collections.Counter(w for r in recs for w in r["mechk"]["struct_why"]))
        out["prefix_chars"] = {"median": int(np.median([r["mechk"].get("prefix_chars", 0) for r in recs])) if recs else None, "max": max((r["mechk"].get("prefix_chars", 0) for r in recs), default=None)}
        out["n_demos"] = dict(collections.Counter(r["mechk"]["n_declared_demos"] for r in recs))
    else:
        out["prefix_chars"] = {"median": int(np.median([r["mech8"]["prefix_chars"] for r in recs])) if recs else None, "max": max((r["mech8"]["prefix_chars"] for r in recs), default=None)}
        out["n_demos"] = dict(collections.Counter(r["mech8"]["n_demos"] for r in recs))
        out["n_note_lines"] = dict(collections.Counter(r["mech7"]["n_note_lines"] for r in recs))
    return out

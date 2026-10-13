"""Answer normalisation, the evaluation-window store and the mechanical checks that every exercise must pass.

Inputs : exercise dicts {context, query, answer, justification, type, (key_term, copy_candidate, prior_candidate for type h)},
         the prompt table (forbidden regex, packed text format, banned dataset class names), a tokenizer, and the
         evaluation-window store (50-character windows of every evaluation prompt, template, pool and label).
Outputs: booleans / detail dicts per check, the list of reasons an exercise fails, and the selection of passing records
         for packing (one record per document: the passing retry replaces the failed first draw).

Checks (fixed before the run):
  check 1  every token of the normalised answer occurs in the normalised context
  check 2  the normalised answer is not the trailing token sequence of a context line carrying a ':' / '=' / 'is written' cue
           (enforced for type a only; reported for the other types)
  check 3  no forbidden string (evaluation words, markers, templates) in context + query + answer
  check 4  no 50-character window of the packed text occurs in any evaluation prompt (the window store)
  check 5  packed length between tok_min and tok_max tokens
  check 6  for type h: (1) copy and prior candidates are demonstration outputs different from the answer and each other,
           (2) the key term's content words are in the query and in a copy-candidate demonstration input, (3) an answer
           demonstration shares a content word with the query, (4) >= 5 distinct outputs with >= 2 unnamed, (5) no output is a
           class name of the symbol-label evaluation datasets; for the other topped-up types only (5)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..io import read_json, read_jsonl

WIN = 50  # characters per evaluation window
HB = np.uint64(1099511628211)
QUOTES = "\"'`“”‘’«»()[]{}"
CUE_SPLIT = re.compile(r":|=| is written ")
STOPWORDS = set(
    """a an the and or but if then else of to in on at by for from with without into onto over under about above below between among through during before after as is are was were be been
being am do does did done have has had having will would shall should can could may might must not no nor so than too very this that these those there here where when why how which who whom whose what
it its they them their theirs we us our ours you your yours he him his she her hers i me my mine one ones all any each every some such both either neither other another more most less least much many few
several own same also just only even still yet again ever never always often sometimes already now once twice first second third last next previous following given using based according within upon via
per each because while although though whether since until unless therefore thus hence however moreover otherwise rather quite really simply same
input inputs output outputs demonstration demonstrations example examples query question questions answer answers rule rules exercise document passage passages text label labels token tokens context
write writes give gives return reply state states item items case cases line lines number set member members correct wrong true false none null apply applies applying convention above below""".split()
)
WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
ARROW_RE = re.compile(r"\s*(?:→|->|=>|⇒|⟶|⟹)\s*")
PREFIX_RE = re.compile(r"^\s*(?:(?:demonstration|demo|example|case)\s*\d*\s*[:.)\-]?\s*)?(?:input\s*:)?\s*", re.I)
CHECK6_REASON = {
    "m6_demos_parsed": "CHECK6_no_demos_parsed", "m6_1_candidates_are_demo_outputs_ne_answer": "CHECK6_1_candidates", "m6_2_key_term": "CHECK6_2_key_term",
    "m6_3_answer_demo_shares_content_word": "CHECK6_3_answer_demo_overlap", "m6_4_five_outputs_two_unnamed": "CHECK6_4_output_set", "m6_5_no_dataset_class_name": "CHECK6_5_class_name",
}


def fill(tpl: str, **kw: str) -> str:
    for k, v in kw.items():
        tpl = tpl.replace("{" + k + "}", v)
    return tpl


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_answer(s: Any) -> str:
    """lowercase; strip; collapse whitespace; strip surrounding quotes/brackets; drop a trailing period; drop a leading article;
    remove punctuation except letters/digits/hyphen; collapse whitespace again (check 1 applies the same function to the context)."""
    s = norm_ws(str(s)).lower()
    while True:
        t = s.strip()
        while t and t[0] in QUOTES:
            t = t[1:]
        while t and t[-1] in QUOTES:
            t = t[:-1]
        if t.endswith("."):
            t = t[:-1]
        t = t.strip()
        if t == s:
            break
        s = t
    s = re.sub(r"^(the|an|a)\s+", "", s)
    s = re.sub(r"[^\w\s\-]", "", s).replace("_", " ")
    return norm_ws(s)


def packed_text(ex: dict, fmt: str) -> str:
    return fill(fmt, context=ex["context"], query=ex["query"], answer=ex["answer"], justification=ex.get("justification") or "")


def ntok(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False)["input_ids"])


def parse_json_obj(s: str) -> dict | None:
    """The first JSON object in a model reply (code fences stripped; raw newlines inside strings allowed)."""
    import json

    t = s.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    for cand in (t, t[t.find("{") : t.rfind("}") + 1] if "{" in t and "}" in t else ""):
        if not cand:
            continue
        try:
            o = json.loads(cand, strict=False)
            if isinstance(o, dict):
                return o
        except Exception:  # noqa: BLE001
            pass
    return None


# ---------------------------------------------------------------- evaluation windows (check 4)
def win_hashes(s: str) -> np.ndarray:
    s = norm_ws(s)
    if len(s) < WIN:
        return np.zeros(0, np.uint64)
    cp = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    n = len(cp) - WIN + 1
    h = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(WIN):
            h = h * HB + cp[j : j + n]
    return h


def window_at(s: str, target: np.uint64) -> str | None:
    s = norm_ws(s)
    hs = win_hashes(s)
    k = np.nonzero(hs == target)[0]
    return s[k[0] : k[0] + WIN] if len(k) else None


def eval_hit(evalh: np.ndarray, s: str) -> int | None:
    h = win_hashes(s)
    if not len(h) or not len(evalh):
        return None
    idx = np.searchsorted(evalh, h)
    idx = np.minimum(idx, len(evalh) - 1)
    hit = np.nonzero(evalh[idx] == h)[0]
    return int(h[hit[0]]) if len(hit) else None


def hash_strings(strings: list[str]) -> np.ndarray:
    parts = [win_hashes(s) for s in sorted(set(strings))]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.uint64)


def load_eval_windows(store_dir: str | Path, required: bool = True):
    """eval_windows.npy (sorted uint64 hashes) + eval_windows.json (sources, sha256) from a directory."""
    store_dir = Path(store_dir)
    npy, meta = store_dir / "eval_windows.npy", store_dir / "eval_windows.json"
    if not (npy.exists() and meta.exists()):
        if required:
            raise SystemExit(f"{npy} missing: build the evaluation-window store first (scripts/build_exercises.py windows)")
        return None, None
    return np.load(npy), read_json(meta)


def evaluation_strings(datasets: bool = True) -> dict[str, list[str]]:
    """Every string whose 50-character windows an exercise may not contain: diagnostic templates and pools, rendered synthetic
    documents of both modes, the symbol-label / exchanged-label prompts, far-assay prompts, and (datasets=True) the test texts
    and label names of the symbol-label evaluation datasets. Dataset loads need network access."""
    import random

    from ..tasks import prompts as tp
    from . import synthetic_docs as sd

    strings: dict[str, list[str]] = {"templates_pools": [], "rendered_synthetic_docs": []}
    strings["templates_pools"] += [h for h, _ in sd.TEMPLATES] + [s for _, s in sd.TEMPLATES] + sorted(sd.ALL_WORDS) + sd.MARKERS
    strings["templates_pools"] += tp.evaluation_strings()
    for s in range(200):
        for mode in ("compose", "attested"):
            strings["rendered_synthetic_docs"].append(sd.make_doc(random.Random(s), mode, ver=3)[0])
    if datasets:
        from .external_sets import evaluation_strings as ext_strings

        strings.update(ext_strings())
    return strings


def build_eval_windows(store_dir: str | Path, datasets: bool = True) -> tuple[np.ndarray, dict]:
    """Build (or rebuild) the window store under store_dir."""
    import json
    import time

    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    strings = evaluation_strings(datasets)
    counts = {}
    parts = []
    for k, ss in strings.items():
        arr = hash_strings(ss)
        counts[k] = {"strings": len(set(ss)), "windows": int(len(arr))}
        parts.append(arr)
    allh = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.uint64)
    npy = store_dir / "eval_windows.npy"
    np.save(str(npy) + ".tmp.npy", allh)
    Path(str(npy) + ".tmp.npy").replace(npy)
    info = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "window_chars": WIN, "n_windows": int(len(allh)), "sources": counts,
            "sha256_npy": hashlib.sha256(npy.read_bytes()).hexdigest(), "seconds": round(time.time() - t0)}
    (store_dir / "eval_windows.json").write_text(json.dumps(info, indent=1))
    return allh, info


# ---------------------------------------------------------------- checks 1-6
def check1_answer_tokens_in_context(ans_n: str, ctx_n: str) -> bool:
    toks = {t.strip(".,;:!?\"'()[]{}") for t in re.split(r"[\s\-]+", ctx_n)}
    need = [t.strip(".,;:!?\"'()[]{}") for t in re.split(r"[\s\-]+", ans_n)]
    need = [t for t in need if t]
    return bool(need) and all(t in toks for t in need)


def check2_answer_not_demo_tail(ans_n: str, context: str) -> bool:
    """the normalised answer is not the trailing token sequence (after the last ':' / '=' / 'is written' cue) of any context line carrying such a cue"""
    for line in context.split("\n"):
        if not line.strip() or not CUE_SPLIT.search(line):
            continue
        tail = norm_answer(CUE_SPLIT.split(line)[-1])
        ln = norm_answer(line)
        if tail == ans_n or ln.endswith(" " + ans_n):
            return False
    return True


def content_words(s: Any) -> set[str]:
    return {w for w in WORD_RE.findall(str(s).lower()) if w not in STOPWORDS}


def parse_demos(context: str) -> tuple[list[dict], str | None]:
    """demonstration lines: 'Output:' (then 'Label:') lines -> output = text after the last cue, input = text before it; fallback: lines with an arrow"""

    def by_cue(cue: str) -> list[dict]:
        out = []
        for line in context.split("\n"):
            ms = list(re.finditer(cue, line, re.I))
            if not ms:
                continue
            o = line[ms[-1].end() :].strip()
            inp = ARROW_RE.sub(" ", PREFIX_RE.sub("", line[: ms[-1].start()], count=1)).strip()
            if o:
                out.append({"input": inp, "output": o, "output_norm": norm_answer(o), "line": line.strip()})
        return out

    for cue in (r"output\s*:", r"label\s*:"):
        d = by_cue(cue)
        if d:
            return d, cue
    out = []
    for line in context.split("\n"):
        ms = list(ARROW_RE.finditer(line))
        if ms and line[ms[-1].end() :].strip():
            out.append({"input": PREFIX_RE.sub("", line[: ms[-1].start()], count=1).strip(), "output": line[ms[-1].end() :].strip(), "output_norm": norm_answer(line[ms[-1].end() :]), "line": line.strip()})
    return out, ("arrow" if out else None)


def check6(ex: dict, t: str, class_norms: set[str]) -> dict:
    """Check 6: sub-checks (1)-(5) for type h; (5) only for the other topped-up types. Returns every sub-check, `enforced` and `pass`."""
    demos, mode = parse_demos(ex["context"])
    ans = norm_answer(ex["answer"])
    outs = [d["output_norm"] for d in demos]
    distinct = set(outs)
    outset = distinct | {ans}
    r: dict[str, Any] = {"mode": mode, "n_demos": len(demos), "n_distinct_outputs": len(distinct), "class_name_hits": sorted(outset & class_norms)}
    r["m6_5_no_dataset_class_name"] = not r["class_name_hits"]
    if t != "h":
        r["enforced"] = ["m6_5_no_dataset_class_name"]
        r["pass"] = r["m6_5_no_dataset_class_name"]
        return r
    cc = norm_answer(ex.get("copy_candidate") or "")
    pc = norm_answer(ex.get("prior_candidate") or "")
    kt = str(ex.get("key_term") or "")
    qw = content_words(ex["query"])
    ktw = content_words(kt)
    copy_demos = [d for d in demos if d["output_norm"] == cc]
    ans_demos = [d for d in demos if d["output_norm"] == ans]
    r["m6_demos_parsed"] = len(demos) > 0
    r["m6_1_candidates_are_demo_outputs_ne_answer"] = bool(cc) and bool(pc) and cc in distinct and pc in distinct and cc != ans and pc != ans and cc != pc
    r["m6_2_key_term"] = bool(ktw) and ktw <= qw and any(ktw <= content_words(d["input"]) for d in copy_demos)
    r["m6_3_answer_demo_shares_content_word"] = any(content_words(d["input"]) & qw for d in ans_demos)
    r["m6_4_five_outputs_two_unnamed"] = len(distinct) >= 5 and len(distinct - {ans, cc, pc}) >= 2
    r["detail"] = {"answer_norm": ans, "copy_norm": cc, "prior_norm": pc, "key_term_words": sorted(ktw), "query_content_words": sorted(qw)[:30], "n_copy_demos": len(copy_demos),
                   "n_answer_demos": len(ans_demos), "answer_demo_overlaps": [sorted(content_words(d["input"]) & qw) for d in ans_demos], "outputs": sorted(distinct)}
    r["enforced"] = ["m6_demos_parsed", "m6_1_candidates_are_demo_outputs_ne_answer", "m6_2_key_term", "m6_3_answer_demo_shares_content_word", "m6_4_five_outputs_two_unnamed", "m6_5_no_dataset_class_name"]
    r["pass"] = all(r[k] for k in r["enforced"])
    return r


def mechanical_checks(ex: dict, t: str, prompts: dict, tok, evalh: np.ndarray | None, class_norms: set[str] | None = None, with_check6: bool = True) -> dict:
    """All mechanical checks for one exercise (no API). Returns the fields of a verified record and the mechanical failure reasons."""
    forbid = re.compile(prompts["forbidden_regex"], re.I | re.M)
    ans_n = norm_answer(ex["answer"])
    ctx_n = norm_answer(ex["context"])
    pt = packed_text(ex, prompts["packed_text_format"])
    n = ntok(tok, pt) if tok is not None else None
    tok_min, tok_max = prompts["tokens"]["min"], prompts["tokens"]["max"]
    reasons = []
    m1 = check1_answer_tokens_in_context(ans_n, ctx_n)
    m2 = check2_answer_not_demo_tail(ans_n, ex["context"])
    m2_enforced = t == "a"
    hits = [m.group(0) for m in forbid.finditer(ex["context"] + "\n" + ex["query"] + "\n" + ex["answer"])]
    m3 = not hits
    hw = eval_hit(evalh, pt) if evalh is not None else None
    m4 = hw is None
    m5 = (tok_min <= n <= tok_max) if n is not None else True
    rec: dict[str, Any] = {"answer_norm": ans_n, "n_tokens": n, "mech1": m1, "mech2": m2, "mech2_enforced": m2_enforced, "mech3": m3, "forbidden_hits": hits[:10], "mech4": m4,
                           "eval_window": window_at(pt, np.uint64(hw)) if hw is not None else None, "mech5": m5}
    if not m1:
        reasons.append("CHECK1_answer_token_not_in_context")
    if not m2 and m2_enforced:
        reasons.append("CHECK2_answer_is_demo_answer")
    if not m3:
        reasons.append("CHECK3_forbidden")
    if not m4:
        reasons.append("CHECK4_eval_overlap")
    if not m5:
        reasons.append("CHECK5_tokens_short" if n < tok_min else "CHECK5_tokens_long")
    if with_check6:
        m6 = check6(ex, t, class_norms or set())
        rec["mech6"] = m6
        reasons += [CHECK6_REASON[k] for k in m6["enforced"] if not m6[k]]
    rec["mechanical_reasons"] = reasons
    return rec


def class_name_norms(prompts: dict) -> set[str]:
    names = prompts.get("banned_dataset_class_names", {}).get("names", {})
    return {norm_answer(x) for lst in names.values() for x in lst} - {""}


# ---------------------------------------------------------------- selection for packing
def retry_candidates(verified_path: str | Path, docs: list[dict], retry_type: str = "h") -> list[dict]:
    """Retry rule: documents of `retry_type` whose NON-retry verified record has pass == false get ONE more generation
    (unparsed generations are not exercises and are not retried; a document already retried is not retried again)."""
    recs = list(read_jsonl(verified_path))
    failed = {r["doc_sha256"] for r in recs if r["type"] == retry_type and not r["pass"] and not r.get("retry")}
    already = {r["doc_sha256"] for r in recs if r.get("retry")}
    return [d for d in docs if d["type"] == retry_type and d["sha256"] in failed and d["sha256"] not in already]


def select_passing(paths: list[str | Path], types: set[str] | list[str], packed_fmt: str, evalh: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """Verified records that enter a training pack: pass == true, type in `types`, de-duplicated by packed text (first by
    (pool, file, row, doc) order), and — when the window store is given — re-checked against it (check 4 at pack time).
    A retried document contributes its passing retry record only."""
    recs: list[dict] = []
    seen: set[str] = set()
    dup = 0
    src_counts: dict[str, int] = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for r in read_jsonl(path):
            if not r.get("pass") or r["type"] not in types:
                continue
            key = hashlib.sha256(packed_text(r["exercise"], packed_fmt).encode("utf-8")).hexdigest()
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            r["_src"] = str(path)
            recs.append(r)
            src_counts[str(path)] = src_counts.get(str(path), 0) + 1
    recs.sort(key=lambda r: (r["pool"], r["file"], r["row"], r["doc_sha256"]))
    keep = [r for r in recs if eval_hit(evalh, packed_text(r["exercise"], packed_fmt)) is None] if evalh is not None else recs
    return keep, {"duplicates_dropped": dup, "check4_dropped_at_pack": len(recs) - len(keep), "per_source": src_counts}

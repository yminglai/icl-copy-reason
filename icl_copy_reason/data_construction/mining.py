"""Mined real definition-use documents: the classifier-selected windows (arm D), the verifier-selected reference-unseen
windows (arm D2) and their dictionary-format rendering (arm D3), the pool surveys and the audits.

Inputs : the pretraining proxy stream and the packed common rows of the admixture run (manifest, packs, validation
         token files; config `data_dir`), the eight public pools (`documents.POOLS`), the API settings for the labelling /
         verification / audit models, and a mining directory for intermediate files (config `mine_dir`).
Outputs: arm D   pool.jsonl / random.jsonl -> cands.jsonl -> labels.jsonl -> clf.pkl + fit.json -> selected.jsonl +
                 synth_defuse_s0.u16 (the replacement rows) -> audit.json, overlap.json, finalize record
         arm D2  cands_<pool>.jsonl -> verified_<pool>.jsonl -> admitted.jsonl + synth_enriched_s0.u16 + pack.json
         arm D3  rendered.jsonl + synth_rendered_s0.u16 + render.json; audit.json, surprisal.json, proxies.json
         surveys pool_survey.json / pool_survey_words.json (candidate-filter yield over open pools); the reference
                 vocabulary refvocab.txt; the arm-D window audit and its three-model re-read.
Every step is deterministic given its seed; model names come from the config (defaults = the models used in the paper).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pickle
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..api_client import ChatClient
from ..config import ApiSettings
from ..io import read_json, write_json
from . import synthetic_docs as sd
from .documents import POOLS, list_pool_files

SEQ = 2048
MAX_CHARS = 8000  # the unit is a WINDOW = the document's first 8,000 characters, later cut to its first 2,048 tokens
SHINGLE = 64  # 64-token shingles (stride 1 in windows, 64-aligned in the packs / sanity samples)
WIN = 1023  # arm D2 window length in tokens (two windows + EOS per packed row)
ROWS = 14801  # replacement rows of the admixture runs
R_TOKENS = ROWS * SEQ
LABEL_MODELS = ["claude-sonnet-5", "gpt-5.5", "deepseek-v4-pro"]  # three-model majority labels (arm D) and audits
VERIFY_MODEL = "gpt-4.1"  # one call per candidate window (arm D2)
AUDIT_MODELS = LABEL_MODELS
CFG: dict[str, Any] = {"mine": None, "data": None, "crit": None, "out_data": None, "max_files": None}


def configure(
    mine_dir: str | Path,
    data_dir: str | Path,
    critique_dir: str | Path | None = None,
    out_data: str | Path | None = None,
    max_files: int | None = None,
) -> None:
    """Set the working directories: mine_dir (intermediate files), data_dir (manifest / packs / replacement-row outputs),
    critique_dir (reference vocabulary, window audits; default mine_dir), out_data (where the u16 row files go; default data_dir)."""
    CFG.update(
        {
            "mine": str(mine_dir),
            "data": str(data_dir),
            "crit": str(critique_dir or mine_dir),
            "out_data": str(out_data or data_dir),
            "max_files": max_files,
        }
    )
    for k in ("mine", "crit", "out_data"):
        os.makedirs(CFG[k], exist_ok=True)


def _p(key: str) -> str:
    if CFG[key] is None:
        raise SystemExit("call mining.configure(mine_dir, data_dir) first (or pass --mine-dir / --data-dir)")
    return CFG[key]


# ---------------------------------------------------------------- shared helpers


def sha(path: str | Path) -> str:
    hh = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            hh.update(chunk)
    return hh.hexdigest()


def jl(path: str | Path) -> list:
    return [json.loads(line) for line in open(path) if line.strip()]


def jl_iter(path: str | Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


_CLIENT: ChatClient | None = None


def client(settings: ApiSettings | None = None) -> ChatClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = ChatClient(settings or ApiSettings.from_env())
    return _CLIENT


def ask_json(
    model: str, prompt: str, max_tokens: int = 200, temperature: float | None = 0
) -> tuple[dict | None, str, str | None]:
    """One JSON-answer call with the record-returning client: a reply cut off by the budget is re-asked once at 2,000 tokens;
    a non-JSON reply is re-asked once with 'Return ONLY the JSON object.'. Returns (object or None, raw reply, error)."""
    c = client()
    r = c.complete(None, prompt, model, temperature, max_tokens)
    if r["error"]:
        return None, "", r["error"]
    m = re.search(r"\{.*\}", r["content"], flags=re.S)
    if m:
        try:
            return json.loads(m.group(0)), r["content"], None
        except Exception:  # noqa: BLE001
            pass
    trunc = r["finish_reason"] == "length"
    r2 = c.complete(None, prompt + "\nReturn ONLY the JSON object.", model, temperature, 2000 if trunc else max_tokens)
    if r2["error"]:
        return None, "", r2["error"]
    m = re.search(r"\{.*\}", r2["content"], flags=re.S)
    if m:
        try:
            return json.loads(m.group(0)), r2["content"], None
        except Exception:  # noqa: BLE001
            return None, r2["content"], "invalid_answer"
    return None, r2["content"], ("truncated_twice" if r2["finish_reason"] == "length" else "invalid_answer")


def load_manifest_tokenizer(data_dir: str | Path):
    """The base checkpoint's tokenizer named in the admixture manifest (manifest_s0.json: base, revision, eos, replacement_rows)."""
    from transformers import AutoTokenizer

    man = read_json(Path(data_dir) / "manifest_s0.json")
    return AutoTokenizer.from_pretrained(man["base"], revision=man["revision"]), man


# ---------------------------------------------------------------- arm D: classifier-selected definition-use windows
PREFILTER = [
    r"\bis defined as\b",
    r"\brefers to\b",
    r"\bis called\b",
    r"\bis known as\b",
    r"\bdenotes?\b",
    r"\bdenoted by\b",
    r"\blet [^.\n]{1,40}\bbe\b",
    r"\bglossary\b",
    r"\bdefinition\b",
    r"\bmeans that\b",
    r"\bdef [a-zA-Z_]\w*\(",
    r":=",
    r"\bthe term\b",
    r"\bwe call\b",
    r"\bis said to be\b",
    r"\bstands for\b",
    r"\bin other words\b",
    r"\brule\b.{0,40}\b(states|says)\b",
]
PREFILTER_RE = re.compile("|".join(PREFILTER), re.IGNORECASE)

# ---- exclusion filter (fixed before the run: evaluation strings may not occur in any arm-D document) -----------------------------------------------
EXCLUDE_STRINGS = ["In this language:", "I_WALK", "I_RUN", "I_LOOK", "I_JUMP", "I_TURN_LEFT", "I_TURN_RIGHT"]
EXCLUDE_WORDS = [
    "dax",
    "wif",
    "lug",
    "zup",
    "fep",
    "kiki",
    "blicket",
    "tufa",
    "gazzer",
    "mip",
    "blorp",
    "kleek",
    "voon",
    "snerp",
    "quab",
    "flim",
    "dorn",
    "yubb",
    "KEL",
    "MOR",
    "TAV",
    "ZUN",
    "PIQ",
    "DOF",
    "WEX",
    "BRU",
    "Quechua",
    "Quichua",
    "Runasimi",
    "Kichwa",
]
EXCLUDE_RE = re.compile(r"\b(" + "|".join(map(re.escape, EXCLUDE_WORDS)) + r")\b|\d [@#&%~^] \d")
PATS_RE = [re.compile(p, re.I) for p in PREFILTER]
EXCL = EXCLUDE_RE


def excluded(text):
    return any(s in text for s in EXCLUDE_STRINGS) or EXCLUDE_RE.search(text) is not None


RUBRIC = """You label documents for a pretraining-data study. Decide whether the document is a DEFINITION-THEN-USE document.

POSITIVE (label 1) iff BOTH hold inside this one document:
 (a) it explicitly INTRODUCES at least one term, symbol, notation, rule or function - e.g. a definition sentence ("X is defined as ...", "we call ...",
     "X refers to ..."), a glossary entry, "let X denote ...", a function or variable definition in code, a stated rule or convention; AND
 (b) LATER in the same document it USES that introduced item in a new instance - an example sentence employing the term, a derivation or
     calculation with the notation, a call of the defined function, an exercise or worked example applying the rule.
NEGATIVE (label 0): a definition with no later use; use of terms without an in-document definition; a bare glossary or list of definitions
with no application; general prose, news, narrative, advertising; code that only defines without calling; documents where (b) is only a
restatement of the definition rather than a new instance.

Answer with ONE line of JSON and nothing else: {"label": 0 or 1, "kind": one of ["glossary","math_notation","code_def_call","rule_exercise",
"term_definition","other","none"], "evidence": "<= 25 words quoting the definition and the later use"}"""


def arm_d_pool(a):
    from ..training.pretrain import stream_docs  # the admixture trainer's document stream

    os.makedirs(a.out, exist_ok=True)
    skip = a.skip_docs
    if skip is None:
        skip = json.load(open(f"{a.data}/manifest_s0.json"))["n_docs"]
    st = {
        "skip_docs": skip,
        "seen": 0,
        "excluded": 0,
        "prefilter_pass": 0,
        "random_written": 0,
        "by_source": {},
        "t0": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    fp = open(f"{a.out}/pool.jsonl", "w")
    fr = open(f"{a.out}/random.jsonl", "w")
    t0 = time.time()
    for k, (src, did, text) in enumerate(stream_docs("smollm17", 0)):
        if k < skip:
            continue
        st["seen"] += 1
        pos = k
        if excluded(text):
            st["excluded"] += 1
            continue
        hit = PREFILTER_RE.search(text[:MAX_CHARS]) is not None
        if st["seen"] % a.random_every == 0:
            fr.write(
                json.dumps(
                    {
                        "pos": pos,
                        "src": src,
                        "id": did,
                        "prefilter": hit,
                        "text": text[:MAX_CHARS],
                        "full_chars": len(text),
                    }
                )
                + "\n"
            )
            st["random_written"] += 1
        if hit:
            fp.write(
                json.dumps({"pos": pos, "src": src, "id": did, "text": text[:MAX_CHARS], "full_chars": len(text)})
                + "\n"
            )
            st["prefilter_pass"] += 1
            st["by_source"][src] = st["by_source"].get(src, 0) + 1
            if st["prefilter_pass"] >= a.target:
                break
        if st["seen"] % 100000 == 0:
            print(
                json.dumps({k: v for k, v in st.items() if k != "by_source"}) + f" {time.time() - t0:.0f}s", flush=True
            )
    fp.close()
    fr.close()
    st["t1"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    st["prefilter_rate_after_exclusion"] = st["prefilter_pass"] / max(1, st["seen"] - st["excluded"])
    json.dump(st, open(f"{a.out}/pool_stats.json", "w"), indent=1)
    print(json.dumps(st, indent=1))


# ------------------------------------------------------------------------------------------------------------------ candidates
def arm_d_candidates(a):
    rng = random.Random(a.seed)
    pool = jl(f"{a.mine}/pool.jsonl")
    seen = set()
    uniq = []
    for i, d in enumerate(pool):
        if d["text"] not in seen:
            seen.add(d["text"])
            uniq.append(i)
    idx = sorted(rng.sample(uniq, a.n))
    print(f"pool {len(pool)} documents, {len(uniq)} unique windows")
    with open(f"{a.mine}/cands.jsonl", "w") as f:
        for i in idx:
            f.write(json.dumps(pool[i]) + "\n")
    rnd = jl(f"{a.mine}/random.jsonl")
    idr = sorted(rng.sample(range(len(rnd)), min(a.n_random, len(rnd))))
    with open(f"{a.mine}/cands_random.jsonl", "w") as f:
        for i in idr:
            f.write(json.dumps(rnd[i]) + "\n")
    print(f"candidates: {len(idx)} from pool ({len(pool)}), {len(idr)} from random ({len(rnd)})")


# ------------------------------------------------------------------------------------------------------------------ label
def _ask(model: str, text: str) -> dict:
    """One rubric call (temperature 0 where the endpoint allows it); up to five attempts; returns label / kind / evidence or an error."""
    doc = text[:MAX_CHARS]
    msg = RUBRIC + "\n\n=== DOCUMENT ===\n" + doc + "\n=== END ==="
    err = None
    for attempt in range(5):
        j, _raw, e = ask_json(model, msg, max_tokens=400)
        try:
            lab = int(j["label"])
            assert lab in (0, 1)
            return {"label": lab, "kind": j.get("kind"), "evidence": str(j.get("evidence", ""))[:200]}
        except Exception as ex:  # noqa: BLE001
            err = e or f"{type(ex).__name__}: {str(ex)[:120]}"
            time.sleep(5 * (attempt + 1))
    return {"label": None, "error": err}


def arm_d_label(a):
    docs = jl(a.inp)
    models = a.models.split(",")
    done = {}
    if os.path.exists(a.out):
        for r in jl(a.out):
            done[r["pos"]] = r
    todo = [
        d
        for d in docs
        if d["pos"] not in done or any(done[d["pos"]]["votes"].get(m, {}).get("label") is None for m in models)
    ]
    print(f"labelling {len(todo)} of {len(docs)} documents with {models}", flush=True)

    def work(d):
        votes = {m: _ask(m, d["text"]) for m in models}
        labs = [v["label"] for v in votes.values() if v["label"] is not None]
        gold = int(sum(labs) * 2 > len(labs)) if len(labs) >= 2 else None
        return {
            "pos": d["pos"],
            "src": d["src"],
            "id": d["id"],
            "votes": votes,
            "gold": gold,
            "n_valid": len(labs),
            "unanimous": len(set(labs)) == 1 if labs else None,
            "truncated": len(d["text"]) > MAX_CHARS,
        }

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, d) for d in todo]
        for k, f in enumerate(as_completed(futs)):
            r = f.result()
            done[r["pos"]] = r
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(todo)}", flush=True)
    with open(a.out, "w") as f:
        for d in docs:
            if d["pos"] in done:
                f.write(json.dumps(done[d["pos"]]) + "\n")
    rows = [done[d["pos"]] for d in docs if d["pos"] in done]
    g = [r["gold"] for r in rows if r["gold"] is not None]
    summ = {
        "n": len(rows),
        "n_gold": len(g),
        "positive_rate": float(np.mean(g)) if g else None,
        "unanimous_rate": float(np.mean([r["unanimous"] for r in rows if r["unanimous"] is not None]))
        if rows
        else None,
        "per_model_positive_rate": {
            m: float(np.mean([r["votes"][m]["label"] for r in rows if r["votes"][m]["label"] is not None]))
            for m in models
        },
        "per_model_failures": {m: sum(1 for r in rows if r["votes"][m]["label"] is None) for m in models},
        "truncated": sum(1 for r in rows if r["truncated"]),
    }
    json.dump(summ, open(a.out.replace(".jsonl", "_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


# ------------------------------------------------------------------------------------------------------------------ fit
def make_clf(C):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    feats = FeatureUnion(
        [
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)),
            (
                "c",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300000
                ),
            ),
        ]
    )
    return Pipeline([("f", feats), ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])


def arm_d_fit(a):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.model_selection import StratifiedKFold, train_test_split

    labs = jl(f"{a.mine}/labels.jsonl")
    texts = {d["pos"]: d["text"][:MAX_CHARS] for d in jl(f"{a.mine}/cands.jsonl")}
    rows = [(texts[r["pos"]], r["gold"]) for r in labs if r["gold"] is not None and r["pos"] in texts]
    X = [t for t, _ in rows]
    y = np.array([g for _, g in rows])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=200, random_state=a.seed, stratify=y)
    cv = {}
    for C in (0.1, 0.3, 1.0):
        aps = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=a.seed).split(Xtr, ytr):
            clf = make_clf(C).fit([Xtr[i] for i in tr], ytr[tr])
            aps.append(average_precision_score(ytr[va], clf.predict_proba([Xtr[i] for i in va])[:, 1]))
        cv[C] = float(np.mean(aps))
    Cbest = max(cv, key=cv.get)
    clf = make_clf(Cbest).fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    prec, rec, thr = precision_recall_curve(yte, s)
    # thresholds thr[i] correspond to prec[i], rec[i] (the last prec/rec point has no threshold)
    qual = [(float(t), float(p), float(r)) for p, r, t in zip(prec[:-1], rec[:-1], thr) if p >= 0.80 and r >= 0.30]
    if qual:
        t_sel, p_sel, r_sel = min(qual, key=lambda x: x[0])
        qualifies = True
    else:
        # no fallback threshold - arm D is not run; the best precision at recall >= .30 is recorded for the report only
        cand = [(float(t), float(p), float(r)) for p, r, t in zip(prec[:-1], rec[:-1], thr) if r >= 0.30]
        t_sel, p_sel, r_sel = max(cand, key=lambda x: x[1]) if cand else (None, None, None)
        qualifies = False
    pickle.dump(clf, open(f"{a.mine}/clf.pkl", "wb"))
    pred = (s >= t_sel).astype(int) if t_sel is not None else np.zeros_like(yte)
    cm = {
        "tp": int(((pred == 1) & (yte == 1)).sum()),
        "fp": int(((pred == 1) & (yte == 0)).sum()),
        "fn": int(((pred == 0) & (yte == 1)).sum()),
        "tn": int(((pred == 0) & (yte == 0)).sum()),
    }
    agree = float(np.mean([r["unanimous"] for r in labs if r["unanimous"] is not None]))
    fit = {
        "n_labelled": len(rows),
        "n_train": len(Xtr),
        "n_validation": len(Xte),
        "positives_train": int(ytr.sum()),
        "positives_validation": int(yte.sum()),
        "cv_average_precision": cv,
        "validation_confusion": cm,
        "labeller_unanimity": agree,
        "note": "the 200 held-out windows choose the threshold (validation, not an independent test); the operational gate is the audit",
        "C": Cbest,
        "validation_average_precision": float(average_precision_score(yte, s)),
        "threshold": t_sel,
        "validation_precision_at_threshold": p_sel,
        "validation_recall_at_threshold": r_sel,
        "qualifies": qualifies,
        "rule": "lowest held-out score with precision >= .80 at recall >= .30; stratified 700/200 split; C by 5-fold CV average precision",
        "pr_curve": [[float(p), float(r), float(t)] for p, r, t in zip(prec[:-1], rec[:-1], thr)],
        "clf_sha256": sha(f"{a.mine}/clf.pkl"),
        "seed": a.seed,
    }
    json.dump(fit, open(f"{a.mine}/fit.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in fit.items() if k != "pr_curve"}, indent=1))


# ------------------------------------------------------------------------------------------------------------------ select
def arm_d_select(a):
    from transformers import AutoTokenizer

    man = json.load(open(f"{a.data}/manifest_s0.json"))
    rows = man["replacement_rows"]
    eos = man["eos"]
    fit = json.load(open(f"{a.mine}/fit.json"))
    assert fit["qualifies"], "classifier did not qualify: arm D is dropped"
    thr = fit["threshold"]
    clf = pickle.load(open(f"{a.mine}/clf.pkl", "rb"))
    tok = AutoTokenizer.from_pretrained(man["base"], revision=man["revision"])
    need = int(1.1 * len(rows) * SEQ)
    got = 0
    sel = []
    scanned = 0
    by_src = {}
    # 64-aligned shingles of the common packed rows (training + calibration slice) and both sanity samples
    t0 = time.time()
    ref = set()
    packs = np.load(f"{a.data}/packs_s0.u16", mmap_mode="r")
    for r0 in range(0, packs.shape[0], 4096):
        blk = np.asarray(packs[r0 : r0 + 4096]).reshape(-1, SEQ // SHINGLE, SHINGLE)
        for row in blk:
            for sh in row:
                ref.add(hash(sh.tobytes()))
    for k in ("fwe", "pile"):
        v = np.load(f"{a.data}/val_{k}.u16", mmap_mode="r")
        n = (len(v) // SHINGLE) * SHINGLE
        for sh in np.asarray(v[:n]).reshape(-1, SHINGLE):
            ref.add(hash(sh.tobytes()))
    print(f"reference shingles: {len(ref)} ({time.time() - t0:.0f}s)", flush=True)

    def shingle_hit(ids):
        arr = np.asarray(ids, dtype=np.uint16)
        if len(arr) < SHINGLE:
            return False
        win = np.lib.stride_tricks.sliding_window_view(arr, SHINGLE)
        return any(hash(w.tobytes()) in ref for w in win)

    fsel = open(f"{a.mine}/selected.jsonl", "w")
    counts = {"shingle_excl": 0, "tok_trunc": 0}
    with open(f"{a.mine}/pool.jsonl") as f:
        batch = []

        def flush():
            nonlocal got, scanned
            if not batch:
                return
            sc = clf.predict_proba([d["text"][:MAX_CHARS] for d in batch])[:, 1]
            scanned += len(batch)
            keep = [d for d, s_ in zip(batch, sc) if s_ >= thr]
            if keep:
                enc = tok([d["text"] for d in keep], add_special_tokens=False)["input_ids"]
                for d, ids, s_ in zip(keep, enc, [s_ for s_ in sc if s_ >= thr]):
                    if got >= need:
                        break
                    if len(ids) > SEQ - 1:
                        ids = ids[: SEQ - 1]
                        trunc = True
                    else:
                        trunc = False
                    if shingle_hit(ids):
                        counts["shingle_excl"] += 1
                        continue
                    counts["tok_trunc"] += int(trunc)
                    d2 = {
                        "pos": d["pos"],
                        "src": d["src"],
                        "id": d["id"],
                        "score": float(s_),
                        "n_tokens": len(ids) + 1,
                        "token_truncated": trunc,
                        "ids": [int(x) for x in ids],
                        "text": tok.decode(ids),
                    }
                    fsel.write(json.dumps(d2) + "\n")
                    sel.append({k: v for k, v in d2.items() if k not in ("text", "ids")})
                    got += len(ids) + 1
                    by_src[d["src"]] = by_src.get(d["src"], 0) + 1
            batch.clear()

        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= 512:
                flush()
                if got >= need:
                    break
        flush()
    fsel.close()
    assert got >= need, f"pool exhausted: {got} < {need} tokens (scanned {scanned})"
    # pack in stream order exactly as synth_packs packs the B / B_ctrl documents: a continuous token stream with EOS separators cut into rows of 2048
    # (the last window of a row is cut at the row boundary in every arm - matched); windows cut at a boundary are counted
    mm = np.lib.format.open_memmap(f"{a.data}/synth_defuse_s0.u16", mode="w+", dtype=np.uint16, shape=(len(rows), SEQ))
    buf = []
    r = 0
    n_docs_packed = 0
    n_cut = 0
    complete_tokens = 0
    contrib = []
    for d in jl(f"{a.mine}/selected.jsonl"):
        ids = d["ids"] + [eos]
        start_row, start_off = r, len(buf)
        buf.extend(ids)
        n_docs_packed += 1
        while len(buf) >= SEQ and r < len(rows):
            mm[r] = np.array(buf[:SEQ], dtype=np.uint16)
            buf = buf[SEQ:]
            r += 1
        cut = (
            (r > start_row)
            and (len(buf) > 0 or r >= len(rows))
            and (start_off + len(ids) > SEQ * (r - start_row) if r > start_row else False)
        )
        n_cut += int(cut)
        complete_tokens += 0 if cut else len(ids)
        contrib.append((d["pos"], len(ids), cut))
        if r >= len(rows):
            break
    mm.flush()
    assert r == len(rows), f"packed {r} rows, need {len(rows)}"
    json.dump(contrib, open(f"{a.mine}/contrib.json", "w"))
    info = {
        "threshold": thr,
        "n_selected": len(sel),
        "n_windows_packed": n_docs_packed,
        "n_scanned": scanned,
        "selection_rate_in_pool": len(sel) / max(1, scanned),
        "tokens_selected": got,
        "tokens_needed": need,
        "rows": len(rows),
        "by_source": by_src,
        "mean_score": float(np.mean([d["score"] for d in sel])),
        "median_window_tokens": float(np.median([d["n_tokens"] for d in sel])),
        "windows_token_truncated_to_2048": counts["tok_trunc"],
        "windows_excluded_by_shingle": counts["shingle_excl"],
        "reference_shingles": len(ref),
        "windows_cut_at_row_boundary": n_cut,
        "share_windows_cut": n_cut / max(1, n_docs_packed),
        "token_share_complete_windows": complete_tokens / (len(rows) * SEQ),
        "sha256_synth_defuse": sha(f"{a.data}/synth_defuse_s0.u16"),
        "seconds": time.time() - t0,
    }
    json.dump(info, open(f"{a.mine}/select.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


# ------------------------------------------------------------------------------------------------------------------ audit
def arm_d_audit(a):
    rng = random.Random(a.seed)
    sel = jl(f"{a.mine}/selected.jsonl")
    contrib = {p: n for p, n, _ in json.load(open(f"{a.mine}/contrib.json"))}
    packed = [i for i, d in enumerate(sel) if d["pos"] in contrib]
    w = [contrib[sel[i]["pos"]] for i in packed]
    chosen = set()
    while len(chosen) < min(a.n, len(packed)):
        chosen.add(
            rng.choices(packed, weights=w, k=1)[0]
        )  # probability proportional to the tokens a window contributes
    idx = sorted(chosen)
    with open(f"{a.mine}/audit_docs.jsonl", "w") as f:
        for i in idx:
            f.write(json.dumps(sel[i]) + "\n")
    a2 = SimpleNamespace(
        inp=f"{a.mine}/audit_docs.jsonl",
        out=f"{a.mine}/audit_labels.jsonl",
        models=",".join(LABEL_MODELS),
        workers=a.workers,
    )
    arm_d_label(a2)
    labs = jl(f"{a.mine}/audit_labels.jsonl")
    g = [r["gold"] for r in labs if r["gold"] is not None]
    {r["pos"]: r["src"] for r in labs}
    by = {}
    for r in labs:
        if r["gold"] is not None:
            by.setdefault(r["src"], []).append(r["gold"])
    res = {
        "n": len(g),
        "operational_consensus_precision": float(np.mean(g)) if g else None,
        "gate": 0.70,
        "passes": bool(g) and float(np.mean(g)) >= 0.70,
        "sampling": "proportional to contributed tokens",
        "by_source": {k: {"n": len(v), "precision": float(np.mean(v))} for k, v in by.items()},
        "seed": a.seed,
    }
    json.dump(res, open(f"{a.mine}/audit.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------------------------ overlap
def arm_d_overlap(a):
    big = "\n".join(d["text"] for d in jl(f"{a.mine}/selected.jsonl"))
    res = {}
    for name in ("scan", "elemental", "miniscan", "arith", "chainmap"):
        p = json.load(open(f"{a.panels}/{name}.json"))
        flagged = 0
        examples = []
        for it in p["items"]:
            lines = [ln for ln in it["prompt"].split("\n") if len(ln) >= 8]
            hit = next((ln for ln in lines if ln in big), None)
            if hit is not None:
                flagged += 1
                if len(examples) < 5:
                    examples.append(hit[:80])
        res[name] = {
            "n_items": p["n"],
            "flagged": flagged,
            "fraction": flagged / p["n"],
            "excluded_for_D": flagged / p["n"] > 0.05,
            "examples": examples,
        }
    json.dump(res, open(f"{a.mine}/overlap.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------------------------ finalize
def arm_d_record(a):
    out = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rubric": RUBRIC,
        "prefilter": PREFILTER,
        "exclude_strings": EXCLUDE_STRINGS,
        "exclude_words": EXCLUDE_WORDS,
        "exclude_regex": EXCLUDE_RE.pattern,
        "label_models": LABEL_MODELS,
        "max_chars": MAX_CHARS,
    }
    for k in ("pool_stats", "labels_summary", "labels_random_summary", "fit", "select", "audit", "overlap"):
        f = f"{a.mine}/{k}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            out[k] = {kk: vv for kk, vv in d.items() if kk != "pr_curve"}
    out["sha256"] = {}
    for f in (
        "pool.jsonl",
        "random.jsonl",
        "cands.jsonl",
        "cands_random.jsonl",
        "labels.jsonl",
        "labels_random.jsonl",
        "clf.pkl",
        "selected.jsonl",
        "audit_labels.jsonl",
    ):
        if os.path.exists(f"{a.mine}/{f}"):
            out["sha256"][f] = sha(f"{a.mine}/{f}")
    for f in ("synth_defuse_s0.u16",):
        if os.path.exists(f"{a.data}/{f}"):
            out["sha256"][f] = sha(f"{a.data}/{f}")
    out["arm_D_runs"] = (
        bool(out.get("fit", {}).get("qualifies"))
        and bool(out.get("audit", {}).get("passes"))
        and "synth_defuse_s0.u16" in out["sha256"]
        and "overlap" in out
    )
    out["estimand"] = (
        "effect of the classifier-selected real-document recipe, not of the definition->use property in isolation"
    )
    out["assays_excluded_for_D"] = [k for k, v in out.get("overlap", {}).items() if v.get("excluded_for_D")]
    json.dump(out, open(f"{_p('mine')}/arm_d_record.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rubric", "prefilter", "exclude_words")}, indent=1))


# ---------------------------------------------------------------- the defined-term extractor and format statistics (window audit)
SEPS = [" : ", " => ", " -> ", " | ", " ~ ", " is written ", " = ", ":=", " - ", " — ", "\t"]
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def extract_term(sent):
    """Heuristic defined-term extraction for the frozen prefilter patterns; returns (term, pattern) or (None, pattern)."""
    s = sent.strip()
    m = re.search(
        r"(?:called|known as|termed|denoted by|denoted|the term)\s+[\"“']?([A-Za-z][\w\-]*(?:\s[A-Za-z][\w\-]*){0,2})[\"”']?",
        s,
    )
    if m:
        return m.group(1), "after"
    m = re.search(r"\b[Ll]et\s+([^\s,]{1,40})\s+(?:be|denote)\b", s)
    if m:
        return m.group(1), "let"
    m = re.search(r"\bdef\s+([a-zA-Z_]\w*)\(", s)
    if m:
        return m.group(1), "def"
    m = re.search(
        r"([A-Za-z][\w\-]*(?:\s[A-Za-z][\w\-]*){0,2})\s+(?:is defined as|refers to|denotes|stands for|means that|is said to be|:=)",
        s,
    )
    if m:
        t = m.group(1)
        toks = t.split()
        drop = {"it", "this", "that", "which", "he", "she", "they", "there", "what", "who"}
        toks = [w for w in toks if w.lower() not in {"the", "a", "an", "term", "word", "this", "that", "of"}]
        if toks and toks[-1].lower() not in drop:
            return " ".join(toks[-2:] if len(toks) > 1 and toks[-2][0].isupper() else toks[-1:]), "before"
    m = re.search(r"\bdefinition of\s+(?:the\s+|a\s+)?([A-Za-z][\w\-]*(?:\s[A-Za-z][\w\-]*)?)", s)
    if m:
        return m.group(1), "defof"
    return None, None


def first_definition(text):
    for i, s in enumerate(sentences(text)):
        for p in PREFILTER:
            if re.search(p, s, flags=re.I):
                t, how = extract_term(s)
                return i, s[:300], t, how, p
    return None, None, None, None, None


def count_uses(term, text):
    return len(re.findall(r"(?<![\w\-])" + re.escape(term) + r"(?![\w\-])", text, flags=re.I))


def fmt_stats(text, tok, ref_vocab):
    lines = [ln for ln in text.split("\n") if ln.strip()]
    tab = sum(1 for ln in lines if any(sep in ln for sep in SEPS) and len(ln) < 120 and not ln.rstrip().endswith("."))
    ltok = [len(tok(ln, add_special_tokens=False)["input_ids"]) for ln in lines] or [0]
    words = [w.lower() for w in WORD.findall(text)]
    nonce = sum(1 for w in words if w not in ref_vocab) / max(1, len(words))
    return {
        "n_lines": len(lines),
        "table_line_share": tab / max(1, len(lines)),
        "mean_line_tokens": float(np.mean(ltok)),
        "nonce_share": nonce,
        "n_words": len(words),
    }


def window_audit(seed: int = 20260925, n: int = 40) -> dict:
    """Fixed-seed audit of the arm-D selected windows (sample proportional to contributed tokens): defined term, piece count,
    reference frequency, later uses, format statistics versus arm-B rows -> <critique_dir>/armD_audit.json."""
    SEED, N = seed, n
    tok, man = load_manifest_tokenizer(_p("data"))
    # reference sample: unigram counts over random.jsonl
    ref = collections.Counter()
    n_ref_words = 0
    n_ref_docs = 0
    with open(f"{_p('mine')}/random.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ws = [w.lower() for w in WORD.findall(r["text"])]
            ref.update(ws)
            n_ref_words += len(ws)
            n_ref_docs += 1
    ref_vocab = set(ref)
    # sample proportional to n_tokens
    recs = []
    with open(f"{_p('mine')}/selected.jsonl") as f:
        for line in f:
            r = json.loads(line)
            recs.append((r["pos"], r["src"], r["n_tokens"], r["score"], r["ids"], r["id"]))
    w = np.array([r[2] for r in recs], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(recs), size=N, replace=False, p=w / w.sum())
    sample = []
    for i in sorted(idx.tolist()):
        pos, src, ntok, score, ids, did = recs[i]
        text = tok.decode(ids)
        si, dsent, term, how, pat = first_definition(text)
        e = {
            "pos": pos,
            "src": src,
            "n_tokens": ntok,
            "score": score,
            "id": did,
            "def_sentence_index": si,
            "def_sentence": dsent,
            "term": term,
            "term_how": how,
            "pattern": pat,
        }
        if term:
            e["pieces"] = len(tok(" " + term, add_special_tokens=False)["input_ids"])
            tl = term.lower()
            e["ref_count"] = sum(ref[x] for x in set(WORD.findall(tl))) if " " in tl else ref.get(tl, 0)
            e["ref_per_million_words"] = 1e6 * e["ref_count"] / n_ref_words
            e["novel_proxy"] = bool(e["ref_count"] == 0 and e["pieces"] >= 2)
            sents = sentences(text)
            after = " ".join(sents[si + 1 :]) if si is not None else ""
            e["later_uses"] = count_uses(term, after)
            e["total_uses"] = count_uses(term, text)
        e.update(fmt_stats(text, tok, ref_vocab))
        e["head"] = text[:400]
        sample.append(e)
    # arm B reference rows
    b = np.load(f"{_p('data')}/synth_compose_s0.u16", mmap_mode="r")  # npy memmap (rows, 2048)
    bidx = np.random.default_rng(SEED).choice(b.shape[0], size=N, replace=False)
    bstats = [fmt_stats(tok.decode(b[i].tolist()), tok, ref_vocab) for i in bidx]

    def agg(xs, k):
        xs = [x[k] for x in xs if k in x and x[k] is not None]
        return {
            "n": len(xs),
            "mean": float(np.mean(xs)) if xs else None,
            "median": float(np.median(xs)) if xs else None,
        }

    with_term = [e for e in sample if e.get("term")]
    summary = {
        "seed": SEED,
        "n": N,
        "n_ref_docs": n_ref_docs,
        "n_ref_words": n_ref_words,
        "ref_vocab_size": len(ref_vocab),
        "by_source": dict(collections.Counter(e["src"] for e in sample)),
        "def_found": sum(1 for e in sample if e["def_sentence"] is not None),
        "term_extracted": len(with_term),
        "pieces": agg(with_term, "pieces"),
        "pieces_eq_1": sum(1 for e in with_term if e["pieces"] == 1),
        "ref_count_zero": sum(1 for e in with_term if e["ref_count"] == 0),
        "novel_proxy": sum(1 for e in with_term if e["novel_proxy"]),
        "ref_per_million": agg(with_term, "ref_per_million_words"),
        "later_uses": agg(with_term, "later_uses"),
        "later_uses_ge2": sum(1 for e in with_term if e["later_uses"] >= 2),
        "later_uses_0": sum(1 for e in with_term if e["later_uses"] == 0),
        "def_sentence_index": agg(sample, "def_sentence_index"),
        "format_D": {k: agg(sample, k) for k in ["table_line_share", "mean_line_tokens", "nonce_share", "n_lines"]},
        "format_B": {k: agg(bstats, k) for k in ["table_line_share", "mean_line_tokens", "nonce_share", "n_lines"]},
    }
    json.dump(
        {"summary": summary, "sample": sample, "armB_rows": bstats},
        open(f"{_p('crit')}/armD_audit.json", "w"),
        indent=1,
    )
    print(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------- arm D2 / D3: verifier-selected reference-unseen windows and the rendering
WORDTERM = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{3,}( [A-Za-z][A-Za-z0-9_\-]+)?$")
PARA = re.compile(r"\n\s*\n")
SENT = re.compile(r"(?<=[.!?])\s+")
BANNED = {w.lower() for w in sd.ALL_WORDS} | {m.lower() for m in sd.MARKERS}
VERIFY_PROMPT = (
    "You check one pretraining text window for a data study. The candidate term is: {T}\n"
    'Answer in JSON only: {{"q1": true/false, "q2": true/false, "q3": true/false}} where\n'
    "q1 = the window explicitly introduces or defines the term (a definition, 'X is called ...', 'let X denote ...', a glossary-style entry); a discourse phrase such as 'in other words' or 'which means that' does NOT count;\n"
    "q2 = the term is NOT a standard English word or phrase and NOT a widely known named entity: it is a coinage, a document-specific name, a code identifier or a specialised technical term whose meaning the text itself supplies;\n"
    "q3 = the term is used at least twice AFTER its definition within the window.\nWindow:\n\n{W}"
)


def list_files(pid: str) -> list[str]:
    """The pinned file urls of a pool (CFG max_files truncates for smoke tests only)."""
    urls = [u for u, _ in list_pool_files(pid)]
    return urls[: CFG["max_files"]] if CFG["max_files"] else urls


def get_tok():
    return load_manifest_tokenizer(_p("data"))


_REF = None


def refvocab():
    global _REF
    if _REF is None:
        _REF = set(open(f"{_p('crit')}/refvocab.txt").read().split("\n"))
    return _REF


def term_ok(T, tok):
    if not WORDTERM.match(T.strip()) or EXCL.search(T):
        return False
    ws = [w.lower() for w in T.split()]
    pieces = [x.lower() for x in WORD.findall(T)]  # the reference grammar [A-Za-z][A-Za-z'\-]+ applied to T
    if any(w in BANNED for w in ws) or not pieces or any(x in refvocab() for x in pieces):
        return False
    return len(tok(" " + T, add_special_tokens=False)["input_ids"]) >= 2


def use_re(T):
    return re.compile(r"(?<![\w\-])" + re.escape(T) + r"(?![\w\-])", re.I)


def candidate(text, tok):
    """The candidate rule of the verifier-selected arm. Returns dict or (None, reason)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    head = text[:20000]
    pos = 0
    for para in PARA.split(head):
        p0 = head.find(para, pos)
        pos = p0 + len(para)
        soff = 0
        for s in SENT.split(para):
            s0 = para.find(s, soff)
            soff = s0 + len(s)
            if not s.strip():
                continue
            hit = next((p for p in PATS_RE if p.search(s)), None)
            if hit is None:
                continue
            T, how = extract_term(s)
            if not T or not term_ok(T, tok):
                continue
            ids = tok(text[p0:], add_special_tokens=False)["input_ids"]
            if len(ids) < WIN:
                return None, "short_doc"
            ids = ids[:WIN]
            wtxt = tok.decode(ids)

            def norm(x):
                return re.sub(r"\s+", " ", x).strip()

            sn = norm(s)
            wn = norm(wtxt)
            k = wn.find(sn)
            if k < 0:
                return None, "def_not_in_window"
            uses = [m for m in use_re(T).finditer(wn) if m.start() >= k + len(sn)]
            if len(uses) < 2:
                return None, "uses_lt2"
            return {
                "term": T,
                "term_how": how,
                "pattern": hit.pattern,
                "def_sentence": s.strip()[:400],
                "para_char": p0,
                "ids": ids,
                "later_uses": len(uses),
                "cue": hit.search(s).group(0),
            }, None
    return None, "no_candidate"


def _scan_file(args):
    pid, fi, url = args
    from datasets import load_dataset

    repo, rev, _, kind, field, _ = POOLS[pid]
    tok, _ = get_tok()
    out = []
    c = collections.Counter()
    t0 = time.time()
    try:
        ds = load_dataset(kind, data_files=url, split="train", streaming=True)
        for ri, r in enumerate(ds):
            text = r.get(field) or ""
            if len(text) < 500:
                c["short_chars"] += 1
                continue
            c["docs"] += 1
            cand, why = candidate(text, tok)
            if cand is None:
                c[why] += 1
                continue
            c["cands"] += 1
            cand.update(
                {
                    "pool": pid,
                    "file": fi,
                    "row": ri,
                    "doc_id": str(r.get("id") or r.get("url") or r.get("meta") or "")[:120],
                }
            )
            out.append(cand)
    except Exception as e:
        c["file_error"] += 1
        out.append({"_error": f"{type(e).__name__}: {str(e)[:200]}", "file": fi})
    return fi, out, dict(c), round(time.time() - t0)


def d2_scan(a):
    import multiprocessing as mp

    urls = list_files(a.pool)
    t0 = time.time()
    json.dump(
        {
            "pool": a.pool,
            "revision": POOLS[a.pool][1],
            "files": urls,
            "sha256_list": hashlib.sha256("\n".join(urls).encode()).hexdigest(),
        },
        open(f"{_p('mine')}/files_{a.pool}.json", "w"),
        indent=1,
    )
    jobs = [(a.pool, i, u) for i, u in enumerate(urls)]
    counts = collections.Counter()
    errors = []
    with open(f"{_p('mine')}/cands_{a.pool}.jsonl", "w") as f, mp.Pool(a.workers) as p:
        for fi, out, c, secs in p.imap(_scan_file, jobs):  # imap keeps FILE order -> deterministic candidate sequence
            counts.update(c)
            for o in out:
                if "_error" in o:
                    errors.append(o)
                    continue
                f.write(json.dumps(o) + "\n")
            print(
                f"{a.pool} file {fi}/{len(urls)} docs={c.get('docs', 0)} cands={c.get('cands', 0)} {secs}s", flush=True
            )
    info = {
        "pool": a.pool,
        "desc": POOLS[a.pool][5],
        "revision": POOLS[a.pool][1],
        "n_files": len(urls),
        "counts": dict(counts),
        "errors": errors,
        "seconds": round(time.time() - t0),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(info, open(f"{_p('mine')}/scan_{a.pool}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "errors"}))


def d2_verify(a):
    import concurrent.futures as cf

    tok, _ = get_tok()
    cands = list(jl(f"{_p('mine')}/cands_{a.pool}.jsonl"))
    t0 = time.time()

    def one(i):
        c = cands[i]
        w = tok.decode(c["ids"])
        js, raw, err = ask_json(VERIFY_MODEL, VERIFY_PROMPT.format(T=c["term"], W=w))
        keep = bool(js) and all(js.get(q) is True for q in ("q1", "q2", "q3"))
        return i, {
            "i": i,
            "pool": a.pool,
            "file": c["file"],
            "row": c["row"],
            "term": c["term"],
            "answer": js,
            "keep": keep,
            "error": err,
            "model": VERIFY_MODEL,
        }

    res = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, r in ex.map(one, range(len(cands))):
            res[i] = r
    n_err = sum(1 for r in res.values() if r["error"] and r["answer"] is None)
    n_keep = sum(1 for r in res.values() if r["keep"])
    with open(f"{_p('mine')}/verify_{a.pool}.jsonl", "w") as f:
        for i in range(len(cands)):
            f.write(json.dumps(res[i]) + "\n")
    with open(f"{_p('mine')}/verified_{a.pool}.jsonl", "w") as f:
        for i in range(len(cands)):
            if res[i]["keep"]:
                f.write(json.dumps(cands[i]) + "\n")
    info = {
        "pool": a.pool,
        "n_candidates": len(cands),
        "n_kept": n_keep,
        "n_failed": n_err,
        "failure_rate": n_err / max(1, len(cands)),
        "keep_rate": n_keep / max(1, len(cands)),
        "q_rates": {
            q: sum(1 for r in res.values() if (r["answer"] or {}).get(q) is True) / max(1, len(cands))
            for q in ("q1", "q2", "q3")
        },
        "model": VERIFY_MODEL,
        "seconds": round(time.time() - t0),
    }
    json.dump(info, open(f"{_p('mine')}/verify_{a.pool}.json", "w"), indent=1)
    print(json.dumps(info))


def ref_shingles():
    ref = set()
    packs = np.load(f"{_p('data')}/packs_s0.u16", mmap_mode="r")
    for r0 in range(0, packs.shape[0], 4096):
        for row in np.asarray(packs[r0 : r0 + 4096]).reshape(-1, SEQ // SHINGLE, SHINGLE):
            for sh in row:
                ref.add(hash(sh.tobytes()))
    for k in ("fwe", "pile"):
        v = np.load(f"{_p('data')}/val_{k}.u16", mmap_mode="r")
        n = (len(v) // SHINGLE) * SHINGLE
        for sh in np.asarray(v[:n]).reshape(-1, SHINGLE):
            ref.add(hash(sh.tobytes()))
    return ref


def stride_hit(ids, ref):
    arr = np.asarray(ids, dtype=np.uint16)
    win = np.lib.stride_tricks.sliding_window_view(arr, SHINGLE)
    return any(hash(w.tobytes()) in ref for w in win)


def aligned(ids):
    arr = np.asarray(ids[: (len(ids) // SHINGLE) * SHINGLE], dtype=np.uint16).reshape(-1, SHINGLE)
    return [hash(w.tobytes()) for w in arr]


def d2_pack(a):
    tok, man = get_tok()
    eos = man["eos"]
    t0 = time.time()
    ref = ref_shingles()
    print(f"reference shingles {len(ref)} ({time.time() - t0:.0f}s)", flush=True)
    queues = {
        p: (list(jl(f"{_p('mine')}/verified_{p}.jsonl")) if os.path.exists(f"{_p('mine')}/verified_{p}.jsonl") else [])
        for p in POOLS
    }
    ptr = {p: 0 for p in POOLS}
    admitted = []
    seen = set()
    U = 0
    drops = collections.Counter()
    pairs = collections.Counter()
    while U < R_TOKENS:
        progressed = False
        for p in POOLS:
            while ptr[p] < len(queues[p]):
                c = queues[p][ptr[p]]
                ptr[p] += 1
                ids = c["ids"]
                if stride_hit(ids, ref):
                    drops[f"{p}:shingle_ref"] += 1
                    continue
                dup = next((s for s in aligned(ids) if s in seen), None)
                if dup is not None:
                    drops[f"{p}:dup_window"] += 1
                    pairs[f"{p}<-{owner[dup]}"] += 1
                    continue
                for s in aligned(ids):
                    seen.add(s)
                    owner[s] = p
                c = dict(c)
                c["i"] = len(admitted)
                admitted.append(c)
                U += WIN + 1
                progressed = True
                break
            if U >= R_TOKENS:
                break
        if not progressed:
            break
    D = (2 * ROWS * (WIN + 1)) / max(1, U)
    yield_gate = U >= R_TOKENS / 2
    with open(f"{_p('mine')}/admitted.jsonl", "w") as f:
        for c in admitted:
            f.write(json.dumps(c) + "\n")
    mm = np.lib.format.open_memmap(
        f"{_p('out_data')}/synth_enriched_s0.u16", mode="w+", dtype=np.uint16, shape=(ROWS, SEQ)
    )
    n = len(admitted)
    exposures = collections.Counter()
    if yield_gate:
        for k in range(ROWS):
            w0, w1 = admitted[(2 * k) % n], admitted[(2 * k + 1) % n]
            exposures[w0["i"]] += 1
            exposures[w1["i"]] += 1
            mm[k] = np.array(w0["ids"] + [eos] + w1["ids"] + [eos], dtype=np.uint16)
    mm.flush()
    info = {
        "unique_windows": n,
        "unique_tokens_U": U,
        "R_tokens": R_TOKENS,
        "duplication_factor_D": D,
        "yield_gate_U_ge_R_half": yield_gate,
        "cycles": (2 * ROWS) / max(1, n),
        "unique_documents": len({(c["pool"], c["file"], c["row"]) for c in admitted}),
        "unique_terms": len({c["term"].lower() for c in admitted}),
        "pool_shares_windows": {p: sum(1 for c in admitted if c["pool"] == p) / max(1, n) for p in POOLS},
        "verified_available": {p: len(q) for p, q in queues.items()},
        "consumed": dict(ptr),
        "drops": dict(drops),
        "dup_pairs": dict(pairs),
        "exposure_hist": dict(collections.Counter(exposures.values())),
        "reference_shingles": len(ref),
        "sha256_synth_enriched": sha(f"{_p('out_data')}/synth_enriched_s0.u16") if yield_gate else None,
        "seconds": round(time.time() - t0),
    }
    json.dump(info, open(f"{_p('mine')}/pack.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


owner = {}


def render_meaning(c):
    s = re.sub(r"\s+", " ", c["def_sentence"])
    s = s.replace(c["cue"], " ") if c.get("cue") else s
    s = use_re(c["term"]).sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:.-—\"'")
    m = " ".join(s.split()[:12])
    return (
        m if m else " ".join(re.sub(r"\s+", " ", c["def_sentence"]).split()[:12])
    )  # fallback (deterministic): first 12 words of the definition sentence


def render_window(c, tok, eos, i):
    header, sep = sd.TEMPLATES[random.Random(29_000_000 + i).randrange(len(sd.TEMPLATES))]
    wtxt = tok.decode(c["ids"])
    wn = re.sub(r"\s+", " ", wtxt).strip()
    sn = re.sub(r"\s+", " ", c["def_sentence"]).strip()
    k = wn.find(sn)
    after = wn[k + len(sn) :] if k >= 0 else wn
    uses = [s.strip() for s in SENT.split(after) if use_re(c["term"]).search(s)][:4]
    for n_use in range(len(uses), -1, -1):  # drop use lines last-to-first until >= 512 prose tokens remain
        prefix = (
            header + f"{render_meaning(c)}{sep}{c['term']}" + "\n" + ("\n".join(uses[:n_use]) + "\n" if n_use else "")
        )  # TEMPLATES headers end with "\n"
        pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        if WIN - len(pre_ids) >= 512 or n_use == 0:
            break
    ids = (pre_ids + c["ids"])[:WIN]
    return ids, len(pre_ids), n_use, header


def d3_render(a):
    tok, man = get_tok()
    eos = man["eos"]
    adm = list(jl(f"{_p('mine')}/admitted.jsonl"))
    json.load(open(f"{_p('mine')}/pack.json"))
    t0 = time.time()
    rend = []
    pre_len = []
    n_uses = []
    val = collections.Counter()
    t_occ = []
    for c in adm:
        ids, pl, nu, header = render_window(c, tok, eos, c["i"])
        rend.append(ids)
        pre_len.append(pl)
        n_uses.append(nu)
        d = tok.decode(ids)
        val["len_ok"] += int(len(ids) == WIN)
        val["header_ok"] += int(d.startswith(header.strip()[:20]))
        val["meaning_nonempty"] += int(bool(render_meaning(c)))
        val["term_in_line2"] += int(c["term"] in d.split("\n")[1] if "\n" in d else 0)
        t_occ.append(len(use_re(c["term"]).findall(d)))
    with open(f"{_p('mine')}/rendered.jsonl", "w") as f:
        for c, ids in zip(adm, rend):
            f.write(json.dumps({"i": c["i"], "ids": ids}) + "\n")
    mm = np.lib.format.open_memmap(
        f"{_p('out_data')}/synth_rendered_s0.u16", mode="w+", dtype=np.uint16, shape=(ROWS, SEQ)
    )
    n = len(adm)
    for k in range(ROWS):
        mm[k] = np.array(rend[(2 * k) % n] + [eos] + rend[(2 * k + 1) % n] + [eos], dtype=np.uint16)
    mm.flush()
    info = {
        "n": n,
        "mean_prefix_tokens": float(np.mean(pre_len)),
        "mean_prose_tokens_removed": float(np.mean(pre_len)),
        "use_lines_hist": dict(collections.Counter(n_uses)),
        "min_prose_tokens": int(WIN - max(pre_len)),
        "validator": dict(val),
        "validator_all_pass": all(val[k] == n for k in ("len_ok", "header_ok", "meaning_nonempty", "term_in_line2")),
        "term_occurrences_in_D3_hist": dict(collections.Counter(min(x, 10) for x in t_occ)),
        "example": tok.decode(rend[0])[:600],
        "sha256_synth_rendered": sha(f"{_p('out_data')}/synth_rendered_s0.u16"),
        "seconds": round(time.time() - t0),
    }
    json.dump(info, open(f"{_p('mine')}/render.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def d2_audit(a):
    import concurrent.futures as cf

    tok, _ = get_tok()
    adm = list(jl(f"{_p('mine')}/admitted.jsonl"))
    rng = random.Random(1)
    idx = sorted(rng.sample(range(len(adm)), min(150, len(adm))))
    t0 = time.time()
    jobs = [(i, m) for i in idx for m in AUDIT_MODELS]

    def one(j):
        i, m = j
        c = adm[i]
        js, raw, err = ask_json(m, VERIFY_PROMPT.format(T=c["term"], W=tok.decode(c["ids"])))
        return j, {
            "i": i,
            "model": m,
            "term": c["term"],
            "pool": c["pool"],
            "answer": js,
            "error": err,
            "all_true": bool(js) and all(js.get(q) is True for q in ("q1", "q2", "q3")),
        }

    res = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for j, r in ex.map(one, jobs):
            res[j] = r
    with open(f"{_p('mine')}/audit.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(res[j]) + "\n")
    pos = sum(1 for i in idx if sum(1 for m in AUDIT_MODELS if res[(i, m)]["all_true"]) >= 2)
    prec = pos / len(idx)
    info = {
        "n": len(idx),
        "seed": 1,
        "consensus_positives": pos,
        "three_model_consensus_precision": prec,
        "gate": 0.70,
        "passes": prec >= 0.70,
        "per_model_positive_rate": {m: sum(1 for i in idx if res[(i, m)]["all_true"]) / len(idx) for m in AUDIT_MODELS},
        "per_model_q_rates": {
            m: {
                q: sum(1 for i in idx if (res[(i, m)]["answer"] or {}).get(q) is True) / len(idx)
                for q in ("q1", "q2", "q3")
            }
            for m in AUDIT_MODELS
        },
        "per_model_failures": {m: sum(1 for i in idx if res[(i, m)]["answer"] is None) for m in AUDIT_MODELS},
        "models_without_temperature": sorted(client().no_temperature),
        "indices": idx,
        "seconds": round(time.time() - t0),
    }
    json.dump(info, open(f"{_p('mine')}/audit.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "indices"}, indent=1))


def d2_surprisal(a):
    import torch
    from transformers import AutoModelForCausalLM

    tok, man = get_tok()
    adm = list(jl(f"{_p('mine')}/admitted.jsonl"))
    idx = json.load(open(f"{_p('mine')}/audit.json"))["indices"]
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8)))
    model = AutoModelForCausalLM.from_pretrained(man["base"], revision=man["revision"], torch_dtype=torch.bfloat16)
    model.eval()
    out = []
    for i in idx:
        c = adm[i]
        ids = torch.tensor([c["ids"]])
        tids = tok(" " + c["term"], add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
            nll = -lp.gather(1, ids[0, 1:, None])[:, 0]
        seq = c["ids"]
        first = next((k for k in range(len(seq) - len(tids)) if seq[k : k + len(tids)] == tids), None)
        if first is None:
            tids2 = tok(c["term"], add_special_tokens=False)["input_ids"]
            first = next((k for k in range(len(seq) - len(tids2)) if seq[k : k + len(tids2)] == tids2), None)
            tids = tids2
        term_nll = float(nll[first - 1 : first - 1 + len(tids)].mean()) if first else None
        out.append(
            {
                "i": i,
                "term": c["term"],
                "first_pos": first,
                "term_pieces": len(tids),
                "term_mean_nll": term_nll,
                "window_mean_nll": float(nll.mean()),
            }
        )
    info = {
        "n": len(out),
        "mean_term_nll": float(np.mean([o["term_mean_nll"] for o in out if o["term_mean_nll"] is not None])),
        "mean_window_nll": float(np.mean([o["window_mean_nll"] for o in out])),
        "n_term_not_found": sum(1 for o in out if o["term_mean_nll"] is None),
        "items": out,
        "seconds": round(time.time() - t0),
        "device": "cpu",
    }
    json.dump(info, open(f"{_p('mine')}/surprisal.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "items"}))


def d2_proxies(a):
    tok, _ = get_tok()
    ref = refvocab()
    t0 = time.time()

    def stats(texts):
        xs = [fmt_stats(t, tok, ref) for t in texts]
        return {
            k: float(np.mean([x[k] for x in xs]))
            for k in ["table_line_share", "mean_line_tokens", "nonce_share", "n_lines"]
        } | {"n": len(xs)}

    adm = list(jl(f"{_p('mine')}/admitted.jsonl"))
    rend = list(jl(f"{_p('mine')}/rendered.jsonl"))
    out = {"D2_all": stats([tok.decode(c["ids"]) for c in adm]), "D3_all": stats([tok.decode(r["ids"]) for r in rend])}
    rng = np.random.default_rng(1)
    b = np.load(f"{_p('data')}/synth_compose_s0.u16", mmap_mode="r")
    d = np.load(f"{_p('data')}/synth_defuse_s0.u16", mmap_mode="r")
    out["B_rows_600"] = stats([tok.decode(b[i].tolist()) for i in rng.choice(b.shape[0], 600, replace=False)])
    out["D_rows_600"] = stats([tok.decode(d[i].tolist()) for i in rng.choice(d.shape[0], 600, replace=False)])
    out["D2_terms_pieces_hist"] = dict(
        collections.Counter(len(tok(" " + c["term"], add_special_tokens=False)["input_ids"]) for c in adm)
    )
    out["D2_later_uses_hist"] = dict(collections.Counter(min(c["later_uses"], 10) for c in adm))
    out["D2_def_para_char_median"] = float(np.median([c["para_char"] for c in adm]))
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(f"{_p('mine')}/proxies.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def d2_record(a):
    """The mined-documents record: pool table, per-pool scan / verify counts, pack / render / audit / proxies / surprisal summaries, gates, sha256 of every artefact."""
    ld = lambda n: json.load(open(f"{_p('mine')}/{n}")) if os.path.exists(f"{_p('mine')}/{n}") else None  # noqa: E731
    pack, aud, rend, prox, sur = (
        ld("pack.json"),
        ld("audit.json"),
        ld("render.json"),
        ld("proxies.json"),
        ld("surprisal.json"),
    )
    scans = {p: ld(f"scan_{p}.json") for p in POOLS}
    vers = {p: ld(f"verify_{p}.json") for p in POOLS}
    fail_rate = sum(v["n_failed"] for v in vers.values() if v) / max(
        1, sum(v["n_candidates"] for v in vers.values() if v)
    )
    d2 = bool(pack and pack["yield_gate_U_ge_R_half"] and aud and aud["passes"] and fail_rate <= 0.05)
    d3 = bool(d2 and rend and rend.get("sha256_synth_rendered") and rend.get("validator_all_pass"))
    files = {
        f: sha(f"{_p('mine')}/{f}")
        for f in sorted(os.listdir(_p("mine")))
        if f.endswith((".jsonl", ".json")) and f != "mined_documents_record.json"
    }
    out = {
        "utc": utc(),
        "pools": {
            p: {"repo": v[0], "revision": v[1], "desc": v[5], "n_files": len(list_files(p))} for p, v in POOLS.items()
        },
        "verify_model": VERIFY_MODEL,
        "audit_models": AUDIT_MODELS,
        "prompt_sha256": hashlib.sha256(VERIFY_PROMPT.encode()).hexdigest(),
        "scan": {p: (s_ or {}).get("counts") for p, s_ in scans.items()},
        "verify": {
            p: ({k: v[k] for k in ("n_candidates", "n_kept", "n_failed", "keep_rate", "q_rates")} if v else None)
            for p, v in vers.items()
        },
        "verify_failure_rate": fail_rate,
        "pack": {k: v for k, v in (pack or {}).items()},
        "audit": {k: v for k, v in (aud or {}).items() if k != "indices"},
        "render": {k: v for k, v in (rend or {}).items() if k != "example"},
        "proxies": prox,
        "surprisal": ({k: v for k, v in sur.items() if k != "items"} if sur else None),
        "arm_D2_runs": d2,
        "arm_D3_runs": d3,
        "gates": {
            "yield": bool(pack and pack["yield_gate_U_ge_R_half"]),
            "audit": bool(aud and aud["passes"]),
            "verify_failure_le_5pct": fail_rate <= 0.05,
        },
        "sha256": {
            "synth_enriched_s0.u16": sha(f"{_p('out_data')}/synth_enriched_s0.u16") if d2 else None,
            "synth_rendered_s0.u16": sha(f"{_p('out_data')}/synth_rendered_s0.u16") if d3 else None,
            "mine_files": files,
        },
        "estimand": "reference-unseen, verifier-selected open-corpus definition->use windows at 3.03 % for 1B tokens (D2) and the same windows with a deterministic dictionary-format rendering prepended (D3); recipe-level, single training seed",
    }
    json.dump(out, open(f"{_p('mine')}/mined_documents_record.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("arm_D2_runs", "arm_D3_runs", "gates", "verify_failure_rate")}))


# ---------------------------------------------------------------- reference vocabulary, pool surveys, three-model re-read
def build_refvocab(random_docs: str | Path, workers: int = 32) -> dict:
    """Every lower-cased alphabetic word of the decoded common packed rows plus the random stream sample -> <critique_dir>/refvocab.txt (+ .json)."""
    import multiprocessing as mp

    packs = np.load(f"{_p('data')}/packs_s0.u16", mmap_mode="r")
    n = packs.shape[0]
    chunks = [(i * n // workers, (i + 1) * n // workers) for i in range(workers)]
    with mp.Pool(workers) as p:
        sets = p.map(_refvocab_rows, chunks)
    vocab = set().union(*sets)
    for d in jl_iter(random_docs):
        vocab.update(w.lower() for w in WORD.findall(d["text"]))
    words = sorted(vocab)
    out = Path(_p("crit")) / "refvocab.txt"
    out.write_text("\n".join(words) + "\n")
    info = {"rows": int(n), "tokens": int(n) * SEQ, "n_words": len(words), "sha256": sha(out)}
    write_json(info, Path(_p("crit")) / "refvocab.json")
    return info


def _refvocab_rows(args: tuple[int, int]) -> set[str]:
    lo, hi = args
    tok, _ = load_manifest_tokenizer(_p("data"))
    packs = np.load(f"{_p('data')}/packs_s0.u16", mmap_mode="r")
    s = set()
    for r in range(lo, hi):
        s.update(w.lower() for w in WORD.findall(tok.decode(packs[r].tolist())))
    return s


SURVEY_POOLS = [  # name, load_dataset kwargs, text field
    ("fineweb-edu CC-MAIN-2025-26 (full dump)", dict(path="HuggingFaceFW/fineweb-edu", name="CC-MAIN-2025-26"), "text"),
    ("fineweb sample-10BT (unfiltered web)", dict(path="HuggingFaceFW/fineweb", name="sample-10BT"), "text"),
    ("dclm-baseline-1.0", dict(path="mlfoundations/dclm-baseline-1.0"), "text"),
    ("smollm-corpus fineweb-edu-dedup", dict(path="HuggingFaceTB/smollm-corpus", name="fineweb-edu-dedup"), "text"),
    ("smollm-corpus cosmopedia-v2", dict(path="HuggingFaceTB/smollm-corpus", name="cosmopedia-v2"), "text"),
    ("marin wikipedia-markdown", dict(path="marin-community/wikipedia-markdown"), "text"),
    ("marin stackexchange-markdown", dict(path="marin-community/stackexchange-markdown"), "text"),
    ("marin ar5iv-no-problem-markdown", dict(path="marin-community/ar5iv-no-problem-markdown"), "text"),
    (
        "proof-pile-2 arxiv",
        dict(path="json", data_files="hf://datasets/EleutherAI/proof-pile-2/arxiv/train/arXiv_00*.jsonl.zst"),
        "text",
    ),
    (
        "proof-pile-2 open-web-math",
        dict(path="json", data_files="hf://datasets/EleutherAI/proof-pile-2/open-web-math/train/shard-000*.jsonl.zst"),
        "text",
    ),
    (
        "proof-pile-2 algebraic-stack (math code)",
        dict(path="json", data_files="hf://datasets/EleutherAI/proof-pile-2/algebraic-stack/train/*.jsonl.zst"),
        "text",
    ),
    ("finemath-4plus", dict(path="HuggingFaceTB/finemath", name="finemath-4plus"), "text"),
    (
        "pile-of-law uscode",
        dict(path="json", data_files="hf://datasets/pile-of-law/pile-of-law/data/train.uscode.jsonl.xz"),
        "text",
    ),
    (
        "pile-of-law cfr",
        dict(path="json", data_files="hf://datasets/pile-of-law/pile-of-law/data/train.cfr.jsonl.xz"),
        "text",
    ),
    (
        "pile-of-law eurlex",
        dict(path="json", data_files="hf://datasets/pile-of-law/pile-of-law/data/train.eurlex.jsonl.xz"),
        "text",
    ),
    ("project_gutenberg (manu, en)", dict(path="manu/project_gutenberg", split="en"), "text"),
    ("stack-edu Python", dict(path="HuggingFaceTB/stack-edu", name="Python"), "content"),
    ("starcoderdata python (gated)", dict(path="bigcode/starcoderdata", data_dir="python"), "content"),
]
WORDTERM_SURVEY = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{3,}(?: [A-Za-z][A-Za-z0-9_\-]{1,})?$")


def pool_survey(
    random_docs: str | Path,
    n_per_pool: int = 1500,
    word_terms: bool = False,
    pools: list | None = None,
    out_name: str | None = None,
) -> dict:
    """Candidate-filter yield over open pools (CPU, streaming, read-only): def -> term -> (word-like term) -> novel (zero reference
    occurrences and >= 2 tokenizer pieces) -> >= 2 later uses. Resumable; failures recorded per pool."""
    from datasets import load_dataset

    tok, _ = load_manifest_tokenizer(_p("data"))
    ref: collections.Counter = collections.Counter()
    for d in jl_iter(random_docs):
        ref.update(w.lower() for w in WORD.findall(d["text"]))
    out_path = Path(_p("crit")) / (out_name or ("pool_survey_words.json" if word_terms else "pool_survey.json"))
    res = (
        read_json(out_path)
        if out_path.exists()
        else {
            "n_per_pool": n_per_pool,
            "filter": "def -> term -> "
            + (
                "wordterm -> novel_word -> novel_word_uses2"
                if word_terms
                else "novel(ref==0 & pieces>=2) -> later_uses>=2"
            ),
            "pools": {},
        }
    )
    for name, kw, field_name in pools or SURVEY_POOLS:
        if name in res["pools"] and "error" not in res["pools"][name]:
            continue
        t0 = time.time()
        c: collections.Counter = collections.Counter()
        ex: list[dict] = []
        terms: collections.Counter = collections.Counter()
        try:
            kw = dict(kw)
            split = kw.pop("split", "train")
            ds = load_dataset(split=split, streaming=True, **kw)
            for r in ds:
                text = r.get(field_name) or ""
                if len(text) < 500:
                    continue
                c["docs"] += 1
                win = text[:MAX_CHARS]
                si, dsent, term, _how, _pat = first_definition(win)
                if dsent is not None:
                    c["def"] += 1
                    if term and (not word_terms or WORDTERM_SURVEY.match(term.strip())):
                        c["term"] += 1
                        tl = term.lower()
                        rc = sum(ref[x] for x in set(WORD.findall(tl))) if " " in tl else ref.get(tl, 0)
                        pieces = len(tok(" " + term, add_special_tokens=False)["input_ids"])
                        later = count_uses(term, " ".join(sentences(win)[si + 1 :]))
                        if later >= 2:
                            c["term_uses2"] += 1
                        if rc == 0 and pieces >= 2:
                            c["novel"] += 1
                            terms[term] += 1
                            if later >= 2:
                                c["novel_uses2"] += 1
                                if len(ex) < 8:
                                    ex.append({"term": term, "pieces": pieces, "later_uses": later, "def": dsent[:220]})
                if c["docs"] >= n_per_pool:
                    break
            res["pools"][name] = {
                "counts": dict(c),
                "rates": {k: c[k] / max(1, c["docs"]) for k in ["def", "term", "term_uses2", "novel", "novel_uses2"]},
                "examples": ex,
                "top_novel_terms": terms.most_common(12),
                "seconds": round(time.time() - t0),
            }
        except Exception as e:  # noqa: BLE001 - a failing pool is recorded, never hidden
            res["pools"][name] = {
                "error": f"{type(e).__name__}: {str(e)[:300]}",
                "counts": dict(c),
                "seconds": round(time.time() - t0),
            }
        write_json(res, out_path)
        print(name, json.dumps(res["pools"][name].get("rates") or res["pools"][name].get("error")), flush=True)
    return res


REREAD_PROMPT = (
    "You audit one pretraining document window for a data study. Answer in JSON only with keys: "
    "has_definition (true/false: the window explicitly introduces/defines a specific term, symbol, function or rule; discourse phrases like 'in other words' or 'which means that' do NOT count), "
    "term (string or null), term_known_to_2T_model (true/false: would a 1.7B-parameter language model pretrained on ~2 trillion tokens of English web text and code already know this term's meaning), "
    "later_uses (integer: how many times the term is used again after its definition inside this window), "
    "format ('prose' | 'list_or_table' | 'code' | 'mixed'). Window follows.\n\n"
)


def window_reread(models: list[str] | None = None, workers: int = 12) -> dict:
    """Three-model re-read of the audited arm-D windows (<critique_dir>/armD_audit.json): definition present, term known to the base, later uses, format."""
    models = models or AUDIT_MODELS
    aud = read_json(Path(_p("crit")) / "armD_audit.json")
    sample = aud["sample"]
    tok, _ = load_manifest_tokenizer(_p("data"))
    pos = {e["pos"] for e in sample}
    texts = {r["pos"]: tok.decode(r["ids"]) for r in jl_iter(f"{_p('mine')}/selected.jsonl") if r["pos"] in pos}
    jobs = [(e["pos"], m) for e in sample for m in models]

    def ask(j):
        js, raw, err = ask_json(j[1], REREAD_PROMPT + texts[j[0]][:9000], max_tokens=4000, temperature=None)
        return js if js is not None else {"error": err or "no json", "raw": raw[:200]}

    with ThreadPoolExecutor(workers) as ex:
        res = dict(zip(jobs, ex.map(ask, jobs)))
    with open(Path(_p("crit")) / "armD_reread.jsonl", "w") as f:
        for (p, m), v in res.items():
            f.write(json.dumps({"pos": p, "model": m, **v}) + "\n")
    summ = {}
    for m in models:
        ok = [v for v in (res[(e["pos"], m)] for e in sample) if "error" not in v]
        summ[m] = {
            "n_ok": len(ok),
            "has_definition": sum(1 for v in ok if v.get("has_definition") is True),
            "term_known": sum(
                1 for v in ok if v.get("has_definition") is True and v.get("term_known_to_2T_model") is True
            ),
            "term_novel": sum(
                1 for v in ok if v.get("has_definition") is True and v.get("term_known_to_2T_model") is False
            ),
            "later_uses_ge2": sum(
                1
                for v in ok
                if v.get("has_definition") is True and isinstance(v.get("later_uses"), int) and v["later_uses"] >= 2
            ),
            "format": {
                k: sum(1 for v in ok if v.get("format") == k) for k in ["prose", "list_or_table", "code", "mixed"]
            },
        }
    maj = {"has_definition": 0, "term_known": 0, "term_novel": 0, "later_uses_ge2": 0}
    for e in sample:
        vs = [v for v in (res[(e["pos"], m)] for m in models) if "error" not in v]
        if sum(1 for v in vs if v.get("has_definition") is True) >= 2:
            maj["has_definition"] += 1
            if sum(1 for v in vs if v.get("term_known_to_2T_model") is True) >= 2:
                maj["term_known"] += 1
            if sum(1 for v in vs if v.get("term_known_to_2T_model") is False) >= 2:
                maj["term_novel"] += 1
            if sum(1 for v in vs if isinstance(v.get("later_uses"), int) and v["later_uses"] >= 2) >= 2:
                maj["later_uses_ge2"] += 1
    out = {"per_model": summ, "majority": maj, "n": len(sample)}
    write_json(out, Path(_p("crit")) / "armD_reread_summary.json")
    return out

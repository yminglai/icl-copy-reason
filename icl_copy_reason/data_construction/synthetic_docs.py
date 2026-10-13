"""Synthetic definition-use documents (mode "compose") and their every-combination-shown twins (mode "attested"), plus the
preparation checks run before an admixture run.

Inputs : a random seed and a mode; generator version `ver` (1 = the 360M admixture of record; 3 = the 1.7B admixture and the
         LoRA transfer: twins share the query count, no query repeats a cell, documents with < 2 held or < 2 shown cells are redrawn).
Outputs: (document text, meta) pairs; `write_sample` writes a jsonl.gz sample per mode; `pools_disjoint_from` checks that
         the generator's word and marker pools never overlap the diagnostic's evaluation pools.

One document = one mini-language: 2-3 features with 2-3 values each; each value is named by an ordinary word (pools
disjoint from the diagnostic's word pairs) and realised by a nonce marker (pool disjoint from the diagnostic's markers and
the alternate-template pool); realisation drawn from {suffix stem-A-B(-C), suffix reversed order, prefix}; header /
separator from six templates none of which equals the diagnostic template; every single-feature form on 1-2 stems, a
random subset of the full-feature combinations shown, then 2-4 queries on fresh stems WITH their answers written out.
  compose   every queried combination is UNSHOWN in the demonstrations (must be composed);
  attested  same template, draws, row counts, query counts and answer lengths, but every queried combination is already
            shown on another stem (copy the bundle onto the new stem; never an unshown cell).
"""
from __future__ import annotations

import gzip
import itertools
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

WORD_SETS = [
    ["soft", "hard", "rough"], ["light", "heavy"], ["fast", "slow"], ["sweet", "sour", "bitter"], ["loud", "quiet"], ["round", "flat", "long"],
    ["first", "last"], ["north", "south", "east"], ["morning", "evening"], ["inside", "outside"], ["past", "future"], ["few", "all"],
    ["male", "female"], ["wild", "tame"], ["deep", "shallow"], ["early", "late"], ["dark", "bright"], ["thick", "thin"],
]
ALL_WORDS = {w for ws in WORD_SETS for w in ws}
MARKERS = ["az", "bek", "cho", "dun", "eem", "fal", "gis", "hup", "ivo", "jal", "kre", "lom", "mib", "nol", "oq", "pev", "quo", "rax", "sut", "tam",
           "uzz", "vek", "wol", "xan", "yeb", "zog", "bri", "clu", "dez", "fon", "gar", "hix", "jun", "kop", "lat", "mur", "nib", "pol", "ruk", "sev"]
ONSETS = ["bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "kl", "kr", "pl", "pr", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr", "tw",
          "d", "f", "g", "k", "m", "n", "p", "r", "t", "v", "w", "z", "b", "l", "sh", "th", "ch"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ee", "oo", "ou", "ea"]
CODAS = ["b", "ck", "d", "f", "g", "k", "l", "m", "n", "p", "rn", "sh", "t", "th", "v", "x", "z", "lt", "mp", "nd", "rb", "sk", "ng", "st"]
TEMPLATES = [
    ("Vocabulary of an invented tongue:\n", " : "),
    ("Grammar notes. Each word is a stem with markers.\n", " => "),
    ("Field notes, morphology:\n", " -> "),
    ("Dictionary entries (meaning, then form):\n", " | "),
    ("Examples from the language:\n", " ~ "),
    ("How words are built here:\n", " is written "),
]
MODES = ("compose", "attested")


def stem(rng: random.Random) -> str:
    return rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)


def realise(st: str, markers: list[str], order: str, kind: str) -> str:
    ms = markers if order == "fwd" else markers[::-1]
    if kind == "prefix":
        return "-".join(ms) + "-" + st
    return st + "-" + "-".join(ms)


def make_doc(rng: random.Random, mode: str = "compose", ver: int = 1) -> tuple[str, dict[str, Any]]:
    """ver=1: the 360M admixture's generator (queries cycle through the held cells in compose mode; attested mode samples shown
    cells WITH replacement). ver=2: both modes draw n_q DISTINCT cells without replacement, n_q = min(draw, |held|, |shown|)
    identically in both modes. ver=3: as v2, but a document with fewer than two held or two shown cells is redrawn (n_q stays
    in 2..4), and the query stems are drawn BEFORE the queried cells so that the twin documents differ only in which cells are queried."""
    while True:
        out = _make_doc_once(rng, mode, ver)
        if out is not None:
            return out


def _make_doc_once(rng: random.Random, mode: str, ver: int):
    n_feat = rng.choice([2, 2, 3])
    sets = rng.sample(WORD_SETS, n_feat)
    values = [rng.sample(s, rng.choice([2, 2, 3]) if len(s) >= 3 else 2) for s in sets]
    used = rng.sample(MARKERS, sum(len(v) for v in values))
    marks, i = [], 0
    for v in values:
        marks.append(used[i : i + len(v)])
        i += len(v)
    kind = rng.choice(["suffix", "suffix", "prefix"])
    order = rng.choice(["fwd", "rev"])
    header, sep = rng.choice(TEMPLATES)
    stems: set[str] = set()

    def fresh() -> str:
        while True:
            s = stem(rng)
            if s not in stems and not (ver >= 2 and s in ALL_WORDS):
                stems.add(s)
                return s

    lines = []
    for f, v in enumerate(values):
        for j, word in enumerate(v):
            for _ in range(rng.choice([1, 2])):
                s = fresh()
                lines.append(f"{word} {s}{sep}{realise(s, [marks[f][j]], 'fwd', kind)}")
    combos = list(itertools.product(*[range(len(v)) for v in values]))
    rng.shuffle(combos)
    n_show = max(1, len(combos) - rng.choice([2, 3, 4]) if len(combos) > 4 else len(combos) - rng.choice([1, 2]))
    shown, held = combos[:n_show], combos[n_show:]
    for c in shown:
        s = fresh()
        words = " ".join(values[f][c[f]] for f in range(n_feat))
        lines.append(f"{words} {s}{sep}{realise(s, [marks[f][c[f]] for f in range(n_feat)], order, kind)}")
    rng.shuffle(lines)
    n_q = rng.choice([2, 3, 4])
    n_q_draw = n_q
    q_stems: list[str] = []
    if ver == 1:
        if mode == "compose":
            queries = (held * 4)[:n_q] if held else []
        else:
            queries = [rng.choice(shown) for _ in range(n_q)]
    elif ver == 2:
        n_q = min(n_q, len(held), len(shown))
        queries = rng.sample(held if mode == "compose" else shown, n_q)
    else:
        n_q = min(n_q, len(held), len(shown))
        if n_q < 2:
            return None
        q_stems = [fresh() for _ in range(n_q)]
        queries = rng.sample(held if mode == "compose" else shown, n_q)
    q_lines = []
    for qi, c in enumerate(queries):
        s = q_stems[qi] if ver >= 3 else fresh()
        words = " ".join(values[f][c[f]] for f in range(n_feat))
        q_lines.append(f"{words} {s}{sep}{realise(s, [marks[f][c[f]] for f in range(n_feat)], order, kind)}")
    meta = {"n_feat": n_feat, "n_shown": len(shown), "n_held": len(held), "n_queries": len(q_lines), "kind": kind, "order": order, "n_q_draw": n_q_draw, "n_q_unshown": sum(1 for c in queries if c not in shown)}
    return header + "\n".join(lines) + "\n" + "\n".join(q_lines) + "\n", meta


def generate(mode: str, seed: int, n: int, ver: int = 1) -> list[dict[str, Any]]:
    """n documents of one mode from one seeded generator (the stream the admixture packer consumes)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        text, meta = make_doc(rng, mode, ver)
        out.append({"i": i, "mode": mode, "seed": seed, "ver": ver, "text": text, **meta})
    return out


def write_sample(mode: str, seed: int, n: int, ver: int, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for d in generate(mode, seed, n, ver):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return n


def pools_disjoint_from(eval_words: set[str], eval_markers: set[str]) -> dict[str, list[str]]:
    """Overlaps between this generator's pools and the diagnostic's evaluation pools (both must be empty)."""
    return {"words": sorted(ALL_WORDS & {w.lower() for w in eval_words}), "markers": sorted(set(MARKERS) & {m.lower() for m in eval_markers})}


def diagnostic_pools() -> tuple[set[str], set[str]]:
    """The diagnostic's word and marker pools (standard + alternate template), read from the tasks package when present,
    else from the shipped data/diagnostic files, else the values of record (kept here so the disjointness test is self-contained)."""
    words_of_record = {"small", "big", "old", "new", "red", "blue", "hot", "cold", "one", "many", "here", "there", "mine", "yours",
                       "tiny", "huge", "young", "ancient", "green", "yellow", "wet", "dry", "single", "several", "near", "far", "ours", "theirs"}
    markers_of_record = {"il", "nak", "or", "um", "esh", "oth", "ka", "tu", "ri", "mo", "lin", "dor", "sa", "ne", "po", "ta", "ku", "zi", "fen", "gor", "lu", "wa", "ex", "an", "ir", "on", "ul", "ek", "ip", "ab",
                         "ash", "ent", "ilk", "oom", "ust", "eck", "arn", "ith", "ulp", "orm", "esk", "ind", "ath", "ock", "ump", "erd", "ipt", "oft", "ung", "ext"}
    try:
        from ..tasks import prompts as tp

        words = set()
        for name in ("WORDS_A", "WORDS_B"):
            for pr in getattr(tp, name, []):
                words.update(pr)
        for t in getattr(tp, "TEMPLATES", {}).values():
            for pr in (t.get("wA") or []) + (t.get("wB") or []):
                words.update(pr)
        markers = set(getattr(tp, "AFFIX_POOL", []))
        for t in getattr(tp, "TEMPLATES", {}).values():
            markers.update(t.get("pool") or [])
        if words and markers:
            return words | words_of_record, markers | markers_of_record
    except Exception:  # noqa: BLE001 - the tasks package is optional for this check
        pass
    return words_of_record, markers_of_record


# ---------------------------------------------------------------- preparation checks before an admixture run
def base_eligibility(results_dir: str | Path, model_tag: str, tag: str, seeds: tuple[int, ...] = (4, 5)) -> dict:
    """Eligibility of the base checkpoint from its ablation / copy-share files on two item seeds: the combined dependence
    D (competitor-marker + queried-word effects) has a 95 % interval above zero, null adherence >= .60, four-way accuracy
    < .50 and exact match < .15 on BOTH panels."""
    results_dir = Path(results_dir)
    res: dict[str, Any] = {"seeds": {}}
    ok_all = True
    for s in seeds:
        ko = json.load(open(results_dir / f"knockout_{model_tag}_s{s}{tag}.json"))
        ko = ko[list(ko.keys())[0]]
        tr = json.load(open(results_dir / f"triage_{model_tag}_s{s}{tag}.json"))
        tr = tr[list(tr.keys())[0]]["compose_h"]
        base = np.array([it["margin"] for it in ko["none"]["items"]])
        D = (np.array([it["margin"] for it in ko["competitor_marker"]["items"]]) - base) + (np.array([it["margin"] for it in ko["wordB"]["items"]]) - base)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(D), size=(2000, len(D)))
        m = D[idx].mean(1)
        lo, hi = float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
        an = float(np.mean([it["a_null"] for it in ko["none"]["items"]]))
        r = {"D": [float(D.mean()), lo, hi], "a_null": an, "four_way": tr["four_way_acc"], "exact": tr["exact"], "ok": lo > 0 and an >= 0.60 and tr["four_way_acc"] < 0.50 and tr["exact"] < 0.15}
        res["seeds"][s] = r
        ok_all &= r["ok"]
    res["eligible"] = ok_all
    return res


def learning_rate_pick(calib_dir: str | Path, rates: list[str]) -> dict:
    """The largest rate whose 64-step calibration run has finite loss throughout, clip fraction <= .20 and
    validation loss(step 64) <= validation loss(step 0) + .02; ties -> lower."""
    calib_dir = Path(calib_dir)
    rows: dict[str, Any] = {}
    for r in rates:
        f = calib_dir / f"lrcalib_{r}" / "train.jsonl"
        if not f.exists():
            rows[r] = {"missing": True}
            continue
        recs = [json.loads(line) for line in open(f)]
        last = recs[-1]
        steps = [x for x in recs if "lm_loss" in x]
        finite = all(np.isfinite(x["lm_loss"]) for x in steps) and not any(x.get("nonfinite") for x in steps) and len(steps) >= 64
        clipf = last.get("clip_fraction", sum(int(x["clipped"]) for x in steps) / max(1, len(steps)))
        v0 = (recs[0].get("val") or {}).get("fwe")
        v64 = (last.get("val") or {}).get("fwe")
        ok = finite and clipf <= 0.20 and v0 is not None and v64 is not None and v64 <= v0 + 0.02
        rows[r] = {"finite": finite, "clip_fraction": clipf, "val_fwe_0": v0, "val_fwe_64": v64, "ok": ok}
    ok_rates = [r for r in rates if rows[r].get("ok")]
    return {"rates": rows, "selected_lr": max(ok_rates, key=float) if ok_rates else None}

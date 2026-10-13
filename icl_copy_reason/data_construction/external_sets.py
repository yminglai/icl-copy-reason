"""External evaluation sets: symbol-label and exchanged-label in-context classification, the SCAN / ELEMENTAL /
MiniSCAN / arithmetic / chain-map panels with their oracles, and the far assays built from BIG-bench, bAbI and CLUTRR.

Inputs : public datasets (loaded from the Hugging Face Hub at the recorded paths; the SCAN and ELEMENTAL files ship under
         data/external/), the label maps data/external/label_maps.json, fixed seeds, and a causal LM for scoring.
Outputs: `build_*` return item lists (prompt, demonstrations, target, group / tier); `score_*` write per-item jsonl and a
         summary.json under <out>/<tag>/ with the same fields as the released per-run results.

Symbol-label sets (TREC-6, DBpedia-7, Yahoo-5, 20 Newsgroups-6; and SST-2 / AG News / Subj for the near assays):
k = 2 demonstrations per class, balanced, drawn from the TRAIN split with a fixed per-item seed; test items = the first n
of one seeded shuffle of the TEST split restricted to the kept classes (prefix-stable); three conditions share the
demonstrations: natural (true label words), symbol (labels replaced by Foo / Bar / ... ; the label mapping must come from the
demonstrations), exchanged (true label words under a cyclic shift; following the context contradicts the prior).
Prediction = arg-max summed log-probability of " <label>" continuations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from ..config import data_root
from ..io import read_json, write_json

EXT = data_root() / "external"
SEED = 1234
SUL = ["Foo", "Bar", "Baz", "Qux", "Quux", "Corge", "Grault"]
LABEL_DATASETS: dict[str, dict] = {
    # symbol-label / exchanged-label sets on datasets used in NO training exercise (k = 2 per class)
    "trec6": dict(
        path="CogComp/trec",
        revision="refs/convert/parquet",
        text=["text"],
        label="coarse_label",
        keep=[0, 1, 2, 3, 4, 5],
        names=["Abbreviation", "Entity", "Description", "Human", "Location", "Number"],
        k=12,
    ),
    "dbpedia7": dict(
        path="fancyzhx/dbpedia_14",
        revision=None,
        text=["content"],
        label="label",
        keep=[0, 1, 2, 3, 4, 5, 6],
        names=["Company", "School", "Artist", "Athlete", "Politician", "Vehicle", "Building"],
        k=14,
    ),
    "yahoo5": dict(
        path="community-datasets/yahoo_answers_topics",
        revision=None,
        text=["question_title", "question_content"],
        label="topic",
        keep=[0, 1, 2, 3, 4],
        names=["Society", "Science", "Health", "Education", "Computers"],
        k=10,
    ),
    "ng6": dict(
        path="SetFit/20_newsgroups",
        revision=None,
        text=["text"],
        label="label",
        keep=[0, 1, 2, 3, 4, 5],
        names=["Atheism", "Graphics", "Windows", "PC", "Mac", "X11"],
        k=12,
    ),
}
NEAR_DATASETS: dict[str, dict] = {
    # the near in-context assays (class-balanced test draw, k = 8)
    "sst2": dict(path="SetFit/sst2", text="text", label="label", names=["negative", "positive"], k=8),
    "agnews": dict(
        path="fancyzhx/ag_news", text="text", label="label", names=["World", "Sports", "Business", "Technology"], k=8
    ),
    "subj": dict(path="SetFit/subj", text="text", label="label", names=["objective", "subjective"], k=8),
}


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def derange(n: int) -> list[int]:
    """cyclic shift (a 2-class swap for two classes)"""
    return [(i + 1) % n for i in range(n)]


def wilson(p: float, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - hw, c + hw]


def label_prompt(item: dict, names: list[str], mapping: list[int]) -> str:
    s = "".join(f"Input: {x}\nLabel: {names[mapping[y]]}\n\n" for x, y in item["demos"])
    return s + f"Input: {item['x']}\nLabel:"


def _clean(s: str) -> str:
    return (s or "").strip().replace("\n", " ")[:600]


def _texts_of(dset, cols: list[str], idx: list[int]) -> dict[int, str]:
    """columnar read of the text columns for the kept row indices"""
    sub = dset.select(idx)
    parts = [sub[c][:] for c in cols]
    return {i: _clean(" ".join((p[j] or "") for p in parts)) for j, i in enumerate(idx)}


def build_label_set(ds: str, n: int) -> tuple[list[dict], list[str], dict]:
    """Symbol-label / exchanged-label items for one of LABEL_DATASETS (prefix-stable in n)."""
    from datasets import load_dataset

    d = LABEL_DATASETS[ds]
    kw = {"revision": d["revision"]} if d["revision"] else {}
    tr = load_dataset(d["path"], split="train", **kw)
    te = load_dataset(d["path"], split="test", **kw)
    keep = {c: j for j, c in enumerate(d["keep"])}
    C = len(d["keep"])
    trl = tr[d["label"]][:]
    tel = te[d["label"]][:]
    tet = _texts_of(te, d["text"], [i for i, y in enumerate(tel) if y in keep])
    trt = _texts_of(tr, d["text"], [i for i, y in enumerate(trl) if y in keep])
    te_idx = [i for i in sorted(tet) if tet[i]]
    random.Random(SEED).shuffle(te_idx)
    test_idx = te_idx[:n]
    trby = {j: [i for i in sorted(trt) if keep[trl[i]] == j and trt[i]] for j in range(C)}
    items = []
    for jdx, ti in enumerate(test_idx):
        r = random.Random(SEED * 1000 + ti)
        demos = sum([r.sample(trby[j], d["k"] // C) for j in range(C)], [])
        r.shuffle(demos)
        items.append(
            {
                "idx": jdx,
                "test_i": ti,
                "x": tet[ti],
                "y": keep[tel[ti]],
                "demos": [(trt[i], keep[trl[i]]) for i in demos],
            }
        )
    return (
        items,
        d["names"],
        {
            "path": d["path"],
            "revision": d["revision"],
            "n_test_kept": len(te_idx),
            "n_train_kept": sum(len(v) for v in trby.values()),
        },
    )


def build_near_set(ds: str, n: int) -> tuple[list[dict], list[str]]:
    """Class-balanced items for one of NEAR_DATASETS (each class pool shuffled once by the seed; the first n/C items taken)."""
    from datasets import load_dataset

    d = NEAR_DATASETS[ds]
    tr = load_dataset(d["path"], split="train")
    te = load_dataset(d["path"], split="test")
    rng = random.Random(SEED)
    C = len(d["names"])
    per = n // C
    by = {c: [i for i, y in enumerate(te[d["label"]]) if y == c] for c in range(C)}
    for c in range(C):
        rng.shuffle(by[c])
    test_idx = sum([by[c][:per] for c in range(C)], [])
    random.Random(SEED + 1).shuffle(test_idx)
    trby = {c: [i for i, y in enumerate(tr[d["label"]]) if y == c] for c in range(C)}
    items = []
    for j, ti in enumerate(test_idx):
        r = random.Random(SEED * 1000 + ti)
        demos = sum([r.sample(trby[c], d["k"] // C) for c in range(C)], [])
        r.shuffle(demos)
        items.append(
            {
                "idx": j,
                "test_i": ti,
                "x": te[ti][d["text"]].strip().replace("\n", " ")[:600],
                "y": te[ti][d["label"]],
                "demos": [(tr[i][d["text"]].strip().replace("\n", " ")[:600], tr[i][d["label"]]) for i in demos],
            }
        )
    return items, d["names"]


def label_maps() -> dict:
    """Label names, kept class indices, symbol lists and exchange maps for every set (shipped as data/external/label_maps.json)."""
    out = {"symbols": SUL, "exchange_rule": "cyclic shift of the label indices (a swap for two classes)", "sets": {}}
    for ds, d in LABEL_DATASETS.items():
        C = len(d["names"])
        out["sets"][ds] = {
            "source": d["path"],
            "revision": d["revision"],
            "label_field": d["label"],
            "text_fields": d["text"],
            "kept_label_indices": d["keep"],
            "names": d["names"],
            "k": d["k"],
            "symbols": SUL[:C],
            "exchange_map": {d["names"][i]: d["names"][derange(C)[i]] for i in range(C)},
            "demonstrations": "2 per class from the train split, per-item seed SEED*1000+test_index",
        }
    for ds, d in NEAR_DATASETS.items():
        C = len(d["names"])
        out["sets"][ds] = {
            "source": d["path"],
            "label_field": d["label"],
            "text_fields": [d["text"]],
            "names": d["names"],
            "k": d["k"],
            "symbols": SUL[:C],
            "exchange_map": {d["names"][i]: d["names"][derange(C)[i]] for i in range(C)},
            "demonstrations": f"{d['k'] // C} per class from the train split, per-item seed SEED*1000+test_index",
        }
    return out


def score_label_options(lm, prompts_opts: list[tuple[str, list[str]]]) -> list[list[float]]:
    """Summed log-probability of each ' <label>' continuation given the prompt (prompt run once with a KV cache)."""
    import torch

    model, tok, dev = lm.model, lm.tok, lm.device
    out = []
    with torch.no_grad():
        for p, opts in prompts_opts:
            pids = tok(p, return_tensors="pt").input_ids.to(dev)
            n0 = pids.shape[1]
            po = model(pids, use_cache=True)
            cache = po.past_key_values
            last = po.logits[0, -1].float().log_softmax(-1)
            lls = []
            for o in opts:
                oids = tok(" " + o, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
                ll = last[oids[0, 0]].item()
                if oids.shape[1] > 1:
                    extra = cache.get_seq_length() - n0
                    if extra > 0:
                        cache.crop(-extra)
                    lg = model(oids[:, :-1], past_key_values=cache, use_cache=True).logits[0].float().log_softmax(-1)
                    ll += lg.gather(1, oids[0, 1:].unsqueeze(1)).sum().item()
                lls.append(ll)
            out.append(lls)
    return out


def score_label_sets(lm, out_dir: str | Path, tag: str, datasets: list[str], limit: int = 1000) -> dict:
    """Natural / symbol / exchanged conditions for every requested set -> <out>/<tag>/<ds>.jsonl + summary.json."""
    out_dir = Path(out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    t0 = time.time()
    for ds in datasets:
        t1 = time.time()
        if ds in LABEL_DATASETS:
            items, names, src = build_label_set(ds, limit)
            k = LABEL_DATASETS[ds]["k"]
        else:
            items, names = build_near_set(ds, limit)
            src = {"path": NEAR_DATASETS[ds]["path"]}
            k = NEAR_DATASETS[ds]["k"]
        C = len(names)
        ident = list(range(C))
        flip = derange(C)
        sym = SUL[:C]
        rows = []
        for it in items:
            conds = {
                "natural": (label_prompt(it, names, ident), names),
                "sul": (label_prompt(it, sym, ident), sym),
                "flipped": (label_prompt(it, names, flip), names),
            }
            lls = score_label_options(lm, list(conds.values()))
            rec: dict[str, Any] = {"idx": it["idx"], "test_i": it["test_i"], "y": it["y"]}
            for (cname, _), ll in zip(conds.items(), lls):
                rec[cname + "_pred"] = max(range(C), key=lambda i: ll[i])
                rec[cname + "_ll"] = ll
            rec["natural_correct"] = int(rec["natural_pred"] == it["y"])
            rec["sul_correct"] = int(rec["sul_pred"] == it["y"])
            rec["flipped_follows_context"] = int(rec["flipped_pred"] == flip[it["y"]])
            rec["flipped_follows_prior"] = int(rec["flipped_pred"] == it["y"])
            rows.append(rec)
        with open(out_dir / f"{ds}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        nat = sum(r["natural_correct"] for r in rows) / n
        summary[ds] = {
            "n": n,
            "n_classes": C,
            "chance": 1 / C,
            "label_names": names,
            "sul_symbols": sym,
            "flip_map": {names[i]: names[flip[i]] for i in range(C)},
            "k": k,
            "source": src,
            "natural_acc": nat,
            "natural_wilson95": wilson(nat, n),
            "sul_acc": sum(r["sul_correct"] for r in rows) / n,
            "flipped_follows_context": sum(r["flipped_follows_context"] for r in rows) / n,
            "flipped_follows_prior": sum(r["flipped_follows_prior"] for r in rows) / n,
            "class_counts": [sum(1 for r in rows if r["y"] == c) for c in range(C)],
            "elapsed_s": time.time() - t1,
        }
        print(ds, json.dumps(summary[ds]), flush=True)
    write_json(
        {
            "model": getattr(lm, "name", None),
            "revision": getattr(lm, "revision", None),
            "limit": limit,
            "seed": SEED,
            "elapsed_s": time.time() - t0,
            "summary": summary,
        },
        out_dir / "summary.json",
    )
    return summary


# ---------------------------------------------------------------- SCAN / ELEMENTAL / MiniSCAN / arithmetic / chain-map panels
PRIMS = {"walk": "I_WALK", "run": "I_RUN", "look": "I_LOOK", "jump": "I_JUMP"}
OPS = ["twice", "thrice", "and", "after", "around", "opposite", "left", "right", "turn"]
MAX_OPS_PRIMARY = 4  # primary SCAN tier = <= 4 operator words; secondary tier >= 5


def read_scan(path):
    out = []
    for line in open(path):
        m = re.match(r"IN: (.*) OUT: (.*)", line.strip())
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


def skeleton(cmd):
    return " ".join("PRIM" if w in PRIMS else w for w in cmd.split())


def scan_interpret(cmd):
    """SCAN semantics (Lake & Baroni 2018, Fig. 6): u ::= walk|run|look|jump|turn ; x ::= u | u left|right | u opposite d | u around d ;
    s ::= x | x twice | x thrice ; c ::= s | s and s | s after s."""

    def prim(u):
        return [] if u == "turn" else [PRIMS[u]]

    def turn(d):
        return ["I_TURN_LEFT" if d == "left" else "I_TURN_RIGHT"]

    def x(ws):
        u = ws[0]
        if len(ws) == 1:
            return prim(u)
        if len(ws) == 2:
            return turn(ws[1]) + prim(u)
        if ws[1] == "opposite":
            return turn(ws[2]) * 2 + prim(u)
        if ws[1] == "around":
            return (turn(ws[2]) + prim(u)) * 4
        raise ValueError(ws)

    def s(ws):
        if ws[-1] == "twice":
            return x(ws[:-1]) * 2
        if ws[-1] == "thrice":
            return x(ws[:-1]) * 3
        return x(ws)

    ws = cmd.split()
    if "and" in ws:
        i = ws.index("and")
        return s(ws[:i]) + s(ws[i + 1 :])
    if "after" in ws:
        i = ws.index("after")
        return s(ws[i + 1 :]) + s(ws[:i])
    return s(ws)


def build_scan(n=200, k=12):
    train = read_scan(f"{EXT}/scan/tasks_train_addprim_jump.txt")
    test = read_scan(f"{EXT}/scan/tasks_test_addprim_jump.txt")
    # oracle check on every gold line
    bad = [c for c, a in train + test if " ".join(scan_interpret(c)) != a]
    nonjump = [(c, a) for c, a in train if "jump" not in c.split()]
    by_skel = {}
    for c, a in nonjump:
        by_skel.setdefault(skeleton(c), []).append((c, a))
    by_op = {}
    for c, a in nonjump:
        for w in set(c.split()) & set(OPS):
            by_op.setdefault(w, []).append((c, a))
    for w in by_op:
        by_op[w].sort(key=lambda t: (len(t[0].split()), h(t[0])))
    fillers = sorted(nonjump, key=lambda t: h(t[0]))
    # stratified frozen order over test items: round-robin over skeletons in hash order, items within skeleton in hash order
    groups = {}
    for c, a in test:
        groups.setdefault(skeleton(c), []).append((c, a))
    for g in groups.values():
        g.sort(key=lambda t: h(t[0]))
    order = []
    skels = sorted(groups, key=h)
    i = 0
    while len(order) < len(test):
        for sk in skels:
            if i < len(groups[sk]):
                order.append(groups[sk][i])
        i += 1
    items, replaced = [], 0
    for c, a in order:
        if len(items) >= n:
            break
        if (
            len([w for w in c.split() if w in OPS]) > MAX_OPS_PRIMARY
        ):  # primary tier = commands with <= 4 operator words
            continue
        demos = [("jump", "I_JUMP")]
        sk = skeleton(c)
        if sk in by_skel:
            demos.append(by_skel[sk][0])
        for w in sorted(set(c.split()) & set(OPS)):
            for d in by_op[w]:
                if d not in demos:
                    demos.append(d)
                    break
        if len(demos) > k:
            replaced += 1
            continue
        for d in fillers:
            if len(demos) >= k:
                break
            if d not in demos:
                demos.append(d)
        prompt = "\n".join(f"IN: {dc} OUT: {da}" for dc, da in demos) + f"\nIN: {c} OUT:"
        n_ops = len([w for w in c.split() if w in OPS])
        items.append(
            {
                "prompt": prompt,
                "target": a,
                "query": c,
                "skeleton": sk,
                "n_ops": n_ops,
                "tier": "primary",
                "demos": demos,
            }
        )
    # secondary tier (harder): up to 100 further test queries with >= 5 operator words (never substitutes for the primary panel)
    prim_q = {it["query"] for it in items}
    for c, a in order:
        if sum(1 for it in items if it["tier"] == "long_command") >= 100:
            break
        if c in prim_q or len([w for w in c.split() if w in OPS]) <= MAX_OPS_PRIMARY:
            continue
        demos = [("jump", "I_JUMP")]
        sk = skeleton(c)
        if sk in by_skel:
            demos.append(by_skel[sk][0])
        for w in sorted(set(c.split()) & set(OPS)):
            for d in by_op[w]:
                if d not in demos:
                    demos.append(d)
                    break
        for d in fillers:
            if len(demos) >= k:
                break
            if d not in demos:
                demos.append(d)
        prompt = "\n".join(f"IN: {dc} OUT: {da}" for dc, da in demos) + f"\nIN: {c} OUT:"
        items.append(
            {
                "prompt": prompt,
                "target": a,
                "query": c,
                "skeleton": sk,
                "n_ops": len([w for w in c.split() if w in OPS]),
                "tier": "long_command",
                "demos": demos,
            }
        )
    return {
        "task": "scan_icl",
        "n": len(items),
        "n_primary": sum(1 for it in items if it["tier"] == "primary"),
        "k_demos": k,
        "replaced_over_k": replaced,
        "oracle_mismatches": len(bad),
        "items": items,
    }


# ------------------------------------------------------------------------------------------------------------------ ElementalTask
def elem_ops():
    simple = list(csv.DictReader(open(f"{EXT}/elemental/simple.csv")))
    lex = {}
    for r in simple:
        lex.setdefault(r["category_name"], {})[r["question"]] = r["answer"]
    words = sorted(
        {
            r["question"]
            for r in simple
            if r["category_name"] in ("present_to_gerund", "singular_to_plural", "translate_eng_fr", "translate_eng_sp")
        }
    )
    lex["reverse"] = {w: w[::-1] for w in words}  # synthesised atomic demonstrations (the repo ships none for reverse)
    lex["uppercase"] = {**lex.get("uppercase", {}), **{w: w.upper() for w in words[:60]}}
    lex["lowercase"] = {**lex.get("lowercase", {}), **{w.upper(): w for w in words[:60]}}
    lex["first_letter"] = {**lex.get("first_letter", {})}
    lex["last_letter"] = {**lex.get("last_letter", {})}
    fn = {
        "uppercase": str.upper,
        "lowercase": str.lower,
        "first_letter": lambda s: s[0],
        "last_letter": lambda s: s[-1],
        "reverse": lambda s: s[::-1],
    }
    return lex, fn


def elem_apply(chain, x, lex, fn):
    for op in chain:
        if op in fn:
            x = fn[op](x)
        elif x in lex.get(op, {}):
            x = lex[op][x]
        elif x.lower() in lex.get(
            op, {}
        ):  # the composite file applies case changes before lexical lookups regardless of the listed order
            x = lex[op][x.lower()]
        else:
            return None
    return x


def build_elemental(per_cat=5, k=4):
    comp = list(csv.DictReader(open(f"{EXT}/elemental/compositional.csv")))
    lex, fn = elem_ops()
    cats = {}
    for r in comp:
        cats.setdefault(r["category_name"], []).append(r)
    items, oracle_ok, oracle_n = [], 0, 0
    for cat in sorted(cats, key=h):
        rows = sorted(cats[cat], key=lambda r: h(r["input"]))
        taken = 0
        for r in rows:
            if taken >= per_cat:
                break
            chain = r["operations"].split("+")
            demos = []
            ok = True
            for op in chain:
                pool = sorted(
                    ((q, a) for q, a in lex.get(op, {}).items() if q != r["input"] and a != r["output"]),
                    key=lambda t: h(op + t[0]),
                )
                if len(pool) < k:
                    ok = False
                    break
                demos += [(op, q, a) for q, a in pool[:k]]
            if not ok:
                continue
            prompt = (
                "\n".join(f"{op}: {q} -> {a}" for op, q, a in demos) + f"\n{'+'.join(chain)}: {r['input']} ->"
            )  # named operations
            o = elem_apply(chain, r["input"], lex, fn)
            oracle_n += 1
            oracle_ok += int(o == r["output"])
            items.append(
                {
                    "prompt": prompt,
                    "target": r["output"],
                    "query": r["input"],
                    "category": cat,
                    "chain": chain,
                    "tier": "primary" if len(chain) == 2 else "three_ops",
                    "demos": demos,
                    "oracle": o,
                }
            )
            taken += 1
    return {
        "task": "elemental_icl",
        "n": len(items),
        "k_demos_per_op": k,
        "oracle_acc": oracle_ok / max(1, oracle_n),
        "items": items,
    }


# ------------------------------------------------------------------------------------------------------------------ MiniSCAN (Lake, Linzen & Baroni 2019)
NONCE = [
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
    "prax",
    "temmo",
    "goll",
    "riff",
    "sket",
    "nurm",
    "hib",
    "walp",
    "zeck",
    "orpa",
    "lini",
    "cheb",
]
COLOURS = ["RED", "GREEN", "BLUE", "YELLOW", "PINK", "PURPLE", "BLACK", "WHITE", "ORANGE", "BROWN", "GRAY", "CYAN"]


def miniscan_interpret(cmd, prim, fw):
    """cmd: list of words. Grammar (Lake, Linzen & Baroni 2019): u -> prim | prim REP (three copies); x -> u | u WRAP u (-> u1 u2 u1); c -> x | x SWAP x (-> x2 x1)."""
    rep, swap, wrap = fw["rep"], fw["swap"], fw["wrap"]

    def u(ws):  # primitive or primitive REP
        if len(ws) == 1:
            return [prim[ws[0]]]
        if len(ws) == 2 and ws[1] == rep:
            return [prim[ws[0]]] * 3
        raise ValueError(ws)

    def x(ws):  # WRAP binds tighter than SWAP: x WRAP y -> x y x
        if wrap in ws:
            i = ws.index(wrap)
            return u(ws[:i]) + u(ws[i + 1 :]) + u(ws[:i])
        return u(ws)

    if swap in cmd:  # SWAP has the lowest precedence: x SWAP y -> y x
        i = cmd.index(swap)
        return x(cmd[i + 1 :]) + x(cmd[:i])
    return x(cmd)


def build_miniscan(n=200, seed=0):
    """Per item: 4 primitive nonce words -> 4 colour tokens, 3 function words (repeat / swap / wrap), the 14-item study set of Lake et al. 2019
    (4 primitives, 2 examples per function word, 4 two-function compositions; the 4th primitive appears only alone) and one held-out compositional
    query that uses the 4th primitive with a function word or a composition absent from the study set."""
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        words = rng.sample(NONCE, 7)
        cols = rng.sample(COLOURS, 4)
        p = dict(zip(words[:4], cols))
        fw = {"rep": words[4], "swap": words[5], "wrap": words[6]}
        a, b, c, d = words[:4]  # d = the 4th primitive, shown only alone in the study set
        study = [
            [a],
            [b],
            [c],
            [d],
            [a, fw["rep"]],
            [c, fw["rep"]],
            [a, fw["swap"], c],
            [c, fw["swap"], b],
            [c, fw["wrap"], b],
            [b, fw["wrap"], a],
            [a, fw["rep"], fw["swap"], c],
            [c, fw["swap"], b, fw["rep"]],
            [b, fw["swap"], a, fw["wrap"], c],
            [c, fw["rep"], fw["swap"], b],
        ]
        pool = [
            [d, fw["rep"]],
            [d, fw["swap"], a],
            [b, fw["swap"], d],
            [d, fw["wrap"], c],
            [a, fw["wrap"], d],
            [d, fw["rep"], fw["swap"], b],
            [c, fw["swap"], d, fw["rep"]],
            [d, fw["wrap"], a, fw["rep"]],
            [b, fw["rep"], fw["wrap"], d],
            [a, fw["swap"], d, fw["wrap"], c],
        ]
        q = rng.choice(pool)
        rng.shuffle(study)
        demos = [(" ".join(s), " ".join(miniscan_interpret(s, p, fw))) for s in study]
        target = " ".join(miniscan_interpret(q, p, fw))
        prompt = "\n".join(f"{c_} -> {o_}" for c_, o_ in demos) + f"\n{' '.join(q)} ->"
        items.append(
            {
                "prompt": prompt,
                "target": target,
                "query": " ".join(q),
                "n_ops": sum(1 for w in q if w in fw.values()),
                "tier": "primary",
                "demos": demos,
                "colours": cols,
            }
        )
    return {"task": "miniscan_icl", "n": len(items), "k_demos": 14, "oracle_mismatches": 0, "items": items}


# ---------------------------------------------------------------- MiniSCAN, arithmetic and chain-map panels (format-far composition assays)
ARITH_FAMILY = {
    "a+b+1": lambda a, b: a + b + 1,
    "a+b+2": lambda a, b: a + b + 2,
    "a+2b": lambda a, b: a + 2 * b,
    "2a+b": lambda a, b: 2 * a + b,
    "a-b": lambda a, b: a - b,
    "b-a": lambda a, b: b - a,
    "a*b": lambda a, b: a * b,
    "max": lambda a, b: max(a, b),
    "min": lambda a, b: min(a, b),
    "|a-b|": lambda a, b: abs(a - b),
    "2a-b": lambda a, b: 2 * a - b,
    "a+b-1": lambda a, b: a + b - 1,
}
ARITH_SYMBOLS = ["@", "#", "&", "%", "~", "^"]


def build_arith(n=200, seed=0, k=6):
    """ArithOp-ICL: two novel binary operators on single-digit integers (0-9), each defined ONLY by k demonstrations, never nested in the
    demonstrations; the query nests them once: (a op1 b) op2 c. Each operator's k demonstration pairs are required to identify it uniquely within
    ARITH_FAMILY (every other family member disagrees on at least one demonstration), so the target is determined by the context alone.
    Oracle = the family functions; exact match on the integer string."""
    rng = random.Random(seed)
    items = []
    names = sorted(ARITH_FAMILY)
    while len(items) < n:
        f1, f2 = rng.sample(names, 2)
        s1, s2 = rng.sample(ARITH_SYMBOLS, 2)

        def demos_for(fn, sym):
            for _ in range(50):
                pairs = [(rng.randrange(10), rng.randrange(10)) for _ in range(k)]
                if len(set(pairs)) < k:
                    continue
                vals = [ARITH_FAMILY[fn](a, b) for a, b in pairs]
                if all(any(ARITH_FAMILY[g](a, b) != v for (a, b), v in zip(pairs, vals)) for g in names if g != fn):
                    return [(f"{a} {sym} {b}", str(v)) for (a, b), v in zip(pairs, vals)]
            return None

        d1, d2 = demos_for(f1, s1), demos_for(f2, s2)
        if d1 is None or d2 is None:
            continue
        a, b, c = rng.randrange(10), rng.randrange(10), rng.randrange(10)
        inner = ARITH_FAMILY[f1](a, b)
        target = str(ARITH_FAMILY[f2](inner, c))
        demos = d1 + d2
        rng.shuffle(demos)
        query = f"({a} {s1} {b}) {s2} {c}"
        prompt = "\n".join(f"{x} = {y}" for x, y in demos) + f"\n{query} ="
        items.append(
            {
                "prompt": prompt,
                "target": target,
                "query": query,
                "ops": [f1, f2],
                "symbols": [s1, s2],
                "inner": inner,
                "tier": "primary",
                "demos": demos,
            }
        )
    return {"task": "arith_icl", "n": len(items), "k_demos": 2 * k, "oracle_mismatches": 0, "items": items}


CHAIN_WORDS = [
    "apple",
    "hammer",
    "river",
    "violin",
    "candle",
    "pillow",
    "sparrow",
    "ladder",
    "marble",
    "helmet",
    "cactus",
    "anchor",
    "basket",
    "compass",
    "lantern",
    "mirror",
    "napkin",
    "oyster",
    "pencil",
    "quilt",
    "ribbon",
    "saddle",
    "teapot",
    "umbrella",
    "wagon",
    "yogurt",
    "zipper",
    "button",
    "feather",
    "goblet",
    "kettle",
    "magnet",
    "needle",
    "orange",
    "puzzle",
    "rocket",
    "shovel",
    "ticket",
    "walnut",
    "blanket",
]
CHAIN_CATS = ["KEL", "MOR", "TAV", "ZUN", "PIQ", "DOF", "WEX", "BRU"]
CHAIN_ACTIONS = ["fold", "spin", "tap", "hum", "wave", "nod", "hop", "clap", "point", "bow", "kneel", "wink"]


def build_chainmap(n=200, seed=0):
    """ChainMap-ICL: an in-context word->category mapping (6 ordinary words assigned at random to 3 nonce categories, 2 each) and a category->action
    mapping (3 categories -> 3 actions), each stated separately; three worked examples give word -> action for ONE word of each category; the query
    asks for the action of a word whose category is shown but whose word->action pair never is (two-hop composition). Oracle = the two lookups."""
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        words = rng.sample(CHAIN_WORDS, 6)
        cats = rng.sample(CHAIN_CATS, 3)
        acts = rng.sample(CHAIN_ACTIONS, 3)
        w2c = {words[i]: cats[i % 3] for i in range(6)}
        c2a = dict(zip(cats, acts))
        shown = [words[0], words[1], words[2]]
        held = [words[3], words[4], words[5]]  # one shown / one held word per category
        q = rng.choice(held)
        target = c2a[w2c[q]]
        rules = [f"{w} belongs to {w2c[w]}" for w in words]
        rng.shuffle(rules)
        maps = [f"{c} means {c2a[c]}" for c in cats]
        rng.shuffle(maps)
        ex = [(w, c2a[w2c[w]]) for w in shown]
        rng.shuffle(ex)
        prompt = (
            "Rules:\n"
            + "\n".join(rules)
            + "\n"
            + "\n".join(maps)
            + "\nExamples:\n"
            + "\n".join(f"{w} -> {a}" for w, a in ex)
            + f"\n{q} ->"
        )
        demos = [(w, a) for w, a in ex] + [(r.split(" belongs to ")[0], r.split(" belongs to ")[1]) for r in rules]
        items.append(
            {
                "prompt": prompt,
                "target": target,
                "query": q,
                "category": w2c[q],
                "tier": "primary",
                "demos": demos,
                "actions": acts,
            }
        )
    return {"task": "chainmap_icl", "n": len(items), "k_demos": 12, "oracle_mismatches": 0, "items": items}


# ------------------------------------------------------------------------------------------------------------------ baselines + scoring
def trigrams(s):
    s = s.lower()
    return {s[i : i + 3] for i in range(max(0, len(s) - 2))}


def baselines(panel):
    n = len(panel["items"])
    copy = near = 0
    for it in panel["items"]:
        copy += int(it["query"] == it["target"])
        demos = it["demos"]
        best = max(demos, key=lambda d: len(trigrams(it["query"]) & trigrams(d[-2] if len(d) == 3 else d[0])))
        near += int((best[-1]) == it["target"])
    return {"query_copy": copy / n, "nearest_demo_output": near / n}


def format_valid(task, pred):
    if task == "scan_icl":
        toks = pred.split()
        return bool(toks) and all(
            t in ("I_WALK", "I_RUN", "I_LOOK", "I_JUMP", "I_TURN_LEFT", "I_TURN_RIGHT") for t in toks
        )
    if task == "miniscan_icl":
        toks = pred.split()
        return bool(toks) and all(t in COLOURS for t in toks)
    if task == "arith_icl":
        return bool(re.fullmatch(r"-?\d+", pred.strip()))
    if task == "chainmap_icl":
        return pred.strip() in CHAIN_ACTIONS
    return bool(pred.strip()) and "\n" not in pred.strip()


def score(lm, panel, bs=8, max_new=None):
    import torch

    task = panel["task"]
    max_new = max_new or (64 if task == "scan_icl" else 16 if task == "elemental_icl" else 24)
    preds = []
    for b in range(0, len(panel["items"]), bs):
        prompts = [it["prompt"] for it in panel["items"][b : b + bs]]
        enc = lm.tok(prompts, return_tensors="pt", padding=True).to(lm.device)
        with torch.no_grad():
            out = lm.model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=lm.tok.pad_token_id)
        for o in out:
            txt = lm.tok.decode(o[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            preds.append(txt.split("\n")[0].strip())
    recs = []
    for it, p in zip(panel["items"], preds):
        recs.append(
            {"pred": p, "exact": int(p == it["target"]), "valid": int(format_valid(task, p)), "tier": it["tier"]}
        )
    prim = [r for r in recs if r["tier"] == "primary"] or recs
    n = len(prim)
    res = {
        "task": task,
        "n": n,
        "exact": sum(r["exact"] for r in prim) / n,
        "format_valid": sum(r["valid"] for r in prim) / n,
        "exact_secondary_tier": None,
        "items": recs,
    }
    sec = [r for r in recs if r["tier"] != "primary"]
    if sec:
        res["exact_secondary_tier"] = sum(r["exact"] for r in sec) / len(sec)
        res["n_secondary"] = len(sec)
    return res


# ---------------------------------------------------------------- far assays from BIG-bench task JSON, bAbI and CLUTRR
K = 8
MAX_NEW = 48
MAX_NEW_BY = {"list_functions": 48, "unnatural_icl": 48}  # default 32 for the other families
BABI_TASKS = [1, 15, 16, 17, 19, 20]
BABI_REPO = "Muennighoff/babi"
CLUTRR_REPO, CLUTRR_CFG = "CLUTRR/v1", "gen_train23_test2to10"
SRC_INFO: dict[str, Any] = {}  # dataset revisions recorded into summary.json
UNNATURAL = [
    "dates",
    "dates_unnatural_content",
    "dates_unnatural_content_and_form",
    "dates_unnatural_form",
    "identity",
    "reverse_natural_content",
    "reverse_to_natural_content",
    "unnatural_addition_2_digit",
]
SRC = str(
    EXT / "bigbench"
)  # BIG-bench task JSON (google/BIG-bench benchmark_tasks/<task>/...), fetched by scripts/build_external_sets.py fetch-bigbench


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def render(demos, query, ip, op, sep):
    return sep.join(ip + x + op + y for x, y in demos) + sep + ip + query + op.rstrip(" ")


def build_list_functions():
    items = []
    for fi in range(1, 251):
        fid = f"c{fi:03d}"
        ex = json.load(open(os.path.join(SRC, "list_functions", fid, "task.json")))["examples"]
        rng = random.Random(f"{SEED}-{fid}")
        held = rng.sample(range(len(ex)), 4)
        for hi in held:
            pool = [j for j in range(len(ex)) if j != hi and ex[j]["input"] != ex[hi]["input"]]
            demos = rng.sample(pool, K)
            items.append(
                {
                    "assay": "list_functions",
                    "group": fid,
                    "ex_i": hi,
                    "demo_i": demos,
                    "target": ex[hi]["target"],
                    "prompt": render(
                        [(ex[j]["input"], ex[j]["target"]) for j in demos],
                        ex[hi]["input"],
                        "Input: ",
                        "\nOutput: ",
                        "\n",
                    ),
                }
            )
    random.Random(SEED).shuffle(items)
    return items


def build_unnatural():
    per = {}
    for s in UNNATURAL:
        t = json.load(open(os.path.join(SRC, "unnatural_in_context_learning", s, "task.json")))
        ex = t["examples"]
        ip, op, sep = t["example_input_prefix"], t["example_output_prefix"], t["few_shot_example_separator"]
        assert t["stop_string"] == "\n"
        rng = random.Random(f"{SEED}-{s}")
        held = rng.sample(range(len(ex)), 125)
        rows = []
        for hi in held:
            demos = []
            while len(demos) < K:  # same-subtask, excluding e (by index and by input string), without replacement
                j = rng.randrange(len(ex))
                if j != hi and j not in demos and ex[j]["input"] != ex[hi]["input"]:
                    demos.append(j)
            rows.append(
                {
                    "assay": "unnatural_icl",
                    "group": s,
                    "ex_i": hi,
                    "demo_i": demos,
                    "target": ex[hi]["target"],
                    "prompt": render([(ex[j]["input"], ex[j]["target"]) for j in demos], ex[hi]["input"], ip, op, sep),
                }
            )
        per[s] = rows
    return [per[s][j] for j in range(125) for s in UNNATURAL]  # round-robin: any prefix is class-balanced


def _repo_sha(repo):
    try:
        from huggingface_hub import dataset_info

        return dataset_info(repo).sha
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def qa_block(story, q, a=None):
    return f"Story: {story}\nQuestion: {q}\nAnswer:" + ("" if a is None else f" {a}")


def build_babi(t):
    from datasets import load_dataset

    ds = load_dataset(BABI_REPO)
    SRC_INFO[f"babi_{t}"] = {"repo": BABI_REPO, "sha": _repo_sha(BABI_REPO), "config": "default"}
    def fmt(r):
        return (
            " ".join(x.strip() for x in r["passage"].strip().split("\n") if x.strip()),
            r["question"].strip(),
            r["answer"].strip(),
        )
    te = [fmt(r) for r in ds["test"] if r["task"] == t]
    tr = [fmt(r) for r in ds["train"] if r["task"] == t]
    rng = random.Random(f"{SEED}-babi{t}")
    order = rng.sample(range(len(te)), min(1000, len(te)))
    items = []
    for hi in order:
        demos = []
        while len(demos) < K:
            j = rng.randrange(len(tr))
            if j not in demos and tr[j][:2] != te[hi][:2]:
                demos.append(j)
        items.append(
            {
                "assay": f"babi_{t}",
                "group": f"task{t}",
                "ex_i": hi,
                "demo_i": demos,
                "target": te[hi][2],
                "prompt": "\n\n".join([qa_block(*tr[j]) for j in demos] + [qa_block(te[hi][0], te[hi][1])]),
            }
        )
    return items


def build_clutrr():
    import ast

    from datasets import load_dataset

    files = {
        s: f"hf://datasets/{CLUTRR_REPO}@refs/convert/parquet/{CLUTRR_CFG}/{s}/0000.parquet" for s in ["train", "test"]
    }
    ds = load_dataset("parquet", data_files=files)
    SRC_INFO["clutrr"] = {
        "repo": CLUTRR_REPO,
        "sha": _repo_sha(CLUTRR_REPO),
        "config": CLUTRR_CFG,
        "branch": "refs/convert/parquet",
    }

    def fmt(r):
        n1, n2 = ast.literal_eval(r["query"])
        story = " ".join(r["story"].replace("[", "").replace("]", "").split())
        return (story, f"How is {n2} related to {n1}?", r["target_text"].strip(), r["task_name"])

    te = [fmt(r) for r in ds["test"]]
    tr = [fmt(r) for r in ds["train"]]
    rng = random.Random(f"{SEED}-clutrr")
    order = rng.sample(range(len(te)), min(1000, len(te)))
    items = []
    for hi in order:
        demos = []
        while len(demos) < K:
            j = rng.randrange(len(tr))
            if j not in demos and tr[j][:2] != te[hi][:2]:
                demos.append(j)
        items.append(
            {
                "assay": "clutrr",
                "group": te[hi][3],
                "ex_i": hi,
                "demo_i": demos,
                "target": te[hi][2],
                "prompt": "\n\n".join([qa_block(*tr[j][:3]) for j in demos] + [qa_block(te[hi][0], te[hi][1])]),
            }
        )
    return items


BUILDERS = {"list_functions": build_list_functions, "unnatural_icl": build_unnatural, "clutrr": build_clutrr}
BUILDERS.update({f"babi_{t}": (lambda t=t: build_babi(t)) for t in BABI_TASKS})


def build(assay, limit):
    items = BUILDERS[assay]()
    assert len(items) == 1000, (assay, len(items))
    for j, it in enumerate(items):
        it["idx"] = j
        it["prompt_sha256"] = sha(it["prompt"])
    return items[:limit] if limit else items


class _StopNL:
    def __init__(self, tok, start):
        self.tok = tok
        self.start = start

    def __call__(self, ids, scores, **kw):
        import torch

        done = []
        for row in ids:
            txt = self.tok.decode(row[self.start :], skip_special_tokens=True)
            done.append("\n" in txt or self.tok.eos_token_id in row[self.start :].tolist())
        return torch.tensor(done, device=ids.device)


def generate(model, tok, prompts, dev, bs, max_new=MAX_NEW):
    import torch
    from transformers import StoppingCriteriaList

    out = []
    for b in range(0, len(prompts), bs):
        chunk = prompts[b : b + bs]
        enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
        start = enc.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=StoppingCriteriaList([_StopNL(tok, start)]),
            )
        for row in gen:
            out.append(tok.decode(row[start:], skip_special_tokens=True))
        print(f"  gen {min(b + bs, len(prompts))}/{len(prompts)}", flush=True)
    return out


def score_far_assays(
    lm, out_dir: str | Path, tag: str, assays: list[str], limit: int = 1000, batch_size: int = 16
) -> dict:
    """Greedy generation (stop at the first newline); prediction = first line; exact match -> <out>/<tag>/far_<assay>.jsonl + summary.json."""
    tok, model, dev = lm.tok, lm.model, lm.device
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out_dir = Path(out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    t0 = time.time()
    for assay in assays:
        items = build(assay, limit)
        gens = generate(model, tok, [it["prompt"] for it in items], dev, batch_size, MAX_NEW_BY.get(assay, 32))
        rows = []
        for it, g in zip(items, gens):
            pred = g.split("\n")[0].strip()
            tgt = it["target"].strip()
            rows.append(
                {
                    "idx": it["idx"],
                    "assay": assay,
                    "group": it["group"],
                    "ex_i": it["ex_i"],
                    "demo_i": it["demo_i"],
                    "prompt_sha256": it["prompt_sha256"],
                    "target": it["target"],
                    "prediction": pred,
                    "raw_generation": g,
                    "correct": int(pred == tgt),
                    "correct_ci": int(pred.lower() == tgt.lower()),
                }
            )
        with open(out_dir / f"far_{assay}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        em = sum(r["correct"] for r in rows) / n
        emci = sum(r["correct_ci"] for r in rows) / n
        groups = sorted(set(r["group"] for r in rows))
        summary[assay] = {
            "n": n,
            "k": K,
            "max_new_tokens": MAX_NEW_BY.get(assay, 32),
            "n_distinct_targets": len(set(r["target"] for r in rows)),
            "em": em,
            "em_wilson95": wilson(em, n),
            "em_ci": emci,
            "em_ci_wilson95": wilson(emci, n),
            "n_groups": len(groups),
            "per_group_em": {
                g: sum(r["correct"] for r in rows if r["group"] == g) / sum(1 for r in rows if r["group"] == g)
                for g in groups
            }
            if len(groups) <= 16
            else None,
        }
        print(assay, json.dumps({k: v for k, v in summary[assay].items() if k != "per_group_em"}), flush=True)
    write_json(
        {
            "model": getattr(lm, "name", None),
            "revision": getattr(lm, "revision", None),
            "limit": limit,
            "seed": SEED,
            "max_new_tokens": MAX_NEW,
            "batch_size": batch_size,
            "hf_sources": SRC_INFO,
            "elapsed_s": time.time() - t0,
            "summary": summary,
        },
        out_dir / "summary.json",
    )
    return summary


def score_panels(lm, panels_dir: str | Path, out_path: str | Path, tasks: list[str], bs: int = 8) -> dict:
    """Score the SCAN / ELEMENTAL / MiniSCAN / arithmetic / chain-map panels (json built by build_panels) with greedy generation."""
    out_path = Path(out_path)
    res = read_json(out_path) if out_path.exists() else {}
    for name in tasks:
        panel = read_json(Path(panels_dir) / f"{name}.json")
        res[name] = score(lm, panel, bs)
        print(name, {k: v for k, v in res[name].items() if k != "items"}, flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(res, out_path, indent=None)
    return res


def build_panels(
    panels_dir: str | Path, tasks: list[str], n_scan: int = 200, per_cat: int = 5, n_miniscan: int = 200
) -> dict:
    """Build the panels deterministically into panels_dir (+ report.json with sha256, baselines, oracle checks)."""
    panels_dir = Path(panels_dir)
    panels_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "scan": lambda: build_scan(n=n_scan),
        "elemental": lambda: build_elemental(per_cat=per_cat),
        "miniscan": lambda: build_miniscan(n=n_miniscan),
        "arith": lambda: build_arith(n=n_miniscan),
        "chainmap": lambda: build_chainmap(n=n_miniscan),
    }
    rep = {}
    for name in tasks:
        p = builders[name]()
        p["baselines"] = baselines(p)
        js = json.dumps(p, indent=1, sort_keys=True)
        (panels_dir / f"{name}.json").write_text(js)
        rep[name] = {
            "n": p["n"],
            "sha256": h(js),
            "baselines": p["baselines"],
            "oracle": p.get("oracle_acc", 1.0 if p.get("oracle_mismatches") == 0 else None),
            "oracle_mismatches": p.get("oracle_mismatches"),
            "replaced_over_k": p.get("replaced_over_k"),
            "n_secondary": sum(1 for it in p["items"] if it["tier"] != "primary"),
        }
    write_json(rep, panels_dir / "report.json")
    return rep


def evaluation_strings() -> dict[str, list[str]]:
    """Strings of the external evaluation sets that no training exercise may contain (for the evaluation-window store):
    the near-assay prompts in every condition, the BIG-bench far-assay prompts and targets, and the test texts + label names of
    the symbol-label datasets. Needs network access to the Hub."""
    from datasets import load_dataset

    strings: dict[str, list[str]] = {}
    for ds in NEAR_DATASETS:
        items, names = build_near_set(ds, 1000)
        C = len(names)
        ident, flip, sym = list(range(C)), derange(C), SUL[:C]
        strings[f"near_{ds}"] = (
            [
                s
                for it in items
                for s in (
                    label_prompt(it, names, ident),
                    label_prompt(it, sym, ident),
                    label_prompt(it, names, flip),
                    it["x"],
                )
            ]
            + names
            + sym
        )
    try:
        for assay in ("list_functions", "unnatural_icl"):
            strings[f"far_{assay}"] = [s for it in build(assay, 0) for s in (it["prompt"], str(it["target"]))]
    except Exception as e:  # noqa: BLE001 - the BIG-bench snapshot may be absent
        strings["far_error"] = [f"{type(e).__name__}: {str(e)[:200]}"]
    loaders = {
        "CogComp/trec": lambda: [
            r["text"] for r in load_dataset("CogComp/trec", revision="refs/convert/parquet", split="test")
        ],
        "fancyzhx/dbpedia_14": lambda: [
            f"{r['title'].strip()} {r['content'].strip()}" for r in load_dataset("fancyzhx/dbpedia_14", split="test")
        ],
        "community-datasets/yahoo_answers_topics": lambda: [
            s
            for r in load_dataset("community-datasets/yahoo_answers_topics", split="test")
            for s in (
                f"{r['question_title'].strip()} {r['question_content'].strip()} {r['best_answer'].strip()}",
                f"{r['question_title'].strip()} {r['best_answer'].strip()}",
            )
        ],
        "SetFit/20_newsgroups": lambda: (
            [r["text"] for r in load_dataset("SetFit/20_newsgroups", split="test")]
            + sorted({r["label_text"] for r in load_dataset("SetFit/20_newsgroups", split="test")})
        ),
    }
    for name, fn in loaders.items():
        strings[name] = fn()
    strings["label_names"] = [x for d in LABEL_DATASETS.values() for x in d["names"]]
    return strings

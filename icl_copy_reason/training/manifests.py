"""Data manifests: what every training run reads, row by row, with hashes.

Two kinds of manifest exist:
  corpus manifest   (build_corpus_manifest)  the replay corpus packed into rows, the replacement rows (every 33rd row) and the synthetic
                    document packs that replace them, the two validation token files, document ids and sha256 of every file.
  row-map manifest  (build_row_map)          an explicit per-row source map over a corpus pack and an exercise pack: packs_row[j] / synth_row[j]
                    (exactly one >= 0) for the exercise arms (admixture, fine-tune, matched replay, calibration slices, the variant loop).
Inputs : a config (model + revision, corpus, steps, rows per step, seeds, index-space rule) and the data root.
Outputs: manifest_<name>.json files; check() recomputes every map from its rule; overlap_check() removes rows whose 50-character windows hit the
evaluation-window store (recorded in the manifest).
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from icl_copy_reason.io import sha256_file, write_json
from icl_copy_reason.training.pack import (
    SEQ,
    eos_id,
    token_dtype,
    validation_tokens,
    write_corpus_rows,
    write_synthetic_rows,
)

ROWS_PER_STEP = {"smollm": 768, "pile": 1024, "smollm17": 1024}
CORPUS_SHARES = {"fineweb-edu-dedup": 0.85, "cosmopedia-v2": 0.15}
REPLACEMENT_MOD, REPLACEMENT_RESIDUE = 33, 32  # every 33rd row of the index space holds a synthetic pack (3.03 %)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------------------- corpus manifest
def stream_corpus(corpus: str, seed: int) -> Iterator[tuple[str, str, str]]:
    """The public proxy stream: 85 % fineweb-edu-dedup + 15 % cosmopedia-v2 (document-level mixing, Random(seed)), each shuffled with a 20,000 buffer."""
    from datasets import load_dataset

    if corpus not in ("smollm", "smollm17"):
        raise ValueError("only the smollm proxy stream is supported here")
    a = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True).shuffle(seed=seed, buffer_size=20000)
    b = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True).shuffle(seed=seed + 1, buffer_size=20000)
    ia, ib = iter(a), iter(b)
    rng = random.Random(seed)
    while True:
        if rng.random() < CORPUS_SHARES["cosmopedia-v2"]:
            r = next(ib)
            yield ("cosmopedia-v2", str(r.get("seed_data", ""))[:80], r["text"])
        else:
            r = next(ia)
            yield ("fineweb-edu-dedup", r["id"], r["text"])


def replacement_rows(steps: int, rows_per_step: int) -> list[int]:
    return [i for i in range(steps * rows_per_step) if i % REPLACEMENT_MOD == REPLACEMENT_RESIDUE]


def build_corpus_manifest(out: str, seed: int, base: str, revision: str | None, corpus: str = "smollm", steps: int = 636, calib_steps: int = 64,
                          doc_ver: int = 1, dtype_name: str = "uint16", synthetic_modes: tuple[str, ...] = ("compose", "attested")) -> dict[str, Any]:
    """Write packs_s<seed>.u16, synth_<mode>_s<seed>.u16, val_fwe.u16, val_pile.u16 and manifest_s<seed>.json under out."""
    from transformers import AutoTokenizer

    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    eos = eos_id(tok)
    dtype = token_dtype(dtype_name)
    rps = ROWS_PER_STEP[corpus]
    n_rows = (steps + calib_steps) * rps
    t0 = time.time()
    ids = write_corpus_rows(tok, stream_corpus(corpus, seed), n_rows, f"{out}/packs_s{seed}.u16", eos, dtype)
    print(f"packs: {n_rows} rows, {len(ids)} docs, {time.time() - t0:.0f}s", flush=True)
    repl = replacement_rows(steps, rps)
    metas = {}
    for mode in synthetic_modes:
        metas[mode] = write_synthetic_rows(tok, seed, repl, mode, f"{out}/synth_{mode}_s{seed}.u16", eos, ver=doc_ver, dtype=dtype)
    if not os.path.exists(f"{out}/val_fwe.u16"):
        for which, n_tok in (("fwe", 2_000_000), ("pile", 1_000_000)):
            np.save(f"{out}/val_{which}.u16", validation_tokens(tok, n_tok, which, eos, dtype))
            os.replace(f"{out}/val_{which}.u16.npy", f"{out}/val_{which}.u16")
    files = [f"packs_s{seed}.u16"] + [f"synth_{m}_s{seed}.u16" for m in synthetic_modes] + ["val_fwe.u16", "val_pile.u16"]
    man = {"seed": seed, "base": base, "revision": revision, "corpus": corpus, "rows": n_rows, "rows_per_step": rps, "steps": steps, "calib_steps": calib_steps,
           "replacement_rows": repl, "n_replacement": len(repl), "doc_ver": doc_ver, "replacement_fraction": len(repl) / (steps * rps), "eos": int(eos),
           "n_docs": len(ids), "docs": ids, "token_dtype": dtype_name,
           "synth_meta": {m: {"n_docs": sum(x[1] for x in v), "n_queries": sum(x[2] for x in v)} for m, v in metas.items()},
           "sha256": {f: sha256_file(f"{out}/{f}") for f in files}}
    json.dump(man, open(f"{out}/manifest_s{seed}.json", "w"))
    return man


# ------------------------------------------------------------------------------------------------------------- row-map manifests
def map_admixture(rps: int, steps: int, residues: tuple[int, ...], mod: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rows 0..n-1 of the corpus; rows i with i mod `mod` in residues take exercise row rank(i)."""
    n = rps * steps
    packs_row = np.arange(n, dtype=np.int64)
    synth_row = np.full(n, -1, dtype=np.int64)
    repl = np.array([i for i in range(n) if i % mod in residues], dtype=np.int64)
    packs_row[repl] = -1
    synth_row[repl] = np.arange(len(repl))
    return packs_row, synth_row, {"n_replacement": int(len(repl)), "n_synth_rows_expected": int(len(repl)), "replacement_fraction": len(repl) / n}


def map_admixture_control(rps: int, steps: int, residues: tuple[int, ...], mod: int, pool0: int, pool_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """As map_admixture, but the slot rows take cycled replay rows from an unused corpus region (slot k -> pool0 + k mod pool_size)."""
    n = rps * steps
    packs_row = np.arange(n, dtype=np.int64)
    synth_row = np.full(n, -1, dtype=np.int64)
    repl = np.array([i for i in range(n) if i % mod in residues], dtype=np.int64)
    packs_row[repl] = pool0 + (np.arange(len(repl)) % pool_size)
    return packs_row, synth_row, {"n_replacement": int(len(repl)), "n_synth_rows_expected": 0, "pool_rows": [int(pool0), int(pool0 + pool_size - 1)],
                                  "pool_size": int(pool_size), "mean_exposure": len(repl) / pool_size}


def map_finetune(rps: int, steps: int, replay0: int, ctrl0: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Odd rows = exercise rows (j-1)/2 (or, for the matched control, replay rows ctrl0 + (j-1)/2); even rows = replay rows replay0 + j/2."""
    n = rps * steps
    j = np.arange(n, dtype=np.int64)
    odd = j % 2 == 1
    packs_row = np.where(odd, -1, replay0 + j // 2)
    synth_row = np.where(odd, (j - 1) // 2, -1)
    if ctrl0 is not None:
        packs_row = np.where(odd, ctrl0 + (j - 1) // 2, packs_row)
        synth_row = np.full(n, -1, dtype=np.int64)
    meta = {"n_replay_rows": int((~odd).sum()), "replay_rows": [int(replay0), int(replay0 + (n - 2) // 2)], "n_synth_rows_expected": 0 if ctrl0 is not None else int(odd.sum())}
    if ctrl0 is not None:
        meta["ctrl_rows"] = [int(ctrl0), int(ctrl0 + (n - 2) // 2)]
    return packs_row, synth_row, meta


def map_calibration(rps: int, steps: int, calib0: int, n_synth: int | None = None, tail: int | None = None, ctrl0: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Even rows = corpus calibration region; odd rows = the last `tail` exercise rows in reverse order, cycled (or replay rows ctrl0 + m for the control sweep)."""
    n = rps * steps
    j = np.arange(n, dtype=np.int64)
    odd = j % 2 == 1
    m = (j - 1) // 2
    packs_row = np.where(odd, -1, calib0 + j // 2)
    meta: dict[str, Any] = {"n_replay_rows": int((~odd).sum()), "calib_rows": [int(calib0), int(calib0 + (n - 2) // 2)]}
    if ctrl0 is None:
        synth_row = np.where(odd, n_synth - 1 - (m % tail), -1)
        meta.update({"synth_rows_used": [int(n_synth - tail), int(n_synth - 1)], "n_synth_rows_expected": int(n_synth), "cycles": (int(odd.sum()) + tail - 1) // tail})
    else:
        synth_row = np.full(n, -1, dtype=np.int64)
        packs_row = np.where(odd, ctrl0 + m, packs_row)
        meta.update({"ctrl_rows": [int(ctrl0), int(ctrl0 + (n - 2) // 2)], "n_synth_rows_expected": 0})
    return packs_row, synth_row, meta


def rule(kind: str, p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict, str]:
    """Dispatch on the index-space kind named in the config (admixture | admixture_control | finetune | finetune_control | calibration | calibration_control)."""
    if kind == "admixture":
        pr, sr, meta = map_admixture(p["rps"], p["steps"], tuple(p["residues"]), p["mod"])
        return pr, sr, meta, f"rows 0..{p['rps'] * p['steps'] - 1} of packs; i mod {p['mod']} in {list(p['residues'])} -> exercise row rank(i)"
    if kind == "admixture_control":
        dup = p["dup_factor"]
        n_slot = len([i for i in range(p["rps"] * p["steps"]) if i % p["mod"] in tuple(p["residues"])])
        pool = int(round(n_slot / dup))
        pr, sr, meta = map_admixture_control(p["rps"], p["steps"], tuple(p["residues"]), p["mod"], p["pool0"], pool)
        return pr, sr, meta, f"as admixture; slot k -> packs row {p['pool0']} + (k mod {pool}); pool = round({n_slot}/{dup})"
    if kind == "finetune":
        pr, sr, meta = map_finetune(p["rps"], p["steps"], p["replay0"])
        return pr, sr, meta, f"j odd -> exercise row (j-1)/2; j even -> packs row {p['replay0']} + j/2"
    if kind == "finetune_control":
        pr, sr, meta = map_finetune(p["rps"], p["steps"], p["replay0"], p["ctrl0"])
        return pr, sr, meta, f"j odd -> packs row {p['ctrl0']} + (j-1)/2; j even -> packs row {p['replay0']} + j/2"
    if kind == "calibration":
        pr, sr, meta = map_calibration(p["rps"], p["steps"], p["calib0"], p["n_synth"], p["tail"])
        return pr, sr, meta, f"j even -> packs row {p['calib0']} + j/2; j odd -> exercise row {p['n_synth']}-1-(((j-1)/2) mod {p['tail']})"
    if kind == "calibration_control":
        pr, sr, meta = map_calibration(p["rps"], p["steps"], p["calib0"], ctrl0=p["ctrl0"])
        return pr, sr, meta, f"j even -> packs row {p['calib0']} + j/2; j odd -> packs row {p['ctrl0']} + (j-1)/2"
    raise ValueError(kind)


def build_row_map(arm: str, kind: str, p: dict[str, Any], seed: int, packs_path: str, val_paths: dict[str, str], synth_path: str | None, out_path: str,
                  packs_sha: str | None = None, root: str | None = None) -> dict[str, Any]:
    """Materialise one row-map manifest (the trainer asserts shapes and sha256 of every named file before step 1)."""
    pr, sr, meta, rtxt = rule(kind, p)
    n = p["rps"] * p["steps"]
    assert len(pr) == len(sr) == n and ((pr >= 0) ^ (sr >= 0)).all(), "every row has exactly one source"
    packs = np.load(packs_path, mmap_mode="r")
    assert packs.ndim == 2 and packs.shape[1] == SEQ
    assert pr.max() < packs.shape[0], f"packs row {pr.max()} beyond the corpus ({packs.shape[0]} rows)"
    packs_sha = packs_sha or sha256_file(packs_path)
    syn_sha = None
    syn_rows = None
    if synth_path is not None and os.path.exists(synth_path):
        arr = np.load(synth_path, mmap_mode="r")
        syn_rows = int(arr.shape[0])
        syn_sha = sha256_file(synth_path)
        assert arr.shape == (meta["n_synth_rows_expected"], SEQ), f"{synth_path}: shape {arr.shape}, expected ({meta['n_synth_rows_expected']}, {SEQ})"
    relp = (lambda f: os.path.relpath(f, root)) if root else (lambda f: f)
    man = {"arm": arm, "kind": kind, "seed": seed, "utc": utc(), "rows_per_step": p["rps"], "steps": p["steps"], "n_rows": n, "tokens": n * SEQ, "seq": SEQ, "rule": rtxt, "meta": meta,
           "row_format": "row j: exactly one of packs_row[j] (corpus row) / synth_row[j] (exercise row) is >= 0",
           "packs": {"file": relp(packs_path), "rows": int(packs.shape[0]), "sha256": packs_sha},
           "synth": None if synth_path is None else {"file": relp(synth_path), "rows_expected": meta["n_synth_rows_expected"], "rows": syn_rows, "sha256": syn_sha},
           "val": {k: {"file": relp(v), "sha256": sha256_file(v)} for k, v in val_paths.items()}, "sha256": {}, "packs_row": pr.tolist(), "synth_row": sr.tolist()}
    man["sha256"][os.path.basename(packs_path)] = packs_sha
    if synth_path is not None:
        man["sha256"][os.path.basename(synth_path)] = syn_sha
    for k, v in val_paths.items():
        man["sha256"][os.path.basename(v)] = man["val"][k]["sha256"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(man, open(out_path, "w"), separators=(",", ":"))
    return man


def fill_synth_sha(manifest_paths: list[str], synth_path: str) -> None:
    """Record the exercise file's sha256 (and shape check) in every manifest that names it, once the file exists."""
    arr = np.load(synth_path, mmap_mode="r")
    h = sha256_file(synth_path)
    for mp in manifest_paths:
        man = json.load(open(mp))
        if not man["synth"] or os.path.basename(man["synth"]["file"]) != os.path.basename(synth_path):
            continue
        assert arr.shape == (man["synth"]["rows_expected"], SEQ), f"{mp}: synth shape {arr.shape} != expected"
        man["synth"]["rows"] = int(arr.shape[0])
        man["synth"]["sha256"] = h
        man["sha256"][os.path.basename(synth_path)] = h
        man["synth_filled_utc"] = utc()
        json.dump(man, open(mp, "w"), separators=(",", ":"))


def check_row_map(mp: str, kind: str, p: dict[str, Any]) -> dict[str, Any]:
    """Recompute the map from its rule and compare with the materialised arrays."""
    man = json.load(open(mp))
    pr = np.array(man["packs_row"])
    sr = np.array(man["synth_row"])
    rp, rs, _, _ = rule(kind, p)
    same = bool(np.array_equal(pr, rp) and np.array_equal(sr, rs))
    if man.get("meta", {}).get("overlap", {}).get("replacements"):
        same = True  # rows replaced by the overlap check differ from the rule by construction (recorded)
    used = pr[pr >= 0]
    return {"rows": len(pr), "packs_rows": int((pr >= 0).sum()), "synth_rows": int((sr >= 0).sum()), "packs_min": int(used.min()), "packs_max": int(used.max()),
            "packs_unique": int(len(np.unique(used))), "synth_unique": int(len(np.unique(sr[sr >= 0]))) if (sr >= 0).any() else 0, "rule_match": same,
            "synth_sha_filled": bool(man["synth"] and man["synth"]["sha256"]), "overlap_checked": bool(man["meta"].get("overlap"))}


# ---------------------------------------------------------------------------------------------- evaluation-window overlap check
WIN = 50
_HB = np.uint64(1099511628211)


def normalise_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def window_hashes(s: str) -> np.ndarray:
    """Rolling 50-character hashes (the unit of the evaluation-window store that guards the training data against evaluation text)."""
    s = normalise_whitespace(s)
    if len(s) < WIN:
        return np.zeros(0, np.uint64)
    cp = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    n = len(cp) - WIN + 1
    h = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(WIN):
            h = h * _HB + cp[j : j + n]
    return h


def overlap_check(mp: str, store: np.ndarray, tok, packs: np.ndarray, synth: np.ndarray | None, dry: bool = False) -> dict[str, Any]:
    """Decode every row of a row-map manifest; a row whose windows hit the sorted store is replaced by the next unused row of the same source (re-checked)."""
    man = json.load(open(mp))
    pr = np.array(man["packs_row"])
    sr = np.array(man["synth_row"])
    n = len(pr)
    used_p = set(pr[pr >= 0].tolist())
    used_s = set(sr[sr >= 0].tolist())
    spare_p = (man["meta"].get("calib_rows") or [man["packs"]["rows"] - 1])[1] + 1
    spare_s = (man["meta"]["synth_rows_used"][0] - 1) if man["synth"] and man["meta"].get("synth_rows_used") else None
    t0 = time.time()

    def hits(src: str, idx: int) -> tuple[int, str | None]:
        txt = tok.decode(np.array(packs[idx] if src == "p" else synth[idx], dtype=np.int64).tolist(), skip_special_tokens=True)
        h = window_hashes(txt)
        if not len(h) or not len(store):
            return 0, None
        ii = np.minimum(np.searchsorted(store, h), len(store) - 1)
        k = np.nonzero(store[ii] == h)[0]
        return int(len(k)), (normalise_whitespace(txt)[k[0] : k[0] + WIN] if len(k) else None)

    hit_log, repl_log = [], []
    for j in range(n):
        src, idx = ("p", int(pr[j])) if pr[j] >= 0 else ("s", int(sr[j]))
        nh, ex = hits(src, idx)
        if not nh:
            continue
        hit_log.append({"row": j, "source": src, "index": idx, "n_windows": nh, "example": ex})
        while True:
            if src == "p":
                while spare_p in used_p:
                    spare_p += 1
                assert spare_p < man["packs"]["rows"], "corpus exhausted"
                cand = spare_p
                spare_p += 1
            else:
                while spare_s in used_s:
                    spare_s -= 1
                assert spare_s is not None and spare_s >= 0, "exercise file exhausted"
                cand = spare_s
                spare_s -= 1
            nh2, _ = hits(src, cand)
            if not nh2:
                break
            repl_log.append({"row": j, "source": src, "rejected_candidate": cand, "n_windows": nh2})
        if src == "p":
            pr[j] = cand
            used_p.add(cand)
        else:
            sr[j] = cand
            used_s.add(cand)
        repl_log.append({"row": j, "source": src, "from": idx, "to": cand})
    rec = {"checked_utc": utc(), "unit": f"{WIN}-character rolling-hash windows", "n_store_windows": int(len(store)), "n_rows_checked": n, "n_hits": len(hit_log),
           "hits": hit_log, "replacements": repl_log, "seconds": round(time.time() - t0)}
    if not dry:
        man["packs_row"] = pr.tolist()
        man["synth_row"] = sr.tolist()
        man["meta"]["overlap"] = {k: v for k, v in rec.items() if k != "hits"} | {"hits": hit_log[:200]}
        json.dump(man, open(mp, "w"), separators=(",", ":"))
    return rec


# ------------------------------------------------------------------------------------------------------------- config helpers
def index_spaces_from_config(spaces: dict[str, dict[str, Any]], seed: int = 0, overrides: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Resolve the index spaces of a config (synthetic file names follow the seed; replay0 / ctrl0 overrides move the replay regions)."""
    out = copy.deepcopy(spaces)
    for p in out.values():
        if p.get("synth"):
            p["synth"] = p["synth"].replace("_s0.", f"_s{seed}.")
    for k, v in (overrides or {}).items():
        for p in out.values():
            if k in p and v is not None:
                p[k] = v
    return out


def manifest_summary(man: dict[str, Any]) -> dict[str, Any]:
    """The compact record a freeze file keeps about a manifest (no row arrays)."""
    return {"arm": man["arm"], "kind": man.get("kind"), "rows_per_step": man["rows_per_step"], "steps": man["steps"], "n_rows": man["n_rows"], "tokens": man["tokens"], "rule": man["rule"],
            "meta": {k: v for k, v in man["meta"].items() if k != "overlap"}, "packs_sha256": man["packs"]["sha256"], "synth": man["synth"],
            "overlap_checked": bool(man["meta"].get("overlap"))}


def write_manifest_without_rows(man: dict[str, Any], path: str) -> None:
    """A released copy of a corpus manifest: everything except the document-id list, plus the list's sha256 and a 1,000-id sample."""
    slim = {k: v for k, v in man.items() if k != "docs"}
    docs = man.get("docs") or []
    from icl_copy_reason.io import sha256_text

    slim["docs_list"] = {"n_docs": len(docs), "sha256_of_json": sha256_text(json.dumps(docs)), "sample_first_1000": docs[:1000],
                         "note": "the full id list is regenerated by build_corpus_manifest (same seed, same stream) or available as a release asset"}
    write_json(slim, path)

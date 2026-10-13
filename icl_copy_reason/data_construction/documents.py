"""Public document pools and the deterministic, content-blind document sampler.

Inputs : the pool table (eight public Hugging Face datasets at pinned revisions; `POOLS`), a sampling spec
         (quotas per pool, eligibility bounds, the residue rule, exclusion records of earlier mining runs) from
         configs/*-data.yaml, and the forbidden-string regex of the prompt table.
Outputs: docs.jsonl (one record per sampled document: pool, file, file_name, row, sha256, chars, text, rank, type)
         and sample_summary.json (quotas, shortfalls, per-file counts) in the output directory.

Sampling rule (fixed before the run). Per pool, files are read in the listed order. A document is ELIGIBLE iff
min_chars <= len(text) <= max_chars, it is not an earlier mining candidate (identity (pool, file name, row)), its sha256 is
not among the classifier-selected documents of the mined-document arm, and the forbidden regex finds no match. Among
eligible documents in stream order, eligible-index % mod == residue is taken until the pool quota. Files are streamed in
parallel but consumed in file order, so the output is deterministic. A worker stops a file after quota*mod+mod eligible
documents. Shortfalls are topped up from the FineWeb-Edu pool (P4) and recorded.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from ..io import read_json, read_jsonl, write_json

POOL_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

# id: (repo, revision, file list or None = list the repo, loader kind, text field, description)
POOLS: dict[str, tuple] = {
    "P1": ("EleutherAI/proof-pile-2", "901a9273a770e9d4138c5ddd91802f9c5c6cdc4b", [f"arxiv/train/arXiv_{i:03d}.jsonl.zst" for i in range(100)], "json", "text", "proof-pile-2 arXiv"),
    "P2": ("EleutherAI/proof-pile-2", "901a9273a770e9d4138c5ddd91802f9c5c6cdc4b", [f"open-web-math/train/shard-{i:04d}.jsonl.zst" for i in range(63)], "json", "text", "proof-pile-2 OpenWebMath"),
    "P3": ("HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594", [f"finemath-4plus/train-{i:05d}-of-00064.parquet" for i in range(64)], "parquet", "text", "FineMath-4+"),
    "P4": ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", [f"data/CC-MAIN-2025-26/000_{i:05d}.parquet" for i in range(25)], "parquet", "text", "FineWeb-Edu CC-MAIN-2025-26 (files 0-24: mining; files 25-49: exercise documents)"),
    "P5": ("marin-community/ar5iv-no-problem-markdown", "aa24ae3b07113ca787c1346aa30275437ee11c01", None, "json", "text", "Marin ar5iv-no-problem markdown"),
    "P6": ("pile-of-law/pile-of-law", "2e96169e7e4b43f8ea36230515ebb44b27423b94", ["data/train.eurlex.jsonl.xz"], "json", "text", "Pile of Law EUR-Lex"),
    "P7": ("marin-community/stackexchange-markdown", "5a39d5689f9b867e76e0a07afa2742a68a78cef9", None, "json", "text", "Marin StackExchange markdown"),
    "P8": ("marin-community/wikipedia-markdown", "4c6c554cca365a075834805cc0d15d11cbdc7aaa", None, "json", "text", "Marin Wikipedia markdown"),
}
# the exercise documents use a FRESH FineWeb-Edu slice (files 25-49) so that they never overlap the mining slice (files 0-24)
P4_EXERCISE_FILES = [f"data/CC-MAIN-2025-26/000_{i:05d}.parquet" for i in range(25, 50)]


def pool_table() -> dict[str, dict[str, Any]]:
    """The pool table as plain JSON (shipped as data/exercises/source_pools.json)."""
    return {p: {"repo": v[0], "revision": v[1], "files": v[2], "loader": v[3], "text_field": v[4], "description": v[5]} for p, v in POOLS.items()}


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def list_pool_files(pid: str, files_record: str | Path | None = None, p4_files: list[str] | None = None) -> list[tuple[str, str]]:
    """(url, file name) pairs for a pool: the recorded file list of the mining run when given, else the pinned table / the Hub listing."""
    repo, rev, files, _, _, _ = POOLS[pid]
    if files_record and Path(files_record).exists():
        urls = read_json(files_record)["files"]
    else:
        if pid == "P4" and p4_files is not None:
            files = p4_files
        if files is None:
            from huggingface_hub import list_repo_files

            fs = [f for f in list_repo_files(repo, repo_type="dataset", revision=rev) if f.endswith(".jsonl.gz")]
            if pid == "P8":
                fs = [f for f in fs if f.startswith("enwiki_namespace_0_")]
            files = sorted(fs)
        urls = [f"hf://datasets/{repo}@{rev}/{f}" for f in files]
    return [(u, u.split(f"@{rev}/", 1)[1]) for u in urls]


def load_exclusions(mining_dir: str | Path | None, selected_docs: str | Path | None) -> tuple[dict, set, dict]:
    """Earlier mining candidates by (pool, file name, row) and classifier-selected documents by text sha256."""
    keys: dict = collections.defaultdict(set)
    n_rows = 0
    if mining_dir:
        mining_dir = Path(mining_dir)
        for pid in POOL_ORDER:
            fp = mining_dir / f"files_{pid}.json"
            if not fp.exists():
                continue
            names = [u.split(f"@{POOLS[pid][1]}/", 1)[1] for u in read_json(fp)["files"]]
            for src in (mining_dir / f"cands_{pid}.jsonl", mining_dir / "admitted.jsonl"):
                if not src.exists():
                    continue
                for d in read_jsonl(src):
                    if d["pool"] != pid:
                        continue
                    keys[(pid, names[int(d["file"])])].add(int(d["row"]))
                    n_rows += 1
    shas = {sha_text(d["text"]) for d in read_jsonl(selected_docs)} if selected_docs and Path(selected_docs).exists() else set()
    return keys, shas, {"mining_rows": n_rows, "mining_files": len(keys), "selected_doc_shas": len(shas)}


def _sample_file(job: tuple) -> tuple:
    pid, fi, url, _name, max_elig, tmp, excl_rows, excl_shas, doc_min, doc_max, forbid_pattern = job
    from datasets import load_dataset

    forbid = re.compile(forbid_pattern, re.I | re.M)
    kind, field = POOLS[pid][3], POOLS[pid][4]
    c: collections.Counter = collections.Counter()
    out = f"{tmp}/{pid}_{fi:05d}.jsonl"
    err = None
    t0 = time.time()
    with open(out, "w") as f:
        try:
            ds = load_dataset(kind, data_files=url, split="train", streaming=True)
            for ri, r in enumerate(ds):
                text = r.get(field) or ""
                c["read"] += 1
                n = len(text)
                if n < doc_min or n > doc_max:
                    c["chars_out"] += 1
                    continue
                if ri in excl_rows:
                    c["excl_mining"] += 1
                    continue
                h = sha_text(text)
                if h in excl_shas:
                    c["excl_selected"] += 1
                    continue
                if forbid.search(text):
                    c["forbidden"] += 1
                    continue
                c["eligible"] += 1
                f.write(json.dumps({"row": ri, "sha256": h, "chars": n, "text": text}) + "\n")
                if c["eligible"] >= max_elig:
                    c["stopped_early"] = 1
                    break
        except Exception as e:  # noqa: BLE001 - a failed file is recorded, not fatal
            c["file_error"] = 1
            err = f"{type(e).__name__}: {str(e)[:200]}"
    return fi, dict(c), out, err, round(time.time() - t0)


class PoolSampler:
    """Streams one pool's files in order and takes every document whose eligible index satisfies the residue rule."""

    def __init__(self, pid: str, mp_pool, batch: int, tmp: str, excl_keys: dict, excl_shas: set, base_quota: int, spec: dict, files: list[tuple[str, str]]):
        self.pid, self.mp, self.batch, self.tmp, self.excl, self.shas, self.spec = pid, mp_pool, batch, tmp, excl_keys, excl_shas, spec
        self.files = files
        self.fi = 0
        self.gidx = 0
        self.sel: list[dict] = []
        self.counts: collections.Counter = collections.Counter()
        self.file_log: list[dict] = []
        self.pending: list = []
        self.max_elig = base_quota * spec["mod"] + spec["mod"]

    def take(self, quota: int) -> list[dict]:
        """Append documents to self.sel until len == quota (or the pool is exhausted); the stream and the index continue across calls."""
        mod, res = self.spec["mod"], self.spec["residue"]
        while len(self.sel) < quota:
            if not self.pending:
                if self.fi >= len(self.files):
                    break
                jobs = [
                    (self.pid, fi, u, n, self.max_elig, self.tmp, self.excl.get((self.pid, n), set()), self.shas, self.spec["min_chars"], self.spec["max_chars"], self.spec["forbidden_regex"])
                    for fi, (u, n) in enumerate(self.files[self.fi : self.fi + self.batch], self.fi)
                ]
                self.fi += len(jobs)
                for r in [self.mp.apply_async(_sample_file, (j,)) for j in jobs]:
                    fi, c, path, err, secs = r.get()
                    self.counts.update(c)
                    self.file_log.append({"file": fi, "name": self.files[fi][1], "counts": c, "error": err, "seconds": secs})
                    print(f"{self.pid} file {fi}/{len(self.files)} read={c.get('read', 0)} eligible={c.get('eligible', 0)} {secs}s {err or ''}", flush=True)
                    self.pending.append((fi, path, open(path)))
            fi, path, fh = self.pending[0]
            for line in fh:
                d = json.loads(line)
                if self.gidx % mod == res:
                    self.sel.append({"pool": self.pid, "file": fi, "file_name": self.files[fi][1], **d})
                self.gidx += 1
                if len(self.sel) >= quota:
                    break
            else:
                fh.close()
                os.unlink(path)
                self.pending.pop(0)
        return self.sel

    def close(self) -> None:
        for _fi, path, fh in self.pending:
            fh.close()
            os.unlink(path)
        self.pending = []


def assign_type_by_hash(text: str, types: list[str]) -> str:
    """Type assignment of the first exercise set: Random(int(sha256(text)[:16], 16)).choice(sorted(types))."""
    return random.Random(int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)).choice(sorted(types))


def sample_documents(spec: dict, outdir: str | Path, workers: int = 32, batch: int = 4, tmp: str | None = None, pilot: int = 0, pilot_skip: int = 0) -> dict:
    """Run the sampler for every pool and write docs.jsonl + sample_summary.json.

    spec keys: quotas (per pool at `target_docs`), target_docs, target (this run), mod, residue, min_chars, max_chars,
    forbidden_regex, type_rule ('hash' with `types`, or 'draw' with `type_weights` and `type_seed`), p4_files (list or
    None), mining_dir / selected_docs (exclusions; optional), files_records (dir of files_<pool>.json; optional),
    pilot_type (type given to pilot documents; 'h' for the top-up pilot, None = by rule), pilot_per_pool.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tmp or str(outdir / "_tmp")
    os.makedirs(tmp, exist_ok=True)
    n_pilot = pilot + pilot_skip
    if pilot:
        quotas = {p: spec.get("pilot_per_pool") or -(-n_pilot // len(POOL_ORDER)) for p in POOL_ORDER}
    else:
        quotas = {p: int(round(q * spec["target"] / spec["target_docs"])) for p, q in spec["quotas"].items()}
    excl_keys, excl_shas, excl_info = load_exclusions(spec.get("mining_dir"), spec.get("selected_docs"))
    print("exclusions", excl_info, "quotas", quotas, flush=True)
    t0 = time.time()
    files_records = spec.get("files_records")
    files = {p: list_pool_files(p, f"{files_records}/files_{p}.json" if files_records and p != "P4" else None, spec.get("p4_files")) for p in POOL_ORDER}
    with mp.Pool(workers) as pool:
        samplers = {p: PoolSampler(p, pool, batch, tmp, excl_keys, excl_shas, quotas[p], spec, files[p]) for p in POOL_ORDER}
        with cf.ThreadPoolExecutor(len(POOL_ORDER)) as ex:
            list(ex.map(lambda p: samplers[p].take(quotas[p]), POOL_ORDER))
        short = {p: quotas[p] - len(samplers[p].sel) for p in POOL_ORDER if len(samplers[p].sel) < quotas[p]}
        topup = 0
        if short and not pilot:
            topup = sum(short.values())
            print(f"shortfall {short}: topping up {topup} from P4", flush=True)
            samplers["P4"].take(quotas["P4"] + topup)
        for s in samplers.values():
            s.close()
    docs: list[dict] = []
    for p in POOL_ORDER:
        for k, d in enumerate(samplers[p].sel):
            docs.append({**d, "rank": k})
    if spec["type_rule"] == "hash":
        for d in docs:
            d["type"] = assign_type_by_hash(d["text"], spec["types"])
    if pilot:
        chosen: list[dict] = []
        maxr = max((len(samplers[p].sel) for p in POOL_ORDER), default=0)
        byp = {p: [d for d in docs if d["pool"] == p] for p in POOL_ORDER}
        per: collections.Counter = collections.Counter()
        for r in range(maxr):
            for p in POOL_ORDER:
                if r >= len(byp[p]):
                    continue
                d = byp[p][r]
                if spec.get("pilot_type"):
                    if len(chosen) < n_pilot:
                        chosen.append({**d, "type": spec["pilot_type"], "pilot_rank": len(chosen)})
                elif per[d["type"]] < pilot // len(spec["types"]):
                    per[d["type"]] += 1
                    chosen.append({**d, "pilot_rank": len(chosen)})
        docs = chosen[pilot_skip:]
        rule = f"pilot: quota {quotas['P1']} per pool by the same rule; round-robin by rank over P1..P8; positions {pilot_skip}..{n_pilot - 1}"
    elif spec["type_rule"] == "draw":
        w = spec["type_weights"]
        ts = sorted(w)
        draws = random.Random(spec["type_seed"]).choices(ts, weights=[w[t] for t in ts], k=len(docs))
        for d, t in zip(docs, draws):
            d["type"] = t
        rule = f"type per document = Random({spec['type_seed']}).choices over {w}"
    else:
        rule = "type per document = Random(int(sha256(text)[:16], 16)).choice(sorted(types))"
    with open(outdir / "docs.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    summ = {
        "utc": utc(), "seconds": round(time.time() - t0), "target": spec["target"], "pilot": pilot, "type_rule": rule, "residue": spec["residue"], "mod": spec["mod"],
        "quotas": quotas, "topup_from_P4": topup, "shortfall": short, "exclusion_sources": excl_info, "forbidden_regex_flags": "re.I|re.M", "n_docs": len(docs),
        "per_type": dict(collections.Counter(d["type"] for d in docs)),
        "pools": {p: {"desc": POOLS[p][5], "revision": POOLS[p][1], "n_files_listed": len(samplers[p].files), "files_streamed": samplers[p].fi, "taken": len(samplers[p].sel),
                      "eligible_indexed": samplers[p].gidx, "counts": dict(samplers[p].counts), "files": samplers[p].file_log} for p in POOL_ORDER},
    }
    write_json(summ, outdir / "sample_summary.json")
    print(json.dumps({k: v for k, v in summ.items() if k != "pools"}, indent=1))
    return summ

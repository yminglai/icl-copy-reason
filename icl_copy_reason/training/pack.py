"""Token packing for every training arm.

Inputs : a tokenizer, document streams or exercise records, a target number of 2,048-token rows.
Outputs: uint16 / uint32 memmap files of shape (rows, 2048) with EOS between documents, plus per-row records.

Four packers share this module:
  write_corpus_rows     stream documents into rows (the replay corpus of the admixture runs)
  write_synthetic_rows  rows of synthetic definition-use documents, one deterministic generator per row (Random(seed * 10**6 + row))
  pack_exercise_rows    first-fit-decreasing packing of exercises into rows with cycle semantics, an exposure gate and raw-source residual fill
  lora_rows             alternating synthetic / replay rows for the LoRA transfer recipe
The numerics are those of the runs of record (same seeds, same tie rules); only paths and names changed.
"""
from __future__ import annotations

import bisect
import collections
import random
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

SEQ = 2048
RESIDUAL_CAP = 307  # 15 % of a row: the largest residual a row may carry before the packer draws from the next cycle


def token_dtype(name: str = "uint16"):
    """uint16 is the dtype of record (vocabularies <= 65,535); uint32 for the 128k-vocabulary 3B block."""
    return np.uint32 if name == "uint32" else np.uint16


def eos_id(tok) -> int:
    if tok.eos_token_id is not None:
        return int(tok.eos_token_id)
    return int(tok.convert_tokens_to_ids("<|endoftext|>"))


def open_rows(path: str, mode: str = "r") -> np.memmap:
    return np.load(path, mmap_mode=mode)


# ----------------------------------------------------------------------------------------------------------------- corpus rows
def write_corpus_rows(tok, docs: Iterable[tuple[str, str, str]], n_rows: int, path: str, eos: int, dtype=np.uint16) -> list[tuple[str, str]]:
    """Pack (source, id, text) documents into n_rows rows of SEQ tokens with EOS after every document; returns the (source, id) list consumed."""
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n_rows, SEQ))
    buf: list[int] = []
    row = 0
    ids: list[tuple[str, str]] = []
    for src, did, text in docs:
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [eos])
        ids.append((src, did))
        while len(buf) >= SEQ and row < n_rows:
            mm[row] = np.array(buf[:SEQ], dtype=dtype)
            buf = buf[SEQ:]
            row += 1
        if row >= n_rows:
            break
    mm.flush()
    return ids


def write_synthetic_rows(tok, seed: int, rows: list[int], mode: str, path: str, eos: int, ver: int = 1, dtype=np.uint16) -> list[tuple[int, int, int]]:
    """One row of synthetic definition-use documents per replacement row; row r draws from Random(seed * 10**6 + r) so both modes share their draws."""
    from icl_copy_reason.data_construction.synthetic_docs import make_doc

    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(len(rows), SEQ))
    meta = []
    for k, r in enumerate(rows):
        rng = random.Random(seed * 10**6 + int(r))
        buf: list[int] = []
        n_docs = 0
        n_q = 0
        while len(buf) < SEQ:
            doc, m = make_doc(rng, mode, ver=ver)
            buf.extend(tok(doc, add_special_tokens=False)["input_ids"] + [eos])
            n_docs += 1
            n_q += m["n_queries"]
        mm[k] = np.array(buf[:SEQ], dtype=dtype)
        meta.append((int(r), n_docs, n_q))
    mm.flush()
    return meta


def validation_tokens(tok, n_tokens: int, which: str, eos: int, dtype=np.uint16, fineweb_config: str = "CC-MAIN-2025-26") -> np.ndarray:
    """The fixed held-out token files: FineWeb-Edu (one crawl) and the Pile validation split, stream order until n_tokens."""
    from datasets import load_dataset

    if which == "fwe":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", fineweb_config, split="train", streaming=True)
    else:
        ds = load_dataset("json", data_files="hf://datasets/monology/pile-uncopyrighted/val.jsonl.zst", split="train", streaming=True)
    buf: list[int] = []
    for r in ds:
        buf.extend(tok(r["text"], add_special_tokens=False)["input_ids"] + [eos])
        if len(buf) >= n_tokens:
            break
    return np.array(buf[:n_tokens], dtype=dtype)


def rows_from_text_stream(tok, docs: Iterable[str], n_tokens: int, eos: int) -> Iterator[tuple[np.ndarray, int]]:
    """Yield SEQ-token rows (uint32) from documents in order until n_tokens tokens were emitted; the last partial row is dropped."""
    buf: list[int] = []
    emitted = 0
    used = 0
    for text in docs:
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [eos])
        used += 1
        while len(buf) >= SEQ and emitted < n_tokens:
            yield np.array(buf[:SEQ], dtype=np.uint32), used
            buf = buf[SEQ:]
            emitted += SEQ
        if emitted >= n_tokens:
            return


# ----------------------------------------------------------------------------------------------------------- exercise rows
class _Cycle:
    """One pass over the shuffled exercise order: exercises sorted by length for largest-fit draws, with a consumption pointer."""

    def __init__(self, order: list[int], lengths: list[int]):
        self.order, self.lengths = order, lengths
        n = len(order)
        keys = sorted(range(n), key=lambda pos: (lengths[order[pos]], pos))
        self.ls = [lengths[order[k]] for k in keys]
        self.ps = keys
        self.consumed: set[int] = set()
        self.ptr = 0

    def take_pos(self, pos: int) -> None:
        i = bisect.bisect_left(self.ls, self.lengths[self.order[pos]])
        while self.ps[i] != pos:
            i += 1
        del self.ls[i]
        del self.ps[i]
        self.consumed.add(pos)

    def largest_fit(self, cap: int) -> int | None:
        i = bisect.bisect_right(self.ls, cap) - 1
        return self.ps[i] if i >= 0 else None

    def exhausted(self) -> bool:
        return len(self.consumed) >= len(self.order)

    def next_first(self) -> int | None:
        n = len(self.order)
        while self.ptr < n and self.ptr in self.consumed:
            self.ptr += 1
        return self.ptr if self.ptr < n else None


def pack_exercise_rows(items: list[list[int]], raws: list[list[int]], target_rows: int, eos: int, once_only: bool = False,
                       max_exposure: float | None = 3.0, seed: int = 10**6 + 13, residual_cap: int = RESIDUAL_CAP, dtype=np.uint16) -> tuple[list[np.ndarray], list[dict], dict]:
    """Pack tokenised exercises into rows.

    Random(seed) shuffle; the row's first exercise is the next unconsumed one of the current cycle, the rest of the row takes the largest remaining
    exercise of the current cycle that fits; when nothing fits and the residual would exceed residual_cap, the largest fitting exercise of the NEXT cycle
    is drawn (that cycle keeps its own state). once_only: a single pass, no cycling. Exposure gate: stop before a row would push the mean exposure
    above max_exposure (the caller reverts the surplus rows to replay). Residual is filled with the tokenised prefixes of the raw source documents of
    the row's exercises, EOS-separated, EOS-padded when they run out.
    """
    n = len(items)
    assert n > 0
    order = list(range(n))
    random.Random(seed).shuffle(order)
    lengths = [len(x) + 1 for x in items]
    minl = min(lengths)
    cycles = [_Cycle(order, lengths)]
    c = 0
    exposures: collections.Counter = collections.Counter()
    rows: list[np.ndarray] = []
    meta: list[dict] = []
    stats: collections.Counter = collections.Counter()
    gate_stopped = False
    while len(rows) < target_rows:
        if once_only and cycles[0].exhausted():
            break
        while cycles[c].exhausted():
            c += 1
            if c == len(cycles):
                cycles.append(_Cycle(order, lengths))
        first = cycles[c].next_first()
        cycles[c].take_pos(first)
        e = order[first]
        buf = items[e] + [eos]
        cap = SEQ - lengths[e]
        row_items = [e]
        while cap >= minl:
            src = cycles[c]
            pos = src.largest_fit(cap)
            if pos is None and cap > residual_cap and not once_only:
                if c + 1 == len(cycles):
                    cycles.append(_Cycle(order, lengths))
                src = cycles[c + 1]
                pos = src.largest_fit(cap)
            if pos is None:
                break
            src.take_pos(pos)
            e = order[pos]
            buf += items[e] + [eos]
            cap -= lengths[e]
            row_items.append(e)
        if max_exposure is not None and not once_only and (sum(exposures.values()) + len(row_items)) / n > max_exposure:
            gate_stopped = True
            break
        for e in row_items:
            exposures[e] += 1
        ex_tok = sum(len(items[e]) for e in row_items)
        stats["exercise_tokens"] += ex_tok
        stats["instances"] += len(row_items)
        residual = cap
        if cap > 0:
            stats["rows_with_residual"] += 1
            stats["rows_residual_over_cap"] += int(cap > residual_cap)
            fill: list[int] = []
            for e in row_items:
                src_ids = raws[e]
                if len(src_ids) >= cap - len(fill):
                    fill += src_ids[: cap - len(fill)]
                    break
                fill += src_ids + [eos]
                if len(fill) >= cap:
                    fill = fill[:cap]
                    break
            if len(fill) < cap:
                stats["eos_filler_tokens"] += cap - len(fill)
                fill += [eos] * (cap - len(fill))
            stats["residual_tokens"] += cap
            buf += fill
        assert len(buf) == SEQ, len(buf)
        rows.append(np.asarray(buf, dtype=dtype))
        meta.append({"n_ex": len(row_items), "exposure_idx": [exposures[e] for e in row_items], "residual": residual, "exercise_tokens": ex_tok})
    hist = collections.Counter(exposures[e] for e in range(n))
    hist[0] = n - len([e for e in exposures if exposures[e] > 0])
    info = {"unique": n, "rows_target": target_rows, "rows": len(rows), "gate_stopped": gate_stopped, "once_only": once_only, "instances": stats["instances"],
            "mean_exposure": stats["instances"] / n, "max_exposure": max(exposures.values()) if exposures else 0,
            "exposure_histogram": {str(k): v for k, v in sorted(hist.items())}, "cycles_started": len(cycles), "exercise_tokens": stats["exercise_tokens"],
            "residual_tokens": stats["residual_tokens"], "rows_with_residual": stats["rows_with_residual"], "rows_residual_over_cap": stats["rows_residual_over_cap"],
            "eos_filler_tokens": stats["eos_filler_tokens"]}
    return rows, meta, info


def equal_shares(total_rows: int, types: Iterable[str]) -> dict[str, int]:
    types = sorted(types)
    base, extra = divmod(total_rows, len(types))
    return {t: base + (1 if i < extra else 0) for i, t in enumerate(types)}


def build_exercise_pack(tok, records: list[dict], source_text: dict[str, str], positions: list[int], render: Callable[[dict], str],
                        once_only_types: Iterable[str] = ("a",), max_exposure: float = 3.0, pack_seed: int = 10**6 + 13, interleave_seed: int = 13,
                        seed_offset: int = 0, revert_row: Callable[[int], np.ndarray] | None = None, dtype=np.uint16,
                        residual_cap: int = RESIDUAL_CAP) -> tuple[np.ndarray, list[dict], dict[str, Any]]:
    """Pack verified exercise records into the slot rows of an index space.

    records: verified exercises (dict with 'type', 'exercise', 'doc_sha256'); source_text: sha256 -> raw document text (residual fill).
    positions: the slot rows of the index space (len = rows to fill). Types in once_only_types are packed once (no cycling) and take their rows
    first; the other types split the remaining rows equally. Rows a type cannot fill under the exposure gate revert to replay rows
    (revert_row(position) -> SEQ tokens). One Random(interleave_seed + seed_offset) shuffle interleaves the per-type rows over the positions.
    """
    n_total = len(positions)
    by: dict[str, list[dict]] = {}
    for r in records:
        by.setdefault(r["type"], []).append(r)
    for t in by:
        by[t].sort(key=lambda r: (r.get("pool", ""), r.get("file", 0), r.get("row", 0), r["doc_sha256"]))
    ids = {t: [tok(render(r["exercise"]), add_special_tokens=False)["input_ids"][: SEQ - 1] for r in rs] for t, rs in by.items()}
    raws = {t: [tok(source_text.get(r["doc_sha256"], "")[: 4 * SEQ * 4], add_special_tokens=False)["input_ids"][:SEQ] for r in rs] for t, rs in by.items()}
    seedkw = {"seed": pack_seed + seed_offset}
    packed: dict[str, list[np.ndarray]] = {}
    meta: dict[str, list[dict]] = {}
    info: dict[str, dict] = {}
    once = [t for t in sorted(by) if t in set(once_only_types)]
    other = [t for t in sorted(by) if t not in set(once_only_types)]
    assert other, "no cycled type retained"
    for t in once:
        packed[t], meta[t], info[t] = pack_exercise_rows(ids[t], raws[t], n_total, eos_id(tok), once_only=True, residual_cap=residual_cap, dtype=dtype, **seedkw)
    n_once = sum(len(packed[t]) for t in once)
    share = equal_shares(n_total - n_once, other)
    for t in other:
        packed[t], meta[t], info[t] = pack_exercise_rows(ids[t], raws[t], share[t], eos_id(tok), max_exposure=max_exposure, residual_cap=residual_cap, dtype=dtype, **seedkw)
    unique_tokens = {t: sum(len(x) for x in ids[t]) for t in ids}
    gate = {}
    for t in sorted(by):
        rows_share = len(packed[t]) if t in once else share[t]
        need = rows_share * SEQ / max_exposure
        gate[t] = {"unique_exercises": len(ids[t]), "unique_tokens": unique_tokens[t], "rows_share": rows_share, "needed_for_gate": int(need),
                   "pass": bool(unique_tokens[t] >= need or t in once), "rows_realised": info[t]["rows"], "mean_exposure": round(info[t]["mean_exposure"], 3)}
    order = [(t, i) for t in sorted(packed) for i in range(len(packed[t]))]
    n_revert = n_total - len(order)
    assert n_revert >= 0
    order += [("replay", k) for k in range(n_revert)]
    random.Random(interleave_seed + seed_offset).shuffle(order)
    if n_revert and revert_row is None:
        raise ValueError("rows revert to replay but no revert_row function was given")
    arr = np.zeros((n_total, SEQ), dtype=dtype)
    layout = []
    for k, (pos, (t, i)) in enumerate(zip(positions, order)):
        if t == "replay":
            arr[k] = revert_row(pos)
            layout.append({"row": pos, "type": "replay"})
        else:
            arr[k] = packed[t][i]
            layout.append({"row": pos, "type": t, **meta[t][i]})
    summary = {"rows_total": n_total, "once_only_rows": n_once, "row_shares": share, "rows_reverted_to_replay": n_revert, "yield_gate": gate,
               "per_type": {t: {**info[t], "unique_tokens": unique_tokens[t]} for t in info},
               "exercise_tokens": sum(info[t]["exercise_tokens"] for t in info), "pack_seed": pack_seed, "interleave_seed": interleave_seed, "seed_offset": seed_offset,
               "max_mean_exposure": max_exposure, "residual_cap": residual_cap}
    return arr, layout, summary


# -------------------------------------------------------------------------------------------------------------- LoRA rows
def lora_rows(tok, syn_docs: Iterable[str], replay_docs: Iterable[str], syn_tokens: int, replay_tokens: int, global_batch: int = 32,
              trim_to_batch: bool = True) -> tuple[np.ndarray, dict]:
    """Rows alternate synthetic (even) and replay (odd) until each token budget is met; a stream that runs dry before its budget is a hard failure."""
    eos = eos_id(tok)
    syn = list(rows_from_text_stream(tok, syn_docs, syn_tokens, eos))
    rep = list(rows_from_text_stream(tok, replay_docs, replay_tokens, eos))
    assert len(syn) * SEQ >= syn_tokens - SEQ + 1 and len(rep) * SEQ >= replay_tokens - SEQ + 1, f"stream exhausted: syn {len(syn) * SEQ}/{syn_tokens}, replay {len(rep) * SEQ}/{replay_tokens}"
    n = min(len(syn), len(rep))
    rows = np.stack([x[0] for pair in zip(syn[:n], rep[:n]) for x in pair])
    n_rows = (len(rows) // global_batch) * global_batch if trim_to_batch else len(rows)
    rows = rows[:n_rows]
    info = {"rows": int(n_rows), "steps": int(n_rows // global_batch), "syn_rows": int(n_rows // 2), "syn_docs_used": int(syn[n - 1][1]) if syn else 0,
            "replay_docs_used": int(rep[n - 1][1]) if rep else 0, "tokens": int(n_rows * SEQ)}
    return rows, info

"""Batching check: does co-batching the four candidate forms of an item (one padded forward over 4 x bs rows, the layout of the
attention ablation) change the summed log-probabilities relative to scoring each candidate as a single unpadded sequence?

Inputs : a model, item count, batch size, dtypes. Outputs: max / mean |difference| per form and per margin, arg-max agreement
(schema of the released batching_check.json). Runs on CPU by default.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..io import write_json
from ..tasks.prompts import build, four_forms, slots_of
from .ablation import line_sets, scored_logprobs
from .copy_share import load_model


def four_form_logprobs(lm, items, bs: int) -> np.ndarray:
    """Summed log P(form) for the four forms of each item; bs = items per forward (4*bs candidate rows); bs=0 -> one row per forward."""
    out = []
    if bs == 0:
        for it in items:
            for _bundle, form in four_forms(it):
                out += scored_logprobs(lm, [it.prompt], [form], [None], [line_sets(it)[1]])
        return np.array(out).reshape(len(items), 4)
    for b0 in range(0, len(items), bs):
        bp = items[b0:b0 + bs]
        prompts, conts, spans, qs = [], [], [], []
        for it in bp:
            for _bundle, form in four_forms(it):
                prompts.append(it.prompt)
                conts.append(form)
                spans.append(None)
                qs.append(line_sets(it)[1])
        out += scored_logprobs(lm, prompts, conts, spans, qs)
    return np.array(out).reshape(len(items), 4)


def _margins(M, items):
    out = []
    for j, it in enumerate(items):
        bundles = [b for b, _ in four_forms(it)]
        comp = (it.ia, 1 - it.ib) if slots_of(it.lang, it.ia, it.ib)[0][0] == "a" else (1 - it.ia, it.ib)
        out.append(M[j, bundles.index((it.ia, it.ib))] - M[j, bundles.index(comp)])
    return np.array(out)


def run(model: str = "EleutherAI/pythia-1.4b", steps: str = "143000", n: int = 50, seed: int = 2, bs: int = 8, dtypes: str = "float32,bfloat16",
        out_path: str | Path = "batching_check.json", device: str = "cpu") -> Path:
    rev = None if steps in ("main", "") else f"step{steps}"
    items = build("compose_h", n, seed=seed + 11)
    res = {"model": model, "revision": rev or "main", "n_items": len(items), "item_seed_label": seed, "generator_seed": seed + 11, "batch_items": bs, "device": device, "per_dtype": {}}
    for dt in dtypes.split(","):
        t0 = time.time()
        lm = load_model(model, revision=rev, dtype=dt, device=device)
        lm.model.set_attn_implementation("eager")
        co = four_form_logprobs(lm, items, bs)
        single = four_form_logprobs(lm, items, 0)
        d = np.abs(co - single)
        dm = np.abs(_margins(co, items) - _margins(single, items))
        res["per_dtype"][dt] = {"max_abs_diff_logprob": float(d.max()), "mean_abs_diff_logprob": float(d.mean()), "max_abs_diff_margin": float(dm.max()),
                                "mean_abs_diff_margin": float(dm.mean()), "argmax_agreement": float(np.mean(co.argmax(1) == single.argmax(1))), "seconds": round(time.time() - t0)}
        print(dt, res["per_dtype"][dt], flush=True)
        lm = None
    return write_json(res, out_path)

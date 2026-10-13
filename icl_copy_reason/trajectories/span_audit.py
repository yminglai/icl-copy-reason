"""Tokenizer span audit: do the ablation spans of the competitor marker cover exactly the marker's characters?

Inputs : a model id (tokenizer only, CPU), item seeds and item count; the items are rebuilt exactly as the ablation builds them.
Outputs: per model {vocab, fast, frac_hyphen, frac_stem, n_aligned, per_seed: {seed: [aligned flags]}}; `aligned` = no letter
         outside the marker is masked by a token that overlaps the span (the aligned-only sensitivity analysis uses these items).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def audit_model(model_id: str, seeds: Sequence[int] = (2, 3), n: int = 160, seed_offset: int = 11, tokenizer=None) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from ..readings.ablation import line_sets
    from ..tasks.prompts import build, four_forms

    tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
    res: dict[str, Any] = {"vocab": len(tok), "fast": bool(getattr(tok, "is_fast", False)), "per_seed": {}}
    hy = st = tot = 0
    for seed in seeds:
        items = build("compose_h", n, seed=seed + seed_offset)
        flags = []
        for it in items:
            spans, _ = line_sets(it)
            correct = next(form for bundle, form in four_forms(it) if bundle == (it.ia, it.ib))
            text = it.prompt + " " + correct
            offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            item_hy = item_st = False
            for s, e in spans["competitor_marker"]:
                for ts, te in offs:
                    if te > s and ts < e and te > ts:
                        outside = text[ts:s] + text[e:te]
                        if "-" in outside:
                            item_hy = True
                        if any(ch.isalpha() for ch in outside):
                            item_st = True
            flags.append(not item_st)
            hy += item_hy
            st += item_st
            tot += 1
        res["per_seed"][str(seed)] = flags
    res["frac_hyphen"] = round(hy / tot, 3)
    res["frac_stem"] = round(st / tot, 3)
    res["n_aligned"] = int(sum(sum(v) for v in res["per_seed"].values()))
    return res


def audit_models(model_ids: Sequence[str], seeds: Sequence[int] = (2, 3), n: int = 160) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in model_ids:
        try:
            out[m] = audit_model(m, seeds, n)
        except Exception as e:  # noqa: BLE001 - one tokenizer failing must not stop the audit
            out[m] = {"error": repr(e)[:200]}
    return out

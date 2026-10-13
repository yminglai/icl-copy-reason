"""Demonstration-line deletion: which demonstration lines does the model use to fill the second slot?

For each withheld-combination item, variants of the SAME prompt with some demonstration lines removed (stems and endings
otherwise identical):
  full                          all demonstration lines
  drop_sham                     remove two single-feature lines of the first feature's other value (length-matched control)
  drop_single2                  remove the single-feature lines whose ending is the correct SECOND ending (composition evidence)
  drop_competitor               remove the shown-combination lines that share the first ending (the copied candidate's source)
  drop_other_bundles            remove the two shown combinations that do NOT share the first ending
  drop_all_bundles              single-feature lines only
  drop_single2_and_competitor   both removals (factorial cell for the interaction estimate)
Scores per variant: four-form likelihood accuracy / copied-candidate rate / margin (with per-item records) and generation exact,
second-ending and copy rates.

Inputs : an LM and items. Outputs: the schema of the released `evidence_<model>_s<seed>_fact3.json` files, via `run`
(`drop_sham` omitted = the older `_fact2` files).
"""
from __future__ import annotations

import gc
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..io import read_json, write_json
from ..tasks.prompts import Lang, build, score, slots_of
from .copy_share import four_way, generate, load_model

VARIANTS = ["full", "drop_sham", "drop_single2", "drop_competitor", "drop_other_bundles", "drop_all_bundles", "drop_single2_and_competitor"]


def variants(it):
    lang = Lang(**it.lang)
    lines = it.prompt.split("\n")
    header, demos, query = lines[0], lines[1:-1], lines[-1]
    sl = slots_of(it.lang, it.ia, it.ib)
    first_f, first_aff = sl[0]
    second_f, second_aff = sl[1]
    if first_f == "a":
        competitor = (it.ia, 1 - it.ib)
    else:
        competitor = (1 - it.ia, it.ib)
    comp_form_suffix = "-".join(a for _, a in lang.slots(*competitor))

    def split_line(line):
        return line.split(" = ") if " = " in line else line.split(" -> ")

    def is_single_second(line):          # "<word> <stem> = <stem>-<aff>" with exactly one affix == second_aff
        lhs, rhs = split_line(line)
        return len(lhs.split(" ")) == 2 and rhs.count("-") == 1 and rhs.endswith("-" + second_aff)

    def is_bundle(line):
        lhs, rhs = split_line(line)
        return len(lhs.split(" ")) == 3

    def is_competitor(line):
        return is_bundle(line) and split_line(line)[1].endswith("-" + comp_form_suffix)

    # sham deletion: two single-feature lines whose suffix belongs to the FIRST realised feature's OTHER value (not the competitor,
    # not the composition evidence) — matches the competitor deletion in line count and roughly in length
    first_other = (lang.affA[1 - it.ia] if first_f == "a" else lang.affB[1 - it.ib])

    def is_sham(line):
        lhs, rhs = line.split(" = ") if " = " in line else line.split(" -> ")
        return len(lhs.split(" ")) == 2 and rhs.count("-") == 1 and rhs.endswith("-" + first_other)

    out = {"full": demos,
           "drop_sham": [d for d in demos if not is_sham(d)],
           "drop_single2": [d for d in demos if not is_single_second(d)],
           "drop_competitor": [d for d in demos if not is_competitor(d)],
           "drop_other_bundles": [d for d in demos if not (is_bundle(d) and not is_competitor(d))],
           "drop_all_bundles": [d for d in demos if not is_bundle(d)],
           "drop_single2_and_competitor": [d for d in demos if not is_single_second(d) and not is_competitor(d)]}
    return {k: "\n".join([header] + v + [query]) for k, v in out.items()}


def delete_revision(lm, items, bs: int = 16, variant_names: list[str] | None = None) -> dict:
    rec = {}
    for vname in variant_names or VARIANTS:
        vitems = [replace(it, prompt=variants(it)[vname]) for it in items]
        acc, comp, margin = four_way(lm, vitems, bs)
        sc = []
        for b in range(0, len(vitems), bs):
            bp = vitems[b:b + bs]
            gens = generate(lm, [it.prompt for it in bp])
            for it, g in zip(bp, gens):
                s = score(it, g)
                second = slots_of(it.lang, it.ia, it.ib)[-1][0]
                sc.append((s["exact"], s["aff_" + second], s["copy"]))
        rec[vname] = {"four_way_acc": acc, "four_way_competitor": comp, "four_way_margin": margin, "items": four_way.last_items,
                      "gen_exact": round(float(np.mean([x[0] for x in sc])), 3), "gen_second": round(float(np.mean([x[1] for x in sc])), 3),
                      "gen_copy": round(float(np.mean([x[2] for x in sc])), 3), "n": len(vitems)}
    rec["fingerprint"] = lm.fingerprint()
    return rec


def run(model: str, steps: str = "main", n: int = 160, seed: int = 0, bs: int = 16, tag: str = "_fact3", out_dir: str | Path = ".",
        device: str | None = None) -> Path:
    """Deletion variants for `model` at every revision in `steps` (items = build("compose_h", n, seed + 11)); resumable.
    Returns <out_dir>/evidence_<model>_s<seed><tag>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mtag = model.split("/")[-1]
    outp = out_dir / f"evidence_{mtag}_s{seed}{tag}.json"
    res = read_json(outp) if outp.exists() else {}
    for step in steps.split(","):
        rev = None if step in ("main", "") else (step if step.startswith("stage") else f"step{step}")
        key = rev or "main"
        if key in res:
            continue
        t0 = time.time()
        lm = load_model(model, revision=rev, device=device)
        items = build("compose_h", n, seed=seed + 11)
        rec = delete_revision(lm, items, bs)
        res[key] = rec
        print(f"{mtag} {key}: " + " | ".join(f"{k}: 4way {v['four_way_acc']:.2f}/copied {v['four_way_competitor']:.2f}/m {v['four_way_margin']:+.2f}"
                                             for k, v in rec.items() if k != "fingerprint") + f" [{time.time() - t0:.0f}s]", flush=True)
        write_json(res, outp)
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outp

#!/usr/bin/env python3
"""Score a Hugging Face causal LM with the readings of the paper.

usage: python scripts/score_model.py <reading> --model <repo or local dir> [--revision main|step143000|<branch>] [--seed 0] [--n 160]
                                      [--experiment finals] [--tag ""] [--fp32] [--config <name>] [--set key=value ...]
readings:
  copy-share   generation, four-form likelihood, copy-probability share (triage_<model>_s<seed><tag>.json)
  ablation     attention ablation of demonstration spans (knockout_<model>_s<seed><tag>.json)   [--sets a,b,c]
  deletion     demonstration-line deletion (evidence_<model>_s<seed><tag>.json)
  patching     word -> slot residual patch / DAS (causal_<model>_s<seed><tag>.json)            [--position word --layers 4 --no-das]
  layer-sweep  the word patch at every layer (patch_sweep_<model>_s<seed>_lsweep.json)
  patch-map    layer x position patch map (patch_<model>_s<seed>.json)
  variants     three scoring rules, optional float32 (variants_<model>_s<seed>_rescored[_fp32].json)
  sham-variants  matched-sham and value mean-ablation controls (knockout_variants_<model>_s<seed>_shamfix.json)
  attribution  direct logit attribution per head (dla_<model>[_<rev>]_s<seed>.json)
  head-ablation  cross-fitted R5/B5 head ablation (ablate_...json; needs the other seed's attribution file)
  cumulative   cumulative ablation curves + OV check (mech_cumulative_<model>_sel<s>_eval<t>.json)  [--other-seed]
  steer        inference-time head steering (steer_...json)                                     [--n-random 19]
  attention    attention signatures at the answer slot (attn_<model>[_<rev>]_s<seed><tag>.json)  [--seeds 2,3]
  code-scheme  the code-scheme strict/prefix generation rates (codestrict_<model>_s<seed>.json)
  quechua-gate / quechua  the real-language check (gate_/triage_/knockout_<model>_s<seed>_real[h].json)  [--hyphen]
  batching     batching check on CPU (batching_check.json)
Outputs go to <ICR_RESULTS_ROOT>/<experiment>/ (default experiment per reading). Revisions: 'main', a Pythia step number, or a branch name.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl_copy_reason.config import add_config_args, config_from_args, results_root  # noqa: E402

DEFAULT_EXPERIMENT = {"copy-share": "finals", "ablation": "finals", "deletion": "deletion", "patching": "patching", "layer-sweep": "patching", "patch-map": "patching",
                      "variants": "finals", "sham-variants": "heads-steer", "attribution": "heads-steer", "head-ablation": "heads-steer", "cumulative": "heads-steer",
                      "steer": "heads-steer", "attention": "heads-attention", "code-scheme": "code-scheme", "quechua-gate": "quechua", "quechua": "quechua", "batching": "finals"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reading", choices=sorted(DEFAULT_EXPERIMENT))
    add_config_args(ap)
    ap.add_argument("--model", required=False, help="Hugging Face repo id or local checkpoint directory")
    ap.add_argument("--revision", default="main", help="main, a Pythia step number, or a branch / tag; comma-separated for several")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--other-seed", type=int, default=None, help="cross-fitting seed (head-ablation, cumulative, steer)")
    ap.add_argument("--seeds", default=None, help="attention: comma-separated item seeds scored with one model load")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--experiment", default=None, help="results sub-directory (default per reading)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--scheme", default="morph", choices=["morph", "code"])
    ap.add_argument("--template", default="default", choices=["default", "alt"])
    ap.add_argument("--order", default="ab", choices=["ab", "ba"])
    ap.add_argument("--n-single", type=int, default=2)
    ap.add_argument("--n-combo", type=int, default=2)
    ap.add_argument("--sets", default=None, help="ablation: comma-separated span sets")
    ap.add_argument("--position", default="word", choices=["eq", "stem", "word"])
    ap.add_argument("--layers", default="")
    ap.add_argument("--ranks", default="8")
    ap.add_argument("--tasks", default="compose_s")
    ap.add_argument("--no-das", action="store_true")
    ap.add_argument("--skip-heads", action="store_true")
    ap.add_argument("--n-random", type=int, default=19)
    ap.add_argument("--hyphen", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    exp = a.experiment or DEFAULT_EXPERIMENT[a.reading]
    out_dir = results_root() / exp
    if a.reading != "batching" and not a.model:
        ap.error("--model is required")
    tag = a.tag
    bs = a.bs
    dev = a.device
    if a.reading == "copy-share":
        from icl_copy_reason.readings import copy_share
        p = copy_share.run(a.model, a.revision, a.n, a.seed, bs or 16, n_single=a.n_single, n_combo=a.n_combo, order=a.order, template=a.template, scheme=a.scheme,
                           tag=tag if tag is not None else "", out_dir=out_dir, dtype="float32" if a.fp32 else "bfloat16", device=dev)
    elif a.reading == "ablation":
        from icl_copy_reason.readings import ablation
        p = ablation.run(a.model, a.revision, a.n, a.seed, bs or 8, a.scheme, a.template, a.sets, tag if tag is not None else "", a.fp32, out_dir, dev)
    elif a.reading == "deletion":
        from icl_copy_reason.readings import deletion
        p = deletion.run(a.model, a.revision, a.n, a.seed, bs or 16, tag if tag is not None else "_fact3", out_dir, dev)
    elif a.reading == "patching":
        from icl_copy_reason.readings import patching
        p = patching.run(a.model, a.revision, a.tasks, a.layers, a.position, a.ranks, bs=bs or 16, seed=a.seed, tag=tag if tag is not None else "",
                         no_das=a.no_das, skip_heads=a.skip_heads, n_test=a.n, out_dir=out_dir, device=dev)
    elif a.reading == "layer-sweep":
        from icl_copy_reason.readings import patching
        p = patching.layer_sweep(a.model, a.seed, a.revision, a.n, bs or 16, a.layers, tag=tag if tag is not None else "_lsweep", out_dir=out_dir, fp32=a.fp32, device=dev)
    elif a.reading == "patch-map":
        from icl_copy_reason.readings import patching
        p = patching.patch_map(a.model, a.seed, a.n, bs or 8, tag if tag is not None else "", None if a.revision == "main" else a.revision, out_dir, dev)
    elif a.reading == "variants":
        from icl_copy_reason.readings import variants
        p = variants.rescore(a.model, a.revision, a.n, a.seed, tag if tag is not None else ("_rescored_fp32" if a.fp32 else "_rescored"), bs or 8, a.fp32, out_dir=out_dir, device=dev)
    elif a.reading == "sham-variants":
        from icl_copy_reason.readings import ablation
        p = ablation.sham_variants(a.model, a.seed, a.revision, a.n, bs or 8, tag=tag if tag is not None else "_shamfix", fp32=a.fp32, out_dir=out_dir, device=dev)
    elif a.reading == "attribution":
        from icl_copy_reason.readings import heads
        p = heads.attribution(a.model, a.revision, a.seed, a.n, bs or 8, a.scheme, tag if tag is not None else "", a.fp32, out_dir=out_dir, device=dev)
    elif a.reading == "head-ablation":
        from icl_copy_reason.readings import heads
        p = heads.ablate_sets(a.model, a.revision, a.seed, a.other_seed, a.n, bs or 8, tag if tag is not None else "", out_dir, device=dev)
    elif a.reading == "cumulative":
        from icl_copy_reason.readings import heads
        other = a.other_seed if a.other_seed is not None else {2: 3, 3: 2}.get(a.seed, a.seed + 1)
        p = heads.cumulative(a.model, a.seed, other, a.revision, a.n, bs or 8, tag=tag if tag is not None else "", fp32=a.fp32, out_dir=out_dir, device=dev)
    elif a.reading == "steer":
        from icl_copy_reason.readings import heads
        p = heads.steer(a.model, a.revision, a.seed, a.other_seed, a.n, bs or 8, a.n_random, tag if tag is not None else "_steer", a.fp32, out_dir, device=dev)
    elif a.reading == "attention":
        from icl_copy_reason.readings import heads
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
        p = heads.attention_signatures(a.model, seeds, None if a.revision == "main" else a.revision, n=a.n, bs=bs or 8, scheme=a.scheme,
                                       tag=tag if tag is not None else "", out_dir=out_dir, device=dev)
    elif a.reading == "code-scheme":
        from icl_copy_reason.readings import code_scheme
        p = code_scheme.run(a.model, a.revision, a.seed, a.n, bs or 16, out_dir, dev)
    elif a.reading == "quechua-gate":
        from icl_copy_reason.tasks import quechua
        p = quechua.gate(a.model, a.revision, a.n, a.seed, bs or 16, a.hyphen, out_dir=out_dir, device=dev)
    elif a.reading == "quechua":
        from icl_copy_reason.tasks import quechua
        p = quechua.measure(a.model, a.revision, a.n, a.seed, bs or 16, hyphen=a.hyphen, out_dir=out_dir, device=dev)
    elif a.reading == "batching":
        from icl_copy_reason.readings import batching
        p = batching.run(a.model or "EleutherAI/pythia-1.4b", a.revision if a.revision != "main" else "143000", a.n, a.seed, bs or 8,
                         out_path=out_dir / "batching_check.json", device=dev or "cpu")
    else:  # pragma: no cover
        raise SystemExit(a.reading)
    _ = cfg
    print("output:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

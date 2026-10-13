#!/usr/bin/env python
"""Real documents -> definition-use exercises (sample / generate / verify / retry / audit), the first set or the topped-up set.

usage:
  python scripts/build_exercises.py windows  --config exercises-data            # evaluation-window store (check 4), needs network
  python scripts/build_exercises.py sample   --config exercises-data [--pilot 40]
  python scripts/build_exercises.py generate --config exercises-data            # ICR_API_BASE / ICR_API_KEY
  python scripts/build_exercises.py verify   --config exercises-data
  python scripts/build_exercises.py retry    --config exercises-data            # one more draw for failed type-h documents
  python scripts/build_exercises.py verify-retry --config exercises-data
  python scripts/build_exercises.py audit    --config exercises-data
  python scripts/build_exercises.py all      --config rewritten-docs-data       # the first set end to end
  python scripts/build_exercises.py check    --records data/exercises/exercises.jsonl.gz --n 200   # offline checks on shipped records
  python scripts/build_exercises.py variant-select --config exercises-data --variant k --n 40 --out <dir>/docs_k.jsonl
  python scripts/build_exercises.py variant-post --config exercises-data --variant k --verified <dir>/verified_k.jsonl --docs <dir>/docs_k.jsonl
Outputs go to <output_dir> of the config (default <ICR_RESULTS_ROOT>/<config name>).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl_copy_reason.config import add_config_args, config_from_args  # noqa: E402
from icl_copy_reason.data_construction import augment_real, exercises, variants  # noqa: E402
from icl_copy_reason.io import read_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["windows", "sample", "generate", "verify", "retry", "verify-retry", "audit", "all", "check", "variant-select", "variant-post"])
    add_config_args(ap)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--pilot", type=int, default=0, help="sample: pilot documents instead of the full target")
    ap.add_argument("--target", type=int, default=None, help="sample: documents to sample (default from the config)")
    ap.add_argument("--no-dataset-windows", action="store_true", help="windows: skip the Hub datasets (offline store of templates / pools / synthetic documents only)")
    ap.add_argument("--records", default=None, help="check: a jsonl / jsonl.gz of verified records")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--variant", choices=["k", "h2"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verified", default=None)
    ap.add_argument("--docs", default=None)
    ap.add_argument("--sorted", action="store_true", help="variant-select: (pool, file, row) order (full run) instead of file order (pilot)")
    a = ap.parse_args(argv)
    cfg = config_from_args(a, default="exercises-data")
    name = cfg.name if cfg else "exercises-data"
    default_prompts = "rewrite_prompts.json" if name == "rewritten-docs-data" else "prompts.json"
    if a.stage == "check":
        P = augment_real.prompt_table(cfg, default_prompts)
        recs = []
        for r in read_jsonl(a.records or Path(__file__).resolve().parents[1] / "data" / "exercises" / "exercises.jsonl.gz"):
            recs.append(r)
            if len(recs) >= a.n:
                break
        out = exercises.run_offline_checks(P, recs)
        n_ok = sum(1 for o in out if not o["mechanical_reasons"])
        print(json.dumps({"n": len(out), "all_mechanical_checks_pass": n_ok, "reasons": sorted({x for o in out for x in o["mechanical_reasons"]})}, indent=1))
        return 0 if n_ok == len(out) else 1
    if a.stage == "variant-select":
        pth = augment_real.paths(cfg, name)
        sel, info = variants.select_documents(a.docs or pth["docs"], a.n, a.out or pth["base"] / f"docs_{a.variant}.jsonl", a.variant, a.sorted)
        print(json.dumps(info, indent=1))
        return 0
    if a.stage == "variant-post":
        print(json.dumps(variants.post_verify(a.verified, a.docs, a.variant), indent=1))
        return 0
    out = augment_real.run_pipeline(cfg, a.stage, name, default_prompts, workers=a.workers, pilot=a.pilot, extend_windows_with_datasets=not a.no_dataset_windows, target=a.target)
    if isinstance(out, dict):
        print(json.dumps({k: v for k, v in out.items() if k not in ("pools", "by_type", "total", "per_type")}, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

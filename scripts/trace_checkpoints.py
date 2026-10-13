#!/usr/bin/env python
"""Trace the two readings over a family's released pretraining checkpoints and summarise the trajectory.

usage:
  python scripts/trace_checkpoints.py --config traj-olmo2-7b                       # trace the grid (all seeds), then summarise
  python scripts/trace_checkpoints.py --config traj-comma-7b --revisions stage1-step010000-tokens21B,stage1-step020000-tokens42B
  python scripts/trace_checkpoints.py --config traj-pythia-2.8b --summarize-only    # summary from existing outputs
  python scripts/trace_checkpoints.py --config traj-smollm3-3b --stage A            # final checkpoint only (survey stage A)
  python scripts/trace_checkpoints.py --list                                        # the listed families
Outputs go to ICR_RESULTS_ROOT/<config>/ (default <repo>/runs/<config>/): knockout_*.json, triage_*.json, trajectory_<family>_s<seeds>.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl_copy_reason.config import add_config_args, config_from_args, results_root  # noqa: E402
from icl_copy_reason.trajectories import crossings, trace  # noqa: E402
from icl_copy_reason.trajectories.registry import families, load_family  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--list", action="store_true", help="list the families in the registry and exit")
    ap.add_argument("--list-finals", action="store_true", help="print '<family> <final model repo>' for every family and exit")
    ap.add_argument("--list-revisions", action="store_true", help="with --config: print the configured grid's revisions, one per line, and exit")
    ap.add_argument("--repo", action="store_true", help="with --config: print the family's checkpoint repo and exit")
    ap.add_argument("--revisions", help="comma-separated revisions instead of the configured grid")
    ap.add_argument("--seeds", help="comma-separated item seeds (default: the config's)")
    ap.add_argument("--stage", choices=["A", "B"], default="B", help="A = final checkpoint only; B = the grid (default)")
    ap.add_argument("--readings", help="comma-separated subset of copy_share,ablation")
    ap.add_argument("--summarize-only", action="store_true", help="skip tracing; summarise existing outputs")
    ap.add_argument("--results-dir", help="where outputs live (default ICR_RESULTS_ROOT/<config>)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    if args.list:
        for f in families():
            F = load_family(f)
            print(f"traj-{f:20s} {F.repo:55s} {len(F.refs()):3d} checkpoints  unit={F.position_unit}  seeds={F.seeds}  grid={F.default_grid}")
        return 0
    if args.list_finals:
        for f in families():
            F = load_family(f)
            print(f, F.final_model)
        return 0
    cfg = config_from_args(args)
    if cfg is None:
        ap.error("--config traj-<family> is required (or --list / --list-finals)")
    if args.repo or args.list_revisions:
        F = trace.family_of(cfg)
        if args.repo:
            print(F.repo)
        else:
            for ref in F.refs():
                print(ref)
        return 0
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    results_dir = Path(args.results_dir) if args.results_dir else results_root() / cfg.name
    if not args.summarize_only:
        F = trace.family_of(cfg)
        revisions = args.revisions.split(",") if args.revisions else None
        if args.stage == "A":
            revisions = [F.final_revision or "main"]
            cfg.set("include_released_model", False)
            cfg.set("repo", F.final_model)
        files = trace.trace_family(cfg, results_dir, revisions=revisions, seeds=seeds, readings=args.readings.split(",") if args.readings else None, device=args.device)
        for k, p in files.items():
            print(f"{k}: {p}")
        if args.stage == "A":
            return 0
    try:
        r = trace.summarize_family(cfg, results_dir, seeds=seeds)
    except crossings.TrajectoryError as e:
        print(f"summary not produced: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in r.items() if k not in ("rows", "D_mean_series", "D_ci_lo", "D_ci_hi")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Trace the two readings over a family's released checkpoints and summarise the trajectory.

Inputs : a traj-<family> config (repo, grid of revisions, seeds, item count, ablation sets, copy-share tasks, dtype) and a
         results directory. The readings are `icl_copy_reason.readings.ablation.run` and `.copy_share.run` (one call per
         revision list; both are resumable per revision and write `knockout_<tag>_s<seed><tag>.json` / `triage_...json`).
Outputs: the per-revision result files, the merged files for separate-repo families, and the trajectory summary
         `trajectory_<family>_s<seeds>.json` (+ .md) in the schema of the released summaries.
"""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, Config, results_root
from ..io import write_json
from . import crossings
from .cohort import SEPARATE_REPO_FAMILIES, merge_separate_repo_series, model_tag
from .registry import Family, load_family, result_key


def family_of(cfg: Config) -> Family:
    return load_family(cfg.get("family") or cfg.name)


def _run_reading(kind: str, model: str, steps: str, n: int, seed: int, tag: str, out_dir: Path, sets: str | None, tasks: str | None,
                 dtype: str, bs: int, device: str | None) -> Path:
    """Call the reading in-process; fall back to the score_model CLI when the readings package is not importable here."""
    try:
        if kind == "ablation":
            from ..readings.ablation import run

            return run(model=model, steps=steps, n=n, seed=seed, bs=bs, sets=sets, tag=tag, fp32=(dtype == "float32"), out_dir=out_dir, device=device)
        from ..readings.copy_share import run as run_cs

        return run_cs(model=model, steps=steps, n=n, seed=seed, bs=max(bs, 16), tasks=tasks, tag=tag, out_dir=out_dir, dtype=dtype, device=device)
    except ImportError:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "score_model.py"), "ablation" if kind == "ablation" else "copy-share", "--model", model,
               "--revisions", steps, "--n", str(n), "--seed", str(seed), "--tag", tag, "--out-dir", str(out_dir)]
        if sets and kind == "ablation":
            cmd += ["--sets", sets]
        if tasks and kind != "ablation":
            cmd += ["--tasks", tasks]
        subprocess.run(cmd, check=True)
        return out_dir / f"{'knockout' if kind == 'ablation' else 'triage'}_{model_tag(model)}_s{seed}{tag}.json"


def entries_by_model(F: Family, grid: str | None, include_released: bool) -> list[tuple[str, str]]:
    """[(model id, comma-separated revisions)] for a grid: same-repo families give one entry; separate-repo families one per checkpoint."""
    out: dict[str, list[str]] = {}
    for e in F.revisions(grid):
        ref = e["ref"]
        if "@" in ref:
            mid, rev = ref.split("@", 1)
        elif "/" in ref:
            mid, rev = ref, "main"
        else:
            mid, rev = F.repo, ref
        out.setdefault(mid, []).append(rev)
    if include_released and F.final_model:
        out.setdefault(F.final_model, []).append(F.final_revision or "main")
    return [(m, ",".join(r)) for m, r in out.items()]


def trace_family(cfg: Config, results_dir: str | Path | None = None, revisions: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
                 readings: Sequence[str] | None = None, device: str | None = None) -> dict[str, Path]:
    """Run the configured readings over the family's grid (or the given revisions) for every seed. Returns the result files."""
    F = family_of(cfg)
    out_dir = Path(results_dir or results_root() / cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = cfg.get("grid") or F.default_grid
    seeds = list(seeds or cfg.get("seeds") or F.seeds)
    readings = list(readings or cfg.get("readings") or ["copy_share", "ablation"])
    n = int(cfg.get("n_items", F.n))
    tag = cfg.get("tag", F.tag)
    sets = ",".join(cfg.get("ablation_sets") or []) or None
    tasks = ",".join(cfg.get("copy_share_tasks") or []) or None
    dtype = cfg.get("dtype", "bfloat16")
    bs = int(cfg.get("batch_size", 8))
    entries = [(F.repo, ",".join(revisions))] if revisions else entries_by_model(F, grid, bool(cfg.get("include_released_model", True)))
    files: dict[str, Path] = {}
    for mid, steps in entries:
        for s in seeds:
            if "ablation" in readings:
                files[f"ablation:{mid}:s{s}"] = _run_reading("ablation", mid, steps, n, s, tag, out_dir, sets, tasks, dtype, bs, device)
            if "copy_share" in readings:
                files[f"copy_share:{mid}:s{s}"] = _run_reading("copy_share", mid, steps, n, s, tag, out_dir, sets, tasks, dtype, bs, device)
    return files


def summarize_family(cfg: Config, results_dir: str | Path | None = None, seeds: Sequence[int] | None = None, descriptive: bool | None = None,
                     out_dir: str | Path | None = None) -> dict[str, Any]:
    """Build the trajectory summary of a family from its result files and write trajectory_<family>_s<seeds>.json (+ .md)."""
    F = family_of(cfg)
    results_dir = Path(results_dir or results_root() / cfg.name)
    out_dir = Path(out_dir or results_dir)
    seeds = list(seeds or cfg.get("seeds") or F.seeds)
    grid = cfg.get("grid") or F.default_grid
    n = int(cfg.get("n_items", F.n))
    tag = cfg.get("tag", F.tag)
    summary_cfg = cfg.get("summary") or {}
    descriptive = summary_cfg.get("descriptive", True) if descriptive is None else descriptive
    if F.family in SEPARATE_REPO_FAMILIES:
        mt, _, table = merge_separate_repo_series(results_dir, F, grid, seeds, tag)
    else:
        mt = model_tag(F.repo)
        table = F.position_table(grid)
    _ = result_key

    def pos(k):
        return table.get(k, F.position_of(k))

    r = crossings.summarize(results_dir, mt, seeds, tag, n, pos, descriptive=descriptive, B=int(summary_cfg.get("bootstrap", crossings.BOOTSTRAP)),
                            onset_B=int(summary_cfg.get("onset_bootstrap", crossings.ONSET_BOOTSTRAP)))
    r.update({"family": F.family, "grid": grid, "position_unit": F.position_unit})
    name = f"trajectory_{F.family}_s{'-'.join(map(str, seeds))}"
    write_json(r, out_dir / f"{name}.json")
    (out_dir / f"{name}.md").write_text(crossings.summary_markdown(r, name))
    return r

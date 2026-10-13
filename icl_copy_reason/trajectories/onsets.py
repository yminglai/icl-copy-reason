"""Fitted onsets: a four-parameter logistic over log position gives the point at which a per-item skill curve crosses a threshold.

Inputs : per-checkpoint arrays of per-item 0/1 outcomes (the same items at every checkpoint) and their log10 positions.
Outputs: the crossing position with an item-bootstrap 95 % interval and the fraction of censored replicates; the skill table
         (stem copying, feature generation, shown-combination exact match, withheld four-way accuracy, withheld exact match).
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..io import read_json

ONSET_BOOTSTRAP = 300


def logistic(x, x0, k, lo, hi):
    return lo + (hi - lo) / (1 + np.exp(-k * (x - x0)))


def fit_onset(x: Sequence[float], per_item: Sequence[Sequence[float]], thr: float = 0.5, B: int = ONSET_BOOTSTRAP, seed: int = 0):
    """x: log10 positions per checkpoint; per_item: one 0/1 array per checkpoint. Fit the logistic to the mean curve and bootstrap
    items for an interval on the x where the fitted curve crosses `thr`. Returns (x_cross, lo, hi, censored_fraction); x_cross is
    +inf when the fitted upper asymptote stays below thr (right-censored), -inf when the lower asymptote is above it, nan when the
    fit fails."""
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=float)

    def cross(y):
        try:
            p, _ = curve_fit(logistic, x, y, p0=[np.median(x), 2.0, max(0, y.min()), min(1, y.max())],
                             bounds=([x.min() - 2, 0.05, 0, 0], [x.max() + 2, 50, 1, 1]), maxfev=20000)
        except Exception:  # noqa: BLE001 - a failed fit is reported as nan
            return np.nan
        lo, hi = p[2], p[3]
        if hi < thr:
            return np.inf
        if lo > thr:
            return -np.inf
        return p[0] + np.log((hi - lo) / (thr - lo) - 1) / (-p[1])

    y = np.array([np.mean(v) for v in per_item])
    xc = cross(y)
    n = min(len(v) for v in per_item)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        boots.append(cross(np.array([np.mean(np.array(v)[idx]) for v in per_item])))
    boots = np.array(boots)
    fin = boots[np.isfinite(boots)]
    cens = float(np.mean(~np.isfinite(boots)))
    lo, hi = (np.percentile(fin, 2.5), np.percentile(fin, 97.5)) if len(fin) > 10 else (np.nan, np.nan)
    return xc, lo, hi, cens


def fit_onset_positions(positions: Sequence[float], per_item_acc: Sequence[Sequence[float]], thr: float = 0.25, B: int = ONSET_BOOTSTRAP) -> dict[str, Any]:
    """fit_onset on log10 positions, reported back in position units: {"x", "ci": [lo, hi], "censored_fraction"} (None when infinite)."""
    xc, lo, hi, cens = fit_onset(np.log10(np.asarray(positions, float)), per_item_acc, thr, B=B)

    def back(v):
        return None if not np.isfinite(v) else float(10**v)

    return {"x": back(xc), "ci": [back(lo), back(hi)], "censored_fraction": float(cens)}


def merge_by_seed(files: Sequence[str | Path], position_of: Callable[[str], float | None]) -> tuple[list[dict], list[str]]:
    """One merged record per item seed (files of the same seed are unioned, first occurrence wins; `_s<d>` in the file name is the
    seed) and the revision keys common to every seed, sorted by position."""
    by_seed: dict[str, dict] = {}
    for f in files:
        f = str(f)
        if not os.path.exists(f):
            continue
        seed = re.search(r"_s(\d)[_.]", os.path.basename(f)).group(1)
        d = by_seed.setdefault(seed, {})
        for k, v in read_json(f).items():
            if k in d and isinstance(v, dict) and isinstance(d[k], dict):
                for kk, vv in v.items():
                    d[k].setdefault(kk, vv)
            else:
                d.setdefault(k, v)
    ds = [by_seed[s] for s in sorted(by_seed)]
    keys = sorted([k for k in ds[0] if position_of(k) is not None and all(k in d for d in ds)], key=position_of)
    return ds, keys


SKILLS: list[tuple[str, str, str, str, float]] = [
    # name, task, item list, metric, threshold
    ("copy (stem)", "compose_s", "items", "copy", 0.85),
    ("feature A generated", "bind_a", "items", "aff_a", 0.70),
    ("feature B generated", "bind_b", "items", "aff_b", 0.70),
    ("shown combination exact", "compose_s", "items", "exact", 0.50),
    ("withheld four-way accuracy", "compose_h", "four_way_items", "acc", 0.25),
    ("withheld exact", "compose_h", "items", "exact", 0.15),
]


def skill_onsets(files: Sequence[str | Path], position_of: Callable[[str], float | None], B: int = ONSET_BOOTSTRAP) -> list[dict[str, Any]]:
    """The fitted-onset table of one run from its copy-share result files (any number of item seeds): one row per skill."""
    ds, keys = merge_by_seed(files, position_of)
    x = [np.log10(position_of(k)) for k in keys]
    rows = []
    for name, task, lst, met, thr in SKILLS:
        per = [np.concatenate([[float(i[met]) for i in d[k][task][lst]] for d in ds]) for k in keys]
        xc, lo, hi, cens = fit_onset(x, per, thr, B=B)
        rows.append({"skill": name, "task": task, "metric": met, "threshold": thr, "onset_log10": _f(xc), "ci_log10": [_f(lo), _f(hi)],
                     "onset": _back(xc), "ci": [_back(lo), _back(hi)], "censored_fraction": float(cens), "n_checkpoints": len(keys)})
    return rows


def _f(v: float) -> float | str | None:
    if v == np.inf:
        return "censored"
    if v == -np.inf:
        return "<range"
    return None if not np.isfinite(v) else float(v)


def _back(v: float) -> float | str | None:
    r = _f(v)
    return float(10**r) if isinstance(r, float) else r

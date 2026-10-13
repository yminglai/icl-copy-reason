"""Paired item bootstrap used by every table and figure of the paper.

Inputs : per-item effects (a vector) or two paired per-item vectors. Outputs: mean and a 95% percentile interval.
Convention of record: numpy default_rng(0) and 2,000 resamples (the appendix table of deletion effects used 3,000 with the same seed,
so its endpoints can differ by up to about 0.03 from a 2,000-resample run).
"""
from __future__ import annotations

import numpy as np


def bootstrap_mean(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lower, upper) of the item bootstrap of the mean."""
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    bs = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def bootstrap_diff(a, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Paired difference a - b per item, then bootstrap_mean."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have the same length")
    return bootstrap_mean(a - b, n_boot, seed, alpha)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)

from __future__ import annotations

import itertools

import numpy as np

from .. import config

def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int | None = None) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if len(v) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    boot_means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi

def paired_permutation_test(a, b, n_perm: int = 10000, seed: int | None = None) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    if np.allclose(d, 0.0):
        return 1.0
    obs = abs(float(d.mean()))

    if n <= 16:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        stats = np.abs(signs @ d) / n
        return float(np.mean(stats >= obs - 1e-12))

    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    signs = rng.choice([1.0, -1.0], size=(n_perm, n))
    stats = np.abs(signs @ d) / n
    return float((np.sum(stats >= obs - 1e-12) + 1) / (n_perm + 1))

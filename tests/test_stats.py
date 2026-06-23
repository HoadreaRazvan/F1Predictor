import numpy as np

from f1pred.evaluation.stats import bootstrap_ci, paired_permutation_test

def test_bootstrap_ci_contains_mean_and_brackets_it():
    v = np.array([0.60, 0.62, 0.64, 0.66, 0.68, 0.70])
    mean, lo, hi = bootstrap_ci(v, n_boot=2000, seed=0)
    assert abs(mean - v.mean()) < 1e-9
    assert lo <= mean <= hi
    assert lo >= v.min() - 1e-9 and hi <= v.max() + 1e-9

def test_bootstrap_ci_narrower_for_lower_variance():
    rng = np.random.default_rng(0)
    tight = np.full(8, 0.7) + rng.normal(scale=0.005, size=8)
    wide = np.full(8, 0.7) + rng.normal(scale=0.08, size=8)
    _, lo_t, hi_t = bootstrap_ci(tight, seed=1)
    _, lo_w, hi_w = bootstrap_ci(wide, seed=1)
    assert (hi_t - lo_t) < (hi_w - lo_w)

def test_permutation_identical_series_p_one():
    a = np.array([0.6, 0.7, 0.65, 0.62, 0.68])
    assert paired_permutation_test(a, a.copy()) == 1.0

def test_permutation_large_consistent_difference_significant():
    a = np.array([0.80, 0.82, 0.79, 0.81, 0.83, 0.80])
    b = a - 0.15
    p = paired_permutation_test(a, b)
    assert p < 0.05

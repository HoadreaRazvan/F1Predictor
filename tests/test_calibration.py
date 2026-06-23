import numpy as np

from f1pred.evaluation import metrics
from f1pred.evaluation.calibration import IsotonicCalibrator, PlattScaler

def _miscalibrated(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.random(n)
    y = (rng.random(n) < s**2).astype(float)
    return s, y

def test_isotonic_is_monotone_and_roughly_preserves_order():
    s, y = _miscalibrated()
    iso = IsotonicCalibrator().fit(s, y)
    grid = np.linspace(0, 1, 50)
    out = iso.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)

    assert abs(metrics.roc_auc(y, iso.transform(s)) - metrics.roc_auc(y, s)) < 0.02

def test_platt_preserves_order_exactly():
    s, y = _miscalibrated()

    assert abs(metrics.roc_auc(y, PlattScaler().fit(s, y).transform(s)) - metrics.roc_auc(y, s)) < 1e-9

def test_calibration_reduces_brier():
    s, y = _miscalibrated()
    brier_raw = metrics.brier_score(y, s)
    brier_platt = metrics.brier_score(y, PlattScaler().fit(s, y).transform(s))
    brier_iso = metrics.brier_score(y, IsotonicCalibrator().fit(s, y).transform(s))
    assert brier_platt < brier_raw
    assert brier_iso < brier_raw

def test_platt_outputs_valid_probabilities():
    s, y = _miscalibrated(n=500, seed=1)
    p = PlattScaler().fit(s, y).transform(s)
    assert p.min() >= 0.0 and p.max() <= 1.0

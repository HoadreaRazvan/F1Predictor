import numpy as np

from f1pred.evaluation import metrics

def test_classification_metrics_known_values():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])

    cm = metrics.confusion_matrix(y_true, y_pred)
    assert (cm["tp"], cm["fp"], cm["fn"], cm["tn"]) == (2, 1, 1, 2)
    assert abs(metrics.accuracy(y_true, y_pred) - 4 / 6) < 1e-9
    assert abs(metrics.precision(y_true, y_pred) - 2 / 3) < 1e-9
    assert abs(metrics.recall(y_true, y_pred) - 2 / 3) < 1e-9
    assert abs(metrics.f1(y_true, y_pred) - 2 / 3) < 1e-9

def test_perfect_prediction():
    y = np.array([0, 1, 1, 0])
    assert metrics.accuracy(y, y) == 1.0
    assert metrics.f1(y, y) == 1.0

def test_probabilistic_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(metrics.roc_auc(y, p) - 1.0) < 1e-9
    assert abs(metrics.pr_auc(y, p) - 1.0) < 1e-9

    assert abs(metrics.roc_auc(y, 1 - p) - 0.0) < 1e-9

    assert abs(metrics.roc_auc(y, np.full(4, 0.5)) - 0.5) < 1e-9

def test_brier_and_logloss():
    y = np.array([1.0, 0.0])

    assert metrics.brier_score(y, np.array([1.0, 0.0])) == 0.0
    assert metrics.log_loss(y, np.array([1.0, 0.0])) < 1e-6

    assert abs(metrics.brier_score(y, np.array([0.5, 0.5])) - 0.25) < 1e-9

def test_ece_perfectly_calibrated_is_zero():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert metrics.expected_calibration_error(y, p, n_bins=10) < 1e-9

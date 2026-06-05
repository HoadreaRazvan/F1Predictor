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


def test_standings_overlap():
    res = metrics.standings_overlap(["VER", "NOR", "LEC"], ["VER", "LEC", "NOR"])
    assert res["champ_top3_overlap"] == 1.0
    assert res["champ_top3_exact"] == 0.0
    res2 = metrics.standings_overlap(["VER", "NOR", "LEC"], ["VER", "NOR", "LEC"])
    assert res2["champ_top3_exact"] == 1.0

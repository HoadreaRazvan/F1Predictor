import numpy as np
import pandas as pd

from f1pred.evaluation import experiments, metrics
from f1pred.features.engineering import add_features, feature_matrix

_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8}


def _toy_feats(seasons=(2020, 2021, 2022), rounds=4, n_drivers=6, seed=0):
    rng = np.random.default_rng(seed)
    drivers = [f"D{i}" for i in range(n_drivers)]
    rows = []
    for season in seasons:
        for rnd in range(1, rounds + 1):
            order = np.argsort(np.arange(n_drivers) + rng.normal(scale=0.5, size=n_drivers))
            for pos, di in enumerate(order, start=1):
                drv = drivers[di]
                rows.append(
                    {
                        "season": season, "round": rnd,
                        "event_name": f"GP{rnd}", "circuit": f"C{rnd}", "country": "X",
                        "event_date": pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=14 * rnd),
                        "is_sprint_weekend": 0.0,
                        "driver": drv, "driver_id": drv,
                        "team": f"T{di // 2}", "team_id": f"T{di // 2}",
                        "grid": float(pos), "quali_pos": float(pos),
                        "position": float(pos), "status": "Finished", "finished": 1.0,
                        "points": float(_POINTS.get(pos, 0)),
                        "rain": 0.0, "track_temp": 30.0, "air_temp": 22.0,
                    }
                )
    return add_features(pd.DataFrame(rows))


def test_sweep_1d_shape_and_ranges():
    feats = _toy_feats()
    df = experiments.sweep_1d(feats, "tree", "max_depth", [2, 4, 6])
    assert len(df) == 3
    assert list(df["value"]) == [2, 4, 6]
    for m in experiments.SWEEP_METRICS:
        assert df[m].between(0.0, 1.0).all()
    assert (df["fit_seconds"] >= 0).all()


def test_grid_2d_shape_and_ranges():
    feats = _toy_feats()
    mat = experiments.grid_2d(feats, "tree", "max_depth", [2, 4], "min_samples_leaf", [1, 5], metric="accuracy")
    assert mat.shape == (2, 2)
    finite = mat[np.isfinite(mat)]
    assert ((finite >= 0.0) & (finite <= 1.0)).all()


def test_learning_curve_uses_more_data():
    feats = _toy_feats()
    lc = experiments.learning_curve(feats, "logreg")
    assert list(lc["n_seasons"]) == [1, 2]
    assert lc["n_train_rows"].is_monotonic_increasing


def test_time_fit_nonnegative():
    feats = _toy_feats()
    X, y, _ = feature_matrix(feats)
    assert experiments.time_fit("tree", X, y, repeats=2) >= 0.0


def test_convergence_logreg_records_decreasing_loss():
    feats = _toy_feats()
    loss = experiments.convergence_logreg(feats, n_iters=200)
    assert len(loss) == 200
    assert loss[-1] < loss[0]


def test_oob_vs_estimators():
    feats = _toy_feats()
    df = experiments.oob_vs_estimators(feats, n_values=[5, 10])
    assert list(df["n_estimators"]) == [5, 10]
    assert df["oob_error"].notna().all()


def test_threshold_sweep_ranges():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    s = rng.random(100)
    df = experiments.threshold_sweep(y, s, thresholds=[0.3, 0.5, 0.7])
    assert list(df["threshold"]) == [0.3, 0.5, 0.7]
    for m in ("accuracy", "precision", "recall", "f1"):
        assert df[m].between(0.0, 1.0).all()


def test_per_season_metrics_columns():
    feats = _toy_feats()
    ps = experiments.per_season_metrics(feats, "tree")
    assert "podium_hit_rate" in ps.columns and "accuracy" in ps.columns
    assert len(ps) >= 1


def test_roc_auc_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1])
    fpr, tpr, _ = metrics.roc_curve(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    assert abs(metrics.auc(fpr, tpr) - 1.0) < 1e-9
    fpr2, tpr2, _ = metrics.roc_curve(y, np.full(6, 0.5))
    assert abs(metrics.auc(fpr2, tpr2) - 0.5) < 1e-9


def test_pr_curve_average_precision():
    y = np.array([0, 0, 1, 1])
    rec, prec, _ = metrics.pr_curve(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert rec[0] == 0.0 and prec[0] == 1.0
    assert abs(metrics.average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-9


def test_calibration_counts_sum_to_n():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.random(200)
    mean_pred, frac_pos, counts = metrics.calibration_curve(y, s, n_bins=10)
    assert len(mean_pred) == 10 and len(frac_pos) == 10
    assert counts.sum() == 200

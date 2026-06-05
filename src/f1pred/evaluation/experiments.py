from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import config
from ..features.engineering import feature_matrix
from ..models import build_model
from . import metrics
from .rolling import rolling_cross_year

_METRIC_KEYS = {
    "accuracy": "mean_accuracy",
    "f1": "mean_f1",
    "podium_hit_rate": "mean_podium_hit_rate",
    "winner_acc": "mean_winner_acc",
    "champ_top3_overlap": "mean_champ_top3_overlap",
}
SWEEP_METRICS = list(_METRIC_KEYS.keys())


def fmt_value(v) -> str:
    if v is None:
        return "all"
    return str(v)


def time_fit(model_key: str, X: np.ndarray, y: np.ndarray, repeats: int = 3, **overrides) -> float:
    times = []
    for _ in range(max(1, repeats)):
        model = build_model(model_key, **overrides)
        t0 = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def train_time_per_model(feats: pd.DataFrame, model_keys=None, repeats: int = 3) -> pd.DataFrame:
    model_keys = list(model_keys) if model_keys else list(config.MODEL_KEYS)
    last = int(feats["season"].max())
    train = feats[feats["season"] < last]
    X, y, _ = feature_matrix(train)
    rows = [
        {"model": k, "fit_seconds": time_fit(k, X, y, repeats=repeats), "n_train_rows": len(train)}
        for k in model_keys
    ]
    return pd.DataFrame(rows)


def sweep_1d(feats: pd.DataFrame, model_key: str, param: str, values, seasons=None) -> pd.DataFrame:
    rows = []
    for v in values:
        res = rolling_cross_year(feats, model_key, seasons=seasons, verbose=False, **{param: v})
        ov = res.overall()
        row = {"param": param, "value": v, "value_str": fmt_value(v)}
        for short, key in _METRIC_KEYS.items():
            row[short] = ov.get(key, float("nan"))
        row["fit_seconds"] = ov.get("fit_seconds_mean", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def grid_2d(
    feats: pd.DataFrame,
    model_key: str,
    param_a: str,
    values_a,
    param_b: str,
    values_b,
    metric: str = "podium_hit_rate",
    seasons=None,
) -> np.ndarray:
    key = _METRIC_KEYS.get(metric, f"mean_{metric}")
    mat = np.full((len(values_a), len(values_b)), np.nan)
    for i, a in enumerate(values_a):
        for j, b in enumerate(values_b):
            res = rolling_cross_year(
                feats, model_key, seasons=seasons, verbose=False, **{param_a: a, param_b: b}
            )
            mat[i, j] = res.overall().get(key, np.nan)
    return mat


def learning_curve(feats: pd.DataFrame, model_key: str, holdout_season: int | None = None, **overrides) -> pd.DataFrame:
    seasons = sorted(feats["season"].unique())
    if holdout_season is None:
        holdout_season = seasons[-1]
    candidates = [s for s in seasons if s < holdout_season]
    test = feats[feats["season"] == holdout_season]
    Xte, yte, _ = feature_matrix(test)

    rows = []
    for k in range(1, len(candidates) + 1):
        train = feats[feats["season"].isin(candidates[:k])]
        Xtr, ytr, _ = feature_matrix(train)
        model = build_model(model_key, **overrides)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)
        y_pred = (proba >= 0.5).astype(int)
        pod = metrics.podium_metrics(test, proba)
        rows.append(
            {
                "n_seasons": k,
                "n_train_rows": len(train),
                "accuracy": metrics.accuracy(yte, y_pred),
                "f1": metrics.f1(yte, y_pred),
                "podium_hit_rate": pod["podium_hit_rate"],
                "winner_acc": pod["winner_acc"],
            }
        )
    return pd.DataFrame(rows)


def collect_probabilities(feats: pd.DataFrame, model_key: str, **overrides):
    res = rolling_cross_year(feats, model_key, verbose=False, **overrides)
    return res.y_true_all, res.y_proba_all


def per_season_metrics(feats: pd.DataFrame, model_key: str, **overrides) -> pd.DataFrame:
    return rolling_cross_year(feats, model_key, verbose=False, **overrides).per_season


def threshold_sweep(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 3)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": metrics.accuracy(y_true, y_pred),
                "precision": metrics.precision(y_true, y_pred),
                "recall": metrics.recall(y_true, y_pred),
                "f1": metrics.f1(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def convergence_logreg(feats: pd.DataFrame, **overrides) -> np.ndarray:
    last = int(feats["season"].max())
    train = feats[feats["season"] < last]
    X, y, _ = feature_matrix(train)
    model = build_model("logreg", **overrides)
    model.fit(X, y)
    return np.asarray(model.loss_history_, dtype=float)


def oob_vs_estimators(feats: pd.DataFrame, n_values=None, **overrides) -> pd.DataFrame:
    n_values = list(n_values) if n_values else list(config.OOB_N_ESTIMATORS)
    last = int(feats["season"].max())
    train = feats[feats["season"] < last]
    X, y, _ = feature_matrix(train)

    rows = []
    for n in n_values:
        model = build_model("forest", n_estimators=n, oob_score=True, **overrides)
        model.fit(X, y)
        oob = model.oob_score_
        rows.append(
            {
                "n_estimators": n,
                "oob_score": oob if oob is not None else float("nan"),
                "oob_error": (1.0 - oob) if oob is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows)

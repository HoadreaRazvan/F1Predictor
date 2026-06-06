from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.engineering import feature_matrix
from ..models import build_model
from . import metrics

_METRICS = ("accuracy", "podium_hit_rate", "exact_podium_set", "winner_acc")


@dataclass
class RollingResult:

    model_key: str
    per_season: pd.DataFrame
    y_true_all: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred_all: np.ndarray = field(default_factory=lambda: np.array([]))
    y_proba_all: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_importances: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    fit_seconds_mean: float = 0.0

    def overall(self) -> dict:
        num = self.per_season.select_dtypes("number").mean(numeric_only=True)
        out = {f"mean_{k}": float(v) for k, v in num.items()}
        out["fit_seconds_mean"] = self.fit_seconds_mean
        if len(self.y_true_all):
            out["overall_accuracy"] = metrics.accuracy(self.y_true_all, self.y_pred_all)
            out["overall_f1"] = metrics.f1(self.y_true_all, self.y_pred_all)
        return out


def _eval_on(model, season_df: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    X, y_true, _ = feature_matrix(season_df)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    y_pred = (proba >= 0.5).astype(int)
    pod = metrics.podium_metrics(season_df, proba)
    m = {
        "accuracy": metrics.accuracy(y_true, y_pred),
        "podium_hit_rate": pod["podium_hit_rate"],
        "exact_podium_set": pod["exact_podium_set"],
        "winner_acc": pod["winner_acc"],
        "n_races": pod["n_races"],
    }
    return m, y_true, y_pred, proba


def rolling_cross_year(
    feats: pd.DataFrame,
    model_key: str,
    seasons=None,
    verbose: bool = True,
    **model_overrides,
) -> RollingResult:
    seasons = sorted(seasons) if seasons is not None else sorted(feats["season"].unique())

    records = []
    y_true_all, y_pred_all, y_proba_all = [], [], []
    fit_times: list[float] = []
    last_importances = None
    feat_names: list[str] = []

    for i in range(2, len(seasons)):
        test_season = seasons[i]
        val_season = seasons[i - 1]
        train = feats[feats["season"].isin(seasons[:i - 1])]
        val = feats[feats["season"] == val_season]
        test = feats[feats["season"] == test_season]
        if train.empty or test.empty:
            continue

        Xtr, ytr, feat_names = feature_matrix(train)

        model = build_model(model_key, **model_overrides)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - t0
        fit_times.append(fit_seconds)

        test_m, yte, y_pred_test, proba_test = _eval_on(model, test)

        val_m = _eval_on(model, val)[0] if not val.empty else {}

        rec = {
            "season": int(test_season),
            "val_season": int(val_season),
            "n_train_rows": int(len(train)),
            **{k: test_m[k] for k in _METRICS},
            "n_races": test_m["n_races"],
            **{f"val_{k}": val_m.get(k, float("nan")) for k in _METRICS},
            "fit_seconds": fit_seconds,
        }
        records.append(rec)

        y_true_all.extend(yte.tolist())
        y_pred_all.extend(y_pred_test.tolist())
        y_proba_all.extend(proba_test.tolist())
        last_importances = model.feature_importances_

        if verbose:
            print(
                f"  [{model_key:6s}] test {test_season} (val {val_season}): "
                f"acc={rec['accuracy']:.3f} podium_hit={rec['podium_hit_rate']:.3f} "
                f"exact_podium={rec['exact_podium_set']:.3f} top1={rec['winner_acc']:.3f}"
            )

    per_season = pd.DataFrame(records)
    return RollingResult(
        model_key=model_key,
        per_season=per_season,
        y_true_all=np.array(y_true_all),
        y_pred_all=np.array(y_pred_all),
        y_proba_all=np.array(y_proba_all),
        feature_importances=last_importances,
        feature_names=feat_names,
        fit_seconds_mean=float(np.mean(fit_times)) if fit_times else 0.0,
    )

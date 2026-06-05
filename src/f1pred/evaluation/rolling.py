from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..features.engineering import feature_matrix
from ..models import build_model
from ..predict.season_standings import actual_standings, predicted_standings
from . import metrics


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

    for test_season in seasons[1:]:
        train = feats[feats["season"] < test_season]
        test = feats[feats["season"] == test_season]
        if train.empty or test.empty:
            continue

        Xtr, ytr, feat_names = feature_matrix(train)
        Xte, yte, _ = feature_matrix(test)

        model = build_model(model_key, **model_overrides)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - t0
        fit_times.append(fit_seconds)
        proba = model.predict_proba(Xte)
        y_pred = (proba >= 0.5).astype(int)

        cls = metrics.classification_report(yte, y_pred)
        pod = metrics.podium_metrics(test, proba)

        pred_tbl = predicted_standings(test, proba)
        act_tbl = actual_standings(test)
        champ = metrics.standings_overlap(
            list(pred_tbl["driver"].head(3)), list(act_tbl["driver"].head(3))
        )

        rec = {
            "season": int(test_season),
            "n_train_rows": int(len(train)),
            **{k: cls[k] for k in ("accuracy", "precision", "recall", "f1")},
            **{k: pod[k] for k in ("podium_hit_rate", "exact_podium_set", "podium_ordered", "winner_acc")},
            "n_races": pod["n_races"],
            **champ,
            "fit_seconds": fit_seconds,
        }
        records.append(rec)
        y_true_all.extend(yte.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_proba_all.extend(np.asarray(proba, dtype=float).tolist())
        last_importances = model.feature_importances_

        if verbose:
            print(
                f"  [{model_key:6s}] {test_season}: "
                f"acc={rec['accuracy']:.3f} F1={rec['f1']:.3f} "
                f"podium_hit={rec['podium_hit_rate']:.3f} winner={rec['winner_acc']:.3f} "
                f"champ_top3={rec['champ_top3_overlap']:.2f}"
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

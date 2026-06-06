from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..predict.race_podium import rank_drivers


def _counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def precision(y_true, y_pred) -> float:
    tp, fp, _, _ = _counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred) -> float:
    tp, _, fn, _ = _counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(y_true, y_pred) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def confusion_matrix(y_true, y_pred) -> dict:
    tp, fp, fn, tn = _counts(y_true, y_pred)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def classification_report(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1(y_true, y_pred),
        **confusion_matrix(y_true, y_pred),
    }


def _safe_mean(seq) -> float:
    return float(np.mean(seq)) if len(seq) else 0.0


def podium_metrics(season_df: pd.DataFrame, proba) -> dict:
    df = season_df.copy()
    df["_proba"] = list(proba)

    hit, exact_set, ordered, winner = [], [], [], []
    for _, race in df.groupby("round", sort=True):
        actual = list(race.sort_values("position")["driver"].head(config.PODIUM_CUTOFF))
        if len(actual) < config.PODIUM_CUTOFF:
            continue
        ranked = rank_drivers(race, race["_proba"].to_numpy())
        pred = list(ranked["driver"].head(config.PODIUM_CUTOFF))

        inter = len(set(pred) & set(actual))
        hit.append(inter / config.PODIUM_CUTOFF)
        exact_set.append(1.0 if set(pred) == set(actual) else 0.0)
        ordered.append(_safe_mean([pred[i] == actual[i] for i in range(config.PODIUM_CUTOFF)]))
        winner.append(1.0 if pred[0] == actual[0] else 0.0)

    return {
        "podium_hit_rate": _safe_mean(hit),
        "exact_podium_set": _safe_mean(exact_set),
        "podium_ordered": _safe_mean(ordered),
        "winner_acc": _safe_mean(winner),
        "n_races": len(hit),
    }

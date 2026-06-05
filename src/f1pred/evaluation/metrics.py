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


def standings_overlap(pred_top3: list[str], actual_top3: list[str]) -> dict:
    n = min(len(pred_top3), len(actual_top3), 3)
    overlap = len(set(pred_top3[:3]) & set(actual_top3[:3])) / 3.0
    exact = 1.0 if pred_top3[:n] == actual_top3[:n] and n == 3 else 0.0
    return {"champ_top3_overlap": overlap, "champ_top3_exact": exact}


def _ranked_counts(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    if len(y_score):
        distinct = np.where(np.diff(y_score) != 0)[0]
        idx = np.r_[distinct, len(y_score) - 1]
    else:
        idx = np.array([], dtype=int)
    return tps[idx], fps[idx], y_score[idx], float((y_true == 1).sum()), float((y_true == 0).sum())


def auc(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def roc_curve(y_true, y_score):
    tp, fp, thr, P, N = _ranked_counts(y_true, y_score)
    tpr = np.r_[0.0, tp / P] if P > 0 else np.r_[0.0, np.zeros(len(tp))]
    fpr = np.r_[0.0, fp / N] if N > 0 else np.r_[0.0, np.zeros(len(fp))]
    thresholds = np.r_[np.inf, thr]
    return fpr, tpr, thresholds


def roc_auc(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def pr_curve(y_true, y_score):
    tp, fp, thr, P, _ = _ranked_counts(y_true, y_score)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / P if P > 0 else np.zeros(len(tp))
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    thresholds = np.r_[np.inf, thr]
    return recall, precision, thresholds


def average_precision(y_true, y_score) -> float:
    recall, precision, _ = pr_curve(y_true, y_score)
    return float(np.sum(np.diff(recall) * precision[1:]))


def calibration_curve(y_true, y_score, n_bins: int = 10):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)

    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        counts[b] = c
        if c > 0:
            mean_pred[b] = float(y_score[m].mean())
            frac_pos[b] = float(y_true[m].mean())
    return mean_pred, frac_pos, counts

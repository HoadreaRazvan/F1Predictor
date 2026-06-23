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

def log_loss(y_true, proba, eps: float = 1e-12) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)
    if len(y) == 0:
        return 0.0
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

def brier_score(y_true, proba) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(y) else 0.0

def roc_auc(y_true, proba) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=float)
    i, n = 0, len(p)
    while i < n:
        j = i
        while j < n and sorted_p[j] == sorted_p[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def pr_auc(y_true, proba) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(p, kind="mergesort")[::-1]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    rec_prev = np.concatenate(([0.0], rec[:-1]))
    return float(np.sum((rec - rec_prev) * prec))

def reliability_curve(y_true, proba, n_bins: int = 10):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        count.append(c)
        conf.append(float(p[m].mean()) if c else float((edges[b] + edges[b + 1]) / 2))
        acc.append(float(y[m].mean()) if c else float("nan"))
    return np.array(conf), np.array(acc), np.array(count)

def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    n = len(y)
    if n == 0:
        return 0.0
    conf, acc, count = reliability_curve(y, proba, n_bins)
    ece = 0.0
    for cf, ac, ct in zip(conf, acc, count):
        if ct and not np.isnan(ac):
            ece += (ct / n) * abs(ac - cf)
    return float(ece)

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

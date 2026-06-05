from __future__ import annotations

import numpy as np

from .base import BaseModel


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    for r in (p, 1.0 - p):
        m = r > 0
        out[m] -= r[m] * np.log2(r[m])
    return out


def _node_entropy(pos: float, total: float) -> float:
    if total == 0:
        return 0.0
    return float(_binary_entropy(np.array([pos / total]))[0])


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "is_leaf")

    def __init__(self):
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: "_Node | None" = None
        self.right: "_Node | None" = None
        self.value: float = 0.0
        self.is_leaf: bool = False


class DecisionTree(BaseModel):

    name = "tree"

    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: int | str | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng()

        self.root: _Node | None = None
        self.n_features_: int = 0
        self._importance_acc: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.n_features_ = X.shape[1]
        self._importance_acc = np.zeros(self.n_features_)

        self.root = self._build(X, y, depth=0)

        total = self._importance_acc.sum()
        if total > 0:
            self._feature_importances = self._importance_acc / total
        else:
            self._feature_importances = self._importance_acc.copy()
        return self

    def _n_features_to_try(self) -> int:
        d = self.n_features_
        if self.max_features is None:
            return d
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if self.max_features == "log2":
            return max(1, int(np.log2(d)))
        return max(1, min(int(self.max_features), d))

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node()
        n = len(y)
        pos = float(y.sum())
        node.value = pos / n if n > 0 else 0.0

        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or pos == 0
            or pos == n
        ):
            node.is_leaf = True
            return node

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            node.is_leaf = True
            return node

        mask = X[:, feat] <= thr
        self._importance_acc[feat] += gain * n

        node.feature = feat
        node.threshold = thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n = len(y)
        total_pos = float(y.sum())
        parent_entropy = _node_entropy(total_pos, n)

        best_gain = 0.0
        best_feat: int | None = None
        best_thr: float | None = None

        n_try = self._n_features_to_try()
        if n_try < self.n_features_:
            features = self.rng.choice(self.n_features_, size=n_try, replace=False)
        else:
            features = range(self.n_features_)

        min_leaf = self.min_samples_leaf

        for feat in features:
            x = X[:, feat]
            order = np.argsort(x, kind="mergesort")
            x_sorted = x[order]
            y_sorted = y[order]

            cum_pos = np.cumsum(y_sorted)
            left_total = np.arange(1, n)
            left_pos = cum_pos[:-1]
            right_total = n - left_total
            right_pos = total_pos - left_pos

            valid = x_sorted[1:] != x_sorted[:-1]
            valid &= (left_total >= min_leaf) & (right_total >= min_leaf)
            if not valid.any():
                continue

            h_left = _binary_entropy(left_pos / left_total)
            h_right = _binary_entropy(right_pos / right_total)
            child_entropy = (left_total / n) * h_left + (right_total / n) * h_right
            gain = parent_entropy - child_entropy

            gain[~valid] = -np.inf
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain = float(gain[k])
                best_feat = int(feat)
                best_thr = float((x_sorted[k] + x_sorted[k + 1]) / 2.0)

        return best_feat, best_thr, best_gain

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self._predict_node(self.root, X)

    def _predict_node(self, node: _Node, X: np.ndarray) -> np.ndarray:
        if node.is_leaf or node.feature is None:
            return np.full(X.shape[0], node.value)
        out = np.empty(X.shape[0], dtype=float)
        mask = X[:, node.feature] <= node.threshold
        if mask.any():
            out[mask] = self._predict_node(node.left, X[mask])
        if (~mask).any():
            out[~mask] = self._predict_node(node.right, X[~mask])
        return out

from __future__ import annotations

import numpy as np

from .base import BaseModel
from .decision_tree import DecisionTree

class RandomForest(BaseModel):
    name = "forest"

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 12,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        max_features: int | str | None = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.trees: list[DecisionTree] = []
        self.n_features_: int = 0
        self.oob_score_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        self.n_features_ = d
        rng = np.random.default_rng(self.random_state)

        self.trees = []

        oob_sum = np.zeros(n)
        oob_count = np.zeros(n)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
            else:
                idx = np.arange(n)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

            if self.oob_score and self.bootstrap:
                oob_mask = np.ones(n, dtype=bool)
                oob_mask[idx] = False
                if oob_mask.any():
                    oob_sum[oob_mask] += tree.predict_proba(X[oob_mask])
                    oob_count[oob_mask] += 1

        imp = np.zeros(d)
        for tree in self.trees:
            fi = tree.feature_importances_
            if fi is not None:
                imp += fi
        self._feature_importances = imp / max(len(self.trees), 1)

        if self.oob_score and self.bootstrap:
            seen = oob_count > 0
            if seen.any():
                oob_proba = oob_sum[seen] / oob_count[seen]
                oob_pred = (oob_proba >= 0.5).astype(int)
                self.oob_score_ = float((oob_pred == y[seen]).mean())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.trees:
            raise RuntimeError("Random Forest neantrenat: apelează fit() întâi.")
        acc = np.zeros(X.shape[0])
        for tree in self.trees:
            acc += tree.predict_proba(X)
        return acc / len(self.trees)

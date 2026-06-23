from __future__ import annotations

import numpy as np

from .base import BaseModel

class RandomPredictor(BaseModel):
    name = "random"

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state
        self._rng: np.random.Generator | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomPredictor":
        self._rng = np.random.default_rng(self.random_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._rng is None:
            self._rng = np.random.default_rng(self.random_state)
        return self._rng.random(X.shape[0])

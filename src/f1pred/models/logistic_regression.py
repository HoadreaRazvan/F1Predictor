from __future__ import annotations

import numpy as np

from .base import BaseModel, Standardizer

def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

class LogisticRegression(BaseModel):
    name = "logreg"

    def __init__(
        self,
        lr: float = 0.1,
        n_iters: int = 3000,
        l2: float = 0.0,
        class_weight: bool = False,
        standardize: bool = True,
    ) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.class_weight = class_weight
        self.standardize = standardize

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self._scaler: Standardizer | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if self.standardize:
            self._scaler = Standardizer().fit(X)
            X = self._scaler.transform(X)

        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        if self.class_weight:
            n_pos = float((y == 1).sum())
            n_neg = float((y == 0).sum())
            w_pos = n / (2.0 * n_pos) if n_pos > 0 else 1.0
            w_neg = n / (2.0 * n_neg) if n_neg > 0 else 1.0
            sample_w = np.where(y == 1, w_pos, w_neg)
        else:
            sample_w = np.ones(n)
        sw_sum = sample_w.sum()

        self.loss_history_ = []
        eps = 1e-12
        for _ in range(self.n_iters):
            z = X @ self.w + self.b
            p = _sigmoid(z)

            ce = -(sample_w * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))).sum() / sw_sum
            self.loss_history_.append(float(ce + 0.5 * self.l2 * float(self.w @ self.w)))

            err = (p - y) * sample_w
            grad_w = X.T @ err / sw_sum + self.l2 * self.w
            grad_b = err.sum() / sw_sum
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        self._feature_importances = np.abs(self.w)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return _sigmoid(X @ self.w + self.b)

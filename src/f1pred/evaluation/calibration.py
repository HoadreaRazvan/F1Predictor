from __future__ import annotations

import numpy as np

from ..features.engineering import feature_matrix
from ..models import LogisticRegression, build_model

class PlattScaler:
    def __init__(self) -> None:
        self._lr = LogisticRegression(lr=0.5, n_iters=2000, l2=0.0,
                                      class_weight=False, standardize=True)

    def fit(self, scores, y) -> "PlattScaler":
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        self._lr.fit(s, np.asarray(y, dtype=float))
        return self

    def transform(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(s)

class IsotonicCalibrator:
    def __init__(self) -> None:
        self.x_thr_: np.ndarray | None = None
        self.y_val_: np.ndarray | None = None

    def fit(self, scores, y) -> "IsotonicCalibrator":
        s = np.asarray(scores, dtype=float)
        t = np.asarray(y, dtype=float)
        order = np.argsort(s, kind="mergesort")
        s_sorted = s[order]
        t_sorted = t[order]

        v_stack: list[float] = []
        w_stack: list[float] = []
        for v in t_sorted:
            v_stack.append(float(v))
            w_stack.append(1.0)
            while len(v_stack) > 1 and v_stack[-2] > v_stack[-1]:
                v2, w2 = v_stack.pop(), w_stack.pop()
                v1, w1 = v_stack.pop(), w_stack.pop()
                w_new = w1 + w2
                v_stack.append((v1 * w1 + v2 * w2) / w_new)
                w_stack.append(w_new)

        fitted = np.empty(len(s_sorted))
        i = 0
        for v, w in zip(v_stack, w_stack):
            cnt = int(round(w))
            fitted[i:i + cnt] = v
            i += cnt

        uniq = np.concatenate((np.diff(s_sorted) > 0, [True]))
        self.x_thr_ = s_sorted[uniq]
        self.y_val_ = fitted[uniq]
        return self

    def transform(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        return np.interp(s, self.x_thr_, self.y_val_)

def calibration_experiment(feats, model_key: str, seasons=None) -> dict:
    seasons = sorted(seasons) if seasons is not None else sorted(feats["season"].unique())
    y_all, raw_all, platt_all, iso_all = [], [], [], []

    for i in range(2, len(seasons)):
        train = feats[feats["season"].isin(seasons[:i - 1])]
        val = feats[feats["season"] == seasons[i - 1]]
        test = feats[feats["season"] == seasons[i]]
        if train.empty or test.empty or val.empty:
            continue

        Xtr, ytr, _ = feature_matrix(train)
        model = build_model(model_key)
        model.fit(Xtr, ytr)

        Xval, yval, _ = feature_matrix(val)
        sval = np.asarray(model.predict_proba(Xval), dtype=float)
        platt = PlattScaler().fit(sval, yval)
        iso = IsotonicCalibrator().fit(sval, yval)

        Xte, yte, _ = feature_matrix(test)
        raw = np.asarray(model.predict_proba(Xte), dtype=float)
        y_all.extend(yte.tolist())
        raw_all.extend(raw.tolist())
        platt_all.extend(platt.transform(raw).tolist())
        iso_all.extend(iso.transform(raw).tolist())

    return {
        "y": np.array(y_all),
        "raw": np.array(raw_all),
        "platt": np.array(platt_all),
        "iso": np.array(iso_all),
    }

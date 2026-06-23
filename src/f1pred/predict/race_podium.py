from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.engineering import FEATURE_COLUMNS

def predict_proba_for_races(model, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    return model.predict_proba(X)

def rank_drivers(race_df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    cols = [c for c in ("driver", "team", "grid", "position") if c in race_df.columns]
    out = race_df[cols].copy()
    out["proba"] = np.asarray(proba, dtype=float)
    out = out.sort_values(["proba", "grid"], ascending=[False, True]).reset_index(drop=True)
    out["pred_position"] = np.arange(1, len(out) + 1)
    return out

def predicted_podium(ranked: pd.DataFrame) -> list[str]:
    return list(ranked["driver"].head(3))

import numpy as np
import pandas as pd

from f1pred.features.engineering import FEATURE_COLUMNS, feature_matrix

def _toy_matrix(n: int = 10) -> pd.DataFrame:
    data = {c: np.arange(n, dtype=float) + i for i, c in enumerate(FEATURE_COLUMNS)}
    data["is_podium"] = (np.arange(n) < 3).astype(float)
    return pd.DataFrame(data)

def test_feature_matrix_default_uses_all_columns():
    df = _toy_matrix()
    X, y, names = feature_matrix(df)
    assert X.shape == (len(df), len(FEATURE_COLUMNS))
    assert names == list(FEATURE_COLUMNS)
    assert y.shape == (len(df),)

def test_feature_matrix_columns_subset_for_ablation():
    df = _toy_matrix()
    drop = FEATURE_COLUMNS[0]
    cols = [c for c in FEATURE_COLUMNS if c != drop]

    X, y, names = feature_matrix(df, columns=cols)
    assert X.shape == (len(df), len(FEATURE_COLUMNS) - 1)
    assert drop not in names
    assert names == cols

    assert np.array_equal(y, feature_matrix(df)[1])

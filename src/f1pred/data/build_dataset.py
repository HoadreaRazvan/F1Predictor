from __future__ import annotations

import pandas as pd

from .. import config
from . import loader


def build_results_long(seasons=None, save: bool = True) -> pd.DataFrame:
    config.ensure_dirs()
    df = loader.load_all(seasons)

    df = df.sort_values(["season", "round", "position"], na_position="last").reset_index(drop=True)

    if save:
        df.to_parquet(config.DATASET_PATH, index=False)
        print(f"Salvat: {config.DATASET_PATH}  ({len(df)} rânduri)")
    return df


def get_results_long(force: bool = False) -> pd.DataFrame:
    if not force and config.DATASET_PATH.exists():
        return pd.read_parquet(config.DATASET_PATH)
    return build_results_long(save=True)

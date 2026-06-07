from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from . import extras as extras_mod
from . import loader

_EXTRA_COLS = ["pit_stop_time", "race_pace", "sc_flag", "tyre_deg"]


def _merge_extras(df: pd.DataFrame) -> pd.DataFrame:
    extras = extras_mod.get_race_extras()
    if extras is None or extras.empty:
        for c in _EXTRA_COLS:
            df[c] = np.nan
        return df
    e = extras[["season", "round", "driver_id", *_EXTRA_COLS]].copy()
    e["season"] = e["season"].astype(int)
    e["round"] = e["round"].astype(int)
    e["driver_id"] = e["driver_id"].astype(str)
    e = e.drop_duplicates(subset=["season", "round", "driver_id"])
    return df.merge(e, on=["season", "round", "driver_id"], how="left")


def build_results_long(seasons=None, save: bool = True) -> pd.DataFrame:
    config.ensure_dirs()
    df = loader.load_all(seasons)
    df = _merge_extras(df)

    df = df.sort_values(["season", "round", "position"], na_position="last").reset_index(drop=True)

    if save:
        df.to_parquet(config.DATASET_PATH, index=False)
        print(f"Salvat: {config.DATASET_PATH}  ({len(df)} rânduri)")
    return df


def get_results_long(force: bool = False) -> pd.DataFrame:
    if not force and config.DATASET_PATH.exists():
        return pd.read_parquet(config.DATASET_PATH)
    return build_results_long(save=True)

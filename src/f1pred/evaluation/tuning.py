from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import config
from .rolling import rolling_cross_year

_METRICS = ("accuracy", "podium_hit_rate", "exact_podium_set", "winner_acc")

_DEFAULT_PARAMS = {
    "logreg": config.LOGREG_PARAMS,
    "tree": config.TREE_PARAMS,
    "forest": config.FOREST_PARAMS,
    "random": config.RANDOM_PARAMS,
}

def _native(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v

def iter_configs(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]

def _default_subset(model_key: str, grid: dict) -> dict:
    base = _DEFAULT_PARAMS[model_key]
    return {k: base[k] for k in grid if k in base}

def tune_model(feats: pd.DataFrame, model_key: str, grid: dict | None = None, *,
               metric: str = None, verbose: bool = True) -> pd.DataFrame:
    grid = grid if grid is not None else config.GRIDS[model_key]
    metric = metric or config.TUNE_METRIC
    configs = iter_configs(grid)

    default = _default_subset(model_key, grid)
    if default and default not in configs:
        configs = configs + [default]

    rows = []
    for i, cfg in enumerate(configs, 1):
        res = rolling_cross_year(feats, model_key, verbose=False, **cfg)
        ov = res.overall()
        row = {k: _native(v) for k, v in cfg.items()}
        for m in _METRICS:
            row[f"val_{m}"] = float(ov.get(f"mean_val_{m}", np.nan))
            row[f"test_{m}"] = float(ov.get(f"mean_{m}", np.nan))
        row["fit_seconds"] = float(ov.get("fit_seconds_mean", np.nan))
        row["is_default"] = (cfg == default)
        rows.append(row)
        if verbose:
            print(f"  [{model_key:6s}] {i:3d}/{len(configs)}  "
                  f"val_{metric}={row[f'val_{metric}']:.3f}  test_{metric}={row[f'test_{metric}']:.3f}  {cfg}")

    df = pd.DataFrame(rows)
    return df.sort_values(f"val_{metric}", ascending=False).reset_index(drop=True)

def best_config(df: pd.DataFrame, metric: str, grid_keys: list[str]) -> dict:
    ranked = df.sort_values(
        [f"val_{metric}", "val_exact_podium_set", "fit_seconds"],
        ascending=[False, False, True],
    )
    top = ranked.iloc[0]
    return {k: _native(top[k]) for k in grid_keys}

def run_tuning(feats: pd.DataFrame, keys, metric: str = None, *,
               save: bool = True, plots: bool = True) -> dict:
    metric = metric or config.TUNE_METRIC
    config.ensure_dirs()
    best = {}

    for key in keys:
        grid = config.GRIDS.get(key, {})
        if not grid:
            print(f"\n=== '{key}': fără hiperparametri de optimizat (se sare peste) ===")
            best[key] = {}
            continue
        n = len(iter_configs(grid))
        print(f"\n=== Optimizare '{key}': {n} configurații "
              f"(selecție după val_{metric}) ===")
        df = tune_model(feats, key, grid, metric=metric, verbose=True)

        if save:
            path = config.OUTPUT_DIR / f"tuning_{key}.csv"
            df.to_csv(path, index=False)
            print(f"  Salvat: {path}")

        bcfg = best_config(df, metric, list(grid.keys()))
        best[key] = bcfg

        brow = df.sort_values(f"val_{metric}", ascending=False).iloc[0]
        print(f"  Cea mai bună configurație: {bcfg}")
        print(f"    validare: {metric}={brow[f'val_{metric}']:.3f}  ||  "
              f"test (informativ): {metric}={brow[f'test_{metric}']:.3f}")
        if df["is_default"].any():
            drow = df[df["is_default"]].iloc[0]
            print(f"  Referință (implicită din config): "
                  f"validare {metric}={drow[f'val_{metric}']:.3f}  ||  "
                  f"test {metric}={drow[f'test_{metric}']:.3f}")

        if plots:
            from ..viz import plots as P

            P.plot_tuning_configs(df, key, metric,
                                  config.PLOTS_DIR / f"tuning_{key}_configs.png")
            P.plot_tuning_curves(df, key, metric, list(grid.keys()),
                                 config.PLOTS_DIR / f"tuning_{key}_curves.png")
            print(f"  Grafice: tuning_{key}_configs.png, tuning_{key}_curves.png")

    return best

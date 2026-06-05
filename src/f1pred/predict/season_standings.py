from __future__ import annotations

import pandas as pd

from .. import config
from .race_podium import rank_drivers


def predicted_standings(season_df: pd.DataFrame, proba) -> pd.DataFrame:
    df = season_df.copy()
    df["_proba"] = list(proba)

    points = {}
    wins = {}
    podiums = {}
    teams = {}

    for _, race in df.groupby("round", sort=True):
        ranked = rank_drivers(race, race["_proba"].to_numpy())
        for _, row in ranked.iterrows():
            drv = row["driver"]
            pos = int(row["pred_position"])
            teams[drv] = row.get("team", "")
            points[drv] = points.get(drv, 0.0) + config.F1_POINTS.get(pos, 0)
            wins[drv] = wins.get(drv, 0) + (1 if pos == 1 else 0)
            podiums[drv] = podiums.get(drv, 0) + (1 if pos <= config.PODIUM_CUTOFF else 0)

    table = pd.DataFrame(
        {
            "driver": list(points.keys()),
            "team": [teams[d] for d in points],
            "pred_points": [points[d] for d in points],
            "pred_wins": [wins[d] for d in points],
            "pred_podiums": [podiums[d] for d in points],
        }
    )
    table = table.sort_values("pred_points", ascending=False).reset_index(drop=True)
    table.insert(0, "pred_rank", range(1, len(table) + 1))
    return table


def actual_standings(season_df: pd.DataFrame) -> pd.DataFrame:
    df = season_df.copy()
    agg = (
        df.groupby("driver")
        .agg(
            team=("team", "last"),
            points=("points", "sum"),
            wins=("position", lambda s: int((s == 1).sum())),
            podiums=("is_podium", "sum"),
        )
        .reset_index()
    )
    agg = agg.sort_values("points", ascending=False).reset_index(drop=True)
    agg.insert(0, "rank", range(1, len(agg) + 1))
    return agg

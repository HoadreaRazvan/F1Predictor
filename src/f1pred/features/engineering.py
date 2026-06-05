from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .. import config

FEATURE_COLUMNS = [
    "grid",
    "quali_pos",
    "is_pole",
    "start_top3",
    "start_top10",
    "form_avg_finish",
    "form_avg_points",
    "form_podium_rate",
    "form_dnf_rate",
    "form_avg_grid",
    "last_finish",
    "season_points",
    "season_position",
    "season_podiums",
    "season_wins",
    "season_races",
    "team_form_avg_finish",
    "team_form_avg_points",
    "team_season_points",
    "team_season_position",
    "circuit_driver_avg_finish",
    "circuit_driver_podium_rate",
    "circuit_team_avg_finish",
    "career_races",
    "round",
    "is_sprint_weekend",
    "rain",
    "track_temp",
    "air_temp",
]

_DEFAULTS = {
    "grid": 20.0,
    "quali_pos": 20.0,
    "form_avg_finish": 15.0,
    "form_avg_points": 0.0,
    "form_podium_rate": 0.0,
    "form_dnf_rate": 0.1,
    "form_avg_grid": 15.0,
    "last_finish": 15.0,
    "season_points": 0.0,
    "season_position": 20.0,
    "season_podiums": 0.0,
    "season_wins": 0.0,
    "season_races": 0.0,
    "team_form_avg_finish": 12.0,
    "team_form_avg_points": 0.0,
    "team_season_points": 0.0,
    "team_season_position": 10.0,
    "circuit_driver_avg_finish": 15.0,
    "circuit_driver_podium_rate": 0.0,
    "circuit_team_avg_finish": 12.0,
    "career_races": 0.0,
    "rain": 0.0,
    "track_temp": 30.0,
    "air_temp": 22.0,
}


def _mean(seq, default):
    return float(np.mean(seq)) if len(seq) else default


def _rank_of(value: float, all_values) -> int:
    return 1 + int(sum(1 for v in all_values if v > value))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    N = config.FORM_WINDOW

    d_finish = defaultdict(lambda: deque(maxlen=N))
    d_points = defaultdict(lambda: deque(maxlen=N))
    d_podium = defaultdict(lambda: deque(maxlen=N))
    d_dnf = defaultdict(lambda: deque(maxlen=N))
    d_grid = defaultdict(lambda: deque(maxlen=N))
    d_last_finish: dict = {}
    d_career = defaultdict(int)

    t_finish = defaultdict(lambda: deque(maxlen=2 * N))
    t_points = defaultdict(lambda: deque(maxlen=2 * N))

    dc_finish = defaultdict(list)
    dc_podium = defaultdict(list)
    tc_finish = defaultdict(list)

    season_pts = defaultdict(float)
    season_pod = defaultdict(float)
    season_win = defaultdict(float)
    season_races_d = defaultdict(float)
    team_season_pts = defaultdict(float)
    current_season = None

    feat_rows: list[dict] = []

    for (season, rnd), grp in df.groupby(["season", "round"], sort=True):
        if season != current_season:
            season_pts = defaultdict(float)
            season_pod = defaultdict(float)
            season_win = defaultdict(float)
            season_races_d = defaultdict(float)
            team_season_pts = defaultdict(float)
            current_season = season

        driver_points_now = dict(season_pts)
        team_points_now = dict(team_season_pts)

        for row in grp.itertuples():
            did = row.driver_id
            tid = row.team_id
            circuit = row.circuit
            grid = float(row.grid) if not pd.isna(row.grid) else _DEFAULTS["grid"]
            if grid <= 0:
                grid = _DEFAULTS["grid"]
            quali = float(row.quali_pos) if not pd.isna(row.quali_pos) else grid

            feat = {
                "grid": grid,
                "quali_pos": quali,
                "is_pole": float(grid == 1),
                "start_top3": float(grid <= 3),
                "start_top10": float(grid <= 10),
                "form_avg_finish": _mean(d_finish[did], _DEFAULTS["form_avg_finish"]),
                "form_avg_points": _mean(d_points[did], _DEFAULTS["form_avg_points"]),
                "form_podium_rate": _mean(d_podium[did], _DEFAULTS["form_podium_rate"]),
                "form_dnf_rate": _mean(d_dnf[did], _DEFAULTS["form_dnf_rate"]),
                "form_avg_grid": _mean(d_grid[did], _DEFAULTS["form_avg_grid"]),
                "last_finish": float(d_last_finish.get(did, _DEFAULTS["last_finish"])),
                "season_points": float(season_pts[did]),
                "season_position": float(
                    _rank_of(season_pts[did], driver_points_now.values())
                ),
                "season_podiums": float(season_pod[did]),
                "season_wins": float(season_win[did]),
                "season_races": float(season_races_d[did]),
                "team_form_avg_finish": _mean(t_finish[tid], _DEFAULTS["team_form_avg_finish"]),
                "team_form_avg_points": _mean(t_points[tid], _DEFAULTS["team_form_avg_points"]),
                "team_season_points": float(team_season_pts[tid]),
                "team_season_position": float(
                    _rank_of(team_season_pts[tid], team_points_now.values())
                ),
                "circuit_driver_avg_finish": _mean(
                    dc_finish[(did, circuit)], _DEFAULTS["circuit_driver_avg_finish"]
                ),
                "circuit_driver_podium_rate": _mean(
                    dc_podium[(did, circuit)], _DEFAULTS["circuit_driver_podium_rate"]
                ),
                "circuit_team_avg_finish": _mean(
                    tc_finish[(tid, circuit)], _DEFAULTS["circuit_team_avg_finish"]
                ),
                "career_races": float(d_career[did]),
                "round": float(rnd),
                "is_sprint_weekend": float(row.is_sprint_weekend),
                "rain": float(row.rain) if not pd.isna(row.rain) else _DEFAULTS["rain"],
                "track_temp": float(row.track_temp)
                if not pd.isna(row.track_temp)
                else _DEFAULTS["track_temp"],
                "air_temp": float(row.air_temp)
                if not pd.isna(row.air_temp)
                else _DEFAULTS["air_temp"],
            }
            feat["_index"] = row.Index
            feat_rows.append(feat)

        for row in grp.itertuples():
            did = row.driver_id
            tid = row.team_id
            circuit = row.circuit
            pos = float(row.position) if not pd.isna(row.position) else 20.0
            pts = float(row.points) if not pd.isna(row.points) else 0.0
            grid = float(row.grid) if not pd.isna(row.grid) else _DEFAULTS["grid"]
            is_pod = 1.0 if pos <= config.PODIUM_CUTOFF else 0.0
            is_win = 1.0 if pos == 1 else 0.0
            is_dnf = 0.0 if float(row.finished) == 1.0 else 1.0

            d_finish[did].append(pos)
            d_points[did].append(pts)
            d_podium[did].append(is_pod)
            d_dnf[did].append(is_dnf)
            d_grid[did].append(grid)
            d_last_finish[did] = pos
            d_career[did] += 1

            t_finish[tid].append(pos)
            t_points[tid].append(pts)

            dc_finish[(did, circuit)].append(pos)
            dc_podium[(did, circuit)].append(is_pod)
            tc_finish[(tid, circuit)].append(pos)

            season_pts[did] += pts
            season_pod[did] += is_pod
            season_win[did] += is_win
            season_races_d[did] += 1
            team_season_pts[tid] += pts

    feat_df = pd.DataFrame(feat_rows).set_index("_index").sort_index()
    overlap = [c for c in feat_df.columns if c in df.columns and c != "round"]
    out = df.drop(columns=overlap).join(feat_df.drop(columns=["round"]))

    for col, default in _DEFAULTS.items():
        if col in out.columns:
            out[col] = out[col].fillna(default)

    out["is_podium"] = (out["position"] <= config.PODIUM_CUTOFF).astype(float)
    return out


def feature_matrix(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["is_podium"].to_numpy(dtype=float)
    return X, y, list(FEATURE_COLUMNS)


def get_feature_dataset(force: bool = False) -> pd.DataFrame:
    from ..data import get_results_long

    if not force and config.FEATURES_PATH.exists():
        return pd.read_parquet(config.FEATURES_PATH)

    config.ensure_dirs()
    raw = get_results_long(force=force)
    feats = add_features(raw)
    feats.to_parquet(config.FEATURES_PATH, index=False)
    print(f"Salvat: {config.FEATURES_PATH}  ({len(feats)} rânduri, {len(FEATURE_COLUMNS)} feature-uri)")
    return feats

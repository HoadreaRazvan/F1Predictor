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
    "quali_gap_to_pole",
    "quali_gap_to_teammate",
    "grid_pos_minus_quali_pos",
    "teammate_grid_pos",
    "driver_points_minus_teammate",
    "team_avg_quali_pos_last_5",
    "team_dnf_rate_last_5",
    "team_avg_pit_stop_time",
    "circuit_overtaking_difficulty",
    "circuit_safety_car_rate",
    "circuit_podium_rate_from_grid_pos",
    "rain_probability",
    "tyre_degradation_level",
    "driver_wet_performance",
    "race_pace_last_5",
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
    "quali_gap_to_pole": 1.5,
    "quali_gap_to_teammate": 0.0,
    "grid_pos_minus_quali_pos": 0.0,
    "teammate_grid_pos": 15.0,
    "driver_points_minus_teammate": 0.0,
    "team_avg_quali_pos_last_5": 12.0,
    "team_dnf_rate_last_5": 0.1,
    "team_avg_pit_stop_time": 24.0,
    "circuit_overtaking_difficulty": 0.5,
    "circuit_safety_car_rate": 0.4,
    "circuit_podium_rate_from_grid_pos": 0.0,
    "rain_probability": 0.15,
    "tyre_degradation_level": 0.05,
    "driver_wet_performance": 12.0,
    "race_pace_last_5": 1.05,
}


def _mean(seq, default):
    return float(np.mean(seq)) if len(seq) else default


def _rank_of(value: float, all_values) -> int:
    return 1 + int(sum(1 for v in all_values if v > value))


def _overtaking_difficulty(poschanges) -> float:
    if not len(poschanges):
        return _DEFAULTS["circuit_overtaking_difficulty"]
    avg = float(np.mean(poschanges))
    return float(np.clip(1.0 - avg / 10.0, 0.0, 1.0))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    N = config.FORM_WINDOW

    for col in ("quali_best", "pit_stop_time", "race_pace", "sc_flag", "tyre_deg"):
        if col not in df.columns:
            df[col] = np.nan

    d_finish = defaultdict(lambda: deque(maxlen=N))
    d_points = defaultdict(lambda: deque(maxlen=N))
    d_podium = defaultdict(lambda: deque(maxlen=N))
    d_dnf = defaultdict(lambda: deque(maxlen=N))
    d_grid = defaultdict(lambda: deque(maxlen=N))
    d_last_finish: dict = {}
    d_career = defaultdict(int)

    t_finish = defaultdict(lambda: deque(maxlen=2 * N))
    t_points = defaultdict(lambda: deque(maxlen=2 * N))
    t_quali = defaultdict(lambda: deque(maxlen=2 * N))
    t_dnf = defaultdict(lambda: deque(maxlen=2 * N))
    t_pit = defaultdict(lambda: deque(maxlen=2 * N))

    dc_finish = defaultdict(list)
    dc_podium = defaultdict(list)
    tc_finish = defaultdict(list)

    d_pace = defaultdict(lambda: deque(maxlen=N))
    d_wet = defaultdict(list)
    c_poschange = defaultdict(list)
    c_sc = defaultdict(list)
    c_rain = defaultdict(list)
    c_tyre = defaultdict(list)
    grid_pod_hits = defaultdict(float)
    grid_pod_total = defaultdict(float)

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

        team_members = defaultdict(list)
        pole_time = np.nan
        for row in grp.itertuples():
            g = float(row.grid) if not pd.isna(row.grid) else _DEFAULTS["grid"]
            if g <= 0:
                g = _DEFAULTS["grid"]
            qb = float(row.quali_best) if not pd.isna(row.quali_best) else np.nan
            team_members[row.team_id].append({"did": row.driver_id, "grid": g, "qbest": qb})
            if not np.isnan(qb):
                pole_time = qb if np.isnan(pole_time) else min(pole_time, qb)
        mate_of = {}
        for members in team_members.values():
            for m in members:
                others = [x for x in members if x["did"] != m["did"]]
                mate_of[m["did"]] = others[0] if others else None

        for row in grp.itertuples():
            did = row.driver_id
            tid = row.team_id
            circuit = row.circuit
            grid = float(row.grid) if not pd.isna(row.grid) else _DEFAULTS["grid"]
            if grid <= 0:
                grid = _DEFAULTS["grid"]
            quali = float(row.quali_pos) if not pd.isna(row.quali_pos) else grid
            qbest = float(row.quali_best) if not pd.isna(row.quali_best) else np.nan
            mate = mate_of.get(did)

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

            g_int = int(grid)
            feat.update({
                "quali_gap_to_pole": float(qbest - pole_time)
                if (not np.isnan(qbest) and not np.isnan(pole_time))
                else _DEFAULTS["quali_gap_to_pole"],
                "quali_gap_to_teammate": float(qbest - mate["qbest"])
                if (mate is not None and not np.isnan(qbest) and not np.isnan(mate["qbest"]))
                else _DEFAULTS["quali_gap_to_teammate"],
                "grid_pos_minus_quali_pos": float(grid - quali),
                "teammate_grid_pos": float(mate["grid"]) if mate is not None
                else _DEFAULTS["teammate_grid_pos"],
                "driver_points_minus_teammate": float(season_pts[did] - season_pts[mate["did"]])
                if mate is not None else _DEFAULTS["driver_points_minus_teammate"],
                "team_avg_quali_pos_last_5": _mean(t_quali[tid], _DEFAULTS["team_avg_quali_pos_last_5"]),
                "team_dnf_rate_last_5": _mean(t_dnf[tid], _DEFAULTS["team_dnf_rate_last_5"]),
                "team_avg_pit_stop_time": _mean(t_pit[tid], _DEFAULTS["team_avg_pit_stop_time"]),
                "circuit_overtaking_difficulty": _overtaking_difficulty(c_poschange[circuit]),
                "circuit_safety_car_rate": _mean(c_sc[circuit], _DEFAULTS["circuit_safety_car_rate"]),
                "circuit_podium_rate_from_grid_pos": float(grid_pod_hits[g_int] / grid_pod_total[g_int])
                if grid_pod_total[g_int] > 0 else _DEFAULTS["circuit_podium_rate_from_grid_pos"],
                "rain_probability": _mean(c_rain[circuit], _DEFAULTS["rain_probability"]),
                "tyre_degradation_level": _mean(c_tyre[circuit], _DEFAULTS["tyre_degradation_level"]),
                "driver_wet_performance": _mean(d_wet[did], _DEFAULTS["driver_wet_performance"]),
                "race_pace_last_5": _mean(d_pace[did], _DEFAULTS["race_pace_last_5"]),
            })

            feat["_index"] = row.Index
            feat_rows.append(feat)

        for row in grp.itertuples():
            did = row.driver_id
            tid = row.team_id
            circuit = row.circuit
            pos = float(row.position) if not pd.isna(row.position) else 20.0
            pts = float(row.points) if not pd.isna(row.points) else 0.0
            grid = float(row.grid) if not pd.isna(row.grid) else _DEFAULTS["grid"]
            grid_clean = grid if grid > 0 else _DEFAULTS["grid"]
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
            t_quali[tid].append(
                float(row.quali_pos) if not pd.isna(row.quali_pos) else grid_clean
            )
            t_dnf[tid].append(is_dnf)

            dc_finish[(did, circuit)].append(pos)
            dc_podium[(did, circuit)].append(is_pod)
            tc_finish[(tid, circuit)].append(pos)

            if not pd.isna(row.race_pace):
                d_pace[did].append(float(row.race_pace))
            if not pd.isna(row.rain) and float(row.rain) == 1.0:
                d_wet[did].append(pos)
            c_poschange[circuit].append(abs(grid_clean - pos))
            g_int = int(grid_clean)
            grid_pod_total[g_int] += 1.0
            grid_pod_hits[g_int] += is_pod

            season_pts[did] += pts
            season_pod[did] += is_pod
            season_win[did] += is_win
            season_races_d[did] += 1
            team_season_pts[tid] += pts

        first = grp.iloc[0]
        circuit0 = str(first["circuit"])
        c_rain[circuit0].append(
            1.0 if (pd.notna(first["rain"]) and float(first["rain"]) == 1.0) else 0.0
        )
        if pd.notna(first["sc_flag"]):
            c_sc[circuit0].append(float(first["sc_flag"]))
        if pd.notna(first["tyre_deg"]):
            c_tyre[circuit0].append(float(first["tyre_deg"]))
        for tid_g, sub in grp.groupby("team_id"):
            vals = sub["pit_stop_time"].dropna()
            if len(vals):
                t_pit[tid_g].append(float(vals.mean()))

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

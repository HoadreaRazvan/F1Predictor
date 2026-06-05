from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import config

warnings.filterwarnings("ignore")

_CACHE_ENABLED = False


def enable_cache() -> None:
    global _CACHE_ENABLED
    if _CACHE_ENABLED:
        return
    import fastf1

    fastf1.Cache.enable_cache(str(config.CACHE_DIR))
    try:
        fastf1.Cache.offline_mode(enabled=True)
    except Exception:
        pass
    try:
        fastf1.set_log_level("ERROR")
    except Exception:
        pass
    _CACHE_ENABLED = True


def _finished_flag(status: str) -> bool:
    if not isinstance(status, str):
        return False
    return status == "Finished" or status.startswith("+")


def _weather_summary(session) -> dict:
    out = {"rain": np.nan, "track_temp": np.nan, "air_temp": np.nan}
    try:
        w = session.weather_data
        if w is not None and len(w) > 0:
            out["rain"] = float(bool(w["Rainfall"].any()))
            out["track_temp"] = float(w["TrackTemp"].mean())
            out["air_temp"] = float(w["AirTemp"].mean())
    except Exception:
        pass
    return out


def _quali_positions(year: int, rnd: int) -> dict:
    import fastf1

    try:
        q = fastf1.get_session(year, rnd, "Q")
        q.load(laps=False, telemetry=False, weather=False, messages=False)
        res = q.results
        return {
            str(row.DriverId): float(row.Position)
            for row in res.itertuples()
            if pd.notna(row.Position)
        }
    except Exception:
        return {}


def load_round(year: int, rnd: int, event) -> pd.DataFrame | None:
    import fastf1

    try:
        race = fastf1.get_session(year, rnd, "R")
        race.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception as exc:
        print(f"  [skip] {year} runda {rnd}: {exc}")
        return None

    res = race.results
    if res is None or len(res) == 0:
        print(f"  [skip] {year} runda {rnd}: fără rezultate")
        return None

    weather = _weather_summary(race)
    quali = _quali_positions(year, rnd)

    event_format = str(event.get("EventFormat", "conventional"))
    rows = []
    for r in res.itertuples():
        did = str(r.DriverId)
        rows.append(
            {
                "season": int(year),
                "round": int(rnd),
                "event_name": str(event.get("EventName", "")),
                "circuit": str(event.get("Location", "")),
                "country": str(event.get("Country", "")),
                "event_date": pd.to_datetime(event.get("EventDate", pd.NaT)),
                "is_sprint_weekend": float(event_format != "conventional"),
                "driver": str(r.Abbreviation),
                "driver_id": did,
                "team": str(r.TeamName),
                "team_id": str(r.TeamId),
                "grid": float(r.GridPosition) if pd.notna(r.GridPosition) else np.nan,
                "quali_pos": quali.get(did, np.nan),
                "position": float(r.Position) if pd.notna(r.Position) else np.nan,
                "status": str(r.Status),
                "finished": float(_finished_flag(str(r.Status))),
                "points": float(r.Points) if pd.notna(r.Points) else 0.0,
                "rain": weather["rain"],
                "track_temp": weather["track_temp"],
                "air_temp": weather["air_temp"],
            }
        )
    return pd.DataFrame(rows)


def load_season(year: int) -> pd.DataFrame:
    import fastf1

    enable_cache()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    frames = []
    for ev in schedule.itertuples():
        rnd = int(ev.RoundNumber)
        if rnd < 1:
            continue
        event = {
            "EventName": ev.EventName,
            "Location": ev.Location,
            "Country": ev.Country,
            "EventDate": ev.EventDate,
            "EventFormat": ev.EventFormat,
        }
        df = load_round(year, rnd, event)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    season_df = pd.concat(frames, ignore_index=True)
    print(f"  {year}: {season_df['round'].nunique()} curse, {len(season_df)} rânduri")
    return season_df


def load_all(seasons=None) -> pd.DataFrame:
    seasons = list(seasons) if seasons is not None else list(config.SEASONS)
    enable_cache()
    frames = []
    for year in seasons:
        print(f"Încarc sezonul {year} ...")
        df = load_season(year)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("Niciun sezon încărcat din cache. Verifică data/cache.")
    return pd.concat(frames, ignore_index=True)

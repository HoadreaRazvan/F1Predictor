"""Acces de nivel jos la datele FastF1 — strict din cache (offline).

Pentru fiecare rundă încărcăm sesiunea de cursă (``R``) și calificările (``Q``)
și extragem un rând per pilot cu: grid, poziție finală, status, puncte, echipă,
plus context (circuit, dată, vreme). Totul vine din cache-ul existent — nicio
descărcare. Sesiunile lipsă (ex. anulate) sunt sărite cu avertizare.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import config

warnings.filterwarnings("ignore")

_CACHE_ENABLED = False
_OFFLINE = None


def enable_cache(offline: bool = True) -> None:
    """Activează cache-ul FastF1 și fixează modul offline.

    ``offline=True`` (implicit): folosește DOAR cache-ul local, fără nicio cerere de
    rețea — modul normal, reproductibil (evită revalidările care primesc 429 de la
    API-ul Ergast/Jolpica). ``offline=False``: permite descărcări; folosit O SINGURĂ
    DATĂ de pasul de îmbogățire (``extras.build_race_extras``).
    """
    global _CACHE_ENABLED, _OFFLINE
    if _CACHE_ENABLED and _OFFLINE == offline:
        return
    import fastf1

    if not _CACHE_ENABLED:
        fastf1.Cache.enable_cache(str(config.CACHE_DIR))
    try:
        fastf1.Cache.offline_mode(enabled=offline)
    except Exception:
        pass
    try:
        fastf1.set_log_level("ERROR")
    except Exception:
        pass
    _CACHE_ENABLED = True
    _OFFLINE = offline


def _finished_flag(status: str) -> bool:
    """True dacă pilotul a fost clasat la final (a terminat sau a fost lapped)."""
    if not isinstance(status, str):
        return False
    return status == "Finished" or status.startswith("+")


def _weather_summary(session) -> dict:
    """Rezumat de vreme la nivel de cursă (din weather_data cache-uit)."""
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


def _quali_info(year: int, rnd: int) -> dict:
    """Mapă DriverId -> {"pos": poziție, "best": cel mai bun timp (s)} din sesiunea Q.

    Timpul cel mai bun e minimul dintre Q1/Q2/Q3 prezente (în secunde); ``nan`` dacă
    pilotul nu are niciun tur cronometrat. {} dacă sesiunea lipsește din cache.
    Datele Q1/Q2/Q3 vin din rezultatele Ergast deja cache-uite (fără descărcare).
    """
    import fastf1

    try:
        q = fastf1.get_session(year, rnd, "Q")
        q.load(laps=False, telemetry=False, weather=False, messages=False)
        res = q.results
    except Exception:
        return {}

    out = {}
    for row in res.itertuples():
        if pd.isna(getattr(row, "Position", np.nan)):
            continue
        times = []
        for q_col in ("Q1", "Q2", "Q3"):
            t = getattr(row, q_col, None)
            if t is not None and pd.notna(t):
                secs = pd.Timedelta(t).total_seconds()
                if secs > 0:
                    times.append(secs)
        out[str(row.DriverId)] = {
            "pos": float(row.Position),
            "best": float(min(times)) if times else np.nan,
        }
    return out


def load_round(year: int, rnd: int, event) -> pd.DataFrame | None:
    """Încarcă o rundă -> DataFrame cu un rând per pilot, sau None dacă indisponibil."""
    import fastf1

    try:
        race = fastf1.get_session(year, rnd, "R")
        race.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception as exc:  # sesiune lipsă din cache / anulată
        print(f"  [skip] {year} runda {rnd}: {exc}")
        return None

    res = race.results
    if res is None or len(res) == 0:
        print(f"  [skip] {year} runda {rnd}: fără rezultate")
        return None

    weather = _weather_summary(race)
    quali = _quali_info(year, rnd)

    event_format = str(event.get("EventFormat", "conventional"))
    rows = []
    for r in res.itertuples():
        did = str(r.DriverId)
        qinfo = quali.get(did)
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
                "quali_pos": qinfo["pos"] if qinfo else np.nan,
                "quali_best": qinfo["best"] if qinfo else np.nan,
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
    """Încarcă toate rundele unui sezon -> DataFrame concatenat."""
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
    """Încarcă toate sezoanele cerute (implicit cele din config)."""
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

"""Îmbogățiri descărcate O SINGURĂ DATĂ din FastF1 (tururi, pit-stops, mesaje).

Pipeline-ul normal rulează strict offline din cache (vezi ``loader.py``), iar
folderele de cursă din cache conțin doar rezultate + vreme. Pentru câteva
caracteristici avem nevoie de date mai bogate, care nu sunt în cache-ul de bază:

  * ``pit_stop_time`` — durata medie a opririlor la boxe (din Ergast/Jolpica);
  * ``race_pace``     — ritmul de cursă al pilotului (median tururi verzi, normalizat);
  * ``sc_flag``       — dacă în cursă a apărut Safety Car / VSC (din mesajele de control);
  * ``tyre_deg``      — degradarea pneurilor (panta timpului pe tur vs. vârsta pneului).

Acest modul le extrage cu rețeaua TEMPORAR activată și salvează rezultatul în
``outputs/race_extras.parquet`` (o linie per sezon/rundă/pilot). După această
descărcare unică, restul proiectului rămâne offline și reproductibil, citind acel
fișier prin :func:`get_race_extras`.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from .. import config
from . import loader

warnings.filterwarnings("ignore")

# Coloanele fișierului de îmbogățiri (în această ordine).
EXTRAS_COLUMNS = [
    "season",
    "round",
    "driver_id",
    "team_id",
    "pit_stop_time",  # per pilot (s)
    "race_pace",      # per pilot (>=1, normalizat pe circuit)
    "sc_flag",        # per cursă (0/1)
    "tyre_deg",       # per cursă (s/tur)
]


# --------------------------------------------------------------------- pit-stops
def _pit_stop_times(year: int, rnd: int) -> dict:
    """DriverId -> durata mediană a opririlor la boxe (s), din Ergast/Jolpica."""
    try:
        from fastf1.ergast import Ergast

        resp = Ergast().get_pit_stops(season=year, round=rnd, limit=1000)
        content = resp.content
    except Exception:
        return {}
    if not content:
        return {}
    df = content[0]
    if df is None or len(df) == 0 or "duration" not in df or "driverId" not in df:
        return {}

    secs = df["duration"]
    if np.issubdtype(np.asarray(secs).dtype, np.timedelta64):
        secs = pd.to_timedelta(secs).dt.total_seconds()
    secs = pd.to_numeric(secs, errors="coerce")

    work = pd.DataFrame({"driver_id": df["driverId"].astype(str), "sec": secs}).dropna()
    # Filtru defensiv: elimină valori absurde (drive-through, erori de raportare).
    work = work[(work["sec"] >= 1.5) & (work["sec"] <= 60.0)]
    if work.empty:
        return {}
    return {did: float(g["sec"].median()) for did, g in work.groupby("driver_id")}


# -------------------------------------------------------------------- safety car
def _safety_car_flag(session, laps) -> float:
    """1.0 dacă în cursă a fost Safety Car / VSC, altfel 0.0 (nan dacă nu se poate)."""
    try:
        msg = session.race_control_messages
    except Exception:
        msg = None

    if msg is not None and len(msg) > 0:
        hit = False
        if "Message" in msg:
            up = msg["Message"].astype(str).str.upper()
            hit = bool(up.str.contains("SAFETY CAR").any() or up.str.contains("VSC").any())
        if not hit and "Category" in msg:
            hit = bool(msg["Category"].astype(str).str.contains("SafetyCar", case=False).any())
        return 1.0 if hit else 0.0

    # Fallback: codurile de status al pistei din tururi (4 = SC, 6/7 = VSC).
    if laps is not None and len(laps) > 0 and "TrackStatus" in laps:
        ts = laps["TrackStatus"].astype(str)
        if ts.str.contains("4").any() or ts.str.contains("6").any() or ts.str.contains("7").any():
            return 1.0
        return 0.0
    return np.nan


# --------------------------------------------------------------- tururi „curate"
def _green_laps(laps) -> pd.DataFrame | None:
    """Tururi verzi (status pistă „1"), fără intrare/ieșire din boxe, cu timp valid."""
    if laps is None or len(laps) == 0 or "LapTime" not in laps:
        return None
    secs = laps["LapTime"].dt.total_seconds()
    green = laps["TrackStatus"].astype(str) == "1"
    nopit = laps["PitInTime"].isna() & laps["PitOutTime"].isna()
    mask = secs.notna() & green & nopit & (secs > 0)
    if not mask.any():
        return None
    sub = laps[mask].copy()
    sub["_sec"] = secs[mask].to_numpy()
    return sub


# --------------------------------------------------------------------- race pace
def _race_pace(laps, abbr2id: dict) -> dict:
    """DriverId -> ritm de cursă normalizat: median(pilot) / cuantila 0.10 a plutonului.

    Normalizarea face valorile comparabile între circuite (independent de lungime):
    1.0 = pe ritmul de top al plutonului, valori >1 = mai lent.
    """
    sub = _green_laps(laps)
    if sub is None:
        return {}
    ref = float(sub["_sec"].quantile(0.10))
    if ref <= 0:
        return {}
    out = {}
    for drv, g in sub.groupby("Driver"):
        did = abbr2id.get(str(drv))
        if did is None:
            continue
        out[did] = float(g["_sec"].median() / ref)
    return out


# ------------------------------------------------------------- tyre degradation
def _tyre_degradation(laps) -> float:
    """Degradare la nivel de cursă: median al pantelor timp/tur vs. vârsta pneului.

    Pentru fiecare stint (>=5 tururi verzi) se estimează panta liniară a timpului pe
    tur în funcție de ``TyreLife``; rezultatul este mediana pantelor (s/tur).
    """
    sub = _green_laps(laps)
    if sub is None or "TyreLife" not in sub or "Stint" not in sub:
        return np.nan
    sub = sub[sub["TyreLife"].notna() & sub["Stint"].notna()]
    slopes = []
    for _, g in sub.groupby(["Driver", "Stint"]):
        if len(g) < 5:
            continue
        x = g["TyreLife"].astype(float).to_numpy()
        y = g["_sec"].to_numpy()
        if np.ptp(x) < 1:
            continue
        try:
            slope = float(np.polyfit(x, y, 1)[0])
        except Exception:
            continue
        if -0.5 <= slope <= 2.0:  # clamp defensiv (s/tur plauzibil)
            slopes.append(slope)
    return float(np.median(slopes)) if slopes else np.nan


# ---------------------------------------------------------------- extracție rundă
def _extract_round(year: int, rnd: int) -> list[dict]:
    """Extrage îmbogățirile unei runde -> listă de rânduri (un rând per pilot)."""
    import fastf1

    try:
        race = fastf1.get_session(year, rnd, "R")
        race.load(laps=True, telemetry=False, weather=False, messages=True)
    except Exception as exc:
        print(f"  [skip] {year} runda {rnd}: {exc}")
        return []

    res = race.results
    if res is None or len(res) == 0:
        return []

    abbr2id, id2team = {}, {}
    for r in res.itertuples():
        did = str(r.DriverId)
        abbr2id[str(r.Abbreviation)] = did
        id2team[did] = str(r.TeamId)

    try:
        laps = race.laps
    except Exception:
        laps = None

    pace = _race_pace(laps, abbr2id)
    tyre = _tyre_degradation(laps)
    sc = _safety_car_flag(race, laps)
    pits = _pit_stop_times(year, rnd)

    rows = []
    for r in res.itertuples():
        did = str(r.DriverId)
        rows.append(
            {
                "season": int(year),
                "round": int(rnd),
                "driver_id": did,
                "team_id": id2team.get(did, str(r.TeamId)),
                "pit_stop_time": pits.get(did, np.nan),
                "race_pace": pace.get(did, np.nan),
                "sc_flag": sc,
                "tyre_deg": tyre,
            }
        )
    return rows


# ------------------------------------------------------------------ API publică
def build_race_extras(seasons=None, *, throttle: float = 0.5, save: bool = True,
                      force: bool = False) -> pd.DataFrame:
    """Descarcă (cu rețea) îmbogățirile pentru toate sezoanele și le salvează în parquet.

    Procesul este \emph{incremental și resumabil}: rundele deja prezente în parquet sunt
    sărite, iar progresul este salvat după fiecare sezon. Astfel, dacă API-ul public
    impune o limită de rată (eng. \emph{rate limiting}), e suficient să se reia comanda
    după resetarea limitei — rundele descărcate anterior nu se mai cer din nou.
    Rundele indisponibile rămân fără rând și primesc valori implicite în aval.
    """
    import fastf1

    seasons = list(seasons) if seasons is not None else list(config.SEASONS)
    config.ensure_dirs()

    # Pornim de la ce avem deja (dacă nu se forțează o reconstrucție completă).
    existing = pd.DataFrame(columns=EXTRAS_COLUMNS)
    if not force and config.RACE_EXTRAS_PATH.exists():
        try:
            existing = pd.read_parquet(config.RACE_EXTRAS_PATH)
        except Exception:
            existing = pd.DataFrame(columns=EXTRAS_COLUMNS)
    done = set(zip(existing["season"].astype(int), existing["round"].astype(int))) \
        if len(existing) else set()

    loader.enable_cache(offline=False)  # PERMITE descărcări — doar aici!

    new_rows: list[dict] = []

    def _persist():
        if not save:
            return
        parts = [existing] if len(existing) else []
        if new_rows:
            parts.append(pd.DataFrame(new_rows, columns=EXTRAS_COLUMNS))
        if parts:
            out = pd.concat(parts, ignore_index=True)
            out = out.drop_duplicates(subset=["season", "round", "driver_id"], keep="last")
            out.to_parquet(config.RACE_EXTRAS_PATH, index=False)

    for year in seasons:
        print(f"Îmbogățesc sezonul {year} ...")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            print(f"  [skip] program {year}: {exc}")
            continue
        n_before = len(new_rows)
        for ev in schedule.itertuples():
            rnd = int(ev.RoundNumber)
            if rnd < 1 or (int(year), rnd) in done:
                continue
            r = _extract_round(year, rnd)
            new_rows.extend(r)
            if throttle:
                time.sleep(throttle)
        print(f"  {year}: {len(new_rows) - n_before} rânduri noi")
        _persist()  # salvăm progresul după fiecare sezon (rezistent la limita de rată)

    parts = [existing] if len(existing) else []
    if new_rows:
        parts.append(pd.DataFrame(new_rows, columns=EXTRAS_COLUMNS))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EXTRAS_COLUMNS)
    df = df.drop_duplicates(subset=["season", "round", "driver_id"], keep="last")
    if save and not df.empty:
        df.to_parquet(config.RACE_EXTRAS_PATH, index=False)
        print(f"Salvat: {config.RACE_EXTRAS_PATH}  ({len(df)} rânduri)")
    return df


def get_race_extras() -> pd.DataFrame:
    """Citește îmbogățirile (offline). Frame gol corect tipat dacă fișierul lipsește."""
    if config.RACE_EXTRAS_PATH.exists():
        try:
            return pd.read_parquet(config.RACE_EXTRAS_PATH)
        except Exception:
            pass
    empty = {c: pd.Series(dtype="object" if c in ("driver_id", "team_id") else "float64")
             for c in EXTRAS_COLUMNS}
    return pd.DataFrame(empty)

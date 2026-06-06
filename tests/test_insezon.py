import numpy as np
import pandas as pd

from f1pred.cli import _fit_before, _race_metrics
from f1pred.features.engineering import add_features

_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8}


def _toy_feats(seasons=(2020, 2021, 2022), rounds=4, n_drivers=6, seed=0):
    rng = np.random.default_rng(seed)
    drivers = [f"D{i}" for i in range(n_drivers)]
    rows = []
    for season in seasons:
        for rnd in range(1, rounds + 1):
            order = np.argsort(np.arange(n_drivers) + rng.normal(scale=0.5, size=n_drivers))
            for pos, di in enumerate(order, start=1):
                drv = drivers[di]
                rows.append(
                    {
                        "season": season, "round": rnd,
                        "event_name": f"GP{rnd}", "circuit": f"C{rnd}", "country": "X",
                        "event_date": pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=14 * rnd),
                        "is_sprint_weekend": 0.0,
                        "driver": drv, "driver_id": drv,
                        "team": f"T{di // 2}", "team_id": f"T{di // 2}",
                        "grid": float(pos), "quali_pos": float(pos),
                        "position": float(pos), "status": "Finished", "finished": 1.0,
                        "points": float(_POINTS.get(pos, 0)),
                        "rain": 0.0, "track_temp": 30.0, "air_temp": 22.0,
                    }
                )
    return add_features(pd.DataFrame(rows))


def test_augmented_uses_more_data_than_baseline():
    feats = _toy_feats()
    _, n_base = _fit_before(feats, "tree", 2022)
    rounds = sorted(int(r) for r in feats[feats["season"] == 2022]["round"].unique())
    _, n_aug = _fit_before(feats, "tree", 2022, upto_round=rounds[-1])
    assert n_aug > n_base
    _, n_first = _fit_before(feats, "tree", 2022, upto_round=rounds[0])
    assert n_first == n_base


def test_race_metrics_keys_and_ranges():
    feats = _toy_feats()
    model, _ = _fit_before(feats, "tree", 2022)
    race = feats[(feats["season"] == 2022) & (feats["round"] == 1)]
    m = _race_metrics(model, race)
    assert set(m) == {"accuracy", "podium_hit_rate", "exact_podium_set", "winner_acc"}
    for v in m.values():
        assert 0.0 <= v <= 1.0

import numpy as np
import pandas as pd

from f1pred.evaluation import tuning
from f1pred.features.engineering import add_features

_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8}

def _toy_feats(seasons=(2019, 2020, 2021, 2022), rounds=4, n_drivers=6, seed=0):
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

def test_iter_configs_is_cartesian_product():
    grid = dict(a=[1, 2, 3], b=["x", "y"])
    cfgs = tuning.iter_configs(grid)
    assert len(cfgs) == 6
    assert set(cfgs[0]) == {"a", "b"}

    assert len({tuple(sorted(c.items())) for c in cfgs}) == 6

def test_tune_model_columns_and_sorted_by_validation():
    feats = _toy_feats()
    grid = dict(max_depth=[3, 6], min_samples_leaf=[2, 5])
    df = tuning.tune_model(feats, "tree", grid, metric="podium_hit_rate", verbose=False)

    assert len(df) >= 4
    for col in ("val_podium_hit_rate", "test_podium_hit_rate", "max_depth",
                "min_samples_leaf", "fit_seconds", "is_default"):
        assert col in df.columns

    v = df["val_podium_hit_rate"].to_numpy()
    assert np.all(v[:-1] >= v[1:] - 1e-9)

def test_best_config_selected_on_validation():
    feats = _toy_feats()
    grid = dict(max_depth=[3, 6], min_samples_leaf=[2, 5])
    df = tuning.tune_model(feats, "tree", grid, metric="podium_hit_rate", verbose=False)
    bc = tuning.best_config(df, "podium_hit_rate", list(grid.keys()))

    assert set(bc.keys()) == {"max_depth", "min_samples_leaf"}

    assert df["val_podium_hit_rate"].iloc[0] == df["val_podium_hit_rate"].max()

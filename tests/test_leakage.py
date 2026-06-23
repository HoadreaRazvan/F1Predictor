import pandas as pd

from f1pred.features.engineering import add_features

def _mini_long():
    rows = []
    for rnd in (1, 2, 3):
        for drv, pos, pts in (("AAA", 1, 25.0), ("BBB", 2, 18.0)):
            rows.append(
                {
                    "season": 2030,
                    "round": rnd,
                    "event_name": f"GP{rnd}",
                    "circuit": "Testville",
                    "country": "TestLand",
                    "event_date": pd.Timestamp("2030-01-01") + pd.Timedelta(days=14 * rnd),
                    "is_sprint_weekend": 0.0,
                    "driver": drv,
                    "driver_id": drv,
                    "team": "T_" + drv,
                    "team_id": "T_" + drv,
                    "grid": float(pos),
                    "quali_pos": float(pos),
                    "position": float(pos),
                    "status": "Finished",
                    "finished": 1.0,
                    "points": pts,
                    "rain": 0.0,
                    "track_temp": 30.0,
                    "air_temp": 22.0,
                }
            )
    return pd.DataFrame(rows)

def test_no_leakage_first_race_has_defaults():
    feats = add_features(_mini_long())
    r1 = feats[feats["round"] == 1]

    assert (r1["career_races"] == 0).all()
    assert (r1["season_points"] == 0).all()
    assert (r1["form_avg_finish"] == 15.0).all()

def test_season_points_exclude_current_race():
    feats = add_features(_mini_long())
    aaa = feats[feats["driver"] == "AAA"].sort_values("round")

    assert list(aaa["season_points"]) == [0.0, 25.0, 50.0]

def test_form_uses_only_past():
    feats = add_features(_mini_long())
    aaa = feats[feats["driver"] == "AAA"].sort_values("round")

    assert aaa.iloc[1]["form_avg_finish"] == 1.0
    assert aaa.iloc[2]["form_avg_finish"] == 1.0

    assert (feats["is_podium"] == 1.0).all()

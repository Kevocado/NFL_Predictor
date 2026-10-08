import numpy as np
import pandas as pd
from nfl_predictor.features.conditions import add_condition_features, CONDITION_COLUMNS
from nfl_predictor.features import build


def _games(**kw):
    base = {"game_id": "g1", "roof": "outdoors", "temp": 30.0, "wind": 22.0}
    base.update(kw)
    return pd.DataFrame([base])


def test_cold_windy_outdoor_game():
    row = add_condition_features(_games()).iloc[0]
    assert row["wx_known"] == 1 and row["wx_dome"] == 0
    assert row["wx_wind_mph"] == 22.0 and row["wx_cold"] == 1 and row["wx_windy"] == 1


def test_dome_neutralises_weather():
    row = add_condition_features(_games(roof="dome", temp=20.0, wind=30.0)).iloc[0]
    assert row["wx_dome"] == 1 and row["wx_wind_mph"] == 0.0 and row["wx_cold"] == 0 and row["wx_windy"] == 0


def test_missing_weather_is_flagged_not_zeroed():
    row = add_condition_features(_games(temp=None, wind=None)).iloc[0]
    assert row["wx_known"] == 0 and row["wx_wind_mph"] == 0.0 and row["wx_cold"] == 0


def test_columns_declared():
    assert set(CONDITION_COLUMNS) <= set(add_condition_features(_games()).columns)


def _season_games(seed=3):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(4)]
    rows = []
    for season in (2023, 2024):
        for week in range(1, 5):
            order = list(rng.permutation(teams))
            for i in range(0, 4, 2):
                home, away = order[i], order[i + 1]
                rows.append({
                    "game_id": f"{season}_{week:02d}_{home}_{away}", "season": season, "week": week,
                    "gameday": pd.Timestamp(f"{season}-09-07") + pd.Timedelta(days=7 * (week - 1)),
                    "home_team": home, "away_team": away,
                    "home_score": int(rng.integers(6, 42)), "away_score": int(rng.integers(6, 42)),
                    "div_game": 0, "roof": "outdoors", "temp": 30.0, "wind": 10.0,
                })
    return pd.DataFrame(rows)


def test_served_conditions_wx_known_when_schedule_has_conditions():
    """Schedule provides roof/temp/wind → wx_known=1 at serving."""
    games = _season_games()
    when = pd.Timestamp("2025-09-07")
    history = games[pd.to_datetime(games["gameday"]) < when]
    # The upcoming game's schedule data
    game_schedule = {"roof": "outdoors", "temp": 30.0, "wind": 10.0}
    served = build.build_features_for_game("T0", "T1", history, gameday=when, blocks=("conditions",), game_schedule=game_schedule)
    assert served["wx_known"] == 1.0
    assert served["wx_wind_mph"] == 10.0


def test_served_conditions_wx_known_zero_when_schedule_missing():
    """Schedule missing roof/temp/wind → wx_known=0 at serving."""
    games = _season_games()
    when = pd.Timestamp("2025-09-07")
    history = games[pd.to_datetime(games["gameday"]) < when].copy()
    # Drop conditions columns from history to simulate missing schedule data
    history = history.drop(columns=["roof", "temp", "wind"])
    # No game_schedule passed → should fall back to NaN
    served = build.build_features_for_game("T0", "T1", history, gameday=when, blocks=("conditions",))
    assert served["wx_known"] == 0.0
    assert served["wx_wind_mph"] == 0.0
import pandas as pd
from nfl_predictor.features.conditions import add_condition_features, CONDITION_COLUMNS


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
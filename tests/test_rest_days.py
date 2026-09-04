import pandas as pd

from nfl_predictor.features import rest_days


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC"},
            {"game_id": "g2", "gameday": "2025-09-11", "home_team": "KC", "away_team": "CIN"},
            {"game_id": "g3", "gameday": "2025-09-25", "home_team": "BAL", "away_team": "CIN"},
        ]
    )


def test_first_appearance_has_no_rest_days():
    result = rest_days.add_rest_days(_games())

    assert pd.isna(result.iloc[0]["home_rest_days"])
    assert pd.isna(result.iloc[0]["away_rest_days"])


def test_rest_days_counts_days_since_last_game():
    result = rest_days.add_rest_days(_games())

    g2 = result.iloc[1]
    assert g2["home_rest_days"] == 7  # KC away in g1 (09-04) -> home in g2 (09-11)

    g3 = result.iloc[2]
    assert g3["home_rest_days"] == 21  # BAL home in g1 (09-04) -> home in g3 (09-25), bye week

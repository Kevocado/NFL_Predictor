import pandas as pd

from nfl_predictor.features import power_ratings


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "season": 2025, "week": 1, "gameday": "2025-09-04",
             "home_team": "BAL", "away_team": "KC", "home_score": 27, "away_score": 20},
            {"game_id": "g2", "season": 2025, "week": 2, "gameday": "2025-09-11",
             "home_team": "KC", "away_team": "BAL", "home_score": 17, "away_score": 24},
        ]
    )


def test_first_game_uses_start_rating_for_both_teams():
    result = power_ratings.compute_pregame_ratings(_games(), start_rating=1500.0)

    first = result.iloc[0]
    assert first["home_pregame_rating"] == 1500.0
    assert first["away_pregame_rating"] == 1500.0


def test_rating_moves_after_a_result():
    result = power_ratings.compute_pregame_ratings(_games(), start_rating=1500.0)

    second = result.iloc[1]
    # KC lost game 1 as the away team, so KC's pregame rating for game 2
    # (now at home) should have dropped below 1500.
    assert second["home_pregame_rating"] < 1500.0
    # BAL won game 1, so BAL's pregame rating for game 2 (now away) should
    # have risen above 1500.
    assert second["away_pregame_rating"] > 1500.0


def test_final_ratings_reflects_every_game():
    ratings = power_ratings.final_ratings(_games(), start_rating=1500.0)

    assert set(ratings) == {"BAL", "KC"}
    # BAL won both meetings on net score margin, should end above start.
    assert ratings["BAL"] > 1500.0
    assert ratings["KC"] < 1500.0

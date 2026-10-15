import pandas as pd

from nfl_predictor.features.player_usage import recent_averages


def _hist():
    rows = [("p1", 2026, w, y, 0.0) for w, y in zip(range(1, 8), (100, 200, 300, 50, 60, 70, 80))]
    rows += [("p1", 2025, 17, 999, 0.0), ("p2", 2026, 1, 5, 0.0)]
    return pd.DataFrame(rows, columns=["player_id", "season", "week", "passing_yards", "rushing_yards"])


def test_is_the_plain_mean_of_the_last_five_games_before_the_week_including_the_latest():
    out = recent_averages("p1", _hist(), 2026, 8)
    assert out["games"] == 5 and out["passing_yards"] == round((300 + 50 + 60 + 70 + 80) / 5, 1)


def test_never_uses_the_game_being_predicted_or_later_or_another_season():
    out = recent_averages("p1", _hist(), 2026, 4)
    assert out["games"] == 3 and out["passing_yards"] == 200.0   # weeks 1-3 only; the 999 is last season


def test_a_qb_rushing_average_mirrors_rushing_yards_and_unknown_players_are_empty():
    assert recent_averages("p1", _hist(), 2026, 8)["qb_rushing_yards"] == 0.0
    assert recent_averages("nobody", _hist(), 2026, 8) == {}

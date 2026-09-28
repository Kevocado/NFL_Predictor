import pandas as pd



def test_div_game_is_read_from_the_target_season_not_the_oldest_matchup():
    """A regression guard for a lookup that was right only for one season.

    `routes._load_game_history` concatenates eight completed seasons and
    `schedules.fetch_schedules` sorts oldest first, so taking the first
    home/away pair on record read a *prior season's* value. A 2026 game whose
    true `div_game` is 0 was served 1.
    """
    from nfl_predictor.features.build import _div_game_for

    games = pd.DataFrame([
        {"season": 2019, "week": 5, "home_team": "BUF", "away_team": "MIA", "div_game": 1},
        {"season": 2021, "week": 6, "home_team": "BUF", "away_team": "MIA", "div_game": 0},
        {"season": 2026, "week": 3, "home_team": "BUF", "away_team": "MIA", "div_game": 0},
    ])
    assert _div_game_for(games, "BUF", "MIA") == 0


def test_div_game_still_reads_a_division_game_in_the_target_season():
    from nfl_predictor.features.build import _div_game_for

    games = pd.DataFrame([
        {"season": 2019, "week": 5, "home_team": "BUF", "away_team": "MIA", "div_game": 0},
        {"season": 2026, "week": 3, "home_team": "BUF", "away_team": "MIA", "div_game": 1},
    ])
    assert _div_game_for(games, "BUF", "MIA") == 1


def test_div_game_takes_the_latest_meeting_inside_the_target_season():
    """Two meetings, same season: the one being predicted is the later week."""
    from nfl_predictor.features.build import _div_game_for

    games = pd.DataFrame([
        {"season": 2025, "week": 1, "home_team": "BUF", "away_team": "MIA", "div_game": 0},
        {"season": 2025, "week": 18, "home_team": "BUF", "away_team": "MIA", "div_game": 1},
    ])
    assert _div_game_for(games, "BUF", "MIA") == 1


def test_div_game_falls_back_to_zero_without_the_column():
    from nfl_predictor.features.build import _div_game_for

    assert _div_game_for(pd.DataFrame([{"season": 2026, "home_team": "BUF", "away_team": "MIA"}]), "BUF", "MIA") == 0
    assert _div_game_for(
        pd.DataFrame([{"season": 2026, "home_team": "BUF", "away_team": "MIA", "div_game": float("nan")}]),
        "BUF", "MIA",
    ) == 0

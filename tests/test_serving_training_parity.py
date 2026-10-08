"""A game's served feature row equals the row training builds for the same game (same code path, same values)."""
import numpy as np
import pandas as pd
import pytest

from nfl_predictor.features import build


def _season_games(seed=3, seasons=(2023, 2024, 2025), weeks=14):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            order = list(rng.permutation(teams))
            n_games = 3 if week % 4 == 0 else 4          # a bye week for some teams every fourth week
            for i in range(0, 2 * n_games, 2):
                home, away = order[i], order[i + 1]
                rows.append({
                    "game_id": f"{season}_{week:02d}_{home}_{away}", "season": season, "week": week,
                    "gameday": pd.Timestamp(f"{season}-09-07") + pd.Timedelta(days=7 * (week - 1)) + pd.Timedelta(days=int(rng.choice([0, 0, 0, 4]))),
                    "home_team": home, "away_team": away,
                    "home_score": int(rng.integers(6, 42)), "away_score": int(rng.integers(6, 42)), "div_game": 0,
                })
    return pd.DataFrame(rows)


def test_the_served_row_equals_the_row_training_builds_for_the_same_game():
    games = _season_games()
    trained, cols = build.build_training_frame(games)
    sample = trained[trained["season"] == 2025].sample(15, random_state=1)
    for _, g in sample.iterrows():
        history = games[pd.to_datetime(games["gameday"]) < pd.Timestamp(g["gameday"])]
        served = build.build_features_for_game(g["home_team"], g["away_team"], history, gameday=g["gameday"])
        np.testing.assert_allclose(served[cols].to_numpy(float), g[cols].to_numpy(float), equal_nan=True, err_msg=g["game_id"])


def test_rest_days_are_measured_to_the_games_own_date_not_to_today():
    games = _season_games()
    last = pd.to_datetime(games["gameday"]).max()
    when = last + pd.Timedelta(days=9)
    team = games.sort_values("gameday").iloc[-1]["home_team"]
    served = build.build_features_for_game(team, "T7" if team != "T7" else "T6", games, gameday=when)
    last_played = pd.to_datetime(games[(games["home_team"] == team) | (games["away_team"] == team)]["gameday"]).max()
    assert served["home_rest_days"] == (when - last_played).days


def test_unplayed_rows_in_the_history_do_not_enter_the_form_or_the_ratings():
    games = _season_games()
    future = games.iloc[[-1]].copy()
    future["game_id"], future["home_score"], future["away_score"] = "future", np.nan, np.nan
    future["gameday"] = pd.Timestamp("2026-01-04")
    with_future = pd.concat([games, future], ignore_index=True)
    a = build.build_features_for_game("T0", "T1", games, gameday="2026-01-11")
    b = build.build_features_for_game("T0", "T1", with_future, gameday="2026-01-11")
    pd.testing.assert_series_equal(a, b)


def test_the_row_has_exactly_the_feature_columns_in_order():
    games = _season_games()
    assert list(build.build_features_for_game("T0", "T1", games, gameday="2026-01-11").index) == build.FEATURE_COLUMNS
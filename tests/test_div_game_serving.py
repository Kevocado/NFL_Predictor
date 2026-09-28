"""`div_game` is a modelled feature that was served as a constant.

`features/build.py::FEATURE_COLUMNS` includes `div_game`, and
`build_training_frame` fills it from the schedule — it toggles for roughly a
third of games. `build_features_for_game` hardcoded it to 0, so the model was
fitted with a covariate that varied and served with it permanently constant, and
the fitted coefficient was dead weight at inference.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.features.build import build_features_for_game


def _games(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["game_id", "home_team", "away_team", "home_score", "away_score", "gameday", "div_game"],
    )


def _played(home: str, away: str, day: str, div: int) -> dict:
    return {
        "game_id": f"{home}-{away}",
        "home_team": home,
        "away_team": away,
        "home_score": 24,
        "away_score": 17,
        "gameday": day,
        "div_game": div,
    }


def test_div_game_is_read_from_the_schedule_for_a_division_matchup():
    games = _games([_played("BUF", "MIA", "2026-09-10", 1)])
    row = build_features_for_game("BUF", "MIA", games)
    assert row["div_game"] == 1


def test_div_game_is_zero_for_a_non_division_matchup():
    games = _games([_played("BUF", "MIA", "2026-09-10", 1), _played("BUF", "NYJ", "2026-09-17", 0)])
    row = build_features_for_game("BUF", "NYJ", games)
    assert row["div_game"] == 0


def test_div_game_defaults_to_zero_when_the_schedule_lacks_the_column():
    """A games frame without div_game must still build, matching the training default."""
    games = _games([_played("BUF", "MIA", "2026-09-10", 1)]).drop(columns=["div_game"])
    row = build_features_for_game("BUF", "MIA", games)
    assert row["div_game"] == 0


def test_div_game_defaults_to_zero_when_the_matchup_is_absent_from_the_frame():
    """Season-projection style calls pass a frame that may not contain this game."""
    games = _games([_played("BUF", "MIA", "2026-09-10", 1)])
    row = build_features_for_game("SF", "SEA", games)
    assert row["div_game"] == 0


def test_div_game_lands_in_the_returned_feature_row():
    """The value must be present and correct, whatever dtype the Series promotes to.

    `pd.Series({...})` over a mixed float/int dict yields float64, and XGBoost
    coerces at the model boundary regardless, so the dtype is not what is worth
    pinning -- the value reaching the feature row is.
    """
    games = _games([_played("BUF", "MIA", "2026-09-10", 1)])
    row = build_features_for_game("BUF", "MIA", games)
    assert row["div_game"] == 1
    assert "div_game" in build_features_for_game("SF", "SEA", games).index

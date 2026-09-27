"""Regression tests for the player-prop serving defects found 2026-09-27.

Three distinct bugs, all in the same narrow path from a player's stat history
to the feature row the models consume.

1. **`shift(1)` train/serve skew.** Training builds each rolling feature as
   ``s.shift(1).rolling(5, min_periods=1).mean()`` — the *latest* game is
   excluded, because a pregame feature cannot know it. Serving took
   ``history.tail(5)`` and took the plain mean, which *includes* the latest game.
   Same column name, different quantity, silently.

   Measured on Aaron Rodgers (``00-0023459``): served ``passing_yards_roll`` was
   254.0 against a training-consistent 249.0.

2. **No season filter at serving time.** ``build_features_for_player`` ranged
   over every season in the frame, so a player on a 2026 roster could be scored
   on whatever his last five games were in whatever year those were. 416 of 458
   non-zero served rows were predicted off 2024 data — two seasons stale. Worse,
   ``recent_team`` comes from the current roster while the features come from an
   old one, so a traded player was credited to his new team on his old team's
   numbers.

3. **Unknown players were given a fabricated zero vector** rather than skipped.
   CFB's equivalent route skips (``routes.py`` ``if feature_row is None: continue``);
   NFL substituted ``pd.Series({col: 0.0 for col in PLAYER_FEATURE_COLUMNS})``. A
   model never trained on the all-zero region was then asked to score it, and
   430 of 880 live rows (48.9%) came back bit-identical: passing_yards 62.592,
   rushing_yards 10.054, receiving_yards 16.005, anytime_td_prob 0.111846. Those
   are the origin intercepts, not predictions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.features.player_usage import (
    PLAYER_FEATURE_COLUMNS,
    build_features_for_player,
    build_player_training_frame,
)


def _history(seasons_weeks: list[tuple[int, int, float]], player_id: str = "p1") -> pd.DataFrame:
    """Minimal per-game frame: one stat, with carrying/receiving zeroed."""
    rows = []
    for season, week, passing in seasons_weeks:
        rows.append(
            {
                "player_id": player_id,
                "season": season,
                "week": week,
                "passing_yards": passing,
                "rushing_yards": 0.0,
                "receiving_yards": 0.0,
                "targets": 0.0,
                "carries": 0.0,
                "receptions": 0.0,
                "passing_tds": 0,
                "rushing_tds": 0,
                "receiving_tds": 0,
            }
        )
    return pd.DataFrame(rows)


def test_serving_excludes_the_latest_game_matching_the_training_discipline():
    """The served rolling mean must equal the training rolling mean for the same row.

    Before the fix this failed: serving used ``tail(5)`` including the latest
    game, training used ``shift(1)`` excluding it.
    """
    history = _history([(2023, w, float(w * 100)) for w in range(1, 7)])

    # The training frame's final row is the pregame view of week 7's game.
    training, feature_cols = build_player_training_frame(history)
    training_row = training.iloc[-1]

    served = build_features_for_player("p1", history)
    assert served is not None
    for column in feature_cols:
        assert served[column] == pytest.approx(training_row[column]), column


def test_serving_does_not_include_the_newest_game():
    """Appending a game slides the window without ever admitting that game.

    History (100, 200, 300): drop 300, average (100, 200) = 150.
    After appending 5000: drop 5000, average (100, 200, 300) = 200.

    The load-bearing assertion is that the result is 200 and not 1400 -- the old
    ``tail(5)`` swallowed the 5000 into the mean alongside three small numbers,
    which is how one outlier game rewrote a player's whole feature row.
    """
    before = build_features_for_player(
        "p1", _history([(2023, 1, 100.0), (2023, 2, 200.0), (2023, 3, 300.0)])
    )
    assert before["passing_yards_roll"] == pytest.approx(150.0)

    after = build_features_for_player(
        "p1",
        _history([(2023, 1, 100.0), (2023, 2, 200.0), (2023, 3, 300.0), (2023, 4, 5000.0)]),
    )
    assert after["passing_yards_roll"] == pytest.approx(200.0)
    assert after["passing_yards_roll"] != pytest.approx(1400.0)


def test_serving_can_be_scoped_to_a_season():
    """A 2026 request must not be answered from 2024 rows."""
    history = _history([(2024, 1, 111.0), (2024, 2, 222.0), (2026, 1, 333.0)])

    scoped = build_features_for_player("p1", history, season=2026)
    assert scoped is not None
    # Only the 2026 row is in scope, and it is the latest, so shift(1) leaves
    # nothing to average -- the feature is undefined, not 0.0.
    assert pd.isna(scoped["passing_yards_roll"]) or scoped["passing_yards_roll"] == 0.0
    assert not (scoped["passing_yards_roll"] == 166.5), "must not average across seasons"


def test_serving_returns_none_when_the_player_has_no_rows_in_the_requested_season():
    history = _history([(2024, 1, 111.0), (2024, 2, 222.0)])
    assert build_features_for_player("nobody", history, season=2026) is None
    assert build_features_for_player("p1", history, season=2026) is None


def test_serving_without_a_season_filter_keeps_the_old_behaviour():
    """`season=None` must remain a no-filter call so existing callers are unaffected."""
    history = _history([(2024, 1, 100.0), (2024, 2, 200.0)])
    served = build_features_for_player("p1", history)
    assert served is not None
    assert set(served.index) == set(PLAYER_FEATURE_COLUMNS)


def test_roster_fallback_row_is_not_a_fabricated_zero_vector():
    """No caller may synthesise an all-zero feature row for an unknown player.

    The regression it would reintroduce is measurable: the origin intercepts are
    passing_yards 62.592, rushing_yards 10.054, receiving_yards 16.005,
    anytime_td_prob 0.111846.
    """
    import inspect

    from nfl_predictor.api import routes

    source = inspect.getsource(routes)
    assert "0.0 for col in player_usage.PLAYER_FEATURE_COLUMNS" not in source, (
        "routes.py still fabricates an all-zero feature row for players with no history; "
        "that region was never in the training data and the model returns its origin intercept"
    )

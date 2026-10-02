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
    SERVING_FEATURE_COLUMNS,
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


def test_serving_matches_the_training_row_for_the_same_week():
    """The one assertion that matters: for a target gameweek W, the served value
    equals `_add_rolling`'s own training row **at W**.

    The previous version of this test compared the training frame's *last* row
    (week 6) against a serving vector built from the whole frame, and called it
    "the same row". Those agreed only because the frame stopped at week 6, so it
    could not have detected an off-by-one-week — and it did not: the `iloc[:-1]`
    change it appeared to justify was wrong, and this test passed anyway.

    Built on a strict arithmetic progression so every week is distinguishable.
    For W the training row is the mean of weeks 1..W-1; the served value with
    `week=W` must be exactly that, and a dropped row gives (W-1)/2 * 10 instead.
    """
    history = _history([(2023, w, float(w * 10)) for w in range(1, 17)])
    training, feature_cols = build_player_training_frame(history)

    for target_week in (6, 10, 12, 16):
        served = build_features_for_player("p1", history, season=2023, week=target_week)
        assert served is not None
        row = training[training["week"] == target_week].iloc[0]
        for column in feature_cols:
            assert served[column] == pytest.approx(row[column]), (
                f"week {target_week}, {column}: served {served[column]} vs training {row[column]}")


def test_the_window_covers_the_most_recent_completed_game():
    """Training's `shift(1)` at week W includes the game at W-1, so serving must
    too. This is the specific case the `iloc[:-1]` change broke: it silently
    excluded a real, completed game from the pregame view."""
    history = _history([(2023, w, float(w * 10)) for w in range(1, 7)])
    served = build_features_for_player("p1", history, season=2023, week=7)
    # Weeks 1..6, window 5 -> weeks 2..6 -> mean 40.
    assert served["passing_yards_roll"] == pytest.approx(40.0)
    # Dropping the last row instead would give mean(1..5) = 30.
    assert served["passing_yards_roll"] != pytest.approx(30.0)


def test_a_target_week_whose_game_is_already_in_the_frame_does_not_leak():
    """`_load_player_history` returns every *played* week, so a team's week-N game
    can be present while another team's is not. Without an explicit `week` bound
    the window would include the game being predicted — the leak `shift(1)` exists
    to prevent."""
    history = _history([(2023, w, 1000.0) for w in range(1, 6)])
    served = build_features_for_player("p1", history, season=2023, week=5)
    assert served["passing_yards_roll"] == pytest.approx(1000.0), "weeks 1-4, all 1000"
    assert build_features_for_player("p1", history, season=2023) is not None
    unbounded = build_features_for_player("p1", history, season=2023)
    assert unbounded["passing_yards_roll"] == pytest.approx(1000.0)


def test_appending_a_completed_game_slides_the_window_and_bounds_it():
    """Appending a game moves the pregame view forward by one week, and the window
    still holds only the last `window` games -- so a large new game enters the mean
    without unboundedly dominating it.

    History 100, 200, 300 predicting week 4: mean(1..3) = 200. After 5000 is
    played and we predict week 5: mean(1..4) bounded to 5 = 1400, and with a
    3-game window the 5000 cannot push the mean past its own share.
    """
    three = _history([(2023, 1, 100.0), (2023, 2, 200.0), (2023, 3, 300.0)])
    four = _history([(2023, 1, 100.0), (2023, 2, 200.0), (2023, 3, 300.0), (2023, 4, 5000.0)])

    # Predicting week 4: the pregame view is weeks 1-3, so mean 200.
    assert build_features_for_player("p1", three, season=2023, week=4)["passing_yards_roll"] == pytest.approx(200.0)
    # Week 4 is now played and we predict week 5: the view is weeks 1-4, mean 1400.
    assert build_features_for_player("p1", four, season=2023, week=5)["passing_yards_roll"] == pytest.approx(1400.0)
    # The same view, bounded to a 3-game window, cannot be dominated by the 5000.
    bounded = build_features_for_player("p1", four, season=2023, week=5, window=3)
    assert bounded["passing_yards_roll"] == pytest.approx((200.0 + 300.0 + 5000.0) / 3)
    # Without `week`, the last row is treated as the most recent completed game.
    assert build_features_for_player("p1", three, season=2023)["passing_yards_roll"] == pytest.approx(200.0)


def test_serving_can_be_scoped_to_a_season():
    """A 2026 request must not be answered from 2024 rows."""
    history = _history([(2024, 1, 111.0), (2024, 2, 222.0), (2026, 1, 333.0)])

    scoped = build_features_for_player("p1", history, season=2026)
    assert scoped is not None
    # Only the 2026 row is in scope, and it is a *completed* game for a week-2
    # fixture, so the pregame view is that one game: 333.0. Averaging across
    # seasons (the old behaviour) gave 166.5.
    assert scoped["passing_yards_roll"] == pytest.approx(333.0)
    assert not (scoped["passing_yards_roll"] == 166.5), "must not average across seasons"

    # A single prior game is a real value, not the "undefined" case: the old
    # `iloc[:-1]` dropped the only row and produced NaN, which then reached the
    # model as a fabricated zero.
    assert not pd.isna(scoped["passing_yards_roll"])


def test_serving_returns_none_when_the_player_has_no_rows_in_the_requested_season():
    history = _history([(2024, 1, 111.0), (2024, 2, 222.0)])
    assert build_features_for_player("nobody", history, season=2026) is None
    assert build_features_for_player("p1", history, season=2026) is None


def test_serving_without_a_season_filter_keeps_the_old_behaviour():
    """`season=None` must remain a no-filter call so existing callers are unaffected."""
    history = _history([(2024, 1, 100.0), (2024, 2, 200.0)])
    served = build_features_for_player("p1", history)
    assert served is not None
    # Superset, not equality: the builder also emits `passing_tds_roll`, which the
    # QB passing-TD model is fitted on. `PLAYER_FEATURE_COLUMNS` itself is
    # unchanged -- it is what the anytime-TD classifier and the yardage
    # regressors are fitted on, so widening it would change their feature count.
    assert set(served.index) == set(SERVING_FEATURE_COLUMNS)
    assert set(PLAYER_FEATURE_COLUMNS) < set(served.index)
    assert "passing_tds_roll" not in PLAYER_FEATURE_COLUMNS


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

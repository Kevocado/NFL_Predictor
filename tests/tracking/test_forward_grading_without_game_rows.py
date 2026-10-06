"""a forward pick must grade even though its game was never a game prediction.

A standalone forward tick writes no `game_predictions` rows -- it has no win
probabilities to write, those come from the game-level model. But the
reconciler LEFT JOINs... no, INNER joins `game_predictions` to read each game's
`commence_time`, and its own docstring calls that "fail closed". For a forward
pick it is not fail-closed, it is fail-NOTHING: the join drops every row, so
the reconciler returns 0 and the forward test grades to zero forever.

That is the identical trap already documented in `forward_report.graded_picks`,
which solved it by deriving season/week from the game_id and keeping the join
as a fallback. The reconciler never got the same treatment, so with correct
outcomes in hand the forward test still reported nothing.

The post-kickoff guard must survive the fix. It cannot: without a kickoff time
there is nothing to compare a snapshot timestamp against, so the guard has to
come from somewhere else. `forward_tick` already refuses to snapshot a
post-kickoff game, so a forward row cannot be a reconstruction -- and the
closing-line capture refuses those too. The guard is therefore kept for rows
that DO have a kickoff time and simply not applied to those that do not, which
is stated here rather than left implicit.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.tracking import store

FUTURE = "2099-09-04T20:20:00"
FWD_GAME = "2026_05_BUF_LA"


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    yield


def _forward_pick(**overrides):
    """A pick exactly as the forward tick writes it, with NO game_predictions row.

    `closing_line` is written by its own pass, not by the snapshot: the close is
    not knowable when the pick is made, so `record_player_prop_predictions`
    deliberately leaves it NULL and the capture UPDATEs it later.
    """
    row = {
        "game_id": FWD_GAME, "player_id": "00-1", "player_name": "James Cook",
        "market": "fwd_rushing_yards", "predicted_value": 75.5,
        "side": "under", "line_at_snapshot": 75.5, "odds_at_snapshot": -114.0,
        "model_p_over": 0.864, "edge_vs_breakeven": 0.331,
    }
    row.update(overrides)
    store.record_player_prop_predictions([row])
    # The close came in BELOW the 75.5 we took an under at, which is 2 yards the
    # right way for that side.
    store.record_closing_lines([{
        "game_id": FWD_GAME, "player_id": "00-1",
        "market": "fwd_rushing_yards", "closing_line": 73.5}])


def test_a_forward_pick_grades_without_a_game_prediction_row():
    _forward_pick()

    stats = pd.DataFrame([{"game_id": FWD_GAME, "player_id": "00-1",
                           "rushing_yards": 40.0}])

    assert store.reconcile_player_prop_predictions(stats) == 1
    with store._connect() as conn:
        row = pd.read_sql("SELECT * FROM player_prop_predictions", conn).iloc[0]
    assert row["actual_value"] == 40.0
    assert row["hit"] == 1, "40.0 is under the 75.5 the call was made at"
    assert row["clv"] == 2.0, "an under gains when the close is lower"


def test_the_post_kickoff_guard_still_holds_where_a_kickoff_is_known():
    """Removing the inner join must not remove the guard. When the game's kickoff
    IS known -- the background tracker writes game_predictions -- a row recorded
    at or after it is a reconstruction and is still refused.

    The kickoff is written directly rather than via `record_game_predictions`,
    which refuses to snapshot a game that has already kicked off and so cannot
    produce the past-kickoff row this needs.
    """
    with store._connect() as conn, conn:
        conn.execute(
            "INSERT INTO game_predictions (game_id, home_team, away_team, "
            "home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, "
            "over_prob, under_prob, commence_time, snapshotted_at) "
            "VALUES (?, 'LAR', 'BUF', 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ?, ?)",
            (FWD_GAME, "2000-09-04T20:20:00", "2000-09-01T00:00:00+00:00"))
    _forward_pick()

    stats = pd.DataFrame([{"game_id": FWD_GAME, "player_id": "00-1",
                           "rushing_yards": 40.0}])

    assert store.reconcile_player_prop_predictions(stats) == 0
    with store._connect() as conn:
        row = pd.read_sql("SELECT resolved, actual_value FROM player_prop_predictions",
                          conn).iloc[0]
    assert row["resolved"] == 0, "the row must be left unresolved, not graded"


def test_the_exemption_is_narrow_a_non_forward_orphan_still_fails_closed():
    """`test_reconcile_skips_orphan_prop_rows_with_no_game` already requires an
    unprovable prop row to stay ungraded. The fix must exempt forward picks only
    -- loosening the join for every market would have dropped that guarantee
    repo-wide to fix one market, silently."""
    store.record_player_prop_predictions([{
        "game_id": FWD_GAME, "player_id": "00-2", "player_name": "Orphan",
        "market": "rushing_yards", "predicted_value": 90.0,
    }])

    stats = pd.DataFrame([{"game_id": FWD_GAME, "player_id": "00-2",
                           "rushing_yards": 92.0}])

    assert store.reconcile_player_prop_predictions(stats) == 0
    with store._connect() as conn:
        row = pd.read_sql("SELECT resolved FROM player_prop_predictions", conn).iloc[0]
    assert row["resolved"] == 0


def test_a_pre_kickoff_pick_still_grades_when_the_game_is_known():
    """The ordinary path must be untouched: known kickoff in the future, row
    snapshotted before it."""
    store.record_game_predictions([{
        "game_id": FWD_GAME, "home_team": "LAR", "away_team": "BUF",
        "season": 2026, "week": 5, "commence_time": FUTURE,
        "home_win_prob": 0.5, "away_win_prob": 0.5,
        "home_cover_prob": 0.5, "away_cover_prob": 0.5,
        "over_prob": 0.5, "under_prob": 0.5,
    }])
    _forward_pick()

    stats = pd.DataFrame([{"game_id": FWD_GAME, "player_id": "00-1",
                           "rushing_yards": 40.0}])

    assert store.reconcile_player_prop_predictions(stats) == 1

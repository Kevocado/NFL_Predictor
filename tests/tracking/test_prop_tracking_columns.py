"""Forward-test columns on player_prop_predictions.

Three properties, each of which has broken something before:

* old rows stay readable after the migration (ALTER TABLE, not a rebuild);
* a snapshot's line/probability/edge are persisted verbatim, because a
  recorded price is the pick;
* reconciliation fills `hit` and `clv` and never rewrites the snapshot side.
  CLV's sign depends on which side was taken, so an over that closes 2 yards
  higher and an under that closes 2 yards lower are both +2.0.
"""
from __future__ import annotations

import contextlib
import sqlite3

import pandas as pd
import pytest

from nfl_predictor import config
from nfl_predictor.tracking import store

GAME_ID = "2099_01_ALB_DEN"
FORWARD_COLUMNS = ["line_at_snapshot", "odds_at_snapshot", "model_p_over",
                   "edge_vs_breakeven", "closing_line", "clv", "hit"]


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _game():
    return {
        "game_id": GAME_ID, "home_team": "ALB", "away_team": "DEN",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.5, "away_win_prob": 0.5,
        "home_cover_prob": 0.5, "away_cover_prob": 0.5,
        "over_prob": 0.5, "under_prob": 0.5,
    }


def _close(line: float):
    """The closing line is not knowable pre-kickoff, so it is written separately."""
    store.record_closing_lines([{"game_id": GAME_ID, "player_id": "00-1",
                                 "market": "receiving_yards", "closing_line": line}])


def _prop(**overrides):
    prop = {
        "game_id": GAME_ID, "player_id": "00-1", "player_name": "Test",
        "position": "WR", "market": "receiving_yards", "predicted_value": 58.0,
    }
    prop.update(overrides)
    return prop


def _columns():
    with contextlib.closing(store._connect()) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(player_prop_predictions)")}


def test_migration_adds_every_forward_column():
    assert set(FORWARD_COLUMNS) <= _columns()


def test_legacy_row_survives_the_migration_with_null_new_columns():
    """A pre-migration database: the table exists without the forward columns."""
    legacy = sqlite3.connect(config.TRACKING_DB_PATH)
    legacy.execute("""
        CREATE TABLE player_prop_predictions (
            game_id TEXT NOT NULL, player_id TEXT NOT NULL, player_name TEXT NOT NULL,
            market TEXT NOT NULL, predicted_value REAL NOT NULL,
            snapshotted_at TEXT NOT NULL, resolved INTEGER NOT NULL DEFAULT 0,
            actual_value REAL,
            PRIMARY KEY (game_id, player_id, market)
        )
    """)
    legacy.execute(
        "INSERT INTO player_prop_predictions VALUES (?,?,?,?,?,?,1,?)",
        ("2020_01_X_Y", "00-9", "Old", "receiving_yards", 40.0, "2020-09-10T00:00:00+00:00", 44.0),
    )
    legacy.commit()
    legacy.close()

    with contextlib.closing(store._connect()) as conn:
        row = conn.execute(
            "SELECT predicted_value, actual_value, line_at_snapshot, hit FROM player_prop_predictions"
        ).fetchone()

    assert row[0] == 40.0, "the old row's own data is intact"
    assert row[1] == 44.0
    assert row[2] is None, "a pre-migration row has no snapshot line"
    assert row[3] is None, "and is not retroactively graded"


def test_record_persists_the_snapshot_price_and_probability():
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop(
        side="over", line_at_snapshot=52.5, odds_at_snapshot=-110,
        model_p_over=0.60, edge_vs_breakeven=0.076,
    )])

    with contextlib.closing(store._connect()) as conn:
        row = conn.execute(
            "SELECT side, line_at_snapshot, odds_at_snapshot, model_p_over, edge_vs_breakeven"
            " FROM player_prop_predictions").fetchone()

    assert row == ("over", 52.5, -110.0, 0.60, 0.076)


def test_hit_and_clv_for_an_over_pick():
    """line 52.5 -> close 54.5 -> actual 60: covered, and the line moved our way."""
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop(side="over", line_at_snapshot=52.5)])
    _close(54.5)

    _resolve(60.0)

    row = _graded()
    assert row["hit"] == 1
    assert row["clv"] == pytest.approx(2.0)
    assert row["actual_value"] == 60.0


def test_hit_and_clv_for_an_under_pick():
    """line 52.5 -> close 50.5 -> actual 48: also covered, also +2.0."""
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop(side="under", line_at_snapshot=52.5)])
    _close(50.5)

    _resolve(48.0)

    row = _graded()
    assert row["hit"] == 1
    assert row["clv"] == pytest.approx(2.0), "an under gains when the line drops"


def test_a_missed_pick_is_recorded_as_a_miss():
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop(side="over", line_at_snapshot=52.5)])
    _close(50.5)

    _resolve(40.0)

    row = _graded()
    assert row["hit"] == 0
    assert row["clv"] == pytest.approx(-2.0), "the line moved against the pick"


def test_no_closing_line_means_no_clv_but_still_a_hit():
    """CLV needs a close. Its absence must not block the hit verdict."""
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop(side="over", line_at_snapshot=52.5)])

    _resolve(60.0)

    row = _graded()
    assert row["hit"] == 1
    assert row["clv"] is None


def test_no_side_means_no_hit():
    """A yardage projection with no over/under call cannot be graded over/under."""
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop(line_at_snapshot=52.5)])
    _close(54.5)

    _resolve(60.0)

    row = _graded()
    assert row["hit"] is None
    assert row["clv"] is None


def test_reconciliation_does_not_rewrite_the_snapshot_line():
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop(side="over", line_at_snapshot=52.5, model_p_over=0.61)])
    _close(54.5)

    _resolve(60.0)

    row = _graded()
    assert row["line_at_snapshot"] == 52.5, "the bettable price is immutable"
    assert row["model_p_over"] == 0.61


def test_an_already_graded_row_is_not_regraded():
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop(side="over", line_at_snapshot=52.5)])
    _close(54.5)
    _resolve(60.0)

    # A second reconciliation with a different actual must change nothing.
    store.reconcile_player_prop_predictions(_stats(99.0))

    row = _graded()
    assert row["actual_value"] == 60.0
    assert row["hit"] == 1


def _stats(actual: float) -> pd.DataFrame:
    return pd.DataFrame([{"game_id": GAME_ID, "player_id": "00-1",
                          "receiving_yards": actual}])


def _resolve(actual: float) -> None:
    store.reconcile_player_prop_predictions(_stats(actual))


def _graded() -> dict:
    with contextlib.closing(store._connect()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM player_prop_predictions").fetchone()
    return dict(row)
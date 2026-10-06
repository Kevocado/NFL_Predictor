"""the closing capture must only close picks that exist, from the book that snapped them.

CodeRabbit found two integrity holes in the capture that the happy-path tests
didn't catch. Both make CLV a number about something other than the pick that
was actually made.

1. The capture fetches lines for whatever markets the MODEL currently loads,
   not for whatever markets have SNAPSHOT ROWS in the database. A market the
   model has dropped, or a market loaded only by the snapshot run, would get a
   close written against a row that does not exist -- or, a row that DOES exist
   would be skipped because the model no longer prices it. The close must
   attach to the pick, not to the model.

2. The capture matches closing lines by (game, player, market) only. The
   snapshot records which book and which line were snapped; a close from a
   different book is a difference between two shops' prices, not CLV. The
   capture must match by book AND line, or at minimum document that it assumes
   the book is the same and verify it.

Both are fixed by reading the snapshot rows FIRST and fetching only those
market/book/line combinations. This is the same 'follow the data, not the
schema' discipline that caught the inner-join bug.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from nfl_predictor.tracking import forward_tick, store

from quantile_stubs import write_artifacts

FUTURE = "2099-09-04T20:20:00"

GAME = {
    "game_id": "G1", "home_team": "BAL", "away_team": "DEN",
    "season": 2026, "week": 5, "commence_time": FUTURE,
    "home_win_prob": 0.5, "away_win_prob": 0.5,
    "home_cover_prob": 0.5, "away_cover_prob": 0.5,
    "over_prob": 0.5, "under_prob": 0.5,
}

EVENT_INDEX = {("Baltimore Ravens", "Denver Broncos"): "e-bal-den"}

PROP_FD = {
    "player_name": "A.J. Brown Jr.", "normalized_name": "aj brown",
    "market": "player_pass_yds", "line": 50.0,
    "over_odds": -110, "under_odds": -110, "book": "fanduel", "team": "BAL",
}

PROP_DK = {**PROP_FD, "book": "draftkings", "line": 52.0}

PLAYERS = [{"player_id": "00-1", "player_name": "A.J. Brown Jr.", "team": "BAL"}]


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    yield


def _files(tmp_path):
    from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS

    games = tmp_path / "games.json"
    games.write_text(json.dumps([GAME]))
    players = tmp_path / "players.json"
    players.write_text(json.dumps(PLAYERS))
    row = {c: 0.0 for c in FORWARD_FEATURE_COLUMNS}
    row.update({"player_id": "00-1", "season": 2026, "week": 1})
    frame = tmp_path / "frame.parquet"
    pd.DataFrame([row]).to_parquet(frame, index=False)
    return ["--games-json", str(games), "--players-path", str(players),
            "--feature-frame", str(frame)]


def _models(tmp_path, markets=("passing_yards",)):
    return write_artifacts(tmp_path, markets=markets)


def _run(models_dir, tmp_path, *extra):
    return forward_tick.main([
        "--models-dir", str(models_dir), *_files(tmp_path), *extra])


def _snapshot_rows():
    with store._connect() as conn:
        return pd.read_sql("SELECT * FROM player_prop_predictions", conn)


def test_capture_only_closes_markets_that_have_snapshot_rows(monkeypatch, tmp_path):
    """The capture must NOT write closes for a market the model prices but the
    snapshot did NOT record. If the model prices passing + rushing but the
    snapshot only has passing, the capture must not invent a rushing close.

    The current code sizes its fetch off `market_quantiles` (what the model
    loads), not off what snapshot rows exist in the DB. That means a market the
    model dropped would get no close, and a market the model prices but the
    book didn't offer would get a fabricated close written against... nothing,
    because `record_closing_lines` UPDATEs. But the fetch still spends credits
    on it, which is a budget leak, and the logic is semantically wrong: the
    close belongs to the pick, not the model."""
    store.record_game_predictions([GAME])
    models = _models(tmp_path, markets=("passing_yards", "rushing_yards"))

    def _stub_snap(monkeypatch):
        monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: True)
        monkeypatch.setattr(forward_tick, "fetch_event_index", lambda: EVENT_INDEX)
        # Snapshot: only passing is offered for this player.
        monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                            lambda e, markets=None, credits_needed=1: [PROP_FD])

    _stub_snap(monkeypatch)
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
    assert _run(models, tmp_path) == 0

    # Capture run: the model prices both, the book now has both.
    # The bug would close both; the fix closes only the passing pick that exists.
    def _stub_capture(monkeypatch):
        monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: True)
        monkeypatch.setattr(forward_tick, "fetch_event_index", lambda: EVENT_INDEX)
        monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                            lambda e, markets=None, credits_needed=1: [
                                {**PROP_FD, "line": 54.5},    # passing moved
                                {**PROP_FD, "market": "player_rush_yds", "line": 120.0},  # rushing appeared
                            ])

    _stub_capture(monkeypatch)
    assert _run(models, tmp_path, "--capture-closing") == 0

    rows = _snapshot_rows()
    assert len(rows) == 1, "only one forward pick was snapshotted"
    assert rows.iloc[0]["market"] == "fwd_passing_yards"
    assert rows.iloc[0]["closing_line"] == 54.5
    # No rushing row was invented by the capture
    assert "fwd_rushing_yards" not in rows["market"].values


def test_capture_matches_the_book_and_line_that_snapped_the_pick(monkeypatch, tmp_path):
    """A pick snapped at one book's line must get its close from THAT book, not
    from another shop. The schema now records `book_at_snapshot` at pick time,
    and the capture matches by (player_id, book, market) so CLV is measured
    against the SAME shop the bet was made at."""
    store.record_game_predictions([GAME])
    models = _models(tmp_path)

    def _stub_snap(monkeypatch):
        monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: True)
        monkeypatch.setattr(forward_tick, "fetch_event_index", lambda: EVENT_INDEX)
        # Snapshot: both books quote the player; the first match wins.
        # The recorded pick will have line_at_snapshot from the FIRST match,
        # and book_at_snapshot from that match.
        monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                            lambda e, markets=None, credits_needed=1: [
                                PROP_DK,    # DraftKings 52.0 (first in list)
                                PROP_FD,    # FanDuel 50.0
                            ])

    _stub_snap(monkeypatch)
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
    assert _run(models, tmp_path) == 0

    rows = _snapshot_rows()
    picked = rows.iloc[0]
    # The pick was made at DK (first in list), so book_at_snapshot = draftkings
    assert picked["book_at_snapshot"] == "draftkings"
    assert picked["line_at_snapshot"] == 52.0

    def _stub_capture(monkeypatch):
        monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: True)
        monkeypatch.setattr(forward_tick, "fetch_event_index", lambda: EVENT_INDEX)
        # At close: FD moved to 54.5, DK moved to 48.0 (opposite directions).
        # The fix matches by book_at_snapshot, so it picks DK's close (48.0).
        monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                            lambda e, markets=None, credits_needed=1: [
                                {**PROP_FD, "line": 54.5},      # FD close
                                {**PROP_DK, "line": 48.0},      # DK close
                            ])

    _stub_capture(monkeypatch)
    assert _run(models, tmp_path, "--capture-closing") == 0

    rows = _snapshot_rows()
    picked = rows.iloc[0]
    # Close MUST come from DK (the book the pick was made at), not FD.
    assert picked["closing_line"] == 48.0, "close must come from the SAME book"


def test_capture_does_not_require_the_model_to_still_price_the_market(monkeypatch, tmp_path):
    """If a model artifact is removed between the snapshot and the close, the
    capture must STILL close the existing pick. The pick's existence is a fact
    about the past; the model's current artifact set is a fact about the
    present. Tying the close to the artifact makes the forward test fragile:
    retraining one market would orphan all its closes.

    The fix reads the DB for what markets have picks, so it works even when
    the artifact for that market is gone."""
    store.record_game_predictions([GAME])
    # Create TWO artifacts: passing_yards (the pick's market) and rushing_yards
    # (unrelated). We'll remove only passing_yards.
    models = _models(tmp_path, markets=("passing_yards", "rushing_yards"))

    def _stub_snap(monkeypatch):
        monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: True)
        monkeypatch.setattr(forward_tick, "fetch_event_index", lambda: EVENT_INDEX)
        monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                            lambda e, markets=None, credits_needed=1: [PROP_FD])

    _stub_snap(monkeypatch)
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
    assert _run(models, tmp_path) == 0

    # REMOVE only the passing_yards artifact -- the model no longer loads this
    # market, but rushing_yards remains. The pick exists in the DB.
    (models / "passing_yards_quantile_2025.pkl").unlink()

    def _stub_capture(monkeypatch):
        monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: True)
        monkeypatch.setattr(forward_tick, "fetch_event_index", lambda: EVENT_INDEX)
        monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                            lambda e, markets=None, credits_needed=1: [{**PROP_FD, "line": 54.5}])

    _stub_capture(monkeypatch)
    # With the fix, capture reads the DB and proceeds even without the artifact.
    assert _run(models, tmp_path, "--capture-closing") == 0

    rows = _snapshot_rows()
    assert rows.iloc[0]["closing_line"] == 54.5, "close written even with artifact removed"
"""the CLI's closing-capture flag.

`run_forward_tick(capture_closing=True)` is unreachable from the command line
without this, and CLV is only ever produced by running the tick -- so an
unreachable flag means the close is never captured and `clv` stays NULL, which
is the bug the whole capture exists to fix.
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

PROP = {
    "player_name": "A.J. Brown Jr.", "normalized_name": "aj brown",
    "market": "player_pass_yds", "line": 50.0,
    "over_odds": -110, "under_odds": -110, "book": "fanduel", "team": "BAL",
}

PLAYERS = [{"player_id": "00-1", "player_name": "A.J. Brown Jr.", "team": "BAL"}]


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    yield


def _stub(monkeypatch, line):
    monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: True)
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda e, markets=None, credits_needed=1: [{**PROP, "line": line}])
    monkeypatch.setattr(forward_tick, "fetch_event_index", lambda: EVENT_INDEX)


def _files(tmp_path):
    from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS

    games = tmp_path / "games.json"
    games.write_text(json.dumps([GAME]))
    players = tmp_path / "players.json"
    players.write_text(json.dumps(PLAYERS))
    # One prior week for 00-1, so `history_row_for` has a row to return and the
    # model has every declared feature column to read.
    row = {c: 0.0 for c in FORWARD_FEATURE_COLUMNS}
    row.update({"player_id": "00-1", "season": 2026, "week": 1})
    frame = tmp_path / "frame.parquet"
    pd.DataFrame([row]).to_parquet(frame, index=False)
    return ["--games-json", str(games), "--players-path", str(players),
            "--feature-frame", str(frame)]


def _run(models_dir, tmp_path, *extra):
    return forward_tick.main([
        "--models-dir", str(models_dir), *_files(tmp_path), *extra])


def test_the_cli_can_capture_the_closing_line(monkeypatch, tmp_path, capsys):
    """The flag has to reach `run_forward_tick`, or the close is uncapturable
    from the command line and CLV is permanently NULL."""
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
    store.record_game_predictions([GAME])
    models = write_artifacts(tmp_path, markets=("passing_yards",))

    # The snapshot.
    _stub(monkeypatch, 50.0)
    assert _run(models, tmp_path) == 0

    # The close, minutes later: the book has moved to 54.5.
    _stub(monkeypatch, 54.5)
    assert _run(models, tmp_path, "--capture-closing") == 0

    out = capsys.readouterr().out
    assert "closing_lines_written: 1" in out, out

    with store._connect() as conn:
        row = pd.read_sql("SELECT * FROM player_prop_predictions", conn).iloc[0]
    assert row["closing_line"] == 54.5
    assert row["line_at_snapshot"] == 50.0, "the taken price is immutable"


def test_a_closing_capture_dry_run_still_spends_nothing(monkeypatch, tmp_path):
    """--dry-run is the budget guard; it has to hold for the capture too."""
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
    store.record_game_predictions([GAME])
    models = write_artifacts(tmp_path, markets=("passing_yards",))
    called = []
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda *a, **k: called.append(a) or [])

    exit_code = _run(models, tmp_path, "--dry-run", "--capture-closing")

    assert exit_code == 0
    assert called == [], "--dry-run must not spend a credit"

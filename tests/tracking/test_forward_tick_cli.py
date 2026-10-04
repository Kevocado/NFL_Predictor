"""the forward_tick CLI.

It spends the free tier's metered credits, so the tests that matter here are the
ones that prove it cannot run by accident: no key, no snapshot, no spend. Every
test stubs the network seam; nothing here reaches The Odds API.
"""
from __future__ import annotations

import json

import pytest

from nfl_predictor.tracking import forward_tick, store

from quantile_stubs import QUANTILE_GRID as QUANTILES, write_artifacts


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    yield


GAME = {
    "game_id": "G1", "home_team": "ALB", "away_team": "DEN",
    "commence_time": "2099-09-04T20:20:00",
    "home_win_prob": 0.5, "away_win_prob": 0.5,
    "home_cover_prob": 0.5, "away_cover_prob": 0.5,
    "over_prob": 0.5, "under_prob": 0.5,
}

PROP = {
    "player_id": "00-1", "player_name": "P", "market": "player_rec_yds",
    "line": 50.0, "over_odds": -110, "under_odds": -110,
    "book": "fanduel", "team": "ALB",
}


def _stub(monkeypatch, credits=500, props=None):
    monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: credits >= needed)
    monkeypatch.setattr(forward_tick, "probe_props_coverage",
                        lambda e: {"has_any": True, "markets": {}, "books": []})
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda e, markets=None, credits_needed=1: (props if props is not None else [PROP]))


def _artifacts(tmp_path, monkeypatch, markets=("receiving_yards",)):
    """Artifacts on disk, loaded exactly as the CLI loads them."""
    return write_artifacts(tmp_path, markets=markets)


def test_loads_an_artifact_into_a_per_market_predictor(tmp_path, monkeypatch):
    models_dir = _artifacts(tmp_path, monkeypatch)

    predictor = forward_tick.predictor_for(models_dir)

    assert set(predictor) == {"player_rec_yds"}
    quantiles = predictor["player_rec_yds"](50.0)
    assert set(quantiles) == set(QUANTILES)
    values = [quantiles[q] for q in QUANTILES]
    assert values == sorted(values), "quantile predictions must ascend with level"
    assert len(set(values)) > 1, "a flat set would be the degenerate case"


def test_missing_artifact_directory_is_an_error_not_an_empty_tick(tmp_path, monkeypatch):
    """Silently ticking with no model would price every prop as 'no edge' and
    report a clean week having looked at nothing."""
    with pytest.raises(FileNotFoundError, match="quantile"):
        forward_tick.predictor_for(tmp_path / "missing")


def test_a_market_absent_from_the_artifacts_is_simply_absent(tmp_path, monkeypatch):
    models_dir = _artifacts(tmp_path, monkeypatch)

    predictor = forward_tick.predictor_for(models_dir)

    assert "player_pass_yds" not in predictor, "an absent market must be absent, not invented"


def test_predictor_expands_a_single_row_to_every_quantile(tmp_path, monkeypatch):
    models_dir = _artifacts(tmp_path, monkeypatch)
    predictor = forward_tick.predictor_for(models_dir)

    row = {"f1": 0.5}
    quantiles = predictor["player_rec_yds"](50.0, row)

    assert set(quantiles) == set(QUANTILES)


def test_tick_through_loaded_artifacts_logs_an_edge(monkeypatch, tmp_path):
    store.record_game_predictions([GAME])
    models_dir = _artifacts(tmp_path, monkeypatch)
    _stub(monkeypatch)

    result = forward_tick.run_forward_tick(
        games=[GAME],
        market_quantiles=forward_tick.predictor_for(models_dir),
    )

    assert result["props_snapshotted"] == 1
    assert result["degenerate_rows"] == 0
    # The stub's distribution straddles the 50.5 line, so the over side clears
    # the gate and exactly one pick is logged.
    assert result["picks_logged"] == 1
    logged = _logged()
    assert logged[0]["side"] == "over"
    assert logged[0]["edge_vs_breakeven"] >= 0.05


def test_cli_refuses_without_an_api_key(monkeypatch, tmp_path):
    # main() reads the key from config at call time, so that is what to patch.
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", None)
    called = []
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda *a, **k: called.append(a) or [])

    exit_code = forward_tick.main(["--models-dir", str(_artifacts(tmp_path, monkeypatch))])

    assert exit_code != 0
    assert called == [], "no key means no request and no snapshot"


def test_cli_dry_run_makes_no_request_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")
    called = []
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda *a, **k: called.append(a) or [])

    exit_code = forward_tick.main([
        "--models-dir", str(_artifacts(tmp_path, monkeypatch)), "--dry-run",
        "--games-json", _games_file(tmp_path),
    ])

    assert exit_code == 0
    assert called == [], "--dry-run must not spend a credit"
    assert _logged() == []


def test_cli_refuses_an_empty_slate_rather_than_reporting_a_clean_week(monkeypatch, tmp_path):
    """No games and "looked at nothing" both produce zero picks. Only the first
    is a real result, so the second must not be reportable."""
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")
    empty = tmp_path / "empty.json"
    empty.write_text("[]")

    with pytest.raises(ValueError, match="empty"):
        forward_tick.main(["--models-dir", str(_artifacts(tmp_path, monkeypatch)),
                           "--games-json", str(empty)])


def test_cli_refuses_when_no_slate_is_given(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")

    with pytest.raises(ValueError, match="no games to tick"):
        forward_tick.main(["--models-dir", str(_artifacts(tmp_path, monkeypatch))])


def test_cli_reports_what_it_would_spend_on_a_dry_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")
    _stub(monkeypatch, props=[])

    forward_tick.main([
        "--models-dir", str(_artifacts(tmp_path, monkeypatch)),
        "--games-json", _games_file(tmp_path), "--dry-run",
    ])

    out = capsys.readouterr().out
    assert "credit" in out.lower()


def test_cli_writes_nothing_when_the_budget_is_insufficient(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")
    store.record_game_predictions([GAME])
    _stub(monkeypatch, credits=0)

    result = forward_tick.run_forward_tick(
        games=[GAME], market_quantiles=forward_tick.predictor_for(
            _artifacts(tmp_path, monkeypatch)))

    assert result["picks_logged"] == 0
    assert _logged() == []


def test_main_writes_a_weekly_report_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")
    _stub(monkeypatch, props=[])
    out_dir = tmp_path / "reports"

    forward_tick.main([
        "--models-dir", str(_artifacts(tmp_path, monkeypatch)),
        "--games-json", _games_file(tmp_path),
        "--report-dir", str(out_dir), "--season", "2026", "--week", "4",
    ])

    assert (out_dir / "2026-W4.md").exists()


def _games_file(tmp_path):
    path = tmp_path / "games.json"
    path.write_text(json.dumps([GAME]))
    return str(path)


def _logged() -> list[dict]:
    import contextlib
    import sqlite3

    with contextlib.closing(store._connect()) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM player_prop_predictions WHERE line_at_snapshot IS NOT NULL")]
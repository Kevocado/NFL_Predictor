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
    "season": 2026, "week": 4,
    "commence_time": "2099-09-04T20:20:00",
    "home_win_prob": 0.5, "away_win_prob": 0.5,
    "home_cover_prob": 0.5, "away_cover_prob": 0.5,
    "over_prob": 0.5, "under_prob": 0.5,
}

#: Shaped like what `props_snapshot._normalize_rows` actually returns: names and a
#: normalized_name, and NO player_id. An earlier version of this fixture carried a
#: hand-seeded player_id, which is precisely why the missing join went unnoticed.
PROP = {
    "player_name": "A.J. Brown Jr.", "normalized_name": "aj brown",
    "market": "player_rec_yds", "line": 50.0,
    "over_odds": -110, "under_odds": -110, "book": "fanduel", "team": "ALB",
}

PLAYERS = [{"player_id": "00-1", "player_name": "A.J. Brown Jr.", "team": "ALB"}]


def _stub(monkeypatch, credits=500, props=None):
    monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: credits >= needed)
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda e, markets=None, credits_needed=1: (props if props is not None else [PROP]))


def _artifacts(tmp_path, monkeypatch, markets=("receiving_yards",)):
    """Artifacts on disk, loaded exactly as the CLI loads them."""
    return write_artifacts(tmp_path, markets=markets)


def test_loads_an_artifact_into_a_per_market_predictor(tmp_path, monkeypatch):
    models_dir = _artifacts(tmp_path, monkeypatch)

    predictor = forward_tick.predictor_for(models_dir)

    assert set(predictor) == {"player_rec_yds"}
    quantiles = predictor["player_rec_yds"]({"passing_yards_roll": 0.0}, 50.0)
    assert set(quantiles) == set(QUANTILES)
    values = [quantiles[q] for q in QUANTILES]
    assert values == sorted(values), "quantile predictions must ascend with level"


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

    quantiles = predictor["player_rec_yds"]({"passing_yards_roll": 0.5}, 50.0)

    assert set(quantiles) == set(QUANTILES)


def test_the_prediction_actually_moves_with_the_feature_row(tmp_path, monkeypatch):
    """The pin for the model-bypass bug.

    The stub is a function of its first feature, so discarding the row shifts
    every quantile. When `predict` built an all-zero frame instead of using the
    row it was given, both calls returned the same numbers and nothing failed --
    the 5% gate would then have fired off the book's line alone.
    """
    models_dir = _artifacts(tmp_path, monkeypatch)
    predictor = forward_tick.predictor_for(models_dir)

    low = predictor["player_rec_yds"]({"passing_yards_roll": 0.0}, 50.0)
    high = predictor["player_rec_yds"]({"passing_yards_roll": 10.0}, 50.0)

    assert low[0.5] != high[0.5]
    assert high[0.5] > low[0.5]


def test_predict_refuses_an_empty_feature_row(tmp_path, monkeypatch):
    """An absent row must raise, not silently become zeros."""
    models_dir = _artifacts(tmp_path, monkeypatch)
    predictor = forward_tick.predictor_for(models_dir)

    with pytest.raises(ValueError, match="feature row"):
        predictor["player_rec_yds"]({}, 50.0)


def test_two_players_get_different_predictions(tmp_path, monkeypatch):
    models_dir = _artifacts(tmp_path, monkeypatch)
    predictor = forward_tick.predictor_for(models_dir)

    a = predictor["player_rec_yds"]({"passing_yards_roll": 1.0}, 50.0)
    b = predictor["player_rec_yds"]({"passing_yards_roll": 9.0}, 50.0)

    assert a[0.5] != b[0.5], "the model is being consulted, not bypassed"


def test_tick_through_loaded_artifacts_logs_an_edge(monkeypatch, tmp_path):
    store.record_game_predictions([GAME])
    models_dir = _artifacts(tmp_path, monkeypatch)
    _stub(monkeypatch)

    result = forward_tick.run_forward_tick(
        games=[GAME],
        market_quantiles=forward_tick.predictor_for(models_dir),
        feature_frame=_feature_frame(), players=PLAYERS,
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
            _artifacts(tmp_path, monkeypatch)), feature_frame=_feature_frame())

    assert result["picks_logged"] == 0
    assert _logged() == []


def test_main_writes_a_weekly_report_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")
    _stub(monkeypatch, props=[])
    out_dir = tmp_path / "reports"

    forward_tick.main([
        "--models-dir", str(_artifacts(tmp_path, monkeypatch)),
        "--games-json", _games_file(tmp_path),
        "--players-path", _players_file(tmp_path),
        "--feature-frame", _feature_frame_path(tmp_path),
        "--report-dir", str(out_dir), "--season", "2026", "--week", "4",
    ])

    assert (out_dir / "2026-W4.md").exists()


def _feature_frame():
    """One prior week for player 00-1, so `history_row_for` has a row to return."""
    import pandas as pd

    from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS

    row = {c: 0.0 for c in FORWARD_FEATURE_COLUMNS}
    row["player_id"] = "00-1"
    row["season"] = 2026
    row["week"] = 1
    return pd.DataFrame([row])


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

# --- the slate must be loadable, and the scope must be real ----------------

def test_slate_rows_carry_commence_time(monkeypatch):
    """nflverse's schedule names the kickoff `gameday`; the tick reads
    `commence_time`. Without the rename every game reads as post-kickoff
    (unparseable -> skipped) and the tick silently does nothing."""
    from nfl_predictor.tracking import forward_tick

    monkeypatch.setattr("nfl_predictor.data.schedules.fetch_upcoming_games",
                        lambda season, week: _schedule_frame())

    games = forward_tick._fetch_slate(2026, 5)

    assert games and all(g.get("commence_time") for g in games)


def _schedule_frame():
    import pandas as pd

    return pd.DataFrame([{
        "game_id": "G1", "season": 2026, "week": 5,
        "gameday": "2026-10-11T18:00:00+00:00",
        "home_team": "ALB", "away_team": "DEN",
    }])


def test_weekly_report_is_scoped_to_its_week(monkeypatch, tmp_path):
    """Two weeks of picks, two reports. Each must show only its own."""
    from nfl_predictor.tracking.forward_report import graded_picks

    for week, game in ((4, "GW4"), (5, "GW5")):
        store.record_game_predictions([{
            "game_id": game, "home_team": "ALB", "away_team": "DEN",
            "commence_time": "2099-09-04T20:20:00", "season": 2026, "week": week,
            "home_win_prob": 0.5, "away_win_prob": 0.5, "home_cover_prob": 0.5,
            "away_cover_prob": 0.5, "over_prob": 0.5, "under_prob": 0.5,
        }])
        store.record_player_prop_predictions([{
            "game_id": game, "player_id": f"00-{week}", "player_name": "P",
            "market": "receiving_yards", "position": "WR", "predicted_value": 50.0,
            "side": "over", "line_at_snapshot": 52.5, "odds_at_snapshot": -110.0,
            "model_p_over": 0.60, "edge_vs_breakeven": 0.076,
        }])

    assert len(graded_picks(season=2026, week=4)) == 1
    assert len(graded_picks(season=2026, week=5)) == 1
    assert len(graded_picks(season=2026)) == 2


def test_a_verified_artifact_is_served_and_a_corrupt_one_is_not(tmp_path, monkeypatch):
    """`load_quantile_artifact` is a bare pickle.loads, so a stale pickle would
    price picks silently -- the incident models/manifest.py documents."""
    import json
    import sqlite3

    from nfl_predictor.models import quantile_registry

    models_dir = _artifacts(tmp_path, monkeypatch, markets=("receiving_yards",))
    # No manifest: predictor_for verifies only when one exists.
    assert forward_tick.predictor_for(models_dir)

    # Now write a manifest whose recorded digest does not match the file.
    manifest = {"quantile_yardage_v1": {"markets": {
        "receiving_yards": {"path": "receiving_yards_quantile_2025.pkl",
                            "sha256": "0" * 64}}}}
    (models_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="do not verify"):
        forward_tick.predictor_for(models_dir)


def test_a_matching_manifest_verifies(tmp_path, monkeypatch):
    import json
    import hashlib

    models_dir = _artifacts(tmp_path, monkeypatch, markets=("receiving_yards",))
    digest = hashlib.sha256(
        (models_dir / "receiving_yards_quantile_2025.pkl").read_bytes()).hexdigest()
    (models_dir / "manifest.json").write_text(json.dumps({"quantile_yardage_v1": {
        "markets": {"receiving_yards": {
            "path": "receiving_yards_quantile_2025.pkl", "sha256": digest}}}}))

    assert forward_tick.predictor_for(models_dir)


# --- the wiring that mutation showed nothing covered ------------------------

def _feature_frame_path(tmp_path):
    """The feature frame on disk, so main() takes the --feature-frame path."""
    frame = _feature_frame()
    path = tmp_path / "frame.parquet"
    frame.to_parquet(path)
    return str(path)


def _players_file(tmp_path):
    path = tmp_path / "players.json"
    path.write_text(json.dumps([
        {"player_id": "00-1", "player_name": "A.J. Brown Jr.", "team": "ALB"},
    ]))
    return str(path)


def test_main_actually_ticks_and_logs_a_pick(monkeypatch, tmp_path):
    """Deleting `run_forward_tick` from `main()` left the whole suite green,
    because every other main() test supplied an empty prop list and took the
    `continue`. This one supplies a real prop, so the wiring is covered."""
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")
    _stub(monkeypatch, props=[PROP])

    exit_code = forward_tick.main([
        "--models-dir", str(_artifacts(tmp_path, monkeypatch)),
        "--games-json", _games_file(tmp_path),
        "--players-path", _players_file(tmp_path),
        "--feature-frame", _feature_frame_path(tmp_path),
    ])

    assert exit_code == 0
    logged = _logged()
    assert len(logged) == 1
    assert logged[0]["player_id"] == "00-1", "the book name was joined to an nflverse id"
    assert logged[0]["edge_vs_breakeven"] >= 0.05


def test_main_reports_a_missing_player_index_as_a_misconfiguration(monkeypatch, tmp_path):
    """A missing flag is not "the book had no props this week". It exits
    non-zero and says so, so it cannot be filed as a clean no-coverage week."""
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")
    _stub(monkeypatch, props=[PROP])

    exit_code = forward_tick.main([
        "--models-dir", str(_artifacts(tmp_path, monkeypatch)),
        "--games-json", _games_file(tmp_path),
        "--feature-frame", _feature_frame_path(tmp_path),
    ])

    assert exit_code == 2
    assert _logged() == []


def test_history_row_for_excludes_the_target_week():
    """Mutation showed `<` -> `<=` in the filter leaves the suite green.

    Returning the target week's own row is a real leak: it would carry that
    week's actuals. The fixture deliberately puts the target week's row LAST so
    a wrong sort or a `<=` picks it."""
    import pandas as pd

    frame = pd.DataFrame([
        {"player_id": "00-1", "season": 2026, "week": 3, "receiving_yards_roll": 30.0},
        {"player_id": "00-1", "season": 2026, "week": 4, "receiving_yards_roll": 99.0},
    ])

    row = forward_tick.history_row_for(frame, "00-1", 2026, 4)

    assert row["receiving_yards_roll"] == 30.0, "the target week's own row leaked in"


def test_history_row_for_excludes_a_later_season():
    import pandas as pd

    frame = pd.DataFrame([
        {"player_id": "00-1", "season": 2026, "week": 12, "receiving_yards_roll": 30.0},
        {"player_id": "00-1", "season": 2027, "week": 1, "receiving_yards_roll": 77.0},
    ])

    row = forward_tick.history_row_for(frame, "00-1", 2027, 1)

    assert row["receiving_yards_roll"] == 30.0


def test_history_row_for_uses_the_TARGET_game_context_not_the_history_row():
    """`is_home`, rest days and weather describe the game being played. Taking
    them from the history row supplies the wrong home/away and the wrong rest
    days -- wrong values, not merely stale ones."""
    import pandas as pd

    frame = pd.DataFrame([{
        "player_id": "00-1", "season": 2026, "week": 3,
        "receiving_yards_roll": 30.0, "is_home": 1, "rest_days": 3.0,
        "is_outdoor": 1, "temp_c": 20.0,
    }])

    # The target game is away, with different rest and weather.
    context = {"is_home": 0, "rest_days": 7.0, "is_outdoor": 0, "temp_c": 2.0}
    row = forward_tick.history_row_for(frame, "00-1", 2026, 4, game_context=context)

    assert row["is_home"] == 0, "week 3's home flag was used for a week-4 away game"
    assert row["rest_days"] == 7.0
    assert row["is_outdoor"] == 0
    assert row["receiving_yards_roll"] == 30.0, "usage still comes from history"


def test_history_row_for_has_no_row_for_an_unknown_player():
    import pandas as pd

    frame = pd.DataFrame([{"player_id": "other", "season": 2026, "week": 3,
                           "receiving_yards_roll": 30.0}])

    assert forward_tick.history_row_for(frame, "00-1", 2026, 4) == {}


# --- the context must come from the TARGET game, not the history row --------

def test_context_columns_come_from_the_game_being_played(monkeypatch, tmp_path):
    """Mutation showed wiring `game_context` had zero coverage.

    Taking `is_home` from the player's last observed row supplies the PREVIOUS
    game's home flag. The prop's team against the game's home_team is exact, and
    the schedule carries `away_rest`/`home_rest`, so both are derivable.
    """
    import pandas as pd

    game = {"home_team": "ALB", "away_team": "DEN",
            "home_rest": 3, "away_rest": 8}
    away_prop = {**PROP, "team": "DEN"}
    home_prop = {**PROP, "team": "ALB"}

    away = forward_tick.game_context_for(game, away_prop)
    home = forward_tick.game_context_for(game, home_prop)

    assert away["is_home"] == 0
    assert home["is_home"] == 1
    assert away["rest_days"] == 8
    assert home["rest_days"] == 3


def test_unknown_weather_is_imputed_from_the_frame_not_set_to_zero():
    """Weather is genuinely unknown at snapshot time. Filling 0 asserts a
    freezing, windless game every week; the frame's median is neutral."""
    import pandas as pd

    frame = pd.DataFrame({"temp_c": [10.0, 20.0, 15.0], "wind_kph": [8.0, 12.0, 10.0]})

    context = forward_tick.game_context_for({"home_team": "ALB"}, PROP, frame)

    assert context["temp_c"] == 15.0
    assert context["wind_kph"] == 10.0
    assert context["temp_c"] != 0.0


def test_a_real_tick_does_not_force_the_context_columns_to_zero(monkeypatch, tmp_path):
    """The end-to-end version: after the tick's own fillna, no context column may
    be 0 purely because it was unknown."""
    monkeypatch.setattr("nfl_predictor.config.ODDS_API_KEY", "test-key")
    _stub(monkeypatch, props=[PROP])

    seen = {}
    original = forward_tick.history_row_for

    def spy(*args, **kwargs):
        row = original(*args, **kwargs)
        seen.update(row)
        return row

    monkeypatch.setattr(forward_tick, "history_row_for", spy)

    forward_tick.main([
        "--models-dir", str(_artifacts(tmp_path, monkeypatch)),
        "--games-json", _games_file(tmp_path),
        "--players-path", _players_file(tmp_path),
        "--feature-frame", _feature_frame_path(tmp_path),
    ])

    assert seen, "the row was never built"
    from nfl_predictor.tracking.forward_tick import TARGET_GAME_COLUMNS

    unknown = [c for c in TARGET_GAME_COLUMNS if c not in seen or seen[c] != seen[c]]
    assert "is_home" not in unknown, "is_home is exactly derivable and was not supplied"

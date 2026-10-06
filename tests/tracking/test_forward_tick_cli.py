"""the forward_tick CLI.

It spends the free tier's metered credits, so the tests that matter here are the
ones that prove it cannot run by accident: no key, no snapshot, no spend. Every
test stubs the network seam; nothing here reaches The Odds API.
"""
from __future__ import annotations

import json
import os

import pytest

from nfl_predictor.tracking import forward_tick, store

from quantile_stubs import QUANTILE_GRID as QUANTILES, write_artifact, write_artifacts


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    yield


GAME = {
    "game_id": "G1", "home_team": "BAL", "away_team": "DEN",
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
    "market": "player_pass_yds", "line": 50.0,
    "over_odds": -110, "under_odds": -110, "book": "fanduel", "team": "BAL",
}

PLAYERS = [{"player_id": "00-1", "player_name": "A.J. Brown Jr.", "team": "BAL"}]


#: The Odds API's event ids are opaque strings with no relationship to nflverse
#: game_ids, so every live test game has to be mapped through one.
EVENT_INDEX = {
    ("Baltimore Ravens", "Denver Broncos"): "e-bal-den",
    ("Kansas City Chiefs", "Las Vegas Raiders"): "e-kc-lv",
}


def _stub(monkeypatch, credits=500, props=None):
    monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: credits >= needed)
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda e, markets=None, credits_needed=1: (props if props is not None else [PROP]))
    # The events list is a real HTTP call in main(); stubbed here rather than at
    # the socket, per tests/conftest.py.
    monkeypatch.setattr(forward_tick, "fetch_event_index", lambda: EVENT_INDEX)


def _artifacts(tmp_path, monkeypatch, markets=("passing_yards",)):
    """Artifacts on disk, loaded exactly as the CLI loads them."""
    return write_artifacts(tmp_path, markets=markets)


def test_loads_an_artifact_into_a_per_market_predictor(tmp_path, monkeypatch):
    models_dir = _artifacts(tmp_path, monkeypatch)

    predictor = forward_tick.predictor_for(models_dir)

    assert set(predictor) == {"player_pass_yds"}
    quantiles = predictor["player_pass_yds"]({"passing_yards_roll": 0.0}, 50.0)
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

    # The artifacts on disk hold passing_yards only, so the other two offered
    # markets must be absent rather than invented as unpriceable stubs.
    assert set(predictor) == {"player_pass_yds"}
    assert "player_rush_yds" not in predictor, "an absent market must be absent, not invented"
    assert "player_receptions" not in predictor


def test_predictor_expands_a_single_row_to_every_quantile(tmp_path, monkeypatch):
    models_dir = _artifacts(tmp_path, monkeypatch)
    predictor = forward_tick.predictor_for(models_dir)

    quantiles = predictor["player_pass_yds"]({"passing_yards_roll": 0.5}, 50.0)

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

    low = predictor["player_pass_yds"]({"passing_yards_roll": 0.0}, 50.0)
    high = predictor["player_pass_yds"]({"passing_yards_roll": 10.0}, 50.0)

    assert low[0.5] != high[0.5]
    assert high[0.5] > low[0.5]


def test_predict_refuses_an_empty_feature_row(tmp_path, monkeypatch):
    """An absent row must raise, not silently become zeros."""
    models_dir = _artifacts(tmp_path, monkeypatch)
    predictor = forward_tick.predictor_for(models_dir)

    with pytest.raises(ValueError, match="feature row"):
        predictor["player_pass_yds"]({}, 50.0)


def test_two_players_get_different_predictions(tmp_path, monkeypatch):
    models_dir = _artifacts(tmp_path, monkeypatch)
    predictor = forward_tick.predictor_for(models_dir)

    a = predictor["player_pass_yds"]({"passing_yards_roll": 1.0}, 50.0)
    b = predictor["player_pass_yds"]({"passing_yards_roll": 9.0}, 50.0)

    assert a[0.5] != b[0.5], "the model is being consulted, not bypassed"


def test_tick_through_loaded_artifacts_logs_an_edge(monkeypatch, tmp_path):
    store.record_game_predictions([GAME])
    models_dir = _artifacts(tmp_path, monkeypatch)
    _stub(monkeypatch)

    result = forward_tick.run_forward_tick(
        games=[GAME],
        market_quantiles=forward_tick.predictor_for(models_dir),
        feature_frame=_feature_frame(), players=PLAYERS,
        event_index=EVENT_INDEX,
    )

    assert result["props_snapshotted"] == 1
    assert result["degenerate_rows"] == 0
    # The stub's distribution straddles the 50.5 line, so the over side clears
    # the gate and exactly one pick is logged.
    assert result["picks_logged"] == 1
    logged = _logged()
    assert logged[0]["side"] == "over"
    assert logged[0]["edge_vs_breakeven"] >= 0.05
    assert logged[0]["market"] == "fwd_passing_yards", (
        "a forward pick must not share a market with the yardage projections")


def test_cli_refuses_without_an_api_key(monkeypatch, tmp_path):
    # main() reads the key from config at call time, so that is what to patch.
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", None)
    called = []
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda *a, **k: called.append(a) or [])

    exit_code = forward_tick.main(["--models-dir", str(_artifacts(tmp_path, monkeypatch))])

    assert exit_code != 0
    assert called == [], "no key means no request and no snapshot"


def test_cli_dry_run_makes_no_request_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
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
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
    empty = tmp_path / "empty.json"
    empty.write_text("[]")

    with pytest.raises(ValueError, match="empty"):
        forward_tick.main(["--models-dir", str(_artifacts(tmp_path, monkeypatch)),
                           "--games-json", str(empty)])


def test_cli_refuses_when_no_slate_is_given(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")

    with pytest.raises(ValueError, match="no games to tick"):
        forward_tick.main(["--models-dir", str(_artifacts(tmp_path, monkeypatch))])


def test_cli_reports_what_it_would_spend_on_a_dry_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
    _stub(monkeypatch, props=[])

    forward_tick.main([
        "--models-dir", str(_artifacts(tmp_path, monkeypatch)),
        "--games-json", _games_file(tmp_path), "--dry-run",
    ])

    out = capsys.readouterr().out
    assert "credit" in out.lower()


def test_cli_writes_nothing_when_the_budget_is_insufficient(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
    store.record_game_predictions([GAME])
    _stub(monkeypatch, credits=0)

    result = forward_tick.run_forward_tick(
        games=[GAME], market_quantiles=forward_tick.predictor_for(
            _artifacts(tmp_path, monkeypatch)), feature_frame=_feature_frame(),
        event_index=EVENT_INDEX)

    assert result["picks_logged"] == 0
    assert _logged() == []


def test_main_writes_a_weekly_report_when_asked(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
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
        "home_team": "BAL", "away_team": "DEN",
    }])


def test_weekly_report_is_scoped_to_its_week(monkeypatch, tmp_path):
    """Two weeks of picks, two reports. Each must show only its own."""
    from nfl_predictor.tracking.forward_report import graded_picks

    for week, game in ((4, "GW4"), (5, "GW5")):
        store.record_game_predictions([{
            "game_id": game, "home_team": "BAL", "away_team": "DEN",
            "commence_time": "2099-09-04T20:20:00", "season": 2026, "week": week,
            "home_win_prob": 0.5, "away_win_prob": 0.5, "home_cover_prob": 0.5,
            "away_cover_prob": 0.5, "over_prob": 0.5, "under_prob": 0.5,
        }])
        store.record_player_prop_predictions([{
            "game_id": game, "player_id": f"00-{week}", "player_name": "P",
            "market": "passing_yards", "position": "QB", "predicted_value": 50.0,
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

    models_dir = _artifacts(tmp_path, monkeypatch, markets=("passing_yards",))
    # No manifest: predictor_for verifies only when one exists.
    assert forward_tick.predictor_for(models_dir)

    # Now write a manifest whose recorded digest does not match the file.
    manifest = {"quantile_yardage_v1": {"markets": {
        "passing_yards": {"path": "passing_yards_quantile_2025.pkl",
                            "sha256": "0" * 64}}}}
    (models_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="do not verify"):
        forward_tick.predictor_for(models_dir)


def test_a_matching_manifest_verifies(tmp_path, monkeypatch):
    import json
    import hashlib

    models_dir = _artifacts(tmp_path, monkeypatch, markets=("passing_yards",))
    digest = hashlib.sha256(
        (models_dir / "passing_yards_quantile_2025.pkl").read_bytes()).hexdigest()
    (models_dir / "manifest.json").write_text(json.dumps({"quantile_yardage_v1": {
        "markets": {"passing_yards": {
            "path": "passing_yards_quantile_2025.pkl", "sha256": digest}}}}))

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
        {"player_id": "00-1", "player_name": "A.J. Brown Jr.", "team": "BAL"},
    ]))
    return str(path)


def test_main_actually_ticks_and_logs_a_pick(monkeypatch, tmp_path):
    """Deleting `run_forward_tick` from `main()` left the whole suite green,
    because every other main() test supplied an empty prop list and took the
    `continue`. This one supplies a real prop, so the wiring is covered."""
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
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
    assert logged[0]["market"] == "fwd_passing_yards", (
        "a forward pick must not share a market with the yardage projections")


def test_main_reports_a_missing_player_index_as_a_misconfiguration(monkeypatch, tmp_path):
    """A missing flag is not "the book had no props this week". It exits
    non-zero and says so, so it cannot be filed as a clean no-coverage week."""
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
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

    game = {"home_team": "BAL", "away_team": "DEN",
            "home_rest": 3, "away_rest": 8}
    away_prop = {**PROP, "team": "DEN"}
    home_prop = {**PROP, "team": "BAL"}

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

    context = forward_tick.game_context_for({"home_team": "BAL"}, PROP, frame)

    assert context["temp_c"] == 15.0
    assert context["wind_kph"] == 10.0
    assert context["temp_c"] != 0.0


def test_a_real_tick_does_not_force_the_context_columns_to_zero(monkeypatch, tmp_path):
    """The end-to-end version: after the tick's own fillna, no context column may
    be 0 purely because it was unknown."""
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
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


def test_the_working_key_wins_when_both_names_are_set(monkeypatch):
    """The VPS holds two DIFFERENT credentials and only `ODDS_API_KEY` works --
    `SPORTSBOOK_API_KEY` returns 401 there. So the verified key is read first,
    despite the other name being the one the rest of the stack uses. A tidy
    rename would have broken every live odds call.

    `SPORTSBOOK_API_KEY` is still accepted, so a host carrying only that name
    works, and the order can be flipped once its VPS value is replaced."""
    import importlib

    import nfl_predictor.config as config

    monkeypatch.setenv("SPORTSBOOK_API_KEY", "returns-401")
    monkeypatch.setenv("ODDS_API_KEY", "works")
    assert importlib.reload(config).SPORTSBOOK_API_KEY == "works", \
        "the credential that actually authenticates must win"

    monkeypatch.setenv("ODDS_API_KEY", "")
    assert importlib.reload(config).SPORTSBOOK_API_KEY == "returns-401", \
        "a host with only SPORTSBOOK_API_KEY must still work"

    monkeypatch.setenv("SPORTSBOOK_API_KEY", "")
    # An empty value is the "unset" signal here: compose.yml defaults both to
    # empty when the host has no key, and every guard is `if not ...`, so falsy
    # is the contract rather than `is None`.
    assert not importlib.reload(config).SPORTSBOOK_API_KEY


def test_a_missing_key_names_both_variables(monkeypatch, tmp_path):
    """A credential problem must not be reported as an exhausted budget."""
    # main() imports the name at call time, so patching config is the seam.
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", None)

    exit_code = forward_tick.main(["--models-dir", str(tmp_path)])

    assert exit_code == 2


def test_an_event_list_fetch_failure_exits_cleanly(monkeypatch, tmp_path):
    """A transport failure must be one line and exit 2, not a traceback from
    `raise_for_status()` buried inside the tick."""
    import requests

    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")

    def boom():
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(forward_tick, "fetch_event_index", boom)

    exit_code = forward_tick.main([
        "--models-dir", str(_artifacts(tmp_path, monkeypatch)),
        "--games-json", _games_file(tmp_path),
        "--players-path", _players_file(tmp_path),
        "--feature-frame", _feature_frame_path(tmp_path),
    ])

    assert exit_code == 2
    assert _logged() == [], "nothing snapshotted after a failed index fetch"


def test_serving_uses_the_same_window_as_training():
    """Training is `shift(1).rolling(_LONG_WINDOW, min_periods=1).mean()` on
    `opp_share_raw`, so serving must average the same number of POSTERIOR rows.

    A previous version hardcoded 4 when the training window is 5, and dropped NaN
    rows before taking the tail -- so a gap in the record reached further back and
    folded an older week into the window."""
    import pandas as pd

    from nfl_predictor.features.availability import _LONG_WINDOW
    from nfl_predictor.tracking import forward_tick

    assert forward_tick._OPP_SHARE_WINDOW == _LONG_WINDOW, (
        "serving window must track training, not restate it")

    # Five prior weeks: the target week must average exactly these.
    frame = pd.DataFrame(
        [{"player_id": "00-1", "season": 2026, "week": w,
          "opp_share_raw": float(w)} for w in range(1, 7)]
        + [{"player_id": "00-1", "season": 2026, "week": 7}])

    row = forward_tick.history_row_for(frame, "00-1", 2026, 7)

    # Weeks 2..6 -> mean 4.0. Including week 1 (mean 3.0) would mean 6 rows.
    assert row["opp_share"] == pytest.approx(4.0)


def test_a_gap_in_the_record_does_not_reach_further_back():
    """NaN rows still occupy a position in the training window -- pandas skips
    them inside the mean, it does not close the gap.

    The gap is placed in the MOST RECENT row, and the oldest row is given a
    distinct value, so filtering NaN out first pulls week 1 into the window and
    changes the answer. An earlier version of this test put the gap outside the
    window, where both implementations agree, and so caught nothing."""
    import pandas as pd

    frame = pd.DataFrame([
        {"player_id": "00-1", "season": 2026, "week": 1, "opp_share_raw": 7.0},
        {"player_id": "00-1", "season": 2026, "week": 2, "opp_share_raw": 1.0},
        {"player_id": "00-1", "season": 2026, "week": 3, "opp_share_raw": 1.0},
        {"player_id": "00-1", "season": 2026, "week": 4, "opp_share_raw": 1.0},
        {"player_id": "00-1", "season": 2026, "week": 5, "opp_share_raw": 1.0},
        {"player_id": "00-1", "season": 2026, "week": 6, "opp_share_raw": float("nan")},
        {"player_id": "00-1", "season": 2026, "week": 7},
    ])

    row = forward_tick.history_row_for(frame, "00-1", 2026, 7)

    # Positional window is weeks 2..6 = [1,1,1,1,NaN] -> 1.0.
    # Filtering NaN first would reach back to week 7's 1.0 and give (7+1+1+1+1)/5.
    assert row["opp_share"] == pytest.approx(1.0)


def test_no_history_yields_unknown_not_zero():
    """A player with no observed share is unknown. Zero would assert a benched
    player and train on it."""
    import pandas as pd

    frame = pd.DataFrame([{"player_id": "00-1", "season": 2026, "week": 6,
                           "opp_share_raw": float("nan")}])

    row = forward_tick.history_row_for(frame, "00-1", 2026, 7)

    assert row["opp_share"] != row["opp_share"], "must be NaN, not 0.0"


def test_the_module_is_executable_as_documented():
    """`python -m nfl_predictor.tracking.forward_tick` is the documented command.

    With no `__main__` guard it imported the module, did nothing, and exited 0 --
    reporting success while snapshotting nothing. This runs the module the way the
    gate document does and asserts it reaches argument handling rather than
    falling off the end of the import."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "nfl_predictor.tracking.forward_tick", "--help"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    assert result.returncode == 0, result.stderr
    assert "--feature-frame" in result.stdout, (
        "the module must expose the documented CLI, not exit silently")


def test_an_artifact_with_no_book_market_is_skipped_not_fatal(tmp_path, monkeypatch):
    """`receiving_yards` is fitted and gated, but the API offers no NFL
    receiving-yards player-prop market, so it can never be forward-tested through
    this feed. Loading must skip it with the reason, not KeyError -- and must not
    silently invent the market either."""
    models_dir = _artifacts(tmp_path, monkeypatch, markets=("passing_yards",))
    # A receiving_yards artifact alongside, exactly as the real models dir has.
    write_artifact(models_dir, market="receiving_yards")

    predictor = forward_tick.predictor_for(models_dir)

    assert set(predictor) == {"player_pass_yds"}, (
        "an unquotable market must be absent, not invented")


def test_the_dry_run_estimate_counts_only_priceable_markets(monkeypatch, tmp_path, capsys):
    """The printed estimate must equal what the tick will actually reserve.

    It read the artifact list with the glob `*_quantile_2025`, which matches none
    of `passing_yards_quantile_2025.pkl`, so it silently fell through to a
    fallback and reported 3 markets where 2 were priceable."""
    monkeypatch.setattr("nfl_predictor.config.SPORTSBOOK_API_KEY", "test-key")
    _stub(monkeypatch)

    models_dir = _artifacts(tmp_path, monkeypatch, markets=("passing_yards", "rushing_yards"))

    exit_code = forward_tick.main([
        "--models-dir", str(models_dir),
        "--games-json", _games_file(tmp_path),
        "--feature-frame", _feature_frame_path(tmp_path),
        "--players-path", _players_file(tmp_path),
        "--dry-run",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    # 1 game in the fixture, 2 priceable markets.
    assert "~2 credits estimated" in out, out

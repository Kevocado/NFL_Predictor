import pandas as pd
import pytest

from nfl_predictor.api import routes


def test_background_tracking_tick_records_and_reconciles(monkeypatch):
    calls = {"recorded": 0, "reconciled": 0, "backfilled": 0}

    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "g1", "home_team": "BAL", "away_team": "KC", "gameday": "2025-09-04",
              "spread_line": -2.5, "total_line": 46.5}]
        ),
    )
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "BAL", "away_team": "KC", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {
        "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
        "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
        "feature_cols": [], "player_feature_cols": [],
    })
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {
        "home_win_prob": 0.6, "away_win_prob": 0.4, "home_cover_prob": 0.55,
        "away_cover_prob": 0.45, "over_prob": 0.52, "under_prob": 0.48,
    })
    monkeypatch.setattr(
        routes.store, "record_game_predictions",
        lambda games: calls.__setitem__("recorded", calls["recorded"] + len(games)) or len(games),
    )
    monkeypatch.setattr(
        routes.store, "reconcile_game_predictions",
        lambda results_df: calls.__setitem__("reconciled", calls["reconciled"] + 1) or 0,
    )
    monkeypatch.setattr(routes.schedules, "fetch_current_season_partial", lambda: pd.DataFrame(columns=["game_id", "home_score", "away_score"]))
    monkeypatch.setattr(
        routes.store, "backfill_unresolved_games",
        lambda schedules_module: calls.__setitem__("backfilled", calls["backfilled"] + 1) or 0,
    )

    routes.background_tracking_tick(season=2025, week=1)

    assert calls["recorded"] == 1
    assert calls["reconciled"] == 1
    assert calls["backfilled"] == 1


def test_a_missing_player_stats_file_does_not_stop_the_game_backfill(monkeypatch, caplog):
    """A regression guard on control flow, not on a log message.

    The tick reconciles props and then backfills *games*, in two separate `try`
    blocks. The first version of the fix for the documented nflverse 404 skipped
    the prop reconcile with a bare `return` -- which exits the whole tick, so a
    missing **player** stats file silently stopped **game** reconciliation too.

    That is strictly worse than the bug it was reporting: game resolution is the
    thing that currently works (30 of 33 resolved on a warm container), and it
    would have gone to 0 with no error anywhere. The existing suite did not catch
    it because its only tick test leaves `completed` empty, so the prop branch is
    never entered -- the same "the test never took this path" hole as everywhere
    else in this session.

    So: `completed` is non-empty here, the prop fetch is empty, and the game
    backfill must still run.
    """
    calls = {"prop_reconciled": 0, "backfilled": 0}

    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: pd.DataFrame(
        [{"game_id": "g1", "home_team": "BAL", "away_team": "KC", "gameday": "2026-09-10",
          "spread_line": -2.5, "total_line": 46.5}]))
    monkeypatch.setattr(routes.schedules, "load_training_data", lambda seasons: pd.DataFrame(
        [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2026-08-29",
          "home_team": "BAL", "away_team": "KC", "home_score": 24, "away_score": 17}]))
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {
        "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
        "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
        "feature_cols": [], "player_feature_cols": []})
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {
        "home_win_prob": 0.6, "away_win_prob": 0.4, "home_cover_prob": 0.55,
        "away_cover_prob": 0.45, "over_prob": 0.52, "under_prob": 0.48})
    monkeypatch.setattr(routes.store, "record_game_predictions", lambda games: len(games))
    monkeypatch.setattr(routes.store, "reconcile_game_predictions", lambda results_df: 0)

    # `completed` is non-empty, so the prop branch IS entered -- the branch the
    # existing test never reaches.
    monkeypatch.setattr(routes.schedules, "fetch_current_season_partial", lambda: pd.DataFrame(
        [{"game_id": "g0", "home_score": 24, "away_score": 17}]))
    # nflverse 404s for the current season; `hub_cache` turns that into an empty
    # frame rather than an error.
    monkeypatch.setattr(routes.player_stats, "fetch_weekly_player_stats",
                        lambda seasons: pd.DataFrame())
    monkeypatch.setattr(
        routes.store, "reconcile_player_prop_predictions",
        lambda stats: calls.__setitem__("prop_reconciled", calls["prop_reconciled"] + 1) or 0)
    monkeypatch.setattr(
        routes.store, "backfill_unresolved_games",
        lambda schedules_module: calls.__setitem__("backfilled", calls["backfilled"] + 1) or 0)

    with caplog.at_level("WARNING"):
        routes.background_tracking_tick(season=2026, week=3)

    assert calls["prop_reconciled"] == 0, "there is nothing to reconcile against an empty frame"
    assert calls["backfilled"] == 1, (
        "a missing PLAYER stats file must not stop GAME reconciliation; "
        "game resolution is the part that works and must not regress because of this"
    )
    assert any("player_props_2026" in r.getMessage() or "player_stats_2026" in r.getMessage()
               or "nflverse" in r.getMessage() for r in caplog.records), (
        "the blocker should be reported at WARNING, not only documented -- it used to be "
        "logged at INFO from hub_cache, which is below the default threshold"
    )

"""Task 2: freeze each game's snapshot inside a lead window, and record which model made it.

`current_season_and_week()` anchors on the date of week 1's first kickoff, so it rolls over mid
week. Two games were therefore never snapshotted pre-kickoff and never tracked at all:

- Thursday-night games, because "this week" had already rolled to the next one by the time the
  window opened;
- CFB week-N+1 games, frozen on the previous Saturday, before that day's results.

A per-game lead window fixes both, and the first snapshot inside it is kept forever
(`INSERT OR IGNORE`), so it is the one the track record grades and the Kalshi feed serves.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from nfl_predictor.api import routes
from nfl_predictor.models import manifest

from fitted_stand_ins import load_pickle as _servable_load_pickle

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def _game(game_id, week, hours_from_now):
    kickoff = (NOW + timedelta(hours=hours_from_now)).replace(tzinfo=None)
    return {"game_id": game_id, "season": 2026, "week": week, "gameday": pd.Timestamp(kickoff),
            "home_team": "CLE", "away_team": "PIT", "home_score": None, "away_score": None,
            "spread_line": -3.0, "total_line": 38.5}


def test_games_to_snapshot_takes_both_weeks_but_only_inside_the_lead_window(monkeypatch):
    weeks = {
        3: pd.DataFrame([_game("in_window", 3, 24), _game("too_early", 3, 100)]),
        4: pd.DataFrame([_game("next_week_soon", 4, 30), _game("next_week_later", 4, 200)]),
    }
    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: weeks.get(week, pd.DataFrame()))

    games = routes._games_to_snapshot(2026, 3, NOW, lead_hours=48)

    assert list(games["game_id"]) == ["in_window", "next_week_soon"]


def test_games_to_snapshot_handles_an_empty_next_week(monkeypatch):
    weeks = {3: pd.DataFrame([_game("in_window", 3, 24)])}
    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: weeks.get(week, pd.DataFrame()))

    games = routes._games_to_snapshot(2026, 3, NOW, lead_hours=48)

    assert list(games["game_id"]) == ["in_window"]


def test_the_lead_window_keeps_past_games_so_the_reconciler_still_sees_them(monkeypatch):
    """The window filters on the upper bound only. `record_game_predictions` rejects a game that
    has already kicked off, and the tick's existing test depends on past games reaching it."""
    weeks = {3: pd.DataFrame([_game("already_played", 3, -30), _game("in_window", 3, 24)])}
    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: weeks.get(week, pd.DataFrame()))

    games = routes._games_to_snapshot(2026, 3, NOW, lead_hours=48)

    assert set(games["game_id"]) == {"already_played", "in_window"}


def test_games_to_snapshot_drops_a_game_with_an_unparseable_kickoff(monkeypatch):
    """A NaT kickoff cannot be placed relative to the window, so it must not be snapshotted at a
    moment that is not 48h before kickoff."""
    bad = _game("no_kickoff", 3, 24)
    bad["gameday"] = pd.NaT
    weeks = {3: pd.DataFrame([bad, _game("in_window", 3, 24)])}
    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: weeks.get(week, pd.DataFrame()))

    games = routes._games_to_snapshot(2026, 3, NOW, lead_hours=48)

    assert list(games["game_id"]) == ["in_window"]


def test_the_lead_window_is_configurable_and_defaults_to_48_hours(monkeypatch):
    assert routes.SNAPSHOT_LEAD_HOURS == 48.0


def test_tracking_tick_records_distribution_and_each_games_own_week(monkeypatch):
    recorded = []
    monkeypatch.setattr(routes, "_games_to_snapshot", lambda season, week, now, lead_hours=None: pd.DataFrame(
        [_game("2026_04_PIT_CLE", 4, 30)]))
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {"model_version": "xgb@t"})
    monkeypatch.setattr(routes, "_load_game_history", lambda season: pd.DataFrame())
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {
        "home_win_prob": 0.4, "away_win_prob": 0.6, "predicted_margin": -3.4, "sigma": 13.2,
        "predicted_total": 39.8, "total_sigma": 12.5, "model_version": "xgb@t",
    })
    monkeypatch.setattr(routes.store, "record_game_predictions", lambda games: recorded.extend(games) or len(games))
    monkeypatch.setattr(routes, "_get_player_props_live", lambda season, week: [])
    monkeypatch.setattr(routes.store, "record_player_prop_predictions", lambda rows: 0)
    monkeypatch.setattr(routes.schedules, "fetch_current_season_partial",
                        lambda: pd.DataFrame(columns=["game_id", "home_score", "away_score"]))
    monkeypatch.setattr(routes.store, "reconcile_game_predictions", lambda df: 0)
    monkeypatch.setattr(routes.store, "backfill_unresolved_games", lambda module: 0)

    routes.background_tracking_tick(season=2026, week=3)

    assert len(recorded) == 1
    row = recorded[0]
    # The game is a week-4 game being snapshotted during the week-3 tick. Labelling it week 3
    # would put it in the wrong week everywhere downstream.
    assert row["week"] == 4
    assert row["commence_time"] == str(_game("x", 4, 30)["gameday"])
    assert row["predicted_margin"] == -3.4 and row["sigma"] == 13.2
    assert row["model_version"] == "xgb@t"


def test_predict_game_from_models_reports_model_version(monkeypatch):
    class _FakeTotalModel:
        def predict(self, _X):
            return [45.0]

    monkeypatch.setattr(
        routes.feature_build, "build_features_for_game",
        lambda home, away, games_df, gameday=None, blocks=None, aux=None, starters=None, game_schedule=None: pd.Series({"rating_diff": 50.0, "home_rest_days": 7.0, "away_rest_days": 7.0}),
    )
    models = {
        "feature_cols": ["rating_diff", "home_rest_days", "away_rest_days"],
        "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
        "total_model": _FakeTotalModel(), "model_version": "elo@2026-09-04T22:12:49+00:00",
    }

    result = routes._predict_game_from_models(models, "KC", "BAL", pd.DataFrame())

    assert result["model_version"] == "elo@2026-09-04T22:12:49+00:00"


def test_model_version_combines_candidate_and_training_time():
    assert manifest.model_version(
        {"chosen_candidate": "xgb", "trained_at": "2026-09-04T22:12:49.750941+00:00"}
    ) == "xgb@2026-09-04T22:12:49.750941+00:00"


def test_load_models_exposes_the_version(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00", "sigma": 12.0,
        "total_sigma": 10.0, "feature_cols": list(manifest.feature_build.FEATURE_COLUMNS),
        # The CODE's player features, not a placeholder and not an arbitrary one:
        # `load_models` refuses a payload whose artefact fingerprint disagrees
        # with `player_usage.PLAYER_FEATURE_COLUMNS` (see
        # `manifest._verify_artifact_fingerprint`), so a hand-built manifest has
        # to carry the real list to get past that guard -- and with it,
        # `assert every fitted player column is one build_features_for_player
        # emits`, which "p" is not.
        #
        # Same story for the top-level `feature_cols` and the game-level half of
        # the fingerprint above it: both must be the CODE's
        # `features.build.FEATURE_COLUMNS`, for the same reason.
        "player_feature_cols": list(manifest.player_usage.PLAYER_FEATURE_COLUMNS),
        "yardage_metrics": [],
        "artifact_fingerprint": manifest.artifact_fingerprint(
            list(manifest.player_usage.PLAYER_FEATURE_COLUMNS),
            list(manifest.feature_build.FEATURE_COLUMNS)),
    })
    # Real estimators carrying the feature lists the committed artefacts carry, not
    # `object()`. `load_models` now reads each artefact's own fitted columns, so a
    # stand-in that records none is refused -- correctly, and this test is about
    # `model_version`, not about the audit. See `tests/fitted_stand_ins.py`.
    #
    # `_artifact_path` is pointed at real paths, as it is in production, so the
    # stand-in can tell a game artefact from a player one by filename. Left as a
    # bare `object()` it could not, and every artefact came back player-shaped.
    tmp_models = tmp_path / "models"
    tmp_models.mkdir()
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)
    monkeypatch.setattr(manifest, "_artifact_path", lambda name: tmp_models / name)

    models = manifest.load_models()

    assert models["model_version"] == "ridge@2026-09-04T22:12:49+00:00"


def test_player_props_are_mapped_to_this_week_s_game_not_next_week_s(monkeypatch):
    """The tick now snapshots this week AND next, but the prop feed is still
    `_get_player_props_live(season, week)` for THIS week and can fall back to the current week. With
    `team_to_game` built from both weeks, a team playing in both had its current-week prop stored
    under next week's game_id -- and `record_player_prop_predictions` is INSERT OR IGNORE, so that
    wrong snapshot would be frozen forever."""
    recorded_props = []
    games = pd.DataFrame([
        {**_game("this_week_game", 3, 24), "home_team": "BAL", "away_team": "KC"},
        {**_game("next_week_game", 4, 24 * 7), "home_team": "BAL", "away_team": "HOU"},
    ])
    monkeypatch.setattr(routes, "_games_to_snapshot", lambda season, week, now, lead_hours=None: games)
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {"model_version": "xgb@t"})
    monkeypatch.setattr(routes, "_load_game_history", lambda season: pd.DataFrame())
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {
        "home_win_prob": 0.4, "away_win_prob": 0.6, "predicted_margin": -3.4, "sigma": 13.2,
        "predicted_total": 39.8, "total_sigma": 12.5, "model_version": "xgb@t",
    })
    monkeypatch.setattr(routes.store, "record_game_predictions", lambda rows: 0)
    monkeypatch.setattr(routes, "_get_player_props_live", lambda season, week: [
        {"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL", "position": "RB",
         "anytime_td_prob": 0.4},
    ])
    monkeypatch.setattr(routes.store, "record_player_prop_predictions",
                        lambda rows: recorded_props.extend(rows) or len(rows))
    monkeypatch.setattr(routes.schedules, "fetch_current_season_partial",
                        lambda: pd.DataFrame(columns=["game_id", "home_score", "away_score"]))
    monkeypatch.setattr(routes.store, "reconcile_game_predictions", lambda df: 0)
    monkeypatch.setattr(routes.store, "backfill_unresolved_games", lambda module: 0)

    routes.background_tracking_tick(season=2026, week=3)

    assert recorded_props, "the prop was dropped entirely, which is a different bug"
    assert {row["game_id"] for row in recorded_props} == {"this_week_game"}

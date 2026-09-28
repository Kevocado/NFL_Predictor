"""The player-props empty state must mean "there are no props", never "props failed".

`GET /api/players/{season}/{week}/props` had one failure mode and one success mode,
both of which came back as `200 []`:

- `_get_player_props_live`'s outer `except Exception: return []` turned a failed
  nflverse fetch (including a read timeout) into an empty list;
- when the season being predicted has no player-stat history at all -- which is
  the live condition today, nflverse 404s `player_stats_2026.parquet` -- every
  player is skipped for want of a pregame feature row, so the same `200 []`
  came back with nothing having failed at all.

A reader cannot tell those apart, and a reader shown "no props" when the pipeline
broke is being told something false.

The simulated timeout below raises the builtin `TimeoutError` rather than
`requests.exceptions.Timeout` on purpose. Nothing in this path uses `requests`:
`nfl_data_py` hands a URL straight to `pandas.read_parquet`, which reads it
through fsspec/urllib, and a stalled socket surfaces there as `socket.timeout` --
an alias of the builtin `TimeoutError` since 3.10. A test that raised the
`requests` exception would pass against code that never sees one.
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api import routes
from nfl_predictor.api.main import app
from nfl_predictor import public_snapshot


def _game_row(season: int, week: int) -> dict:
    return {
        "game_id": f"{season}_{week:02d}_BAL_KC", "season": season, "week": week,
        "gameday": "2026-09-10", "home_team": "BAL", "away_team": "KC",
        "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5,
    }


@pytest.fixture
def client(monkeypatch):
    """A client whose only stub is the schedule fetch -- every other dependency
    (models, player history, feature building, the prop models) is real, so a
    test cannot pass by stubbing away the thing it is meant to exercise."""
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame([_game_row(season, week)]),
    )
    # Carries the real columns the fallback indexes on (`week`). A bare
    # pd.DataFrame() made the code take a KeyError branch rather than the
    # "no games this week" branch the test is about.
    monkeypatch.setattr(
        routes.schedules, "fetch_current_season_partial",
        lambda: pd.DataFrame(columns=["week", "home_team", "away_team"]),
    )
    # The roster fallback is a real nflverse call. Left unstubbed it both reached
    # the network from the test suite and -- worse -- supplied real rows, so a
    # test asserting "this season has no stats" could be satisfied by the
    # per-player path instead of the one it claimed to cover. An empty roster
    # with the right columns keeps every test in this file offline.
    monkeypatch.setattr(
        routes.player_stats, "fetch_seasonal_roster",
        lambda season: pd.DataFrame(columns=["player_id", "player_name", "position", "recent_team"]),
    )
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {"player_models": {"feature_cols": [], "anytime_td": None}},
    )
    return TestClient(app)


def _stat_rows(season: int, week: int) -> pd.DataFrame:
    """One week of real history for one player, so `build_features_for_player`
    has something to roll and the happy path produces a real prop row."""
    return pd.DataFrame([{
        "player_id": "00-001", "player_name": "Pat Mahomes", "position": "QB",
        "recent_team": "KC", "season": season, "week": week,
        "passing_yards": 275.0, "passing_tds": 2, "rushing_yards": 0.0, "rushing_tds": 0,
        "receiving_yards": 0.0, "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 0,
    }])


class _FakeClassifier:
    def predict_proba(self, X):
        return np.array([[0.6, 0.4]])


class _FakeRegressor:
    def predict(self, X):
        return np.array([275.0])


@pytest.fixture
def scoring_models(monkeypatch):
    """Real feature building and real iteration; only the two xgboost models are
    replaced, so `build_features_for_player` returning None -- the condition that
    actually empties the feed -- is exercised for real."""
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {"player_models": {
            "feature_cols": ["passing_yards_roll"],
            "anytime_td": _FakeClassifier(),
            "passing_yards": _FakeRegressor(),
        }},
    )


# --- the timeout -----------------------------------------------------------


def test_a_read_timeout_in_the_schedule_fetch_is_not_served_as_an_empty_200(client, monkeypatch):
    """The whole point: a stalled upstream socket must not reach a reader as
    "no props for this game"."""
    def _timed_out(season, week):
        raise TimeoutError("timed out reading https://github.com/nflverse/nflverse-data")

    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", _timed_out)

    response = client.get("/api/players/2026/3/props")

    assert response.status_code != 200, (
        f"a failed fetch was served as a successful empty list: {response.status_code} {response.text}"
    )
    assert response.status_code == 503
    # The body has to say what happened. A bare 503 is a different kind of
    # silence -- the reader still can't tell it from a genuine outage.
    detail = response.json()["detail"]
    assert "props" in detail.lower()


def test_a_read_timeout_does_not_reach_a_reader_as_an_empty_list_at_the_snapshot_layer(monkeypatch):
    """`public_snapshot._build_week` catches the same failure and used to write
    `[]` into the committed artifact. A transient blip during the nightly build
    would then be served as a real, permanent empty state until the next
    rebuild -- and in PUBLIC_MODE that file IS the response, so fixing the route
    alone would change nothing a reader sees."""
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    monkeypatch.setattr(routes, "_get_games_live", lambda season, week: [])
    monkeypatch.setattr(
        routes, "_load_models_cached", lambda: {"player_models": {"feature_cols": [], "anytime_td": None}}
    )
    previous = {"player_props": [{"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL",
                                 "position": "RB", "anytime_td_prob": 0.42}]}

    week = public_snapshot._build_week(2026, 3, previous=previous)

    assert week["player_props"], "a failed rebuild overwrote good props with an empty list"
    assert week["player_props"] == previous["player_props"]
    assert week["player_props_status"] == "stale"


# --- the data gap that is actually happening ------------------------------


def test_a_season_with_no_player_stats_is_reported_unavailable_not_empty(client, monkeypatch):
    """nflverse 404s `player_stats_2026.parquet` today (see
    docs/player-prop-accuracy-blocker.md). `_load_player_history` therefore
    returns other seasons only, no player in the predicted season has a pregame
    feature row, every player is skipped -- and the route answered `200 []`,
    which reads as "this game has no props"."""
    monkeypatch.setattr(
        routes, "_load_player_history",
        lambda season: _stat_rows(season - 1, 1),
    )

    response = client.get("/api/players/2026/3/props")

    assert response.status_code == 503, (
        f"a season with no stats was served as an empty success: {response.status_code} {response.text}"
    )
    assert "2026" in response.json()["detail"]


def test_the_data_gap_is_raised_not_returned_as_an_empty_list(client, monkeypatch):
    """Pin the distinction at the function the other two callers use, so the
    empty list cannot come back by someone wrapping the raise in a try/except."""
    monkeypatch.setattr(
        routes, "_load_player_history",
        lambda season: _stat_rows(season - 1, 1),
    )

    with pytest.raises(routes.PlayerPropsUnavailable):
        routes._get_player_props_live(2026, 3)


def test_a_per_player_prediction_failure_is_not_silently_dropped_from_a_good_week(client, monkeypatch, scoring_models):
    """The inner `except Exception: continue` swallows a per-player failure
    without recording it. One bad player is a tolerable gap; every player
    failing is an outage wearing the same empty state."""
    monkeypatch.setattr(
        routes, "_load_player_history", lambda season: _stat_rows(season, 1)
    )
    monkeypatch.setattr(
        routes.player_props, "predict_props",
        lambda models, feature_row, position: (_ for _ in ()).throw(
            TimeoutError("model backend timed out")
        ),
    )

    response = client.get("/api/players/2026/3/props")

    assert response.status_code == 503, (
        f"every player failed to predict and the reader got {response.status_code} {response.text}"
    )


# --- the empty state that is honest must survive the fix ------------------


def test_a_week_with_no_games_still_serves_an_empty_list(client, monkeypatch):
    """No games means no props. That is a real answer and must stay a 200 --
    a fix that turns every empty result into a 503 has just moved the lie."""
    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: pd.DataFrame())
    monkeypatch.setattr(
        routes.schedules, "fetch_current_season_partial",
        lambda: pd.DataFrame(columns=["week", "home_team", "away_team"]),
    )

    response = client.get("/api/players/2026/3/props")

    assert response.status_code == 200
    assert response.json() == []


def test_a_fully_scored_week_still_serves_its_props(client, monkeypatch, scoring_models):
    monkeypatch.setattr(
        routes, "_load_player_history", lambda season: _stat_rows(season, 1)
    )

    response = client.get("/api/players/2026/3/props")

    assert response.status_code == 200
    body = response.json()
    assert [row["player_name"] for row in body] == ["Pat Mahomes"]
    assert body[0]["anytime_td_prob"] == pytest.approx(0.4)


# --- PUBLIC_MODE, which is what production actually serves ----------------


def test_a_snapshot_week_with_games_and_no_props_is_not_served_as_an_empty_200(monkeypatch):
    """In PUBLIC_MODE the committed snapshot *is* the response, so the live path
    never runs. This is the surface a reader is actually looking at, and it is
    the one that serves the lie today: data/public_snapshot.json has 14-16 games
    and zero props for weeks 2-7, the window the last build rebuilt."""
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot_cache", {
        "season": 2026,
        "weeks": {"3": {"games": [_game_row(2026, 3)], "predictions": {}, "player_props": []}},
    })

    response = TestClient(app).get("/api/players/2026/3/props")

    assert response.status_code == 503, (
        f"a snapshot week with games and no props served {response.status_code} {response.text}"
    )


def test_a_snapshot_week_with_no_games_and_no_props_still_serves_an_empty_list(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot_cache", {
        "season": 2026,
        "weeks": {"22": {"games": [], "predictions": {}, "player_props": []}},
    })

    response = TestClient(app).get("/api/players/2026/22/props")

    assert response.status_code == 200
    assert response.json() == []


def test_a_snapshot_week_with_props_serves_them(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot_cache", {
        "season": 2026,
        "weeks": {"1": {"games": [_game_row(2026, 1)], "predictions": {}, "player_props": [
            {"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL",
             "position": "RB", "anytime_td_prob": 0.42},
        ], "player_props_status": "ok"}},
    })

    response = TestClient(app).get("/api/players/2026/1/props")

    assert response.status_code == 200
    assert [row["player_name"] for row in response.json()] == ["A. Back"]


def test_a_failed_rebuild_with_nothing_previous_is_marked_unavailable(monkeypatch):
    """Nothing to carry over, so the week has to say so rather than assert
    "no props" -- otherwise the next build repeats the same silent empty."""
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    monkeypatch.setattr(routes, "_get_games_live", lambda season, week: [])
    monkeypatch.setattr(
        routes, "_load_models_cached", lambda: {"player_models": {"feature_cols": [], "anytime_td": None}}
    )

    week = public_snapshot._build_week(2026, 3, previous=None)

    assert week["player_props"] == []
    assert week["player_props_status"] == "unavailable"


def test_the_committed_snapshot_never_serves_an_empty_200_for_a_week_with_games(monkeypatch):
    """The regression guard on the real artifact, not a fixture.

    `data/public_snapshot.json` currently holds a full slate and zero props for
    weeks 2-7 -- the window the last three builds rebuilt. This asserts the
    invariant that makes that honest: any week with games either serves rows or
    answers 503, and only a week with no games at all may serve an empty list.
    Written against the actual file so it keeps holding after the next refresh
    commits a different set of numbers.
    """
    from nfl_predictor.config import PUBLIC_SNAPSHOT_PATH

    if not PUBLIC_SNAPSHOT_PATH.exists():
        pytest.skip("no committed snapshot in this checkout")
    snap = json.loads(PUBLIC_SNAPSHOT_PATH.read_text())
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot_cache", snap)
    client = TestClient(app)

    empty_but_played = []
    for week, block in snap["weeks"].items():
        games, props = block.get("games") or [], block.get("player_props") or []
        response = client.get(f"/api/players/{snap['season']}/{week}/props")
        if games and not props:
            empty_but_played.append(int(week))
            assert response.status_code == 503, (
                f"week {week} has {len(games)} game(s) and no props but served "
                f"{response.status_code} {response.text[:120]}"
            )
        else:
            assert response.status_code == 200
            assert len(response.json()) == len(props), f"week {week} served the wrong rows"

    # Recorded rather than asserted away: the count is the live symptom, and a
    # future snapshot that fixes it should show up as a change in this number
    # rather than as a silently different file.
    print(f"\nweeks with games but no props (all now answered 503): {empty_but_played}")


def test_a_successful_rebuild_is_marked_ok(monkeypatch):
    """The marker is only worth anything if the success path sets it."""
    monkeypatch.setattr(
        routes, "_get_games_live", lambda season, week: [_game_row(season, week)]
    )
    monkeypatch.setattr(
        routes, "_get_player_props_live", lambda season, week: [
            {"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL",
             "position": "RB", "anytime_td_prob": 0.42},
        ]
    )

    week = public_snapshot._build_week(2026, 3, previous=None)

    assert week["player_props_status"] == "ok"

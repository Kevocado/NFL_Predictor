"""The player-props empty state must mean "there are no props", never "props failed".

`GET /api/players/{season}/{week}/props` had one failure mode and one success mode,
both of which came back as `200 []`:

- `_get_player_props_live`'s outer `except Exception: return []` turned a failed
  nflverse fetch into an empty list;
- when the season being predicted has no player-stat history at all -- which is
  the live condition today, nflverse 404s `player_stats_2026.parquet` -- every
  player is skipped for want of a pregame feature row, so the same `200 []` came
  back with nothing having failed at all.

A reader cannot tell those apart, and a reader shown "no props" when the pipeline
broke is being told something false.

**On the simulated failures.** These tests raise the builtin `TimeoutError` and
plain `Exception` subclasses, and none of that is a claim about how nflverse
behaves. Traced against the installed library: `nfl_data_py` hands a URL straight
to `pandas.read_parquet`, which reaches `urllib.request.urlopen` with **no
timeout kwarg**. So a genuinely stalled socket does not raise at all -- it blocks
forever, which is a different and worse failure than anything modelled here, and
one this suite does not and cannot fix by raising an exception. What these tests
actually exercise is the branch under test, `except Exception`: `HTTPError` (the
404), `URLError`, `IncompleteRead` and `ConnectionResetError` are all `Exception`
subclasses, so the branch is identical for all of them. The exception *type* in
each test is a stand-in, not a reproduction, and no assertion here depends on it.

The one thing worth being precise about: an earlier version of this docstring
claimed the failure arrives as `socket.timeout` via "fsspec/urllib". There is no
fsspec and no pyarrow HTTP layer in this path -- it is `urllib.request` directly.
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api import facts as facts_mod
from nfl_predictor.api import routes
from nfl_predictor.api.main import app
from nfl_predictor import public_snapshot
from nfl_predictor.data import player_stats
from tracked_artifacts import require

SEASON = 2026
WEEK = 3

#: The committed artefact this module reads twice, named once. `require()`
#: re-derives it from the path and refuses if the two disagree, so this spelling
#: cannot drift from `config`'s.
SNAPSHOT_REL = "data/public_snapshot.json"

#: Why the artefact has to be there. Quoted into the failure message: a reader
#: told their checkout is broken should not have to go and find out what they
#: are missing.
WHY_SNAPSHOT_REQUIRED = (
    "In PUBLIC_MODE the snapshot IS the response -- routes._public_snapshot_cache "
    "is served verbatim and the live pipeline never runs -- so this is the only "
    "place the empty-state invariant is checked against the file the public "
    "deployment actually serves."
)

#: What would have to be true for its absence to be expected. Nothing in this
#: repository: the file is committed at `main`, so this branch is unreachable here
#: and says so rather than inventing a reason.
WHY_SNAPSHOT_ABSENT_IS_EXPECTED = (
    "Nothing in this repository predicts it: data/public_snapshot.json is "
    "committed at main. If you are reading this, your checkout is older than the "
    "file or was cloned without it -- update the branch rather than committing a "
    "test that tolerates its absence."
)


def _game_row(season: int = SEASON, week: int = WEEK) -> dict:
    return {
        "game_id": f"{season}_{week:02d}_BAL_KC", "season": season, "week": week,
        "gameday": "2026-09-10", "home_team": "BAL", "away_team": "KC",
        "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5,
    }


def _empty_history() -> pd.DataFrame:
    """An empty player-stat frame carrying the real column names.

    The names matter: the code under test indexes `player_history["season"]`, and
    a bare `pd.DataFrame()` raises KeyError there and takes a branch that has
    nothing to do with the one under test. Same reasoning as the `week` column on
    the partial-games stub below.
    """
    return pd.DataFrame(columns=list(player_stats.KEEP_COLUMNS))


def _stat_rows(season: int, week: int, team: str = "KC", player_id: str = "00-001") -> pd.DataFrame:
    """One week of real history for one player, so `build_features_for_player`
    has something to roll and the happy path produces a real prop row."""
    return pd.DataFrame([{
        "player_id": player_id, "player_name": "Pat Mahomes", "position": "QB",
        "recent_team": team, "season": season, "week": week,
        "passing_yards": 275.0, "passing_tds": 2, "rushing_yards": 0.0, "rushing_tds": 0,
        "receiving_yards": 0.0, "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 0,
    }])


class _FakeClassifier:
    def predict_proba(self, X):
        return np.array([[0.6, 0.4]])


class _FakeRegressor:
    def predict(self, X):
        return np.array([275.0])


def _player_models() -> dict:
    return {"player_models": {
        "feature_cols": ["passing_yards_roll"],
        "anytime_td": _FakeClassifier(),
        "passing_yards": _FakeRegressor(),
    }}


@pytest.fixture
def data_seams(monkeypatch):
    """Stub the three data modules the props path reads from, and nothing else.

    Every one of these is a real network seam. `_load_player_history` in
    particular: `routes.py` force-refreshes the current season on every call, so
    leaving it alone made `nfl_data_py` open nine real connections to nflverse
    while every test here passed. `data/cache/` is gitignored, so a clean CI
    checkout has no cached parquet and always reaches the network -- the same
    tests are more exposed on CI than locally, and that is the wrong way round.

    Everything downstream of the seams is the real code: the active-team filter,
    the roster fallback, `build_features_for_player`, the per-player loop and
    every error path. A test cannot pass by stubbing the thing it exercises.
    """
    monkeypatch.setattr(routes, "_load_player_history", lambda season: _empty_history())
    # The same seam for game history. `_build_week` also builds predictions, and
    # `_get_game_prediction_live` -> `_load_game_history` -> `fetch_schedules` ->
    # `nfl_data_py.import_schedules`, which reads http://www.habitatring.com/games.csv
    # over plain HTTP. That one was invisible before because `_build_week`'s
    # per-game `except Exception` swallowed it; the offline guard is what made it
    # visible, which is the guard doing its job.
    monkeypatch.setattr(routes, "_load_game_history", lambda season: pd.DataFrame())
    # `_build_week` also stores each upcoming game's matchup duels, which reads current-season play-by-play
    # (`load_pbp_agg` -> nflverse over the network). Empty efficiency means no duels; that path is unrelated to this one.
    monkeypatch.setattr(public_snapshot, "_current_season_efficiency", lambda season: pd.DataFrame())
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
    monkeypatch.setattr(
        routes.player_stats, "fetch_seasonal_roster",
        lambda season: pd.DataFrame(columns=["player_id", "player_name", "position", "recent_team"]),
    )
    monkeypatch.setattr(routes, "_load_models_cached", lambda: _player_models())
    # The depth chart is the fourth network seam, and the one this file's first
    # round missed: `_get_player_props_live` calls `flags_for_season_week`
    # (routes.py:632) on every request that gets past the roster filter, and
    # with `data/cache/` gitignored a clean checkout has no cached parquet, so
    # it reached for depth_charts_2026.parquet on GitHub. The call is wrapped
    # in `except Exception` downstream, so every such test still passed while
    # the offline guard caught the traffic -- the same shape as the
    # `_load_player_history` leak this fixture was written to close.
    monkeypatch.setattr(
        routes.depth_charts, "flags_for_season_week",
        lambda season, week, cache_dir: {},
    )
    return monkeypatch


@pytest.fixture
def client(data_seams):
    return TestClient(app)


@pytest.fixture
def scoring_models(monkeypatch):
    """Swap in models that predict. Real feature building and real iteration, so
    `build_features_for_player` returning None -- the condition that actually
    empties the feed -- is exercised for real."""
    monkeypatch.setattr(routes, "_load_models_cached", _player_models)


# --- the failure must not arrive as an empty 200 --------------------------


def test_a_failed_upstream_fetch_is_not_served_as_an_empty_200(client, monkeypatch):
    """The whole point: a stalled upstream read must not reach a reader as
    "no props for this game"."""
    def _timed_out(season, week):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", _timed_out)

    response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code != 200, (
        f"a failed fetch was served as a successful empty list: {response.status_code} {response.text}"
    )
    assert response.status_code == 503
    # A bare 503 is a different kind of silence -- the reader still cannot tell
    # this week from a genuine outage.
    assert response.json()["detail"] == routes.PROPS_UNAVAILABLE_DETAIL


def test_the_public_503_body_does_not_leak_the_upstream_cause(client, monkeypatch, caplog):
    """`str(exc)` used to go straight into the response body, and on a urllib
    failure that carries the upstream host: a reader was served
    `HTTPSConnectionPool(host='github.com', port=443): Read timed out`. The detail
    belongs in the log."""
    def _died(season, week):
        raise ConnectionError("HTTPSConnectionPool(host='github.com', port=443): Read timed out")

    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", _died)

    with caplog.at_level("ERROR", logger="nfl_predictor.api.routes"):
        response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    body = response.text
    assert response.status_code == 503
    for leak in ("github.com", "HTTPSConnectionPool", "Read timed out", "ConnectionError"):
        assert leak not in body, f"the upstream cause leaked into the public body: {leak!r} in {body!r}"
    # The log is where the detail belongs, and it must actually be there.
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "github.com" in logged, f"the cause is not in the log either: {logged!r}"


def test_the_live_path_never_leaks_a_raw_exception(client, monkeypatch):
    """The contract that makes the route's single `except PlayerPropsUnavailable`
    sufficient.

    There is deliberately no `except Exception` backstop on the route, so this is
    what stands in for one. A mutation deleting the route handler must be caught
    HERE, by a test that reaches the real wrapping code, rather than by a branch
    at the call site that nothing can reach.
    """
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: (_ for _ in ()).throw(ValueError("something new and unrelated")),
    )

    with pytest.raises(routes.PlayerPropsUnavailable) as caught:
        routes._get_player_props_live(SEASON, WEEK)
    assert "something new and unrelated" in str(caught.value), (
        "the wrapper dropped the cause, so the log would have nothing to say"
    )


def test_a_season_with_no_player_stats_is_reported_unavailable_not_empty(client, monkeypatch, caplog):
    """nflverse 404s `player_stats_2026.parquet` today (see
    docs/player-prop-accuracy-blocker.md). `_load_player_history` therefore
    returns other seasons only, no player in the predicted season has a pregame
    feature row, every player is skipped -- and the route answered `200 []`,
    which reads as "this game has no props"."""
    monkeypatch.setattr(
        routes, "_load_player_history", lambda season: _stat_rows(season - 1, 1)
    )

    with caplog.at_level("ERROR", logger="nfl_predictor.api.routes"):
        response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 503, (
        f"a season with no stats was served as an empty success: {response.status_code} {response.text}"
    )
    # The public body is generic on purpose, so the identifying detail -- which
    # season, and that it is a data gap -- is asserted on the log instead.
    assert response.json()["detail"] == routes.PROPS_UNAVAILABLE_DETAIL
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert f"season {SEASON}" in logged
    assert "player stats" in logged


def test_the_data_gap_is_raised_not_returned_as_an_empty_list(client, monkeypatch):
    """Pin the distinction at the function the other callers use, so the empty
    list cannot come back by someone wrapping the raise in a try/except."""
    monkeypatch.setattr(
        routes, "_load_player_history", lambda season: _stat_rows(season - 1, 1)
    )

    with pytest.raises(routes.PlayerPropsUnavailable):
        routes._get_player_props_live(SEASON, WEEK)


def test_a_prediction_error_on_every_player_is_not_silently_dropped(client, monkeypatch):
    """The inner `except Exception: continue` swallows a per-player error without
    recording it. One bad player is a tolerable gap; every player failing is an
    outage wearing the same empty state."""
    monkeypatch.setattr(
        routes, "_load_player_history", lambda season: _stat_rows(season, 1)
    )
    monkeypatch.setattr(
        routes.player_props, "predict_props",
        lambda models, feature_row, position: (_ for _ in ()).throw(
            ValueError("xgboost model file is corrupt")
        ),
    )

    response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 503, (
        f"every player failed to predict and the reader got {response.status_code} {response.text}"
    )


def test_the_failure_message_does_not_claim_every_player_failed_when_none_did(client, monkeypatch, caplog):
    """The first cut of the total-failure message read

        every player in season 2026 week 3 failed to predict (0 of 1 failed)

    which is self-contradictory, and it is exactly what a `skipped` player
    produced. `skipped` and `failed` are different upstream bugs -- a missing
    pregame feature row versus a dead model -- and the message has to say which.
    """
    # A history row for a team that is NOT in this week's games, so the
    # data-gap check passes (the season does have rows) and the roster fallback
    # supplies the names for the teams that are. Those roster players have no
    # rows at all, so every one of them is skipped.
    monkeypatch.setattr(
        routes, "_load_player_history", lambda season: _stat_rows(season, 1, team="NYJ")
    )
    monkeypatch.setattr(
        routes.player_stats, "fetch_seasonal_roster",
        lambda season: pd.DataFrame([
            {"player_id": "00-777", "player_name": "R. Rookie", "position": "RB", "recent_team": "KC"},
        ]),
    )

    with caplog.at_level("ERROR", logger="nfl_predictor.api.routes"):
        response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 503, (
        f"a week where nobody had a feature row served {response.status_code} {response.text}"
    )
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "every player" not in logged, f"the log still claims every player failed: {logged!r}"
    assert "skipped" in logged, f"the log does not mention the skip: {logged!r}"
    assert "failed to predict" not in logged, f"the log blames a prediction error it did not have: {logged!r}"


# --- the empty state that is honest must survive the fix ------------------


def test_a_week_with_no_games_still_serves_an_empty_list(client, monkeypatch):
    """No games means no props. That is a real answer and must stay a 200 -- a fix
    that turns every empty result into a 503 has just moved the lie."""
    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: pd.DataFrame())
    monkeypatch.setattr(
        routes.schedules, "fetch_current_season_partial",
        lambda: pd.DataFrame(columns=["week", "home_team", "away_team"]),
    )

    response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 200
    assert response.json() == []


def test_a_fully_scored_week_still_serves_its_props(client, monkeypatch):
    monkeypatch.setattr(
        routes, "_load_player_history", lambda season: _stat_rows(season, 1)
    )

    response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 200
    body = response.json()
    assert [row["player_name"] for row in body] == ["Pat Mahomes"]
    assert body[0]["anytime_td_prob"] == pytest.approx(0.4)


# --- PUBLIC_MODE, which is what production actually serves ----------------


def _public(monkeypatch, weeks: dict) -> TestClient:
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot_cache", {"season": SEASON, "weeks": weeks})
    return TestClient(app)


def test_a_snapshot_week_with_games_and_no_props_is_not_served_as_an_empty_200(monkeypatch):
    """In PUBLIC_MODE the committed snapshot *is* the response, so the live path
    never runs. This is the surface a reader actually looks at, and it is the one
    that serves the lie today: data/public_snapshot.json has 14-16 games and zero
    props for weeks 2-7, the window the last build rebuilt."""
    client = _public(monkeypatch, {"3": {"games": [_game_row()], "predictions": {}, "player_props": []}})

    response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 503, (
        f"a snapshot week with games and no props served {response.status_code} {response.text}"
    )


def test_a_snapshot_week_with_no_games_and_no_props_still_serves_an_empty_list(monkeypatch):
    client = _public(monkeypatch, {"22": {"games": [], "predictions": {}, "player_props": []}})

    response = client.get(f"/api/players/{SEASON}/22/props")

    assert response.status_code == 200
    assert response.json() == []


def test_a_snapshot_week_with_props_serves_them(monkeypatch):
    client = _public(monkeypatch, {"1": {
        "games": [_game_row(SEASON, 1)], "predictions": {}, "player_props_status": "ok",
        "player_props": [{"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL",
                          "position": "RB", "anytime_td_prob": 0.42}],
    }})

    response = client.get(f"/api/players/{SEASON}/1/props")

    assert response.status_code == 200
    assert [row["player_name"] for row in response.json()] == ["A. Back"]


# --- the recorded reason is read, not just written ------------------------


def test_a_stale_week_is_served_with_a_stale_marker(monkeypatch):
    """`public_snapshot` writes `player_props_status` on every week. While nothing
    read it, a `stale` week was served as a plain 200 and a reader could not tell
    the rows were from an earlier build -- which is a key nothing reads, i.e. the
    same class of defect as the empty state itself."""
    client = _public(monkeypatch, {"3": {
        "games": [_game_row()], "predictions": {}, "player_props_status": "stale",
        "player_props": [{"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL",
                          "position": "RB", "anytime_td_prob": 0.42}],
    }})

    response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 200
    assert response.headers.get(routes.PROPS_STALE_HEADER) == "true"
    assert [row["player_name"] for row in response.json()] == ["A. Back"]


def test_a_fresh_week_carries_no_stale_marker(monkeypatch):
    """The marker has to mean something. If it were always on, the previous test
    would pass for the wrong reason."""
    client = _public(monkeypatch, {"3": {
        "games": [_game_row()], "predictions": {}, "player_props_status": "ok",
        "player_props": [{"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL",
                          "position": "RB", "anytime_td_prob": 0.42}],
    }})

    response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 200
    assert routes.PROPS_STALE_HEADER not in response.headers


def test_a_recorded_unavailable_status_is_honoured_with_no_games_at_all(monkeypatch):
    """The recorded reason outranks the `games` inference. This shape cannot arise
    from the builder -- a week with no games is recorded `ok`, not `unavailable` --
    so the only way to see it is a hand-edited or future artifact, and a build
    that says "I could not produce these" must be believed."""
    client = _public(monkeypatch, {"22": {
        "games": [], "predictions": {}, "player_props": [], "player_props_status": "unavailable",
    }})

    response = client.get(f"/api/players/{SEASON}/22/props")

    assert response.status_code == 503, (
        f"a build that recorded 'unavailable' was served as an empty week: {response.status_code}"
    )


def test_rows_beat_a_stale_status(monkeypatch):
    """Ordering guard on the shared rule. A `stale` week still carries real rows,
    and blanking them would discard exactly what the carry-forward exists to
    preserve, so the rows win."""
    client = _public(monkeypatch, {"3": {
        "games": [_game_row()], "predictions": {}, "player_props_status": "stale",
        "player_props": [{"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL",
                          "position": "RB", "anytime_td_prob": 0.42}],
    }})

    response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 200
    assert len(response.json()) == 1


# --- the second consumer: the facts bundle --------------------------------


def test_the_facts_panel_agrees_with_the_props_route_about_the_same_week(monkeypatch):
    """Critical, and the reason this file exists twice over.

    `facts._props` is a second reader of the same snapshot, and while it kept its
    own `snap.get("player_props") or []` rule the two panels disagreed about one
    week for one reader: the props page said "could not be loaded" while the
    facts panel said nothing. They now share `snapshot_props_unavailable`.
    """
    client = _public(monkeypatch, {"3": {"games": [_game_row()], "predictions": {}, "player_props": []}})

    rows, unavailable = facts_mod._props(SEASON, WEEK)

    assert unavailable is True
    assert rows == []
    # Same week, same rule: the route is loud, and the panel is not silent.
    assert client.get(f"/api/players/{SEASON}/{WEEK}/props").status_code == 503


def test_the_facts_panel_does_not_claim_unavailable_for_a_week_that_simply_has_no_games(monkeypatch):
    """The flag must not fire on the honest empty, or the panel will cry wolf on
    every late-season week."""
    _public(monkeypatch, {"22": {"games": [], "predictions": {}, "player_props": []}})

    rows, unavailable = facts_mod._props(SEASON, 22)

    assert unavailable is False
    assert rows == []


def test_a_raising_props_route_does_not_take_the_facts_bundle_down_with_it(monkeypatch):
    """The regression: `facts._props` called `routes.get_player_props`, which now
    raises. With no handler at the call site the whole `/facts/{game_id}` bundle
    500'd -- pick, markets, drivers, context and record, all of it, because a
    props panel could not be filled. Before that it degraded to `"players": []`.
    A panel is allowed to be empty; it is not allowed to take the page with it.

    The stub here raises for real. The `live` fixture in tests/test_facts.py
    stubs `get_player_props` with `lambda season, week: []`, which is precisely
    why that file could not see this.
    """
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", False)
    from fastapi import HTTPException
    monkeypatch.setattr(
        facts_mod.routes, "get_player_props",
        lambda season, week: (_ for _ in ()).throw(
            HTTPException(status_code=503, detail="unavailable")
        ),
    )

    rows, unavailable = facts_mod._props(SEASON, WEEK)

    assert (rows, unavailable) == ([], True)


def test_a_404_from_the_props_route_is_not_absorbed(monkeypatch):
    """Only a 503 means 'unavailable'. A 404 means the route itself is gone, and
    turning that into an empty panel would hide a deployment mistake behind a
    quiet `players: []`."""
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", False)
    from fastapi import HTTPException
    monkeypatch.setattr(
        facts_mod.routes, "get_player_props",
        lambda season, week: (_ for _ in ()).throw(HTTPException(status_code=404, detail="gone")),
    )

    with pytest.raises(HTTPException):
        facts_mod._props(SEASON, WEEK)


# --- the snapshot builder ------------------------------------------------


def test_a_failed_rebuild_keeps_the_previous_props(data_seams, monkeypatch):
    """`_build_week` used to catch the failure and write `[]`, which is how the
    committed file came to hold a full slate and no props for weeks 2-7: a
    transient blip frozen into a tracked artifact and then served as a permanent
    empty state. In PUBLIC_MODE that file IS the response, so fixing the route
    alone would change nothing a reader sees."""
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: (_ for _ in ()).throw(TimeoutError("read timed out")),
    )
    monkeypatch.setattr(routes, "_get_games_live", lambda season, week: [])
    previous = {"player_props": [{"player_id": "p1", "player_name": "A. Back", "recent_team": "BAL",
                                 "position": "RB", "anytime_td_prob": 0.42}]}

    week = public_snapshot._build_week(SEASON, WEEK, previous=previous)

    assert week["player_props"], "a failed rebuild overwrote good props with an empty list"
    assert week["player_props"] == previous["player_props"]
    assert week["player_props_status"] == "stale"


def test_a_failed_rebuild_with_nothing_previous_is_marked_unavailable(data_seams, monkeypatch):
    """Nothing to carry, so the week has to say so rather than assert "no props" --
    otherwise the next build repeats the same silent empty."""
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: (_ for _ in ()).throw(TimeoutError("read timed out")),
    )
    monkeypatch.setattr(routes, "_get_games_live", lambda season, week: [])

    week = public_snapshot._build_week(SEASON, WEEK, previous=None)

    assert week["player_props"] == []
    assert week["player_props_status"] == "unavailable"


def test_a_successful_rebuild_is_marked_ok(data_seams, monkeypatch):
    """The marker is only worth anything if the success path sets it."""
    monkeypatch.setattr(routes, "_get_games_live", lambda season, week: [_game_row()])
    # Predictions are not what this file is about, and building one for real walks
    # straight into the schedule fetch.
    monkeypatch.setattr(
        routes, "_get_game_prediction_live",
        lambda season, week, game_id: {"home_win_prob": 0.5, "away_win_prob": 0.5},
    )
    def fake_props(season, week, out_players=None):
        # `out_players` is the collector `_build_week` passes so the artifact
        # stores the out entries beside the rows they were removed from.
        # Accepting it is not optional: production is PUBLIC_MODE and serves this
        # artifact, so an out list that only existed in the live path would never
        # reach a page.
        if out_players is not None:
            out_players.append({"player_id": "p9", "player_name": "Gone Guy", "report_status": "Out"})
        return [{"player_id": "p1", "player_name": "A. Back",
                 "recent_team": "BAL", "position": "RB", "anytime_td_prob": 0.42}]

    monkeypatch.setattr(routes, "_get_player_props_live", fake_props)

    week = public_snapshot._build_week(SEASON, WEEK, previous=None)

    assert week["player_props_status"] == "ok"
    assert [e["player_name"] for e in week["player_props_out"]] == ["Gone Guy"]


def _assert_the_empty_state_invariant(client, snap: dict) -> list[int]:
    """Every week in `snap` honours the empty-state invariant. Returns the offenders.

    Named, and taking the snapshot as an argument, so the three "this ran but
    examined nothing" guards inside it are reachable from a test. Inline in the
    one test that runs against the committed artefact they would be decoration:
    the artefact has 22 weeks and 18 of them carry games, so disabling any of
    those three assertions changes nothing observable and nothing would say so.
    `test_the_guards_that_stop_this_from_being_vacuous_bite` is what makes them
    real, and it can only exist because this is a function.

    The invariant: a week with games either serves rows or answers 503; only a
    week with no games at all may serve an empty list.
    """
    weeks = snap["weeks"]
    assert weeks, (
        f"the snapshot holds no weeks, so this check asked the props route "
        f"nothing at all. An empty `weeks` is a real state -- a build before the "
        f"window opens -- and a guard has to be able to say so rather than pass."
    )

    empty_but_played: list[int] = []
    weeks_with_games = 0
    for week, block in weeks.items():
        # `"games" in block` rather than `block.get("games") or []`: a renamed or
        # dropped key would read as "a week with no games" and sail through the
        # else branch below asserting 200 and zero rows, so the check would pass
        # on a snapshot whose every week it could not see.
        assert "games" in block, (
            f"week {week} has no `games` key (keys: {sorted(block)}). The "
            f"games/player_props spellings are what make 'empty but played' "
            f"distinguishable from 'not played'; if either was renamed, this loop "
            f"can no longer tell those apart and must be rewritten, not left to pass."
        )
        games, props = block.get("games") or [], block.get("player_props") or []
        weeks_with_games += bool(games)
        response = client.get(f"/api/players/{snap['season']}/{week}/props")
        if games and not props:
            empty_but_played.append(int(week))
            assert response.status_code == 503, (
                f"week {week} has {len(games)} game(s) and no props but served "
                f"{response.status_code} {response.text[:120]}"
            )
        else:
            assert response.status_code == 200, (
                f"week {week} served {response.status_code} {response.text[:120]}"
            )
            assert len(response.json()) == len(props), f"week {week} served the wrong rows"

    # The 503 arm is the point of the check, so a run in which it was never
    # reached proved nothing -- including a run against a snapshot that had quietly
    # stopped carrying any games at all.
    assert weeks_with_games, (
        f"none of the {len(weeks)} weeks carries a game, so the empty-but-played "
        f"case never arose and this asserted only that empty weeks serve an "
        f"empty list."
    )
    return empty_but_played


#: Payloads that must each make `_assert_the_empty_state_invariant` raise, with
#: the fragment of each message that says why. The committed snapshot is never any
#: of these, so these are the only way the guards are ever exercised.
UNJUDGEABLE_SNAPSHOTS = (
    pytest.param(
        {}, "holds no weeks",
        id="no-weeks",
    ),
    pytest.param(
        {"22": {"games": [], "player_props": [], "predictions": {}}},
        "none of the 1 weeks carries a game",
        id="no-week-has-games",
    ),
    pytest.param(
        {"3": {"player_props": [], "predictions": {}}},
        "week 3 has no `games` key",
        id="a-week-lost-its-games-key",
    ),
)


@pytest.mark.parametrize("weeks, expected", UNJUDGEABLE_SNAPSHOTS)
def test_the_guards_that_stop_this_from_being_vacuous_bite(weeks, expected, monkeypatch):
    """The three guards inside `_assert_the_empty_state_invariant`, each proved.

    Without this they are assertions that cannot fail. The committed snapshot is
    non-empty, carries games, and spells the key `games`, so on the real artefact
    all three are satisfied trivially -- deleting them changes nothing observable
    and the next person to run a mutation check has no signal that the coverage
    went. Here each guard is handed the one payload it exists to reject.

    The third is the subtle one and the reason the first two were not enough: a
    snapshot that lost its `games` key entirely would read as "every week had no
    games", take the else branch, and assert `200` with zero rows for each -- a
    complete, silent, green pass over weeks it could not see.
    """
    snap = {"season": SEASON, "weeks": weeks}
    with pytest.raises(AssertionError, match=expected):
        _assert_the_empty_state_invariant(_public(monkeypatch, weeks), snap)


def test_the_invariant_check_passes_on_a_snapshot_it_can_judge(monkeypatch):
    """The other direction, so the guards above cannot be passed by raising always."""
    weeks = {
        "3": {"games": [_game_row()], "player_props": [], "predictions": {}},
        "22": {"games": [], "player_props": [], "predictions": {}},
    }
    offenders = _assert_the_empty_state_invariant(
        _public(monkeypatch, weeks), {"season": SEASON, "weeks": weeks}
    )
    assert offenders == [3], offenders


def test_the_committed_snapshot_never_serves_an_empty_200_for_a_week_with_games(monkeypatch):
    """The regression guard on the real artifact, not a fixture.

    This asserts the invariant, not a set of numbers: any week with games either
    serves rows or answers 503, and only a week with no games at all may serve an
    empty list. It is written against the actual file so it keeps holding after
    the next refresh commits different numbers -- which it has already done
    twice, and is why no week is named here.

    Three ways this could stop testing and still pass, all closed: the artefact is
    absent (`require`, which fails where this used to skip); the artefact holds no
    weeks, so the loop body never ran; and no week carries games, so the 503
    branch was never reached. The latter two were open, and the first of them is a
    live risk rather than a thought experiment -- a snapshot written before the
    season window opens has an empty `weeks`, and this would have passed on it
    having asked nothing of the route.

    The loop itself lives in `_assert_the_empty_state_invariant` so those two
    guards can be reached; see
    `test_the_guards_that_stop_this_from_being_vacuous_bite`.
    """
    from nfl_predictor.config import PUBLIC_SNAPSHOT_PATH

    path = require(
        PUBLIC_SNAPSHOT_PATH,
        rel=SNAPSHOT_REL,
        required_because=WHY_SNAPSHOT_REQUIRED,
        expected_absent_because=WHY_SNAPSHOT_ABSENT_IS_EXPECTED,
    )
    snap = json.loads(path.read_text())
    empty_but_played = _assert_the_empty_state_invariant(
        _public(monkeypatch, snap["weeks"]), snap
    )

    # Recorded rather than asserted away: the count is the live symptom, and a
    # future snapshot that fixes it should show up as a change in this number
    # rather than as a silently different file.
    print(f"\nweeks with games but no props (all now answered 503): {empty_but_played}")


# --- coverage holes the review's probes found ------------------------------
#
# Both of the mutations below SURVIVED round 1, which is the only reason they are
# written down here. Neither is a behavioural change: they are the two places
# where the code has a branch and no test looks at it.


def test_a_roster_fallback_failure_does_not_turn_a_good_week_into_an_empty_one(client, monkeypatch, caplog):
    """`routes.py:590`: the roster fallback's `except` logs a warning and carries
    on. Nothing asserted that, so `except Exception: pass` survived every mutation
    run -- and passing would make the code *quieter* on a real nflverse outage,
    which is the wrong direction for a file whose whole point is that failures are
    visible.

    The shape: the week has real history for one of its teams, so `latest_players`
    is non-empty, but the other team has no current-season rows, so `missing_teams`
    is non-empty and the roster fetch runs. It fails. The good rows must still be
    served, and the failure must be recorded.
    """
    monkeypatch.setattr(
        routes, "_load_player_history",
        lambda season: pd.concat(
            [_stat_rows(season, 1, team="KC"), _stat_rows(season, 1, team="NYJ", player_id="00-002")],
            ignore_index=True,
        ),
    )
    # Only KC is in the week, so missing_teams is empty and the fallback never
    # runs -- which is the point: the test has to make it run to be worth anything.
    active = pd.DataFrame([{**_game_row(), "home_team": "BAL", "away_team": "KC"}])
    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: active)
    monkeypatch.setattr(
        routes.player_stats, "fetch_seasonal_roster",
        lambda season: (_ for _ in ()).throw(ConnectionError("nflverse roster fetch failed")),
    )

    with caplog.at_level("WARNING", logger="nfl_predictor.api.routes"):
        response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert response.status_code == 200, (
        f"a roster-fallback failure took out a week that had real props: "
        f"{response.status_code} {response.text[:160]}"
    )
    names = {row["player_name"] for row in response.json()}
    assert names == {"Pat Mahomes"}, f"the good row was lost: {names}"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "roster" in logged.lower(), f"the fallback failure was swallowed silently: {logged!r}"


def test_the_roster_fallback_actually_runs_in_the_test_above(client, monkeypatch):
    """A guard on the guard.

    `test_a_roster_fallback_failure_...` is only meaningful if the fallback is
    entered. If a future change made `missing_teams` empty in this shape, the
    exception would never be raised, `caplog` would be empty, and the test would
    still pass on the 200 and the one row -- asserting nothing about the failure
    path. So the seam is proved to be reached.
    """
    calls = []

    def _boom(season):
        calls.append(season)
        raise ConnectionError("nflverse roster fetch failed")

    monkeypatch.setattr(
        routes, "_load_player_history",
        lambda season: _stat_rows(season, 1, team="KC"),
    )
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame([{**_game_row(), "home_team": "BAL", "away_team": "KC"}]),
    )
    monkeypatch.setattr(routes.player_stats, "fetch_seasonal_roster", _boom)

    response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

    assert calls, (
        "the roster fallback was never reached, so the failure-path test beside this "
        "one would pass without exercising anything"
    )
    assert response.status_code == 200

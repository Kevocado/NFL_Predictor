"""Tests for the read-only /facts bundle the match explainer consumes.

Every test here is offline: the snapshot, the schedule and the tracking
store are all injected, so nothing reaches nflverse or the model files.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, model_validator

from nfl_predictor.api import facts as facts_mod
from nfl_predictor.api.main import app

# The contract, copied from predictor-hub/services/explainer/explainer/facts.py
# so a change on either side has to be made deliberately on both.
class Market(BaseModel):
    market: str
    model_config = ConfigDict(extra="allow")


class Facts(BaseModel):
    sport: Literal["pl", "f1", "nfl", "cfb", "nba"]
    id: str
    title: str
    starts_at: str
    status: Literal["upcoming", "live", "final"]
    pick_timing: Literal["pre_kickoff", "rebuilt", "none"]
    pick: dict | None = None
    markets: list[Market] = []
    drivers: list[dict] = []
    context: dict = {}
    players: list[dict] = []
    record: dict | None = None
    result: dict | None = None

    @model_validator(mode="after")
    def _rebuilt_never_won(self) -> "Facts":
        if self.pick_timing == "rebuilt" and self.result and "pick_won" in self.result:
            raise ValueError("a rebuilt pick cannot carry result.pick_won")
        return self


NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
GAME_ID = "2026_05_KC_BAL"


def _game(**over):
    game = {
        "game_id": GAME_ID,
        "season": 2026,
        "week": 5,
        "gameday": "2026-10-05T00:20:00",  # zoneless UTC, as the schedule stores it
        "home_team": "BAL",
        "away_team": "KC",
        "home_score": None,
        "away_score": None,
        "home_rest": 7,
        "away_rest": 6,
        "div_game": 0,
        "roof": "outdoors",
        "surface": "grass",
        "temp": 62.0,
        "wind": 8.0,
        "spread_line": 2.5,  # nflverse: positive means HOME is favoured
        "total_line": 45.5,
    }
    game.update(over)
    return game


def _prediction(**over):
    pred = {
        "home_win_prob": 0.62,
        "away_win_prob": 0.38,
        "home_cover_prob": 0.56,
        "away_cover_prob": 0.44,
        "over_prob": 0.61,
        "under_prob": 0.39,
        "predicted_margin": 3.4,
        "predicted_total": 47.8,
    }
    pred.update(over)
    return pred


def _prop(player, team, position, **yards):
    prop = {
        "player_id": f"00-{player}",
        "player_name": f"Player {player}",
        "recent_team": team,
        "position": position,
        "anytime_td_prob": 0.5,
    }
    prop.update(yards)
    return prop


def _snapshot(game=None, prediction=None, props=None):
    return {
        "generated_at": "2026-10-01T06:00:00Z",
        "season": 2026,
        "current_week": 5,
        "weeks": {
            "5": {
                "games": [game if game is not None else _game()],
                "predictions": {GAME_ID: prediction if prediction is not None else _prediction()},
                "player_props": props if props is not None else [
                    _prop("001", "BAL", "QB", passing_yards=248.0),
                    _prop("002", "BAL", "WR", receiving_yards=92.0),
                    _prop("003", "KC", "RB", rushing_yards=88.0),
                    _prop("004", "KC", "WR", receiving_yards=61.0),
                ],
            }
        },
    }


def _week_rows(**over):
    row = {
        "game_id": GAME_ID,
        "status": "pending",
        "rebuilt": False,
        "home_win_prob": 0.62,
        "away_win_prob": 0.38,
        "verdict": None,
    }
    row.update(over)
    return [row]


@pytest.fixture
def public(monkeypatch):
    """PUBLIC_MODE on, with a fake snapshot and no tracking DB."""
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod.routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(facts_mod, "_now", lambda: NOW)
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: _week_rows())
    monkeypatch.setattr(
        facts_mod.store, "get_track_record",
        lambda: {"games": {"n_resolved": 66, "pct_moneyline_correct": 0.62, "n_rebuilt": 3}},
    )
    return TestClient(app)


def _install_snapshot(monkeypatch, snapshot):
    monkeypatch.setattr(facts_mod, "_public_snapshot", lambda: snapshot)


# --- the contract -------------------------------------------------------

def test_public_mode_bundle_validates_against_the_contract(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    Facts(**body)  # raises if the shape drifted from the shared contract
    assert body["sport"] == "nfl"
    assert body["id"] == GAME_ID
    assert body["title"] == "KC at BAL"
    assert body["starts_at"] == "2026-10-05T00:20:00Z"
    assert body["status"] == "upcoming"
    assert body["pick"] == {"label": "BAL", "prob": 0.62}
    assert {m["market"] for m in body["markets"]} == {"moneyline", "spread", "total"}


def test_markets_use_site_wording_and_nflverse_sign(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()
    by_market = {m["market"]: m for m in body["markets"]}

    # nflverse spread_line +2.5 means BAL is favoured, so the site's wording
    # is the home team giving points: "BAL -2.5".
    assert by_market["spread"]["line"] == "BAL -2.5"
    assert by_market["spread"]["model_margin"] == pytest.approx(3.4)
    assert by_market["spread"]["model_cover_prob"] == pytest.approx(0.56)
    assert by_market["total"]["line"] == pytest.approx(45.5)
    assert by_market["total"]["model_total"] == pytest.approx(47.8)
    assert by_market["total"]["model_over_prob"] == pytest.approx(0.61)
    assert by_market["moneyline"]["model"] == {"BAL": pytest.approx(0.62), "KC": pytest.approx(0.38)}


def test_context_carries_weather_roof_and_rest_but_no_cached_injuries(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["context"]["roof"] == "outdoors"
    assert "62" in body["context"]["weather"]
    assert body["context"]["rest"] == "Rest 7 v 6 days"
    # The cached nflverse injury file goes stale; ESPN news covers injuries.
    assert "injuries" not in body["context"]


def test_players_are_the_top_three_by_projected_yards(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert [p["name"] for p in body["players"]] == ["Player 001", "Player 002", "Player 003"]
    assert {p["team"] for p in body["players"]} == {"BAL", "KC"}


def test_record_reports_pre_kickoff_hits_over_settled(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["record"] == {
        "label": "Picks made before kickoff",
        "hits": 41,
        "settled": 66,
    }


# --- pick_timing: the three cases ---------------------------------------

def test_pick_timing_is_none_when_no_stored_row(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: [])

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "none"
    assert body["pick"] is None


def test_pick_timing_is_rebuilt_when_snapshotted_after_kickoff(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(rebuilt=True),
    )

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "rebuilt"
    assert body["pick"] is not None


def test_pick_timing_is_pre_kickoff_for_a_snapshotted_row(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick_timing"] == "pre_kickoff"
    assert body["pick"]["label"] == "BAL"


# --- started games: the pick is only ever the stored pre-start one -------

def test_started_game_uses_the_stored_pre_start_pick(public, monkeypatch):
    # The public snapshot rebuilds the current and previous week every 3 h,
    # so for a started game its prediction is today's model, recomputed after
    # kickoff. Only the tracking row holds the pick made before kickoff.
    started = _game(gameday="2026-09-20T00:20:00")  # already kicked off
    _install_snapshot(monkeypatch, _snapshot(game=started, prediction=_prediction(home_win_prob=0.71, away_win_prob=0.29)))

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "live"
    assert body["pick"] == {"label": "BAL", "prob": 0.62}  # the stored row, not the rebuilt snapshot
    # The row carries no margin/total, so no post-kickoff spread or total is quoted.
    assert {m["market"] for m in body["markets"]} <= {"moneyline"}
    assert "0.71" not in str(body["markets"])


def test_final_judges_the_stored_pick_even_when_the_rebuilt_snapshot_flipped(public, monkeypatch):
    final = _game(gameday="2026-09-20T00:20:00", home_score=20, away_score=27)
    # After kickoff the snapshot rebuilt KC as favourite; before kickoff the stored pick was BAL.
    _install_snapshot(monkeypatch, _snapshot(game=final, prediction=_prediction(home_win_prob=0.40, away_win_prob=0.60)))
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week",
                        lambda season, week, games_df: _week_rows(status="resolved"))

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick"] == {"label": "BAL", "prob": 0.62}
    assert body["result"]["pick_won"] is False  # BAL lost 20-27
    Facts(**body)


def test_started_game_without_a_stored_pick_has_no_pick(public, monkeypatch):
    started = _game(gameday="2026-09-20T00:20:00")
    snap = _snapshot(game=started)
    _install_snapshot(monkeypatch, snap)
    # Nothing was stored before kickoff: no tracking row (the snapshot's own
    # prediction is a post-kickoff rebuild and must not stand in for one).
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: [])

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "live"
    assert body["pick"] is None
    assert body["pick_timing"] == "none"
    # No pre-start snapshot means no honest spread/total to quote either.
    assert body["markets"] == []


def test_started_game_never_calls_a_live_model(public, monkeypatch):
    started = _game(gameday="2026-09-20T00:20:00")
    _install_snapshot(monkeypatch, _snapshot(game=started))

    def explode(*args, **kwargs):
        raise AssertionError("public mode must not compute a live model")

    monkeypatch.setattr(facts_mod.routes, "_get_game_prediction_live", explode)
    monkeypatch.setattr(facts_mod.routes, "_load_models_cached", explode)

    assert public.get(f"/facts/{GAME_ID}").status_code == 200


def test_public_mode_never_computes_a_live_model(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    def explode(*args, **kwargs):
        raise AssertionError("public mode must not compute a live model")

    monkeypatch.setattr(facts_mod.routes, "_get_game_prediction_live", explode)
    monkeypatch.setattr(facts_mod.routes, "_load_models_cached", explode)
    monkeypatch.setattr(facts_mod.routes.feature_build, "build_features_for_game", explode)

    assert public.get(f"/facts/{GAME_ID}").status_code == 200


# --- finals -------------------------------------------------------------

def test_final_has_a_score_and_pick_won_when_pre_kickoff(public, monkeypatch):
    final = _game(gameday="2026-09-20T00:20:00", home_score=27, away_score=20)
    _install_snapshot(monkeypatch, _snapshot(game=final))
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(status="resolved"),
    )

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "final"
    assert body["result"]["score"] == "BAL 27-20"
    # The pick was BAL at 0.62 and BAL won 27-20.
    assert body["result"]["pick_won"] is True
    Facts(**body)


def test_final_omits_pick_won_for_a_rebuilt_pick(public, monkeypatch):
    final = _game(gameday="2026-09-20T00:20:00", home_score=20, away_score=27)
    _install_snapshot(monkeypatch, _snapshot(game=final))
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(status="resolved", rebuilt=True),
    )

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "final"
    assert body["pick_timing"] == "rebuilt"
    assert "score" in body["result"]
    assert "pick_won" not in body["result"]


def test_upcoming_game_has_no_result(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "upcoming"
    assert body["result"] is None


# --- /facts/upcoming ----------------------------------------------------

def test_upcoming_lists_only_games_inside_the_window(public, monkeypatch):
    soon = _game(game_id="2026_05_KC_BAL", gameday=(NOW + timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%S"))
    later = _game(game_id="2026_05_SF_PIT", gameday=(NOW + timedelta(hours=100)).strftime("%Y-%m-%dT%H:%M:%S"))
    past = _game(game_id="2026_05_NE_CIN", gameday=(NOW - timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%S"))
    snap = _snapshot()
    snap["weeks"]["5"]["games"] = [soon, later, past]
    _install_snapshot(monkeypatch, snap)

    body = public.get("/facts/upcoming?hours=72").json()

    assert body["ids"] == [GAME_ID]


def test_upcoming_defaults_to_72_hours(public, monkeypatch):
    soon = _game(gameday=(NOW + timedelta(hours=50)).strftime("%Y-%m-%dT%H:%M:%S"))
    _install_snapshot(monkeypatch, _snapshot(game=soon))

    assert public.get("/facts/upcoming").json()["ids"] == [GAME_ID]


# --- live (non-public) mode: the rule that actually bites ---------------

@pytest.fixture
def live(monkeypatch):
    """PUBLIC_MODE off: the facts bundle must still never call the model for
    a game that has started."""
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", False)
    monkeypatch.setattr(facts_mod, "_now", lambda: NOW)
    monkeypatch.setattr(
        facts_mod.routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame([_game(gameday="2026-09-20T00:20:00")]),
    )
    monkeypatch.setattr(facts_mod.routes, "get_games", lambda season, week: [_game(gameday="2026-09-20T00:20:00")])
    monkeypatch.setattr(facts_mod.routes, "get_player_props", lambda season, week: [])
    monkeypatch.setattr(
        facts_mod.store, "get_track_record",
        lambda: {"games": {"n_resolved": 10, "pct_moneyline_correct": 0.5, "n_rebuilt": 0}},
    )
    return TestClient(app)


def test_live_started_game_never_computes_a_model_and_uses_the_stored_row(live, monkeypatch):
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(home_win_prob=0.71, away_win_prob=0.29),
    )

    def explode(*args, **kwargs):
        raise AssertionError("a started game must never be described by today's model")

    monkeypatch.setattr(facts_mod.routes, "get_game_prediction", explode)
    monkeypatch.setattr(facts_mod.routes, "_get_game_prediction_live", explode)
    monkeypatch.setattr(facts_mod.routes, "_load_models_cached", explode)

    body = live.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "live"
    assert body["pick"] == {"label": "BAL", "prob": 0.71}
    # The stored pre-start row carries no margin or total, so those markets
    # are honestly absent rather than recomputed.
    assert [m["market"] for m in body["markets"]] == ["moneyline"]


def test_live_upcoming_game_does_use_the_current_model(live, monkeypatch):
    monkeypatch.setattr(
        facts_mod.routes.schedules, "fetch_week_games", lambda season, week: pd.DataFrame([_game()])
    )
    monkeypatch.setattr(facts_mod.routes, "get_games", lambda season, week: [_game()])
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: _week_rows())
    monkeypatch.setattr(
        facts_mod.routes, "get_game_prediction",
        lambda season, week, game_id: _prediction(home_win_prob=0.66, away_win_prob=0.34),
    )

    body = live.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "upcoming"
    assert body["pick"] == {"label": "BAL", "prob": 0.66}
    assert {m["market"] for m in body["markets"]} == {"moneyline", "spread", "total"}


# --- unknowns -----------------------------------------------------------

def test_unknown_game_id_is_404(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    assert public.get("/facts/2026_05_XX_YY").status_code == 404


def test_started_game_quotes_no_rebuilt_player_projections(public, monkeypatch):
    # The snapshot's player props for the current/previous week are rebuilt
    # after kickoff, like its predictions; a started game quotes none.
    started = _game(gameday="2026-09-20T00:20:00")
    _install_snapshot(monkeypatch, _snapshot(game=started))

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "live"
    assert body["players"] == []

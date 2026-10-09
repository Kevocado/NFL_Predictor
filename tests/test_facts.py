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
    # Declared, not just tolerated. The local model is a copy of predictor-hub's
    # contract, and pydantic ignores extras, so leaving this out would mean the
    # contract test could not notice if `players_unavailable` were dropped from the
    # bundle -- the exact "a key nothing reads" defect. Declaring it here makes the
    # consumer's obligation explicit on this side too, and the field is documented
    # at facts.py:526 as having no reader in this repo.
    players_unavailable: bool = False
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
    # `pre_kickoff` is what `_record()` reads, because the block it renders is labelled "Picks
    # made before kickoff" and the headline now counts every recorded pick. Stubbed with BOTH
    # figures and different numbers on purpose: a fixture that made them agree could not catch a
    # call site that read the wrong one.
    monkeypatch.setattr(
        facts_mod.store, "get_track_record",
        lambda: {"games": {
            "n_resolved": 66, "pct_moneyline_correct": 0.62, "n_rebuilt": 3,
            # 0.80 x 50 = 40 hits, so `hits` and `settled` both differ from what the headline
            # would have produced (0.62 x 66 = 41 of 66) and a mix-up cannot pass unnoticed.
            "pre_kickoff": {"n_resolved": 50, "pct_moneyline_correct": 0.80},
        }},
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
    """The block is LABELLED "Picks made before kickoff", so its two numbers must be the
    pre-kickoff figure's. The fixture offers a 66-pick headline at 62% and a 50-pick pre-kickoff
    figure at 80%; reading the headline instead would render 41 of 66, and this asserts 40 of 50."""
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["record"] == {
        "label": "Picks made before kickoff",
        "hits": 40,
        "settled": 50,
    }


# --- pick_timing: the three cases ---------------------------------------

def test_pick_timing_is_none_only_when_there_is_no_forecast_at_all(public, monkeypatch):
    # No tracking row, and no forecast either: there is genuinely nothing to
    # claim. (A missing row alone is NOT enough — with a live forecast the pick
    # is that forecast; see test_upcoming_game_with_no_stored_row_...)
    _install_snapshot(monkeypatch, _snapshot(prediction={"game_id": None}))
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
    # Stubbed so no test in this module can reach nflverse: _load_game_history
    # reads (and can refresh) the cached parquet. Offline is a hard rule here,
    # not a nicety, so the stub is in the fixture rather than in one test.
    monkeypatch.setattr(
        facts_mod.routes, "_load_game_history",
        lambda season: pd.DataFrame([{"game_id": "other", "rating_diff": 0.0, "home_rest_days": 6, "away_rest_days": 6}]),
    )
    # Same offline rule for the efficiency aux the task-3 duels read: a live
    # facts request would otherwise fetch 8 seasons of nflverse play-by-play.
    monkeypatch.setattr(
        facts_mod.routes, "_load_aux_cached",
        lambda season: facts_mod.routes.feature_build.Aux(
            efficiency=pd.DataFrame(), qb_games=pd.DataFrame(), upcoming_starters={}),
    )
    # Not stubbed: `routes._load_models_cached` runs the real `manifest.load_models`
    # against the committed artefacts. That is deliberate -- this fixture stubs the
    # feature builder because a live facts request would otherwise need a schedule
    # fetch, but the model load itself is local and cheap, and stubbing it would
    # have hidden the fact that `manifest.load_models` calls
    # `build_features_for_game` too.
    #
    # It is also why a `X does not have valid feature names` UserWarning appears
    # under this fixture: `_predict_game_from_models` scores the committed Ridge on
    # `X.to_numpy()`, which drops the column names scikit-learn was fitted with.
    # That is pre-existing production behaviour -- serving has always gone through
    # `.to_numpy()` -- and not something this change introduced.
    # Emits EVERY `features.build.FEATURE_COLUMNS` entry, not just the four the
    # assertions below happen to read.
    #
    # `manifest.load_models` calls `build_features_for_game` to learn what serving
    # actually emits, so this stub is not a private detail of the facts route -- it
    # IS the serving builder as far as the fitted-vs-served audit is concerned. A
    # four-column stub is a builder that genuinely stopped emitting six columns the
    # committed game models are fitted on, and the audit correctly refuses to serve
    # under it. The real builder emits all ten.
    monkeypatch.setattr(
        facts_mod.routes.feature_build, "build_features_for_game",
        lambda home, away, history, gameday=None, blocks=None, aux=None, starters=None, game_schedule=None: pd.Series(
            {c: float(i + 1) for i, c in enumerate(facts_mod.routes.feature_build.FEATURE_COLUMNS)}
            | {"rating_diff": 7.0, "home_rest_days": 6, "away_rest_days": 6},
            dtype=float,
        ),
    )
    monkeypatch.setattr(
        facts_mod.store, "get_track_record",
        lambda: {"games": {
            "n_resolved": 10, "pct_moneyline_correct": 0.5, "n_rebuilt": 0,
            "pre_kickoff": {"n_resolved": 10, "pct_moneyline_correct": 0.5},
        }},
    )
    return TestClient(app)


@pytest.fixture
def upcoming(monkeypatch):
    """PUBLIC_MODE off and the game not yet started: the bundle may compute
    duels here, and the pick comes from today's live read (`_current_prediction`)
    rather than a stored row. Every data source is stubbed so no test reaches
    nflverse or the model files (offline is a hard rule for this module)."""
    monkeypatch.setattr(facts_mod, "PUBLIC_MODE", False)
    monkeypatch.setattr(facts_mod, "_now", lambda: NOW)
    future = lambda: pd.DataFrame([_game(gameday="2026-10-11T00:20:00")])  # noqa: E731
    records = lambda: [_game(gameday="2026-10-11T00:20:00")]  # noqa: E731
    monkeypatch.setattr(facts_mod.routes.schedules, "fetch_week_games", lambda season, week: future())
    monkeypatch.setattr(facts_mod.routes, "get_games", lambda season, week: records())
    monkeypatch.setattr(facts_mod.routes, "get_player_props", lambda season, week: [])
    monkeypatch.setattr(facts_mod.routes, "_load_game_history", lambda season: pd.DataFrame([{"game_id": "other"}]))
    monkeypatch.setattr(facts_mod.routes, "_load_aux_cached",
                        lambda season: facts_mod.routes.feature_build.Aux(
                            efficiency=pd.DataFrame(), qb_games=pd.DataFrame(), upcoming_starters={}))
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda season, week, game_id: _prediction())
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: [])
    monkeypatch.setattr(
        facts_mod.routes.feature_build, "build_features_for_game",
        lambda home, away, history, gameday=None, blocks=None, aux=None, starters=None, game_schedule=None: pd.Series(
            {c: float(i + 1) for i, c in enumerate(facts_mod.routes.feature_build.FEATURE_COLUMNS)}
            | {"rating_diff": 7.0, "home_rest_days": 6, "away_rest_days": 6},
            dtype=float,
        ),
    )
    monkeypatch.setattr(
        facts_mod.store, "get_track_record",
        lambda: {"games": {"n_resolved": 0, "pct_moneyline_correct": 0.0, "n_rebuilt": 0,
                           "pre_kickoff": {"n_resolved": 0, "pct_moneyline_correct": 0.0}}},
    )
    return TestClient(app)


def test_facts_carry_matchups_for_an_upcoming_game(upcoming, monkeypatch):
    """AI plan Task 3: an upcoming game's facts bundle carries the duels, and
    `pick_side` is derived from the bundle's own pick (BAL, home, here)."""
    duel_rows = [{"id": "pass_off_vs_pass_def:home", "attacker": "KC", "defender": "BAL", "stat": "passing offence",
                  "foil": "pass defence", "attacker_rank": 3, "defender_rank": 28, "n_teams": 32, "toward_pick": True}]
    seen = {}

    def fake_matchup_rows(home, away, games_df, efficiency, as_of, season, pick_side):
        seen["pick_side"] = pick_side
        return duel_rows

    monkeypatch.setattr(facts_mod, "_matchup_rows", fake_matchup_rows)
    body = upcoming.get(f"/facts/{GAME_ID}").json()
    assert body["context"]["matchups"] == duel_rows
    assert seen["pick_side"] == "home"
    assert seen["pick_side"] != None  # noqa: E711 - a None pick_side would mean duels were never wired


def test_public_bundle_never_carries_matchups(public, monkeypatch):
    """Task 3's honesty gate: PUBLIC_MODE reads the snapshot only, so a public
    bundle carries no duels at all until the snapshot itself carries what they
    need (same rule as the rating-gap driver)."""
    _install_snapshot(monkeypatch, _snapshot())
    body = public.get(f"/facts/{GAME_ID}").json()
    assert "matchups" not in body["context"]


def test_matchup_rows_returns_empty_without_efficiency_data():
    """No efficiency frame -> no duels, never a fabricated row."""
    from nfl_predictor.api.facts import _matchup_rows
    assert _matchup_rows("KC", "BAL", pd.DataFrame(), pd.DataFrame(),
                         as_of="2026-10-01", season=2026, pick_side="home") == []


def test_matchup_rows_builds_real_duels_from_ranked_efficiency(monkeypatch):
    """The real composition: efficiency ranks -> duels -> context rows, with the
    gap-scaled strength fallback (data/duel_gaps.json is absent on a fresh tree).
    T1 (best offence, best defence) hosting T10 (worst at both) makes both duels
    by the min_gap=8 rule, favourite T1. Without data/duel_lift.json the gate
    has never run, so every row is neutral -- direction only returns once the
    lift gate proves the types (claude-review wiring rule)."""
    import nfl_predictor.api.facts as facts_mod
    from nfl_predictor.api.facts import _matchup_rows
    teams = [f"T{i}" for i in range(1, 11)]
    rows = []
    for game_no in (1, 2, 3):
        for i, team in enumerate(teams):
            off = 1.0 - i * 0.2          # T1 best, T10 worst
            defence = -1.0 + i * 0.2     # T1 best (most negative), T10 worst
            rows.append({"game_id": f"g{game_no}", "team": team,
                         "epa_off_pass": off, "epa_def_pass": defence,
                         "epa_off_rush": off * 0.5, "epa_def_rush": defence * 0.5})
    efficiency = pd.DataFrame(rows)
    games = pd.DataFrame({"game_id": ["g1", "g2", "g3"],
                          "gameday": pd.to_datetime(["2026-09-10", "2026-09-17", "2026-09-24"]),
                          "season": [2026, 2026, 2026]})
    out = _matchup_rows("T1", "T10", games, efficiency, as_of="2026-10-01", season=2026, pick_side="home")
    assert out, "expected real duels between the strongest and weakest teams"
    assert out[0]["attacker"] == "T1" and out[0]["defender"] == "T10"
    assert all(r["stat"] and r["foil"] for r in out), "the DUELS nouns must fill stat/foil, never ''"
    # The :home duels read strongest-offence vs weakest-defence (1 vs 10); the
    # :away duels read the weakest offence vs the strongest defence (10 vs 1).
    assert all(1 <= r["attacker_rank"] <= 10 and 1 <= r["defender_rank"] <= 10 for r in out)
    # No data/duel_lift.json on this tree -> the Task 10 gate proves nothing,
    # so even a favourite facing the worst team stays neutral context.
    assert all(r["toward_pick"] is None for r in out), \
        "an absent duel_lift.json must mean toward_pick None everywhere, even with a pick"

    # Wire the gate: once the types are proven, direction returns.
    monkeypatch.setattr(facts_mod, "load_lift_results",
                        lambda: {"pass_off_vs_pass_def": True, "rush_off_vs_rush_def": True})
    directed = _matchup_rows("T1", "T10", games, efficiency, as_of="2026-10-01", season=2026, pick_side="home")
    # Once a type is proven only its STRONGEST duel (the one the lift test validated) carries a direction; the
    # opposite-direction duel of the same type stays neutral context.
    by_type = {}
    for r in directed:
        by_type.setdefault(r["id"].split(":")[0], []).append(r["toward_pick"])
    assert by_type, "expected rows for both duel types"
    for duel_type, flags in by_type.items():
        assert sum(f is not None for f in flags) == 1, f"{duel_type}: exactly one directed row, got {flags}"
        assert all(f is True for f in flags if f is not None), \
            "a favourite facing the worst team edges toward the pick once proven"


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
    # The pick is the stored row, so the number on screen is the record that
    # will be judged; the SPREAD and TOTAL still come from the live model,
    # which is legitimate for a game that has not been played.
    assert body["pick"] == {"label": "BAL", "prob": 0.62}
    assert {m["market"] for m in body["markets"]} == {"moneyline", "spread", "total"}


def test_an_upcoming_game_reports_the_live_rating_gap(live, monkeypatch):
    """Covers _drivers' live branch, which the started-game tests skip.

    The rating gap comes from the stubs the `live` fixture installs — that
    fixture is what keeps this module offline (see the comment there). Note
    this test does NOT itself prove offline behaviour: a raising stub cannot,
    because _drivers swallows exceptions, and removing the fixture stubs did
    not make this test fail. The offline guarantee is structural.
    """
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
    gap = next(d for d in body["drivers"] if d.get("name") == "Rating gap")
    assert gap["value"] == "+7.0 pts"


# --- the pick number and its timing label must share one source ----------

def test_upcoming_game_with_no_stored_row_still_shows_the_live_forecast(public, monkeypatch):
    # Weeks out there is no tracking row yet, but the model has a current
    # forecast. Gating the pick on the row left pick null and
    # pick_timing "none" beside a full set of markets — the panel showed
    # numbers it refused to make a pick from.
    monkeypatch.setattr(facts_mod.store, "get_predictions_for_week", lambda season, week, games_df: [])
    _install_snapshot(monkeypatch, _snapshot(prediction=_prediction(home_win_prob=0.66, away_win_prob=0.34)))

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "upcoming"
    assert body["pick"] == {"label": "BAL", "prob": 0.66}
    assert body["pick_timing"] == "pre_kickoff"
    assert body["markets"]


def test_upcoming_game_prefers_the_stored_row_so_the_number_matches_the_verdict(public, monkeypatch):
    # With a row present the pick is the row, so the number on screen is the
    # very record that will be judged — and a row written late still reports
    # itself as 'rebuilt' rather than borrowing the live forecast's label.
    monkeypatch.setattr(
        facts_mod.store, "get_predictions_for_week",
        lambda season, week, games_df: _week_rows(home_win_prob=0.62, away_win_prob=0.38, rebuilt=True),
    )
    _install_snapshot(monkeypatch, _snapshot(prediction=_prediction(home_win_prob=0.66, away_win_prob=0.34)))

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "upcoming"
    assert body["pick"] == {"label": "BAL", "prob": 0.62}  # the row, not 0.66
    assert body["pick_timing"] == "rebuilt"
    assert body["markets"]


def test_started_game_still_takes_its_pick_and_label_from_the_row(public, monkeypatch):
    # The counterpart: the upcoming rule above must not leak into a started
    # game, where the row remains the only honest source.
    started = _game(gameday="2026-09-20T00:20:00")
    _install_snapshot(monkeypatch, _snapshot(game=started, prediction=_prediction(home_win_prob=0.80, away_win_prob=0.20)))

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "live"
    assert body["pick"] == {"label": "BAL", "prob": 0.62}  # the row, not 0.80
    assert body["pick_timing"] == "pre_kickoff"


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


# --- one pick, one number ------------------------------------------------
#
# The snapshot rebuilds the current and previous week every few hours, so its
# prediction for an UPCOMING game is today's model, recomputed. The tracking row
# is the number that was snapshotted before kickoff. Both are real, and they can
# differ — so a bundle built from one for the pick and the other for the markets
# states the same pick twice at two different probabilities.
#
# That is not a cosmetic disagreement. The v2 panel computes its confidence band
# from `pick.prob` and draws its moneyline tile and split bar from
# `markets[].model`, so the panel would show 58% beside a band derived from 62%.

def test_the_moneyline_market_agrees_with_the_pick(public, monkeypatch):
    """The invariant, as a property of the bundle rather than of one fixture."""
    # The stored row says BAL at 62; today's recomputation says 55.
    _install_snapshot(
        monkeypatch,
        _snapshot(prediction=_prediction(home_win_prob=0.55, away_win_prob=0.45)),
    )

    body = public.get(f"/facts/{GAME_ID}").json()

    pick = body["pick"]
    moneyline = next(m for m in body["markets"] if m["market"] == "moneyline")
    assert moneyline["model"][pick["label"]] == pick["prob"], (
        f"the bundle states the pick twice: pick says {pick['prob']}, the moneyline "
        f"market says {moneyline['model'][pick['label']]}. A panel that draws its "
        f"figure from one and its confidence from the other would show both."
    )


def test_the_pick_is_the_stored_number_even_when_the_model_moved(public, monkeypatch):
    """The stored row still wins, because it is the record that will be judged."""
    _install_snapshot(
        monkeypatch,
        _snapshot(prediction=_prediction(home_win_prob=0.55, away_win_prob=0.45)),
    )

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["pick"]["label"] == "BAL"
    assert body["pick"]["prob"] == 0.62
    assert body["pick_timing"] == "pre_kickoff"


# --- player props: a panel may be empty, it may not be silent ------------
#
# `facts._props` is a second reader of the same props data as
# `routes.get_player_props`. When that route started answering 503 for an
# unavailable week, this function propagated it, and with no handler at the call
# site the whole `/facts/{game_id}` bundle 500'd -- pick, markets, drivers,
# context and record, all of it, because a props panel could not be filled.
#
# The `public` and `live` fixtures above stub `get_player_props` (or the
# snapshot's props) with data that makes props available, which is exactly why
# this file could not see the regression. These tests break that arrangement
# deliberately: they make props unavailable, in both modes, and assert the rest
# of the bundle is intact and that the panel says something true.


def test_a_snapshot_week_with_games_but_no_props_flags_the_panel_instead_of_claiming_no_players(public, monkeypatch):
    # games present, props absent: a build that failed, not a week without props
    _install_snapshot(monkeypatch, _snapshot(props=[]))

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["players"] == []
    assert body["players_unavailable"] is True
    # The rest of the bundle is the point: it must be whole.
    assert body["pick"] == {"label": "BAL", "prob": 0.62}
    assert {m["market"] for m in body["markets"]} == {"moneyline", "spread", "total"}
    assert body["drivers"] and body["context"] and body["record"]


def test_a_snapshot_week_with_players_does_not_flag_the_panel(public, monkeypatch):
    _install_snapshot(monkeypatch, _snapshot())

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["players_unavailable"] is False
    assert len(body["players"]) == 3


def test_a_recorded_unavailable_status_is_read_by_the_facts_panel(public, monkeypatch):
    """The panel and the props route share one rule, so a build that recorded
    "unavailable" is believed by both. Without this the panel would show an empty
    list for a week the props route is refusing outright."""
    snapshot = _snapshot(props=[])
    snapshot["weeks"]["5"]["player_props_status"] = "unavailable"
    _install_snapshot(monkeypatch, snapshot)

    body = public.get(f"/facts/{GAME_ID}").json()

    assert body["players_unavailable"] is True
    assert body["pick"] is not None


def test_a_raising_props_route_does_not_take_the_bundle_down(live, monkeypatch):
    """The regression itself, in the mode where the route computes live.

    `live` stubs `routes.get_player_props` to return `[]`. Overriding it with a
    raiser is the point: a stub that returns an empty list makes the very failure
    under test unfailable, which is how this shipped.

    The game is pushed into the future because the `live` fixture's own game is
    already finished, and a finished game deliberately quotes no props at all --
    which would make the flag False for a reason that has nothing to do with the
    exception handling under test.
    """
    from fastapi import HTTPException

    def _unavailable(season, week):
        raise HTTPException(status_code=503, detail="player props unavailable")

    upcoming = _game(gameday="2026-10-05T00:20:00")
    monkeypatch.setattr(facts_mod.routes.schedules, "fetch_week_games",
                        lambda season, week: pd.DataFrame([upcoming]))
    monkeypatch.setattr(facts_mod.routes, "get_games", lambda season, week: [upcoming])
    monkeypatch.setattr(facts_mod.routes, "get_player_props", _unavailable)

    body = live.get(f"/facts/{GAME_ID}").json()

    assert body["status"] == "upcoming"
    assert body["players"] == []
    assert body["players_unavailable"] is True
    # Everything else still ships. This is the assertion that fails when the
    # exception is not caught: the request 500s and there is no body at all.
    assert body["context"] is not None
    assert body["pick"] is not None or body["pick_timing"] == "none"


def test_a_finished_game_never_claims_its_props_are_unavailable(live, monkeypatch):
    """The flag has to stay quiet where the bundle deliberately shows no props.
    A finished game quotes no props by rule, so "unavailable" would be a false
    claim about something that was never on the page to begin with."""
    from fastapi import HTTPException

    def _unavailable(season, week):
        raise HTTPException(status_code=503, detail="player props unavailable")

    monkeypatch.setattr(facts_mod.routes, "get_player_props", _unavailable)

    body = live.get(f"/facts/{GAME_ID}").json()

    # The `live` fixture's game kicked off before NOW, so it is in progress.
    assert body["status"] == "live"
    assert body["players"] == []
    assert body["players_unavailable"] is False


def test_the_contract_model_round_trips_players_unavailable(public, monkeypatch):
    """The flag has to survive the contract, not just the dict.

    `Facts` is the local copy of the explainer's contract and pydantic ignores
    extras, so a bundle carrying `players_unavailable` validated perfectly well
    against a model that had never heard of it -- which is how a field can be
    added to the bundle and never once checked. Reading it back off the model
    fails loudly if the declaration is ever dropped, and it is the only assertion
    here that ties the producer to the contract rather than to a dict key.
    """
    snapshot = _snapshot(props=[])
    snapshot["weeks"]["5"]["player_props_status"] = "unavailable"
    _install_snapshot(monkeypatch, snapshot)

    body = public.get(f"/facts/{GAME_ID}").json()
    parsed = Facts(**body)

    assert parsed.players_unavailable is True
    assert "players_unavailable" in Facts.model_fields, (
        "the local contract model no longer declares the field, so nothing pins it"
    )

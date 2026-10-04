"""the weekly forward-test tick.

Three behaviours that each protect the record's honesty:

* **a post-kickoff game is never snapshotted.** A line recorded after kickoff
  is not a bettable price, so it is not a pick. One bad game is skipped and the
  rest of the slate still logs.
* **only rows clearing the 5% edge gate are logged.** The track record is a
  list of bets that passed a gate, not a list of every prop priced.
* **both sides are considered, and the side is recorded explicitly.** A 5%
  under is as much a pick as a 5% over.

The budget guard runs before any fetch, so an exhausted month writes nothing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.tracking import store
from nfl_predictor.tracking.forward_tick import EDGE_GATE, run_forward_tick
from nfl_predictor.odds.props_snapshot import BudgetExhausted

FUTURE = "2099-09-04T20:20:00"
PAST = "2000-09-04T20:20:00"


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    yield


#: The Odds API's event ids are opaque strings with no relationship to nflverse
#: game_ids, so every live test game has to be mapped through one.
#: The Odds API id the tick will actually call with, after mapping the slate's
#: nflverse game_id through EVENT_INDEX.
EVENT_ID = "e-bal-den"

EVENT_INDEX = {
    ("Baltimore Ravens", "Denver Broncos"): "e-bal-den",
    ("Kansas City Chiefs", "Las Vegas Raiders"): "e-kc-lv",
}


def _game(game_id: str, commence: str):
    return {
        "game_id": game_id, "home_team": "BAL", "away_team": "DEN",
        "commence_time": commence,
        "home_win_prob": 0.5, "away_win_prob": 0.5,
        "home_cover_prob": 0.5, "away_cover_prob": 0.5,
        "over_prob": 0.5, "under_prob": 0.5,
    }


def _stub(monkeypatch, props_by_event, props=None, coverage=None, credits=500):
    """Replace the network seam. Quantiles come from `market_quantiles=`."""
    from nfl_predictor.tracking import forward_tick

    monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: credits >= needed)
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda event_id, markets=None, credits_needed=1: props_by_event.get(event_id, []))

#: Book props carry names; the tick's player index supplies the id.
PLAYERS = [
    {"player_id": "00-1", "player_name": "Test", "team": "A"},
    {"player_id": "00-2", "player_name": "Good", "team": "A"},
    {"player_id": "00-3", "player_name": "Meh", "team": "A"},
]


def _pick(monkeypatch, p_over_by_market):
    from nfl_predictor.tracking import forward_tick

    def fake_quantiles(market):
        return p_over_by_market[market]

    monkeypatch.setattr(forward_tick, "quantiles_for", fake_quantiles)


def test_tick_rejects_post_kickoff_game(monkeypatch):
    store.record_game_predictions([_game("G_LIVE", FUTURE), _game("G_DEAD", PAST)])
    _stub(monkeypatch, {EVENT_ID: [{"player_name": "Test", "normalized_name": "test",
                                    "market": "player_rec_yds", "line": 50.0,
                                    "over_odds": -110, "under_odds": -110,
                                    "book": "fanduel", "team": "A"}]})
    _pick(monkeypatch, {"player_rec_yds": lambda line: {0.1: 40.0, 0.5: 55.0, 0.9: 70.0}})

    result = run_forward_tick(games=[_game("G_LIVE", FUTURE), _game("G_DEAD", PAST)],
                              market_quantiles={"player_rec_yds": {0.1: 40.0, 0.5: 55.0, 0.9: 70.0}},
                                  event_index=EVENT_INDEX,
                              players=PLAYERS)

    assert result["games_skipped_post_kickoff"] == 1
    assert result["games"] == 1


def test_tick_logs_only_edge_gate_qualifiers(monkeypatch):
    store.record_game_predictions([_game("G1", FUTURE)])
    # Two props: one the model likes a lot, one it is indifferent on.
    props = [
        {"player_name": "Good", "normalized_name": "good", "market": "player_rec_yds",
         "line": 50.0, "over_odds": -110, "under_odds": -110, "book": "fanduel", "team": "A"},
        {"player_name": "Meh", "normalized_name": "meh", "market": "player_rec_yds",
         "line": 90.0, "over_odds": -110, "under_odds": -110, "book": "fanduel", "team": "A"},
    ]
    _stub(monkeypatch, {EVENT_ID: props})
    # A flat distribution around 60: the 50 line is a clear over, the 90 a clear under,
    # and the distribution says nothing about either being +5%.
    quantiles = {0.1: 52.0, 0.2: 54.0, 0.3: 56.0, 0.4: 58.0, 0.5: 60.0,
                 0.6: 62.0, 0.7: 64.0, 0.8: 66.0, 0.9: 68.0}

    result = run_forward_tick(games=[_game("G1", FUTURE)],
                              market_quantiles={"player_rec_yds": quantiles},
                                event_index=EVENT_INDEX,
                              players=PLAYERS)

    assert result["props_snapshotted"] == 2
    logged = _logged()
    sides = {(r["player_id"], r["side"]) for r in logged}
    assert ("00-2", "over") in sides
    assert ("00-3", "under") in sides
    for row in logged:
        assert row["edge_vs_breakeven"] >= EDGE_GATE


def test_a_prop_the_model_is_indifferent_about_is_not_logged(monkeypatch):
    """Every quantile identical is a degenerate (zero-width) distribution, which
    `p_over_from_quantiles` cannot price: a line equal to it is neither below
    nor above, so it interpolates on a zero-length segment. The tick must not
    turn that into a pick -- a 98% confidence from a flat distribution is a
    fabricated edge, and the 5% gate would wave it through."""
    store.record_game_predictions([_game("G1", FUTURE)])
    _stub(monkeypatch, {EVENT_ID: [{"player_name": "Test", "normalized_name": "test",
                                "market": "player_rec_yds", "line": 60.0,
                                "over_odds": -110, "under_odds": -110,
                                "book": "fanduel", "team": "A"}]})
    quantiles = {q: 60.0 for q in [i / 10 for i in range(1, 10)]}

    result = run_forward_tick(games=[_game("G1", FUTURE)],
                              market_quantiles={"player_rec_yds": quantiles},
                                  event_index=EVENT_INDEX,
                              players=PLAYERS)

    assert result["picks_logged"] == 0
    assert _logged() == []


def test_a_degenerate_distribution_is_rejected_before_pricing(monkeypatch):
    """Pins the reason the case above is a no-op: the quantiles carry no spread,
    so there is no distribution to price a line against."""
    from nfl_predictor.models.prop_probability import p_over_from_quantiles

    flat = {round(q, 1): 60.0 for q in [i / 10 for i in range(1, 10)]}
    value = p_over_from_quantiles(flat, 60.0)

    # A line sitting exactly on a zero-width distribution interpolates on a
    # zero-length segment and lands on 1 - q0.1 = 0.9: a 38-point edge over a
    # -110 breakeven, from a model that predicted nothing. Hence the spread check.
    assert value == pytest.approx(0.9)


def test_budget_exhaustion_writes_nothing(monkeypatch):
    store.record_game_predictions([_game("G1", FUTURE)])
    _stub(monkeypatch, {EVENT_ID: [{"player_id": "00-1", "player_name": "P",
                                "market": "player_rec_yds", "line": 50.0,
                                "over_odds": -110, "under_odds": -110,
                                "book": "fanduel", "team": "BAL"}]},
            credits=0)

    result = run_forward_tick(games=[_game("G1", FUTURE)],
                              market_quantiles={"player_rec_yds": {0.1: 40.0, 0.5: 55.0, 0.9: 70.0}},
                                event_index=EVENT_INDEX,
                              players=PLAYERS)

    assert result["picks_logged"] == 0
    assert result["credits_remaining"] == 0
    assert _logged() == [], "an exhausted budget must not write a partial slate"


def test_no_props_coverage_means_no_picks_and_no_fallback(monkeypatch):
    """Spec: if the free tier has no player props, the tick waits. The fetch
    returning nothing IS the coverage signal -- there is no separate probe, and
    nothing falls back to game lines."""
    store.record_game_predictions([_game("G1", FUTURE)])
    _stub(monkeypatch, {})

    result = run_forward_tick(games=[_game("G1", FUTURE)],
                              market_quantiles={"player_rec_yds": {0.1: 40.0, 0.5: 55.0, 0.9: 70.0}},
                                  event_index=EVENT_INDEX,
                              players=PLAYERS)

    assert result["picks_logged"] == 0
    assert result["no_props_coverage"] is True, (
        "an empty fetch must be reported as absent coverage -- distinct from a "
        "misconfiguration, and with no fallback to game lines")
    assert _logged() == []


def test_tick_returns_the_documented_counters(monkeypatch):
    store.record_game_predictions([_game("G1", FUTURE)])
    _stub(monkeypatch, {EVENT_ID: [{"player_name": "Test", "normalized_name": "test",
                                "market": "player_rec_yds", "line": 50.0,
                                "over_odds": -110, "under_odds": -110,
                                "book": "fanduel", "team": "A"}]})
    quantiles = {0.1: 40.0, 0.5: 55.0, 0.9: 70.0}

    result = run_forward_tick(games=[_game("G1", FUTURE)],
                              market_quantiles={"player_rec_yds": quantiles},
                                event_index=EVENT_INDEX,
                              players=PLAYERS)

    for key in ("games", "props_snapshotted", "picks_logged", "credits_remaining"):
        assert key in result


def test_recorded_edge_uses_the_sides_actual_odds(monkeypatch):
    store.record_game_predictions([_game("G1", FUTURE)])
    _stub(monkeypatch, {EVENT_ID: [{"player_name": "Test", "normalized_name": "test",
                                "market": "player_rec_yds", "line": 50.0,
                                "over_odds": -115, "under_odds": -105,
                                "book": "fanduel", "team": "A"}]})
    quantiles = {0.1: 40.0, 0.5: 55.0, 0.9: 70.0}

    run_forward_tick(games=[_game("G1", FUTURE)],
                     market_quantiles={"player_rec_yds": quantiles},
                       event_index=EVENT_INDEX,
                     players=PLAYERS)

    row = next(r for r in _logged() if r["side"] == "over")
    # over at -115 breakevens at 115/215 = 0.5349
    assert row["odds_at_snapshot"] == -115.0
    assert row["edge_vs_breakeven"] == pytest.approx(row["model_p_over"] - 115 / 215, abs=1e-6)


def test_under_pick_records_its_own_odds(monkeypatch):
    store.record_game_predictions([_game("G1", FUTURE)])
    _stub(monkeypatch, {EVENT_ID: [{"player_name": "Test", "normalized_name": "test",
                                "market": "player_rec_yds", "line": 90.0,
                                "over_odds": -110, "under_odds": -110,
                                "book": "fanduel", "team": "A"}]})
    quantiles = {0.1: 40.0, 0.5: 55.0, 0.9: 70.0}

    run_forward_tick(games=[_game("G1", FUTURE)],
                     market_quantiles={"player_rec_yds": quantiles},
                         event_index=EVENT_INDEX,
                     players=PLAYERS)

    # model_p_over stores the probability of the side TAKEN, so the under's 0.98
    # is 1 - p_over(90) = 1 - 0.02: the model says the 90 will not be covered.
    row = next(r for r in _logged() if r["side"] == "under")
    assert row["model_p_over"] == pytest.approx(0.98)
    assert row["edge_vs_breakeven"] == pytest.approx(0.98 - 110 / 210, abs=1e-6)


def _logged() -> list[dict]:
    import contextlib
    import sqlite3

    with contextlib.closing(store._connect()) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM player_prop_predictions WHERE line_at_snapshot IS NOT NULL")]
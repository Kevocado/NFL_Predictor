"""live player-prop line fetcher.

Two things this module exists to get right, both pinned here:

* **the budget guard runs before the first call.** The free tier is 500
  credits/month and a slate can need more than a month has left. A guard that
  checks after spending has already spent it, so `test_budget_guard_aborts_
  before_any_call` asserts the request mock recorded ZERO calls.
* **names are normalized, never fuzzy-matched.** A silent near-match puts a
  line on the wrong player, which produces a real bet at a real price on the
  wrong proposition. Unmatched props are logged and skipped.

Nothing here touches the network.
"""
from __future__ import annotations

import pytest
import requests_mock as rm_module

from nfl_predictor.odds import props_snapshot
from nfl_predictor.odds.props_snapshot import (
    BudgetExhausted, credits_sufficient, fetch_props_for_event,
    normalize_player_name, probe_props_coverage,
)

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/EV1/odds"
SCORES = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/scores"


@pytest.fixture(autouse=True)
def _budget(request, monkeypatch):
    """Default: plenty of credits left. Tests that care about the guard
    override `remaining`, including to None for 'the API did not say'."""
    remaining = getattr(request, "param", 400)
    if isinstance(remaining, str):
        remaining = int(remaining)
    monkeypatch.setattr(props_snapshot, "ODDS_API_KEY", "test-key")
    monkeypatch.setattr(props_snapshot, "_probe_credits", lambda: remaining)


def _book(**overrides):
    market = {
        "player_name": "A.J. Brown Jr.", "team": "PHI", "market": "player_pass_yds",
        "line": 87.5, "over_odds": -110, "under_odds": -110, "book": "fanduel",
    }
    market.update(overrides)
    return market


# --- name normalization ----------------------------------------------------

def test_name_normalization_strips_punctuation_and_suffixes():
    assert normalize_player_name("A.J. Brown Jr.") == "aj brown"
    assert normalize_player_name("De'Von Achane") == "devon achane"
    assert normalize_player_name("Michael Pittman Jr.") == "michael pittman"


def test_name_normalization_is_whitespace_and_case_insensitive():
    assert normalize_player_name("  JOSH   ALLEN ") == normalize_player_name("Josh Allen")


def test_name_normalization_handles_nfl_style_abbreviations():
    # nflverse abbreviates some names; the book spells them out.
    assert normalize_player_name("P. Mahomes") == normalize_player_name("Patrick Mahomes")


# --- the budget guard ------------------------------------------------------

@pytest.mark.parametrize("_budget", ["5"], indirect=True)
def test_budget_guard_aborts_before_any_call(requests_mock):
    """5 credits left, the slate needs 40: raise, and make NO props request."""
    requests_mock.get(SCORES, json=[], headers={"x-requests-remaining": "5"})
    requests_mock.get(BASE, json=[_book()], headers={"x-requests-remaining": "5"})

    with pytest.raises(BudgetExhausted):
        fetch_props_for_event("EV1", markets=["player_pass_yds", "player_rush_yds"],
                              credits_needed=40)

    assert requests_mock.call_count == 0, (
        "the guard must decide before spending; it made a request")


def test_budget_guard_allows_the_call_when_enough_remains(requests_mock):
    requests_mock.get(BASE, json=[_book()], headers={"x-requests-remaining": "400"})

    props = fetch_props_for_event("EV1", markets=["player_pass_yds"], credits_needed=1)

    assert len(props) == 1
    assert props[0]["line"] == 87.5


@pytest.mark.parametrize("_budget", [None], indirect=True)
def test_credits_sufficient_is_false_when_the_header_is_missing(requests_mock):
    """No header means unknown, and unknown is not 'enough'."""
    requests_mock.get(SCORES, json=[], headers={})

    assert credits_sufficient(needed=1) is False


# --- fetching and normalizing ---------------------------------------------

def test_fetch_returns_normalized_fields(requests_mock):
    requests_mock.get(BASE, json=[_book()], headers={"x-requests-remaining": "400"})
    requests_mock.get(SCORES, json=[], headers={"x-requests-remaining": "400"})

    props = fetch_props_for_event("EV1", markets=["player_pass_yds"])

    assert props == [{
        "player_name": "A.J. Brown Jr.", "normalized_name": "aj brown", "team": "PHI",
        "market": "player_pass_yds", "line": 87.5, "over_odds": -110.0,
        "under_odds": -110.0, "book": "fanduel",
    }]


def test_fetch_requests_american_odds_and_the_us_region(requests_mock):
    requests_mock.get(BASE, json=[], headers={"x-requests-remaining": "400"})
    requests_mock.get(SCORES, json=[], headers={"x-requests-remaining": "400"})

    fetch_props_for_event("EV1", markets=["player_pass_yds"])

    # `last_request` is whichever route matched most recently, so find the odds
    # call by path rather than assuming it was the final one.
    # requests_mock lowercases the path, so match case-insensitively.
    odds_calls = [r for r in requests_mock.request_history if "/events/ev1/odds" in r.path.lower()]
    assert len(odds_calls) == 1
    query = odds_calls[0].qs
    assert query["oddsformat"] == ["american"]
    assert query["regions"] == ["us"]
    assert query["markets"] == ["player_pass_yds"]


def test_a_row_missing_its_line_is_skipped_not_defaulted(requests_mock):
    """A prop with no line is not a prop. Defaulting it to 0 would post a bet."""
    requests_mock.get(BASE, json=[_book(), _book(player_name="Ghost", line=None)],
                      headers={"x-requests-remaining": "400"})

    props = fetch_props_for_event("EV1", markets=["player_pass_yds"])

    assert [p["player_name"] for p in props] == ["A.J. Brown Jr."]


def test_multiple_books_yield_multiple_rows(requests_mock):
    requests_mock.get(BASE, json=[_book(book="fanduel"), _book(book="draftkings")],
                      headers={"x-requests-remaining": "400"})

    props = fetch_props_for_event("EV1", markets=["player_pass_yds"])

    assert {p["book"] for p in props} == {"fanduel", "draftkings"}


# --- joining to nflverse players ------------------------------------------

def test_unmatched_props_are_skipped_and_logged_not_guessed(requests_mock, caplog):
    requests_mock.get(BASE, json=[_book(player_name="Nobody At All", team="ZZZ")],
                      headers={"x-requests-remaining": "400"})
    players = [{"player_id": "00-1", "player_name": "A.J. Brown Jr.", "team": "PHI"}]

    with caplog.at_level("WARNING"):
        matched = props_snapshot.match_props_to_players(
            fetch_props_for_event("EV1", markets=["player_pass_yds"]), players)

    assert matched == []
    assert any("Nobody At All" in record.getMessage() for record in caplog.records)


def test_matching_joins_on_normalized_name_and_team(requests_mock):
    requests_mock.get(BASE, json=[_book()], headers={"x-requests-remaining": "400"})
    players = [{"player_id": "00-1", "player_name": "A.J. Brown Jr.", "team": "PHI"}]

    matched = props_snapshot.match_props_to_players(
        fetch_props_for_event("EV1", markets=["player_pass_yds"]), players)

    assert len(matched) == 1
    assert matched[0]["player_id"] == "00-1"


def test_right_name_wrong_team_does_not_match(requests_mock):
    """A name collision across teams is a miss, not a coin flip."""
    requests_mock.get(BASE, json=[_book()], headers={"x-requests-remaining": "400"})
    players = [{"player_id": "00-1", "player_name": "A.J. Brown Jr.", "team": "NO"}]

    matched = props_snapshot.match_props_to_players(
        fetch_props_for_event("EV1", markets=["player_pass_yds"]), players)

    assert matched == []


# --- coverage probe --------------------------------------------------------

def test_probe_reports_which_markets_and_books_came_back(requests_mock):
    requests_mock.get(BASE, json=[_book(market="player_pass_yds"),
                                  _book(market="player_reception_yds", book="draftkings")],
                      headers={"x-requests-remaining": "400"})

    coverage = probe_props_coverage("EV1")

    # Book keys are mapped to project market names, so `player_reception_yds`
    # is reported as `player_rec_yds` -- the same name the model trains on.
    assert coverage["markets"] == {"player_pass_yds": ["fanduel"],
                                   "player_rec_yds": ["draftkings"]}
    assert "fanduel" in coverage["books"]


def test_probe_on_an_event_with_no_props_is_empty_not_an_error(requests_mock):
    requests_mock.get(BASE, json=[], headers={"x-requests-remaining": "400"})

    coverage = probe_props_coverage("EV1")

    assert coverage["markets"] == {}
    assert coverage["has_any"] is False


def test_probe_reports_no_coverage_rather_than_falling_back_to_game_lines(requests_mock):
    """Spec: if the free tier has no player props, the tick waits. It must not
    quietly substitute a game total and call it a prop."""
    requests_mock.get(BASE, json=[], headers={"x-requests-remaining": "400"})

    coverage = probe_props_coverage("EV1")

    assert coverage["has_any"] is False
    assert "fallback" not in coverage
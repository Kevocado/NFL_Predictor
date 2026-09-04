import pandas as pd
import pytest

from nfl_predictor.odds import value_bets


def test_devig_h2h_removes_the_bookmaker_margin():
    result = value_bets.devig_h2h(home_price=1.91, away_price=1.91)

    assert result is not None
    assert result["home_win"] == pytest.approx(0.5, abs=0.01)
    assert result["home_win"] + result["away_win"] == pytest.approx(1.0, abs=1e-6)


def test_devig_totals_removes_the_bookmaker_margin():
    result = value_bets.devig_totals(over_price=1.91, under_price=1.91)

    assert result is not None
    assert result["over"] == pytest.approx(0.5, abs=0.01)


def test_build_value_bet_table_flags_a_positive_edge():
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "BAL", "away_team": "KC", "commence_time": "2025-09-04T20:20:00"}]
    )
    odds_df = pd.DataFrame(
        [
            {"event_id": "g1", "market": "h2h", "outcome_name": "BAL", "price": 2.20, "point": None, "bookmaker": "dk", "odds_fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
            {"event_id": "g1", "market": "h2h", "outcome_name": "KC", "price": 1.75, "point": None, "bookmaker": "dk", "odds_fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
        ]
    )
    predictions = {"g1": {"home_win_prob": 0.60, "away_win_prob": 0.40}}

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    row = table.iloc[0]
    assert row["home_win_edge"] > 0
    assert "home_win" in row["value_bet_flags"]


def test_build_value_bet_table_recommends_only_the_largest_single_game_edge():
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "BAL", "away_team": "KC", "commence_time": "2025-09-04T20:20:00"}]
    )
    odds_df = pd.DataFrame(
        [
            {"event_id": "g1", "market": "h2h", "outcome_name": "BAL", "price": 2.20},
            {"event_id": "g1", "market": "h2h", "outcome_name": "KC", "price": 1.75},
            {"event_id": "g1", "market": "totals", "outcome_name": "Over", "price": 2.20},
            {"event_id": "g1", "market": "totals", "outcome_name": "Under", "price": 1.75},
        ]
    )
    predictions = {
        "g1": {"home_win_prob": 0.60, "away_win_prob": 0.40, "over_prob": 0.70, "under_prob": 0.30}
    }

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    assert table.iloc[0]["value_bet_flags"] == ["over"]


def test_build_value_bet_table_only_pairs_h2h_prices_from_the_same_bookmaker():
    """Best-of-book selection must not stitch together a home price from one
    bookmaker and an away price from another -- that pairing was never a real
    two-sided market and produces a false edge."""
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "BAL", "away_team": "KC", "commence_time": "2025-09-04T20:20:00"}]
    )
    odds_df = pd.DataFrame(
        [
            # dk is the only bookmaker quoting both sides -- the only valid pair.
            {"event_id": "g1", "market": "h2h", "outcome_name": "BAL", "price": 2.30, "bookmaker": "dk"},
            {"event_id": "g1", "market": "h2h", "outcome_name": "KC", "price": 1.55, "bookmaker": "dk"},
            # fd only quotes the home side, at a better price than dk.
            {"event_id": "g1", "market": "h2h", "outcome_name": "BAL", "price": 2.40, "bookmaker": "fd"},
            # mgm only quotes the away side, at a better price than dk.
            {"event_id": "g1", "market": "h2h", "outcome_name": "KC", "price": 1.60, "bookmaker": "mgm"},
        ]
    )
    predictions = {"g1": {"home_win_prob": 0.55, "away_win_prob": 0.45}}

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    row = table.iloc[0]
    expected = value_bets.devig_h2h(2.30, 1.55)
    cross_book = value_bets.devig_h2h(2.40, 1.60)
    # Sanity check the two candidate pairings actually disagree, so this test
    # would catch a regression rather than passing by coincidence.
    assert expected["home_win"] != pytest.approx(cross_book["home_win"], abs=1e-6)
    assert row["home_win_edge"] == pytest.approx(0.55 - expected["home_win"], abs=1e-6)


def test_build_value_bet_table_ignores_mismatched_point_totals():
    """Over/Under prices must only be paired when they share both a
    bookmaker AND a point; otherwise it isn't a real two-sided market. The
    only valid pair here is dk's -- fd and mgm each only quote one side, at
    different points, with much better prices that a naive best-price scan
    would wrongly prefer."""
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "BAL", "away_team": "KC", "commence_time": "2025-09-04T20:20:00"}]
    )
    odds_df = pd.DataFrame(
        [
            {"event_id": "g1", "market": "totals", "outcome_name": "Over", "point": 47.5, "price": 1.91, "bookmaker": "dk"},
            {"event_id": "g1", "market": "totals", "outcome_name": "Under", "point": 47.5, "price": 1.91, "bookmaker": "dk"},
            {"event_id": "g1", "market": "totals", "outcome_name": "Over", "point": 44.5, "price": 2.50, "bookmaker": "fd"},
            {"event_id": "g1", "market": "totals", "outcome_name": "Under", "point": 51.5, "price": 2.50, "bookmaker": "mgm"},
        ]
    )
    predictions = {"g1": {"over_prob": 0.70, "under_prob": 0.30}}

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    row = table.iloc[0]
    expected = value_bets.devig_totals(1.91, 1.91)
    assert row["over_edge"] == pytest.approx(0.70 - expected["over"], abs=1e-6)
    assert "over" in row["value_bet_flags"]


def test_build_value_bet_table_returns_no_edge_when_no_bookmaker_quotes_both_sides():
    """If no single bookmaker quotes both the home and away price, there is
    no real market to de-vig -- the table must degrade gracefully to no edge
    rather than crash or synthesize a cross-book pairing."""
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "BAL", "away_team": "KC", "commence_time": "2025-09-04T20:20:00"}]
    )
    odds_df = pd.DataFrame(
        [
            {"event_id": "g1", "market": "h2h", "outcome_name": "BAL", "price": 2.30, "bookmaker": "dk"},
            {"event_id": "g1", "market": "h2h", "outcome_name": "KC", "price": 1.60, "bookmaker": "fd"},
        ]
    )
    predictions = {"g1": {"home_win_prob": 0.55, "away_win_prob": 0.45}}

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    row = table.iloc[0]
    assert row["home_win_edge"] is None
    assert row["away_win_edge"] is None
    assert row["value_bet_flags"] == []

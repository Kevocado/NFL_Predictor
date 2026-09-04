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

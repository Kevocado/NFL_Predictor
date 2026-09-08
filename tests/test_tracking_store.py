import pandas as pd
import pytest

from nfl_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _game():
    return {
        "game_id": "2025_01_BAL_KC", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00", "home_win_prob": 0.58, "away_win_prob": 0.42,
        "home_cover_prob": 0.52, "away_cover_prob": 0.48, "over_prob": 0.55, "under_prob": 0.45,
    }


def test_record_game_predictions_is_idempotent():
    n1 = store.record_game_predictions([_game()])
    n2 = store.record_game_predictions([_game()])

    assert n1 == 1
    assert n2 == 0  # already logged, INSERT OR IGNORE


def test_record_game_predictions_rejects_snapshots_after_kickoff():
    game = _game() | {"commence_time": "2000-09-04T20:20:00"}

    with pytest.raises(ValueError, match="before kickoff"):
        store.record_game_predictions([game])


def test_reconcile_game_predictions_fills_actual_outcome():
    store.record_game_predictions([_game()])
    results = pd.DataFrame(
        [{"game_id": "2025_01_BAL_KC", "home_score": 27, "away_score": 20}]
    )

    n = store.reconcile_game_predictions(results)

    assert n == 1
    record = store.get_track_record()
    assert record["n_resolved_games"] == 1
    assert record["pct_moneyline_correct"] == 1.0  # predicted home win, home won


def test_reconcile_game_predictions_counts_duplicate_results_once():
    store.record_game_predictions([_game()])
    results = pd.DataFrame(
        [
            {"game_id": "2025_01_BAL_KC", "home_score": 27, "away_score": 20},
            {"game_id": "2025_01_BAL_KC", "home_score": 20, "away_score": 27},
        ]
    )

    assert store.reconcile_game_predictions(results) == 1
    assert store.get_track_record()["n_resolved_games"] == 1


def test_record_and_reconcile_player_prop_predictions():
    prop = {"game_id": "2025_01_BAL_KC", "player_id": "p1", "player_name": "Runner",
            "market": "rushing_yards", "predicted_value": 85.0}
    store.record_player_prop_predictions([prop])

    player_stats_df = pd.DataFrame(
        [{"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 92, "receiving_yards": 5,
          "passing_yards": 0, "rushing_tds": 1, "receiving_tds": 0, "passing_tds": 0}]
    )
    n = store.reconcile_player_prop_predictions(player_stats_df)

    assert n == 1


def test_reconcile_player_prop_predictions_counts_duplicate_stats_once():
    prop = {"game_id": "2025_01_BAL_KC", "player_id": "p1", "player_name": "Runner",
            "market": "rushing_yards", "predicted_value": 85.0}
    store.record_player_prop_predictions([prop])
    player_stats_df = pd.DataFrame(
        [
            {"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 92},
            {"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 10},
        ]
    )

    assert store.reconcile_player_prop_predictions(player_stats_df) == 1


def _future_game(**overrides):
    game = {
        "game_id": "2025_01_BAL_KC", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": -3.5, "total_line": 51.5,
    }
    game.update(overrides)
    return game


def test_record_game_predictions_persists_spread_and_total_lines():
    import contextlib

    store.record_game_predictions([_future_game()])

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn).iloc[0]

    assert row["home_spread_line"] == -3.5
    assert row["total_line"] == 51.5


def test_reconcile_grades_moneyline_ats_and_totals():
    import contextlib

    store.record_game_predictions([_future_game()])
    # home favored by -3.5 and predicted to cover (home_cover_prob=0.55 > away);
    # over predicted (over_prob=0.5 == under_prob=0.5, home_win predicted).
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 30, "away_score": 20}])

    resolved = store.reconcile_game_predictions(results)

    assert resolved == 1
    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn).iloc[0]

    assert row["moneyline_hit"] == 1  # home won, home was favored
    # home won by 10, spread was -3.5 => home covered; predicted_home_cover=True
    assert row["ats_hit"] == 1
    # total = 50, line = 51.5 => actual under; predicted was a coin flip (over_prob==under_prob)
    # tie-break must not crash -- assert it resolved to 0 or 1, not None
    assert row["total_hit"] in (0, 1)


def test_reconcile_leaves_ats_and_total_hit_null_when_lines_were_never_recorded():
    import contextlib

    store.record_game_predictions([_future_game(home_spread_line=None, total_line=None)])
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 30, "away_score": 20}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn).iloc[0]

    assert row["moneyline_hit"] == 1
    assert pd.isna(row["ats_hit"])
    assert pd.isna(row["total_hit"])


def test_reconcile_catches_a_game_missed_by_a_prior_tick():
    """Simulates a deploy/restart: the game was snapshotted, its results
    became available, but no tick ran to reconcile it until now."""
    store.record_game_predictions([_future_game(game_id="g1", commence_time="2099-01-01T00:00:00Z")])
    results = pd.DataFrame([{"game_id": "g1", "home_score": 21, "away_score": 14}])

    resolved = store.reconcile_game_predictions(results)

    assert resolved == 1


def test_backfill_catches_a_prior_season_row_current_season_partial_would_miss(monkeypatch):
    store.record_game_predictions([_future_game(game_id="g_old", season=2024, commence_time="2099-01-01T00:00:00Z")])
    from nfl_predictor.data import schedules
    monkeypatch.setattr(
        schedules, "load_training_data",
        lambda seasons: pd.DataFrame([{"game_id": "g_old", "home_score": 10, "away_score": 24}]),
    )

    resolved = store.backfill_unresolved_games(schedules)

    assert resolved == 1


def test_get_game_verdict_returns_none_for_unresolved_game():
    store.record_game_predictions([_future_game()])

    verdict = store.get_game_verdict("2025_01_BAL_KC")

    assert verdict is None


def test_get_game_verdict_summarizes_all_three_markets():
    store.record_game_predictions([_future_game()])
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 30, "away_score": 20}])
    store.reconcile_game_predictions(results)

    verdict = store.get_game_verdict("2025_01_BAL_KC")

    assert verdict is not None
    assert verdict["resolved"] is True
    assert verdict["game_id"] == "2025_01_BAL_KC"
    assert verdict["moneyline"]["hit"] == 1
    assert verdict["moneyline"]["predicted"] == "home_win"
    assert verdict["moneyline"]["actual"] == "home_win"
    assert verdict["ats"]["hit"] == 1
    assert verdict["ats"]["predicted"] == "home_cover"
    assert verdict["totals"]["hit"] in (0, 1)
    assert verdict["totals"]["predicted"] == "over"

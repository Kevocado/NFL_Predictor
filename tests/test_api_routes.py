import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api.main import app
from nfl_predictor.api import routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "2025_01_BAL_KC", "season": season, "week": week,
              "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
              "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5}]
        ),
    )
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "BAL", "away_team": "KC", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {
            "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
            "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
            "feature_cols": [], "player_feature_cols": [],
        },
    )
    monkeypatch.setattr(
        routes, "_predict_game_from_models",
        lambda models, home, away, games_df, spread_line=None, total_line=None: {
            "home_win_prob": 0.6, "away_win_prob": 0.4, "home_cover_prob": 0.55, "away_cover_prob": 0.45,
            "over_prob": 0.52, "under_prob": 0.48,
        },
    )
    monkeypatch.setattr(routes.store, "get_track_record", lambda: {"n_resolved_games": 0, "pct_moneyline_correct": None})
    return TestClient(app)


def test_get_games_returns_week_slate(client):
    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["game_id"] == "2025_01_BAL_KC"


def test_get_game_prediction(client):
    response = client.get("/api/games/2025/1/2025_01_BAL_KC/prediction")

    assert response.status_code == 200
    body = response.json()
    assert body["home_win_prob"] == 0.6


def test_get_game_prediction_404s_for_unknown_game(client):
    response = client.get("/api/games/2025/1/nonexistent/prediction")

    assert response.status_code == 404


def test_get_track_record(client):
    response = client.get("/api/track-record")

    assert response.status_code == 200
    assert response.json()["n_resolved_games"] == 0


def test_get_games_handles_nan_scores_for_unplayed_games(client, monkeypatch):
    # Real upcoming-game data (unplayed games) carries home_score/away_score
    # as genuine pandas/numpy float NaN, not Python None -- reproduce that
    # exactly (float("nan") forces a float64 column, matching the real
    # schedules data) rather than the `client` fixture's None, which pandas
    # keeps as an object-dtype None and never actually reproduces the bug.
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "2025_01_BAL_KC", "season": season, "week": week,
              "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
              "home_score": float("nan"), "away_score": float("nan"),
              "spread_line": -2.5, "total_line": 46.5}]
        ),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["home_score"] is None
    assert body[0]["away_score"] is None

def test_get_player_props_includes_recent_team_and_position(client, monkeypatch):
    monkeypatch.setattr(
        routes, "_load_player_history",
        lambda season: pd.DataFrame(
            [{"player_id": "00-001", "player_name": "Pat Mahomes", "position": "QB",
              "recent_team": "KC", "season": season}]
        ),
    )
    monkeypatch.setattr(
        routes.player_usage, "build_features_for_player",
        lambda player_id, history: pd.Series({"dummy_feature": 1.0}),
    )
    monkeypatch.setattr(
        routes.player_props, "predict_props",
        lambda player_models, feature_row, position: {"anytime_td_prob": 0.42, "passing_yards": 275.0},
    )

    response = client.get("/api/players/2025/1/props")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["recent_team"] == "KC"
    assert body[0]["position"] == "QB"

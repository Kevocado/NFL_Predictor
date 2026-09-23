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
        routes.schedules, "fetch_week_games",
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


def test_predict_game_from_models_includes_sigma_and_total_sigma(monkeypatch):
    class _FakeTotalModel:
        def predict(self, _X):
            return [45.0]

    monkeypatch.setattr(
        routes.feature_build, "build_features_for_game",
        lambda home, away, games_df: pd.Series({"rating_diff": 50.0, "home_rest_days": 7.0, "away_rest_days": 7.0}),
    )
    models = {
        "feature_cols": ["rating_diff", "home_rest_days", "away_rest_days"],
        "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
        "total_model": _FakeTotalModel(),
    }

    result = routes._predict_game_from_models(models, "KC", "BAL", pd.DataFrame())

    assert result["sigma"] == 12.0
    assert result["total_sigma"] == 10.0
    assert "predicted_margin" in result
    assert "predicted_total" in result


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
        routes.schedules, "fetch_week_games",
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


def test_get_games_includes_finished_games(client, monkeypatch):
    # A week with a mix of finished and still-upcoming games must return
    # both -- fetch_upcoming_games alone would silently drop the finished
    # one, which is exactly the bug this route must not have.
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [
                {"game_id": "2025_01_BAL_KC", "season": season, "week": week,
                 "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
                 "home_score": 27, "away_score": 20, "spread_line": -2.5, "total_line": 46.5},
                {"game_id": "2025_01_PHI_GB", "season": season, "week": week,
                 "gameday": "2025-09-05", "home_team": "GB", "away_team": "PHI",
                 "home_score": None, "away_score": None, "spread_line": 1.5, "total_line": 45.0},
            ]
        ),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    game_ids = {g["game_id"] for g in response.json()}
    assert game_ids == {"2025_01_BAL_KC", "2025_01_PHI_GB"}


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


def test_get_game_verdict_404s_for_untracked_game(client, monkeypatch):
    monkeypatch.setattr(routes.store, "get_game_verdict", lambda game_id: None)

    response = client.get("/api/games/nope/verdict")

    assert response.status_code == 404


def test_get_predictions_for_week_returns_prediction_status(client, monkeypatch):
    predictions = [
        {"game_id": "2025_01_BAL_KC", "status": "pending", "home_win_prob": 0.6, "away_win_prob": 0.4, "verdict": None},
    ]
    monkeypatch.setattr(routes.store, "get_predictions_for_week", lambda season, week, games: predictions)

    response = client.get("/api/predictions/2025/1")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["game_id"] == "2025_01_BAL_KC"
    assert body[0]["status"] == "pending"
    assert body[0]["home_win_prob"] == 0.6


def test_get_predictions_for_week_includes_both_resolved_and_pending_games(client, monkeypatch):
    """A week in progress has both finished games (from load_training_data)
    and still-upcoming ones (from fetch_upcoming_games). The old if/else
    fallback dropped the finished game whenever any game that week was
    still unplayed (see this plan's final review, finding B3); the route
    must union both sources instead. The client fixture's default mocks
    already return one pending game ("2025_01_BAL_KC", week=1) from
    fetch_upcoming_games and one finished game ("g0", week=1) from
    load_training_data."""
    captured = {}

    def fake_get_predictions_for_week(season, week, games):
        captured["game_ids"] = set(games["game_id"])
        return [
            {"game_id": gid, "status": "resolved" if gid == "g0" else "pending", "verdict": None}
            for gid in games["game_id"]
        ]

    monkeypatch.setattr(routes.store, "get_predictions_for_week", fake_get_predictions_for_week)

    response = client.get("/api/predictions/2025/1")

    assert response.status_code == 200
    assert captured["game_ids"] == {"2025_01_BAL_KC", "g0"}
    body = response.json()
    statuses = {row["game_id"]: row["status"] for row in body}
    assert statuses["2025_01_BAL_KC"] == "pending"
    assert statuses["g0"] == "resolved"


def test_get_power_rankings_sorts_descending_by_rating_and_includes_division(client, monkeypatch):
    monkeypatch.setattr(
        routes.power_ratings, "final_ratings",
        lambda history: {"BAL": 1550.0, "KC": 1480.0},
    )
    monkeypatch.setattr(
        routes.teams_data, "fetch_team_conferences",
        lambda: pd.DataFrame(
            [
                {"team": "BAL", "conference": "AFC", "division": "AFC North"},
                {"team": "KC", "conference": "AFC", "division": "AFC West"},
            ]
        ),
    )

    response = client.get("/api/power-rankings?season=2025")

    assert response.status_code == 200
    body = response.json()
    assert body["season"] == 2025
    rankings = body["rankings"]
    assert [r["team"] for r in rankings] == ["BAL", "KC"]
    assert rankings[0]["rank"] == 1
    assert rankings[1]["rank"] == 2
    assert rankings[0]["division"] == "AFC North"
    assert rankings[1]["division"] == "AFC West"


def test_get_power_rankings_includes_win_loss_record(client, monkeypatch):
    # Override load_training_data to return a played game for the exact
    # requested season -- BAL beat KC 24-17, so BAL should show 1 win.
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": 2025, "week": 1, "gameday": "2025-09-04",
              "home_team": "BAL", "away_team": "KC", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(
        routes.power_ratings, "final_ratings",
        lambda history: {"BAL": 1550.0, "KC": 1480.0},
    )
    monkeypatch.setattr(
        routes.teams_data, "fetch_team_conferences",
        lambda: pd.DataFrame(
            [
                {"team": "BAL", "conference": "AFC", "division": "AFC North"},
                {"team": "KC", "conference": "AFC", "division": "AFC West"},
            ]
        ),
    )

    response = client.get("/api/power-rankings?season=2025")

    body = response.json()
    by_team = {r["team"]: r for r in body["rankings"]}
    assert by_team["BAL"]["wins"] == 1
    assert by_team["BAL"]["losses"] == 0
    assert by_team["KC"]["wins"] == 0
    assert by_team["KC"]["losses"] == 1


def test_get_predictions_batch_is_keyed_by_game_id(client):
    response = client.get("/api/predictions/2025/1/batch")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"2025_01_BAL_KC"}
    assert body["2025_01_BAL_KC"]["home_win_prob"] == 0.6


def test_get_predictions_batch_skips_one_failing_game_without_failing_the_rest(client, monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [
                {"game_id": "bad_game", "season": season, "week": week,
                 "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
                 "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5},
                {"game_id": "good_game", "season": season, "week": week,
                 "gameday": "2025-09-05", "home_team": "GB", "away_team": "PHI",
                 "home_score": None, "away_score": None, "spread_line": 1.5, "total_line": 45.0},
            ]
        ),
    )

    def flaky_predict(models, home, away, games_df, spread_line=None, total_line=None):
        if home == "BAL":
            raise ValueError("feature build blew up")
        return {"home_win_prob": 0.6, "away_win_prob": 0.4}

    monkeypatch.setattr(routes, "_predict_game_from_models", flaky_predict)

    response = client.get("/api/predictions/2025/1/batch")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"good_game"}


def test_get_team_form_returns_last_n_games_with_result(client, monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [
                {"game_id": "g1", "season": 2025, "week": 1, "gameday": "2025-09-04",
                 "home_team": "BAL", "away_team": "KC", "home_score": 27, "away_score": 20},
                {"game_id": "g2", "season": 2025, "week": 2, "gameday": "2025-09-11",
                 "home_team": "KC", "away_team": "BAL", "home_score": 24, "away_score": 10},
            ]
        ),
    )

    response = client.get("/api/teams/BAL/form?season=2025&n=5")

    assert response.status_code == 200
    body = response.json()
    assert body["team"] == "BAL"
    form = body["recent_form"]
    assert len(form) == 2
    # g1: BAL home, won 27-20
    assert form[0]["opponent"] == "KC"
    assert form[0]["is_home"] is True
    assert form[0]["result"] == "W"
    assert form[0]["team_score"] == 27
    # g2: BAL away, lost 10-24
    assert form[1]["opponent"] == "KC"
    assert form[1]["is_home"] is False
    assert form[1]["result"] == "L"
    assert form[1]["team_score"] == 10


def test_get_team_form_respects_n_limit(client, monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [
                {"game_id": f"g{i}", "season": 2025, "week": i, "gameday": f"2025-09-{i:02d}",
                 "home_team": "BAL", "away_team": "KC", "home_score": 20 + i, "away_score": 10}
                for i in range(1, 8)
            ]
        ),
    )

    response = client.get("/api/teams/BAL/form?season=2025&n=3")

    assert response.status_code == 200
    body = response.json()
    assert len(body["recent_form"]) == 3
    # Should be the last 3 chronologically (g5, g6, g7)
    assert [f["game_id"] for f in body["recent_form"]] == ["g5", "g6", "g7"]


def test_get_head_to_head_returns_past_meetings(client, monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "2025_09_LV_KC", "season": season, "week": week,
              "gameday": "2025-11-02", "home_team": "KC", "away_team": "LV",
              "home_score": None, "away_score": None}]
        ),
    )
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [
                {"game_id": "g_old", "season": 2024, "week": 15, "gameday": "2024-12-14",
                 "home_team": "LV", "away_team": "KC", "home_score": 17, "away_score": 24},
                {"game_id": "g_unrelated", "season": 2024, "week": 15, "gameday": "2024-12-14",
                 "home_team": "DEN", "away_team": "LAC", "home_score": 10, "away_score": 3},
            ]
        ),
    )

    response = client.get("/api/games/2025_09_LV_KC/head-to-head?season=2025&week=9&n_seasons=8")

    assert response.status_code == 200
    body = response.json()
    assert body["game_id"] == "2025_09_LV_KC"
    meetings = body["meetings"]
    assert len(meetings) == 1
    assert meetings[0]["game_id"] == "g_old"
    assert meetings[0]["home_team"] == "LV"
    assert meetings[0]["away_team"] == "KC"


def test_get_head_to_head_404s_for_unknown_game(client, monkeypatch):
    monkeypatch.setattr(routes.schedules, "fetch_week_games", lambda season, week: pd.DataFrame(columns=["game_id", "home_team", "away_team"]))

    response = client.get("/api/games/nope/head-to-head?season=2025&week=9")

    assert response.status_code == 404

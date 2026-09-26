import math

from fastapi.testclient import TestClient

from nfl_predictor.api import routes
from nfl_predictor.api.main import app
from nfl_predictor.models import game_outcome


def test_nan_lines_are_treated_as_missing():
    """nflverse leaves spread_line/total_line as NaN (not None) for games without a line yet.
    `margin_to_probabilities` only checked `is not None`, so a NaN line produced NaN cover and
    over probabilities -- which is how a missing line became a nonsense number instead of an
    absent one."""
    result = game_outcome.margin_to_probabilities(
        -3.0, 13.0, spread_line=float("nan"), total_line=float("nan"), predicted_total=40.0, total_sigma=12.0,
    )

    assert result["home_cover_prob"] is None and result["away_cover_prob"] is None
    assert "over_prob" not in result
    assert math.isclose(result["home_win_prob"] + result["away_win_prob"], 1.0)


def test_batch_from_a_snapshot_with_nan_values_serves_null_not_500(monkeypatch):
    """The live 500. Old weeks in public_snapshot.json are reused verbatim, so serving has to
    sanitize as well as computing: Starlette refuses to serialize NaN and answers 500."""
    snapshot = {"season": 2026, "weeks": {"4": {"games": [], "player_props": [], "predictions": {
        "2026_04_GB_TB": {"home_win_prob": 0.55, "away_win_prob": 0.45,
                          "home_cover_prob": float("nan"), "over_prob": float("inf")},
    }}}}
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snapshot)

    response = TestClient(app).get("/api/predictions/2026/4/batch")

    assert response.status_code == 200
    assert response.json()["2026_04_GB_TB"] == {
        "home_win_prob": 0.55, "away_win_prob": 0.45, "home_cover_prob": None, "over_prob": None,
    }


def test_json_safe_leaves_ordinary_values_alone():
    """The sanitizer must not turn a legitimate 0 or a string into null."""
    payload = {"a": 0, "b": 0.0, "c": "nan", "d": [1, None, False], "e": {"f": 2.5}}
    assert routes._json_safe(payload) == payload

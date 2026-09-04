import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models import game_outcome


def _toy_frame():
    rng = np.random.default_rng(42)
    n = 60
    rating_diff = rng.normal(0, 100, n)
    margin = rating_diff * 0.05 + rng.normal(0, 10, n)
    return pd.DataFrame(
        {
            "home_pregame_rating": 1500 + rating_diff / 2,
            "away_pregame_rating": 1500 - rating_diff / 2,
            "rating_diff": rating_diff,
            "home_points_scored_roll": rng.normal(24, 5, n),
            "home_points_allowed_roll": rng.normal(21, 5, n),
            "away_points_scored_roll": rng.normal(23, 5, n),
            "away_points_allowed_roll": rng.normal(22, 5, n),
            "home_rest_days": 7,
            "away_rest_days": 7,
            "div_game": 0,
            "margin": margin,
        }
    )


FEATURE_COLS = [
    "home_pregame_rating", "away_pregame_rating", "rating_diff",
    "home_points_scored_roll", "home_points_allowed_roll",
    "away_points_scored_roll", "away_points_allowed_roll",
    "home_rest_days", "away_rest_days", "div_game",
]


def test_fit_margin_regression_predicts_signed_margin():
    df = _toy_frame()
    model = game_outcome.fit_margin_regression(df[FEATURE_COLS], df["margin"])

    preds = model.predict(df[FEATURE_COLS])
    # A model fit on data where margin correlates with rating_diff should
    # recover a positive relationship.
    assert np.corrcoef(preds, df["margin"])[0, 1] > 0.3


def test_fit_xgb_margin_predicts_signed_margin():
    df = _toy_frame()
    model = game_outcome.fit_xgb_margin(df[FEATURE_COLS], df["margin"])

    preds = model.predict(df[FEATURE_COLS])
    assert np.corrcoef(preds, df["margin"])[0, 1] > 0.3


def test_residual_sigma_is_positive():
    df = _toy_frame()
    model = game_outcome.fit_margin_regression(df[FEATURE_COLS], df["margin"])

    sigma = game_outcome.residual_sigma(model, df[FEATURE_COLS], df["margin"])

    assert sigma > 0


def test_margin_to_probabilities_favors_positive_margin():
    result = game_outcome.margin_to_probabilities(predicted_margin=7.0, sigma=13.0)

    assert result["home_win_prob"] > 0.5
    assert result["home_win_prob"] + result["away_win_prob"] == pytest.approx(1.0)


def test_margin_to_probabilities_includes_cover_and_total_when_lines_given():
    result = game_outcome.margin_to_probabilities(
        predicted_margin=7.0, sigma=13.0, spread_line=-3.0,
        total_line=45.0, predicted_total=48.0, total_sigma=10.0,
    )

    assert "home_cover_prob" in result
    assert "away_cover_prob" in result
    assert result["home_cover_prob"] + result["away_cover_prob"] == pytest.approx(1.0)
    assert "over_prob" in result
    assert "under_prob" in result
    assert result["over_prob"] + result["under_prob"] == pytest.approx(1.0)
    assert result["over_prob"] > 0.5  # predicted_total (48) is above total_line (45)


def test_margin_to_probabilities_spread_line_uses_nflverse_expected_margin_convention():
    # nflverse's spread_line is the home team's *expected margin*: positive
    # means home favored by that many points, and home covers only when its
    # actual margin exceeds spread_line. Here the model predicts home wins
    # by 3, but the line has home favored by 6 (home needs to beat that
    # expected margin to cover) -- so home_cover_prob should be well under
    # 0.5. Under the old (buggy) `margin > -spread_line` convention this
    # scenario would incorrectly come out well over 0.5 (~0.76 vs. the
    # correct ~0.41), so this pins the direction against regression.
    result = game_outcome.margin_to_probabilities(
        predicted_margin=3.0, sigma=13.0, spread_line=6.0,
    )

    assert result["home_cover_prob"] < 0.5

"""build.py — the single feature-construction entry point for game-outcome
models. Every consumer (training, walk-forward evaluation, live serving)
must call build_training_frame / build_features_for_game rather than
reimplementing feature logic inline — same discipline PL_Predictor's
features/build.py documents.
"""

from __future__ import annotations

import pandas as pd

from . import power_ratings, rest_days, rolling_form

FEATURE_COLUMNS = [
    "home_pregame_rating", "away_pregame_rating", "rating_diff",
    "home_points_scored_roll", "home_points_allowed_roll",
    "away_points_scored_roll", "away_points_allowed_roll",
    "home_rest_days", "away_rest_days",
    "div_game",
]


def _assemble(games_df: pd.DataFrame) -> pd.DataFrame:
    df = power_ratings.compute_pregame_ratings(games_df)
    df = rolling_form.add_rolling_form(df)
    df = rest_days.add_rest_days(df)
    df["rating_diff"] = df["home_pregame_rating"] - df["away_pregame_rating"]
    df["home_rest_days"] = df["home_rest_days"].fillna(7)
    df["away_rest_days"] = df["away_rest_days"].fillna(7)
    if "div_game" not in df.columns:
        df["div_game"] = 0
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    return df


def build_training_frame(games_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _assemble(games_df)
    played = df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)
    played["margin"] = played["home_score"] - played["away_score"]
    played["total_points"] = played["home_score"] + played["away_score"]
    return played, FEATURE_COLUMNS


def build_features_for_game(home_team: str, away_team: str, games_df: pd.DataFrame) -> pd.Series:
    """One live feature row for an upcoming home_team vs away_team game,
    computed from every played game in games_df (ratings/rolling form as of
    right now)."""
    ratings = power_ratings.final_ratings(games_df)
    played = games_df[games_df["home_score"].notna() & games_df["away_score"].notna()]

    def _recent_form(team: str) -> tuple[float, float]:
        appearances = pd.concat(
            [
                played[played["home_team"] == team][["gameday", "home_score", "away_score"]].rename(
                    columns={"home_score": "scored", "away_score": "allowed"}
                ),
                played[played["away_team"] == team][["gameday", "away_score", "home_score"]].rename(
                    columns={"away_score": "scored", "home_score": "allowed"}
                ),
            ]
        ).sort_values("gameday")
        recent = appearances.tail(5)
        if recent.empty:
            return float("nan"), float("nan")
        return float(recent["scored"].mean()), float(recent["allowed"].mean())

    def _rest_days(team: str) -> float | None:
        appearances = pd.concat(
            [
                played[played["home_team"] == team][["gameday"]],
                played[played["away_team"] == team][["gameday"]],
            ]
        ).sort_values("gameday")
        if appearances.empty:
            return None
        last_game = pd.to_datetime(appearances.iloc[-1]["gameday"])
        return float((pd.Timestamp.now().normalize() - last_game).days)

    home_scored, home_allowed = _recent_form(home_team)
    away_scored, away_allowed = _recent_form(away_team)
    home_rating = ratings.get(home_team, power_ratings.DEFAULT_START_RATING)
    away_rating = ratings.get(away_team, power_ratings.DEFAULT_START_RATING)
    home_rest = _rest_days(home_team)
    away_rest = _rest_days(away_team)

    return pd.Series(
        {
            "home_pregame_rating": home_rating,
            "away_pregame_rating": away_rating,
            "rating_diff": home_rating - away_rating,
            "home_points_scored_roll": home_scored,
            "home_points_allowed_roll": home_allowed,
            "away_points_scored_roll": away_scored,
            "away_points_allowed_roll": away_allowed,
            "home_rest_days": home_rest if home_rest is not None else 7.0,
            "away_rest_days": away_rest if away_rest is not None else 7.0,
            "div_game": 0,
        }
    )

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


def _div_game_for(games_df: pd.DataFrame, home_team: str, away_team: str) -> int:
    """Whether this matchup is a divisional game, read from the schedule.

    `div_game` is a real modelled feature (`FEATURE_COLUMNS`) and training fills it
    from the schedule, where it toggles for roughly a third of games. It used to be
    hardcoded to 0 at serving time, so the model was fitted with the covariate
    varying and served with it permanently constant -- the fitted coefficient was
    dead weight at inference.

    Extracted from `build_features_for_game` so the season-scoping rule can be
    tested without reconstructing a full valid games frame.

    Two rules, both of which the first version of this lookup got wrong:

    - **Scope to the most recent season.** `routes._load_game_history`
      concatenates eight completed seasons and `schedules.fetch_schedules` sorts
      `["season", "week", "gameday"]` -- oldest first -- so the first home/away
      pair on record is from the *earliest* season available. Demonstrated: a frame
      where 2019's BUF-MIA was divisional and 2021's and 2026's were not served
      `div_game = 1` for a 2026 game whose true value is 0.
    - **Take the latest matchup inside that season**, not the first, since a
      division's two meetings are both in the frame.

    Falls back to 0 for any frame without the column, which is also the training
    default, so a games frame that never carried `div_game` still serves.
    """
    if "div_game" not in games_df.columns:
        return 0
    candidates = games_df
    if "season" in games_df.columns and games_df["season"].notna().any():
        candidates = games_df[games_df["season"] == games_df["season"].max()]
    matchup = candidates[
        (candidates["home_team"] == home_team) & (candidates["away_team"] == away_team)
    ]
    if matchup.empty:
        return 0
    if "week" in matchup.columns:
        matchup = matchup.sort_values("week")
    value = matchup.iloc[-1]["div_game"]
    return 0 if pd.isna(value) else int(value)


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

    # `div_game` is a real modelled feature (FEATURE_COLUMNS) and training fills
    # it from the schedule, where it toggles for roughly a third of games. It
    # used to be hardcoded to 0 here, so the model was fitted with the covariate
    # varying and served with it permanently constant -- the fitted coefficient
    # was dead weight at inference. Read it from the schedule for this matchup
    # when the column is present, and fall back to 0 for any games frame that
    # does not carry it (which is also the training default).
    div_game = _div_game_for(games_df, home_team, away_team)

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
            "div_game": div_game,
        }
    )

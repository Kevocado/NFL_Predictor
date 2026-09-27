"""player_usage.py — player-level rolling usage/production features for
anytime-TD and yardage prop models. Same shift(1)-then-rolling discipline as
features/rolling_form.py, applied per player instead of per team."""

from __future__ import annotations

import pandas as pd

ROLL_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "targets", "carries", "receptions"]
PLAYER_FEATURE_COLUMNS = [f"{stat}_roll" for stat in ROLL_STATS]


def _add_rolling(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = df.groupby("player_id")
    for stat in ROLL_STATS:
        df[f"{stat}_roll"] = grouped[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df


def build_player_training_frame(player_stats_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _add_rolling(player_stats_df)
    df["anytime_td"] = (
        (df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0) + df["passing_tds"].fillna(0)) > 0
    ).astype(int)
    return df, PLAYER_FEATURE_COLUMNS


def build_features_for_player(
    player_id: str,
    player_stats_df: pd.DataFrame,
    season: int | None = None,
    window: int = 5,
) -> pd.Series | None:
    """Pregame rolling features for one player, matching the training discipline.

    `season` scopes the history to a single season. This is not a refinement: the
    previous version ranged over every season in the frame, so a player on a 2026
    roster was scored on whatever his last five games were in whatever year those
    were. Measured 2026-09-27, 416 of 458 non-zero served rows were predicted off
    2024 data, and because `recent_team` comes from the current roster while the
    features came from an old one, a traded player was credited to his new team on
    his former team's numbers.

    The rolling mean deliberately **excludes the most recent game**, reproducing
    `_add_rolling`'s `shift(1).rolling(window, min_periods=1)`. A pregame feature
    cannot know the game being predicted, so including it makes the served quantity
    differ from the trained one under the same column name. That skew was real and
    measurable: Aaron Rodgers (`00-0023459`) was served `passing_yards_roll` 254.0
    against a training-consistent 249.0.

    Returns None when the player has no rows in scope. Callers must skip rather than
    substitute a default: the model is never trained on the all-zero region, and
    scoring it returns the origin intercept (passing_yards 62.592, anytime_td_prob
    0.111846), which is how 430 of 880 live rows -- 48.9% -- came back bit-identical
    before this was fixed.
    """
    history = player_stats_df[player_stats_df["player_id"] == player_id]
    if season is not None:
        history = history[history["season"] == season]
    history = history.sort_values(["season", "week"])
    if history.empty:
        return None
    # Drop the latest game, then average the window before it, exactly as training does.
    prior = history.iloc[:-1].tail(window)
    return pd.Series(
        {
            f"{stat}_roll": float(prior[stat].mean()) if not prior.empty else float("nan")
            for stat in ROLL_STATS
        }
    )

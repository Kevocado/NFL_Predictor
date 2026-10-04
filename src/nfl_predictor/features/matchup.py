"""matchup.py -- opponent-defense and game-context features.

Every rolling value uses shift(1)-then-rolling: a week-W feature sees only
games < W. This is the anti-leakage contract and `tests/features/test_matchup.py`
pins it. The discipline mirrors `features/rolling_form.py`, which is this repo's
existing shift(1)-then-groupby pattern.
"""
from __future__ import annotations

import pandas as pd

from ..data.weather import HIGH_WIND_KPH, INDOOR_STADIUMS, STADIUM_COORDS

#: (feature source column, opponent-allowed feature) per position group.
_OPPONENT_ROLLS = (
    ("passing_yards", "opp_pass_yds_allowed_roll"),
    ("rushing_yards", "opp_rush_yds_allowed_roll"),
    ("receiving_yards", "opp_rec_yds_allowed_roll"),
)

#: Schedule columns carried onto every player row.
_SCHEDULE_COLS = ["season", "week", "recent_team", "opponent_team", "game_id",
                  "spread_line", "total_line", "gameday", "stadium", "is_home"]


def _past_rolling_mean(s: pd.Series, window: int = 5) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def _opponent_allowed(weekly: pd.DataFrame) -> pd.DataFrame:
    """Yards each defense allowed to each position group, rolled on its own history.

    Grouping on `opponent_team` makes the row "what the defense this player is
    facing gave up", which is what the feature means. The rolling window is
    shift(1) within (defense, position), so the defense's week-W game -- the very
    game the player is about to play -- is excluded.
    """
    keys = ["season", "week", "opponent_team", "position"]
    allowed = (
        weekly.groupby(keys, as_index=False)
        .agg(**{dst: (src, "sum") for src, dst in _OPPONENT_ROLLS})
        .sort_values(["opponent_team", "position", "season", "week"])
    )
    grouped = allowed.groupby(["opponent_team", "position"])
    for _, dst in _OPPONENT_ROLLS:
        allowed[dst] = grouped[dst].transform(_past_rolling_mean)
    return allowed[[*keys, *[dst for _, dst in _OPPONENT_ROLLS]]]


def _team_appearances(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game) with home/away resolved, so rest days and the
    game's own columns can be joined on the player's own team."""
    shared = ["season", "week", "game_id", "spread_line", "total_line", "gameday", "stadium"]
    home = schedules[[*shared, "home_team", "away_team"]].rename(
        columns={"home_team": "recent_team", "away_team": "opponent_team"}
    ).assign(is_home=1)
    away = schedules[[*shared, "away_team", "home_team"]].rename(
        columns={"away_team": "recent_team", "home_team": "opponent_team"}
    ).assign(is_home=0)

    both = pd.concat([home, away], ignore_index=True)
    both["gameday"] = pd.to_datetime(both["gameday"])
    both = both.sort_values(["recent_team", "gameday"])
    both["rest_days"] = both.groupby("recent_team")["gameday"].diff().dt.days
    return both[_SCHEDULE_COLS + ["rest_days"]]


def _outdoor_flag(stadium: pd.Series) -> pd.Series:
    """1 outdoor, 0 roofed, NaN for a stadium we have no record for.

    NaN rather than a guess: an unrecognised stadium must not be silently
    treated as open air and handed a wind feature it never had.
    """
    return pd.Series(
        [None if not isinstance(s, str) or (s not in INDOOR_STADIUMS and s not in STADIUM_COORDS)
         else int(s not in INDOOR_STADIUMS) for s in stadium],
        index=stadium.index, dtype="float64",
    )


def add_matchup_features(weekly_df: pd.DataFrame, schedules_df: pd.DataFrame,
                         weather_by_game: dict | None = None) -> pd.DataFrame:
    df = weekly_df.merge(
        _opponent_allowed(weekly_df),
        on=["season", "week", "opponent_team", "position"], how="left",
    )
    df = df.merge(_team_appearances(schedules_df),
                  on=["season", "week", "recent_team", "opponent_team"], how="left")

    # Spread is quoted from the home team's perspective, so the home side's
    # implied total is (total - spread)/2 and the away side's is (total + spread)/2.
    side = 2 * df["is_home"] - 1
    df["implied_team_total"] = (df["total_line"] - side * df["spread_line"]) / 2
    df["game_total"] = df["total_line"]

    df["is_outdoor"] = _outdoor_flag(df["stadium"])
    weather = weather_by_game or {}
    df["temp_c"] = df["game_id"].map(lambda g: weather.get(g, {}).get("temp_c"))
    df["wind_kph"] = df["game_id"].map(lambda g: weather.get(g, {}).get("wind_kph"))
    df["precip_mm"] = df["game_id"].map(lambda g: weather.get(g, {}).get("precip_mm"))
    df["high_wind_flag"] = ((df["wind_kph"] >= HIGH_WIND_KPH) & (df["is_outdoor"] == 1)).astype(int)

    return df
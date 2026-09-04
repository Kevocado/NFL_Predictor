"""rest_days.py — days since each team's previous game, no lookahead.

Bye weeks (roughly one per team per season) make this a bigger swing for
NFL than PL_Predictor's midweek-fixture-congestion equivalent — a team off
a bye typically shows 14 rest days instead of the usual 7.
"""

from __future__ import annotations

import pandas as pd


def _team_appearances(games_df: pd.DataFrame) -> pd.DataFrame:
    home = games_df[["game_id", "gameday", "home_team"]].rename(columns={"home_team": "team"})
    away = games_df[["game_id", "gameday", "away_team"]].rename(columns={"away_team": "team"})
    appearances = pd.concat([home, away], ignore_index=True)
    return appearances.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def add_rest_days(games_df: pd.DataFrame) -> pd.DataFrame:
    games_df = games_df.copy()
    games_df["gameday"] = pd.to_datetime(games_df["gameday"])
    appearances = _team_appearances(games_df)
    appearances["gameday"] = pd.to_datetime(appearances["gameday"])
    appearances["prior_gameday"] = appearances.groupby("team")["gameday"].shift(1)
    appearances["rest_days"] = (appearances["gameday"] - appearances["prior_gameday"]).dt.days

    home_rest = appearances.rename(columns={"team": "home_team", "rest_days": "home_rest_days"})[
        ["game_id", "home_team", "home_rest_days"]
    ]
    away_rest = appearances.rename(columns={"team": "away_team", "rest_days": "away_rest_days"})[
        ["game_id", "away_team", "away_rest_days"]
    ]

    result = games_df.merge(home_rest, on=["game_id", "home_team"], how="left")
    result = result.merge(away_rest, on=["game_id", "away_team"], how="left")
    return result

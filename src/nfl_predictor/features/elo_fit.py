"""Pick Elo K and home-field by out-of-sample log loss on past games only."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .power_ratings import ELO_SCALE, compute_pregame_ratings

K_GRID = (10.0, 15.0, 20.0, 25.0, 30.0)
HFA_GRID = (30.0, 45.0, 65.0, 80.0)


def _log_loss(games: pd.DataFrame, k: float, hfa: float) -> float:
    rated = compute_pregame_ratings(games, k=k, home_field=hfa)
    # Only score completed games (both scores present)
    completed = rated["home_score"].notna() & rated["away_score"].notna()
    if not completed.any():
        return float("inf")
    diff = rated.loc[completed, "home_pregame_rating"] - rated.loc[completed, "away_pregame_rating"] + hfa
    p = np.clip(1.0 / (1.0 + 10 ** (-diff / ELO_SCALE)), 1e-6, 1 - 1e-6)
    # Handle ties as 0.5, not as away win
    y = np.where(
        rated.loc[completed, "home_score"] > rated.loc[completed, "away_score"], 1.0,
        np.where(rated.loc[completed, "home_score"] < rated.loc[completed, "away_score"], 0.0, 0.5)
    )
    # Score only the settled part (burn-in)
    n = len(y)
    burn = n // 4
    if n <= burn:
        return float("inf")
    return float(-np.mean(y[burn:] * np.log(p[burn:]) + (1 - y[burn:]) * np.log(1 - p[burn:])))


def fit_elo_constants(games: pd.DataFrame) -> dict:
    ordered = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    best = min(product(K_GRID, HFA_GRID), key=lambda kh: (_log_loss(ordered, *kh), kh))
    return {"k": best[0], "home_field": best[1]}
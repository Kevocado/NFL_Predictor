"""Join model predictions to live odds and surface single-game value edges."""

from __future__ import annotations

import math

import pandas as pd
from scipy.optimize import brentq

MAX_ODDS_AGE_SECONDS = 60 * 60


def _shin_two_way(price_a: float, price_b: float) -> tuple[float, float] | None:
    """Return two Shin-de-vigged probabilities, or ``None`` for invalid lines."""
    if not all(isinstance(price, (int, float)) and math.isfinite(price) and price > 1 for price in (price_a, price_b)):
        return None

    pi_a, pi_b = 1.0 / price_a, 1.0 / price_b
    overround = pi_a + pi_b
    if overround <= 1.0:
        return None

    def true_prob(pi: float, z: float) -> float:
        inside = z * z + 4 * (1 - z) * (pi * pi) / overround
        return (max(inside, 0.0) ** 0.5 - z) / (2 * (1 - z))

    def z_equation(z: float) -> float:
        return true_prob(pi_a, z) + true_prob(pi_b, z) - 1.0

    try:
        z = brentq(z_equation, 0.0, 0.2)
    except ValueError:
        return pi_a / overround, pi_b / overround

    p_a, p_b = true_prob(pi_a, z), true_prob(pi_b, z)
    total = p_a + p_b
    return p_a / total, p_b / total


def devig_h2h(home_price: float, away_price: float) -> dict | None:
    """Return Shin-de-vigged home and away win probabilities."""
    result = _shin_two_way(home_price, away_price)
    if result is None:
        return None
    home, away = result
    return {"home_win": home, "away_win": away}


def devig_totals(over_price: float, under_price: float) -> dict | None:
    """Return Shin-de-vigged over and under probabilities."""
    result = _shin_two_way(over_price, under_price)
    if result is None:
        return None
    over, under = result
    return {"over": over, "under": under}


def _best_price(odds_df: pd.DataFrame, event_id: str, market: str, outcome_name: str) -> float | None:
    required_columns = {"event_id", "market", "outcome_name", "price"}
    if not required_columns.issubset(odds_df.columns):
        return None
    rows = odds_df[
        (odds_df["event_id"] == event_id)
        & (odds_df["market"] == market)
        & (odds_df["outcome_name"] == outcome_name)
    ]
    if rows.empty:
        return None
    price = rows.loc[rows["price"].idxmax(), "price"]
    return float(price) if pd.notna(price) else None


def _recommended_side(row: dict, edge_threshold: float) -> list[str]:
    """Choose one best positive edge so output can never describe a parlay."""
    candidates = [
        (side, row.get(f"{side}_edge"))
        for side in ("home_win", "away_win", "over", "under")
        if row.get(f"{side}_edge") is not None and row[f"{side}_edge"] > edge_threshold
    ]
    if not candidates:
        return []
    return [max(candidates, key=lambda candidate: candidate[1])[0]]


def build_value_bet_table(
    games_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    predictions: dict[str, dict],
    edge_threshold: float = 0.05,
) -> pd.DataFrame:
    """Build one row per game with de-vigged edges and at most one recommendation."""
    rows = []
    for _, game in games_df.iterrows():
        game_id, home, away = game["game_id"], game["home_team"], game["away_team"]
        prediction = predictions.get(game_id, {})
        row = {
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "commence_time": game["commence_time"],
            **prediction,
        }

        home_price = _best_price(odds_df, game_id, "h2h", home)
        away_price = _best_price(odds_df, game_id, "h2h", away)
        h2h_implied = devig_h2h(home_price, away_price) if home_price and away_price else None

        over_price = _best_price(odds_df, game_id, "totals", "Over")
        under_price = _best_price(odds_df, game_id, "totals", "Under")
        totals_implied = devig_totals(over_price, under_price) if over_price and under_price else None

        row["home_win_edge"] = prediction.get("home_win_prob", 0) - h2h_implied["home_win"] if h2h_implied else None
        row["away_win_edge"] = prediction.get("away_win_prob", 0) - h2h_implied["away_win"] if h2h_implied else None
        row["over_edge"] = prediction["over_prob"] - totals_implied["over"] if totals_implied and "over_prob" in prediction else None
        row["under_edge"] = prediction["under_prob"] - totals_implied["under"] if totals_implied and "under_prob" in prediction else None
        row["value_bet_flags"] = _recommended_side(row, edge_threshold)
        rows.append(row)

    return pd.DataFrame(rows)

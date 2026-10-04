"""forward_tick.py -- snapshot a slate of book lines and log the ones worth taking.

For each pre-kickoff game: fetch the book's prop lines, turn them into
P(over) with the quantile models, and record every side whose edge clears 5%.
Either side qualifies -- a 5% under is as much a pick as a 5% over -- and the
side taken is written explicitly, because CLV's sign and the hit test both
depend on it.

**A post-kickoff game is skipped, not corrected.** A line recorded after
kickoff is not a bettable price, so it is not a pick. The rest of the slate
still runs.

**The budget guard runs before the first fetch.** An exhausted month writes
nothing at all rather than a partial slate that looks like a record.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from ..models.prop_probability import american_to_breakeven, edge_vs_line, p_over_from_quantiles
from ..odds.props_snapshot import (
    BudgetExhausted, credits_sufficient, fetch_props_for_event, probe_props_coverage,
)
from . import store

logger = logging.getLogger(__name__)

#: Kevin's gate: singles only, 5% edge against the line.
EDGE_GATE = 0.05

#: Book market key -> the market name the quantile models were trained on.
MARKET_MAP = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_rec_yds": "receiving_yards",
    "player_receptions": "receptions",
}


def quantiles_for(market: str) -> dict[float, float]:
    """Predicted quantiles for one prop row. Replaced at the call site by the
    trained model; named here so the tick's seam is one function."""
    raise NotImplementedError("wire a trained quantile artifact in via quantiles_for")


def _has_spread(quantiles: dict) -> bool:
    """False when every predicted quantile is the same value.

    A degenerate distribution is a model that failed to fit this row, not a
    confident one, and pricing it produces a confident-looking number.
    """
    values = [float(v) for v in quantiles.values() if v is not None and not pd.isna(v)]
    return len(values) >= 2 and (max(values) - min(values)) > 0


def _is_post_kickoff(commence_time: str) -> bool:
    try:
        kickoff = store._utc_instant(commence_time)
    except Exception:
        # An unparseable kickoff cannot be proven pre-kickoff, so it is skipped.
        return True
    if kickoff is None:
        return True
    return kickoff <= datetime.now(timezone.utc)


def run_forward_tick(games: list[dict], market_quantiles: dict[str, dict] | None = None,
                     edge_gate: float = EDGE_GATE) -> dict:
    """Snapshot one slate.

    `market_quantiles` maps a book market key to that row's predicted quantiles
    -- either a plain `{level: value}` dict, or a callable taking the line and
    returning one, which is how a trained model is injected. Injecting rather
    than importing is what lets the tick be tested without an artifact on disk.

    Returns the counters the forward report reads.
    """
    games = list(games)
    result = {
        "games": 0, "games_skipped_post_kickoff": 0,
        "props_snapshotted": 0, "picks_logged": 0, "credits_remaining": 0,
        "no_props_coverage": False,
    }

    live = []
    for game in games:
        if _is_post_kickoff(game["commence_time"]):
            result["games_skipped_post_kickoff"] += 1
            logger.info("skipping %s: kickoff has passed", game["game_id"])
            continue
        live.append(game)
    result["games"] = len(live)
    if not live:
        return result

    # One props call per game per tick. The whole slate is costed up front so a
    # month that cannot afford it writes nothing rather than half a slate.
    needed = len(live)
    if not credits_sufficient(needed):
        logger.warning("budget: %d credits needed, not fetching anything", needed)
        result["credits_remaining"] = 0
        result["no_props_coverage"] = True
        return result

    rows: list[dict] = []
    for game in live:
        event_id = game["game_id"]
        coverage = probe_props_coverage(event_id)
        if not coverage.get("has_any"):
            logger.warning("no player-prop coverage for %s; waiting, not falling back", event_id)
            result["no_props_coverage"] = True
            continue
        try:
            props = fetch_props_for_event(event_id, credits_needed=1)
        except BudgetExhausted:
            logger.warning("budget exhausted mid-slate at %s; stopping", event_id)
            break
        result["props_snapshotted"] += len(props)

        for prop in props:
            market = MARKET_MAP.get(prop["market"])
            predictor = (market_quantiles or {}).get(prop["market"])
            if market is None or predictor is None:
                continue
            quantiles = predictor(prop["line"]) if callable(predictor) else predictor
            if not _has_spread(quantiles):
                # A zero-width distribution cannot price a line: `p_over_from_
                # quantiles` returns the ceiling for a line sitting on it, which
                # would clear the 5% gate on a fabricated 98% edge.
                logger.warning("degenerate quantiles for %s %s; skipped",
                               prop["player_name"], prop["market"])
                continue
            p_over = p_over_from_quantiles(quantiles, prop["line"])

            over_edge = edge_vs_line(p_over, prop["over_odds"])
            under_edge = edge_vs_line(1.0 - p_over, prop["under_odds"])
            if over_edge >= edge_gate:
                rows.append(_row(game, prop, market, "over", p_over, over_edge, prop["over_odds"]))
            if under_edge >= edge_gate:
                rows.append(_row(game, prop, market, "under", 1.0 - p_over, under_edge, prop["under_odds"]))

    if rows:
        result["picks_logged"] = store.record_player_prop_predictions(rows)
    return result


def _row(game, prop, market, side, p_side, edge, odds) -> dict:
    return {
        "game_id": game["game_id"],
        "player_id": prop["player_id"],
        "player_name": prop["player_name"],
        "position": prop.get("position"),
        "market": market,
        "side": side,
        # predicted_value is the q50, kept for continuity with the existing
        # yardage columns; the forward test grades against the book line.
        "predicted_value": float(prop["line"]),
        "line_at_snapshot": float(prop["line"]),
        "odds_at_snapshot": float(odds) if odds is not None else None,
        "model_p_over": float(p_side),
        "edge_vs_breakeven": float(edge),
    }
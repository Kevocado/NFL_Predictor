"""Per-player season table and position leaderboards from nflverse weekly
stats. EPA is the headline efficiency number; missing values stay None."""
from __future__ import annotations

import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]
SUM_COLS = ["completions", "attempts", "passing_yards", "passing_tds", "interceptions", "carries",
            "rushing_yards", "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds"]
EPA_COLS = ["passing_epa", "rushing_epa", "receiving_epa"]


def _mean(s: pd.Series) -> float | None:
    s = s.dropna()
    return None if s.empty else round(float(s.mean()), 3)


def player_season(weekly: pd.DataFrame, season: int) -> dict:
    df = weekly[(weekly["season"] == season) & weekly["position"].isin(POSITIONS)].sort_values("week")
    players = []
    for pid, g in df.groupby("player_id"):
        last = g.iloc[-1]
        epa_present = g[EPA_COLS].notna().any().any()
        row = {"player_id": pid, "name": last["player_display_name"], "team": last["recent_team"],
               "position": last["position"], "games": int(g["week"].nunique())}
        row.update({c: int(g[c].fillna(0).sum()) for c in SUM_COLS})
        row["epa_total"] = round(float(g[EPA_COLS].astype(float).fillna(0).to_numpy().sum()), 2) if epa_present else None
        row["target_share"] = _mean(g["target_share"].astype(float))
        row["air_yards_share"] = _mean(g["air_yards_share"].astype(float))
        row["fantasy_ppr_pg"] = round(float(g["fantasy_points_ppr"].fillna(0).sum()) / row["games"], 1)
        players.append(row)
    boards = {}
    for pos in POSITIONS:
        ranked = [p for p in players if p["position"] == pos and p["epa_total"] is not None]
        boards[pos] = sorted(ranked, key=lambda p: p["epa_total"], reverse=True)[:5]
    return {"players": players, "leaderboards": boards}

"""Per-team-game efficiency and per-game starting QB, aggregated from nflverse play-by-play.

Two small tables are all the game model needs from the ~50,000 plays a season: one row per team per game (offence
and defence EPA, split pass/rush, success rate) and one row per team per game naming the starting QB (the passer with
the most dropbacks) with his EPA. Both are cached per season so training reads kilobytes, not play-by-play.
"""
from __future__ import annotations

import pandas as pd

from ..config import CACHE_DIR, CURRENT_SEASON
from .hub_cache import cached_frame

PBP_AGG_COLUMNS = [
    "game_id", "season", "week", "posteam", "defteam", "play_type", "epa", "success",
    "qb_dropback", "passer_player_id", "passer_player_name",
]
PBP_AGG_CACHE_DIR = CACHE_DIR / "pbp_agg"

EFFICIENCY_COLUMNS = [
    "game_id", "team", "epa_off", "epa_off_pass", "epa_off_rush", "success_off",
    "epa_def", "epa_def_pass", "epa_def_rush", "success_def",
]
QB_GAME_COLUMNS = ["game_id", "team", "qb_id", "qb_name", "dropbacks", "epa_sum"]


def _import_pbp(years: list[int], columns: list[str]) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_pbp_data(years, columns=columns)


def load_pbp_agg(season: int) -> pd.DataFrame:
    """The play-by-play columns the aggregates need, one season, cached (the current season is refreshed)."""
    return cached_frame(
        PBP_AGG_CACHE_DIR / f"pbp_agg_{season}.parquet",
        lambda: _import_pbp([season], PBP_AGG_COLUMNS), PBP_AGG_COLUMNS, current=season == CURRENT_SEASON,
    )


def _plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Scrimmage plays with an EPA value. Penalties, kickoffs, punts and two-point tries carry no pass/run type here."""
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=PBP_AGG_COLUMNS)
    return pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()]


def team_game_efficiency(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team): what the team's OFFENCE did and what its DEFENCE allowed, per play."""
    plays = _plays(pbp)
    if plays.empty:
        return pd.DataFrame(columns=EFFICIENCY_COLUMNS)

    def side(group_col: str, prefix: str) -> pd.DataFrame:
        g = plays.groupby(["game_id", group_col])
        out = pd.DataFrame({
            f"epa_{prefix}": g["epa"].mean(),
            f"success_{prefix}": g["success"].mean(),
            f"epa_{prefix}_pass": plays[plays["play_type"] == "pass"].groupby(["game_id", group_col])["epa"].mean(),
            f"epa_{prefix}_rush": plays[plays["play_type"] == "run"].groupby(["game_id", group_col])["epa"].mean(),
        })
        out.index = out.index.set_names(["game_id", "team"])
        return out

    merged = side("posteam", "off").join(side("defteam", "def"), how="outer").reset_index()
    return merged[EFFICIENCY_COLUMNS]


def qb_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team, qb): every passer who took a dropback that game, with his dropbacks and EPA.

    The team-game STARTER (most dropbacks, lowest id on a tie) is selected separately in
    `features/qb.py`, not by dropping rows here: a backup's relief dropbacks are the history
    his rating and experience are built from when he later starts. Dropping non-starters here
    made every promoted backup arrive as a "new" QB with the prior.
    """
    plays = _plays(pbp)
    if plays.empty:
        return pd.DataFrame(columns=QB_GAME_COLUMNS)
    db = plays[(plays["qb_dropback"] == 1) & plays["passer_player_id"].notna()]
    if db.empty:
        return pd.DataFrame(columns=QB_GAME_COLUMNS)
    per_qb = (
        db.groupby(["game_id", "posteam", "passer_player_id"])
        .agg(dropbacks=("epa", "size"), epa_sum=("epa", "sum"), qb_name=("passer_player_name", "first"))
        .reset_index()
    )
    per_qb = per_qb.sort_values(["game_id", "posteam", "dropbacks", "passer_player_id"], ascending=[True, True, False, True])
    return per_qb.rename(columns={"posteam": "team", "passer_player_id": "qb_id"})[QB_GAME_COLUMNS].reset_index(drop=True)
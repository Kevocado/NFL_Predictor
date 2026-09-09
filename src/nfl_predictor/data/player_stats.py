"""player_stats.py — weekly player-level stats, cache-or-fetch from
nfl_data_py. Same per-season parquet caching pattern as data/schedules.py."""

from __future__ import annotations

import logging

import pandas as pd

from ..config import PLAYER_STATS_CACHE_DIR

logger = logging.getLogger(__name__)

KEEP_COLUMNS = [
    "player_id", "player_name", "position", "recent_team", "season", "week",
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions", "targets", "carries",
]


def _import_weekly_data(years: list[int], columns: list[str] | None = None) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_weekly_data(years, columns=columns)


def fetch_seasonal_roster(season: int) -> pd.DataFrame:
    """Current team/position for every rostered player, independent of
    whether they've played a game yet this season. Not fed into the
    rolling-usage features (those still come from fetch_weekly_player_stats)
    — this is only for identifying which players belong to which team
    before any current-season stats exist, e.g. props for week 1."""
    import nfl_data_py as nfl

    roster = nfl.import_seasonal_rosters([season])
    roster = roster.rename(columns={"team": "recent_team"})
    return roster[["player_id", "player_name", "position", "recent_team"]].drop_duplicates("player_id")


def _season_cache_path(season: int) -> "Path":
    return PLAYER_STATS_CACHE_DIR / f"{season}.parquet"


def fetch_weekly_player_stats(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    frames = []
    missing = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(season)

    if missing:
        # Fetch one season at a time — nflverse can lag in publishing a
        # season's weekly stats (e.g. right after it "completes" by our own
        # calendar-based season/week estimate) even though prior seasons are
        # fine, and a single missing year shouldn't 404 out every season in
        # the batch.
        for season in missing:
            try:
                fetched = _import_weekly_data([season], columns=KEEP_COLUMNS)[KEEP_COLUMNS].copy()
            except Exception:
                logger.info("No weekly player stats available yet for season=%s", season)
                continue
            season_df = fetched[fetched["season"] == season].reset_index(drop=True)
            season_df.to_parquet(_season_cache_path(season))
            frames.append(pd.read_parquet(_season_cache_path(season)))

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week"]).reset_index(drop=True)

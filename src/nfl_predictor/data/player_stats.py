"""player_stats.py — weekly player-level stats, cache-or-fetch from
nfl_data_py. Same per-season parquet caching pattern as data/schedules.py."""

from __future__ import annotations

import pandas as pd

from ..config import PLAYER_STATS_CACHE_DIR

KEEP_COLUMNS = [
    "player_id", "player_name", "position", "recent_team", "season", "week",
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions", "targets", "carries",
]


def _import_weekly_data(years: list[int], columns: list[str] | None = None) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_weekly_data(years, columns=columns)


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
        fetched = _import_weekly_data(missing, columns=KEEP_COLUMNS)[KEEP_COLUMNS].copy()
        for season in missing:
            season_df = fetched[fetched["season"] == season].reset_index(drop=True)
            season_df.to_parquet(_season_cache_path(season))
            frames.append(pd.read_parquet(_season_cache_path(season)))

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week"]).reset_index(drop=True)

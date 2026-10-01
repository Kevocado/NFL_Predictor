"""injuries.py — official weekly injury reports, cache-or-fetch from
nfl_data_py. Supplements ESPN's closer-to-kickoff scoreboard for gating
player-prop predictions on real availability."""

from __future__ import annotations

import time

import pandas as pd
from pathlib import Path

from ..config import INJURIES_CACHE_DIR

KEEP_COLUMNS = ["season", "week", "team", "gsis_id", "full_name", "position", "report_status"]


def _import_injuries(years: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_injuries(years)


def _season_cache_path(season: int) -> "Path":
    return INJURIES_CACHE_DIR / f"{season}.parquet"


def _cache_is_fresh(path, max_age_seconds: float | None) -> bool:
    """A cached season file is reusable when it exists and, if a limit is given, is
    younger than it. No limit means permanent, which is right for a finished season."""
    if not path.exists():
        return False
    if max_age_seconds is None:
        return True
    try:
        return (time.time() - path.stat().st_mtime) < max_age_seconds
    except OSError:
        return False


def fetch_injuries(
    seasons: list[int], force_refresh: bool = False, max_age_seconds: float | None = None
) -> pd.DataFrame:
    """Official weekly injury reports, cache-or-fetch per season.

    `max_age_seconds` makes a season's cache expire. A gate that reads this on every
    request MUST pass one for the current season: the report is updated through the
    week, and a cache written once is frozen at the first fetch, which by week 2 holds
    no row for the current week and silently gates nobody.
    """
    frames = []
    missing = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and _cache_is_fresh(path, max_age_seconds):
            frames.append(pd.read_parquet(path))
        else:
            missing.append(season)

    if missing:
        fetched = _import_injuries(missing)[KEEP_COLUMNS].copy()
        for season in missing:
            season_df = fetched[fetched["season"] == season].reset_index(drop=True)
            season_df.to_parquet(_season_cache_path(season))
            frames.append(pd.read_parquet(_season_cache_path(season)))

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def current_status_by_player(injuries_df: pd.DataFrame, season: int, week: int) -> dict[str, str]:
    """gsis_id -> report_status for players actually flagged (Out/Doubtful/
    Questionable) in a given season/week — players with no report_status
    (the common case: healthy, no injury report entry) are omitted rather
    than included with a None value, so callers can treat "in this dict" as
    "has a real status to gate on"."""
    week_df = injuries_df[(injuries_df["season"] == season) & (injuries_df["week"] == week)]
    flagged = week_df[week_df["report_status"].notna()]
    return dict(zip(flagged["gsis_id"], flagged["report_status"]))

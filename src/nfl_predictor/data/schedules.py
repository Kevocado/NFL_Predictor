"""schedules.py — game schedule and results, cache-or-fetch from nfl_data_py.

One parquet file per season under SCHEDULES_CACHE_DIR, mirroring the
per-season cache-or-fetch pattern PL_Predictor's data/football_data.py and
F1_Predictor's data/jolpica.py both use — a season's results never change
once played, so caching per-season is safe and avoids re-hitting nfl_data_py
on every call.
"""

from __future__ import annotations

import pandas as pd

from ..config import CURRENT_SEASON, SCHEDULES_CACHE_DIR

KEEP_COLUMNS = [
    "game_id", "season", "week", "gameday", "home_team", "away_team",
    "home_score", "away_score", "home_rest", "away_rest", "div_game",
    "roof", "surface", "temp", "wind", "spread_line", "total_line",
]


def _import_schedules(years: list[int]) -> pd.DataFrame:
    """Thin wrapper around nfl_data_py so tests can monkeypatch just this
    one function rather than the whole nfl_data_py import machinery."""
    import nfl_data_py as nfl

    return nfl.import_schedules(years)


def _season_cache_path(season: int) -> "Path":
    return SCHEDULES_CACHE_DIR / f"{season}.parquet"


def fetch_schedules(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """One row per game across every requested season. Seasons already
    cached on disk are read from cache; anything missing (or force_refresh)
    is fetched from nfl_data_py in one batched call, then split back out to
    per-season cache files."""
    frames = []
    missing = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(season)

    if missing:
        fetched = _import_schedules(missing)[KEEP_COLUMNS].copy()
        fetched["gameday"] = pd.to_datetime(fetched["gameday"])
        for season in missing:
            season_df = fetched[fetched["season"] == season].reset_index(drop=True)
            season_df.to_parquet(_season_cache_path(season))
            frames.append(pd.read_parquet(_season_cache_path(season)))

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def default_completed_seasons(n: int = 8) -> list[int]:
    return list(range(CURRENT_SEASON - n, CURRENT_SEASON))


def load_training_data(seasons: list[int]) -> pd.DataFrame:
    """Only games with a final score — excludes future/postponed games from
    the same fetch_schedules() call."""
    df = fetch_schedules(seasons)
    return df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)


def fetch_current_season_partial() -> pd.DataFrame:
    """Completed games so far in CURRENT_SEASON, refetched every call (no
    per-season cache for the still-in-progress season, since its cache file
    would go stale after every week's games)."""
    df = fetch_schedules([CURRENT_SEASON], force_refresh=True)
    return df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)


def fetch_upcoming_games(season: int, week: int) -> pd.DataFrame:
    """Games in a given season/week that haven't been played yet."""
    df = fetch_schedules([season], force_refresh=(season == CURRENT_SEASON))
    week_df = df[df["week"] == week]
    return week_df[week_df["home_score"].isna()].reset_index(drop=True)


def fetch_week_games(season: int, week: int) -> pd.DataFrame:
    """Every game in a season/week, finished and upcoming together -- unlike
    fetch_upcoming_games, a game that has since been played still shows up
    here instead of silently vanishing from the week."""
    df = fetch_schedules([season], force_refresh=(season == CURRENT_SEASON))
    return df[df["week"] == week].reset_index(drop=True)

"""Parquet cache for the Data Hub's nflverse pulls. A finished season is
fetched once; the season being played is refetched when the copy is older
than REFRESH_HOURS. If nflverse fails, a stale copy beats nothing, and with
no copy at all the hub gets an empty frame (dashes on the site) rather than
an error. Empty or failed pulls are never cached."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

REFRESH_HOURS = 12


def cached_frame(path: Path, fetch: Callable[[], pd.DataFrame], columns: list[str], current: bool) -> pd.DataFrame:
    fresh = path.exists() and (not current or time.time() - path.stat().st_mtime < REFRESH_HOURS * 3600)
    if fresh:
        return pd.read_parquet(path)
    try:
        df = fetch()
    except Exception as exc:  # nflverse down, or the season isn't published yet
        logger.info("hub fetch failed for %s: %s", path.name, exc)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)
    if df.empty:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df

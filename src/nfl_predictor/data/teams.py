"""teams.py — team conference/division metadata, cache-or-fetch from
nfl_data_py. Static within a season (realignment is rare and always
between seasons), so this caches to a single file with no per-season
split needed."""

from __future__ import annotations

import pandas as pd

from ..config import CACHE_DIR

_CACHE_PATH = CACHE_DIR / "team_conferences.parquet"


def _import_team_desc() -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_team_desc()


def fetch_team_conferences(force_refresh: bool = False) -> pd.DataFrame:
    """team, conference ("AFC"/"NFC"), division ("AFC East", ...)."""
    if not force_refresh and _CACHE_PATH.exists():
        return pd.read_parquet(_CACHE_PATH)
    raw = _import_team_desc()
    df = raw.rename(columns={"team_abbr": "team", "team_conf": "conference", "team_division": "division"})
    df = df[["team", "conference", "division"]].drop_duplicates("team").reset_index(drop=True)
    df.to_parquet(_CACHE_PATH)
    return df

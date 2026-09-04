"""injuries.py — official weekly injury reports, cache-or-fetch from
nfl_data_py. Supplements ESPN's closer-to-kickoff scoreboard for gating
player-prop predictions on real availability."""

from __future__ import annotations

import pandas as pd

from ..config import INJURIES_CACHE_DIR

KEEP_COLUMNS = ["season", "week", "team", "gsis_id", "full_name", "position", "report_status"]
STRING_COLUMNS = ["team", "gsis_id", "full_name", "position", "report_status"]


def _import_injuries(years: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_injuries(years)


def _season_cache_path(season: int) -> "Path":
    return INJURIES_CACHE_DIR / f"{season}.parquet"


def _ensure_object_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure string columns are object dtype and convert NaN to None.

    Parquet round-trips can change string dtypes to object; this ensures
    consistency between fresh fetches and cached reads.
    """
    df = df.copy()
    # Convert all string columns to object dtype for consistency
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("object")
            # For report_status specifically, convert NaN to None
            if col == "report_status":
                mask = pd.isna(df[col])
                df.loc[mask, col] = None
    return df


def fetch_injuries(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    frames = []
    missing = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and path.exists():
            df = pd.read_parquet(path)
            # Convert NaN back to None to match original DataFrame semantics
            # (parquet converts None to NaN; we restore it for consistency)
            df = _ensure_object_dtype(df)
            frames.append(df)
        else:
            missing.append(season)

    if missing:
        fetched = _import_injuries(missing)[KEEP_COLUMNS].copy()
        for season in missing:
            season_df = fetched[fetched["season"] == season].reset_index(drop=True)
            season_df.to_parquet(_season_cache_path(season))
            # Ensure consistent dtype before appending
            season_df = _ensure_object_dtype(season_df)
            frames.append(season_df)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    result = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    # Final cleanup to ensure object dtype with None (not NaN)
    result = _ensure_object_dtype(result)
    return result


def current_status_by_player(injuries_df: pd.DataFrame, season: int, week: int) -> dict[str, str]:
    """gsis_id -> report_status for players actually flagged (Out/Doubtful/
    Questionable) in a given season/week — players with no report_status
    (the common case: healthy, no injury report entry) are omitted rather
    than included with a None value, so callers can treat "in this dict" as
    "has a real status to gate on"."""
    week_df = injuries_df[(injuries_df["season"] == season) & (injuries_df["week"] == week)]
    flagged = week_df[week_df["report_status"].notna()]
    return dict(zip(flagged["gsis_id"], flagged["report_status"]))

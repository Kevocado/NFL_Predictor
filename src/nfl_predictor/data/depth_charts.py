"""Depth charts, from nflverse, into the public snapshot.

WHY THIS EXISTS

The predicted box score needs to say who actually starts. A per-player prop list
with no lineups makes the reader assume the top-scoring name is the starter,
which is wrong often enough to matter.

WHAT THE FEED ACTUALLY LOOKS LIKE -- read this before "fixing" anything

`nflverse-data` publishes `depth_charts_{season}.parquet` under the
`depth_charts` release tag. Verified against the real 2024 and 2026 files
(37312 rows for 2024, columns: season, club_code, week, game_type, depth_team,
last_name, first_name, football_name, formation, gsis_id, jersey_number,
position, elias_id, depth_position, full_name). Four things about it are not
what you would guess, and each has already cost time:

1. `depth_team` is a **string**, not an integer. Its values are '1', '2', '3'.
   So `depth_team == 1` -- the rule written in the plan, and the rule that reads
   correctly -- matches **zero rows** and reports that a real team has no
   starters at all. It fails silently: no exception, no warning, just an empty
   starter list on a public page. Compare as a string, or normalise on load.

2. There is **no `depth_slot` column**. The plan assumed one. Depth *order*
   within a `depth_position` is the row order in the file, so a stable order
   has to come from the frame itself, not from a sort key that does not exist.

3. A player appears at more than one `depth_position` in the same week. Isaiah
   Wynn is MIA's LG1 and RG2 in week 19; Malik Washington is KR1, PR1 and WR3.
   A player is therefore a starter if **any** of their rows is depth_team '1',
   and a naive per-row flag reports the same man twice with contradictory
   labels.

4. `depth_position` is sometimes an empty string (a player with no position
   group), and `week` is a float with NaN for offseason rows. Neither is an
   error; both have to be handled without raising.

The plan also asserted as a fixture that "Jaylen appears at RB1 and WR1" for
Miami 2024 week 19. That is two different people: **Jaylen Wright** is RB at
depth_team 2 (bench) and **Jaylen Waddle** is WR at depth_team 1 (starter). The
multi-position case the test was reaching for is real, but the player named in
the plan is wrong, and a test written from that sentence would have asserted a
false thing and passed.

THE STARTER RULE

    starter  <=>  a row for that player, at that `depth_position`, with
                  depth_team == '1'

Keyed on `depth_position`, because a player can be '1' at one position and '2'
at another in the same week, and that is not a contradiction -- it is a backup
who also starts somewhere else.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEPTH_CHARTS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/"
    "depth_charts_{season}.parquet"
)

# The first string that means "this player is on the field to start". Kept as a
# constant so the comparison is visibly a string comparison and a reader does
# not "tidy" it into an int -- which yields zero starters and no error.
STARTER_DEPTH_TEAM = "1"


def _cache_path(cache_dir: Path, season: int) -> Path:
    return Path(cache_dir) / f"depth_charts_{season}.parquet"


#: How long a cached chart for the CURRENT season is trusted. The feed publishes a new snapshot several times a
#: week; a cache that never expires served the 28 Sept chart on 14 Oct.
CACHE_TTL_SECONDS = 6 * 3600


def _is_live_season(season: int) -> bool:
    today = date.today()
    return season >= today.year - (1 if today.month < 3 else 0)


def _nfl_week(snapshot: "pd.Timestamp", season: int) -> int:
    """NFL week a snapshot belongs to: weeks run Tuesday to Monday, week 1 containing the Thursday after Labor Day.
    Offseason/preseason snapshots are week 0."""
    sept1 = date(season, 9, 1)
    labor_day = sept1 + timedelta(days=(0 - sept1.weekday()) % 7)
    window_start = labor_day + timedelta(days=1)  # Tuesday before the Thursday kickoff
    return max(0, (snapshot.date() - window_start).days // 7 + 1)


def normalize_depth_chart(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    """Give the 2025+ nflverse depth-chart schema the columns the rest of this module reads.

    The new feed has no `season`/`week`/`depth_team`/`full_name`/`depth_position`: it has `dt` (snapshot
    timestamp, many per week), `team`, `player_name`, `pos_abb` and `pos_rank`. Without this every 2026 row failed the
    `season` filter, `starter_flags` returned {} and every prop row shipped `is_starter=None`.
    Only the LATEST snapshot of each week is kept, so a player demoted mid-week is not still a starter from an
    earlier snapshot. Legacy-schema frames pass through untouched.
    """
    if frame is None or frame.empty or "pos_abb" not in frame.columns or "depth_team" in frame.columns:
        return frame
    out = frame.copy()
    stamp = pd.to_datetime(out["dt"], errors="coerce", utc=True)
    out = out.assign(_stamp=stamp).dropna(subset=["_stamp"])
    out["week"] = out["_stamp"].map(lambda t: _nfl_week(t, season)).astype(float)
    out = out[out["_stamp"] == out.groupby("week")["_stamp"].transform("max")]
    out["season"] = season
    out["game_type"] = "REG"
    out["club_code"] = out["team"]
    out["full_name"] = out["player_name"]
    out["depth_position"] = out["pos_abb"]
    out["depth_team"] = out["pos_rank"].astype(int).astype(str)
    return out.drop(columns=["_stamp"]).reset_index(drop=True)


def load_depth_charts(season: int, cache_dir: Path, *, timeout: int = 60) -> pd.DataFrame | None:
    """Cache-or-fetch one season of depth charts. Never raises.

    A depth chart is an enhancement. If the download fails the props must still
    be produced, with no starter information, so the UI shows "Projected
    order" rather than losing the feature that called this.
    """
    path = _cache_path(cache_dir, season)
    stale_ok = None
    if path.exists():
        fresh = not _is_live_season(season) or (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS
        try:
            cached = pd.read_parquet(path)
            if fresh:
                return normalize_depth_chart(cached, season)
            stale_ok = cached  # expired: try the network, fall back to this if it fails
        except Exception as exc:  # noqa: BLE001 - a corrupt cache is not fatal
            logger.warning("depth charts: unreadable cache %s (%s); refetching", path, exc)
    try:
        local = _fetch_to(DEPTH_CHARTS_URL.format(season=season), path, timeout)
        return normalize_depth_chart(pd.read_parquet(local), season)
    except Exception as exc:  # noqa: BLE001 - enhancement, never fatal
        if stale_ok is not None:
            logger.warning("depth charts: refresh failed for %s (%s); using the stale cache", season, exc)
            return normalize_depth_chart(stale_ok, season)
        logger.warning("depth charts: fetch failed for %s (%s); continuing without starters", season, exc)
        return None


def _fetch_to(url: str, dest: Path, timeout: int) -> str:
    """Download `url` to `dest` and return `dest`. Raises on failure."""
    import requests  # imported lazily: a cache hit must not need the network stack

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return str(dest)


def resolve_chart(frame: pd.DataFrame, season: int, week: int, game_type: str = "REG") -> pd.DataFrame:
    """The most recent published chart at or before `week` for that season.

    Never returns a chart dated after `week`: that would leak post-kickoff
    information. Empty means "nothing published yet this season".
    """
    season_rows = frame[frame["season"] == season]
    if season_rows.empty:
        return season_rows

    typed = pd.to_numeric(season_rows["week"], errors="coerce")
    eligible = season_rows[typed <= week]
    if eligible.empty:
        # Every chart this season is dated AFTER the game. Using one would feed the
        # game information from its own future, so answer "no chart" instead; the
        # caller may fall back to the previous season's final chart.
        logger.warning("depth charts: no chart at or before week %s in %s", week, season)
        return season_rows.iloc[0:0]

    if game_type and "game_type" in eligible.columns:
        same_type = eligible[eligible["game_type"] == game_type]
        if not same_type.empty:
            eligible = same_type

    latest = pd.to_numeric(eligible["week"], errors="coerce").max()
    return eligible[pd.to_numeric(eligible["week"], errors="coerce") == latest]


def starter_flags(chart: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """One entry per player in the chart: their position, slot and starter flag.

    Keyed by `full_name` because that is what the props carry. A player listed
    at two positions is a starter if EITHER row is depth_team '1', and takes the
    position where they are '1' -- so Isaiah Wynn reports as a starting guard
    rather than as a contradictory LG1/RG2.
    """
    if chart is None or chart.empty:
        return {}

    starters: dict[str, bool] = {}
    positions: dict[str, str] = {}
    starter_position_taken: dict[str, bool] = {}
    order: dict[str, int] = {}
    seen_counter = 0

    for _, row in chart.iterrows():
        name = row.get("full_name")
        if not isinstance(name, str) or not name:
            continue
        position = row.get("depth_position")
        position = position.strip() if isinstance(position, str) else ""
        # The string comparison. An int here reports no starters at all.
        is_starter = str(row.get("depth_team")) == STARTER_DEPTH_TEAM

        if name not in order:
            order[name] = seen_counter
            seen_counter += 1
            starters.setdefault(name, False)
            positions.setdefault(name, position)

        if is_starter:
            starters[name] = True
            # The FIRST position at which they start wins, and the choice is
            # arbitrary -- Malik Washington is KR1, PR1 and WR3, and any one of
            # those is defensible. First-in-file-order is used because it is
            # deterministic: the same feed always produces the same grouping, so
            # the box score does not reshuffle between reloads. A plain
            # assignment here would let a later starter row silently overwrite
            # the first, which is how the position ended up as "PR".
            if not starter_position_taken.get(name):
                positions[name] = position or positions.get(name, "")
                starter_position_taken[name] = True

    flags = {
        name: {
            "is_starter": bool(starters[name]),
            "position": positions.get(name, ""),
            "depth_slot": order[name],
        }
        for name in order
    }
    # The props carry abbreviated names ("S.Darnold") but the feed carries full ones ("Sam Darnold"), so a name join
    # silently misses. A gsis id ("00-0034869") is what both sides share; id keys cannot collide with names.
    if "gsis_id" in chart.columns:
        for name, gsis in chart.drop_duplicates("full_name")[["full_name", "gsis_id"]].itertuples(index=False):
            if isinstance(name, str) and name in flags and isinstance(gsis, str) and gsis:
                flags.setdefault(gsis, flags[name])
    return flags


def flags_for_season_week(season: int, week: int, cache_dir: Path, *, game_type: str = "REG") -> dict[str, dict[str, Any]]:
    """Starter flags for every player published for that season and week.

    Returns `{}` when the feed is unavailable. Callers must treat an empty map
    as "no depth-chart data", not "nobody starts".
    """
    frame = load_depth_charts(season, cache_dir)
    if frame is None or frame.empty:
        return {}
    chart = resolve_chart(frame, season, week, game_type)
    if chart is None or chart.empty:
        return {}
    return starter_flags(chart)

"""player_stats.py — weekly player-level stats, cache-or-fetch from nflverse.
Same per-season parquet caching pattern as data/schedules.py.

Two upstream releases carry this table and they are not interchangeable:

* `player_stats/player_stats_{season}.parquet` -- what `nfl_data_py` reads, and
  hardcodes its URL for. **Published 1999-2024 only**; 2025 and 2026 are 404.
* `stats_player/stats_player_week_{season}.parquet` -- the same table under a
  different release tag, published continuously through the current season.

Why `player_stats` stays primary and `stats_player` is only a fallback
------------------------------------------------------------------
`player_stats` is tried first for every season and used whenever it answers.
That is not inertia; switching history over to `stats_player` was measured and
it moves the models' inputs:

* The two agree on 99.93% of the 124,564 offensive player-weeks in 2001-2024.
* But `stats_player` is a denser file. It carries a row for every player who
  appeared in a game, including ~13,121 zero-stat player-weeks that
  `player_stats` omits, and every defensive player and lineman as well.
  `build_features_for_player` takes `history.tail(5)`, so those extra zero weeks
  land in the rolling window. Recomputing the exact serving features over
  127,102 offensive player-weeks, `targets_roll` moves on 26.3% of them and
  `receiving_yards_roll` on 10.7% (median absolute shift 0.00, p95 3.5 targets
  / 4.05 yards).
* The committed models were trained on `player_stats`. Re-reading history from
  `stats_player` would feed them a different distribution than they were fitted
  on. Confining the fallback to seasons `player_stats` has stopped publishing
  keeps all 26 seasons of training and cached history byte-identical, and
  limits the shift to the current season's own weeks.

Where they disagree and an independent third source can adjudicate, the new
file is the correct one: against nflverse weekly play-by-play, over 1999-2000
the `player_stats` values match on 0 of 735 disagreeing player-weeks and the
`stats_player` values match on 735 of 735. In 2001-2024 it is a 12-to-11 split
and the two are equivalent. So the fallback is a correction for the oldest
seasons and neutral for the recent ones -- it is never the worse source.

One defect is carried over, deliberately and disclosed here: `stats_player`
nets a player's *own* fumbles out of every yardage column rather than only the
matching one. T. Lawrence, 2024 week 4, lost a fumble on a sack and the file
books it as **-5 receiving yards**. `player_stats` floors those at 0. It
affects 12 receiving_yards and 1 rushing_yards player-weeks in 24 seasons
(2001-2024), all of them quarterbacks or a wide receiver with no receiving
plays that week, so any positive prop line grades the same either way. It is
not corrected here because there is no second opinion to correct it with; it is
recorded because it is real. See docs/player-prop-accuracy-blocker.md.

The negative yardages themselves are not a defect of the new file and are not
guarded against. They are a long-standing nflverse convention -- official yard
totals are net of own fumbles -- and `player_stats` has them too, at a larger
count: over 2001-2024 the old file has 1,235 negative `receiving_yards` rows to
the new file's 1,247, and 4,104 negative `rushing_yards` rows to 4,405. A
clamp added here would have silently changed long-settled grading behaviour for
every historical season to work around a twelve-row difference.

A note on `game_id`: `stats_player` carries a native `game_id`, and it is
deliberately dropped rather than kept. `api/routes._attach_game_id` derives
`game_id` by joining the stats frame on `["season", "week", "recent_team"]`
against the schedule, so a frame that already had `game_id` would come out of
that merge with `game_id_x`/`game_id_y` and `reconcile_player_prop_predictions`
would then fail on its `["game_id", "player_id"]` join. `team` is renamed to
`recent_team` for the same reason -- that join needs it.
"""

from __future__ import annotations

import logging

import pandas as pd
from pathlib import Path

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


def _source_marker_path(season: int) -> "Path":
    """Sidecar recording which release wrote `{season}.parquet`.

    Two releases can now write the same cache path, so a bare
    `{season}.parquet` cannot say which values it holds. That matters
    concretely: `stats_player` carries three times the rows and moves the
    rolling features (see the module docstring), so a cache dir that quietly
    held both would be serving one season's history on the other's semantics
    with nothing in the frame to reveal it. The sidecar is one small file per
    season and makes a mixed cache dir *readable* rather than something to
    reason about in a comment.

    Deliberately NOT a rename of the cache path. `{season}.parquet` is where
    seven already-populated caches (2018-2024) sit, and those seven are
    coherent as they stand -- they were all written by `player_stats`, which is
    still the source this rule picks for every one of them.

    What the marker deliberately does NOT do: trigger a re-fetch on mismatch.
    Detecting that `player_stats` has started publishing a season again needs an
    HTTP probe, and a marker read cannot supply one -- so a mismatch rule would
    re-probe on every single call and defeat the cache entirely. (It did, in
    the first draft of this: a fallback-written cache was re-downloaded every
    tick.) `force_refresh=True` is the documented way to re-settle a season, and
    it is what `api/routes._load_player_history` already passes for the live
    season.
    """
    return PLAYER_STATS_CACHE_DIR / f"{season}.source"


def _read_source_marker(season: int) -> "str | None":
    """Which release wrote this season's cache, or None if it predates markers."""
    try:
        return _source_marker_path(season).read_text().strip()
    except OSError:
        return None


def _write_source_marker(season: int, source_id: str) -> None:
    try:
        PLAYER_STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _source_marker_path(season).write_text(f"{source_id}\n")
    except OSError as err:  # a read-only cache dir must not fail the fetch
        logger.info("Could not record player-stats source for season=%s: %s", season, err)


def _weekly_data_fallback(season: int) -> pd.DataFrame:
    """Fetch one season from `stats_player`, the release that still publishes.

    `team` becomes `recent_team` and the file's own `game_id` is dropped; see
    the module docstring for why both are necessary rather than cosmetic.

    Reads the parquet with `pd.read_parquet` straight off the release URL, the
    same call `nfl_data_py.import_weekly_data` makes for the old release -- a
    404 there raises out of pandas, which is the failure this whole path is
    built around.
    """
    url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"stats_player/stats_player_week_{season}.parquet"
    )
    columns = [c for c in KEEP_COLUMNS if c != "recent_team"] + ["team"]
    raw = pd.read_parquet(url, columns=columns, engine="auto")
    return raw.rename(columns={"team": "recent_team"})[KEEP_COLUMNS].copy()


def _fetch_season(season: int) -> "tuple[pd.DataFrame, str]":
    """Return `(frame, source_id)` for one season, falling back on a 404.

    Every failure mode -- `player_stats` absent, `stats_player` absent, both
    absent, or either download failing -- surfaces as an exception to the
    caller, which is where the existing per-season `logger.info` and the
    empty-frame fallthrough already live.
    """
    try:
        return _import_weekly_data([season], columns=KEEP_COLUMNS)[KEEP_COLUMNS].copy(), "player_stats"
    except Exception:
        logger.info(
            "player_stats has not published season=%s; falling back to stats_player", season
        )
    return _weekly_data_fallback(season), "stats_player"


def fetch_weekly_player_stats(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    frames = []
    missing = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and path.exists():
            if _read_source_marker(season) is None:
                # A cache from before the marker existed, so it predates the
                # fallback: every one of this repo's seven was written by
                # `player_stats`, which is still what this rule picks for each
                # of those seasons. Claim it rather than re-downloading 26
                # seasons to establish what the sidecar would have said.
                _write_source_marker(season, "player_stats")
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
                fetched, source_id = _fetch_season(season)
            except Exception:
                logger.info("No weekly player stats available yet for season=%s", season)
                continue
            season_df = fetched[fetched["season"] == season].reset_index(drop=True)
            PLAYER_STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            season_df.to_parquet(_season_cache_path(season))
            _write_source_marker(season, source_id)
            frames.append(pd.read_parquet(_season_cache_path(season)))

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week"]).reset_index(drop=True)

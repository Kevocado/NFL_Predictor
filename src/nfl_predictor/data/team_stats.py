"""team_stats.py — per-team, per-game yardage, from nflverse's `stats_team` release.

Why this module exists. The dashboard labels a number "Team Yardage Predictions".
That number is the sum of every active player's position-specific yardage
projection across the whole roster (`Sports_Predictor/src/lib/playerRank.ts`
`keyYardage`). Measured 2026-09-27 across the 16 live week-3 games, all 32
team totals fell outside the realistic 300-450 band: median 1110, max 1452.
Two errors compound — it adds four quarterbacks' passing projections as if all
four started, and `POSITION_MARKETS` gives QB no `rushing_yards` and RB no
`receiving_yards`, so real rushing/receiving yardage is unrecoverable from it.

There was no yardage model to aggregate because **no yardage data existed**.
`nfl.import_schedules()` returns 46 columns and none of them are yardage; this
project's cached schedule parquets have 17 columns and none of them are yardage
either. The only yardage source previously reachable was aggregating
play-by-play — 27 seasons at ~20 MB and 372 columns each.

This module fetches the `stats_team` release instead: 138 columns, one row per
team per game, ~0.2 MB per season, no API key. The whole 1999-2025 history is a
few MB against roughly 540 MB of play-by-play. **No `nfl_data_py` import
function wraps this release tag**, so the URL is read directly.

The target is `passing_yards + rushing_yards` -- net offensive yards per team
per game, `passing_yards` already being net of `sack_yards_lost`.

Reconciliation constraint. Because the team total and the player totals come
from the same match, `sum(players.passing_yards) + sum(players.rushing_yards)`
equals the team figure **exactly**. Verified on the 2024 regular season: 544 of
544 team-games matched with a maximum absolute difference of 0.0 yards. That
makes the correct architecture project the team total first and *allocate* it
by share; summing independent player projections cannot reproduce it, which is
precisely what the dashboard does today.
"""

from __future__ import annotations

import pandas as pd

from ..config import TEAM_STATS_CACHE_DIR

# nflverse-data release assets. `stats_team` is a release tag with no
# `nfl_data_py` wrapper, so the asset URL is built here rather than through the
# library. Keyless and public.
_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/stats_team"

# The identity columns plus the yardage/volume block worth modelling. The full
# file has 138 columns spanning offense, defense, kicking, punting and penalties;
# these are the ones a team-offence model would read first. Names verified
# against the real 2024 asset -- note there is no `turnovers` or `fumbles_lost`
# column, only the per-phase `*_fumbles_lost` trio and an `fumbles_lost_total`,
# and no turnover column at all. `_download` rejects an unknown column loudly so
# this list cannot rot into a silent feature drop.
KEEP_COLUMNS = [
    "game_id", "season", "week", "team", "opponent_team", "season_type",
    # yardage
    "passing_yards", "rushing_yards", "receiving_yards",
    "sack_yards_lost", "passing_air_yards", "passing_yards_after_catch",
    "receiving_air_yards", "receiving_yards_after_catch",
    # volume
    "attempts", "completions", "carries", "receptions", "targets", "sacks_suffered",
    # scoring / efficiency
    "passing_tds", "rushing_tds", "receiving_tds",
    "passing_first_downs", "rushing_first_downs", "receiving_first_downs",
    "passing_interceptions", "fumbles_lost_total",
    # advanced
    "passing_epa", "rushing_epa", "receiving_epa", "passing_cpoe",
]

# The identity the whole architecture rests on. Exposed as a function so both
# the test suite and any future reconciliation job call the same definition.
TARGET_COLUMNS = ["passing_yards", "rushing_yards"]


def _season_cache_path(season: int):
    return TEAM_STATS_CACHE_DIR / f"{season}.parquet"


def _download(season: int) -> pd.DataFrame:
    url = f"{_RELEASE}/stats_team_week_{season}.parquet"
    raw = pd.read_parquet(url)
    missing = [column for column in KEEP_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(
            f"nflverse stats_team_week_{season}.parquet is missing expected columns: {missing}. "
            "The upstream schema changed; update KEEP_COLUMNS deliberately rather than "
            "silently dropping them."
        )
    return raw[KEEP_COLUMNS].copy()


def fetch_team_stats(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """One row per team per game across the requested seasons.

    Per-season cache-or-fetch, matching `data/schedules.py`. A played season's
    team yardage never changes, so caching per season is safe.
    """
    TEAM_STATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = _season_cache_path(season)
        if path.exists() and not force_refresh:
            frames.append(pd.read_parquet(path))
            continue
        frame = _download(season)
        frame.to_parquet(path, index=False)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["season", "week", "team"]).reset_index(drop=True)


def add_total_yards(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `total_yards` = net offensive yards, the model target.

    Split out from `fetch_team_stats` so any frame carrying the two columns can
    be scored without a round trip through the cache.
    """
    out = frame.copy()
    out["total_yards"] = out[TARGET_COLUMNS].sum(axis=1, min_count=len(TARGET_COLUMNS))
    return out


def fetch_total_yards(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """`fetch_team_stats` with the `total_yards` target attached."""
    return add_total_yards(fetch_team_stats(seasons, force_refresh=force_refresh))


def reconcile_against_players(
    team_frame: pd.DataFrame, player_frame: pd.DataFrame
) -> pd.DataFrame:
    """Per (season, week, team): team total, the player sum, and the difference.

    This is the check that makes an allocate-by-share architecture sound. A
    roster sum that misses by 3x is not a noisy estimate, it is a category
    error, and this function is how that would be caught rather than shipped.

    `player_frame` needs `season`, `week`, `team`, and the same TARGET_COLUMNS
    (nflverse's `stats_player` release carries per-player passing/rushing
    yards). Rows are summed per team-game before comparison.
    """
    keys = ["season", "week", "team"]
    # Both `min_count`s are load-bearing and only one of them was present.
    #
    # `min_count` on a *groupby* sum is per column -- it requires N non-null
    # observations of that one column within the group, not N populated
    # components -- so putting it here NaN'd out any team-game with a single row
    # for a column. Hence the axis-level `min_count` below, which is the right
    # instrument.
    #
    # But pandas' `groupby.sum()` also treats an **all-NaN column as 0.0**. So a
    # team-game where no player row carried a `rushing_yards` value came back with
    # `player_rushing_yards = 0` -- a real zero -- and the axis sum then saw two
    # populated numbers and never objected. The guard the comment above describes
    # therefore did not exist, and the failure it would have caught is the exact
    # "missing value becomes a plausible number" shape: a partial player total read
    # as a complete one, understating the diff by the whole team rushing total.
    #
    # `min_count=1` on the groupby stops an all-NaN component collapsing to zero;
    # the axis `min_count` then requires both components to be genuinely present.
    # Verified: groupby.sum() -> 0.0, axis sum -> 250.0 (guard inert);
    # groupby.sum(min_count=1) -> NaN, axis sum -> NaN.
    player_totals = (
        player_frame.groupby(keys, as_index=False)[TARGET_COLUMNS]
        .sum(min_count=1)
        .rename(columns={column: f"player_{column}" for column in TARGET_COLUMNS})
    )
    summed = [f"player_{column}" for column in TARGET_COLUMNS]
    player_totals["player_total_yards"] = player_totals[summed].sum(axis=1, min_count=len(summed))
    team_totals = add_total_yards(team_frame)[keys + ["total_yards"]]
    merged = team_totals.merge(player_totals[keys + ["player_total_yards"]], on=keys, how="inner")
    merged["diff"] = merged["total_yards"] - merged["player_total_yards"]
    return merged

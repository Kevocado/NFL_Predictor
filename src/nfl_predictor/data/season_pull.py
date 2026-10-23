"""season_pull.py -- scripted, cached pulls of free nflverse season data.

Every pull writes `{name}_{season}.parquet` and is a no-op when that file
already exists, so a rerun is deterministic and costs no network.

`IMPORTERS` is a mapping from cache name to the `nfl_data_py` function name, kept
as data rather than as four near-identical wrappers so a version difference is a
one-line edit. `rosters` maps to `import_depth_charts`: the installed
`nfl_data_py` (0.3.3) has no `import_rosters`, and depth charts are what the
availability features actually read (`depth_chart_position`).
"""
from __future__ import annotations

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

#: NGS is published per stat type, and `import_ngs_data` requires one.
NGS_STAT_TYPES: tuple[str, ...] = ("passing", "rushing", "receiving")

IMPORTERS: dict[str, str] = {
    "weekly": "import_weekly_data",
    "pbp": "import_pbp_data",
    "injuries": "import_injuries",
    "rosters": "import_depth_charts",
    "schedules": "import_schedules",
}


def _cached_pull(name: str, seasons: list[int], cache_dir: Path, importer) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = cache_dir / f"{name}_{season}.parquet"
        if not path.exists():
            importer([season]).to_parquet(path, index=False)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def pull_weekly(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    return _cached_pull("weekly", seasons, Path(cache_dir), nfl.import_weekly_data)


def pull_pbp(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    return _cached_pull("pbp", seasons, Path(cache_dir), nfl.import_pbp_data)


def pull_injuries(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    return _cached_pull("injuries", seasons, Path(cache_dir), nfl.import_injuries)


#: Slot labels in the 2025+ depth-chart schema (`pos_abb`) -> the roster
#: `position` vocabulary of 2017-2024. Return roles carry no position.
_SLOT_TO_POSITION = {
    "LT": "T", "RT": "T", "LG": "G", "RG": "G",
    "LDE": "DE", "RDE": "DE", "LDT": "DT", "RDT": "DT",
    "LCB": "CB", "RCB": "CB", "NB": "CB",
    "LILB": "LB", "RILB": "LB", "MLB": "LB", "SLB": "LB", "WLB": "LB",
    "PK": "K", "KR": None, "PR": None,
}


def normalize_depth_chart(df: pd.DataFrame) -> pd.DataFrame:
    """Give 2025+ depth charts the old schema's `position` and `season`.

    nflverse replaced the depth-chart schema for 2025: no `position`/`season`/
    `week`, instead `pos_abb` (slot) and `dt` (snapshot timestamp). Without this
    every 2025/2026 row had a null position and was dropped by every
    position-filtered model. `week`/`depth_position` stay NaN (not derivable).
    """
    if "pos_abb" not in df.columns:
        return df
    df = df.copy()
    if "position" not in df.columns:
        df["position"] = pd.NA
    slot = df["pos_abb"].map(lambda a: _SLOT_TO_POSITION.get(a, a))
    df["position"] = df["position"].where(df["position"].notna(), slot)
    season = pd.to_numeric(df["season"], errors="coerce") if "season" in df.columns \
        else pd.Series(float("nan"), index=df.index)
    if "dt" in df.columns:
        ts = pd.to_datetime(df["dt"], errors="coerce", utc=True)
        # Aug-Mar snapshots belong to the season that began in August.
        season = season.fillna(ts.dt.year - (ts.dt.month < 6))
    df["season"] = season
    # The 2017-2024 charts carry game_type and fill_positions keeps only REG rows when the column exists. Concatenated
    # with an older season, new-schema rows would arrive with NaN there and be silently filtered out - which left
    # every 2025/2026 weekly row without a position even after the schema fix.
    # pull_rosters concatenates seasons BEFORE normalising, so the column usually exists (from the old-schema
    # seasons) and is NaN on exactly the new-schema rows.
    df["game_type"] = df["game_type"].fillna("REG") if "game_type" in df.columns else "REG"
    return df


def pull_rosters(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    return normalize_depth_chart(
        _cached_pull("rosters", seasons, Path(cache_dir), nfl.import_depth_charts))


def pull_ngs(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    """Next Gen Stats.

    `import_ngs_data(stat_type, years)` takes the stat type *first* and requires
    one -- calling it as `import_ngs_data([season])` silently becomes
    stat_type=[2024]. All three stat types are cached into one file per season.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = cache_dir / f"ngs_{season}.parquet"
        if not path.exists():
            frames_for_season = [nfl.import_ngs_data(stat, [season]) for stat in NGS_STAT_TYPES]
            pd.concat(frames_for_season, ignore_index=True).to_parquet(path, index=False)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


#: Per-role pbp columns: (player id column, player name column).
_ROLES = (
    ("passer_player_id", "passer_player_name"),
    ("rusher_player_id", "rusher_player_name"),
    ("receiver_player_id", "receiver_player_name"),
)

#: Output stat -> pbp column, split because the two aggregate differently:
#: yardage sums, flags count.
_YARDAGE_SOURCES = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}
_FLAG_SOURCES = {
    "carries": "rush_attempt",
    "receptions": "complete_pass",
    "targets": "pass_attempt",
    # Passing volume for the passer. `pass_attempt` is shared with `targets`,
    # but role ownership keeps them apart: a pass attempt is a target for the
    # receiver and an attempt for the passer, never both for one player.
    "attempts": "pass_attempt",
    "completions": "complete_pass",
    # Touchdowns by role. `pass_touchdown` credits both the passer and the
    # receiver on a passing TD (role ownership keeps them apart, same as
    # attempts/targets above); `rush_touchdown` credits the rusher.
    "passing_tds": "pass_touchdown",
    "rushing_tds": "rush_touchdown",
    "receiving_tds": "pass_touchdown",
}

#: Which stats each role owns. A completed pass is one row for the passer *and*
#: one for the receiver, so without this a QB would be credited with his
#: receiver's yardage and a WR with his quarterback's attempts.
_ROLE_STATS = {
    "passer_player_id": ("passing_yards", "attempts", "completions", "passing_tds"),
    "rusher_player_id": ("rushing_yards", "carries", "rushing_tds"),
    "receiver_player_id": ("receiving_yards", "receptions", "targets", "receiving_tds"),
}

_ALL_STATS = tuple(_YARDAGE_SOURCES) + tuple(_FLAG_SOURCES)

WEEKLY_COLUMNS = ["player_id", "player_name", "position", "season", "week",
                  "recent_team", "opponent_team", "passing_yards", "rushing_yards",
                  "receiving_yards", "targets", "carries", "receptions",
                  "attempts", "completions",
                  "passing_tds", "rushing_tds", "receiving_tds"]


def _role_frame(games: pd.DataFrame, id_column: str, name_column: str) -> pd.DataFrame:
    """One row per (player, week) appearance in this role, with only this role's
    stats populated and every other stat zeroed."""
    sub = games[games[id_column].notna()].reset_index(drop=True)
    owned = _ROLE_STATS[id_column]
    frame = pd.DataFrame({
        "player_id": sub[id_column].to_numpy(),
        "player_name": sub[name_column].to_numpy(),
        "season": sub["season"].to_numpy(),
        "week": sub["week"].to_numpy(),
        "posteam": sub["posteam"].to_numpy(),
        "defteam": sub["defteam"].to_numpy(),
    })
    for stat, source in _YARDAGE_SOURCES.items():
        frame[stat] = sub[source].fillna(0) if stat in owned else 0
    for stat, source in _FLAG_SOURCES.items():
        # A season missing the source column (or a synthetic test frame) reads
        # the stat as 0 rather than raising -- same discipline as the
        # absent-column handling in the quantile script's build_feature_frame.
        if stat in owned and source in sub.columns:
            frame[stat] = sub[source].fillna(False).astype(bool).astype(int)
        else:
            frame[stat] = 0
    return frame


def weekly_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Derive weekly player yardage stats from nflverse play-by-play.

    Needed because nflverse's `player_stats` release ends at 2024 while its pbp
    release runs to the current season. Measured against the official 2024
    weekly stats: passing yards and receptions match exactly, rushing and
    receiving yards agree on >=99.6% of player-weeks with correlation >0.999
    (the residual is laterals, which nflverse folds into the official totals).
    Touchdowns (passing/rushing/receiving, from `pass_touchdown`/`rush_touchdown`)
    match the official series on 100%/100%/99.9% of player-weeks.

    `position` is left as NA: pbp carries no roster, and the training script
    fills it from the depth-chart cache.
    """
    if pbp.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)

    games = pbp[pbp["season_type"].astype(str).str.upper() == "REG"]
    keys = ["player_id", "season", "week"]

    numeric_parts, identity_parts = [], []
    for id_column, name_column in _ROLES:
        frame = _role_frame(games, id_column, name_column)
        if frame.empty:
            continue
        numeric_parts.append(frame.groupby(keys, as_index=False)[list(_ALL_STATS)].sum())
        identity_parts.append(frame[[*keys, "player_name", "posteam", "defteam"]])

    if not numeric_parts:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)

    # Identity comes from every role, not just one: a running back never appears
    # in the passer frame, so taking it from there leaves his team blank.
    identity = (pd.concat(identity_parts, ignore_index=True).drop_duplicates(subset=keys)
                .rename(columns={"posteam": "recent_team", "defteam": "opponent_team"}))

    weekly = pd.concat(numeric_parts, ignore_index=True).groupby(keys, as_index=False)[list(_ALL_STATS)].sum()
    weekly = weekly.merge(identity[[*keys, "player_name", "recent_team", "opponent_team"]],
                          on=keys, how="left")
    weekly["position"] = pd.NA
    return weekly.sort_values(keys)[WEEKLY_COLUMNS].reset_index(drop=True)


def fill_positions(weekly: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Attach a position to weekly rows that have none.

    pbp carries no roster, so derived weeks arrive with `position` empty. The
    depth-chart cache does carry one. Matched on (player, season) rather than
    week, because a player's position does not change week to week and a
    week-level match would miss anyone who missed a depth chart.
    """
    if weekly.empty or rosters.empty or "position" not in rosters.columns:
        return weekly

    regular = rosters
    if "game_type" in rosters.columns:
        regular = rosters[rosters["game_type"].astype(str).str.upper() == "REG"]
    positions = (regular.dropna(subset=["position"])
                 .groupby(["gsis_id", "season"], as_index=False)["position"]
                 .agg(lambda s: s.value_counts().index[0]))
    merged = weekly.merge(positions.rename(columns={"gsis_id": "player_id"}),
                          on=["player_id", "season"], how="left",
                          suffixes=("", "_roster"))
    if "position_roster" in merged.columns:
        filled = merged["position"].notna() & (merged["position"] != "")
        merged.loc[~filled, "position"] = merged.loc[~filled, "position_roster"]
        merged = merged.drop(columns=["position_roster"])
    return merged


def pull_schedules(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    return _cached_pull("schedules", seasons, Path(cache_dir), nfl.import_schedules)
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


def pull_rosters(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    return _cached_pull("rosters", seasons, Path(cache_dir), nfl.import_depth_charts)


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


def pull_schedules(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    return _cached_pull("schedules", seasons, Path(cache_dir), nfl.import_schedules)
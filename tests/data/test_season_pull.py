"""season_pull tests.

These stub `nfl_data_py` at the seam the code calls. They do NOT hit the
network: `tests/conftest.py` blocks connects for the whole suite, and the
plan's literal test (a real `pull_weekly([2024])`) would trip that guard. One
`@pytest.mark.network` test at the bottom checks the live importer's real
column names, which is the part a stub cannot tell us.
"""
from __future__ import annotations

import pandas as pd
import pytest

# The nflverse weekly columns this pipeline actually depends on. `recent_team`,
# not `team` -- verified against the live importer in the network test below.
WEEKLY_COLUMNS = [
    "player_id", "player_name", "position", "season", "week",
    "passing_yards", "rushing_yards", "receiving_yards",
    "targets", "carries", "receptions", "recent_team", "opponent_team",
]


def _fake_weekly(seasons: list[int]) -> pd.DataFrame:
    return pd.DataFrame([{
        "player_id": "00-0023459", "player_name": "A.Rodgers", "position": "QB",
        "season": season, "week": 1, "passing_yards": 100, "rushing_yards": 0,
        "receiving_yards": 0, "targets": 0, "carries": 0, "receptions": 0,
        "recent_team": "NYJ", "opponent_team": "SF",
    } for season in seasons])


def test_pull_weekly_writes_parquet_cache(tmp_path, monkeypatch):
    import nfl_data_py as nfl
    from nfl_predictor.data import season_pull

    calls: list[tuple[int, ...]] = []

    def fake(seasons):
        calls.append(tuple(seasons))
        return _fake_weekly(seasons)

    monkeypatch.setattr(nfl, "import_weekly_data", fake)

    df = season_pull.pull_weekly([2024], cache_dir=tmp_path)

    assert (tmp_path / "weekly_2024.parquet").exists()
    assert set(WEEKLY_COLUMNS).issubset(df.columns)
    assert calls == [(2024,)]

    # Second call hits the cache: same rows, and the importer is NOT called again.
    df2 = season_pull.pull_weekly([2024], cache_dir=tmp_path)
    assert len(df2) == len(df)
    assert calls == [(2024,)], "cached season was re-downloaded"


def test_pull_weekly_concatenates_multiple_seasons(tmp_path, monkeypatch):
    import nfl_data_py as nfl
    from nfl_predictor.data import season_pull

    monkeypatch.setattr(nfl, "import_weekly_data", _fake_weekly)

    df = season_pull.pull_weekly([2023, 2024], cache_dir=tmp_path)

    assert (tmp_path / "weekly_2023.parquet").exists()
    assert (tmp_path / "weekly_2024.parquet").exists()
    assert sorted(df["season"].unique()) == [2023, 2024]


@pytest.mark.parametrize("fn,name", [
    ("pull_pbp", "pbp"), ("pull_injuries", "injuries"),
    ("pull_rosters", "rosters"),
])
def test_pull_aux_writes_cache(tmp_path, monkeypatch, fn, name):
    import nfl_data_py as nfl
    from nfl_predictor.data import season_pull

    def fake(seasons):
        return pd.DataFrame([{"season": s, "week": 1, "value": 1} for s in seasons])

    monkeypatch.setattr(nfl, season_pull.IMPORTERS[name], fake)

    df = getattr(season_pull, fn)([2024], cache_dir=tmp_path)

    assert (tmp_path / f"{name}_2024.parquet").exists()
    assert len(df) > 0


def test_pull_ngs_requests_every_stat_type(tmp_path, monkeypatch):
    """`import_ngs_data(stat_type, years)` takes the type FIRST and requires one.
    Calling it as import_ngs_data([season]) turns the season list into a stat
    type, which is a ValueError at best and the wrong dataset at worst."""
    import nfl_data_py as nfl
    from nfl_predictor.data import season_pull

    seen: list[tuple] = []

    def fake(*args, **kwargs):
        seen.append((args, kwargs))
        stat = args[0] if args else kwargs.get("stat_type")
        return pd.DataFrame([{"stat_type": stat, "season": 2024, "week": 1}])

    monkeypatch.setattr(nfl, "import_ngs_data", fake)

    df = season_pull.pull_ngs([2024], cache_dir=tmp_path)

    assert (tmp_path / "ngs_2024.parquet").exists()
    assert sorted(df["stat_type"]) == sorted(season_pull.NGS_STAT_TYPES)
    assert len(seen) == len(season_pull.NGS_STAT_TYPES)


def test_pull_schedules_writes_cache(tmp_path, monkeypatch):
    import nfl_data_py as nfl
    from nfl_predictor.data import season_pull

    def fake(seasons):
        return pd.DataFrame([{"season": s, "week": 1, "total_line": 45.0} for s in seasons])

    monkeypatch.setattr(nfl, "import_schedules", fake)

    df = season_pull.pull_schedules([2024], cache_dir=tmp_path)

    assert (tmp_path / "schedules_2024.parquet").exists()
    assert len(df) > 0


@pytest.mark.network
def test_live_weekly_has_the_columns_we_depend_on(tmp_path):
    """The stub above cannot prove the real importer's column names."""
    from nfl_predictor.data import season_pull

    df = season_pull.pull_weekly([2024], cache_dir=tmp_path)

    missing = set(WEEKLY_COLUMNS) - set(df.columns)
    assert not missing, f"nflverse weekly is missing {sorted(missing)}"


# --- fill_positions ---------------------------------------------------------

GSIS = "00-0000001"


def test_fill_positions_backfills_from_the_depth_chart_cache():
    from nfl_predictor.data.season_pull import fill_positions

    weekly = pd.DataFrame([{"player_id": GSIS, "season": 2025, "week": 1,
                            "position": pd.NA, "receiving_yards": 40}])
    rosters = pd.DataFrame([{"gsis_id": GSIS, "season": 2025, "week": 1.0,
                             "position": "WR", "game_type": "REG",
                             "depth_position": 1, "club_code": "A"}])

    out = fill_positions(weekly, rosters)

    assert out.iloc[0]["position"] == "WR"


def test_fill_positions_does_not_overwrite_a_known_position():
    from nfl_predictor.data.season_pull import fill_positions

    weekly = pd.DataFrame([{"player_id": GSIS, "season": 2025, "week": 1,
                            "position": "TE", "receiving_yards": 40}])
    rosters = pd.DataFrame([{"gsis_id": GSIS, "season": 2025, "week": 1.0,
                             "position": "WR", "game_type": "REG",
                             "depth_position": 1, "club_code": "A"}])

    assert fill_positions(weekly, rosters).iloc[0]["position"] == "TE"


def test_fill_positions_takes_the_most_common_depth_chart_position():
    from nfl_predictor.data.season_pull import fill_positions

    weekly = pd.DataFrame([{"player_id": GSIS, "season": 2025, "week": 3,
                            "position": pd.NA, "receiving_yards": 40}])
    # Listed as aWR more often than aTE across the season.
    rosters = pd.DataFrame([
        {"gsis_id": GSIS, "season": 2025, "week": w, "position": pos,
         "game_type": "REG", "depth_position": 1, "club_code": "A"}
        for w, pos in [(1, "WR"), (2, "WR"), (3, "WR"), (4, "TE")]
    ])

    assert fill_positions(weekly, rosters).iloc[0]["position"] == "WR"


def test_fill_positions_leaves_unknown_players_empty():
    from nfl_predictor.data.season_pull import fill_positions

    weekly = pd.DataFrame([{"player_id": "00-9999999", "season": 2025, "week": 1,
                            "position": pd.NA, "receiving_yards": 40}])
    rosters = pd.DataFrame([{"gsis_id": GSIS, "season": 2025, "week": 1.0,
                             "position": "WR", "game_type": "REG",
                             "depth_position": 1, "club_code": "A"}])

    out = fill_positions(weekly, rosters)

    assert pd.isna(out.iloc[0]["position"])
    assert len(out) == 1
"""Tests for the team-week yardage source and the reconciliation identity.

Two tiers:

- **Offline** (`_frame`/`_players` fixtures): the cache-or-fetch contract, the
  target definition, and the reconciliation arithmetic. These must never touch
  the network.
- **Network** (`test_reconciliation_identity_holds_on_real_data`): the actual
  claim the architecture rests on, checked against real nflverse data. Skipped
  by default so CI stays hermetic; run it explicitly when the upstream schema
  moves.

The identity is the load-bearing assertion. If `sum(players) == team` stops
holding, an allocate-by-share architecture is unsound and the whole approach
needs rethinking rather than retuning.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.data import team_stats


def _frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["game_id", "season", "week", "team", "opponent_team", "season_type",
                 "passing_yards", "rushing_yards", "receiving_yards"],
    )


def _players(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["season", "week", "team", "player", "passing_yards", "rushing_yards"]
    )


# ---------------------------------------------------------------- offline


def test_total_yards_is_passing_plus_rushing():
    frame = _frame([("g1", 2024, 1, "PHI", "ATL", "REG", 250, 90, 300)])
    out = team_stats.add_total_yards(frame)
    assert out.iloc[0]["total_yards"] == 340


def test_total_yards_is_not_passing_plus_rushing_plus_receiving():
    """Receiving yards are already inside passing yards. Including them would
    double-count every pass completion and inflate a team total by ~200."""
    frame = _frame([("g1", 2024, 1, "PHI", "ATL", "REG", 250, 90, 300)])
    out = team_stats.add_total_yards(frame)
    assert out.iloc[0]["total_yards"] == 340, "receiving_yards must not be added on top"


def test_total_yards_is_nan_when_a_component_is_missing():
    """`min_count` means a partially-known team-game is NaN, not a silently
    understated total -- the failure mode that made the original dashboard
    number indefensible."""
    frame = _frame([("g1", 2024, 1, "PHI", "ATL", "REG", 250, None, 300)])
    out = team_stats.add_total_yards(frame)
    assert pd.isna(out.iloc[0]["total_yards"])


def test_reconcile_reports_zero_diff_for_consistent_rows():
    team = _frame([("g1", 2024, 1, "PHI", "ATL", "REG", 250, 90, 300)])
    players = _players([
        (2024, 1, "PHI", "qb", 250, 0),
        (2024, 1, "PHI", "rb1", 0, 60),
        (2024, 1, "PHI", "rb2", 0, 30),
    ])
    out = team_stats.reconcile_against_players(team, players)
    assert len(out) == 1
    assert out.iloc[0]["total_yards"] == 340
    assert out.iloc[0]["player_total_yards"] == 340
    assert out.iloc[0]["diff"] == 0


def test_reconcile_surfaces_a_mismatch_rather_than_hiding_it():
    """The reason this function exists. A roster sum that misses must show up as
    a non-zero `diff`, not be quietly averaged away."""
    team = _frame([("g1", 2024, 1, "PHI", "ATL", "REG", 250, 90, 300)])
    players = _players([(2024, 1, "PHI", "qb", 250, 0)])
    out = team_stats.reconcile_against_players(team, players)
    assert out.iloc[0]["diff"] == 90


def test_fetch_caches_per_season_without_hitting_the_network_twice(monkeypatch, tmp_path):
    monkeypatch.setattr(team_stats, "TEAM_STATS_CACHE_DIR", tmp_path)
    calls: list[int] = []

    def fake_download(season: int) -> pd.DataFrame:
        calls.append(season)
        return _frame([(f"g{season}", season, 1, "PHI", "ATL", "REG", 250, 90, 300)])

    monkeypatch.setattr(team_stats, "_download", fake_download)

    first = team_stats.fetch_team_stats([2023, 2024])
    assert sorted(calls) == [2023, 2024]
    assert len(first) == 2

    second = team_stats.fetch_team_stats([2023, 2024])
    assert calls == [2023, 2024], "cached seasons must not be re-downloaded"
    # Compare values, not dtypes: a parquet round-trip can promote a string
    # column to pandas' StringDtype, which is not a behaviour change.
    pd.testing.assert_frame_equal(first.astype(object), second.astype(object))

    team_stats.fetch_team_stats([2023], force_refresh=True)
    assert calls == [2023, 2024, 2023]


def test_download_rejects_an_upstream_schema_change(monkeypatch):
    """A silently-dropped column would quietly remove a model feature. Fail loudly."""
    monkeypatch.setattr(
        team_stats.pd, "read_parquet",
        lambda url: pd.DataFrame({"game_id": ["g"], "season": [2024]}),
    )
    with pytest.raises(ValueError, match="missing expected columns"):
        team_stats._download(2024)


def test_keep_columns_are_unique():
    assert len(team_stats.KEEP_COLUMNS) == len(set(team_stats.KEEP_COLUMNS))


# ---------------------------------------------------------------- network


@pytest.mark.network
def test_reconciliation_identity_holds_on_real_data():
    """The architecture's load-bearing claim, against real nflverse data.

    Skipped by default. Run with:
        PYTHONPATH=src pytest tests/test_team_stats.py -m network -q

    Measured 2026-09-27 on the 2024 regular season: **544 of 544 team-games
    exact, maximum absolute difference 0.0 yards.**

    Source choice matters, and is why this reads the release URL directly rather
    than going through `nfl_data_py`. `stats_team_week` and `stats_player_week`
    are generated together in the same nflverse release and agree to the yard.
    `player_stats` -- the separate file `nfl_data_py.import_weekly_data` returns
    -- disagrees at one team-game (IND 2024 week 8, by 13 yards), so pairing a
    `stats_team` total against a `player_stats` sum gives 543/544. Both facts are
    recorded rather than one being quietly averaged away.
    """
    pytest.importorskip("nfl_data_py")
    try:
        team = team_stats.fetch_team_stats([2024])
        players = pd.read_parquet(
            "https://github.com/nflverse/nflverse-data/releases/download/"
            "stats_player/stats_player_week_2024.parquet"
        ).rename(columns={"recent_team": "team"})
    except Exception as exc:  # network unavailable
        pytest.skip(f"nflverse unreachable: {exc}")

    if "season_type" in players.columns:
        players = players[players["season_type"] == "REG"]
    team_reg = team[team["season_type"] == "REG"]

    out = team_stats.reconcile_against_players(team_reg, players)
    assert not out.empty, "no overlapping team-games; upstream schema probably changed"
    worst = out["diff"].abs().max()
    assert worst <= 1, (
        f"reconciliation identity broken: max |diff| {worst} across {len(out)} team-games. "
        f"An allocate-by-share architecture is unsound if the player sum no longer "
        f"reproduces the team total. Offenders: "
        f"{out.loc[out['diff'].abs() > 1, ['season', 'week', 'team', 'diff']].to_dict('records')[:5]}"
    )


@pytest.mark.network
def test_player_stats_source_has_a_known_one_game_discrepancy():
    """`nfl_data_py.import_weekly_data` reads a different file from the one
    `stats_team_week` is generated alongside, and the two disagree at exactly one
    team-game in 2024 (IND week 8, 13 yards).

    Pinned so the discrepancy stays a known quantity rather than a surprise: if a
    future reconciliation is off by a small constant, check this first.
    """
    pytest.importorskip("nfl_data_py")
    try:
        team = team_stats.fetch_team_stats([2024])
        import nfl_data_py as nfl

        players = nfl.import_weekly_data([2024]).rename(columns={"recent_team": "team"})
    except Exception as exc:
        pytest.skip(f"upstream unreachable: {exc}")

    if "season_type" in players.columns:
        players = players[players["season_type"] == "REG"]
    out = team_stats.reconcile_against_players(team[team["season_type"] == "REG"], players)
    offenders = out[out["diff"].abs() > 1]
    assert len(offenders) <= 2, (
        f"the known player_stats discrepancy grew from 1 team-game to {len(offenders)}; "
        f"re-examine before trusting either source"
    )


def test_a_missing_component_yields_a_missing_total_not_a_zero():
    """Kills the `min_count=len(summed)`-without-`min_count=1` mutant.

    `groupby.sum()` coerces an all-NaN column to 0.0, so a team-game where no
    player row carried `rushing_yards` looked like a team that rushed nothing, and
    the axis-level presence check saw two real numbers and never fired. The result
    was a *partial* player total that reads as complete — worse than an absent one,
    because an absent one shows up in a count and a partial one does not.
    """
    team = pd.DataFrame([{"season": 2024, "week": 1, "team": "A", "passing_yards": 250, "rushing_yards": 100}])
    players = pd.DataFrame([
        {"season": 2024, "week": 1, "team": "A", "passing_yards": 250, "rushing_yards": float("nan")},
    ])
    out = team_stats.reconcile_against_players(team, players)
    total = out["player_total_yards"].iloc[0]
    assert total != total, f"a missing rushing component must report a missing total, not {total}"
    assert out["diff"].iloc[0] != out["diff"].iloc[0], "and a missing diff, not a number"

"""Source selection for weekly player stats.

The `player_stats` release stops at 2024; `stats_player` carries the same table
under a different release tag. `fetch_weekly_player_stats` therefore tries the
first and falls back to the second. These tests pin which one wins and when,
that the per-season isolation survives, and that the cache cannot silently
serve a file from the wrong release.

None of this hits the network: the `player_stats` URL is reached through
`_import_weekly_data` and the `stats_player` URL through `pd.read_parquet`,
both patched. The `pd.read_parquet` patch only claims `http` paths and
delegates everything else, so the on-disk cache round-trip still runs for real.
"""

import logging

import pandas as pd
import pytest

from nfl_predictor.data import player_stats

NOT_FOUND = "HTTP Error 404: Not Found"


def _missing(*args, **kwargs):
    raise pd.errors.ParserError(NOT_FOUND)


def _raw_player_stats_frame(season=2024):
    """A frame in `player_stats` release shape: `recent_team`, no `game_id`."""
    return pd.DataFrame(
        [
            {
                "player_id": "00-0033873", "player_name": "P. Mahomes", "position": "QB",
                "recent_team": "KC", "season": season, "week": 1,
                "passing_yards": 291, "passing_tds": 2, "rushing_yards": 12, "rushing_tds": 0,
                "receiving_yards": 0, "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 3,
            },
            {
                "player_id": "00-0035211", "player_name": "I. Pacheco", "position": "RB",
                "recent_team": "KC", "season": season, "week": 2,
                "passing_yards": 0, "passing_tds": 0, "rushing_yards": 21, "rushing_tds": 1,
                "receiving_yards": 34, "receiving_tds": 0, "receptions": 3, "targets": 4, "carries": 9,
            },
        ]
    )


def _raw_stats_player_frame(season):
    """A frame in `stats_player` release shape: `team`, plus a native `game_id`."""
    return pd.DataFrame(
        [
            {
                "player_id": "00-0033873", "player_name": "P. Mahomes", "position": "QB",
                "team": "PIT", "game_id": f"{season}_01_PIT_CIN", "season": season, "week": 1,
                "passing_yards": 291, "passing_tds": 2, "rushing_yards": 12, "rushing_tds": 0,
                "receiving_yards": 0, "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 3,
            },
            {
                # A defender. `player_stats` would never publish this row;
                # `stats_player` publishes every player who appeared in a game.
                # It must reach the frame intact -- this module does not decide
                # who gets props.
                "player_id": "00-0034857", "player_name": "M. Rapp", "position": "LB",
                "team": "BAL", "game_id": f"{season}_02_CIN_BAL", "season": season, "week": 2,
                "passing_yards": 0, "passing_tds": 0, "rushing_yards": 0, "rushing_tds": 0,
                "receiving_yards": 0, "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 0,
            },
        ]
    )


@pytest.fixture
def upstream(monkeypatch):
    """Patch both upstream reads and record what each was asked for.

    `player_missing` / `stats_player_missing` are collections of seasons that
    404; anything not listed answers. Both default to nothing missing, so a
    test that does not care about a release says nothing about it.
    """
    state = {"player_missing": set(), "stats_player_missing": set(),
             "player_calls": [], "parquet_urls": []}

    def fake_import(years, columns=None):
        state["player_calls"].append(list(years))
        if years[0] in state["player_missing"]:
            _missing()
        return _raw_player_stats_frame(years[0])

    monkeypatch.setattr(player_stats, "_import_weekly_data", fake_import)

    real_read_parquet = pd.read_parquet

    def fake_read_parquet(path, columns=None, engine="auto"):
        url = str(path)
        if not url.startswith("http"):
            return real_read_parquet(path, columns=columns, engine=engine)
        state["parquet_urls"].append(url)
        season = int(url[: -len(".parquet")].rsplit("_", 1)[1])
        if season in state["stats_player_missing"]:
            _missing()
        return _raw_stats_player_frame(season)

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    return state


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    return tmp_path


def _marker(cache_dir, season):
    return (cache_dir / f"{season}.source").read_text().strip()


def test_player_stats_release_is_used_when_it_answers(upstream, cache_dir):
    frame = player_stats.fetch_weekly_player_stats([2024])

    assert upstream["player_calls"] == [[2024]]
    assert upstream["parquet_urls"] == [], "the fallback ran even though player_stats answered"
    assert frame["player_id"].tolist() == ["00-0033873", "00-0035211"]
    assert frame["recent_team"].tolist() == ["KC", "KC"]


def test_stats_player_fallback_answers_the_season_player_stats_drops(upstream, cache_dir):
    """2025 and 2026 are 404 on `player_stats` and exist only under `stats_player`."""
    upstream["player_missing"] = {2026}

    frame = player_stats.fetch_weekly_player_stats([2026])

    assert upstream["player_calls"] == [[2026]], "player_stats should have been tried first"
    assert upstream["parquet_urls"] == [
        "https://github.com/nflverse/nflverse-data/releases/download/"
        "stats_player/stats_player_week_2026.parquet"
    ]
    # The release's `team` is renamed, because `_attach_game_id` joins the
    # stats frame to the schedule on `recent_team`.
    assert frame["recent_team"].tolist() == ["PIT", "BAL"]
    # Its native game_id is dropped, because `_attach_game_id` would merge a
    # second one in and leave `game_id_x`/`game_id_y` behind, breaking
    # reconcile's `["game_id", "player_id"]` join.
    assert "game_id" not in frame.columns
    assert "team" not in frame.columns
    assert frame["player_id"].tolist() == ["00-0033873", "00-0034857"]


def test_the_fallback_frame_carries_the_same_column_contract(upstream, cache_dir):
    upstream["player_missing"] = {2025}

    frame = player_stats.fetch_weekly_player_stats([2025])

    assert frame.columns.tolist() == player_stats.KEEP_COLUMNS


def test_the_release_that_answered_records_itself_beside_the_cache(upstream, cache_dir):
    upstream["player_missing"] = {2026}

    player_stats.fetch_weekly_player_stats([2026])

    assert _marker(cache_dir, 2026) == "stats_player"
    upstream["player_missing"] = set()
    player_stats.fetch_weekly_player_stats([2024])
    assert _marker(cache_dir, 2024) == "player_stats"


def test_a_season_missing_from_both_releases_returns_an_empty_frame(upstream, cache_dir, caplog):
    """Preserved behaviour: log at INFO, return empty, raise nothing.

    This is the silent-failure path `docs/player-prop-accuracy-blocker.md`
    describes, so it is pinned rather than left implicit.
    """
    upstream["player_missing"] = {2031}
    upstream["stats_player_missing"] = {2031}

    with caplog.at_level(logging.INFO, logger=player_stats.__name__):
        frame = player_stats.fetch_weekly_player_stats([2031])

    assert frame.empty
    assert frame.columns.tolist() == player_stats.KEEP_COLUMNS
    assert any("No weekly player stats available yet" in r.getMessage() for r in caplog.records)
    assert not (cache_dir / "2031.parquet").exists(), "a missing season must not be cached"


def test_one_missing_season_does_not_404_out_the_rest_of_the_batch(upstream, cache_dir):
    """2031 is published by neither release; 2024 still has to arrive."""
    upstream["player_missing"] = {2024, 2031}
    upstream["stats_player_missing"] = {2031}

    frame = player_stats.fetch_weekly_player_stats([2024, 2031])

    assert frame["season"].unique().tolist() == [2024]
    assert frame["player_id"].tolist() == ["00-0033873", "00-0034857"]


def test_negative_yards_pass_through_unclamped(upstream, cache_dir, monkeypatch):
    """No guard, deliberately.

    Official yard totals are net of a player's own fumbles, so nflverse emits
    negatives in *both* releases: measured over 2001-2024 the old file has
    1,235 negative `receiving_yards` player-weeks to the new file's 1,247. A
    clamp would silently rewrite grading for every historical season to paper
    over the twelve rows where the new file books a sack fumble against
    receiving yards. Pinned so the decision is not reversed by accident.
    """
    negatives = pd.DataFrame(
        [
            {"player_id": "00-0022924", "player_name": "B. Roethlisberger", "position": "QB",
             "recent_team": "PIT", "season": 2024, "week": 4, "passing_yards": -6,
             "passing_tds": 0, "rushing_yards": -3, "rushing_tds": 0, "receiving_yards": -5,
             "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 0},
            {"player_id": "00-0034857", "player_name": "M. Rapp", "position": "LB",
             "recent_team": "BAL", "season": 2024, "week": 2, "passing_yards": 0,
             "passing_tds": 0, "rushing_yards": -7, "rushing_tds": 0, "receiving_yards": -2,
             "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 0},
        ]
    )
    monkeypatch.setattr(player_stats, "_import_weekly_data", lambda years, columns=None: negatives)

    frame = player_stats.fetch_weekly_player_stats([2024])

    # keyed by player: the frame comes back sorted by (season, week), so row
    # order is not the fixture's order and asserting on it would be brittle
    yards = frame.set_index("player_id")
    assert yards.loc["00-0022924", "receiving_yards"] == -5
    assert yards.loc["00-0022924", "rushing_yards"] == -3
    assert yards.loc["00-0022924", "passing_yards"] == -6
    assert yards.loc["00-0034857", "receiving_yards"] == -2
    assert yards.loc["00-0034857", "rushing_yards"] == -7
    assert yards.loc["00-0034857", "passing_yards"] == 0


def test_reading_2024_is_unchanged_by_the_fallback_being_added(upstream, cache_dir):
    """Regression guard for the season `player_stats` still publishes.

    2024 comes back from the old release, with the old column set and the old
    row count -- 5,597 rows in production, two here -- and it survives the
    parquet cache round-trip as exactly the frame it was fetched with.
    """
    raw = _raw_player_stats_frame(2024)

    frame = player_stats.fetch_weekly_player_stats([2024])

    assert frame.columns.tolist() == raw.columns.tolist() == player_stats.KEEP_COLUMNS
    assert len(frame) == len(raw)
    # check_dtype=False: the parquet round-trip is already not dtype-preserving for
    # string columns. That predates this change, so asserting dtypes here would be
    # pinning a property of the cache format rather than of source selection.
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True), raw.reset_index(drop=True), check_dtype=False
    )
    assert upstream["parquet_urls"] == []


def test_a_cached_season_is_served_without_touching_either_release(upstream, cache_dir):
    upstream["player_missing"] = {2026}
    player_stats.fetch_weekly_player_stats([2026])
    assert upstream["player_calls"] == [[2026]], "precondition: the season was fetched"

    again = player_stats.fetch_weekly_player_stats([2026])

    assert upstream["player_calls"] == [[2026]], "a cached season must not be re-fetched"
    assert len(again) == 2


def test_a_legacy_cache_with_no_release_marker_is_served_and_claimed(upstream, cache_dir):
    """The seven caches already on this machine (2018-2024) carry no marker.

    Every one of them was written by `player_stats`, which is still the source
    this rule selects for all of those seasons, so they are served as-is --
    re-downloading 26 seasons to establish that would be absurd. The marker is
    written so the next read does not have to re-reason it.
    """
    legacy = _raw_player_stats_frame(2024)
    legacy.to_parquet(cache_dir / "2024.parquet")
    assert not (cache_dir / "2024.source").exists(), "precondition: no marker"

    frame = player_stats.fetch_weekly_player_stats([2024])

    assert upstream["player_calls"] == [], "a legacy cache must not trigger a fetch"
    assert len(frame) == len(legacy)
    assert _marker(cache_dir, 2024) == "player_stats"


def test_a_fallback_written_cache_is_not_refetched_on_every_call(upstream, cache_dir):
    """The trap this design walked into, pinned shut.

    The first draft re-fetched whenever a cache's marker named `stats_player`,
    on the theory that `player_stats` might have started publishing that season
    again. It cannot detect that from the marker alone -- only an HTTP probe
    could, and the marker read is not a probe -- so the rule just re-downloaded
    2026 on every call, forever, which is the opposite of a cache. A
    fallback-written cache is now served like any other.
    """
    upstream["player_missing"] = {2026}
    player_stats.fetch_weekly_player_stats([2026])
    assert _marker(cache_dir, 2026) == "stats_player", "precondition: cached from the fallback"
    assert upstream["player_calls"] == [[2026]]

    # Even now that player_stats answers again, the cache is served.
    upstream["player_missing"] = set()
    player_stats.fetch_weekly_player_stats([2026])
    player_stats.fetch_weekly_player_stats([2026])

    assert upstream["player_calls"] == [[2026]], "the fallback cache was re-fetched"
    assert len(player_stats.fetch_weekly_player_stats([2026])) == 2


def test_force_refresh_re_settles_a_cached_season_and_its_marker(upstream, cache_dir):
    """`force_refresh` is the documented way to move a season onto a new source."""
    upstream["player_missing"] = {2026}
    player_stats.fetch_weekly_player_stats([2026])
    assert _marker(cache_dir, 2026) == "stats_player"

    upstream["player_missing"] = set()
    frame = player_stats.fetch_weekly_player_stats([2026], force_refresh=True)

    assert upstream["player_calls"] == [[2026], [2026]], "force_refresh must re-fetch"
    # Now player_stats publishes 2026, so the cache is that source's and says so.
    assert frame["player_id"].tolist() == ["00-0033873", "00-0035211"]
    assert _marker(cache_dir, 2026) == "player_stats"


def test_a_mixed_cache_dir_records_provenance_per_season(upstream, cache_dir):
    """The coherence property that actually matters, and that is cheap.

    A cache dir holding 2024 from `player_stats` and 2026 from `stats_player`
    is the expected steady state, not a fault. What would be a fault is not
    being able to see it -- so each season's marker names the release its
    values came from, and the two are independently readable.
    """
    upstream["player_missing"] = {2026}

    player_stats.fetch_weekly_player_stats([2024, 2026])

    assert _marker(cache_dir, 2024) == "player_stats"
    assert _marker(cache_dir, 2026) == "stats_player"
    assert player_stats._read_source_marker(2024) == "player_stats"
    assert player_stats._read_source_marker(2026) == "stats_player"
    assert player_stats._read_source_marker(2019) is None, "an uncached season has no marker"

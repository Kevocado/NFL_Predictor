import pandas as pd

from nfl_predictor.data import player_stats


def _raw_weekly_frame():
    return pd.DataFrame(
        [
            {
                "player_id": "00-0033873", "player_name": "P. Mahomes", "position": "QB",
                "recent_team": "KC", "season": 2025, "week": 1,
                "passing_yards": 291, "passing_tds": 2, "rushing_yards": 12, "rushing_tds": 0,
                "receiving_yards": 0, "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 3,
            }
        ]
    )


def test_fetch_weekly_player_stats_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    calls = []

    def fake_import(years, columns=None):
        calls.append(list(years))
        return _raw_weekly_frame()

    monkeypatch.setattr(player_stats, "_import_weekly_data", fake_import)

    first = player_stats.fetch_weekly_player_stats([2025])
    second = player_stats.fetch_weekly_player_stats([2025])

    assert len(calls) == 1
    assert len(first) == 1
    assert second.equals(first)


def test_fetch_weekly_player_stats_force_refresh_refetches(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        player_stats, "_import_weekly_data",
        lambda years, columns=None: calls.append(years) or _raw_weekly_frame(),
    )

    player_stats.fetch_weekly_player_stats([2025])
    player_stats.fetch_weekly_player_stats([2025], force_refresh=True)

    assert len(calls) == 2


def test_an_empty_pull_is_not_cached_and_cannot_poison_the_cache(monkeypatch, tmp_path):
    """A 200 with no rows for the season must not become a permanent absence.

    `fetch_weekly_player_stats` wrote `season_df` to the per-season parquet
    unconditionally, including when it was empty, and the cache check was a bare
    `path.exists()`. So a single empty pull was cached and then returned forever:

        1st call: 0 rows, empty=True, cache file written
        2nd call (upstream now published): 0 rows, empty=True

    That is the shape this API has *before* a season is published, so prop
    reconciliation could never recover even after nflverse published -- which breaks
    the remediation sequence in docs/player-prop-accuracy-blocker.md at step 1. The
    `hub_cache` sibling module states the right discipline in its own docstring --
    "Empty or failed pulls are never cached" -- and this one did not follow it.
    """
    from nfl_predictor.data import player_stats

    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)

    state = {"frame": pd.DataFrame(columns=player_stats.KEEP_COLUMNS), "calls": 0}

    def fake_import(years, columns=None):
        state["calls"] += 1
        return state["frame"]

    monkeypatch.setattr(player_stats, "_import_weekly_data", fake_import)

    first = player_stats.fetch_weekly_player_stats([2025])
    assert first.empty
    assert not player_stats._season_cache_path(2025).exists(), (
        "an empty pull must not be written to the cache; if it is, the absence is permanent"
    )

    # Upstream now has the data. The next call must see it.
    state["frame"] = _raw_weekly_frame()
    second = player_stats.fetch_weekly_player_stats([2025])

    assert len(second) == 1, "an empty pull must not shadow the real data"
    assert state["calls"] == 2


def test_an_already_written_empty_cache_is_discarded(monkeypatch, tmp_path):
    """A cache file left by the old build must not be honoured either.

    Deployments that already ran the old code may have an empty parquet on disk for
    the current season, and the fix has to clear that too -- otherwise the bug survives
    the deploy that fixes it.
    """
    from nfl_predictor.data import player_stats

    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    path = player_stats._season_cache_path(2025)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=player_stats.KEEP_COLUMNS).to_parquet(path)
    assert path.exists(), "precondition: an empty cache file is on disk"

    state = {"calls": 0}

    def fake_import(years, columns=None):
        state["calls"] += 1
        return _raw_weekly_frame()

    monkeypatch.setattr(player_stats, "_import_weekly_data", fake_import)

    out = player_stats.fetch_weekly_player_stats([2025])

    assert state["calls"] == 1, "the empty cache was treated as a hit"
    assert len(out) == 1

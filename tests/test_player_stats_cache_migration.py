"""A cached season written before KEEP_COLUMNS grew (attempts/completions) must be refreshed, not served.

Production keeps a PERSISTENT cache volume (/opt/stack/volumes/nfl-cache -> /app/data/cache). Its 2025/2026 files predate
`attempts`/`completions`; reading them as-is made `player_usage._add_rolling` raise KeyError and would have taken every
player prop down on the first deploy that requires the columns (CodeRabbit, NFL#68). Offline: `_fetch_season` is stubbed."""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.data import player_stats as ps


def _season_frame(season: int, columns=None) -> pd.DataFrame:
    row = {c: 0 for c in ps.KEEP_COLUMNS}
    row.update(player_id="p1", player_name="Pat QB", position="QB", recent_team="KC", season=season, week=1,
               passing_yards=250, attempts=33, completions=22)
    frame = pd.DataFrame([row])
    return frame if columns is None else frame[columns]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "PLAYER_STATS_CACHE_DIR", tmp_path)
    return tmp_path


def _write_old_cache(season: int) -> None:
    old_columns = [c for c in ps.KEEP_COLUMNS if c not in ("attempts", "completions")]
    _season_frame(season, old_columns).to_parquet(ps._season_cache_path(season))


def test_a_cache_missing_required_columns_is_refreshed(cache, monkeypatch):
    _write_old_cache(2025)
    calls = []

    def fake_fetch(season):
        calls.append(season)
        return _season_frame(season), "player_stats"

    monkeypatch.setattr(ps, "_fetch_season", fake_fetch)
    out = ps.fetch_weekly_player_stats([2025])

    assert calls == [2025], "the stale cache must be re-fetched"
    assert set(ps.KEEP_COLUMNS) <= set(out.columns)
    assert out.loc[0, "attempts"] == 33
    # and the refreshed file replaced the stale one on disk
    assert {"attempts", "completions"} <= set(pd.read_parquet(ps._season_cache_path(2025)).columns)


def test_a_complete_cache_is_not_refetched(cache, monkeypatch):
    _season_frame(2025).to_parquet(ps._season_cache_path(2025))
    monkeypatch.setattr(ps, "_fetch_season", lambda s: (_ for _ in ()).throw(AssertionError("must not fetch")))
    out = ps.fetch_weekly_player_stats([2025])
    assert len(out) == 1 and out.loc[0, "attempts"] == 33


def test_if_the_refresh_fails_the_stale_history_is_kept_with_nan_not_dropped(cache, monkeypatch):
    """nflverse unreachable: dropping the season would blank real history, and serving it with a missing column would
    raise. Keep the rows, mark the new columns as unknown (NaN), never invent zeros."""
    _write_old_cache(2025)
    monkeypatch.setattr(ps, "_fetch_season", lambda s: (_ for _ in ()).throw(OSError("nflverse unreachable")))
    out = ps.fetch_weekly_player_stats([2025])

    assert len(out) == 1 and out.loc[0, "passing_yards"] == 250, "the stale rows are still served"
    assert set(ps.KEEP_COLUMNS) <= set(out.columns), "no column is missing, so no KeyError downstream"
    assert out["attempts"].isna().all() and out["completions"].isna().all()


def test_stale_and_current_seasons_are_served_together(cache, monkeypatch):
    _write_old_cache(2025)
    _season_frame(2024).to_parquet(ps._season_cache_path(2024))
    monkeypatch.setattr(ps, "_fetch_season", lambda s: (_season_frame(s), "player_stats"))
    out = ps.fetch_weekly_player_stats([2024, 2025])
    assert sorted(out["season"]) == [2024, 2025]
    assert out["attempts"].notna().all()

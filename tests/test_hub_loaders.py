"""Review fixes: the hub's loaders must fetch what the hub reads, refresh the
current season, and survive nflverse having nothing yet."""
import os
import time

import pandas as pd
import pytest

from nfl_predictor.data import player_season as ps
from nfl_predictor.data import team_efficiency as te


def test_player_season_runs_on_exactly_the_columns_the_hub_fetch_returns():
    row = {c: 0 for c in ps.HUB_WEEKLY_COLUMNS}
    row.update(player_id="q1", player_display_name="Pat QB", position="QB", recent_team="KC", season=2026,
               week=1, passing_epa=1.5, rushing_epa=None, receiving_epa=None, target_share=None,
               air_yards_share=None, fantasy_points_ppr=20.0)
    out = ps.player_season(pd.DataFrame([row], columns=ps.HUB_WEEKLY_COLUMNS), 2026)
    assert out["players"][0]["epa_total"] == 1.5


def test_hub_weekly_fetch_asks_for_the_hub_columns_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "HUB_CACHE_DIR", tmp_path)
    calls = []

    def fake_import(years, columns):
        calls.append((years, columns))
        return pd.DataFrame([{c: 0 for c in columns} | {"season": years[0]}])

    monkeypatch.setattr(ps, "_import_weekly", fake_import)
    ps.load_hub_weekly(2020)
    ps.load_hub_weekly(2020)
    assert len(calls) == 1 and calls[0] == ([2020], ps.HUB_WEEKLY_COLUMNS)


def _age(path, hours):
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


@pytest.mark.parametrize("mod,loader,importer,cache_attr", [
    (ps, "load_hub_weekly", "_import_weekly", "HUB_CACHE_DIR"),
    (te, "load_pbp", "_import_pbp", "PBP_CACHE_DIR"),
])
def test_current_season_refreshes_when_stale_and_keeps_stale_copy_on_error(monkeypatch, tmp_path, mod, loader, importer, cache_attr):
    monkeypatch.setattr(mod, cache_attr, tmp_path)
    season = mod.CURRENT_SEASON
    n = [0]

    def fetch(years, columns):
        n[0] += 1
        return pd.DataFrame([{c: n[0] if c == "week" else 0 for c in columns}])

    monkeypatch.setattr(mod, importer, fetch)
    assert getattr(mod, loader)(season)["week"].iloc[0] == 1
    assert getattr(mod, loader)(season)["week"].iloc[0] == 1  # fresh: served from cache
    _age(next(tmp_path.iterdir()), 13)
    assert getattr(mod, loader)(season)["week"].iloc[0] == 2  # stale: refetched
    _age(next(tmp_path.iterdir()), 13)
    monkeypatch.setattr(mod, importer, lambda years, columns: (_ for _ in ()).throw(OSError("nflverse down")))
    assert getattr(mod, loader)(season)["week"].iloc[0] == 2  # error: stale copy beats nothing


@pytest.mark.parametrize("mod,loader,importer,cache_attr,cols", [
    (ps, "load_hub_weekly", "_import_weekly", "HUB_CACHE_DIR", "HUB_WEEKLY_COLUMNS"),
    (te, "load_pbp", "_import_pbp", "PBP_CACHE_DIR", "PBP_COLUMNS"),
])
def test_nothing_published_yet_gives_empty_frame_and_caches_nothing(monkeypatch, tmp_path, mod, loader, importer, cache_attr, cols):
    monkeypatch.setattr(mod, cache_attr, tmp_path)
    monkeypatch.setattr(mod, importer, lambda years, columns: (_ for _ in ()).throw(ValueError("404")))
    df = getattr(mod, loader)(mod.CURRENT_SEASON)
    assert df.empty and list(df.columns) == getattr(mod, cols)
    assert list(tmp_path.iterdir()) == []

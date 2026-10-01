import os
import time
import pytest
import pandas as pd

from nfl_predictor.data import injuries


def _raw_injury_frame():
    return pd.DataFrame(
        [
            {"season": 2025, "week": 1, "team": "KC", "gsis_id": "00-0033873",
             "full_name": "Patrick Mahomes", "position": "QB", "report_status": "Questionable"},
            {"season": 2025, "week": 1, "team": "KC", "gsis_id": "00-0031234",
             "full_name": "Backup Guy", "position": "RB", "report_status": None},
        ]
    )


def test_fetch_injuries_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(injuries, "INJURIES_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        injuries, "_import_injuries",
        lambda years: calls.append(years) or _raw_injury_frame(),
    )

    first = injuries.fetch_injuries([2025])
    second = injuries.fetch_injuries([2025])

    assert len(calls) == 1
    assert len(first) == 2
    assert second.equals(first)


def test_current_status_by_player_only_includes_flagged_players(monkeypatch, tmp_path):
    monkeypatch.setattr(injuries, "INJURIES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(injuries, "_import_injuries", lambda years: _raw_injury_frame())

    df = injuries.fetch_injuries([2025])
    status = injuries.current_status_by_player(df, season=2025, week=1)

    assert status == {"00-0033873": "Questionable"}

# --- the injury cache expires (NFL#24 review) -----------------------------------
# A gate that reads this on every request cannot reuse a season file written once: by
# week 2 it holds no row for the current week and silently gates nobody (same defect and
# fix as NBA#20).
@pytest.fixture
def ttl_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(injuries, "INJURIES_CACHE_DIR", tmp_path)
    calls = []

    def _import(years):
        calls.append(list(years))
        row = {c: None for c in injuries.KEEP_COLUMNS}
        row.update(season=years[0], week=1, report_status="Out")
        return pd.DataFrame([row])

    monkeypatch.setattr(injuries, "_import_injuries", _import)
    return calls, tmp_path


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_a_fresh_cache_is_reused(ttl_cache):
    calls, _ = ttl_cache
    injuries.fetch_injuries([2026], max_age_seconds=3600)
    injuries.fetch_injuries([2026], max_age_seconds=3600)
    assert len(calls) == 1


def test_an_expired_cache_is_refetched(ttl_cache):
    calls, tmp = ttl_cache
    injuries.fetch_injuries([2026], max_age_seconds=3600)
    _age(tmp / "2026.parquet", 7200)
    injuries.fetch_injuries([2026], max_age_seconds=3600)
    assert len(calls) == 2, "an expired injury report was served from cache"


def test_without_a_max_age_the_cache_never_expires_as_before(ttl_cache):
    calls, tmp = ttl_cache
    injuries.fetch_injuries([2024])
    _age(tmp / "2024.parquet", 10**7)
    injuries.fetch_injuries([2024])
    assert len(calls) == 1, "a past season's cache must stay permanent"


def test_only_the_expired_season_is_refetched(ttl_cache):
    calls, tmp = ttl_cache
    injuries.fetch_injuries([2025, 2026], max_age_seconds=3600)
    _age(tmp / "2026.parquet", 7200)
    calls.clear()
    injuries.fetch_injuries([2025, 2026], max_age_seconds=3600)
    assert calls == [[2026]]

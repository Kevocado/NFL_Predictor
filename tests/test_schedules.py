import pandas as pd
import pytest

from nfl_predictor.data import schedules


def _raw_schedule_frame():
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_KC_BAL", "season": 2025, "week": 1,
                "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
                "home_score": 27, "away_score": 20, "home_rest": 7, "away_rest": 7,
                "div_game": 0, "roof": "outdoors", "surface": "grass",
                "temp": 72.0, "wind": 5.0, "spread_line": -2.5, "total_line": 46.5,
            },
            {
                "game_id": "2025_01_PHI_GB", "season": 2025, "week": 1,
                "gameday": "2025-09-05", "home_team": "GB", "away_team": "PHI",
                "home_score": None, "away_score": None, "home_rest": 7, "away_rest": 7,
                "div_game": 0, "roof": "outdoors", "surface": "grass",
                "temp": None, "wind": None, "spread_line": 1.5, "total_line": 45.0,
            },
        ]
    )


def test_fetch_schedules_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    calls = []

    def fake_import(years):
        calls.append(list(years))
        return _raw_schedule_frame()

    monkeypatch.setattr(schedules, "_import_schedules", fake_import)

    first = schedules.fetch_schedules([2025])
    second = schedules.fetch_schedules([2025])

    assert len(calls) == 1  # second call hit the cache, not the network
    assert len(first) == 2
    assert list(first.columns).__contains__("home_team")
    assert second.equals(first)


def test_load_training_data_drops_unplayed_games(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(schedules, "_import_schedules", lambda years: _raw_schedule_frame())

    df = schedules.load_training_data([2025])

    assert len(df) == 1
    assert df.iloc[0]["game_id"] == "2025_01_KC_BAL"


def test_default_completed_seasons_excludes_current_season(monkeypatch):
    monkeypatch.setattr(schedules, "CURRENT_SEASON", 2026)
    seasons = schedules.default_completed_seasons(n=3)
    assert seasons == [2023, 2024, 2025]


def test_fetch_upcoming_games_filters_season_week(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(schedules, "_import_schedules", lambda years: _raw_schedule_frame())

    upcoming = schedules.fetch_upcoming_games(2025, 1)

    assert list(upcoming["game_id"]) == ["2025_01_PHI_GB"]


def test_fetch_week_games_includes_finished_and_upcoming(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(schedules, "_import_schedules", lambda years: _raw_schedule_frame())

    week_games = schedules.fetch_week_games(2025, 1)

    assert set(week_games["game_id"]) == {"2025_01_KC_BAL", "2025_01_PHI_GB"}

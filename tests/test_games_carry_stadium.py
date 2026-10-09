"""The weekly schedule rows carried no `stadium`, so the games route looked every forecast up under None and attached no
weather to any game in production. The stadium comes through the REAL schedule path now (not injected by the test)."""
import pandas as pd

from nfl_predictor.data import schedules


def _raw(with_stadium=True):
    rows = [
        {"game_id": "2026_05_PHI_JAX", "season": 2026, "week": 5, "gameday": "2026-10-11", "gametime": "13:00",
         "home_team": "JAX", "away_team": "PHI", "home_score": None, "away_score": None, "home_rest": 7, "away_rest": 7,
         "div_game": 0, "roof": "outdoors", "surface": "grass", "temp": None, "wind": None, "spread_line": -7.5,
         "total_line": 41.5, "stadium": "TIAA Bank Stadium"},
        {"game_id": "2026_05_MIN_NO", "season": 2026, "week": 5, "gameday": "2026-10-11", "gametime": "13:00",
         "home_team": "NO", "away_team": "MIN", "home_score": None, "away_score": None, "home_rest": 7, "away_rest": 6,
         "div_game": 0, "roof": "dome", "surface": "turf", "temp": None, "wind": None, "spread_line": 2.5,
         "total_line": 42.5, "stadium": "Mercedes-Benz Superdome"},
    ]
    df = pd.DataFrame(rows)
    return df if with_stadium else df.drop(columns=["stadium"])


def test_weekly_schedule_keeps_the_stadium(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(schedules, "_import_schedules", lambda years: _raw())
    week = schedules.fetch_week_games(2026, 5)
    assert list(week["stadium"]) == ["TIAA Bank Stadium", "Mercedes-Benz Superdome"]


def test_an_upstream_frame_without_stadium_still_works(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(schedules, "_import_schedules", lambda years: _raw(with_stadium=False))
    week = schedules.fetch_week_games(2026, 5)
    assert len(week) == 2 and "stadium" not in week.columns


def test_games_route_attaches_conditions_through_the_real_schedule_path(monkeypatch, tmp_path):
    """End to end with NOTHING injected: schedule (stadium from the upstream frame) -> _get_games_live -> forecast."""
    from nfl_predictor.api import routes

    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(schedules, "_import_schedules", lambda years: _raw())
    seen = []

    def fake_forecast(stadium, kickoff_iso, now):
        seen.append(stadium)
        return {"kind": "rain", "temp_f": 50, "wind_mph": 10, "precip_pct": 60, "source": "open-meteo"}

    monkeypatch.setattr(routes._forecast_cache, "_fetch", fake_forecast)
    monkeypatch.setattr(routes._forecast_cache, "_store", {})
    out = routes._get_games_live(2026, 5)
    assert sorted(seen) == ["Mercedes-Benz Superdome", "TIAA Bank Stadium"]
    assert all(g.get("conditions", {}).get("kind") == "rain" for g in out)

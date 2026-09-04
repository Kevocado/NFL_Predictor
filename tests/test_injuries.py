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

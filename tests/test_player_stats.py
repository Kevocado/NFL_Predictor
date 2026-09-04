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

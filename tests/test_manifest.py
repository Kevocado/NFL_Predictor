import json

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models import manifest


def _fake_games(seasons):
    rng = np.random.default_rng(3)
    rows = []
    teams = [f"T{i}" for i in range(8)]
    for season in seasons:
        for week in range(1, 6):
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                rows.append(
                    {
                        "game_id": f"{season}_{week}_{home}_{away}", "season": season, "week": week,
                        "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)),
                        "home_team": home, "away_team": away,
                        "home_score": int(rng.integers(10, 35)), "away_score": int(rng.integers(10, 35)),
                    }
                )
    return pd.DataFrame(rows)


def _fake_player_stats(seasons):
    rng = np.random.default_rng(4)
    rows = []
    for season in seasons:
        for week in range(1, 6):
            rows.append(
                {
                    "player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "T0",
                    "season": season, "week": week,
                    "passing_yards": 0, "passing_tds": 0,
                    "rushing_yards": int(rng.integers(40, 120)), "rushing_tds": int(rng.integers(0, 2)),
                    "receiving_yards": int(rng.integers(0, 30)), "receiving_tds": 0,
                    "receptions": 2, "targets": 3, "carries": 18,
                }
            )
    return pd.DataFrame(rows)


def test_train_all_writes_a_manifest_with_chosen_candidate(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))

    result = manifest.train_all(seasons=seasons)

    assert result["chosen_candidate"] in ("elo", "ridge", "xgb")
    assert (tmp_path / "manifest.json").exists()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["chosen_candidate"] == result["chosen_candidate"]


def test_load_models_round_trips_after_train_all(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))

    manifest.train_all(seasons=seasons)
    models = manifest.load_models()

    assert "game_outcome_model" in models
    assert "player_models" in models
    assert "anytime_td" in models["player_models"]

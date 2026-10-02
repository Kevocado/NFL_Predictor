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
    assert (tmp_path / "game_outcome_model.pkl").exists()
    assert (tmp_path / "total_points_model.pkl").exists()
    assert (tmp_path / "anytime_td_model.pkl").exists()
    assert (tmp_path / "rushing_yards_model.pkl").exists()
    assert (tmp_path / "receiving_yards_model.pkl").exists()


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


def test_train_all_rejects_training_data_without_walk_forward_fold(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games([2024]))

    with pytest.raises(ValueError, match="walk-forward validation fold"):
        manifest.train_all(seasons=[2024])


def test_load_models_refuses_a_player_model_fitted_on_an_unservable_column(monkeypatch, tmp_path):
    """The general guard, at the entry point that matters.

    `predict_props` scores a live row with `reindex(feature_cols).fillna(0)`, so a
    fitted column the serving builder cannot emit is served as a constant zero for
    every player and every week -- which is exactly how `passing_tds_roll` shipped
    fitted-but-unserved. `load_models` now refuses the payload instead.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    good = {"chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
            "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
            "player_feature_cols": ["passing_yards_roll"], "yardage_metrics": [],
            "qb_passing_td": None}
    monkeypatch.setattr(manifest, "load_manifest", lambda: good)
    assert "anytime_td" in manifest.load_models()["player_models"]

    bad = {**good, "player_feature_cols": ["passing_yards_roll", "passing_tds_roll_not_emitted"]}
    monkeypatch.setattr(manifest, "load_manifest", lambda: bad)
    with pytest.raises(ValueError, match="does not emit"):
        manifest.load_models()


def test_load_models_ignores_stale_yardage_artifacts_after_retrain(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))
    manifest.train_all(seasons=seasons)
    assert (tmp_path / "receiving_yards_model.pkl").exists()

    without_receiving = _fake_player_stats(seasons)
    without_receiving["receiving_yards"] = 0
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: without_receiving)
    manifest.train_all(seasons=seasons)

    assert not (tmp_path / "receiving_yards_model.pkl").exists()
    assert "receiving_yards" not in manifest.load_models()["player_models"]

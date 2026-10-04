"""quantile artifact registry tests.

These write to tmp_path and never touch models/. The committed production
artifacts are checked by tests/test_tracked_artifacts.py, not here -- a unit test
that requires a multi-minute training run before it can pass is a test nobody
runs.

What is pinned here is the part that goes wrong quietly: the artifact's shape,
that extending the manifest leaves the existing keys untouched, and that the
recorded fingerprint actually describes the bytes on disk.
"""
from __future__ import annotations

import json
import pickle

import pandas as pd
import pytest

from nfl_predictor.models.player_props import QUANTILES, fit_yardage_quantile_models
from nfl_predictor.models.quantile_registry import (
    ARTIFACT_KEY, load_quantile_artifact, save_quantile_artifacts,
)

MARKETS = ["passing_yards", "rushing_yards", "receiving_yards"]


def _fitted(n=200):
    rng = pd.Series(range(n), dtype=float)
    X = pd.DataFrame({"a": rng, "b": rng * 0.5})
    y = 50 + 2 * rng + rng.rolling(7, min_periods=1).mean()
    return fit_yardage_quantile_models(X, y, quantiles=[0.1, 0.5, 0.9])


@pytest.fixture(scope="module")
def fitted():
    return _fitted()


def _payload(fitted):
    return {
        "quantile_models": fitted,
        "feature_cols": ["a", "b"],
        "trained_seasons": [2017, 2018, 2019],
        "walkforward_mae": 12.5,
        "walkforward_calibration": {"0.5-0.6": {"empirical": 0.54, "n": 900}},
    }


def _artifacts(fitted):
    return {market: _payload(fitted) for market in MARKETS}


def _manifest():
    return {"player_feature_cols": ["passing_yards_roll"], "sigma": 13.1}


def test_save_writes_one_pickle_per_market(tmp_path, fitted):
    manifest = save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path,
                                       manifest=_manifest(), manifest_path=tmp_path / "manifest.json")

    for market in MARKETS:
        assert (tmp_path / f"{market}_quantile_2025.pkl").exists()
    assert ARTIFACT_KEY in manifest


def test_artifact_payload_carries_the_documented_keys(tmp_path, fitted):
    save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path, manifest=_manifest(),
                            manifest_path=tmp_path / "manifest.json")

    payload = pickle.loads((tmp_path / "rushing_yards_quantile_2025.pkl").read_bytes())
    for key in ("quantile_models", "feature_cols", "trained_seasons",
                "walkforward_mae", "walkforward_calibration"):
        assert key in payload, f"missing {key}"
    assert sorted(payload["quantile_models"]) == [0.1, 0.5, 0.9]


def test_load_round_trips(tmp_path, fitted):
    save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path, manifest=_manifest(),
                            manifest_path=tmp_path / "manifest.json")

    payload = load_quantile_artifact("rushing_yards", out_dir=tmp_path)

    assert payload["feature_cols"] == ["a", "b"]
    assert sorted(payload["quantile_models"]) == [0.1, 0.5, 0.9]
    assert payload["walkforward_mae"] == 12.5


def test_manifest_extension_is_additive(tmp_path, fitted):
    original = _manifest()
    manifest = save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path,
                                       manifest=original, manifest_path=tmp_path / "manifest.json")

    assert manifest["player_feature_cols"] == ["passing_yards_roll"], "existing key was altered"
    assert manifest["sigma"] == 13.1


def test_manifest_is_written_to_disk(tmp_path, fitted):
    path = tmp_path / "manifest.json"
    save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path,
                            manifest=_manifest(), manifest_path=path)

    written = json.loads(path.read_text())
    assert ARTIFACT_KEY in written
    assert "player_feature_cols" in written


def test_manifest_records_a_fingerprint_that_matches_the_file(tmp_path, fitted):
    manifest = save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path,
                                       manifest=_manifest(), manifest_path=tmp_path / "manifest.json")

    entry = manifest[ARTIFACT_KEY]["markets"]["receiving_yards"]
    import hashlib
    digest = hashlib.sha256((tmp_path / "receiving_yards_quantile_2025.pkl").read_bytes()).hexdigest()

    assert entry["sha256"] == digest


def test_stale_fingerprint_is_detectable(tmp_path, fitted):
    """A manifest that no longer describes the bytes on disk must not pass."""
    from nfl_predictor.models.quantile_registry import verify_quantile_artifacts

    save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path, manifest=_manifest(),
                            manifest_path=tmp_path / "manifest.json")
    (tmp_path / "receiving_yards_quantile_2025.pkl").write_bytes(b"corrupted")

    problems = verify_quantile_artifacts(json.loads((tmp_path / "manifest.json").read_text()),
                                        out_dir=tmp_path)

    assert any("receiving_yards" in problem for problem in problems)


def test_verify_passes_on_a_clean_save(tmp_path, fitted):
    from nfl_predictor.models.quantile_registry import verify_quantile_artifacts

    manifest = save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path, manifest=_manifest(),
                                       manifest_path=tmp_path / "manifest.json")

    assert verify_quantile_artifacts(manifest, out_dir=tmp_path) == []


def test_verify_reports_a_missing_artifact(tmp_path, fitted):
    from nfl_predictor.models.quantile_registry import verify_quantile_artifacts

    manifest = save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path, manifest=_manifest(),
                                       manifest_path=tmp_path / "manifest.json")
    (tmp_path / "passing_yards_quantile_2025.pkl").unlink()

    problems = verify_quantile_artifacts(manifest, out_dir=tmp_path)

    assert any("passing_yards" in problem for problem in problems)


def test_save_refuses_a_market_missing_its_models(tmp_path, fitted):
    broken = _artifacts(fitted)
    broken["receiving_yards"] = {k: v for k, v in broken["receiving_yards"].items()
                                 if k != "quantile_models"}

    with pytest.raises(ValueError, match="receiving_yards"):
        save_quantile_artifacts(broken, out_dir=tmp_path, manifest=_manifest(),
                                manifest_path=tmp_path / "manifest.json")


def test_default_quantiles_are_recorded_in_the_manifest(tmp_path, fitted):
    manifest = save_quantile_artifacts(_artifacts(fitted), out_dir=tmp_path,
                                       manifest=_manifest(), manifest_path=tmp_path / "manifest.json")

    assert manifest[ARTIFACT_KEY]["quantiles"] == QUANTILES
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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models.player_props import QUANTILES, fit_yardage_quantile_models
from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS, train_all
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


def _passing_verdict():
    """What Task 14 hands over on a PASS: a calibration table with no failing bucket."""
    return {"walkforward_mae": 20.0,
            "walkforward_calibration": {"0.5-0.6": {"predicted": 0.55, "empirical": 0.56,
                                                     "n": 900, "gap": 0.01,
                                                     "within_tolerance": True}}}


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

# --- the artifacts Task 9 actually writes -----------------------------------

def test_train_all_writes_one_pickle_per_market(tmp_path, monkeypatch):
    """The production path, at the smallest scale that exercises it."""
    import json

    from nfl_predictor.models import quantile_registry

    frame = _training_frame()
    written = train_all(frame=frame, trained_seasons=[2017, 2018], out_dir=tmp_path,
                        manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                        **_passing_verdict())

    assert set(written) == set(MARKETS)
    for market in MARKETS:
        assert (tmp_path / f"{market}_quantile_2025.pkl").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert quantile_registry.verify_quantile_artifacts(manifest, out_dir=tmp_path) == []


def test_written_artifacts_carry_the_full_fitted_grid(tmp_path):
    """An artifact that stored only the tenths would reintroduce the flat-0.02
    tail at serving time, where nothing would notice."""
    from nfl_predictor.models.player_props import QUANTILES

    train_all(frame=_training_frame(), trained_seasons=[2017], out_dir=tmp_path,
              manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
              **_passing_verdict())

    payload = load_quantile_artifact("receiving_yards", out_dir=tmp_path)

    assert sorted(payload["quantile_models"]) == sorted(QUANTILES)
    assert payload["quantile_models"][QUANTILES[0]].get_params()["quantile_alpha"] == QUANTILES[0]


def test_written_artifacts_record_the_walkforward_verdict(tmp_path):
    train_all(frame=_training_frame(), trained_seasons=[2017, 2018], out_dir=tmp_path,
              manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
              **_passing_verdict())

    for market in MARKETS:
        payload = load_quantile_artifact(market, out_dir=tmp_path)
        assert "walkforward_mae" in payload
        assert "walkforward_calibration" in payload
        assert payload["walkforward_mae"] > 0, "an artifact with no MAE cannot be compared"


def test_train_all_refuses_to_write_without_a_walkforward_verdict(tmp_path):
    """The gate is what authorises the write. No verdict, no artifacts."""
    frame = _training_frame()

    with pytest.raises(ValueError, match="walkforward"):
        train_all(frame=frame, trained_seasons=[2017], out_dir=tmp_path,
                  manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                  walkforward_mae=None, walkforward_calibration=None)


def test_train_all_refuses_to_write_on_a_failed_calibration_gate(tmp_path):
    frame = _training_frame()
    failing = {"0.5-0.6": {"predicted": 0.6, "empirical": 0.4, "n": 500,
                           "gap": 0.2, "within_tolerance": False}}

    with pytest.raises(ValueError, match="calibration"):
        train_all(frame=frame, trained_seasons=[2017], out_dir=tmp_path,
                  manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                  walkforward_mae=20.0, walkforward_calibration=failing)


def test_train_all_is_additive_to_an_existing_manifest(tmp_path):
    from nfl_predictor.models import quantile_registry

    original = {"player_feature_cols": ["passing_yards_roll"], "sigma": 13.1,
                "quantile_yardage_v0": {"note": "an older generation"}}

    train_all(frame=_training_frame(), trained_seasons=[2017], out_dir=tmp_path,
              manifest=original, manifest_path=tmp_path / "manifest.json",
              **_passing_verdict())

    # train_all returns the artifacts; the manifest is what it wrote.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["player_feature_cols"] == ["passing_yards_roll"]
    assert manifest["sigma"] == 13.1
    assert manifest["quantile_yardage_v0"] == {"note": "an older generation"}
    assert quantile_registry.ARTIFACT_KEY in manifest


def test_train_all_does_not_write_a_production_pickle(tmp_path):
    train_all(frame=_training_frame(), trained_seasons=[2017], out_dir=tmp_path,
              manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
              **_passing_verdict())

    on_disk = {p.name for p in tmp_path.glob("*.pkl")}
    assert on_disk == {f"{m}_quantile_2025.pkl" for m in MARKETS}, (
        "the live site's pickles must never be a write target")


def _training_frame(with_features: bool = True):
    """A frame carrying every market, each on its own positions, as the real one
    is. A WR-only frame would make the passing and rushing artifacts unwritable,
    which is the correct behaviour but not what these tests are checking.

    `with_features=False` gives the bare frame, used to show that a frame
    without the declared feature set produces no artifact at all.
    """
    rng = np.random.default_rng(0)
    positions = {"WR": "receiving_yards", "TE": "receiving_yards",
                 "RB": "rushing_yards", "QB": "passing_yards"}
    rows = []
    for position, market in positions.items():
        for player in range(8):
            for week in range(1, 13):
                row = {
                    "player_id": f"{position}{player}", "player_name": f"{position}{player}",
                    "position": position, "season": 2017, "week": week,
                    "recent_team": "A", "opponent_team": "B",
                    "passing_yards": 0.0, "rushing_yards": 0.0, "receiving_yards": 0.0,
                    "targets": 5.0, "carries": 0.0, "receptions": 3.0,
                    "f1": float(rng.normal()), "f2": float(rng.normal()),
                }
                row[market] = float(40 + 5 * player + rng.normal(scale=10))
                rows.append(row)
    frame = pd.DataFrame(rows)
    if with_features:
        for column in FORWARD_FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = float(rng.normal())
    return frame


def test_train_all_fits_every_market_it_is_given(tmp_path):
    written = train_all(frame=_training_frame(), trained_seasons=[2017], out_dir=tmp_path,
                        manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                        **_passing_verdict())

    assert written["receiving_yards"]["feature_cols"]
    assert written["receiving_yards"]["trained_seasons"] == [2017]


def test_same_week_raw_stats_are_never_features(tmp_path):
    """The leak this exists to prevent.

    Deriving features as "every numeric column that is not the label" sweeps in
    passing_yards, targets, completions, attempts, fantasy_points -- outcomes of
    the very game being predicted. The offline gate is scored against the
    declared list, so it passes; an artifact fitted on the derived list is fitted
    on the answer and collapses at serving, where those columns do not exist."""
    frame = _training_frame()
    frame["targets"] = 7.0
    frame["completions"] = 20.0
    frame["attempts"] = 30.0
    frame["fantasy_points"] = 18.0
    frame["target_share"] = 0.3

    artifacts = train_all(frame=frame, trained_seasons=[2017], out_dir=tmp_path,
                          manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                          **_passing_verdict())

    forbidden = {"targets", "completions", "attempts", "fantasy_points", "target_share"}
    for market, payload in artifacts.items():
        assert not (forbidden & set(payload["feature_cols"])), (
            f"{market} was fitted on same-week outcomes")


def test_identifier_and_text_columns_are_never_features(tmp_path):
    frame = _training_frame()
    frame["headshot_url"] = "https://example.invalid/p.png"
    frame["stadium"] = "Soldier Field"
    frame["game_id"] = "G1"

    artifacts = train_all(frame=frame, trained_seasons=[2017], out_dir=tmp_path,
                          manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                          **_passing_verdict())

    for payload in artifacts.values():
        assert not ({"headshot_url", "stadium", "game_id"} & set(payload["feature_cols"]))


def test_a_market_is_never_fitted_on_its_own_raw_label(tmp_path):
    """The RAW same-week column is the answer, so it is excluded.

    The matching `_roll` is NOT: a QB's passing_yards_roll is his previous
    games' average, i.e. lagged history, and it is the most predictive feature the
    model has. Excluding it would be over-correcting; the leak is the unlagged
    column, and that is what the filter drops.
    """
    frame = _training_frame()

    artifacts = train_all(frame=frame, trained_seasons=[2017], out_dir=tmp_path,
                          manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                          **_passing_verdict())

    for market, payload in artifacts.items():
        assert market not in payload["feature_cols"], "a model cannot be fitted on its answer"
        assert f"{market}_roll" in payload["feature_cols"], (
            f"{market}'s lagged rolling mean is a legitimate feature for {market}")


def test_feature_columns_are_a_subset_of_the_declared_list(tmp_path):
    """The strongest form: nothing outside FORWARD_FEATURE_COLUMNS can get in."""
    frame = _training_frame()

    artifacts = train_all(frame=frame, trained_seasons=[2017], out_dir=tmp_path,
                          manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                          **_passing_verdict())

    for market, payload in artifacts.items():
        assert set(payload["feature_cols"]) <= set(FORWARD_FEATURE_COLUMNS), (
            f"{market} fitted on columns outside the declared, gate-validated set")


def test_rolling_features_are_used_when_present(tmp_path):
    """The lagged features ARE the model's basis, so they must survive the filter."""
    frame = _training_frame()

    artifacts = train_all(frame=frame, trained_seasons=[2017], out_dir=tmp_path,
                          manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                          **_passing_verdict())

    assert "receiving_yards_roll" in artifacts["receiving_yards"]["feature_cols"]


def test_a_frame_without_the_declared_features_writes_nothing(tmp_path):
    """The leakage filter can starve a model, and that must show up as no
    artifact rather than as one fitted on whatever numeric columns remain."""
    artifacts = train_all(frame=_training_frame(with_features=False),
                          trained_seasons=[2017], out_dir=tmp_path,
                          manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                          **_passing_verdict())

    assert artifacts == {}
    assert list(tmp_path.glob("*.pkl")) == []



# --- the gate and the artifact must fit the SAME features -------------------

def test_the_gate_and_the_artifacts_share_one_feature_list():
    """The drift this pins: the gate validated 28 features while the artifacts
    were fitted on 30, and nothing compared the two lists.

    Asserted against the COMMITTED manifest rather than by importing the script
    -- `tests/test_runtime_dependencies.py` treats any unlisted import as a
    third-party distribution, and re-declaring the list in a test would only
    create a third copy to drift. The manifest is what a fresh clone actually
    loads, so it is the better thing to check.
    """
    from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS

    manifest_path = Path(__file__).resolve().parents[2] / "models" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("no committed manifest")
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get("quantile_yardage_v1")
    if not entry:
        pytest.skip("quantile artifacts not registered yet")

    assert set(entry["feature_cols"]) <= set(FORWARD_FEATURE_COLUMNS), (
        "the registered artifact carries features outside the shared list")

    for market, meta in entry["markets"].items():
        assert meta["n_features"] == len(FORWARD_FEATURE_COLUMNS) - (1 if market in FORWARD_FEATURE_COLUMNS else 0), (
            f"{market} was fitted on a different feature count than the gate validates")


def test_closing_line_features_are_excluded():
    """nflverse's weekly spread_line/total_line are CLOSING lines, and
    implied_team_total/game_total are derived from them. A pre-kickoff snapshot
    has the opening line at best, so carrying these fits the model on
    information it will not have when it matters -- and it is the number the
    offline calibration is scored by, so the skew flatters the gate."""
    from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS

    closing = {"spread_line", "total_line", "implied_team_total", "game_total"}

    assert not (closing & set(FORWARD_FEATURE_COLUMNS))


def test_train_all_refuses_on_an_empty_calibration_report(tmp_path):
    """Zero buckets means no bucket reached the n>=100 floor: no evidence at all.
    `calibration_passes` treats that as not-a-pass and train_all must agree, or a
    run that measured nothing writes artifacts as though it had."""
    with pytest.raises(ValueError, match="no calibration bucket"):
        train_all(frame=_training_frame(), trained_seasons=[2017], out_dir=tmp_path,
                  manifest=_manifest(), manifest_path=tmp_path / "manifest.json",
                  walkforward_mae=20.0, walkforward_calibration={})

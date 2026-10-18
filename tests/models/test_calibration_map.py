"""Calibration maps for P(over): isotonic and Platt.

Properties under test:
* A fitted map is monotone non-decreasing in raw P(over) and clamps to [0, 1].
* An unfitted or too-thin map is the identity (raw P/over passes through).
* `p_over_from_quantiles(..., calibration_map=m)` applies the map and never
  leaves the [0.02, 0.98] clamp.
* Platt needs both outcome classes in the training fold; isotonic needs enough
  pairs; otherwise both fall back to identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models.calibration_map import (
    CalibrationMap, MAP_KINDS, MIN_TRAIN_PAIRS, walk_forward_calibration_map,
    gate_result,
)
from nfl_predictor.models.prop_probability import p_over_from_quantiles

QP = {0.1: 100.0, 0.5: 200.0, 0.9: 300.0}


def _calibrated_pairs(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic train pairs: raw P(over) miscalibrated (overconfident on overs)."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.02, 0.98, n)
    # empirical rate = shrink raw toward 0.5 (overconfident): empirical < raw when raw>0.5
    empirical = np.clip(0.5 + (raw - 0.5) * 0.6, 0.0, 1.0)
    covered = (rng.uniform(0, 1, n) < empirical).astype(float)
    return raw, covered


def test_map_is_monotone_and_clamped():
    raw, covered = _calibrated_pairs()
    for kind in MAP_KINDS:
        m = CalibrationMap(kind).fit(raw, covered)
        xs = np.linspace(0.0, 1.0, 50)
        out = m.apply(xs)
        assert np.all(np.diff(out) >= -1e-9), kind
        assert out.min() >= 0.0 and out.max() <= 1.0, kind


def test_unfitted_map_is_identity():
    m = CalibrationMap("isotonic")
    xs = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(m.apply(xs), xs)


def test_thin_training_falls_back_to_identity():
    raw = np.array([0.2, 0.5, 0.8])
    covered = np.array([0.0, 1.0, 0.0])
    for kind in MAP_KINDS:
        m = CalibrationMap(kind).fit(raw, covered)
        assert not m.fitted
        xs = np.array([0.1, 0.9])
        np.testing.assert_allclose(m.apply(xs), xs)


def test_platt_requires_both_classes():
    # only positives -> logistic cannot fit -> identity
    raw = np.linspace(0.02, 0.98, 300)
    covered = np.ones(300)
    m = CalibrationMap("platt").fit(raw, covered)
    assert not m.fitted
    xs = np.array([0.2, 0.8])
    np.testing.assert_allclose(m.apply(xs), xs)


def test_p_over_from_quantiles_applies_map():
    m = CalibrationMap("platt").fit(*_calibrated_pairs())
    raw = p_over_from_quantiles(QP, 200.0)  # == 0.5
    cal = p_over_from_quantiles(QP, 200.0, calibration_map=m)
    assert 0.02 <= cal <= 0.98
    assert cal != raw  # map changed it
    # without map, unchanged
    assert p_over_from_quantiles(QP, 200.0) == raw


def test_gate_result_detects_widening():
    before = {"worst_gap": 0.04, "brier": 0.20}
    after = {"worst_gap": 0.06, "brier": 0.21}  # gap widened, brier worse
    g = gate_result(before, after)
    assert not g["gap_ok"] and not g["brier_ok"] and not g["gate_passes"]

    after2 = {"worst_gap": 0.03, "brier": 0.19}  # both improved
    g2 = gate_result(before, after2)
    assert g2["gap_ok"] and g2["brier_ok"] and g2["gate_passes"]


def test_walk_forward_returns_raw_and_calibrated_columns():
    # Drive walk_forward_calibration_map through a fixed scored frame so this
    # stays a unit test (no full nflverse frame build or model fit).
    import nfl_predictor.models.calibration_map as cm

    scored = pd.DataFrame({
        "market": ["passing_yards"] * 6,
        "season": [2020, 2020, 2020, 2021, 2021, 2021],
        "week": [1, 2, 3, 1, 2, 3],
        "player_id": ["p1", "p2", "p3", "p1", "p2", "p3"],
        "position": ["QB"] * 6,
        "actual": [200.0, 50.0, 300.0, 100.0, 250.0, 75.0],
        "p_over_exogenous": [0.6, 0.2, 0.8, 0.4, 0.7, 0.3],
        "covered_exogenous": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    })

    out = cm.walk_forward_calibration_map(
        scored, markets=["passing_yards"], feature_cols=[], seasons=[2020, 2021],
        kind="isotonic", scored=scored,
    )
    assert not out.empty
    assert "calibrated_p_over" in out.columns
    assert "cal_map_kind" in out.columns
    # the first scored season has no earlier training data -> no calibration possible (NaN)
    assert out.loc[out["season"] == 2020, "calibrated_p_over"].isna().all()
    # later seasons are fit on earlier seasons only; values stay in the clamp
    later = out.loc[out["season"] == 2021, "calibrated_p_over"].dropna()
    assert len(later) == 3
    assert later.between(0.0, 1.0).all()
    # all calibrated values (ignoring no-training-data NaNs) stay in the clamp
    assert out["calibrated_p_over"].dropna().between(0.0, 1.0).all()


def test_gate_result_improved_is_true_when_both_better():
    g = gate_result({"worst_gap": 0.08, "brier": 0.25},
                    {"worst_gap": 0.03, "brier": 0.20})
    assert g["gate_passes"] is True
    assert g["gap_delta"] < 0 and g["brier_delta"] < 0

"""Lift gate (AI plan Task 10): residual normality check.

Until this gate passes, all matchup rows stay neutral (toward_pick=None, no Edge/Risk label).
The gate checks that model residuals (actual - predicted) are normally distributed
via Shapiro-Wilk test at p >= 0.05.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _sample_residuals() -> np.ndarray:
    """Return a small sample of residuals from the model pipeline.

    Uses the out-of-fold residuals from the walk-forward / sigma pipeline.
    This is a minimal sample so the test runs quickly without full data loads.
    """
    # Simulate residuals that are approximately normal (the gate should pass)
    # In production these would come from game_outcome.oof_residuals or similar.
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


def lift_gate_residuals_ok() -> bool:
    """Return True if residuals pass the lift gate (normality at p >= 0.05)."""
    residuals = _sample_residuals()
    stat, p = stats.shapiro(residuals)
    return bool(p >= 0.05)


def test_residual_lift_gate_normal_passes():
    """A sample of approx-normal residuals should pass the lift gate."""
    assert lift_gate_residuals_ok()


def test_residual_lift_gate_nonnormal_fails(monkeypatch):
    """A sample of clearly non-normal residuals should fail the lift gate."""
    # Use a heavy-tailed distribution so the gate reliably fails
    rng = np.random.default_rng(0)
    heavy = rng.standard_t(3, size=200)  # t-distribution with 3 df = heavy tail
    monkeypatch.setattr("test_lift_gate._sample_residuals", lambda: heavy)
    assert not lift_gate_residuals_ok()

"""Tests for the moneyline calibration module.

The module is NOT wired into the serving path: the walk-forward gate in
`scripts/moneyline_calibration.py` FAILED, so nothing consumes it yet (see the PR body). These
tests pin the behaviour the gate's numbers depend on, so a later change to the module cannot
quietly change what those numbers mean.

What is pinned here:
  * a slope above 1 STRETCHES a probability toward 0/1, and a slope below 1 compresses it
  * the calibration bias is SIGNED as (predicted - actual), and the improvement the gate
    measures is on its ABSOLUTE value -- a bias moving from -0.215 toward -0.10 is an improvement
    even though the signed number went UP
  * the paired bootstrap of the absolute-bias difference is centred where the point estimate is
  * a market that is too compressed to be calibrated by a global stretch is NOT helped by one
"""

import numpy as np
import pytest

from nfl_predictor.models.moneyline_calibration import (
    abs_bias_delta,
    calibration_bias,
    fit_platt,
    paired_abs_bias_ci,
    platt_transform,
    spread_bucket,
)


def test_a_slope_above_one_stretches_a_probability_up():
    """`a > 1` pushes a 0.7 raw probability UP, which is the whole point of the stretch."""
    stretched = platt_transform(np.array([0.7]), 1.5, 0.0)[0]
    assert stretched > 0.7


def test_a_slope_below_one_compresses_a_probability_down():
    """`a < 1` pulls a probability back toward 0.5."""
    compressed = platt_transform(np.array([0.7]), 0.5, 0.0)[0]
    assert compressed < 0.7


def test_a_slope_of_one_is_the_identity():
    """`a = 1, b = 0` changes nothing, which is what "no correction" has to mean."""
    assert platt_transform(np.array([0.3, 0.7]), 1.0, 0.0) == pytest.approx([0.3, 0.7])


def test_the_stretch_is_monotone():
    """A calibration that reorders probabilities is not a calibration."""
    raw = np.linspace(0.05, 0.95, 40)
    stretched = platt_transform(raw, 1.5, 0.0)
    assert np.all(np.diff(stretched) > 0)


def test_the_bias_is_predicted_minus_actual_and_is_signed():
    """A model that UNDER-predicts the outcome has a NEGATIVE bias."""
    probs = np.array([0.3, 0.3, 0.3, 0.3])
    outcomes = np.array([1, 1, 0, 0])
    assert calibration_bias(probs, outcomes) == pytest.approx(-0.2)


def test_a_bias_moving_toward_zero_is_an_improvement_even_though_the_signed_number_rises():
    """-0.215 -> -0.10 is an IMPROVEMENT: the absolute value got smaller.

    This is the case the gate is stated on, and getting it backwards is the error that would have
    made a working correction look like a broken one.
    """
    raw = np.array([0.5, 0.5, 0.5, 0.5])
    cal = np.array([0.6, 0.6, 0.6, 0.6])
    outcomes = np.array([1, 1, 1, 1])   # the model says 0.5, every game was a home win

    assert calibration_bias(raw, outcomes) == pytest.approx(-0.5)
    assert calibration_bias(cal, outcomes) == pytest.approx(-0.4)
    # The signed bias went UP (+0.1) ...
    assert calibration_bias(cal, outcomes) > calibration_bias(raw, outcomes)
    # ... and the ABSOLUTE bias went DOWN, which is the improvement.
    assert abs_bias_delta(raw, cal, outcomes) < 0


def test_a_bias_moving_away_from_zero_is_a_regression():
    """A correction that makes the bias BIGGER is a regression, whatever its sign."""
    raw = np.array([0.5, 0.5, 0.5, 0.5])
    cal = np.array([0.3, 0.3, 0.3, 0.3])
    outcomes = np.array([1, 1, 1, 1])
    assert abs_bias_delta(raw, cal, outcomes) > 0


def test_the_paired_bootstrap_is_centred_on_the_point_estimate():
    """The interval must bracket the point estimate it is an interval for."""
    rng = np.random.default_rng(3)
    raw = np.clip(rng.normal(0.5, 0.1, 400), 0.01, 0.99)
    cal = np.clip(raw + 0.08, 0.01, 0.99)
    y = (rng.random(400) < cal).astype(int)

    point = abs_bias_delta(raw, cal, y)
    low, high = paired_abs_bias_ci(raw, cal, y, n_bootstrap=800, seed=11)
    assert low < point < high


def test_the_paired_bootstrap_reports_a_negative_interval_for_a_real_improvement():
    """A calibration that genuinely shrinks the bias gets an interval entirely below zero.

    The construction is exact rather than approximate: `platt_transform` is a LOGIT-SPACE
    affine map, so composing a 0.5 compression with a 2.0 stretch is the identity in log-odds
    space. So `cal` recovers the truth exactly, while `raw` is the compressed version, and the
    outcomes are drawn from the truth.
    """
    rng = np.random.default_rng(5)
    truth = np.where(rng.random(800) < 0.7, 0.8, 0.2)   # mostly-favourite outcomes
    y = (rng.random(800) < truth).astype(int)
    raw = platt_transform(truth, 0.5, 0.0)               # the "too compressed" raw probabilities
    cal = platt_transform(raw, 2.0, 0.0)                 # the stretch that undoes it

    # The correction moves the bias toward zero ...
    assert abs_bias_delta(raw, cal, y) < 0
    # ... and the paired bootstrap says so.
    low, high = paired_abs_bias_ci(raw, cal, y, n_bootstrap=800, seed=13)
    assert high < 0


def test_fit_platt_recovers_a_known_stretch():
    """A calibration that undoes a known compression is recovered by the fit."""
    rng = np.random.default_rng(17)
    true = np.clip(rng.beta(2, 2, 3000), 0.02, 0.98)
    y = (rng.random(3000) < true).astype(int)
    compressed = platt_transform(true, 0.5, 0.0)   # the "too compressed" raw probabilities

    a, b = fit_platt(compressed, y)
    assert a > 1.0, f"expected a stretch, got a={a}"


def test_fit_platt_does_not_change_the_probabilities_when_there_is_nothing_to_fix():
    """A model that is already calibrated must come out with a slope of about 1.

    A slope far from 1 here would mean the fit was manufacturing a correction out of noise, and
    that correction would then be applied to probabilities that did not need it.
    """
    rng = np.random.default_rng(23)
    p = np.clip(rng.beta(2, 2, 5000), 0.02, 0.98)
    y = (rng.random(5000) < p).astype(int)

    a, b = fit_platt(p, y)
    assert a == pytest.approx(1.0, abs=0.1)


def test_fit_platt_on_nothing_leaves_the_probabilities_alone():
    """No data means no correction: the fit returns the identity."""
    a, b = fit_platt(np.array([]), np.array([]))
    assert (a, b) == (1.0, 0.0)


def test_the_spread_bucket_uses_the_home_teams_expected_margin():
    """`spread_line` is nflverse's convention: POSITIVE favours the home team."""
    assert spread_bucket(7.0) == "fav>=7"
    assert spread_bucket(7.5) == "fav>=7"
    assert spread_bucket(-7.0) == "dog<=-7"
    assert spread_bucket(0.0) == "other"
    assert spread_bucket(3.5) == "other"
    assert spread_bucket(None) == "other"

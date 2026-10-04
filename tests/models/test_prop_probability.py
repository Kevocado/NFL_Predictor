"""P(over | line) and edge maths.

The interpolation is the whole deliverable: it turns nine fitted quantiles into
the one number a bet needs. Two properties carry the weight -- P(over) must be
monotone decreasing in the line, and quantiles that cross (q90 below q50, which
independent GBM fits do produce) must be repaired rather than trusted.
"""
from __future__ import annotations

import pytest

from nfl_predictor.models.prop_probability import (
    american_to_breakeven, edge_vs_line, p_over_from_quantiles,
)

QP = {0.1: 180.0, 0.5: 225.0, 0.9: 275.0}


def test_p_over_interpolation_and_clamps():
    assert p_over_from_quantiles(QP, 225.0) == pytest.approx(0.5)
    # Exactly at the fitted q10, the model says 10% of outcomes fall below, so
    # P(over) is 0.9 -- not the far-below clamp. Equality has to interpolate.
    assert p_over_from_quantiles(QP, 180.0) == pytest.approx(0.9)
    assert p_over_from_quantiles(QP, 100.0) == 0.98
    assert p_over_from_quantiles(QP, 400.0) == 0.02


def test_p_over_is_monotone_decreasing_in_the_line():
    lines = [150, 180, 200, 225, 250, 275, 300]
    probabilities = [p_over_from_quantiles(QP, line) for line in lines]

    assert probabilities == sorted(probabilities, reverse=True), probabilities


def test_interpolates_between_adjacent_fitted_quantiles():
    # halfway between q10=180 and q50=225 is 202.5, so alpha=0.3, P=0.7
    assert p_over_from_quantiles(QP, 202.5) == pytest.approx(0.7)


def test_crossing_quantiles_are_repaired_not_fatal():
    crossed = {0.1: 250.0, 0.5: 225.0, 0.9: 275.0}

    # Repair is forward-max, so the repaired floor is 250, not 225. A 230 line
    # therefore sits below the whole distribution and takes the ceiling.
    assert p_over_from_quantiles(crossed, 230.0) == 0.98

    # Inside the repaired range the answer interpolates off the repaired values:
    # between (q0.5, 250) and (q0.9, 275), alpha = 0.5 + 0.4 * (260-250)/25.
    assert p_over_from_quantiles(crossed, 260.0) == pytest.approx(0.34)


def test_crossing_does_not_break_monotonicity():
    crossed = {0.1: 250.0, 0.2: 240.0, 0.3: 100.0, 0.9: 275.0}
    lines = [90, 120, 200, 240, 260, 300]
    probabilities = [p_over_from_quantiles(crossed, line) for line in lines]

    assert probabilities == sorted(probabilities, reverse=True), probabilities


def test_flat_segment_does_not_divide_by_zero():
    flat = {0.1: 200.0, 0.5: 200.0, 0.9: 200.0}
    value = p_over_from_quantiles(flat, 200.0)

    assert 0.02 <= value <= 0.98


def test_single_quantile_does_not_crash():
    assert 0.02 <= p_over_from_quantiles({0.5: 100.0}, 100.0) <= 0.98


def test_probability_is_always_within_the_clamp():
    for line in [-50, 0, 50, 1000]:
        assert 0.02 <= p_over_from_quantiles(QP, line) <= 0.98


def test_breakeven_math():
    assert american_to_breakeven(-110) == pytest.approx(110 / 210)
    assert american_to_breakeven(100) == pytest.approx(0.5)
    assert edge_vs_line(0.60, -110) == pytest.approx(0.60 - 110 / 210)


def test_breakeven_of_minus_110_is_the_specs_52_4_percent():
    """The spec's forward gate is stated against 52.4% breakeven."""
    assert american_to_breakeven(-110) == pytest.approx(0.524, abs=0.0005)


def test_edge_is_the_number_the_five_percent_gate_compares():
    # 5% edge at -110 needs p_over >= 0.524 + 0.05
    assert edge_vs_line(0.574, -110) >= 0.05
    assert edge_vs_line(0.573, -110) < 0.05
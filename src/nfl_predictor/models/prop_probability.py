"""prop_probability.py -- turn quantile predictions into P(over | line).

This is the deliverable's centre: the models produce nine conditional quantiles,
and a bet needs one number. Interpolation is linear in quantile space between
fitted levels, with no normality assumption and no residual-variance fudge.
"""
from __future__ import annotations

#: Probabilities are clamped here rather than at 0/1. A line below the lowest
#: fitted quantile is genuinely "very likely over", but the fitted quantiles stop
#: at q0.1, so anything beyond is extrapolation. 0.98 is the edge gate's budget:
#: it can still clear 5% against a 52.4% breakeven and never claims certainty.
P_OVER_FLOOR, P_OVER_CEILING = 0.02, 0.98


def p_over_from_quantiles(quantile_preds: dict[float, float], line: float) -> float:
    """P(actual yardage > line), interpolated from predicted quantiles.

    Crossing quantiles -- q90 below q50, which independent GBM fits really do
    produce -- are repaired to be non-decreasing first. Left unrepaired they
    invert the answer.
    """
    levels = sorted(quantile_preds)
    preds = [float(quantile_preds[q]) for q in levels]
    for i in range(1, len(preds)):
        if preds[i] < preds[i - 1]:
            preds[i] = preds[i - 1]

    # Strict, not inclusive: a line sitting exactly on q0.1 has 10% of outcomes
    # below it, so P(over) is 0.9 and must interpolate rather than clamp.
    if line < preds[0]:
        return P_OVER_CEILING
    if line > preds[-1]:
        return P_OVER_FLOOR

    for (low_level, low_value), (high_level, high_value) in zip(
            zip(levels, preds), zip(levels[1:], preds[1:])):
        if low_value <= line <= high_value:
            if high_value == low_value:
                alpha = low_level
            else:
                share = (line - low_value) / (high_value - low_value)
                alpha = low_level + (high_level - low_level) * share
            return min(P_OVER_CEILING, max(P_OVER_FLOOR, 1.0 - alpha))

    return 0.5  # unreachable given the bounds checks above; defensive


def american_to_breakeven(odds: float) -> float:
    """Win probability at which a bet on these American odds breaks even."""
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)


def edge_vs_line(p_over: float, odds: float) -> float:
    """Model probability minus breakeven. The forward test's 5% gate reads this."""
    return p_over - american_to_breakeven(odds)
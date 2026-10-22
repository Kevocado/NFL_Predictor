"""moneyline_calibration.py — Platt scaling for the moneyline (home_win_prob) probability.

**Moneyline only.** The spread (cover) and total (over/under) probabilities are produced by the
same `margin_to_probabilities` call, and they are ALREADY WELL CALIBRATED — a calibration on them
would be a change to code that is not broken. So the correction is applied to `home_win_prob` /
`away_win_prob` ONLY, and the raw probability is kept beside the corrected one so every published
number can be traced back to what the model actually said.

**What the calibration is.** The raw probability comes from a Normal tail on the predicted margin,
and a Normal tail is narrower than the real distribution of game outcomes. That makes the raw
probabilities COMPRESSED: 0.70 comes out more often than it should, and 0.90 comes out more often
than it should. Platt scaling is the standard fix — a one-parameter-per-degree-of-freedom logistic
stretch on the log-odds:

    p_corrected = sigmoid(a * logit(p_raw) + b)

`a > 1` STRETCHES the probabilities toward 0 and 1 (a 0.70 raw probability becomes higher); `a < 1`
compresses them. `b` shifts the whole curve. Both are fitted by minimising the negative
log-likelihood, which is the same proper scoring rule the model is graded on.

**The fit is WALK-FORWARD.** `a` and `b` are fitted on seasons STRICTLY BEFORE the season being
predicted, and the correction is then applied to that season. Nothing from the test season reaches
the fit. See `scripts/moneyline_calibration.py` for the walk-forward harness and the ship gate.

**The bias is signed, and the sign is the whole question.** `calibration_bias` is
`mean(predicted) - mean(actual)`, so a NEGATIVE bias means the model UNDER-predicts the outcome
(the favourites win more often than the model said). Whether a correction IMPROVES the bias is
therefore about the ABSOLUTE value, not the signed one: a bias moving from -0.215 to -0.10 is an
improvement even though the signed number went UP. The ship gate is on
`|bias_corrected| - |bias_raw| < 0`, and `paired_abs_bias_ci` is the paired bootstrap of exactly
that difference, resampling GAMES within the season.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

#: The smallest and largest slope the fit will consider. A slope outside this range is not a
#: calibration, it is a different model wearing the calibration's name: `a` near 0 collapses every
#: probability to 0.5, and a huge `a` is a step function.
MIN_SLOPE = 0.1
MAX_SLOPE = 10.0

#: The intercept's range, for the same reason.
MIN_INTERCEPT = -5.0
MAX_INTERCEPT = 5.0

#: Probabilities are clipped to this band before the logit, so a 0 or 1 never produces an infinity.
PROB_EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), PROB_EPS, 1.0 - PROB_EPS)


def platt_transform(p, a: float, b: float):
    """Apply the Platt stretch to raw probabilities: `sigmoid(a * logit(p) + b)`."""
    p = _clip(p)
    return 1.0 / (1.0 + np.exp(-(a * np.log(p / (1.0 - p)) + b)))


def fit_platt(p_raw, y) -> tuple[float, float]:
    """Fit the Platt slope and intercept by minimising the negative log-likelihood.

    `p_raw` are the model's raw probabilities and `y` the binary outcomes (1 = home won). Returns
    `(a, b)`. A fit that does not converge returns the identity `(1.0, 0.0)`, which leaves the
    probabilities exactly as they were — a calibration that cannot be fitted must not change
    anything, and the gate catches the case where that is hiding a real problem.
    """
    p = _clip(p_raw)
    y = np.asarray(y, dtype=float)
    if len(p) == 0 or len(p) != len(y):
        return 1.0, 0.0
    logit_p = np.log(p / (1.0 - p))

    def nll(params):
        a, b = params
        p_cal = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
        p_cal = np.clip(p_cal, 1e-15, 1.0 - 1e-15)
        return -np.sum(y * np.log(p_cal) + (1.0 - y) * np.log(1.0 - p_cal))

    result = minimize(
        nll,
        x0=[1.0, 0.0],
        method="L-BFGS-B",
        bounds=[(MIN_SLOPE, MAX_SLOPE), (MIN_INTERCEPT, MAX_INTERCEPT)],
    )
    if not result.success:
        return 1.0, 0.0
    return float(result.x[0]), float(result.x[1])


def calibration_bias(probs, y) -> float:
    """The signed calibration bias: `mean(predicted) - mean(actual)`.

    NEGATIVE means the model under-predicts the outcome (favourites win more often than the model
    said). The bias is what the ship gate measures, and because it is SIGNED, an improvement is a
    movement TOWARD ZERO — which is why the gate is stated on the absolute value.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(probs) == 0:
        return float("nan")
    return float(np.mean(probs) - np.mean(y))


def abs_bias_delta(p_raw, p_cal, y) -> float:
    """`|bias_corrected| - |bias_raw|`: negative means the calibration made the bias SMALLER.

    This is the quantity the ship gate is stated on, and it is what `paired_abs_bias_ci` bootstraps.
    """
    before = abs(calibration_bias(p_raw, y))
    after = abs(calibration_bias(p_cal, y))
    return float(after - before)


def paired_abs_bias_ci(p_raw, p_cal, y, n_bootstrap: int = 2000, seed: int = 20261022):
    """Paired bootstrap CI for `|bias_corrected| - |bias_raw|`, resampling GAMES within the sample.

    The pairing is what makes this a bootstrap of a DIFFERENCE rather than of two independent
    quantities: the same resampled games are scored before and after, so the sampling noise of the
    two cancels instead of being added. The CI is entirely below zero when the correction reliably
    shrinks the bias.

    Returns `(lower, upper)`.
    """
    p_raw = np.asarray(p_raw, dtype=float)
    p_cal = np.asarray(p_cal, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        deltas[i] = abs_bias_delta(p_raw[idx], p_cal[idx], y[idx])

    lower = float(np.percentile(deltas, 2.5))
    upper = float(np.percentile(deltas, 97.5))
    return (lower, upper)


def spread_bucket(spread_line, favourite_points: float = 7.0) -> str:
    """Which bucket a game falls in: a heavy favourite, a heavy underdog, or neither.

    `spread_line` is the HOME team's expected margin (nflverse convention: positive favours home),
    so `>= favourite_points` means the home team is the heavy favourite.
    """
    if spread_line is None:
        return "other"
    try:
        value = float(spread_line)
    except (TypeError, ValueError):
        return "other"
    if value >= favourite_points:
        return "fav>=7"
    if value <= -favourite_points:
        return "dog<=-7"
    return "other"

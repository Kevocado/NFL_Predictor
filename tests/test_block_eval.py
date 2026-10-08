import numpy as np
from nfl_predictor.tools.block_eval import paired_bootstrap, calibration_gap


def test_bootstrap_interval_excludes_zero_for_a_real_gain():
    rng = np.random.default_rng(1)
    base = rng.normal(10, 2, 800)
    better = base - 0.8 + rng.normal(0, 0.5, 800)
    lo, hi = paired_bootstrap(base, better)
    assert lo > 0


def test_bootstrap_interval_spans_zero_for_noise():
    rng = np.random.default_rng(2)
    a = rng.normal(10, 2, 800)
    lo, hi = paired_bootstrap(a, a + rng.normal(0, 0.5, 800))
    assert lo < 0 < hi


def test_calibration_gap_is_zero_for_perfect_buckets():
    p = np.repeat([0.1, 0.3, 0.5, 0.7, 0.9], 1000)
    y = np.concatenate([np.r_[np.ones(int(1000 * q)), np.zeros(1000 - int(1000 * q))] for q in (0.1, 0.3, 0.5, 0.7, 0.9)])
    assert calibration_gap(p, y) < 1e-9
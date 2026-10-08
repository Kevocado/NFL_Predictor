import numpy as np
import pandas as pd
from nfl_predictor.tools.block_eval import paired_bootstrap, calibration_gap, evaluate_block
from nfl_predictor.features import build
from epa_fixtures import make_games, make_pbp
from nfl_predictor.data import pbp_agg


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


def test_evaluate_block_runs_on_epa_fixtures():
    """evaluate_block runs the full walk-forward path with EPA block on fixture data."""
    games = make_games(seasons=(2023, 2024, 2025), weeks=6)
    pbp = make_pbp(games)
    aux = build.Aux(pbp_agg.team_game_efficiency(pbp), pbp_agg.qb_games(pbp))
    result = evaluate_block(games, aux, "epa", candidate="ridge")
    assert "mae_delta" in result and "brier_delta" in result
    assert "mae_ci" in result and "brier_ci" in result
    assert "gap_base" in result and "gap_block" in result
    assert "clears" in result
    assert isinstance(result["clears"], bool)
    assert result["n_games"] > 0
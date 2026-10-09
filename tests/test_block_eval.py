import numpy as np
import pandas as pd
from nfl_predictor.tools.block_eval import paired_bootstrap, calibration_gap, evaluate_block
from nfl_predictor.features import build
from epa_fixtures import make_games, make_pbp
from nfl_predictor.data import pbp_agg
from nfl_predictor.evaluate import walk_forward
from nfl_predictor.tools import block_eval


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

# --- added by the block-eval results PR ---------------------------------------

from nfl_predictor.tools.block_eval import (  # noqa: E402
    paired_bootstrap_gap, _per_fold_gap, _per_game_totals, load_recent_nfl_data, main as block_eval_main,
)
from nfl_predictor.tools import qb_agreement  # noqa: E402


def test_pandas_is_imported_so_load_recent_nfl_data_can_concat():
    """Regression: the module referenced pd.* with pandas never imported, so every
    run of the CLI died with NameError after loading all the fixtures."""
    assert block_eval.pd is pd


def test_gap_bootstrap_positive_when_the_block_narrows_the_gap():
    base = [0.050, 0.055, 0.060, 0.058, 0.062]
    blk = [0.020, 0.019, 0.021, 0.022, 0.018]
    lo, hi = paired_bootstrap_gap(base, blk)
    assert lo > 0, f"a real narrowing must have a CI above zero, got {lo}"


def test_gap_bootstrap_spans_zero_when_the_two_gaps_are_indistinguishable():
    """0.0287 vs 0.0288 is noise, not an improvement, and the interval has to say so."""
    base = [0.0287, 0.0288, 0.0286, 0.0289, 0.0285]
    blk = [0.0288, 0.0287, 0.0289, 0.0286, 0.0290]
    lo, hi = paired_bootstrap_gap(base, blk)
    assert lo < 0 < hi, f"an indistinguishable pair must straddle zero, got {lo}, {hi}"


def test_gap_bootstrap_handles_no_folds():
    assert paired_bootstrap_gap([], []) == (0.0, 0.0)


def _games_with_weather(seasons=(2023, 2024, 2025), weeks=6):
    """make_games plus the roof/temp/wind the conditions block reads.

    The shared epa fixture is intentionally weather-free, and the conditions block
    raises KeyError on `roof` rather than defaulting it, so this test supplies it
    instead of widening the shared fixture for everyone.
    """
    games = make_games(seasons=seasons, weeks=weeks)
    rng = np.random.default_rng(7)
    games = games.copy()
    games["roof"] = rng.choice(["outdoors", "dome"], size=len(games))
    games["temp"] = rng.integers(8, 95, size=len(games)).astype(float)
    games["wind"] = rng.integers(0, 22, size=len(games)).astype(float)
    return games


def test_evaluate_block_reports_gap_interval_and_total_mae_for_conditions():
    """conditions is claimed for TOTALS, so the result carries total MAE."""
    games = _games_with_weather()
    pbp = make_pbp(games)
    aux = build.Aux(pbp_agg.team_game_efficiency(pbp), pbp_agg.qb_games(pbp))
    result = evaluate_block(games, aux, "conditions", candidate="ridge")
    assert result["gap_ci"] is not None
    assert result["total_mae"] is not None
    assert set(result["total_mae"]) >= {"delta", "ci", "clears"}
    assert isinstance(result["total_mae"]["clears"], bool)


def test_per_game_totals_matches_error_length():
    games = make_games(seasons=(2023, 2024, 2025), weeks=6)
    folds = walk_forward.prepare_folds(games)
    errs = _per_game_totals(folds, "ridge")
    assert errs.size > 0
    assert (errs >= 0).all(), "abs error is never negative"


def test_qb_agreement_module_exposes_the_expected_and_actual_starter_paths():
    for fn in ("agreement", "get_expected_starters", "get_actual_starters",
               "expected_starters_serving_view", "main"):
        assert hasattr(qb_agreement, fn), f"qb_agreement is missing {fn}"

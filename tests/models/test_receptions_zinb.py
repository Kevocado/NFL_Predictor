"""The receptions ZINB (models/receptions_zinb.py): gate arithmetic, a fit/predict smoke test on synthetic
data with known parameters, fit/serve parity, the serving guardrail and a walk-forward smoke test.

The module shipped with no tests. What is pinned here, and why:

* The GATE arithmetic -- `train_all` refuses to write the receptions artifact without a passing verdict, so the
  verdict's arithmetic (same 10 bins / tol 0.05 / n>=100 as the quantile gate) must be exactly right in both
  directions: a calibrated model passes, a miscalibrated one fails, and "not enough data" is a stop, not a pass.
* FIT/SERVE PARITY -- features are derived in one function (`_derive_features`) used by both; a missing roll column
  raises instead of being filled with zeros, and the winsorization bounds learned at fit time are applied at serving.
* The tail probabilities -- zeros are always "under", and over + under == 1 because a half-point line cannot push.
* The guardrail -- `mu > 7.0` is flagged `high_mu`.

Offline, synthetic data only.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.evaluate.walk_forward import calibration_passes
from nfl_predictor.models import receptions_zinb as rz
from nfl_predictor.models.training import _receptions_gate_passes

RNG_SEED = 7


def _frame(n: int = 3000, seasons=(2017, 2018, 2019, 2020, 2021, 2022, 2023), seed: int = RNG_SEED) -> pd.DataFrame:
    """Synthetic WR/TE/RB history drawn FROM a zero-inflated negative binomial, so the fit has a truth to find.

    mu_nb rises with targets; a structural zero (the fringe player who dresses but is not targeted) is more likely
    for low-target rows -- the shape the real data has."""
    rng = np.random.default_rng(seed)
    position = rng.choice(["WR", "TE", "RB"], size=n, p=[0.5, 0.25, 0.25])
    targets_roll = np.where(position == "RB", rng.uniform(0.5, 5.0, n), rng.uniform(1.0, 10.0, n))
    receptions_roll = 0.65 * targets_roll * rng.uniform(0.85, 1.15, n)
    carries_roll = np.where(position == "RB", rng.uniform(4.0, 18.0, n), rng.uniform(0.0, 1.0, n))
    df = pd.DataFrame({
        "player_id": rng.integers(0, 400, n).astype(str),
        "season": rng.choice(list(seasons), n),
        "week": rng.integers(1, 19, n),
        "position": position,
        "targets_roll": targets_roll,
        "receptions_roll": receptions_roll,
        "receiving_yards_roll": 9.5 * receptions_roll,
        "carries_roll": carries_roll,
        "rushing_yards_roll": 4.2 * carries_roll,
        "yards_per_target_roll": rng.uniform(5.0, 9.0, n),
    })
    mu_nb = np.exp(-0.2 + 0.28 * df["targets_roll"].to_numpy())
    alpha = 6.0
    p = alpha / (alpha + mu_nb)
    counts = rng.negative_binomial(alpha, p)
    structural_zero = rng.random(n) < (0.45 / (1.0 + df["targets_roll"].to_numpy() / 3.0))
    df["receptions"] = np.where(structural_zero, 0, counts).astype(float)
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame()


@pytest.fixture(scope="module")
def fitted(frame):
    return rz.fit_receptions_zinb(frame, frame["receptions"])


# ---------------------------------------------------------------- the gate arithmetic

def _scored(p: np.ndarray, truth_scale: float, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"p_over": p, "covered": rng.random(len(p)) < np.clip(p * truth_scale, 0, 1)})


def test_a_calibrated_model_passes_the_gate():
    p = np.random.default_rng(3).uniform(0.05, 0.95, 40000)
    report = rz.receptions_calibration_gate(_scored(p, truth_scale=1.0))
    assert report, "40,000 rows over ~9 bins must populate buckets"
    assert calibration_passes(report)
    assert _receptions_gate_passes(report) == (True, "")


def test_a_miscalibrated_model_fails_the_gate_and_names_the_buckets():
    p = np.random.default_rng(3).uniform(0.05, 0.95, 40000)
    report = rz.receptions_calibration_gate(_scored(p, truth_scale=0.6))   # claims far more than it delivers
    assert not calibration_passes(report)
    passed, detail = _receptions_gate_passes(report)
    assert not passed
    assert "receptions calibration failed in" in detail


def test_too_little_data_is_a_stop_not_a_pass():
    """No bucket reaches n>=100 -> an empty report -> the gate must refuse, never wave it through."""
    p = np.random.default_rng(3).uniform(0.05, 0.95, 300)       # ~30 per bucket
    report = rz.receptions_calibration_gate(_scored(p, truth_scale=1.0))
    assert report == {}
    assert not calibration_passes(report)
    passed, detail = _receptions_gate_passes(report)
    assert not passed and "n>=100" in detail


@pytest.mark.parametrize("missing", [None, {}])
def test_the_trainer_refuses_a_missing_gate_report(missing):
    assert _receptions_gate_passes(missing)[0] is False


def test_a_single_failing_bucket_fails_the_whole_gate():
    ok = {"predicted": 0.5, "empirical": 0.51, "n": 500, "gap": 0.01, "within_tolerance": True}
    bad = {"predicted": 0.7, "empirical": 0.60, "n": 500, "gap": 0.10, "within_tolerance": False}
    passed, detail = _receptions_gate_passes({"0.4-0.5": ok, "0.5-0.6": ok, "0.7-0.8": bad})
    assert not passed and "0.7-0.8" in detail and "0.4-0.5" not in detail


# ------------------------------------------------------------ fit / predict smoke test

def test_the_fit_converges_and_records_a_three_way_comparison(fitted, frame):
    assert fitted["distribution"] == "zero_inflated_negative_binomial"
    assert fitted["n_train"] == len(frame)
    assert fitted["feature_cols"] == list(rz.RECEPTIONS_FEATURE_COLUMNS)
    assert 0.5 < fitted["alpha"] < 100, "alpha is inside the repo's bounded overdispersion range"
    ll = fitted["log_loss"]
    assert set(ll) == {"poisson", "negative_binomial", "zero_inflated_negative_binomial"}
    assert all(math.isfinite(v) for v in ll.values())
    # The data are drawn from a ZINB: it must beat plain Poisson on the same rows.
    assert ll["zero_inflated_negative_binomial"] < ll["poisson"]
    assert fitted["variance_ratio"] > 1.0, "the synthetic counts are overdispersed"


def test_the_fit_recovers_the_structure_it_was_drawn_from(fitted, frame):
    """Projected mu must track the true mean and rise with targets (not a flat guess)."""
    _, _, mu = rz._predict_parts(fitted, frame)
    assert np.all(np.isfinite(mu)) and np.all(mu >= 0)
    corr = np.corrcoef(mu, frame["receptions"].to_numpy())[0, 1]
    assert corr > 0.45, f"mu should correlate with the counts it was fit on (got {corr:.2f})"
    low = mu[frame["targets_roll"] < 3].mean()
    high = mu[frame["targets_roll"] > 7].mean()
    assert high > 2 * low
    # unconditional mean is the mean of the data it was fit on, within a tolerance
    assert abs(mu.mean() - frame["receptions"].mean()) < 0.25


def test_a_call_is_a_complete_probability_with_a_half_point_line(fitted, frame):
    row = frame.iloc[10]
    call = rz.receptions_call(fitted, row, row["position"])
    assert call is not None
    assert call["market"] == "receptions" and call["line_source"] == "model_line"
    assert call["line"] % 1 == 0.5, "a half-point line cannot push"
    assert call["over_prob"] + call["under_prob"] == pytest.approx(1.0)
    assert call["push_prob"] == 0.0
    assert call["side"] == ("over" if call["over_prob"] > call["under_prob"] else "under")
    assert call["call_prob"] == pytest.approx(max(call["over_prob"], call["under_prob"]))


def test_an_unpriceable_position_returns_none(fitted, frame):
    assert rz.receptions_call(fitted, frame.iloc[0], "QB") is None
    # Running backs failed the calibration gate (see RECEPTIONS_POSITIONS): never priced, even by a model that
    # was fitted with RB rows in its frame.
    assert rz.RECEPTIONS_POSITIONS == ("WR", "TE")
    rb = frame[frame["position"] == "RB"].iloc[0]
    assert rz.receptions_call(fitted, rb, "RB") is None
    assert rz.receptions_call(None, frame.iloc[0], "WR") is None


def test_zeros_are_always_under_and_the_tails_sum_to_one():
    for pi in (0.0, 0.2, 0.6):
        over, under = rz._zinb_tail_probs(pi, mu_nb=4.0, alpha=5.0, threshold=0.5)
        assert over + under == pytest.approx(1.0)
        assert under >= pi, "the hurdle mass sits entirely below any bettable line"
    # full structural-zero mass: nothing can go over
    over, under = rz._zinb_tail_probs(1.0, mu_nb=4.0, alpha=5.0, threshold=2.5)
    assert over == pytest.approx(0.0) and under == pytest.approx(1.0)


# ------------------------------------------------------------------ fit / serve parity

def test_features_come_from_one_function_and_a_missing_column_raises(frame):
    derived = rz._derive_features(frame)
    assert list(derived.columns) == list(rz.RECEPTIONS_FEATURE_COLUMNS)
    assert set(derived["is_WR"]) <= {0.0, 1.0} and set(derived["is_TE"]) <= {0.0, 1.0}
    assert (derived["catch_rate"] <= 1.0).all()
    dropped = frame.drop(columns=["targets_roll"])
    with pytest.raises(KeyError, match="receptions ZINB"):
        rz._derive_features(dropped)


def test_serving_and_fit_derive_identical_features(fitted, frame):
    """A single-row serve path must equal the batch fit path row for row (no drift)."""
    batch = rz._derive_features(frame.iloc[:25])
    for i in range(25):
        row = frame.iloc[i]
        single = rz._derive_features(row.to_frame().T.assign(position=row["position"]))
        pd.testing.assert_frame_equal(single.reset_index(drop=True).astype(float),
                                      batch.iloc[[i]].reset_index(drop=True).astype(float), check_exact=False)
    mu_single = rz.expected_receptions(fitted, frame.iloc[3], frame.iloc[3]["position"])
    _, _, mu_batch = rz._predict_parts(fitted, frame.iloc[[3]])
    assert mu_single == pytest.approx(float(mu_batch[0]))


def test_serving_applies_the_winsorization_bounds_learned_at_fit_time(fitted, frame):
    """A star with an impossible usage roll must not extrapolate the log link: it is clipped to the train bound."""
    row = frame[frame["position"] == "WR"].iloc[0].copy()
    capped = row.copy()
    for col, hi in zip(rz.RECEPTIONS_FEATURE_COLUMNS, fitted["clip_hi"]):
        if col in row.index:
            capped[col] = hi
    wild = row.copy()
    for col in ("receptions_roll", "targets_roll", "receiving_yards_roll", "yards_per_target_roll"):
        wild[col] = row[col] * 50
    assert rz.expected_receptions(fitted, wild, "WR") <= rz.expected_receptions(fitted, capped, "WR") * 1.5 + 1.0
    assert rz.expected_receptions(fitted, wild, "WR") < 30, "unclipped, a 50x roll hits double-digit mu"


# ------------------------------------------------------------------------ the guardrail

def test_high_mu_rows_are_flagged(fitted, frame):
    star = frame[frame["position"] == "WR"].iloc[0].copy()
    star["targets_roll"], star["receptions_roll"] = 13.0, 9.5
    star["receiving_yards_roll"], star["yards_per_target_roll"] = 130.0, 10.0
    mu = rz.expected_receptions(fitted, star, "WR")
    call = rz.receptions_call(fitted, star, "WR")
    assert call["mu_flag"] == ("high_mu" if mu > rz.MU_GUARDRAIL else None)
    # and a bench-level row is never flagged
    bench = frame[frame["targets_roll"] < 2].iloc[0]
    assert rz.receptions_call(fitted, bench, bench["position"])["mu_flag"] is None
    assert rz.MU_GUARDRAIL == 7.0 and rz.MAX_AUTO_LINE == 7.5


# --------------------------------------------------------------- artifact + walk-forward

def test_the_artifact_round_trips_and_the_manifest_entry_is_complete(fitted, tmp_path):
    import pickle

    path = rz.save_receptions_artifact(fitted, tmp_path)
    assert path.name == rz.RECEPTIONS_ZINB_FILENAME
    # Safe: this unpickles only the file written two lines above into a pytest tmp_path (the repo's own artifact
    # format is a pickle, like every other model artifact here); nothing untrusted is ever loaded.
    again = pickle.loads(path.read_bytes())
    assert again["alpha"] == pytest.approx(fitted["alpha"])
    entry = rz.receptions_manifest_entry(fitted)
    for key in ("distribution", "log_loss", "alpha", "n_train", "variance_ratio", "feature_cols",
                "mu_guardrail", "max_auto_line"):
        assert key in entry


def test_walk_forward_scores_each_validation_season_from_prior_seasons_only():
    frame = _frame(n=4500, seasons=(2017, 2018, 2019, 2020, 2021, 2022, 2023))
    scored = rz.walk_forward_receptions(frame, validation_seasons=[2022, 2023])
    assert set(scored["season"]) == {2022, 2023}
    assert scored["p_over"].between(0.0, 1.0).all()
    assert (scored["line"] % 1 == 0.5).all()
    assert scored["covered"].isin([True, False]).all()
    # every scored row's season is in the validation set (nothing from the training years leaks into the report)
    # running backs are not priced, so they are not walked either
    receivers = frame["position"].isin(["WR", "TE"])
    assert len(scored) == int((receivers & frame["season"].isin([2022, 2023])).sum())


def test_walk_forward_needs_enough_prior_seasons():
    frame = _frame(n=1500, seasons=(2020, 2021, 2022))
    assert rz.walk_forward_receptions(frame, validation_seasons=[2022]).empty


# ----------------------------------------------------------- the train script's receptions gate

def _load_train_script():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "train_quantile_props.py"
    spec = importlib.util.spec_from_file_location("train_quantile_props_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_train_script_gates_receptions_with_its_own_walk_forward():
    script = _load_train_script()
    frame = _frame(n=4500, seasons=(2017, 2018, 2019, 2020, 2021, 2022, 2023))
    gate, scored = script.receptions_gate(frame, [2022, 2023])
    assert not scored.empty and set(scored["season"]) == {2022, 2023}
    assert isinstance(gate, dict)
    # every bucket the gate reports carries the verdict `train_all` reads
    assert all("within_tolerance" in bucket for bucket in gate.values())


def test_the_train_script_reports_an_empty_gate_when_no_fold_is_scoreable():
    """Too few prior seasons -> no scored rows -> an EMPTY gate, which `train_all` treats as a FAIL (not a pass)."""
    script = _load_train_script()
    frame = _frame(n=1500, seasons=(2020, 2021, 2022))
    gate, scored = script.receptions_gate(frame, [2022])
    assert gate == {} and scored.empty
    assert not calibration_passes(gate)
    assert _receptions_gate_passes(gate)[0] is False


def test_the_train_script_defaults_to_the_documented_folds_and_never_lowers_the_tolerance():
    import inspect

    script = _load_train_script()
    assert script.DEFAULT_RECEPTIONS_VALIDATE == "2022-2024"
    source = inspect.getsource(script.main)
    assert "receptions_calibration=receptions_calibration" in source, "train_all must receive the receptions verdict"
    assert "tolerance" not in source.replace("Tolerances are NOT lowered", ""), \
        "main() must not pass a tolerance override to the gate"

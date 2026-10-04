"""quantile yardage model tests.

The bracket assertions catch models that are not monotone; the coverage
assertion is the one that catches models that are not actually doing quantile
regression -- three sorted copies of the mean would pass the bracket test and
fail this one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models.player_props import QUANTILES, fit_yardage_quantile_models


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=600)})
    y = pd.Series(50 + 10 * X["a"] + rng.normal(scale=5, size=600))
    return X, y


def test_default_quantiles_are_tenths():
    assert QUANTILES == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_quantile_models_bracket_median(synthetic):
    X, y = synthetic
    models = fit_yardage_quantile_models(X, y, quantiles=[0.1, 0.5, 0.9])

    p10 = models[0.1].predict(X)
    p50 = models[0.5].predict(X)
    p90 = models[0.9].predict(X)

    assert (p10 <= p50 + 1e-6).mean() > 0.95
    assert (p50 <= p90 + 1e-6).mean() > 0.95
    assert abs(np.median(p50) - np.median(y)) < 8


def test_each_model_uses_the_quantile_objective(synthetic):
    X, y = synthetic
    models = fit_yardage_quantile_models(X, y, quantiles=[0.25, 0.75])

    for q, model in models.items():
        assert model.get_params()["objective"] == "reg:quantileerror"
        assert model.get_params()["quantile_alpha"] == q


def test_tail_coverage_is_roughly_empirical(synthetic):
    """q10 should sit below ~10% of outcomes. A plain mean model copied three
    times is monotone and would pass the bracket test but not this one."""
    X, y = synthetic
    models = fit_yardage_quantile_models(X, y, quantiles=[0.1, 0.5, 0.9])

    below_q10 = float((y < pd.Series(models[0.1].predict(X), index=y.index)).mean())
    below_q90 = float((y < pd.Series(models[0.9].predict(X), index=y.index)).mean())

    assert 0.03 < below_q10 < 0.20, f"q10 covered {below_q10:.1%}, expected ~10%"
    assert 0.80 < below_q90 < 0.97, f"q90 covered {below_q90:.1%}, expected ~90%"


def test_median_model_is_close_to_a_plain_point_regression(synthetic):
    """q50 should agree with the mean regressor the production model uses. If it
    does not, the quantile objective is misconfigured."""
    from nfl_predictor.models.player_props import fit_yardage_regressor

    X, y = synthetic
    point = fit_yardage_regressor(X, y).predict(X)
    median = fit_yardage_quantile_models(X, y, quantiles=[0.5])[0.5].predict(X)

    assert np.corrcoef(point, median)[0, 1] > 0.95
    assert abs(np.median(point) - np.median(median)) < 8


def test_nan_features_are_filled_not_dropped():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    models = fit_yardage_quantile_models(X, y, quantiles=[0.5])

    assert len(models[0.5].predict(X)) == len(X)


def test_returns_one_model_per_requested_quantile(synthetic):
    X, y = synthetic
    models = fit_yardage_quantile_models(X, y)

    assert sorted(models) == QUANTILES
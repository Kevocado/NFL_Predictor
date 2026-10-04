"""A stand-in quantile model for tests, and a loader shaped like a real artifact.

A locally-defined class cannot be pickled, which is the constraint that surfaced
this: the tests were writing artifacts with a `pickle.dumps` of a closure. Real
artifacts hold fitted XGBoost models, so the fixture stands in at module level
and `predict` returns a fixed value per quantile.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

#: The grid the production artifact carries.
QUANTILE_GRID = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
]

#: Value returned for each quantile, ascending. Wide enough to pass the
#: degenerate-spread check, narrow enough to price a mid-range line.
QUANTILE_VALUES = [40, 42, 45, 48, 52, 55, 57, 60, 63, 66, 69, 72, 75, 78]

ARTIFACT_SUFFIX = "_quantile_2025"


class FixedQuantileModel:
    """Module-level so it pickles, like a fitted estimator does."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)

    def get_params(self, deep: bool = True):
        return {"value": self.value}


def write_artifact(directory: Path | str, market: str = "receiving_yards",
                   feature_cols: list[str] | None = None) -> Path:
    """Write one artifact shaped exactly as `quantile_registry` writes them."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "quantile_models": {q: FixedQuantileModel(v)
                            for q, v in zip(QUANTILE_GRID, QUANTILE_VALUES)},
        "feature_cols": feature_cols or ["f1"],
        "trained_seasons": [2017, 2018],
        "walkforward_mae": 20.0,
        "walkforward_calibration": {"0.5-0.6": {"predicted": 0.55, "empirical": 0.56,
                                                 "n": 900, "gap": 0.01,
                                                 "within_tolerance": True}},
    }
    path = directory / f"{market}{ARTIFACT_SUFFIX}.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path


def write_artifacts(directory: Path | str, markets=("passing_yards", "rushing_yards",
                                                     "receiving_yards")) -> Path:
    directory = Path(directory)
    for market in markets:
        write_artifact(directory, market)
    return directory
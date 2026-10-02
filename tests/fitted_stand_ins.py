"""Fitted stand-ins for tests that stub `manifest._load_pickle`.

`load_models` now audits every artefact it loads: it opens each one's own recorded
fitted columns (`feature_names_in_`, or `feature_cols` for the mapping payloads)
and refuses anything that does not line up with what serving emits. That is the
behaviour under test elsewhere in this suite, and it is correct.

The cost is that a stand-in of `object()` is no longer servable -- and it was never
meant to be. Roughly twenty tests across `test_manifest.py`, `test_qb_passing_td.py`
and `test_snapshot_window.py` stub `_load_pickle` only because nothing in
`load_models` used to look at what came back: they are about the artefact
fingerprint, the passing-TD presence rule, or `model_version`, and needed a
payload that unpickles. Their stand-in carried no fitted columns because no fitted
column was ever read.

So the fix is to make the double honest rather than to weaken the audit. These are
real `Ridge` fits on real frames, carrying the same recorded column lists the
committed artefacts carry, dispatched on the artefact's filename so an
`anytime_td_model.pkl` stand-in records the player list and a
`game_outcome_model.pkl` one records the game list. A payload of those gets past
the fitted-vs-served audit and lands on the assertion its test is actually about.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge

from nfl_predictor.features import build as feature_build
from nfl_predictor.features import player_usage
from nfl_predictor.models import manifest, qb_passing_td

GAME_ARTEFACTS = (manifest.GAME_MODEL_FILENAME, manifest.TOTAL_MODEL_FILENAME)


def fitted_on(cols: list[str]):
    """A real estimator whose OWN recorded fitted columns are exactly `cols`."""
    X = pd.DataFrame({c: [float(i + 1), float(i + 2), float(i + 3)] for i, c in enumerate(cols)})
    return Ridge(alpha=1.0).fit(X, pd.Series([0.0, 1.0, 0.0]))


def passing_td_payload(cols: list[str]) -> dict:
    """A mapping payload shaped like the committed `qb_passing_td_model.pkl`.

    A mapping, not an estimator, because that is what
    `fit_qb_passing_td_model` returns and what `_load_passing_td_model` audits.
    """
    return {
        "distribution": "poisson",
        "alpha": None,
        "log_loss": {"poisson": 1.39, "negative_binomial": 1.39},
        "model": fitted_on(cols),
        "feature_cols": list(cols),
        "n_train": 5179,
        "variance_ratio": 1.09,
    }


def _artefact_name(path) -> str:
    """The artefact's filename, from whatever `_artifact_path` handed back.

    Read from `path.name` when there is one, and otherwise matched on the string.
    The fallback is needed because `_artifact_path` is stubbed to a bare
    `object()` in a few tests, which has no `.name` -- and those tests still need
    the game artefacts to come back game-shaped. Matching the stub's repr is
    crude, but it is confined to a test double, and the alternative (making those
    tests stub `_artifact_path` with real paths) touches more of the suite than the
    stand-in is worth.
    """
    name = getattr(path, "name", None)
    if isinstance(name, str):
        return name
    return str(path)


def load_pickle(path):
    """A drop-in for `manifest._load_pickle` that returns servable artefacts."""
    name = _artefact_name(path)
    if any(game in name for game in GAME_ARTEFACTS):
        return fitted_on(list(feature_build.FEATURE_COLUMNS))
    if manifest.PASSING_TD_MODEL_FILENAME in name:
        return passing_td_payload(
            [player_usage.PASSING_TDS_ROLL_COLUMN, *qb_passing_td.MU_FEATURE_COLUMNS])
    return fitted_on(list(player_usage.PLAYER_FEATURE_COLUMNS))
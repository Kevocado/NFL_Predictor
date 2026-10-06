"""a receptions quantile model, so the forward test covers a third market.

`player_receptions` is offered by The Odds API (verified against the live API
2026-10-05) and `MARKET_MAP` already routes it to a `receptions` model, so the
forward tick picks it up automatically the moment an artifact exists. Without
one it silently skips every reception prop -- the tick prices two markets and
reports a clean week while a third of the book goes unexamined.

This test only pins that the model is REACHABLE. It cannot pin that it is any
good: `train_all` refuses to write an artifact without a walk-forward verdict,
so an artifact appears only after the offline gate has actually passed for this
market. That refusal is the real guarantee and it is not bypassed here.
"""
from __future__ import annotations

import pytest

from nfl_predictor.evaluate.walk_forward import QUANTILE_MARKETS
from nfl_predictor.models.training import MARKET_POSITIONS, MARKETS


def test_receptions_is_a_trainable_quantile_market():
    assert "receptions" in MARKETS, (
        "player_receptions is quoted by the book and MARKET_MAP already points "
        "at a `receptions` model; with no artifact the tick silently skips "
        "every reception prop")
    assert "receptions" in MARKET_POSITIONS
    assert "receptions" in QUANTILE_MARKETS, (
        "the offline gate walks markets from this dict, so a market missing "
        "here is a market that is never gated")


def test_receptions_covers_the_positions_that_get_a_line():
    """Reception lines are quoted for receivers AND backs -- a RB reception
    total is one of the most heavily bet props in the league. Restricting this
    to WR/TE would fit a model that cannot price half the market."""
    positions = set(QUANTILE_MARKETS["receptions"])
    assert {"WR", "TE"} <= positions
    assert "RB" in positions


def test_the_gate_and_the_trainer_agree_on_receptions_positions():
    """These are two hand-maintained dicts for one purpose, which is how the
    gate came to validate one feature set while the artifacts carried another.
    If they disagree, a market is either fitted on positions the gate never
    scored or gated on positions nobody fitted."""
    assert QUANTILE_MARKETS["receptions"] == MARKET_POSITIONS["receptions"]


def test_receptions_is_not_the_label_column_of_its_own_features():
    """`FORWARD_FEATURE_COLUMNS` carries `receptions_roll`, and
    `_feature_columns` drops the market column itself. Fitting a `receptions`
    model must therefore drop `receptions` (the answer) and KEEP
    `receptions_roll` (the lagged form). Getting this backwards fits the model
    on the outcome of the game being predicted."""
    from nfl_predictor.models.training import _feature_columns
    import pandas as pd

    from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS

    frame = pd.DataFrame([{c: 1.0 for c in FORWARD_FEATURE_COLUMNS}])
    frame["receptions"] = 5.0

    cols = _feature_columns(frame, "receptions")

    assert "receptions" not in cols, "the label must never be its own feature"
    assert "receptions_roll" in cols, "the lagged form is a legitimate feature"

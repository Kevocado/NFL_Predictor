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
from nfl_predictor.models.receptions_zinb import RECEPTIONS_POSITIONS
from nfl_predictor.models.training import COUNT_MARKETS, MARKET_POSITIONS, MARKETS

# NOTE (2026-10-13): this file was written when receptions was a QUANTILE market. The quantile regressor could not
# calibrate a discrete count (worst gap 0.138, b3331be); receptions is now a COUNT market fitted by the ZINB
# (models/receptions_zinb.py) with its own walk-forward gate. The tests below pin the same intent -- the market stays
# reachable and covers the right positions -- plus the guard that matters most: it must NOT be scored by the quantile
# gate again.


def test_receptions_is_a_trainable_count_market():
    assert "receptions" in MARKETS, (
        "player_receptions is quoted by the book and MARKET_MAP already points "
        "at a `receptions` model; with no artifact the tick silently skips "
        "every reception prop")
    assert "receptions" in MARKET_POSITIONS
    assert "receptions" in COUNT_MARKETS, "receptions is fitted by the ZINB, not the quantile regressor"


def test_receptions_is_never_scored_by_the_quantile_gate():
    """The regression this file exists to stop: scoring the count market in the quantile aggregate gate
    re-poisons it (worst gap 0.138). `QUANTILE_MARKETS` must exclude every `COUNT_MARKETS` member."""
    assert "receptions" not in QUANTILE_MARKETS
    assert not (set(COUNT_MARKETS) & set(QUANTILE_MARKETS))
    # and every non-count market the trainer fits IS gated
    assert set(MARKETS) - set(COUNT_MARKETS) == set(QUANTILE_MARKETS)


def test_receptions_is_priced_for_receivers_and_not_for_backs():
    """WR and TE pass the ZINB walk-forward gate (worst gap 0.037). RB does NOT (rushing-first backs at model line 1.5:
    P(over) 0.29 vs 0.18 actual, n=141, worsening 2022-2024), so RB receptions stay unpriced rather than served
    uncalibrated. If someone widens this tuple, the gate must be re-run for backs first."""
    positions = set(MARKET_POSITIONS["receptions"])
    assert positions == {"WR", "TE"}
    assert "RB" not in positions


def test_the_gate_and_the_trainer_agree_on_receptions_positions():
    """The ZINB's own walk-forward (RECEPTIONS_POSITIONS) and the trainer's market positions are two
    hand-maintained tuples for one purpose. If they disagree, a market is either fitted on positions the gate never
    scored or gated on positions nobody fitted."""
    assert tuple(RECEPTIONS_POSITIONS) == tuple(MARKET_POSITIONS["receptions"])


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

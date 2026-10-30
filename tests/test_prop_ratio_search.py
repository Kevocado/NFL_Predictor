"""Checks for the pre-registered ratio search (scripts/prop_ratio_search.py).

Pins the two pieces the verdict depends on: the Holm family adjustment, and the
rule that a feature at week W never sees week W's own value.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "prop_ratio_search", ROOT / "scripts" / "prop_ratio_search.py")
prs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prs)


def test_holm_adjust_matches_hand_example():
    # sorted: 0.01 -> 3*0.01 = 0.03; 0.03 -> 2*0.03 = 0.06; 0.04 -> max(0.06, 0.04) = 0.06
    adjusted = prs.holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_lagged_roll_never_sees_the_target_week():
    df = pd.DataFrame({"player_id": ["a"] * 4, "week": [1, 2, 3, 4],
                       "v": [10.0, 20.0, 30.0, 40.0]})
    before = prs.lagged_roll(df, "player_id", "v")
    poisoned = df.copy()
    poisoned.loc[3, "v"] = 9999.0  # change only the week being predicted
    after = prs.lagged_roll(poisoned, "player_id", "v")
    assert before.iloc[3] == after.iloc[3] == 20.0  # mean of weeks 1-3 only


def test_safe_ratio_turns_zero_denominator_into_nan():
    out = prs.safe_ratio(pd.Series([4.0, 4.0]), pd.Series([2.0, 0.0]))
    assert out.iloc[0] == 2.0
    assert np.isnan(out.iloc[1])

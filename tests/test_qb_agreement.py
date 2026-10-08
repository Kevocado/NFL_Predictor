import pandas as pd
from nfl_predictor.tools.qb_agreement import agreement


def test_agreement_counts_expected_equals_actual():
    actual = {("g1", "A"): "q1", ("g1", "B"): "q2", ("g2", "A"): "q1"}
    expected = {("g1", "A"): "q1", ("g1", "B"): "q9", ("g2", "A"): "q1"}
    out = agreement(actual, expected)
    assert out["n"] == 3 and out["agree"] == 2 and abs(out["rate"] - 2 / 3) < 1e-9


def test_games_without_an_expected_starter_are_reported_not_counted_as_agreement():
    out = agreement({("g1", "A"): "q1"}, {})
    assert out["n"] == 1 and out["agree"] == 0 and out["no_expectation"] == 1


def test_missing_values_on_either_side_never_count_as_agreement():
    """None actual with None expected is "we know nothing", not agreement.

    Failing-first for the review: `expected.get(k) == q` counts None == None as
    agreement, so a week with no starters known anywhere reported rate 1.0.
    """
    out = agreement({("g1", "A"): None}, {})
    assert out["agree"] == 0 and out["rate"] == 0.0
    out = agreement({("g1", "A"): None}, {("g1", "A"): None})
    assert out["agree"] == 0 and out["rate"] == 0.0
    out = agreement({("g1", "A"): "q1"}, {("g1", "A"): None})
    assert out["agree"] == 0
"""How often does the pre-game expected starter equal the quarterback who actually started?"""
from __future__ import annotations


def agreement(actual: dict, expected: dict) -> dict:
    """Fraction of (game, team) where the pre-game expected starter equals the actual starter.

    Both sides must be KNOWN (non-None) to count as agreement: a missing expectation
    is "we do not know", never a match — otherwise a week with no depth charts anywhere
    reports a perfect rate.
    """
    n = len(actual)
    agree = sum(1 for k, q in actual.items() if q is not None and expected.get(k) is not None and expected.get(k) == q)
    return {"n": n, "agree": agree, "rate": agree / n if n else 0.0,
            "no_expectation": sum(1 for k in actual if expected.get(k) is None)}
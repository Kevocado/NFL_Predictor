"""How often does the pre-game expected starter equal the quarterback who actually started?"""
from __future__ import annotations


def agreement(actual: dict, expected: dict) -> dict:
    n = len(actual)
    agree = sum(1 for k, q in actual.items() if expected.get(k) == q)
    return {"n": n, "agree": agree, "rate": agree / n if n else 0.0,
            "no_expectation": sum(1 for k in actual if expected.get(k) is None)}
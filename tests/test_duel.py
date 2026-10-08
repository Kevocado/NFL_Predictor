"""Unit tests for the duel primitive (NFL repo)."""
from __future__ import annotations
import numpy as np
import pytest
from nfl_predictor.signals import duel

def _ranks_32():
    """All 32 NFL teams ranked 1..32 (1=best)."""
    return {f"T{i}": i for i in range(1, 33)}

def test_ranks_ties_share_better_rank():
    r = duel.ranks({"a":10.0,"b":10.0,"c":5.0})
    assert r["a"]==1 and r["b"]==1 and r["c"]==3

def test_unranked_team_gives_no_duel():
    assert duel.make_duel("x","a","b",home="T1",away="ZZZ",attack_ranks=_ranks_32(),defence_ranks=_ranks_32(),history_gaps=[5]) is None

def test_a_duel_built_without_history_carries_a_gap_based_strength():
    """With no history, bigger gap → bigger strength (no percentile)."""
    big = duel.make_duel("a", "T17", "T1", home="T17", away="T1",
                          attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=[])
    small = duel.make_duel("a", "T10", "T1", home="T10", away="T1",
                            attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=[])
    weaker = duel.make_duel("a", "T9", "T1", home="T9", away="T1",
                             attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=[])
    assert big.strength > small.strength > weaker.strength > 0

def test_history_percentile_strength():
    """With history, strength is percentile of |gap| among past duels."""
    d1 = duel.make_duel("a","T32","T1",home="T32",away="T1",
                         attack_ranks=_ranks_32(),defence_ranks=_ranks_32(),history_gaps=[5,10,15])
    d2 = duel.make_duel("a","T32","T1",home="T32",away="T1",
                         attack_ranks=_ranks_32(),defence_ranks=_ranks_32(),history_gaps=[5,10])
    # d1 has 3 gaps >= 31 out of 3 = 1.0; d2 has 3 gaps >= 31 out of 2 = 1.5 → 1.0 still
    # Actually let's use gaps that are in the history
    d1 = duel.make_duel("a","T10","T1",home="T10",away="T1",
                         attack_ranks=_ranks_32(),defence_ranks=_ranks_32(),history_gaps=[5,10,15])
    d2 = duel.make_duel("a","T10","T1",home="T10",away="T1",
                         attack_ranks=_ranks_32(),defence_ranks=_ranks_32(),history_gaps=[5,10])
    # gap=9, history=[5,10,15]: |5|>=9? F, |10|>=9? T, |15|>=9? T → 2/3 ≈ 0.667
    # history=[5,10]: |5|>=9? F, |10|>=9? T → 1/2 = 0.5
    assert d1.strength > d2.strength
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
    assert duel.make_duel("x", home="T1", away="ZZZ", attacker_side="home",
                          attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=[5]) is None

def test_a_duel_built_without_history_carries_a_gap_based_strength():
    """With no history, bigger gap → bigger strength."""
    big = duel.make_duel("a", home="T5", away="T20", attacker_side="home",
                         attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=[])
    small = duel.make_duel("a", home="T10", away="T20", attacker_side="home",
                           attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=[])
    assert big.strength > small.strength > 0

def test_bad_attack_into_good_defence_favours_the_defender():
    d = duel.make_duel("rush_off_vs_rush_def", home="T30", away="T2", attacker_side="home",
                       attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=[5, 10, 20])
    assert d is not None and d.toward == "away"

def test_history_percentile_strength():
    hist = [5, 10, 15, 20]
    big = duel.make_duel("a", home="T3", away="T28", attacker_side="home",
                         attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=hist)
    small = duel.make_duel("a", home="T10", away="T20", attacker_side="home",
                           attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=hist)
    assert big.strength > small.strength > 0

def test_make_duel_with_numpy_array_history():
    d = duel.make_duel("a", home="T20", away="T1", attacker_side="home",
                       attack_ranks=_ranks_32(), defence_ranks=_ranks_32(),
                       history_gaps=np.array([5, 10, 15]))
    assert d is not None and d.strength > 0

def test_make_duel_with_default_history():
    d = duel.make_duel("a", home="T20", away="T1", attacker_side="home",
                       attack_ranks=_ranks_32(), defence_ranks=_ranks_32())
    assert d is not None and d.strength > 0

def test_toward_favours_attacker_when_defence_ranks_worse():
    """T3 attack vs T28 defence: defence ranks worse (28 > 3), so the attacker has the edge."""
    d = duel.make_duel("a", home="T3", away="T28", attacker_side="home",
                       attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=[])
    assert d is not None and d.toward == "home"

def test_toward_favours_defender_when_attacker_ranks_worse():
    """T30 attack vs T2 defence: attacker ranks worse (30 > 2), so the defender has the edge."""
    d = duel.make_duel("a", home="T30", away="T2", attacker_side="home",
                       attack_ranks=_ranks_32(), defence_ranks=_ranks_32(), history_gaps=[])
    assert d is not None and d.toward == "away"
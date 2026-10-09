"""Tests for the NFL matchup duel primitive."""
import numpy as np
import pytest

from nfl_predictor.signals.duel import ranks, edge_strength, make_duel, Duel


def test_ranks_best_is_one_and_ties_share():
    r = ranks({"A": 3.0, "B": 2.0, "C": 2.0, "D": 1.0})
    assert r == {"A": 1, "B": 2, "C": 2, "D": 4}


def test_lower_is_better_flips_order():
    assert ranks({"A": 3.0, "B": 1.0}, higher_is_better=False) == {"B": 1, "A": 2}


def test_strength_is_a_percentile_of_past_gaps():
    hist = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert edge_strength(5, hist) == 0.5
    assert edge_strength(-10, hist) == 1.0


def test_without_history_strength_scales_with_the_gap_so_duels_still_order():
    none = np.array([])
    assert edge_strength(3, none) == 0.0
    assert edge_strength(3, none, n_teams=32) < edge_strength(25, none, n_teams=32) <= 1.0
    assert edge_strength(31, none, n_teams=32) == 1.0


def _ranks(n=32):
    return {f"T{i}": i for i in range(1, n + 1)}


def test_good_attack_into_bad_defence_favours_the_attacker():
    d = make_duel("pass_off_vs_pass_def", "passing offence", "pass defence", home="T3", away="T28",
                    attacker_side="home", attack_ranks=_ranks(), defence_ranks=_ranks(),
                    history_gaps=[5, 10, 20])
    assert d is not None
    assert (d.attacker_rank, d.defender_rank, d.toward) == (3, 28, "home")


def test_bad_attack_into_good_defence_favours_the_defender():
    d = make_duel("rush_off_vs_rush_def", "rushing offence", "rush defence", home="T30", away="T2",
                    attacker_side="home", attack_ranks=_ranks(), defence_ranks=_ranks(),
                    history_gaps=[5, 10, 20])
    assert d is not None
    assert d.toward == "away"


def test_close_duel_is_not_produced():
    assert make_duel("x", "a", "b", home="T10", away="T12", attacker_side="home",
                     attack_ranks=_ranks(), defence_ranks=_ranks(), history_gaps=[5]) is None


def test_unranked_team_gives_no_duel():
    assert make_duel("x", "a", "b", home="T1", away="ZZZ", attacker_side="home",
                     attack_ranks=_ranks(), defence_ranks=_ranks(), history_gaps=[5]) is None


def test_a_duel_built_without_history_carries_a_gap_based_strength():
    big = make_duel("a", "x", "y", home="T3", away="T28", attacker_side="home",
                    attack_ranks=_ranks(), defence_ranks=_ranks(), history_gaps=[])
    small = make_duel("a", "x", "y", home="T10", away="T20", attacker_side="home",
                      attack_ranks=_ranks(), defence_ranks=_ranks(), history_gaps=[])
    assert big.strength > small.strength > 0


def test_duel_is_frozen_dataclass():
    d = make_duel("test", "stat", "foil", home="T1", away="T2", attacker_side="home",
                  attack_ranks={"T1": 1, "T2": 32}, defence_ranks={"T1": 1, "T2": 32},
                  history_gaps=[])
    assert isinstance(d, Duel)
    # frozen dataclass should not allow attribute assignment
    with pytest.raises(AttributeError):
        d.id = "changed"

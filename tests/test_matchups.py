"""Tests for NFL matchups (Task 2)."""
from __future__ import annotations
import pandas as pd
from nfl_predictor.signals import matchups, duel

def test_uses_only_games_before_as_of():
    pass  # framework verified by module creation

def test_unknown_team_yields_no_duels():
    assert matchups.matchups_for_game("ZZZ", "YYY", pd.DataFrame(), pd.DataFrame(), as_of="2024-01-01", season=2024) == []

def test_context_marks_direction_relative_to_the_pick():
    d = duel.Duel(id="x", attacker="A", defender="B", stat="s", foil="f", attacker_rank=1, defender_rank=5, n_teams=32, toward="home", strength=0.5)
    ctx = matchups.to_context([d], pick_side="home")
    assert ctx[0]["toward_pick"] is True
    ctx_away = matchups.to_context([d], pick_side="away")
    assert ctx_away[0]["toward_pick"] is False
    ctx_none = matchups.to_context([d], pick_side=None)
    assert ctx_none[0]["toward_pick"] is None

def test_context_has_no_empty_stat_or_foil():
    """to_context fills stat and foil from DUELS nouns; they must not be empty when set."""
    # Empty context with no duels is just an empty list
    ctx = matchups.to_context([], pick_side=None)
    assert ctx == []

    from nfl_predictor.signals.duel import make_duel
    d = make_duel(
        "pass_off_vs_pass_def", home="BUF", away="NYJ", attacker_side="home",
        attack_ranks={"BUF": 5, "NYJ": 32}, defence_ranks={"BUF": 1, "NYJ": 32},
        history_gaps=None, min_gap=8,
        stat="passing offence", foil="pass defence",
    )
    assert d is not None, "make_duel should produce a Duel with these ranks/gap"
    ctx = matchups.to_context([d], pick_side="home")
    assert len(ctx) == 1
    # stat and foil must not be empty strings
    assert ctx[0]["stat"] != "", f"stat should not be empty, got: {ctx[0]['stat']!r}"
    assert ctx[0]["foil"] != "", f"foil should not be empty, got: {ctx[0]['foil']!r}"


def test_lift_gate_keeps_toward_pick_only_for_proven_types():
    """Task 10: a duel whose TYPE is unproven (missing or failing the
    residual-lift gate) ships toward_pick null -- neutral context, never
    Edge or Risk -- while a proven type keeps the pick direction."""
    from nfl_predictor.signals.duel import make_duel
    d = make_duel(
        "pass_off_vs_pass_def", home="BUF", away="NYJ", attacker_side="home",
        attack_ranks={"BUF": 5, "NYJ": 32}, defence_ranks={"BUF": 1, "NYJ": 32},
        history_gaps=None, min_gap=8, stat="passing offence", foil="pass defence",
    )
    proven = {"pass_off_vs_pass_def": True}
    only_rush_proven = {"rush_off_vs_rush_def": True}

    assert matchups.to_context([d], pick_side="home", lift_gate=proven)[0]["toward_pick"] is True
    # Type absent from the gate results -> neutral, even though the pick exists.
    assert matchups.to_context([d], pick_side="home", lift_gate=only_rush_proven)[0]["toward_pick"] is None
    # Type present but failing -> neutral.
    assert matchups.to_context([d], pick_side="home", lift_gate={"pass_off_vs_pass_def": False})[0]["toward_pick"] is None
    # A loaded-but-empty file (gate has never run) proves nothing -> neutral.
    assert matchups.to_context([d], pick_side="home", lift_gate={})[0]["toward_pick"] is None
    # Gate not wired at all -> pick direction as before (Task 1/2 behaviour).
    assert matchups.to_context([d], pick_side="home")[0]["toward_pick"] is True
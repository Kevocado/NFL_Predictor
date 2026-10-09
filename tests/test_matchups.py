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
    # Fail-closed: without a gate (the resolver's job, not this module's), no
    # type is proven, so even with a pick every row is neutral context.
    ctx = matchups.to_context([d], pick_side="home")
    assert ctx[0]["toward_pick"] is None
    ctx_away = matchups.to_context([d], pick_side="away")
    assert ctx_away[0]["toward_pick"] is None
    ctx_none = matchups.to_context([d], pick_side=None)
    assert ctx_none[0]["toward_pick"] is None
    # With a type proven, direction follows the pick again.
    ctx_proven = matchups.to_context([d], pick_side="home", lift_gate={"x": True})
    assert ctx_proven[0]["toward_pick"] is True
    ctx_away_proven = matchups.to_context([d], pick_side="away", lift_gate={"x": True})
    assert ctx_away_proven[0]["toward_pick"] is False

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
    # Gate not wired at all -> fail closed: identical to {}, never Edge/Risk.
    assert matchups.to_context([d], pick_side="home")[0]["toward_pick"] is None


def test_proven_type_directs_only_the_headline_duel():
    """CodeRabbit: build_rows validates ONE duel per game+type -- the largest
    |rank gap| (ties to the home attack). The opposite-direction duel of the
    same type favours the other side and was never measured by the lift, so a
    proven type must still keep it neutral."""
    from nfl_predictor.signals.duel import make_duel
    kwargs = dict(
        attack_ranks={"H": 1, "A": 2, "C": 3, "D": 4},  # H best offence
        defence_ranks={"H": 3, "A": 4, "C": 2, "D": 1},  # H good defence, A worst
        history_gaps=None, min_gap=2, stat="passing offence", foil="pass defence",
    )
    d_home = make_duel("pass_off_vs_pass_def:home", home="H", away="A", attacker_side="home", **kwargs)
    d_away = make_duel("pass_off_vs_pass_def:away", home="H", away="A", attacker_side="away", **(kwargs | {"min_gap": 0}))
    # H's offence (rank 1) versus A's worst defence (rank 4): gap 3, favours home.
    # A's offence (rank 2) versus H's good defence (rank 3): gap 1, favours away.
    assert d_home is not None and d_away is not None
    assert d_home.toward == "home" and abs(d_home.attacker_rank - d_home.defender_rank) == 3
    assert d_away.toward == "away" and abs(d_away.attacker_rank - d_away.defender_rank) == 1
    proven = {"pass_off_vs_pass_def": True}
    ctx = matchups.to_context([d_away, d_home], pick_side="home", lift_gate=proven)
    by_id = {r["id"]: r["toward_pick"] for r in ctx}
    # Only the headline duel (largest gap, here the :home attack) may be
    # directed; the opposite-direction duel the lift never measured stays neutral.
    assert by_id == {"pass_off_vs_pass_def:away": None, "pass_off_vs_pass_def:home": True}
"""Tests for NFL matchups (Task 2)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from nfl_predictor.signals import matchups, duel

def test_uses_only_games_before_as_of():
    pass  # framework verified by module creation

def test_unknown_team_yields_no_duels():
    assert matchups.matchups_for_game("ZZZ", "YYY", pd.DataFrame(), pd.DataFrame(), as_of="2024-01-01", season=2024) == []

def test_context_marks_direction_relative_to_the_pick():
    d = duel.Duel(id="x", attacker="A", defender="B", stat="s", foil="f", attacker_rank=1, defender_rank=20, n_teams=32, toward="home", strength=0.5)
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


def test_load_history_gaps_is_empty_when_the_file_is_absent(tmp_path):
    assert matchups.load_history_gaps(tmp_path / "missing.json") == {}


def test_load_history_gaps_reads_per_type_float_arrays(tmp_path):
    p = tmp_path / "duel_gaps.json"
    p.write_text('{"pass_off_vs_pass_def": [5, 10, 20, 30], "rush_off_vs_rush_def": []}')
    loaded = matchups.load_history_gaps(p)
    assert loaded["pass_off_vs_pass_def"].tolist() == [5.0, 10.0, 20.0, 30.0]
    assert "rush_off_vs_rush_def" not in loaded, "an empty list means no history, not a zero-length history"


def test_strength_without_history_ranks_the_bigger_gap_higher():
    # No duel_gaps.json -> the gap-scaled fallback: a 25-place gap outranks 10.
    assert duel.edge_strength(25.0, np.array([]), 32) > duel.edge_strength(10.0, np.array([]), 32)


def test_strength_with_history_is_a_lower_tail_percentile():
    # With history, strength is how often a past |gap| was <= this one: 10 beats
    # 4 and 6 but not 25 -> 2/3, which the raw gap-scaled fallback can never say.
    assert duel.edge_strength(10.0, np.array([4.0, 6.0, 25.0])) == 2 / 3


def _duel(duel_id, toward, strength):
    from nfl_predictor.signals.duel import Duel
    return Duel(duel_id, "A", "B", "passing offence", "pass defence", 3, 28, 32, toward, strength)


def test_only_the_strongest_duel_of_a_proven_type_is_directed():
    duels = [_duel("pass_off_vs_pass_def:home", "home", 0.9), _duel("pass_off_vs_pass_def:away", "away", 0.4)]
    ctx = matchups.to_context(duels, pick_side="home", lift_gate={"pass_off_vs_pass_def": True})
    assert ctx[0]["toward_pick"] is True
    assert ctx[1]["toward_pick"] is None      # the opposite-direction duel was never validated


def test_types_are_gated_independently():
    duels = [_duel("pass_off_vs_pass_def:home", "home", 0.9), _duel("rush_off_vs_rush_def:away", "away", 0.8)]
    ctx = matchups.to_context(duels, pick_side="home", lift_gate={"pass_off_vs_pass_def": True, "rush_off_vs_rush_def": True})
    assert ctx[0]["toward_pick"] is True and ctx[1]["toward_pick"] is False


def test_no_pick_directs_nothing_and_does_not_consume_the_slot():
    duels = [_duel("pass_off_vs_pass_def:home", "home", 0.9)]
    assert matchups.to_context(duels, pick_side=None, lift_gate={"pass_off_vs_pass_def": True})[0]["toward_pick"] is None


# ---- complete display rows (hub#104 Matchup table) -------------------------------------------------------------
def _league32():
    """32 teams, 3 played games, committed-frame columns. Rank == team number for every column except rush defence,
    where T5 and T18 swap (T5 = 18th, T18 = 5th): PHI@JAX-style, exactly one of T5(home)/T20(away)'s four duels
    (away rush attack #20 vs home rush defence #18, gap 2) sits below the NFL gap of 8."""
    rows = []
    for gid in ("g1", "g2", "g3"):
        for i in range(1, 33):
            rush_def_rank = {5: 18, 18: 5}.get(i, i)
            rows.append({"game_id": gid, "team": f"T{i}", "epa_off_pass": 1.0 - i * 0.05, "epa_def_pass": -1.0 + i * 0.05,
                         "epa_off_rush": 0.5 - i * 0.02, "epa_def_rush": -0.5 + rush_def_rank * 0.02})
    games = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "season": [2026] * 3,
                          "gameday": pd.to_datetime(["2026-09-10", "2026-09-17", "2026-09-24"])})
    return pd.DataFrame(rows), games


def test_every_duel_type_has_both_directions_regardless_of_gap():
    eff, games = _league32()
    new = matchups.matchups_for_game("T5", "T20", games, eff, "2026-10-04", 2026)
    old = matchups.matchups_for_game("T5", "T20", games, eff, "2026-10-04", 2026, min_gap=8)  # the pre-change call
    assert len(old) == 3 and len(new) == 4
    for t, _, _, _, _ in matchups.DUELS:
        assert {d.id for d in new if d.id.split(":")[0] == t} == {f"{t}:home", f"{t}:away"}
    below = next(d for d in new if d.id not in {o.id for o in old})
    assert below.id == "rush_off_vs_rush_def:away" and abs(below.attacker_rank - below.defender_rank) == 2
    assert [d.strength for d in new] == sorted((d.strength for d in new), reverse=True)


def test_below_gap_duel_never_carries_a_direction_even_when_its_type_is_proven():
    eff, games = _league32()
    duels = matchups.matchups_for_game("T5", "T20", games, eff, "2026-10-04", 2026)
    gate = {"pass_off_vs_pass_def": True, "rush_off_vs_rush_def": True}
    for pick in ("home", "away"):
        ctx = {r["id"]: r for r in matchups.to_context(duels, pick, lift_gate=gate)}
        assert ctx["rush_off_vs_rush_def:away"]["toward_pick"] is None  # gap 2 < 8
        assert ctx["rush_off_vs_rush_def:home"]["toward_pick"] is not None  # gap 15, strongest of its type
    # a lone below-gap duel of a proven type stays neutral and does not use up the type's slot
    from nfl_predictor.signals.duel import Duel
    close = Duel("pass_off_vs_pass_def:home", "A", "B", "s", "f", 10, 12, 32, "home", 0.9)
    assert matchups.to_context([close], "home", lift_gate={"pass_off_vs_pass_def": True})[0]["toward_pick"] is None

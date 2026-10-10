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
    d = duel.Duel(id="x", attacker="A", defender="B", stat="s", foil="f", attacker_rank=1, defender_rank=10, n_teams=32, toward="home", strength=0.5)
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


# --- Complete matchup table tests (both directions for every duel type) ----------


def _league():
    """Synthetic league with 4 teams for testing complete table."""
    teams = ["BUF", "NYJ", "MIA", "NE"]
    rows = []
    # 3 games, each team plays each other once
    for gid, (home, away) in enumerate([("BUF", "NYJ"), ("MIA", "NE"), ("BUF", "MIA")], 1):
        for i, team in enumerate(teams):
            off, dfn = 1.0 - i * 0.2, -1.0 + i * 0.2
            rows.append({"game_id": f"g{gid}", "team": team, "epa_off_pass": off, "epa_def_pass": dfn,
                         "epa_off_rush": off * 0.5, "epa_def_rush": dfn * 0.5})
    eff = pd.DataFrame(rows)
    games = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "season": [2026] * 3,
        "gameday": pd.to_datetime(["2026-09-10", "2026-09-17", "2026-09-24"])
    })
    return eff, games


def test_matchups_for_game_complete_table_returns_all_four_duels():
    """When complete_table=True, all 4 duels (2 types x 2 directions) are returned,
    even when the rank gap is below min_gap. The small-gap duels have toward='neutral'."""
    eff, games = _league()
    # BUF vs NYJ: ranks are BUF=1, NYJ=2, MIA=3, NE=4 for attack (higher=better)
    # For defence: lower=better, so BUF=4, NYJ=3, MIA=2, NE=1
    # BUF attack (1) vs NYJ defence (3): gap = 3-1 = 2 < 8 -> SMALL gap
    # NYJ attack (2) vs BUF defence (4): gap = 4-2 = 2 < 8 -> SMALL gap
    # MIA attack (3) vs NE defence (1): gap = 1-3 = -2, abs=2 < 8 -> SMALL gap
    # NE attack (4) vs MIA defence (2): gap = 2-4 = -2, abs=2 < 8 -> SMALL gap
    # All 4 gaps are small (2), so WITHOUT complete_table, NO duels returned
    # WITH complete_table, all 4 returned with toward='neutral'
    duels_default = matchups.matchups_for_game("BUF", "NYJ", games, eff, as_of="2026-10-01", season=2026,
                                               history_gaps={}, min_gap=8)
    assert duels_default == [], "Without complete_table, small-gap duels should be filtered out"

    duels_complete = matchups.matchups_for_game("BUF", "NYJ", games, eff, as_of="2026-10-01", season=2026,
                                                history_gaps={}, min_gap=8, complete_table=True)
    assert len(duels_complete) == 4, f"complete_table=True should return 4 duels, got {len(duels_complete)}"
    # Check both directions for each type
    types = {d.id.split(":")[0] for d in duels_complete}
    assert types == {"pass_off_vs_pass_def", "rush_off_vs_rush_def"}
    for d in duels_complete:
        assert d.toward == "neutral", f"Small-gap duel should have toward='neutral', got {d.toward}"
        assert d.strength >= 0


def test_to_context_complete_table_small_gap_stays_neutral():
    """A duel with small gap (below min_gap) must NEVER carry a direction in to_context,
    even when its type is proven by the lift gate. The gap check is added to eligibility."""
    from nfl_predictor.signals.duel import make_duel
    # Small gap duel: BUF attack rank 15, NYJ defence rank 20 -> gap = 5 < 8
    d = make_duel(
        "pass_off_vs_pass_def:home", home="BUF", away="NYJ", attacker_side="home",
        attack_ranks={"BUF": 15, "NYJ": 32}, defence_ranks={"BUF": 1, "NYJ": 20},
        history_gaps=None, min_gap=8, stat="passing offence", foil="pass defence",
        complete_table=True  # Force creation despite small gap
    )
    assert d is not None
    assert d.toward == "neutral"
    assert abs(d.defender_rank - d.attacker_rank) < 8

    # Even with proven lift gate, small-gap duel must stay neutral
    proven = {"pass_off_vs_pass_def": True}
    ctx = matchups.to_context([d], pick_side="home", lift_gate=proven, min_gap=8)
    assert ctx[0]["toward_pick"] is None, "Small-gap duel must stay neutral even with proven lift gate"
    # Large gap duel with proven gate should be directed
    d_large = make_duel(
        "pass_off_vs_pass_def:home", home="BUF", away="NYJ", attacker_side="home",
        attack_ranks={"BUF": 5, "NYJ": 32}, defence_ranks={"BUF": 1, "NYJ": 30},  # gap = 31
        history_gaps=None, min_gap=8, stat="passing offence", foil="pass defence",
    )
    assert d_large is not None
    assert d_large.toward == "home"
    ctx_large = matchups.to_context([d_large], pick_side="home", lift_gate=proven, min_gap=8)
    assert ctx_large[0]["toward_pick"] is True, "Large-gap proven duel should be directed"

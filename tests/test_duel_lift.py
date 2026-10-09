"""Residual-lift gate (AI plan Task 10): a duel type only passes when its edge
is bigger than the model's own noise across enough held-out games."""
from __future__ import annotations

import numpy as np

from nfl_predictor.tools.duel_lift import lift, load_lift_results


def _rows(effect: float, n: int = 600, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    toward = rng.choice([-1, 1], n)
    model = rng.normal(0, 6, n)
    actual = model + effect * toward + rng.normal(0, 10, n)
    return [{"toward": int(t), "actual_margin": float(a), "model_margin": float(m)}
            for t, a, m in zip(toward, actual, model)]


def test_a_real_edge_passes():
    out = lift(_rows(effect=3.0))
    assert out["n"] == 600
    assert out["passes"] is True


def test_pure_noise_fails():
    out = lift(_rows(effect=0.0))
    assert out["passes"] is False


def test_too_few_games_never_passes():
    out = lift(_rows(effect=3.0, n=50))
    assert out["passes"] is False
    assert out["n"] == 50


def test_load_absent_file_is_empty(tmp_path):
    assert load_lift_results(tmp_path / "missing.json") == {}


def test_load_reads_passes_flags(tmp_path):
    p = tmp_path / "duel_lift.json"
    p.write_text('{"pass_off_vs_pass_def": {"n": 300, "passes": true}, '
                 '"rush_off_vs_rush_def": {"n": 100, "passes": false}}')
    assert load_lift_results(p) == {"pass_off_vs_pass_def": True, "rush_off_vs_rush_def": False}


def test_load_rejects_non_object_roots_and_non_bool_passes(tmp_path):
    """A top-level list/null is not a results table, and a string like
    \"false\" must not be truthy -- anything unverifiable is simply unproven."""
    for payload in ('[]', 'null', '"passes"',
                    '{"pass_off_vs_pass_def": {"passes": "false"}}',
                    '{"pass_off_vs_pass_def": {"passes": 1}}'):
        p = tmp_path / "bad.json"
        p.write_text(payload)
        assert load_lift_results(p) == {}, f"payload {payload!r} must be unproven"


def test_lift_counts_independent_games_not_rows():
    """MIN_N applies to unique games: one game entering twice (both directional
    duels, pre-collapse data) must not nearly double the sample size."""
    rows = _rows(effect=3.0, n=150)
    # Duplicate each row: 150 games, 300 rows. Still below MIN_N games.
    doubled = [dict(r, game_id=f"g{i // 2}") for i, r in enumerate(rows * 2)]
    out = lift(doubled)
    assert out["n_games"] == 150
    assert out["passes"] is False, "150 independent games must not pass the 200-game gate"
    assert out["n"] == 300


def test_build_rows_runs_the_full_walk_forward_pipeline(tmp_path):
    """The generator produces per-duel-type rows from out-of-fold margins on the
    same fixture data block_eval walks forward on -- proving the tool runs end
    to end (folds -> ridge margins -> duels -> rows), not just the pure core.
    One row per (game, duel type): the two directional lenses of a type cancel
    in the lift mean, so rows must never double-count a game."""
    import pandas as pd
    from epa_fixtures import make_games, make_pbp
    from nfl_predictor.data import pbp_agg
    from nfl_predictor.features import build
    from nfl_predictor.tools.duel_lift import build_rows, lift

    games = make_games(seasons=(2023, 2024, 2025), weeks=6)
    pbp = make_pbp(games)
    aux = build.Aux(pbp_agg.team_game_efficiency(pbp), pbp_agg.qb_games(pbp))

    rows = build_rows(games, aux, min_gap=4)
    assert rows, "the walk-forward held-out games must produce some duels"
    assert all(r["toward"] in (-1, 1) for r in rows)
    assert all(r["gap"] >= 1 for r in rows)
    types = {r["duel"] for r in rows}
    assert types, "duel types are the split key (pass_off_vs_pass_def, ...)"
    # One observation per game per type -- independent games are the unit.
    for t in types:
        game_ids = [r["game_id"] for r in rows if r["duel"] == t]
        assert len(game_ids) == len(set(game_ids)), f"type {t} double-counts games"

    summary = {t: lift([r for r in rows if r["duel"] == t]) for t in types}
    assert all("passes" in s and "ci" in s for s in summary.values())
    assert all(s["n_games"] == sum(1 for r in rows if r["duel"] == t) for t, s in summary.items())
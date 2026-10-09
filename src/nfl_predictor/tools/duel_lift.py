"""Residual-lift honesty gate for duel types (AI plan Task 10).

A duel can be true and explain nothing the model has not already priced.
Before a duel type may claim Edge or Risk, check that, across the
walk-forward held-out games, games where it favoured a side ended better for
that side than the model expected (`toward * (actual_margin - model_margin)`
averaged positively).

`lift()` is the pure core. `__main__` builds the rows from this repository's
own walk-forward fold machinery (evaluate.walk_forward) plus the duel
evaluator -- out-of-fold predicted margins only, never in-sample -- and
writes `data/duel_lift.json` ({duel type -> lift summary}) and
`data/duel_gaps.json` ({duel type -> past |rank gap|s}). An absent file means
the gate has not run; `to_context` then keeps every duel neutral.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_N = 200  # below this the CI cannot be trusted; the type stays neutral

_ROOT = Path(__file__).resolve().parents[3]  # src/nfl_predictor/tools -> repo root


def lift(rows, seed: int = 0, n_boot: int = 2000) -> dict:
    """Bootstrap 95% CI of the mean residual lift. Passes when there are enough
    games and the lower bound is positive -- an effect the noise could have
    produced is not proof."""
    rows = [r for r in rows if r["toward"] in (-1, 1)]
    n = len(rows)
    if n < MIN_N:
        return {"n": n, "mean_lift": None, "ci": (None, None), "passes": False}
    x = np.asarray([r["toward"] * (float(r["actual_margin"]) - float(r["model_margin"])) for r in rows], float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"n": n, "mean_lift": float(x.mean()), "ci": (lo, hi), "passes": bool(lo > 0)}


def load_lift_results(path: str | Path | None = None) -> dict[str, bool]:
    """duel type -> proven, from data/duel_lift.json. Absent/unreadable file is
    {} -- the gate has simply not run, so nothing is proven."""
    path = Path(path) if path is not None else _ROOT / "data" / "duel_lift.json"
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: bool(v.get("passes")) for k, v in raw.items() if isinstance(v, dict) and "passes" in v}


def build_rows(games_df, aux, min_gap: int = 8, candidate: str = "ridge") -> list[dict]:
    """Per-duel-type rows over the walk-forward held-out games (out-of-fold
    margins only). as_of for each val game is its own kickoff, so a duel never
    sees post-kickoff efficiency. Each row also carries the duel's |rank gap|
    so `data/duel_gaps.json` comes from the same games, not a second pass.
    `min_gap` mirrors facts' gate (default 8); the tests drop it to 4 because
    the synthetic fixture has only 8 teams, which cannot span a gap of 8."""
    import pandas as pd

    from nfl_predictor.evaluate import walk_forward as wf
    from nfl_predictor.signals.matchups import matchups_for_game

    rows: list[dict] = []
    for fold in wf.prepare_folds(games_df, aux=aux):
        preds, _ = wf._predict_margins(candidate, fold["train_df"], fold["val_df"], fold["feature_cols"])
        for game, model_margin in zip(fold["val_df"].itertuples(index=False), preds):
            as_of = pd.Timestamp(game.gameday)
            duels = matchups_for_game(game.home_team, game.away_team, games_df,
                                      aux.efficiency, as_of, int(game.season), min_gap=min_gap)
            for d in duels:
                rows.append({
                    "duel": d.id.split(":")[0],
                    "toward": 1 if d.toward == "home" else -1,
                    "gap": abs(d.attacker_rank - d.defender_rank),
                    "actual_margin": float(game.margin),
                    "model_margin": float(model_margin),
                })
    return rows


if __name__ == "__main__":
    from nfl_predictor.api import routes
    from nfl_predictor.data import schedules

    games = schedules.load_training_data(schedules.default_completed_seasons(n=8))
    print(f"building rows over {len(games)} completed-season games (this refits the margin model per fold)...")
    rows = build_rows(games, routes._load_completed_aux_cached())

    types = sorted({r["duel"] for r in rows})
    lifts = {t: lift([r for r in rows if r["duel"] == t]) for t in types}
    gaps = {t: [r["gap"] for r in rows if r["duel"] == t] for t in types}

    out = _ROOT / "data"
    out.mkdir(exist_ok=True)
    (out / "duel_lift.json").write_text(json.dumps(lifts, indent=2) + "\n")
    (out / "duel_gaps.json").write_text(json.dumps(gaps, indent=2) + "\n")

    for t, s in sorted(lifts.items()):
        ci = f"[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]" if s["ci"][0] is not None else "n/a"
        print(f"{t:28s} n={s['n']:4d}  lift={s['mean_lift']}  ci={ci}  passes={s['passes']}")
    print(f"\nwrote {out / 'duel_lift.json'} and {out / 'duel_gaps.json'}")
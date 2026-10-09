"""Decide whether a feature block earns its place. Paired bootstrap on identical held-out games."""
from __future__ import annotations

import numpy as np

from nfl_predictor.evaluate import walk_forward as wf
from nfl_predictor.models import game_outcome


def paired_bootstrap(a, b, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of mean(a - b). Positive means b is better when lower is better."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def calibration_gap(probs, y, buckets: int = 5) -> float:
    probs, y = np.asarray(probs, float), np.asarray(y, float)
    edges = np.quantile(probs, np.linspace(0, 1, buckets + 1))
    idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, buckets - 1)
    gaps = [abs(probs[idx == b].mean() - y[idx == b].mean()) for b in range(buckets) if (idx == b).any()]
    return float(max(gaps)) if gaps else 0.0


def _per_game(folds, candidate):
    errs, briers, probs, ys = [], [], [], []
    for fold in folds:
        preds, sigma = wf._predict_margins(candidate, fold["train_df"], fold["val_df"], fold["feature_cols"])
        margin = fold["val_df"]["margin"].to_numpy()
        p = np.array([wf.game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds])
        errs.append(np.abs(preds - margin))
        y = (margin > 0).astype(float)
        briers.append((p - y) ** 2)
        probs.append(p)
        ys.append(y)
    return tuple(np.concatenate(x) for x in (errs, briers, probs, ys))


def evaluate_block(games_df, aux, block: str, candidate: str = "ridge") -> dict:
    base = _per_game(wf.prepare_folds(games_df), candidate)
    withb = _per_game(wf.prepare_folds(games_df, blocks=(block,), aux=aux), candidate)
    assert len(base[0]) == len(withb[0]), f"block {block} drops games: {len(base[0])} vs {len(withb[0])}"
    mae_ci, brier_ci = paired_bootstrap(base[0], withb[0]), paired_bootstrap(base[1], withb[1])
    gap_base, gap_block = calibration_gap(base[2], base[3]), calibration_gap(withb[2], withb[3])
    return {
        "block": block, "n_games": int(len(base[0])),
        "mae_delta": float(base[0].mean() - withb[0].mean()), "mae_ci": mae_ci,
        "brier_delta": float(base[1].mean() - withb[1].mean()), "brier_ci": brier_ci,
        "gap_base": gap_base, "gap_block": gap_block,
        "clears": bool(mae_ci[0] > 0 and brier_ci[0] > 0 and gap_block <= gap_base),
    }
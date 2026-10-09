"""Decide whether a feature block earns its place. Paired bootstrap on identical held-out games."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_predictor.evaluate import walk_forward as wf
from nfl_predictor.models import game_outcome
from nfl_predictor.data import schedules, pbp_agg
from nfl_predictor.data.pbp_agg import load_pbp_agg, team_game_efficiency, qb_games
from nfl_predictor.features import build as feature_build


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


def load_recent_nfl_data(start_year: int = 2000, end_year: int = None) -> tuple[pd.DataFrame, feature_build.Aux]:
    """Load NFL data for evaluation."""
    if end_year is None:
        end_year = datetime.now().year
    seasons = list(range(start_year, end_year + 1))
    
    print(f"Loading NFL data for seasons {seasons}...")
    
    # Load games data
    games_df = schedules.load_training_data(seasons)
    
    # Load PBP aggregates for aux
    efficiency_frames = []
    qb_frames = []
    
    for season in seasons:
        try:
            pbp = load_pbp_agg(season)
            if not pbp.empty:
                efficiency_frames.append(team_game_efficiency(pbp))
                qb_frames.append(qb_games(pbp))
        except Exception as e:
            print(f"Warning: Could not load PBP data for season {season}: {e}")
    
    efficiency = pd.concat(efficiency_frames, ignore_index=True) if efficiency_frames else pd.DataFrame()
    qb_games_df = pd.concat(qb_frames, ignore_index=True) if qb_frames else pd.DataFrame()
    
    # Create Aux object
    aux = feature_build.Aux(efficiency=efficiency, qb_games=qb_games_df, upcoming_starters={})
    
    return games_df, aux


def main():
    parser = argparse.ArgumentParser(description="Evaluate NFL feature blocks")
    parser.add_argument(
        "--blocks", 
        nargs="+", 
        required=True,
        help="Feature blocks to evaluate (e.g., epa qb conditions)"
    )
    parser.add_argument(
        "--start-year", 
        type=int, 
        default=2000,
        help="First season to include (default: 2000)"
    )
    parser.add_argument(
        "--end-year", 
        type=int, 
        default=None,
        help="Last season to include (default: current year)"
    )
    parser.add_argument(
        "--candidate", 
        type=str, 
        default="ridge",
        help="Candidate model to use (default: ridge)"
    )
    
    args = parser.parse_args()
    
    # Load data
    games_df, aux = load_recent_nfl_data(start_year=args.start_year, end_year=args.end_year)
    
    if games_df.empty:
        print("Error: No games data loaded.")
        return 1
        
    print(f"Loaded {len(games_df)} games for evaluation.")
    print(f"Evaluating blocks: {', '.join(args.blocks)}")
    print(f"Using candidate model: {args.candidate}")
    print()
    
    # Evaluate each block
    results = []
    for block in args.blocks:
        print(f"Evaluating block: {block}...")
        try:
            result = evaluate_block(games_df, aux, block, args.candidate)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating block {block}: {e}")
            import traceback
            traceback.print_exc()
            results.append(None)
    
    # Print results in a table format suitable for PR description
    print("\n" + "="*80)
    print("NFL BLOCK EVALUATION RESULTS")
    print("="*80)
    print(f"{'Block':<12} {'N Games':<10} {'MAE Δ':<12} {'MAE 95% CI':<20} {'Brier Δ':<12} {'Brier 95% CI':<20} {'Gap Base':<10} {'Gap Block':<10} {'Clears'}")
    print("-"*80)
    
    for result in results:
        if result is None:
            print(f"{'ERROR':<12} {'N/A':<10} {'N/A':<12} {'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<10} {'N/A':<10} {'N/A'}")
            continue
            
        block = result['block']
        n_games = result['n_games']
        mae_delta = result['mae_delta']
        mae_ci = result['mae_ci']
        brier_delta = result['brier_delta']
        brier_ci = result['brier_ci']
        gap_base = result['gap_base']
        gap_block = result['gap_block']
        clears = result['clears']
        
        # Format confidence intervals as strings
        mae_ci_str = f"[{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]"
        brier_ci_str = f"[{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]"
        
        print(f"{block:<12} {n_games:<10} {mae_delta:<12.4f} {mae_ci_str:<20} {brier_delta:<12.4f} {brier_ci_str:<20} {gap_base:<10.4f} {gap_block:<10.4f} {'Yes' if clears else 'No'}")
    
    print("="*80)
    
    return 0


if __name__ == "__main__":
    main()

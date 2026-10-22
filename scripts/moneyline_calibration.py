#!/usr/bin/env python3
"""moneyline_calibration.py — walk-forward Platt scaling for moneyline (home_win_prob)."""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
import nfl_data_py as nfl


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for margin regression."""
    df = games.copy()
    
    # Targets
    df['margin'] = df['home_score'] - df['away_score']
    df['actual_home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    # Pre-game features
    df['spread_line'] = df['spread_line'].fillna(0)
    df['home_rest_days'] = df['home_rest'].fillna(7)
    df['away_rest_days'] = df['away_rest'].fillna(7)
    df['div_game'] = df['div_game'].fillna(0).astype(int)
    
    # Rolling points scored/allowed (season-to-date)
    df['home_points_scored_roll'] = df.groupby('home_team')['home_score'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['home_score'].mean())
    df['home_points_allowed_roll'] = df.groupby('home_team')['away_score'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['away_score'].mean())
    df['away_points_scored_roll'] = df.groupby('away_team')['away_score'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['away_score'].mean())
    df['away_points_allowed_roll'] = df.groupby('away_team')['home_score'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['home_score'].mean())
    
    # Rolling point differential (Elo proxy)
    df['home_point_diff_roll'] = df.groupby('home_team').apply(
        lambda x: (x['home_score'] - x['away_score']).shift(1).expanding().mean()
    ).reset_index(level=0, drop=True).fillna(0)
    df['away_point_diff_roll'] = df.groupby('away_team').apply(
        lambda x: (x['away_score'] - x['home_score']).shift(1).expanding().mean()
    ).reset_index(level=0, drop=True).fillna(0)
    df['rating_diff'] = df['home_point_diff_roll'] - df['away_point_diff_roll']
    
    # Fill any remaining NaN
    for col in ['home_points_scored_roll', 'home_points_allowed_roll',
                'away_points_scored_roll', 'away_points_allowed_roll',
                'rating_diff']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


FEATURE_COLS = [
    'rating_diff',
    'home_points_scored_roll', 'home_points_allowed_roll',
    'away_points_scored_roll', 'away_points_allowed_roll',
    'home_rest_days', 'away_rest_days',
    'div_game',
    'spread_line',
]


def run_walkforward_calibration() -> dict:
    """Run walk-forward calibration on historical NFL data."""
    
    # Load historical NFL data
    print("Loading NFL data...")
    import nfl_data_py as nfl
    games = nfl.import_schedules(range(2018, 2027))
    completed = games[games['home_score'].notna() & games['away_score'].notna()].copy()
    print(f"Total completed games: {len(completed)}")
    print(f"Seasons: {sorted(completed['season'].unique())}")
    
    # Build features
    print("Building features...")
    df = build_features(completed)
    
    # Keep only rows with all features
    feature_cols = [
        'rating_diff',
        'home_points_scored_roll', 'home_points_allowed_roll',
        'away_points_scored_roll', 'away_points_allowed_roll',
        'home_rest_days', 'away_rest_days',
        'div_game',
        'spread_line',
    ]
    df = df.dropna(subset=['margin', 'actual_home_win'] + 
                   ['rating_diff', 'home_points_scored_roll', 'home_points_allowed_roll',
                    'away_points_scored_roll', 'away_points_allowed_roll',
                    'home_rest_days', 'away_rest_days', 'div_game', 'spread_line']).copy()
    
    print(f"Games with full features: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    
    # Walk-forward by season
    seasons = sorted(df['season'].unique())
    results = []
    
    for val_season in sorted(df['season'].unique()):
        train_df = df[df['season'] < val_season]
        val_df = df[df['season'] == val_season]
        
        if len(train_df) < 50 or len(val_df) < 10:
            print(f"Season {val_season}: insufficient data (train={len(train_df)}, val={len(val_df)})")
            continue
        
        print(f"\nSeason {val_season}: train={len(train_df)}, val={len(val_df)}")
        
        # Fit Ridge on training
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        feat_cols = ['rating_diff', 'home_points_scored_roll', 'home_points_allowed_roll',
                     'away_points_scored_roll', 'away_points_allowed_roll',
                     'home_rest_days', 'away_rest_days', 'div_game', 'spread_line']
        
        X_train = train_df[['rating_diff', 'home_points_scored_roll', 'home_points_allowed_roll',
                            'away_points_scored_roll', 'away_points_allowed_roll',
                            'home_rest_days', 'away_rest_days', 'div_game', 'spread_line']].fillna(0)
        y_margin = train_df['margin']
        model.fit(X_train, y_margin)
        
        # Predict on validation
        X_val = val_df[['rating_diff', 'home_points_scored_roll', 'home_points_allowed_roll',
                        'away_points_scored_roll', 'away_points_allowed_roll',
                        'home_rest_days', 'away_rest_days', 'div_game', 'spread_line']].fillna(0)
        val_preds = model.predict(X_val)
        
        # Sigma from training residuals (honest)
        train_preds = model.predict(train_df[['rating_diff', 'home_points_scored_roll', 'home_points_allowed_roll',
                            'away_points_scored_roll', 'away_points_allowed_roll',
                            'home_rest_days', 'away_rest_days', 'div_game', 'spread_line']].fillna(0))
        train_residuals = train_df['margin'].values - train_preds
        sigma = float(np.std(train_residuals, ddof=1))
        
        # Predict probabilities on validation
        val_probs_raw = np.array([1 / (1 + np.exp(-m / sigma)) for m in model.predict(X_val)])
        y_val = (val_df['home_score'] > val_df['away_score']).astype(int).values
        spreads = val_df['spread_line'].values
        
        # Fit Platt scaling on training raw probs
        train_probs_raw = np.array([1 / (1 + np.exp(-m / sigma)) for m in model.predict(train_df[['rating_diff', 'home_points_scored_roll', 'home_points_allowed_roll',
                            'away_points_scored_roll', 'away_points_allowed_roll',
                            'home_rest_days', 'away_rest_days', 'div_game', 'spread_line']].fillna(0))])
        train_y = (train_df['home_score'] > train_df['away_score']).astype(int).values
        
        # Fit Platt on training
        p_train = np.clip(train_probs_raw, 1e-6, 1 - 1e-6)
        y_train = (train_df['home_score'] > train_df['away_score']).astype(int).values
        logit_p = np.log(p_train / (1 - p_train))
        
        def nll(params):
            a, b = params
            logit_cal = a * logit_p + b
            p_cal = 1 / (1 + np.exp(-logit_cal))
            p_cal = np.clip(p_cal, 1e-15, 1 - 1e-15)
            return -np.sum(y_train * np.log(p_cal) + (1 - y_train) * np.log(1 - p_cal))
        
        res = minimize(nll, x0=[1.5, 0.0], bounds=[(0.1, 10.0), (-5.0, 5.0)], method="L-BFGS-B")
        if not res.success:
            a, b = 1.0, 0.0
        else:
            a, b = float(res.x[0]), float(res.x[1])
        
        # Apply to validation
        val_probs_raw = np.array([1 / (1 + np.exp(-m / sigma)) for m in model.predict(val_df[['rating_diff', 'home_points_scored_roll', 'home_points_allowed_roll',
                            'away_points_scored_roll', 'away_points_allowed_roll',
                            'home_rest_days', 'away_rest_days', 'div_game', 'spread_line']].fillna(0))])
        y_val = (val_df['home_score'] > val_df['away_score']).astype(int).values
        spreads = val_df['spread_line'].values
        
        p_cal = 1 / (1 + np.exp(-(a * np.log(np.clip(val_probs_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(val_probs_raw, 1e-6, 1 - 1e-6))) + b)))
        
        from sklearn.metrics import brier_score_loss, log_loss
        
        brier_raw = brier_score_loss(y_val, val_probs_raw)
        brier_cal = brier_score_loss(y_val, p_cal)
        logloss_raw = log_loss(y_val, np.clip(val_probs_raw, 1e-6, 1 - 1e-6))
        logloss_cal = log_loss(y_val, np.clip(p_cal, 1e-15, 1 - 1e-15))
        
        brier_delta = brier_cal - brier_raw
        logloss_delta = logloss_cal - log_loss(y_val, np.clip(val_probs_raw, 1e-6, 1 - 1e-6))
        
        # Fav >=7pt bias
        fav_mask = val_df['spread_line'].values >= 7
        if fav_mask.any():
            fav_y = val_df.loc[fav_mask, 'actual_home_win'].values
            fav_raw = val_probs_raw[fav_mask]
            fav_p = np.clip(fav_raw, 1e-6, 1 - 1e-6)
            fav_cal = 1 / (1 + np.exp(-(a * np.log(fav_p / (1 - fav_p)) + b)))
            
            bias_raw = np.mean(fav_raw) - np.mean(fav_y)
            bias_cal = np.mean(p_cal[fav_mask]) - np.mean(fav_y)
            bias_delta = bias_cal - bias_raw
            n_fav7 = int(fav_mask.sum())
        else:
            bias_raw = 0
            bias_cal = 0
            bias_delta = 0
            n_fav7 = 0
        
        result = {
            "season": int(val_season),
            "n_games": int(len(val_df)),
            "brier_raw": float(brier_score_loss(y_val, val_probs_raw)),
            "brier_cal": float(brier_cal),
            "brier_delta": float(brier_cal - brier_score_loss(y_val, val_probs_raw)),
            "logloss_raw": float(log_loss(y_val, np.clip(val_probs_raw, 1e-6, 1 - 1e-6))),
            "logloss_cal": float(log_loss(y_val, np.clip(p_cal, 1e-15, 1 - 1e-15))),
            "logloss_delta": float(log_loss(y_val, np.clip(p_cal, 1e-15, 1 - 1e-15)) - log_loss(y_val, np.clip(val_probs_raw, 1e-6, 1 - 1e-6))),
            "a": float(a),
            "b": float(b),
            "bias_raw": float(bias_raw) if fav_mask.any() else 0,
            "bias_cal": float(bias_cal) if fav_mask.any() else 0,
            "bias_delta": float(bias_delta) if fav_mask.any() else 0,
            "n_fav7": int(fav_mask.sum()),
        }
        print(f"  Brier: {brier_score_loss(y_val, val_probs_raw):.4f} -> {brier_cal:.4f} (delta: {brier_delta:.4f})")
        print(f"  LogLoss: {log_loss(y_val, np.clip(val_probs_raw, 1e-6, 1 - 1e-6)):.4f} -> {log_loss(y_val, np.clip(p_cal, 1e-15, 1 - 1e-15)):.4f} (delta: {logloss_delta:.4f})")
        if fav_mask.any():
            print(f"  Fav>=7pt bias: {bias_raw:.4f} -> {bias_cal:.4f} (delta: {bias_delta:.4f})")
        
        results.append(result)
    
    if not results:
        return {"error": "No valid walk-forward folds"}
    
    # Bootstrap CIs over seasons
    print("\nRunning bootstrap CIs (2000 samples)...")
    n_boot = 2000
    
    brier_boot = []
    logloss_boot = []
    bias_boot = []
    
    for _ in range(2000):
        # Resample seasons with replacement
        idx = np.random.choice(len(results), len(results), replace=True)
        brier_boot.append(np.mean([results[i]['brier_delta'] for i in idx]))
        logloss_boot.append(np.mean([results[i]['logloss_delta'] for i in idx]))
        fav = [results[i] for i in idx if results[i]['n_fav7'] > 0]
        if fav:
            bias_boot.append(np.mean([r['bias_delta'] for r in fav]))
    
    brier_ci = (float(np.percentile(brier_boot, 2.5)), float(np.percentile(brier_boot, 97.5)))
    logloss_ci = (float(np.percentile(logloss_boot, 2.5)), float(np.percentile(logloss_boot, 97.5)))
    bias_ci = (float(np.percentile(bias_boot, 2.5)), float(np.percentile(bias_boot, 97.5))) if bias_boot else (np.nan, np.nan)
    
    print(f"\nBootstrap CIs (2000 samples):")
    print(f"  Brier delta CI: [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]")
    print(f"  LogLoss delta CI: [{logloss_ci[0]:.4f}, {logloss_ci[1]:.4f}]")
    print(f"  Fav>=7pt bias delta CI: [{bias_ci[0]:.4f}, {bias_ci[1]:.4f}]")
    
    # Gate evaluation
    gate_passed = True
    gate_reasons = []
    
    if brier_ci[1] >= 0:
        gate_passed = False
        gate_reasons.append(f"Brier delta CI not entirely < 0: [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]")
    
    if logloss_ci[1] >= 0:
        gate_passed = False
        gate_reasons.append(f"LogLoss delta CI not entirely < 0: [{logloss_ci[0]:.4f}, {logloss_ci[1]:.4f}]")
    
    if not np.isnan(bias_ci[0]) and bias_ci[1] >= 0:
        gate_passed = False
        gate_reasons.append(f"Fav>=7pt bias delta CI not entirely < 0: [{bias_ci[0]:.4f}, {bias_ci[1]:.4f}]")
    
    # Also check per-season
    for r in results:
        if r['brier_delta'] >= 0:
            gate_passed = False
            gate_reasons.append(f"Season {r['season']}: Brier delta >= 0 ({r['brier_delta']:.4f})")
        if r['logloss_delta'] >= 0:
            gate_passed = False
            gate_reasons.append(f"Season {r['season']}: LogLoss delta >= 0 ({r['logloss_delta']:.4f})")
        if r['n_fav7'] > 0 and r['bias_delta'] >= 0:
            gate_passed = False
            gate_reasons.append(f"Season {r['season']}: Fav>=7pt bias delta >= 0 ({r['bias_delta']:.4f})")
    
    gate_result = "PASSED" if gate_passed else "FAILED"
    print(f"\n=== SHIP GATE: {gate_result} ===")
    if not gate_passed:
        for reason in gate_reasons:
            print(f"  - {reason}")
    
    return {
        "per_season": results,
        "overall": {
            "mean_brier_delta": float(np.mean([r['brier_delta'] for r in results])),
            "mean_logloss_delta": float(np.mean([r['logloss_delta'] for r in results])),
            "mean_fav7_bias_delta": float(np.mean([r['bias_delta'] for r in results if r['n_fav7'] > 0])),
        },
        "cis": {
            "brier_delta": brier_ci,
            "logloss_delta": logloss_ci,
            "fav7_bias_delta": bias_ci,
        },
        "gate": {
            "passed": gate_passed,
            "reasons": gate_reasons,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward moneyline calibration")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()
    
    result = run_walkforward_calibration()
    
    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Exit with error code if gate failed
    if not result.get('gate', {}).get('passed', False):
        sys.exit(1)


if __name__ == "__main__":
    main()

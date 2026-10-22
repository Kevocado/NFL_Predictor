#!/usr/bin/env python3
"""moneyline_calibration.py — walk-forward moneyline calibration: the numbers and the ship gate.

Fits a Platt stretch on the moneyline probability WALK-FORWARD (train on season N-1 and earlier,
test on season N) and reports, for every test season and pooled:

  * Brier score and log-loss, raw and corrected, with a paired bootstrap CI on the delta
  * the SIGNED calibration bias (predicted - actual) before and after, per bucket
  * the PAIRED bootstrap CI of |bias_corrected| - |bias_raw| per bucket

The ship gate:
  1. Brier delta CI entirely < 0
  2. log-loss delta CI entirely < 0
  3. |bias| delta CI entirely < 0 for the >=7pt favourite bucket
  4. no OTHER bucket's |bias| significantly worse (its |bias| delta CI not entirely > 0)

Run:
    python scripts/moneyline_calibration.py --output calibration_results.json

Exits 0 when the gate passes and 1 when it fails.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from nfl_predictor.evaluate import walk_forward
from nfl_predictor.models import game_outcome
from nfl_predictor.models.moneyline_calibration import (
    abs_bias_delta,
    calibration_bias,
    fit_platt,
    paired_abs_bias_ci,
    platt_transform,
    spread_bucket,
)

BUCKETS = ("fav>=7", "dog<=-7", "other")
N_BOOT = 2000
SEED = 20261022


def _brier_delta_ci(p_raw, p_cal, y, n_boot: int = N_BOOT, seed: int = SEED):
    """Paired bootstrap CI of the Brier delta (corrected - raw) over the same resampled games."""
    n = len(y)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        deltas[i] = brier_score_loss(yy, p_cal[idx]) - brier_score_loss(yy, p_raw[idx])
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def _logloss_delta_ci(p_raw, p_cal, y, n_boot: int = N_BOOT, seed: int = SEED):
    """Paired bootstrap CI of the log-loss delta (corrected - raw) over the same resampled games."""
    n = len(y)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        deltas[i] = (log_loss(yy, np.clip(p_cal[idx], 1e-15, 1 - 1e-15))
                     - log_loss(yy, np.clip(p_raw[idx], 1e-6, 1 - 1e-6)))
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def _load_games(seasons) -> pd.DataFrame:
    """The completed games for the given seasons, from nflverse.

    Only finished games (both scores present) are usable: a game with no score has no outcome to
    calibrate against, and including it would put an unresolved row in the denominator of every
    metric in this report.
    """
    import nfl_data_py as nfl

    games = nfl.import_schedules(list(seasons))
    return games[games["home_score"].notna() & games["away_score"].notna()].copy()


def _diagnostic(p, y, s) -> dict:
    """What the OUT-OF-SAMPLE probabilities actually look like, before any correction.

    The audit that started this work reported a calibration slope of 3.65 on three bins, which
    says the probabilities are far too compressed and need stretching. This computes that number
    from the same walk-forward out-of-sample probabilities the gate is scored on, so the premise
    is checked against the data rather than assumed.

    The slope is fitted WITHOUT regularisation. `LogisticRegression` penalises by default, and an
    L2 penalty shrinks a slope toward zero, so a penalised fit would understate the compression
    and make the model look BETTER calibrated than it is -- which is the opposite of what this
    diagnostic is for. `C=1e6` is the unpenalised limit on every scikit-learn version.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    slope = float(lr.coef_[0][0])
    bins = np.clip((p * 3).astype(int), 0, 2)
    reliability = []
    for b in range(3):
        sel = bins == b
        if not sel.any():
            continue
        reliability.append({
            "bin": f"{b / 3:.2f}-{(b + 1) / 3:.2f}",
            "n": int(sel.sum()),
            "predicted": float(p[sel].mean()),
            "actual": float(y[sel].mean()),
            "actual_minus_predicted": float(y[sel].mean() - p[sel].mean()),
        })
    return {
        "pooled_logistic_slope": slope,
        "pooled_logistic_intercept": float(lr.intercept_[0]),
        "reliability_3bin": reliability,
        "buckets": {
            bucket: {
                "n": int((s >= 7).sum() if bucket == "fav>=7" else
                         (s <= -7).sum() if bucket == "dog<=-7" else
                         ((s > -7) & (s < 7)).sum()),
            }
            for bucket in BUCKETS
        },
    }


def run(min_train_seasons: int = 2) -> dict:
    """Run the walk-forward calibration and the ship gate, and return both.

    `min_train_seasons` is how many seasons a fold must have behind it before it can be tested,
    so a season with no earlier seasons is never scored against a model that saw them.
    """
    games = _load_games(range(2018, 2027))
    print(f"Completed games: {len(games)}  seasons: {sorted(games['season'].unique())}")

    folds = walk_forward.prepare_folds(games, min_train_seasons=min_train_seasons)
    print(f"Walk-forward folds: {[(f['val_season'], len(f['train_df']), len(f['val_df'])) for f in folds]}")

    # The SERVING sigma, not a per-fold one.
    #
    # `manifest.fit_sigmas` -- the function that fills the manifest the serving path reads -- takes
    # its sigma from the POOLED out-of-fold residuals over every fold. A walk-forward that used a
    # per-fold sigma instead would be evaluating a model that is not the one being served, and the
    # raw probabilities it calibrated would come from a different distribution than the served ones.
    sigma = game_outcome.sigma_from_residuals(walk_forward.oof_residuals(folds, "ridge", "margin"))
    print(f"Serving sigma (pooled OOF residuals): {sigma:.4f}")

    per_season = []
    pooled = {b: {"raw": [], "cal": [], "y": []} for b in BUCKETS}
    all_raw, all_cal, all_y, all_s = [], [], [], []

    for fold in folds:
        season = fold["val_season"]
        train_df, val_df = fold["train_df"], fold["val_df"]
        feature_cols = fold["feature_cols"]

        # The model, fitted on the training seasons only.
        train_margins, _ = walk_forward._predict_margins("ridge", train_df, train_df, feature_cols)
        val_margins, _ = walk_forward._predict_margins("ridge", train_df, val_df, feature_cols)

        train_raw = np.array([
            game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in train_margins
        ])
        val_raw = np.array([
            game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in val_margins
        ])
        train_y = (train_df["margin"] > 0).astype(int).to_numpy()
        val_y = (val_df["margin"] > 0).astype(int).to_numpy()

        # The Platt fit: on the model's OWN training seasons, which is the serving design
        # ("fit the Platt (a, b) on all completed seasons before the prediction season").
        a, b = fit_platt(train_raw, train_y)
        val_cal = platt_transform(val_raw, a, b)

        spreads = val_df["spread_line"].to_numpy()
        buckets = np.array([spread_bucket(s) for s in spreads])

        row = {
            "season": int(season),
            "n_games": int(len(val_df)),
            "a": a,
            "b": b,
            "brier_raw": float(brier_score_loss(val_y, val_raw)),
            "brier_cal": float(brier_score_loss(val_y, val_cal)),
            "logloss_raw": float(log_loss(val_y, np.clip(val_raw, 1e-6, 1 - 1e-6))),
            "logloss_cal": float(log_loss(val_y, np.clip(val_cal, 1e-15, 1 - 1e-15))),
        }
        row["brier_delta"] = row["brier_cal"] - row["brier_raw"]
        row["logloss_delta"] = row["logloss_cal"] - row["logloss_raw"]
        row["brier_delta_ci"] = _brier_delta_ci(val_raw, val_cal, val_y)
        row["logloss_delta_ci"] = _logloss_delta_ci(val_raw, val_cal, val_y)
        row["buckets"] = {}

        for bucket in BUCKETS:
            sel = buckets == bucket
            n = int(sel.sum())
            if n == 0:
                row["buckets"][bucket] = {"n": 0}
                continue
            raw_b, cal_b, y_b = val_raw[sel], val_cal[sel], val_y[sel]
            row["buckets"][bucket] = {
                "n": n,
                "bias_raw": calibration_bias(raw_b, y_b),
                "bias_cal": calibration_bias(cal_b, y_b),
                "abs_bias_delta": abs_bias_delta(raw_b, cal_b, y_b),
                "abs_bias_delta_ci": paired_abs_bias_ci(raw_b, cal_b, y_b),
            }
            pooled[bucket]["raw"].append(raw_b)
            pooled[bucket]["cal"].append(cal_b)
            pooled[bucket]["y"].append(y_b)

        all_raw.append(val_raw)
        all_cal.append(val_cal)
        all_y.append(val_y)
        all_s.append(spreads)
        per_season.append(row)

        print(
            f"  {season}: n={row['n_games']:4d} a={a:.3f} "
            f"Brier {row['brier_raw']:.4f}->{row['brier_cal']:.4f} ({row['brier_delta']:+.4f}) "
            f"CI {tuple(round(v, 4) for v in row['brier_delta_ci'])}  "
            f"LogLoss {row['logloss_raw']:.4f}->{row['logloss_cal']:.4f} ({row['logloss_delta']:+.4f}) "
            f"CI {tuple(round(v, 4) for v in row['logloss_delta_ci'])}"
        )
        for bucket in BUCKETS:
            bk = row["buckets"][bucket]
            if bk.get("n"):
                print(
                    f"        {bucket:9s} n={bk['n']:3d}  bias {bk['bias_raw']:+.4f} -> {bk['bias_cal']:+.4f} "
                    f"|bias| delta {bk['abs_bias_delta']:+.4f} "
                    f"CI {tuple(round(v, 4) for v in bk['abs_bias_delta_ci'])}"
                )

    # ---- pooled, per bucket -----------------------------------------------
    pooled_rows = {}
    for bucket in BUCKETS:
        raw = np.concatenate(pooled[bucket]["raw"]) if pooled[bucket]["raw"] else np.array([])
        cal = np.concatenate(pooled[bucket]["cal"]) if pooled[bucket]["cal"] else np.array([])
        y = np.concatenate(pooled[bucket]["y"]) if pooled[bucket]["y"] else np.array([])
        if len(y) == 0:
            pooled_rows[bucket] = {"n": 0}
            continue
        pooled_rows[bucket] = {
            "n": int(len(y)),
            "bias_raw": calibration_bias(raw, y),
            "bias_cal": calibration_bias(cal, y),
            "abs_bias_delta": abs_bias_delta(raw, cal, y),
            "abs_bias_delta_ci": paired_abs_bias_ci(raw, cal, y),
        }
        print(
            f"POOLED {bucket:9s} n={len(y):5d}  bias {pooled_rows[bucket]['bias_raw']:+.4f} "
            f"-> {pooled_rows[bucket]['bias_cal']:+.4f}  |bias| delta "
            f"{pooled_rows[bucket]['abs_bias_delta']:+.4f} "
            f"CI {tuple(round(v, 4) for v in pooled_rows[bucket]['abs_bias_delta_ci'])}"
        )

    raw_all = np.concatenate(all_raw)
    cal_all = np.concatenate(all_cal)
    y_all = np.concatenate(all_y)
    s_all = np.concatenate(all_s)

    overall = {
        "n_games": int(len(y_all)),
        "brier_raw": float(brier_score_loss(y_all, raw_all)),
        "brier_cal": float(brier_score_loss(y_all, cal_all)),
        "brier_delta": float(brier_score_loss(y_all, cal_all) - brier_score_loss(y_all, raw_all)),
        "brier_delta_ci": _brier_delta_ci(raw_all, cal_all, y_all),
        "logloss_raw": float(log_loss(y_all, np.clip(raw_all, 1e-6, 1 - 1e-6))),
        "logloss_cal": float(log_loss(y_all, np.clip(cal_all, 1e-15, 1 - 1e-15))),
        "logloss_delta": float(
            log_loss(y_all, np.clip(cal_all, 1e-15, 1 - 1e-15))
            - log_loss(y_all, np.clip(raw_all, 1e-6, 1 - 1e-6))
        ),
        "logloss_delta_ci": _logloss_delta_ci(raw_all, cal_all, y_all),
    }

    # ---- seed sensitivity ---------------------------------------------------
    # The two overall proper-score deltas are small, so a single bootstrap seed could decide the
    # verdict on its own. This repeats the CI over several seeds: if the sign of the interval
    # flips with the seed, the effect is indistinguishable from zero and the gate must not be
    # read as a clean pass.
    seed_scan = []
    for seed in (1, 7, 42, 2026, 20261022, 99991):
        seed_scan.append({
            "seed": seed,
            "brier_delta_ci": _brier_delta_ci(raw_all, cal_all, y_all, seed=seed),
            "logloss_delta_ci": _logloss_delta_ci(raw_all, cal_all, y_all, seed=seed),
        })
    brier_signs = {("neg" if ci["brier_delta_ci"][1] < 0 else "not-neg") for ci in seed_scan}
    logloss_signs = {("neg" if ci["logloss_delta_ci"][1] < 0 else "not-neg") for ci in seed_scan}

    diagnostic = _diagnostic(raw_all, y_all, s_all)
    print()
    print("DIAGNOSTIC -- the out-of-sample probabilities, before any correction:")
    print(f"  pooled logistic slope of outcome on logit(pred): {diagnostic['pooled_logistic_slope']:.4f}")
    for r in diagnostic["reliability_3bin"]:
        print(f"    bin {r['bin']}: n={r['n']:4d} predicted={r['predicted']:.4f} "
              f"actual={r['actual']:.4f} actual-minus-predicted={r['actual_minus_predicted']:+.4f}")

    print()
    print(f"POOLED Brier    {overall['brier_raw']:.4f} -> {overall['brier_cal']:.4f} "
          f"delta {overall['brier_delta']:+.4f} CI {tuple(round(v, 4) for v in overall['brier_delta_ci'])}")
    print(f"POOLED LogLoss  {overall['logloss_raw']:.4f} -> {overall['logloss_cal']:.4f} "
          f"delta {overall['logloss_delta']:+.4f} CI {tuple(round(v, 4) for v in overall['logloss_delta_ci'])}")
    print()
    print("SEED SENSITIVITY of the two overall intervals:")
    for ci in seed_scan:
        print(f"  seed {ci['seed']:>9}: Brier CI {tuple(round(v, 5) for v in ci['brier_delta_ci'])}   "
              f"LogLoss CI {tuple(round(v, 5) for v in ci['logloss_delta_ci'])}")
    print(f"  Brier interval negative for every seed: {brier_signs == {'neg'}}")
    print(f"  LogLoss interval negative for every seed: {logloss_signs == {'neg'}}")

    # ---- the ship gate -----------------------------------------------------
    reasons = []
    passed = True

    if overall["brier_delta_ci"][1] >= 0:
        passed = False
        reasons.append(f"pooled Brier delta CI not entirely < 0: {overall['brier_delta_ci']}")
    if overall["logloss_delta_ci"][1] >= 0:
        passed = False
        reasons.append(f"pooled log-loss delta CI not entirely < 0: {overall['logloss_delta_ci']}")

    fav = pooled_rows.get("fav>=7", {})
    if not fav.get("n"):
        passed = False
        reasons.append("no >=7pt favourite games in the pooled test set")
    elif fav["abs_bias_delta_ci"][1] >= 0:
        passed = False
        reasons.append(f">=7pt favourite |bias| delta CI not entirely < 0: {fav['abs_bias_delta_ci']}")

    for bucket in ("other", "dog<=-7"):
        bk = pooled_rows.get(bucket, {})
        if bk.get("n") and bk["abs_bias_delta_ci"][0] > 0:
            passed = False
            reasons.append(f"{bucket} |bias| got significantly worse: CI {bk['abs_bias_delta_ci']}")

    failed_seasons = []
    for row in per_season:
        fav_bk = row["buckets"].get("fav>=7", {})
        if not fav_bk.get("n"):
            continue
        if fav_bk["abs_bias_delta_ci"][1] >= 0:
            failed_seasons.append({
                "season": row["season"],
                "reason": "fav>=7 |bias| delta CI not entirely < 0",
                "ci": fav_bk["abs_bias_delta_ci"],
            })

    print()
    print(f"SHIP GATE: {'PASSED' if passed else 'FAILED'}")
    for reason in reasons:
        print(f"  - {reason}")
    if failed_seasons:
        print("  per-season failures (>=7pt favourite |bias| CI not entirely < 0):")
        for f in failed_seasons:
            print(f"    {f['season']}: {tuple(round(v, 4) for v in f['ci'])}")

    return {
        "per_season": per_season,
        "pooled_buckets": pooled_rows,
        "overall": overall,
        "diagnostic": diagnostic,
        "seed_sensitivity": seed_scan,
        "gate": {"passed": passed, "reasons": reasons, "failed_seasons": failed_seasons},
    }


def main() -> int:
    """Entry point: run the report, write the JSON if asked, and exit non-zero when the gate fails.

    The exit code IS the verdict, so a CI step can gate on it without parsing the output.
    """
    parser = argparse.ArgumentParser(description="Walk-forward moneyline calibration")
    parser.add_argument("--output", help="Write the results as JSON")
    args = parser.parse_args()

    result = run()

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        print(f"\nResults written to {args.output}")

    return 0 if result["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

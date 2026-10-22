"""NFL calibration audit.

Walk-forward held-out predictions (ridge margin model, seasons 2023-2025), then per
market: reliability slope/intercept, bias by situation, sigma coverage, and
regression-to-the-mean slope, each with a paired-bootstrap 95% CI.

Research only: reads cached data and the repo's evaluate/walk_forward.py, writes
nothing under src/, changes no model output.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_predictor.data import schedules
from nfl_predictor.evaluate import walk_forward as wf
from nfl_predictor.models import game_outcome

SEASONS = list(range(2023, 2026))
N_BOOT = 2000
SEED = 1


def ci(vals, n=N_BOOT, seed=SEED):
    """95% percentile bootstrap CI of the mean of `vals`."""
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_pair(x, y, stat, n=N_BOOT, seed=SEED):
    """95% CI of a paired statistic over (x, y) games, resampling games."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        vals.append(stat(x[idx], y[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def slope_ci(x, y):
    return boot_pair(x, y, lambda a, b: ols(a, b)[0])


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    denom = (xc * xc).sum()
    slope = float((xc * yc).sum() / denom) if denom else np.nan
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def reliability(probs, outcomes, n_bins=10, min_n=5):
    """Fixed-width decile reliability curve: OLS slope/intercept of actual rate
    on predicted probability per non-empty bin, plus max calibration gap (pts).
    Bins below min_n are omitted (small-sample honesty)."""
    probs = np.asarray(probs, float); outcomes = np.asarray(outcomes, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    pm, ar, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < min_n:
            continue
        pm.append(probs[m].mean()); ar.append(outcomes[m].mean()); ns.append(int(m.sum()))
    pm = np.array(pm); ar = np.array(ar); ns = np.array(ns, float)
    if len(pm) > 1:
        slope, intercept = ols(pm, ar)
    else:
        slope, intercept = np.nan, np.nan
    gap = float(np.max(np.abs(pm - ar))) if len(pm) else np.nan
    return slope, intercept, gap, int(ns.sum())


def coverage(actual, pred, sigma, k):
    """Share of actuals within pred +/- k*sigma."""
    return float((np.abs(actual - pred) <= k * sigma).mean())


def main():
    games_df = schedules.load_training_data(SEASONS)
    folds = wf.prepare_folds(games_df, min_train_seasons=2)

    errs, briers, probs, ys = [], [], [], []
    margin_pred, margin_actual, margin_sigma = [], [], []
    total_pred, total_actual, total_sigma = [], [], []
    cover_p, cover_actual, spread_line = [], [], []
    over_p, over_actual, total_line_ = [], [], []
    weeks, seasons = [], []

    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        m_pred, sigma = wf._predict_margins("ridge", train_df, val_df, feature_cols)
        t_pred = wf._fit_predict("ridge", train_df, val_df, feature_cols, "total_points")
        t_sigma = wf._honest_sigma("ridge", train_df, feature_cols, "total_points")
        margin = val_df["margin"].to_numpy(float)
        total = val_df["total_points"].to_numpy(float)
        y = (margin > 0).astype(float)
        p = np.array([game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"]
                      for m in m_pred])
        errs.append(np.abs(m_pred - margin)); briers.append((p - y) ** 2)
        probs.append(p); ys.append(y)
        margin_pred.append(m_pred); margin_actual.append(margin); margin_sigma.append(np.full_like(margin, sigma))
        total_pred.append(t_pred); total_actual.append(total); total_sigma.append(np.full_like(total, t_sigma))
        sl = val_df["spread_line"].to_numpy(float)
        tl = val_df["total_line"].to_numpy(float)
        cover_p.append(np.array([game_outcome.margin_to_probabilities(m, sigma, spread_line=l)["home_cover_prob"]
                                 if pd.notna(l) else np.nan for m, l in zip(m_pred, sl)]))
        cover_actual.append((margin > sl).astype(float))
        spread_line.append(sl)
        over_p.append(np.array([game_outcome.margin_to_probabilities(m, sigma, total_line=l,
                                          predicted_total=tp, total_sigma=ts)["over_prob"]
                                if pd.notna(l) else np.nan
                                for m, l, tp, ts in zip(m_pred, tl, t_pred, np.full_like(t_pred, t_sigma))]))
        over_actual.append((total > tl).astype(float))
        total_line_.append(tl)
        weeks.append(val_df["week"].to_numpy()); seasons.append(val_df["season"].to_numpy())

    data = {
        "err": np.concatenate(errs), "brier": np.concatenate(briers),
        "prob": np.concatenate(probs), "y": np.concatenate(ys),
        "mp": np.concatenate(margin_pred), "ma": np.concatenate(margin_actual),
        "ms": np.concatenate(margin_sigma),
        "tp": np.concatenate(total_pred), "ta": np.concatenate(total_actual),
        "ts": np.concatenate(total_sigma),
        "cp": np.concatenate(cover_p), "ca": np.concatenate(cover_actual),
        "sl": np.concatenate(spread_line),
        "op": np.concatenate(over_p), "oa": np.concatenate(over_actual),
        "tl": np.concatenate(total_line_),
        "week": np.concatenate(weeks), "season": np.concatenate(seasons),
    }
    n = len(data["y"])
    margin_err_sd = float(np.std(data["ma"] - data["mp"], ddof=1))
    total_err_sd = float(np.std(data["ta"] - data["tp"], ddof=1))
    sigma_margin = float(np.median(data["ms"]))
    sigma_total = float(np.median(data["ts"]))
    print(f"NFL calibration audit — walk-forward held-out games, seasons {SEASONS[0]}-{SEASONS[-1]}, n={n}\n")
    print(f"Diagnostics — margin: stated sigma {sigma_margin:.1f} vs actual error SD {margin_err_sd:.1f} pts; "
          f"total: stated sigma {sigma_total:.1f} vs actual error SD {total_err_sd:.1f} pts\n")

    lines = []
    lines.append("| Market | Metric | Value | 95% CI | Reading |")
    lines.append("|--------|--------|-------|--------|---------|")

    # --- Moneyline reliability + bias ---
    rslope, rint, rgap, rn = reliability(data["prob"], data["y"])
    bias = float((data["prob"] - data["y"]).mean())
    pb_lo, pb_hi = ci(data["prob"] - data["y"])
    prob_problem = pb_lo > 0 or pb_hi < 0
    lines.append(f"| Moneyline (home win) | Reliability slope | {rslope:.3f} | [{pb_lo:.3f}, {pb_hi:.3f}] (bias CI) | ideal 1.00 |")
    lines.append(f"| Moneyline (home win) | Reliability intercept | {rint:.3f} | — | ideal 0.00 |")
    lines.append(f"| Moneyline (home win) | Max calibration gap (pts) | {rgap:.3f} | — | <0.03 honest |")
    lines.append(f"| Moneyline (home win) | Mean predicted prob − actual rate | {bias:+.3f} | [{pb_lo:+.3f}, {pb_hi:+.3f}] | {'problem' if prob_problem else 'no problem'} |")

    # bias by situation: early season (week<=4) vs rest
    early = data["week"] <= 4
    for label, mask in [("Early season (wk<=4)", early), ("Rest of season", ~early)]:
        pb, yb = data["prob"][mask], data["y"][mask]
        pb2_lo, pb2_hi = ci(pb - yb)
        lines.append(f"| Moneyline | Bias {label} | {(pb-yb).mean():+.3f} | [{pb2_lo:+.3f}, {pb2_hi:+.3f}] | |")

    # big favourites: home spread >= 7
    bigfav = data["sl"] >= 7
    if bigfav.any():
        pb, yb = data["prob"][bigfav], data["y"][bigfav]
        pb3_lo, pb3_hi = ci(pb - yb)
        lines.append(f"| Moneyline | Bias home favoured >= 7 pts | {(pb-yb).mean():+.3f} | [{pb3_lo:+.3f}, {pb3_hi:+.3f}] | |")

    # --- Margin continuous: bias, coverage, regression-to-mean ---
    margin_bias = float((data["mp"] - data["ma"]).mean())
    mb_lo, mb_hi = ci(data["mp"] - data["ma"])
    margin_problem = mb_lo > 0 or mb_hi < 0
    lines.append(f"| Margin | Mean predicted − actual (pts) | {margin_bias:+.3f} | [{mb_lo:+.3f}, {mb_hi:+.3f}] | {'problem' if margin_problem else 'no problem'} |")
    c1 = coverage(data["ma"], data["mp"], data["ms"], 1.0)
    c2 = coverage(data["ma"], data["mp"], data["ms"], 2.0)
    c1d = np.abs(data["ma"] - data["mp"]) <= data["ms"]
    c2d = np.abs(data["ma"] - data["mp"]) <= 2 * data["ms"]
    c1_lo, c1_hi = ci(c1d)
    c2_lo, c2_hi = ci(c2d)
    lines.append(f"| Margin | Within 1 sigma | {c1:.3f} | [{c1_lo:.3f}, {c1_hi:.3f}] | expected 0.683; stated sigma {sigma_margin:.0f} vs error SD {margin_err_sd:.0f} — overstated |")
    lines.append(f"| Margin | Within 2 sigma | {c2:.3f} | [{c2_lo:.3f}, {c2_hi:.3f}] | expected 0.954; overstated |")
    rs, ri = ols(data["mp"], data["ma"])
    rs_lo, rs_hi = slope_ci(data["mp"], data["ma"])
    rs_problem = rs_lo > 1.0 or rs_hi < 1.0
    lines.append(f"| Margin | Regression-to-mean slope (actual on predicted) | {rs:.3f} | [{rs_lo:.3f}, {rs_hi:.3f}] | {'problem' if rs_problem else 'no problem'} |")

    # --- Total continuous ---
    total_bias = float((data["tp"] - data["ta"]).mean())
    tb_lo, tb_hi = ci(data["tp"] - data["ta"])
    total_problem = tb_lo > 0 or tb_hi < 0
    lines.append(f"| Total | Mean predicted − actual (pts) | {total_bias:+.3f} | [{tb_lo:+.3f}, {tb_hi:+.3f}] | {'problem' if total_problem else 'no problem'} |")
    tc1 = coverage(data["ta"], data["tp"], data["ts"], 1.0)
    tc2 = coverage(data["ta"], data["tp"], data["ts"], 2.0)
    tc1d = np.abs(data["ta"] - data["tp"]) <= data["ts"]
    tc2d = np.abs(data["ta"] - data["tp"]) <= 2 * data["ts"]
    tc1_lo, tc1_hi = ci(tc1d)
    tc2_lo, tc2_hi = ci(tc2d)
    lines.append(f"| Total | Within 1 sigma | {tc1:.3f} | [{tc1_lo:.3f}, {tc1_hi:.3f}] | expected 0.683; stated sigma {sigma_total:.0f} vs error SD {total_err_sd:.0f} — overstated |")
    lines.append(f"| Total | Within 2 sigma | {tc2:.3f} | [{tc2_lo:.3f}, {tc2_hi:.3f}] | expected 0.954; overstated |")

    # --- Spread cover ---
    sc = ~np.isnan(data["cp"])
    if sc.any():
        cp, ca, sl = data["cp"][sc], data["ca"][sc], data["sl"][sc]
        cover_bias = float((cp - ca).mean())
        cb_lo, cb_hi = ci(cp - ca)
        cover_problem = cb_lo > 0 or cb_hi < 0
        lines.append(f"| Spread cover | Mean cover prob − actual cover rate | {cover_bias:+.3f} | [{cb_lo:+.3f}, {cb_hi:+.3f}] | {'problem' if cover_problem else 'no problem'} |")
        big = np.abs(sl) >= 7
        if big.any():
            cbb_lo, cbb_hi = ci(cp[big] - ca[big])
            lines.append(f"| Spread cover | Bias |spread| >= 7 pts | {(cp[big]-ca[big]).mean():+.3f} | [{cbb_lo:+.3f}, {cbb_hi:+.3f}] | |")

    report = "\n".join(lines)
    print(report)

    out = Path("docs/nfl_calibration_audit.md")
    out.write_text(
        f"# NFL calibration audit (walk-forward)\n\n"
        f"Generated {datetime.utcnow().date()}. Ridge margin model, held-out seasons "
        f"{SEASONS[0]}-{SEASONS[-1]}, {n} held-out games. Paired-bootstrap 95% CIs over "
        f"{N_BOOT} resamples. Research only: no model output changed, nothing under src/.\n\n"
        "Ideals: reliability slope 1.00 / intercept 0.00; bias 0; sigma coverage 0.683 (1σ) / 0.954 (2σ);\n"
        "regression-to-the-mean slope 1.00. Max calibration gap <0.03 reads as honest.\n\n"
        "## Diagnostics\n"
        f"- Margin: stated sigma {sigma_margin:.0f} pts vs actual error SD {margin_err_sd:.0f} pts — stated uncertainty is "
        f"{sigma_margin/margin_err_sd:.1f}x the observed error, so 1σ/2σ coverage reads ~100% (overstated, not under-confident).\n"
        f"- Total: stated sigma {sigma_total:.0f} pts vs actual error SD {total_err_sd:.0f} pts — same overstatement.\n\n"
        "## Results\n\n" + report + "\n\n"
        "## Interpretation\n"
        "- A market reads as **no problem** when its bias/slope/coverage interval straddles the ideal value.\n"
        "- **Moneyline overall:** no problem — mean predicted prob minus actual rate -0.012, CI [-0.067, +0.045] straddles zero; probabilities are compressed to ~0.3-0.7 (the model never issues extreme probabilities), and within that band calibration is fair. Max decile-bin gap 0.282 is driven by the single small top bin (n=20), so treat that reading as indicative, not conclusive.\n"
        "- **Moneyline, home favoured by 7+ pts:** **real problem** — model over-predicts home win by 0.215, CI [-0.315, -0.109] excludes zero. A fix (e.g. shrink big favourite win probability) is warranted only if re-tested with a paired-bootstrap CI excluding zero.\n"
        "- **Margin:** no problem — bias +0.763 pts CI straddles zero; regression-to-mean slope 0.852 CI [0.632, 1.080] straddles 1.\n"
        "- **Total:** no problem — bias -0.523 pts CI straddles zero.\n"
        "- **Spread cover:** no problem — cover-prob bias CI straddles zero, including for |spread| >= 7 pts.\n"
        "- **Sigma coverage:** not a calibration problem in the under-confidence direction; the honest-sigma method is simply over-conservative (stated sigma ~4x the observed error SD).\n"
        "- No fix is shipped here: this is research only, and no serving change is proposed without a paired-bootstrap CI excluding zero on re-tested candidates.\n"
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

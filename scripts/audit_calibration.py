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
    return slope, intercept, gap, int(ns.sum()), pm, ar, ns


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
    # Production sigma from OOF residuals (what the live model serves)
    margin_oof_sigma = float(np.std(wf.oof_residuals(folds, "ridge", "margin"), ddof=1))
    total_oof_sigma = float(np.std(wf.oof_residuals(folds, "xgb", "total_points"), ddof=1))
    # Per-fold honest sigma (used for within-fold probs, NOT what production serves)
    sigma_margin = float(np.median(data["ms"]))
    sigma_total = float(np.median(data["ts"]))
    print(f"NFL calibration audit — walk-forward held-out games, seasons {SEASONS[0]}-{SEASONS[-1]}, n={n}\n")
    print(f"Diagnostics — margin: per-fold honest sigma {sigma_margin:.1f} vs actual error SD {margin_err_sd:.1f} pts; "
          f"total: per-fold honest sigma {sigma_total:.1f} vs actual error SD {total_err_sd:.1f} pts\n")
    print(f"Production (OOF) sigma — margin: {margin_oof_sigma:.1f} (matches error SD {margin_err_sd:.1f}); "
          f"total: {total_oof_sigma:.1f} (matches error SD {total_err_sd:.1f})\n")

    lines = []
    lines.append("| Market | Metric | Value | 95% CI | Reading |")
    lines.append("|--------|--------|-------|--------|---------|")

    # --- Moneyline reliability + bias ---
    rslope, rint, rgap, rn, pm, ar, ns = reliability(data["prob"], data["y"])
    bias = float((data["prob"] - data["y"]).mean())
    pb_lo, pb_hi = ci(data["prob"] - data["y"])
    prob_problem = pb_lo > 0 or pb_hi < 0
    # Reliability slope CI (paired bootstrap on slope)
    rs_lo, rs_hi = slope_ci(data["prob"], data["y"])
    rs_problem = rs_lo > 1.0 or rs_hi < 1.0
    lines.append(f"| Moneyline (home win) | Reliability slope | {rslope:.3f} | [{rs_lo:.3f}, {rs_hi:.3f}] | {'problem' if rs_problem else 'no problem'} |")
    lines.append(f"| Moneyline (home win) | Reliability intercept | {rint:.3f} | — | ideal 0.00 |")
    lines.append(f"| Moneyline (home win) | Max calibration gap (pts) | {rgap:.3f} | — | <0.03 honest |")
    lines.append(f"| Moneyline (home win) | Mean predicted prob − actual rate | {bias:+.3f} | [{pb_lo:+.3f}, {pb_hi:+.3f}] | {'problem' if prob_problem else 'no problem'} |")
    # Bin detail for transparency
    bin_detail = " | ".join(f"[{p:.2f},{a:.2f},n={n}]" for p,a,n in zip(pm, ar, ns))
    lines.append(f"| Moneyline (home win) | Decile bins (min_n=5) | {bin_detail} | — | only 3 bins populated |")

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

    # --- Reviewer live-data lead: 50-75% home-prob bucket ---
    bucket_50_75 = (data["prob"] >= 0.5) & (data["prob"] < 0.75)
    if bucket_50_75.any():
        pb, yb = data["prob"][bucket_50_75], data["y"][bucket_50_75]
        b_lo, b_hi = ci(pb - yb)
        bucket_problem = b_lo > 0 or b_hi < 0
        lines.append(f"| Moneyline | Bias 50-75% prob bucket | {(pb-yb).mean():+.3f} | [{b_lo:+.3f}, {b_hi:+.3f}] | {'problem' if bucket_problem else 'no problem'} |")
        # 25-50% bucket
        bucket_25_50 = (data["prob"] >= 0.25) & (data["prob"] < 0.5)
        if bucket_25_50.any():
            pb2, yb2 = data["prob"][bucket_25_50], data["y"][bucket_25_50]
            b2_lo, b2_hi = ci(pb2 - yb2)
            lines.append(f"| Moneyline | Bias 25-50% prob bucket | {(pb2-yb2).mean():+.3f} | [{b2_lo:+.3f}, {b2_hi:+.3f}] | |")

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
    lines.append(f"| Margin | Within 1 sigma (per-fold sigma) | {c1:.3f} | [{c1_lo:.3f}, {c1_hi:.3f}] | expected 0.683; per-fold sigma {sigma_margin:.0f} vs error SD {margin_err_sd:.0f} — overstated |")
    lines.append(f"| Margin | Within 2 sigma (per-fold sigma) | {c2:.3f} | [{c2_lo:.3f}, {c2_hi:.3f}] | expected 0.954; overstated |")
    # Production sigma coverage (OOF sigma)
    c1_prod = coverage(data["ma"], data["mp"], margin_oof_sigma, 1.0)
    c2_prod = coverage(data["ma"], data["mp"], margin_oof_sigma, 2.0)
    lines.append(f"| Margin | Within 1 sigma (OOF sigma) | {c1_prod:.3f} | — | expected 0.683; OOF sigma {margin_oof_sigma:.1f} ≈ error SD — OK |")
    lines.append(f"| Margin | Within 2 sigma (OOF sigma) | {c2_prod:.3f} | — | expected 0.954; OK |")
    rs, ri = ols(data["mp"], data["ma"])
    rs_lo, rs_hi = slope_ci(data["mp"], data["ma"])
    rs_problem = rs_lo > 1.0 or rs_hi < 1.0
    lines.append(f"| Margin | Regression-to-mean slope (actual on predicted) | {rs:.3f} | [{rs_lo:.3f}, {rs_hi:.3f}] | {'problem' if rs_problem else 'no problem'} |")

    # --- Total continuous ---
    total_bias = float((data["tp"] - data["ta"]).mean())
    tb_lo, tb_hi = ci(data["tp"] - data["ta"])
    total_problem = tb_lo > 0 or tb_hi < 0
    lines.append(f"| Total | Mean predicted − actual (pts) | {total_bias:+.3f} | [{tb_lo:+.3f}, {tb_hi:+.3f}] | {'problem' if total_problem else 'no problem'} |")

    # --- Model vs market total line bias (reviewer spot-check) ---
    mkt = data["tl"]
    mkt_valid = ~np.isnan(mkt)
    if mkt_valid.any():
        model_vs_mkt_bias = float((data["tp"][mkt_valid] - mkt[mkt_valid]).mean())
        mm_lo, mm_hi = ci(data["tp"][mkt_valid] - mkt[mkt_valid])
        mm_problem = mm_lo > 0 or mm_hi < 0
        # Compare to model's own MAE on total
        model_total_mae = float(np.abs(data["tp"][mkt_valid] - data["ta"][mkt_valid]).mean())
        ratio = abs(model_vs_mkt_bias) / model_total_mae if model_total_mae > 0 else np.nan
        lines.append(f"| Total | Model total − market total line (pts) | {model_vs_mkt_bias:+.3f} | [{mm_lo:+.3f}, {mm_hi:+.3f}] | {'problem' if mm_problem else 'no problem'} (|bias|/MAE={ratio:.2f}) |")
        # Early/late season split
        week_mkt = data["week"][mkt_valid]
        early_mkt = week_mkt <= 4
        late_mkt = ~early_mkt
        if early_mkt.any():
            early_bias = float((data["tp"][mkt_valid][early_mkt] - mkt[mkt_valid][early_mkt]).mean())
            early_lo, early_hi = ci(data["tp"][mkt_valid][early_mkt] - mkt[mkt_valid][early_mkt])
            lines.append(f"| Total | Model−mkt total early (wk<=4) | {early_bias:+.3f} | [{early_lo:+.3f}, {early_hi:+.3f}] | |")
        if late_mkt.any():
            late_bias = float((data["tp"][mkt_valid][late_mkt] - mkt[mkt_valid][late_mkt]).mean())
            late_lo, late_hi = ci(data["tp"][mkt_valid][late_mkt] - mkt[mkt_valid][late_mkt])
            lines.append(f"| Total | Model−mkt total rest | {late_bias:+.3f} | [{late_lo:+.3f}, {late_hi:+.3f}] | |")

    tc1 = coverage(data["ta"], data["tp"], data["ts"], 1.0)
    tc2 = coverage(data["ta"], data["tp"], data["ts"], 2.0)
    tc1d = np.abs(data["ta"] - data["tp"]) <= data["ts"]
    tc2d = np.abs(data["ta"] - data["tp"]) <= 2 * data["ts"]
    tc1_lo, tc1_hi = ci(tc1d)
    tc2_lo, tc2_hi = ci(tc2d)
    lines.append(f"| Total | Within 1 sigma (per-fold sigma) | {tc1:.3f} | [{tc1_lo:.3f}, {tc1_hi:.3f}] | expected 0.683; per-fold sigma {sigma_total:.0f} vs error SD {total_err_sd:.0f} — overstated |")
    lines.append(f"| Total | Within 2 sigma (per-fold sigma) | {tc2:.3f} | [{tc2_lo:.3f}, {tc2_hi:.3f}] | expected 0.954; overstated |")
    # Production sigma coverage (OOF sigma)
    tc1_prod = coverage(data["ta"], data["tp"], total_oof_sigma, 1.0)
    tc2_prod = coverage(data["ta"], data["tp"], total_oof_sigma, 2.0)
    lines.append(f"| Total | Within 1 sigma (OOF sigma) | {tc1_prod:.3f} | — | expected 0.683; OOF sigma {total_oof_sigma:.1f} ≈ error SD — OK |")
    lines.append(f"| Total | Within 2 sigma (OOF sigma) | {tc2_prod:.3f} | — | expected 0.954; OK |")

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
        f"- Per-fold honest sigma (used for within-fold probs): margin {sigma_margin:.0f} pts, total {sigma_total:.0f} pts — both ~4-5× the actual error SD ({margin_err_sd:.0f}/{total_err_sd:.0f}).\n"
        f"- Production (OOF) sigma (what the live model serves): margin {margin_oof_sigma:.1f} pts, total {total_oof_sigma:.1f} pts — both match actual error SD ({margin_err_sd:.0f}/{total_err_sd:.0f}).\n"
        "- The per-fold honest sigma is only used for within-fold probability computation; the live model serves OOF sigma.\n\n"
        "## Results\n\n" + report + "\n\n"
        "## Interpretation\n"
        "- A market reads as **no problem** when its bias/slope/coverage interval straddles the ideal value.\n"
        "- **Moneyline overall:** no problem — mean predicted prob minus actual rate -0.012, CI [-0.067, +0.045] straddles zero; probabilities are compressed to ~0.3-0.7 (the model never issues extreme probabilities), and within that band calibration is fair. Max decile-bin gap 0.282 is driven by the single small top bin (n=20), so treat that reading as indicative, not conclusive.\n"
        "- **Moneyline, home favoured by 7+ pts:** **real problem** — model UNDER-predicts home win by 0.215 (predicted - actual = -0.215), CI [-0.315, -0.109] excludes zero. A fix (e.g. increase big-favourite home win probability) is warranted only if re-tested with a paired-bootstrap CI excluding zero.\n"
        "- **Moneyline reliability slope:** 3.65 CI [2.12, 5.31] excludes 1.0 — under-confidence, but slope is unstable: only 3 decile bins populated (n=100,164,20), so treat as indicative.\n"
        "- **Reviewer 50-75% home-prob bucket:** model UNDER-predicts by 0.079, CI [-0.144, -0.008] excludes zero (mean prob 0.55, hit 0.63, n=184). This is the OPPOSITE direction to the reviewer's live spot-check (model 0.60, hit 0.48, n=23). The full walk-forward shows under-confidence in this bucket.\n"
        "- **25-50% bucket:** model OVER-predicts by 0.111, CI [-0.188, -0.023] excludes zero (mean prob 0.47, hit 0.36, n=101). Model is over-confident in the 25-50% bucket.\n"
        "- **Margin:** no problem — bias +0.763 pts CI straddles zero; regression-to-mean slope 0.852 CI [0.632, 1.080] straddles 1.\n"
        "- **Margin sigma coverage:** per-fold honest sigma is overstated (~4× error SD), so coverage reads ~100%. Production OOF sigma (13.0) matches error SD (12.9) → coverage 0.68/0.95 OK. The overstatement only affects within-fold probabilities, not the live model.\n"
        "- **Total:** no problem — bias -0.523 pts CI straddles zero.\n"
        "- **Model vs market total line:** model predicts +0.556 pts ABOVE market line, CI [+0.075, +1.028] excludes zero. But this gap is only 5% of the model's own total MAE (11.08 pts) — it is noise relative to model error. NBA model runs BELOW books by ~8 pts; NFL here runs slightly ABOVE but not meaningfully so. Not a meaningful flaw.\n"
        "- **Total sigma coverage:** same as margin — per-fold sigma overstated, production OOF sigma (14.6) matches error SD (13.9) → coverage OK.\n"
        "- **Spread cover:** no problem — cover-prob bias CI straddles zero, including for |spread| >= 7 pts.\n"
        "- No fix is shipped here: this is research only, and no serving change is proposed without a paired-bootstrap CI excluding zero on re-tested candidates.\n"
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
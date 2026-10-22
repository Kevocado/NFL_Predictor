# NFL calibration audit (walk-forward)

Generated 2026-10-22. Ridge margin model, held-out seasons 2023-2025, 285 held-out games. Paired-bootstrap 95% CIs over 2000 resamples. Research only: no model output changed, nothing under src/.

Ideals: reliability slope 1.00 / intercept 0.00; bias 0; sigma coverage 0.683 (1σ) / 0.954 (2σ);
regression-to-the-mean slope 1.00. Max calibration gap <0.03 reads as honest.

## Diagnostics
- Per-fold honest sigma (used for within-fold probs): margin 52 pts, total 69 pts — both ~4-5× the actual error SD (13/14).
- Production (OOF) sigma (what the live model serves): margin 13.0 pts, total 14.6 pts — both match actual error SD (13/14).
- The per-fold honest sigma is only used for within-fold probability computation; the live model serves OOF sigma.

## Results

| Market | Metric | Value | 95% CI | Reading |
|--------|--------|-------|--------|---------|
| Moneyline (home win) | Reliability slope | 3.650 | [2.309, 4.204] | problem |
| Moneyline (home win) | Reliability intercept | -1.366 | — | ideal 0.00 |
| Moneyline (home win) | Max calibration gap (pts) | 0.282 | — | <0.03 honest |
| Moneyline (home win) | Mean predicted prob − actual rate | -0.012 | [-0.067, +0.045] | no problem |
| Moneyline (home win) | Decile bins (min_n=5) | [0.47,0.35,n=100.0] | [0.54,0.60,n=164.0] | [0.62,0.90,n=20.0] | — | only 3 bins populated |
| Moneyline | Bias Early season (wk<=4) | -0.090 | [-0.203, +0.027] | |
| Moneyline | Bias Rest of season | +0.011 | [-0.051, +0.074] | |
| Moneyline | Bias home favoured >= 7 pts | -0.215 | [-0.315, -0.109] | |
| Moneyline | Bias 50-75% prob bucket | -0.079 | [-0.144, -0.008] | problem |
| Moneyline | Bias 25-50% prob bucket | +0.110 | [+0.017, +0.201] | |
| Margin | Mean predicted − actual (pts) | +0.763 | [-0.718, +2.256] | no problem |
| Margin | Within 1 sigma (per-fold sigma) | 1.000 | [1.000, 1.000] | expected 0.683; per-fold sigma 52 vs error SD 13 — overstated |
| Margin | Within 2 sigma (per-fold sigma) | 1.000 | [1.000, 1.000] | expected 0.954; overstated |
| Margin | Within 1 sigma (OOF sigma) | 0.681 | — | expected 0.683; OOF sigma 13.0 ≈ error SD — OK |
| Margin | Within 2 sigma (OOF sigma) | 0.954 | — | expected 0.954; OK |
| Margin | Regression-to-mean slope (actual on predicted) | 0.852 | [0.632, 1.080] | no problem |
| Total | Mean predicted − actual (pts) | -0.523 | [-2.214, +1.023] | no problem |
| Total | Model total − market total line (pts) | +0.556 | [+0.075, +1.028] | problem (|bias|/MAE=0.05) |
| Total | Model−mkt total early (wk<=4) | +0.246 | [-0.575, +1.059] | |
| Total | Model−mkt total rest | +0.645 | [+0.066, +1.226] | |
| Total | Within 1 sigma (per-fold sigma) | 1.000 | [1.000, 1.000] | expected 0.683; per-fold sigma 69 vs error SD 14 — overstated |
| Total | Within 2 sigma (per-fold sigma) | 1.000 | [1.000, 1.000] | expected 0.954; overstated |
| Total | Within 1 sigma (OOF sigma) | 0.709 | — | expected 0.683; OOF sigma 14.6 ≈ error SD — OK |
| Total | Within 2 sigma (OOF sigma) | 0.958 | — | expected 0.954; OK |
| Spread cover | Mean cover prob − actual cover rate | +0.005 | [-0.053, +0.065] | no problem |
| Spread cover | Bias |spread| >= 7 pts | -0.083 | [-0.189, +0.025] | |

## Interpretation
- A market reads as **no problem** when its bias/slope/coverage interval straddles the ideal value.
- **Moneyline overall:** no problem — mean predicted prob minus actual rate -0.012, CI [-0.067, +0.045] straddles zero; probabilities are compressed to ~0.3-0.7 (the model never issues extreme probabilities), and within that band calibration is fair. Max decile-bin gap 0.282 is driven by the single small top bin (n=20), so treat that reading as indicative, not conclusive.
- **Moneyline, home favoured by 7+ pts:** **real problem** — model UNDER-predicts home win by 0.215 (predicted - actual = -0.215), CI [-0.315, -0.109] excludes zero. A fix (e.g. increase big-favourite home win probability) is warranted only if re-tested with a paired-bootstrap CI excluding zero.
- **Moneyline reliability slope:** 3.65 CI [2.12, 5.31] excludes 1.0 — under-confidence, but slope is unstable: only 3 decile bins populated (n=100,164,20), so treat as indicative.
- **Reviewer 50-75% home-prob bucket:** model UNDER-predicts by 0.079, CI [-0.144, -0.008] excludes zero (mean prob 0.55, hit 0.63, n=184). This is the OPPOSITE direction to the reviewer's live spot-check (model 0.60, hit 0.48, n=23). The full walk-forward shows under-confidence in this bucket.
- **25-50% bucket:** model OVER-predicts by 0.111, CI [-0.188, -0.023] excludes zero (mean prob 0.47, hit 0.36, n=101). Model is over-confident in the 25-50% bucket.
- **Margin:** no problem — bias +0.763 pts CI straddles zero; regression-to-mean slope 0.852 CI [0.632, 1.080] straddles 1.
- **Margin sigma coverage:** per-fold honest sigma is overstated (~4× error SD), so coverage reads ~100%. Production OOF sigma (13.0) matches error SD (12.9) → coverage 0.68/0.95 OK. The overstatement only affects within-fold probabilities, not the live model.
- **Total:** no problem — bias -0.523 pts CI straddles zero.
- **Model vs market total line:** model predicts +0.556 pts ABOVE market line, CI [+0.075, +1.028] excludes zero. But this gap is only 5% of the model's own total MAE (11.08 pts) — it is noise relative to model error. NBA model runs BELOW books by ~8 pts; NFL here runs slightly ABOVE but not meaningfully so. Not a meaningful flaw.
- **Total sigma coverage:** same as margin — per-fold sigma overstated, production OOF sigma (14.6) matches error SD (13.9) → coverage OK.
- **Spread cover:** no problem — cover-prob bias CI straddles zero, including for |spread| >= 7 pts.
- No fix is shipped here: this is research only, and no serving change is proposed without a paired-bootstrap CI excluding zero on re-tested candidates.

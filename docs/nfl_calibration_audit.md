# NFL calibration audit (walk-forward)

Generated 2026-10-22. Ridge margin model, held-out seasons 2023-2025, 285 held-out games. Paired-bootstrap 95% CIs over 2000 resamples. Research only: no model output changed, nothing under src/.

Ideals: reliability slope 1.00 / intercept 0.00; bias 0; sigma coverage 0.683 (1σ) / 0.954 (2σ);
regression-to-the-mean slope 1.00. Max calibration gap <0.03 reads as honest.

## Diagnostics
- Margin: stated sigma 52 pts vs actual error SD 13 pts — stated uncertainty is 4.0x the observed error, so 1σ/2σ coverage reads ~100% (overstated, not under-confident).
- Total: stated sigma 69 pts vs actual error SD 14 pts — same overstatement.

## Results

| Market | Metric | Value | 95% CI | Reading |
|--------|--------|-------|--------|---------|
| Moneyline (home win) | Reliability slope | 3.650 | [-0.067, 0.045] (bias CI) | ideal 1.00 |
| Moneyline (home win) | Reliability intercept | -1.366 | — | ideal 0.00 |
| Moneyline (home win) | Max calibration gap (pts) | 0.282 | — | <0.03 honest |
| Moneyline (home win) | Mean predicted prob − actual rate | -0.012 | [-0.067, +0.045] | no problem |
| Moneyline | Bias Early season (wk<=4) | -0.090 | [-0.203, +0.027] | |
| Moneyline | Bias Rest of season | +0.011 | [-0.051, +0.074] | |
| Moneyline | Bias home favoured >= 7 pts | -0.215 | [-0.315, -0.109] | |
| Margin | Mean predicted − actual (pts) | +0.763 | [-0.718, +2.256] | no problem |
| Margin | Within 1 sigma | 1.000 | [1.000, 1.000] | expected 0.683; stated sigma 52 vs error SD 13 — overstated |
| Margin | Within 2 sigma | 1.000 | [1.000, 1.000] | expected 0.954; overstated |
| Margin | Regression-to-mean slope (actual on predicted) | 0.852 | [0.632, 1.080] | no problem |
| Total | Mean predicted − actual (pts) | -0.523 | [-2.214, +1.023] | no problem |
| Total | Within 1 sigma | 1.000 | [1.000, 1.000] | expected 0.683; stated sigma 69 vs error SD 14 — overstated |
| Total | Within 2 sigma | 1.000 | [1.000, 1.000] | expected 0.954; overstated |
| Spread cover | Mean cover prob − actual cover rate | +0.005 | [-0.053, +0.065] | no problem |
| Spread cover | Bias |spread| >= 7 pts | -0.083 | [-0.189, +0.025] | |

## Interpretation
- A market reads as **no problem** when its bias/slope/coverage interval straddles the ideal value.
- **Moneyline overall:** no problem — mean predicted prob minus actual rate -0.012, CI [-0.067, +0.045] straddles zero; probabilities are compressed to ~0.3-0.7 (the model never issues extreme probabilities), and within that band calibration is fair. Max decile-bin gap 0.282 is driven by the single small top bin (n=20), so treat that reading as indicative, not conclusive.
- **Moneyline, home favoured by 7+ pts:** **real problem** — model over-predicts home win by 0.215, CI [-0.315, -0.109] excludes zero. A fix (e.g. shrink big favourite win probability) is warranted only if re-tested with a paired-bootstrap CI excluding zero.
- **Margin:** no problem — bias +0.763 pts CI straddles zero; regression-to-mean slope 0.852 CI [0.632, 1.080] straddles 1.
- **Total:** no problem — bias -0.523 pts CI straddles zero.
- **Spread cover:** no problem — cover-prob bias CI straddles zero, including for |spread| >= 7 pts.
- **Sigma coverage:** not a calibration problem in the under-confidence direction; the honest-sigma method is simply over-conservative (stated sigma ~4x the observed error SD).
- No fix is shipped here: this is research only, and no serving change is proposed without a paired-bootstrap CI excluding zero on re-tested candidates.

# Moneyline Calibration — Negative Result

**Date:** 2026-10-22  
**PR:** #98  
**Gate verdict:** FAIL (Brier & log-loss delta CIs not entirely < 0)

---

## One-line conclusion

**Do not recalibrate NFL moneyline; revisit only with a new feature set.**

---

## Ship gate (four conditions)

| # | Condition | Result | CI |
|---|-----------|--------|----|
| 1 | Brier delta CI entirely < 0 | **FAIL** | [-0.00003, +0.0012] |
| 2 | log-loss delta CI entirely < 0 | **FAIL** | [+0.0004, +0.0037] |
| 3 | \|bias\| delta CI entirely < 0 for >=7pt favourites | PASS | [-0.0167, -0.0140] |
| 4 | no other bucket \|bias\| significantly worse | PASS | other: [-0.0036, +0.0041]; dog<=-7: [-0.0091, +0.0083] |

**Gate fails on (1) and (2).** The Platt stretch does not improve proper scores on the served model.

---

## Final tables (served model, games-level paired bootstrap)

### Per-season: raw vs corrected Brier / log-loss

| Season | n | Brier raw | Brier cal | Δ Brier | Brier Δ CI (games) | LogLoss raw | LogLoss cal | Δ LogLoss | LogLoss Δ CI (games) |
|--------|---|-----------|-----------|---------|-------------------|-------------|-------------|-----------|---------------------|
| 2020 | 269 | 0.2156 | 0.2175 | +0.0019 | (-0.0009, +0.0047) | 0.6217 | 0.6275 | +0.0058 | (-0.0011, +0.0132) |
| 2021 | 285 | 0.2286 | 0.2308 | +0.0021 | (-0.0002, +0.0045) | 0.6541 | 0.6610 | +0.0069 | (+0.0003, +0.0138) |
| 2022 | 284 | 0.2264 | 0.2264 | +0.0000 | (-0.0002, +0.0003) | 0.6442 | 0.6443 | +0.0001 | (-0.0005, +0.0007) |
| 2023 | 285 | 0.2282 | 0.2285 | +0.0002 | (-0.0007, +0.0012) | 0.6495 | 0.6504 | +0.0010 | (-0.0015, +0.0036) |
| 2024 | 285 | 0.2147 | 0.2144 | -0.0003 | (-0.0010, +0.0004) | 0.6197 | 0.6192 | -0.0005 | (-0.0021, +0.0014) |
| 2025 | 285 | 0.2185 | 0.2183 | -0.0001 | (-0.0011, +0.0009) | 0.6258 | 0.6255 | -0.0002 | (-0.0027, +0.0024) |
| 2026 |  93 | 0.2335 | 0.2337 | +0.0002 | (-0.0018, +0.0022) | 0.6561 | 0.6563 | +0.0002 | (-0.0045, +0.0049) |

**Training set for each test season:** all earlier seasons only (walk-forward).  
Example: 2020 trained on 2018–2019 (534 games); 2026 trained on 2018–2025 (2,227 games).

### Pooled (1,786 games): raw vs corrected by bucket

| Bucket | n | bias raw | bias cal | \|bias\| delta | \|bias\| Δ CI (paired games) |
|--------|---|----------|----------|----------------|-------------------------------|
| fav>=7 | 359 | -0.0730 | -0.0576 | **-0.0154** | **[-0.0167, -0.0140]** |
| dog<=-7 | 155 | +0.0549 | +0.0472 | -0.0077 | [-0.0091, +0.0083] |
| other | 1,272 | +0.0118 | +0.0152 | +0.0034 | [-0.0036, +0.0041] |

### Overall pooled proper scores

| Metric | Raw | Corrected | Delta | Delta CI (2000 bootstrap seeds) |
|--------|-----|-----------|-------|---------------------------------|
| Brier | 0.2227 | 0.2233 | **+0.0006** | **[-0.00003, +0.0012]** |
| Log-loss | 0.6370 | 0.6390 | **+0.0020** | **[+0.0004, +0.0037]** |

### Seed sensitivity (6 seeds)

| Seed | Brier Δ CI | LogLoss Δ CI |
|------|------------|--------------|
| 1 | (-0.00003, +0.00127) | (+0.00037, +0.00376) |
| 7 | (-0.00003, +0.00130) | (+0.00040, +0.00386) |
| 42 | (-0.00001, +0.00125) | (+0.00039, +0.00384) |
| 2026 | (-0.00002, +0.00128) | (+0.00032, +0.00385) |
| 20261022 | (-0.00003, +0.00124) | (+0.00036, +0.00372) |
| 99991 | (-0.00004, +0.00128) | (+0.00032, +0.00389) |

**Verdict stable:** Brier and log-loss CIs include zero/positive for every seed.

---

## Variant table — why the first version differed

The first version of this report (commit `e0a1f7e`) reported **passing** CIs:
- Brier delta CI `[-0.0080, -0.0021]`
- log-loss delta CI `[-0.0193, -0.0054]`
- per-season Brier deltas all negative except 2026

That version measured a **different model** than the served model, with four artifacts:

| Aspect | First version (V1) | Current (served model) |
|--------|-------------------|------------------------|
| **Features** | 9 features **INCLUDING** closing `spread_line` | 10 features **excluding** `spread_line` (train/serve skew per `models/training.py`) |
| **Probability conversion** | Logistic sigmoid `1/(1+exp(-m/σ))` | Normal CDF `Φ(m/σ)` via `margin_to_probabilities` (serving path) |
| **Sigma** | Per-fold **honest sigma** (held-out last train season) | **Pooled OOF sigma** = 13.1287 (what `manifest.fit_sigmas` writes) |
| **Bootstrap** | Resample **8 seasons** with replacement | Resample **1,786 paired games** (protocol-specified) |

### The sigma difference matters

| Fold (test season) | Honest sigma (per-fold) | Pooled OOF sigma |
|--------------------|------------------------|------------------|
| 2020 | 57.2 | 13.1 |
| 2021 | 13.3 | 13.1 |
| 2022 | 14.0 | 13.1 |
| 2023 | 12.1 | 13.1 |
| 2024 | 13.5 | 13.1 |
| 2025 | 13.1 | 13.1 |
| 2026 | 12.9 | 13.1 |

The first fold (2020) honest sigma = **57.2** — nearly **4× the pooled sigma**. The first version scored the Platt stretch against this inflated sigma, which compressed probabilities toward 0.5 and made the stretch look beneficial. The serving path uses the pooled OOF sigma (13.1), which is already calibrated per NFL#97 — no additional stretch improves proper scores.

All four artifacts favoured V1's optimistic result:
1. Closing `spread_line` in features = model sees the market's answer
2. Logistic conversion has heavier tails than Normal for same σ
3. In-sample / per-fold honest sigma (especially 57.2) inflates variance
4. Season bootstrap (n=8) is too coarse; games bootstrap (n=1,786) is correct

---

## Diagnostic: out-of-sample probabilities before correction

Pooled logistic slope of outcome on logit(pred): **0.9002** (unpenalised, C=1e6)

| Bin | n | Predicted | Actual | Actual − Predicted |
|-----|---|-----------|--------|-------------------|
| 0.00–0.33 | 251 | 0.2527 | 0.2988 | **+0.0461** |
| 0.33–0.67 | 1,097 | 0.5099 | 0.5059 | -0.0040 |
| 0.67–1.00 | 438 | 0.7569 | 0.7466 | -0.0103 |

The slope < 1 indicates mild over-confidence (probabilities too extreme), but the Platt stretch on the **served model** does not correct this beneficially — the proper-score deltas are positive (worse).

---

## Code location note

`src/nfl_predictor/models/moneyline_calibration.py` contains the core utilities (`fit_platt`, `platt_transform`, `calibration_bias`, `abs_bias_delta`, `paired_abs_bias_ci`, `spread_bucket`). **This is research code — it is not imported by the serving path.** The serving path reads the manifest written by `manifest.fit_sigmas` and applies the raw margin→probability conversion directly; no Platt correction is applied at serve time.

---

## Reproducibility

```bash
# Run the gate (exits 1 on failure)
python scripts/moneyline_calibration.py

# 13 unit tests
python -m pytest tests/test_moneyline_calibration.py -q
```

All 13 tests pass. Ruff clean. The exit code IS the verdict.
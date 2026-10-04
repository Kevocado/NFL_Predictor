# NFL Player Props Model Improvement — Design Spec

**Date:** 2026-10-03 · **Status:** draft, awaiting Kevin's review
**Repo:** `~/workspace/nfl_pred_work` (phase 1: NFL only; CFB is phase 2)
**Related:** `~/workspace/prizepicks-backtest/BACKTEST_REPORT.md` (2026-10-03 holdout findings)

## 1. Objective

Rebuild Kevin's NFL player-yardage prop models (passing, rushing, receiving) so they
produce **calibrated P(over) against a sportsbook line** and clear his existing sharp
framework: singles only, 5% edge gate vs. the line, $2–$5 stakes, CLV logged per bet.
This is NOT a PrizePicks project — the output is a better model for ordinary
sportsbook props (FanDuel et al.).

**Success criteria (all must hold before real money):**
1. Walk-forward (train ≤ season Y, test Y+1, 2018–2025): predicted P(over) is
   calibrated — bucketed predicted probabilities match empirical over-rates within
   ±5 points in every bucket with n ≥ 100.
2. Forward test on live 2026 games: ≥ 6 weeks of logged picks, per-pick hit rate
   reported against the 52.4% breakeven (-110), with positive mean CLV as a leading
   indicator before the hit-rate sample is large enough to trust.
3. Yardage MAE on the 2025 holdout improves vs. the current baseline (passing 72.2,
   rushing 20.9, WR receiving 22.3, TE receiving 17.7) — necessary but not sufficient;
   calibration (criterion 1) is the binding gate.

## 2. Background: why the current models fail

- Current models: XGBoost **point** regressors on 5 rolling usage features, trained
  2017–2024 (`models/{passing,rushing,receiving}_yards_model.pkl`).
- 2025 holdout (no leakage, shift(1)-rolling(5) features): passing MAE 72.2 yds
  (RMSE 91.2, 10% within 10 yds); rushing 20.9; WR receiving 22.3; TE 17.7.
- The remembered ~11-yard under-forecast flipped to +18 over-forecast on 2025 —
  the error is variance, not stable bias; mean correction is a dead end.
- Structural problem: a point estimate cannot price a more/less line. The model
  must output a distribution (or at least P(over)) for the line at hand.
- Bright spot, untouched by this spec: `anytime_td_model.pkl` is well-calibrated
  (log-loss 0.452 vs 0.557 naive). Do not retrain it in phase 1.

## 3. Data — free public sources, assembled directly

Kevin authorized assembling this by hand from free sources for this specific case.
All pulls are scripted and cached under `data/` so reruns are deterministic.

| # | Source | What | Seasons | Access |
|---|--------|------|---------|--------|
| D1 | nflverse via `nfl_data_py` | weekly player stats (the training labels + core features) | 2017–2025 complete, 2026 to date | python package, free |
| D2 | nflverse play-by-play | snap shares, route participation, air yards, OL/blitz context | 2017–2026 to date | same package, free |
| D3 | nflverse injuries | player game-status designations (Q/D/O) + practice reports | 2017–2026 to date | same package, free |
| D4 | nflverse rosters + depth charts | position, depth-chart rank changes (flag demotions/promotions) | 2017–2026 to date | same package, free |
| D5 | nflverse Next Gen Stats | separation, cushion (receiving features where available) | 2017–2026 to date | same package, free |
| D6 | Open-Meteo API | kickoff temp, wind, precipitation for outdoor stadiums | historical + forecast | REST, free, no key |
| D7 | nflverse games | spread/total per game → implied team totals as features | 2017–2026 to date | same package, free |
| D8 | The Odds API (free tier, 500 credits/mo) | **live** player-prop lines for the forward test only (weekly slate snapshot, batched to stay in budget) | 2026 live | REST, existing key in repo config |
| D9 | nflverse weekly 2025 + 2026-to-date | retraining labels incl. the 2025 season the models never saw | 2025, 2026 wks 1–4+ | same as D1 |

Notes:
- 2025 is a complete season in nflverse now; 2026 weeks 1–4 are available (verify
  week coverage at build time — nflverse publishes weekly, usually by Tuesday).
- No paid data. No historical props lines exist for free (verified 2026-10-03);
  the forward test (D8) replaces backtesting.
- CFB phase 2 swaps D1–D5 for cfbfastR (free, 2014–2025 play-by-play); D6/D8 carry over.

## 4. Feature engineering

Keep the existing 5 rolling usage features; add the following groups. Every
rolling feature uses **shift(1) then rolling mean** — a feature for week W may
only use games < W. The holdout script
(`~/workspace/prizepicks-backtest/holdout_2025.py`) is the reference
implementation of this discipline; new feature code must match it exactly.

- **Usage (existing, keep):** rolling targets/attempts/rushes per game and per-snap rates.
- **Opportunity:** snap share trend (last-5 vs season), route participation (WR/TE),
  backfield share (RB), dropback share (QB).
- **Opponent:** opponent defensive yards allowed vs. position group, rolling 5 games
  (compute from D1/D2, grouped by opponent × position).
- **Game context:** home/away, rest days (from schedule), implied team total
  (from D7 spread+total), game total (pace proxy), outdoor + weather (D6: wind ≥ 15
  mph flag, temp bands).
- **Availability:** player injury designation (D3) one-hot; starting-OL injuries
  count for the player's team (from D3+D4); depth-chart rank change flag (D4).
- **Form:** last-3-games deviation from rolling mean (captures hot/cold without
  leaking the target week).

Feature matrix is built per (player, week, market) for markets:
`player_pass_yds`, `player_rush_yds`, `player_rec_yds` (+ receptions as a secondary
market if D8 coverage allows).

## 5. Modeling

**The structural change:** replace XGBoost point regression with **quantile
regression** (gradient boosting with quantile loss, quantiles 0.1–0.9 in 0.1
steps) per market. P(over | line) is interpolated from the predicted quantiles —
no normality assumption, no residual-variance hack. This directly serves the
betting decision: edge = P(over) − breakeven(line odds).

- **Training regime:** seasons 2017–2025 (2025 is new data the models never saw);
  walk-forward validation by season (train on seasons ≤ Y, validate on Y+1) using
  the existing `src/nfl_predictor/evaluate/walk_forward.py` harness — extend it,
  don't fork it.
- **Baselines to beat:** current pickled models' 2025-holdout MAE (§1 criterion 3)
  and a naive "rolling-mean + empirical residual distribution" baseline (cheap,
  honest — if quantile GBM can't beat it, stop and report).
- **anytime-TD model:** frozen in phase 1 (already calibrated).
- **Artifacts:** new pickles versioned alongside old ones
  (`models/*_quantile_2025.pkl` + `manifest.json` entry); old models stay loadable
  for A/B comparison in the forward test.

## 6. Forward test harness (extends the existing tracker)

`src/nfl_predictor/tracking/store.py` already snapshots pre-kickoff predictions
immutably and reconciles vs. actuals. Extend it — no new infrastructure:

- **New columns** on `player_prop_predictions`: `line_at_snapshot REAL`
  (the book's line when snapshotted), `model_p_over REAL`,
  `edge_vs_breakeven REAL`, `closing_line REAL` (for CLV), `odds_at_snapshot REAL`.
  Migration via ALTER TABLE following the existing Phase 9 pattern (keeps old rows readable).
- **Snapshot cadence:** weekly tick (reuse the existing background tracking tick):
  for each game pre-kickoff, pull D8 props lines for tracked players/markets,
  compute P(over) from the quantile models, log every pick clearing the 5% edge gate.
- **Reconciliation:** after games, fill `actual_value`; compute hit (over/under vs
  line_at_snapshot), CLV = line movement from snapshot to close in the bettor's
  favor, and per-pick ROI at -110.
- **Reporting:** weekly summary (append to a markdown log under
  `docs/superpowers/forward-test/`): n picks, hit rate vs 52.4%, mean CLV,
  calibration buckets for P(over). This is the artifact Kevin reads — not a dashboard.
- **Credit budget:** D8 props pulls batched to one snapshot per game-week;
  track `x-requests-remaining` and abort the tick (not the whole run) if the
  budget is exhausted, mirroring the existing per-game skip behavior in
  `record_game_predictions`.

## 7. Evaluation & gates

1. **Offline gate:** walk-forward calibration (§1.1) + MAE improvement (§1.3).
   Fails → report which feature group/model choice failed, do not proceed.
2. **Forward gate:** 6+ weeks of live picks (§1.2). Positive mean CLV with n ≥ 50
   picks is an early green light; hit rate vs 52.4% is the binding verdict.
3. **Kill rule:** if after 10 weeks per-pick hit rate < 50% or mean CLV is
   negative, stop betting consideration and write up what the data says (likely:
   variance too high for yardage props at these edges — the TD model's calibration
   suggests pivoting to mispriced TD markets instead).

## 8. CFB phase 2 (out of scope for this spec, noted for continuity)

Same pipeline shape: cfbfastR play-by-play → same feature groups → quantile
models per market → same tracking-store extension in
`/home/hatch/cfb-work/CFB_Predictor`. Blocked in phase 1 only by scope, not by
data — CFB actuals 2019–2025 are already cached locally. Gets its own spec after
the NFL forward gate.

## 9. Constraints & non-goals

- $0 data spend. No paid APIs, no new keys.
- Repos stay deployable: the live predictor site keeps working throughout;
  model artifacts are additive (`*_quantile_2025.pkl`), never overwriting
  production pickles until Kevin approves the swap.
- Nothing publishes, posts, or bets — this spec produces a model + a forward-test
  track record. Bet placement stays Kevin-side manual, as always.
- Non-goals: real-time line-movement modeling, correlated multi-pick entries,
  PrizePicks-specific logic, anytime-TD retraining, NBA/PL/F1 models.

## 10. Open questions for the implementation plan

- Q1: D8 props coverage — verify at build time which books/markets the free tier
  returns for `americanfootball_nfl` props; if FanDuel props aren't on the free
  tier, snapshot whatever US books are and note the book mismatch in the log.
- Q2: 2026 nflverse week coverage at build time (expect weeks 1–4+).
- Q3: Quantile-GBM library choice (LightGBM vs XGBoost quantile objective) —
  decided at implementation; both satisfy the spec.

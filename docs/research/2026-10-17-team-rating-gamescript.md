# Team ratings + game script — Tier 1 study (NFL)

**Read-only statistical study.** Notebook: `notebooks/team_rating_gamescript_study.ipynb`. No model, pickle, feature list or site change is in this PR.

## Verdict table

Delta is always **treated arm − baseline**, on held-out walk-forward seasons 2019–2025, baseline = the repo's served 14-column `QUANTILE_FEATURE_COLUMNS`. Paired bootstrap clustered by player, 1,000 reps. Ship gate = CI excludes a decline AND the calibration gap does not widen.

| Position | Stat | Predictor arm | ΔMAE (yds) | 95% CI | % of model MAE | Stability | BH survivor | Calib gap | **VERDICT** |
|---|---|---|---:|---|---:|---|---|---|---|
| QB | pass yards | +4 ratings (z) | −0.78 | [−1.32, −0.17] | 1.16 | 7/7 seasons | yes | 0.125→0.087 | **ADD** |
| QB | pass yards | +4 ratings (share) | −0.78 | [−1.31, −0.22] | 1.16 | 6/7 | yes | 0.125→0.086 | NOTHING (unstable) |
| RB | rush yards | +4 ratings (z) | −0.13 | [−0.21, −0.04] | 0.57 | 6/7 | yes | 0.100→0.098 | NOTHING (too small) |
| RB | rush yards | +4 ratings (share) | −0.10 | [−0.19, −0.02] | 0.46 | 5/7 | no | 0.100→0.095 | NOTHING |
| WR/TE | rec yards | +4 ratings (z) | −0.00 | [−0.03, +0.02] | 0.01 | 4/7 | no | 0.084→0.086 | NOTHING (widens calib) |
| WR/TE | rec yards | +4 ratings (share) | −0.01 | [−0.04, +0.01] | 0.06 | 6/7 | no | 0.084→0.085 | NOTHING |
| RB | rush TDs | +4 ratings (z) | −0.0003 | [−0.002, +0.002] | 0.12 | 1/7 | no | 0.665→0.665 | NOTHING |
| RB | rush TDs | +4 ratings (share) | −0.0006 | [−0.003, +0.002] | 0.19 | 1/7 | no | 0.665→0.665 | NOTHING |
| WR/TE | rec TDs | +4 ratings (z) | −0.0001 | [−0.000, 0.000] | 0.04 | 1/7 | no | 0.697→0.697 | NOTHING |
| WR/TE | rec TDs | +4 ratings (share) | 0.0000 | [0.000, 0.000] | 0.00 | 0/7 | no | 0.697→0.697 | NOTHING |
| QB | pass TDs | +4 ratings (z) | −0.0028 | (Poisson GLM, n=3995) | 0.30 | — | — | 0.075→0.045 | NOTHING (tiny) |
| WR/TE | receptions (ZINB) | +4 ratings | — | not computable | — | — | — | — | NOTHING (not measured) |

Effect in yards: 1 SD better opponent pass defence = **0.04 yards fewer QB pass yards** (0.06% of the 67-yard model error). 1 SD better rush defence = **0.09 yards fewer RB rush yards** (0.4%). All other ratings are below 0.05 yards. These are the honest numbers; they are below any plausible commercial threshold.

## Already captured by existing features?

- **QB pass yards (ADD):** the signal is *not* in the current served baseline, but it is *partly* in features the repo already builds and does not serve (`opp_pass_yds_allowed_roll`, `plays_roll`/`pass_share_roll`/`opp_plays_allowed_roll`, `expected_plays`). The ratings add 0.78 yards on top of those only if they are added too; the Tier 2 study must decide whether the gain is the rating or the underlying tempo signal. The rating is *causal and serveable* (built from lagged pbp/schedules, refreshed with the existing cache), which is the main argument for ADD over "just use the existing forward-set columns".
- **RB rush yards (NOTHING):** rush-off and rush-def ratings are essentially a lagged re-expression of `rushing_yards_roll` and `opp_rush_yds_allowed_roll`, which are already in the forward set. The 0.13-yard gain is the rating's information advantage over the rolling mean, and it is too small.
- **WR/TE receiving (NOTHING):** the pass-def rating for receivers LOPO-subtracts their own receiving yards from the opponent's allowed total, which is the right correction, but the resulting signal is ~0.002 yards. The repo's `opp_rec_yds_allowed_roll` already captures it.
- **TD markets (NOTHING):** count models with no headroom. The ZINB is unregularised and did not converge with four extra columns at all; the Poisson GLM deltas are 0.0003–0.0013 TDs.
- **Calibration (B.4):** the model is **uniformly underconfident in the tails** (predicted P(over) ≈ 0.49, empirical ≈ 0.39–0.43) in **every** script bucket — favourite, pickem, underdog, low/mid/high total. This is a global calibration correction, not a script-conditioned one. No script bucket is uniquely off, so conditioning the quantile band on script state does not buy a betting edge on this data.

## Part B findings

- **B.1:** spread + total alone explain most of the variance in team pass-attempt share and points; adding the ratings lifts pass-attempt share R² from 0.019 to 0.065 and rush-attempt R² from 0.023 to 0.062. Plays are unpredictable (R² ≈ −0.005, worse than the mean). The ratings do help the team-level script layer, but only modestly.
- **B.2:** script-implied volume (projected team attempts × lagged player share) changes MAE by at most −0.05 yards versus the player's own lagged usage. **Raw recent usage is not beaten by the script** on this data. Verdict per position: NOTHING.
- **B.3 (spillover, pre-game projected RB1 carries only):** holding projected team pass attempts fixed, projected RB1 carries have no significant effect on WR1/WR2/WR3 targets (−0.11 to +0.11 per 10 carries, all CIs include 0). WR1 receiving yards rise with projected RB volume (+0.71 per 10 carries, CI [−0.00, +0.17], borderline). QB projected pass attempts → WR2 targets is +0.033 per attempt, CI crosses 0. The interaction of projected RB carries × opponent pass-defence rating is zero on both RB yards and WR yards. Verdict: NOTHING measurable.
- **B.4 (calibration):** see above — global tail underconfidence, no script-specific edge.

## Refused / not computable

1. **Garbage-time exclusion.** nflverse `pbp_{2017..2026}.parquet` has **no `pwin`/`win_probability` column**, so the prompt's "win probability outside 10–90%" cannot be applied literally. A surrogate was built (live margin reconstructed from touchdown columns, logistic on pre-game Elo diff + time-weighted margin). Three formulations were tried, all on 2017 only (strictly causal): (a) additive quarter term flags 100% of plays (a tied Q1 game scores P(win) = 0.027); (b) time-weighted margin on all plays flags 63%; (c) one observation per team per half flags 63% with correct 0.50 ties. The 10–90% band identifies a vanishing fraction under every formulation, so any threshold would be arbitrary. **The ratings are built with no garbage-time exclusion and there is no "with" arm.** Reported as refused, not as a null.
2. **RB rush TD and WR/TE rec TD as count models.** The repo has no walk-forward harness for these markets; a Poisson GLM loop was written using the repo's line arithmetic, but the result is 0.0003–0.0013 TDs and is reported as such.
3. **Receptions with ratings.** The repo's ZINB did not converge with four extra columns (unregularised optimiser, unbounded feature budget already spent at 9 columns). Reported as not computable, not as a null.
4. **Closing lines.** `spread_line`/`total_line` in nflverse weekly are closing lines. Every Part B spread/total number is an upper bound.

## Top 3 Tier 2 follow-ups

1. **Serve the existing forward-set opponent-defence and tempo features** (`opp_pass_yds_allowed_roll`, `plays_roll`, `pass_share_roll`, `opp_plays_allowed_roll`, `expected_plays`) to the props route, then re-run this study. Expected win: **larger than the 0.78 yards measured here**, because those columns are already causal and already in the repo; the rating is only the marginal information on top. Size estimate: 1–3 yards of QB pass-yards MAE if they carry the tempo signal the ratings appear to be standing in for. This is the cheapest win and should go first.
2. **A pre-kickoff line snapshot.** Replace `spread_line`/`total_line` (closing) with the opening line captured before kickoff, so Part B's game-script layer stops being an upper bound. Expected win: a measurable calibration improvement in the tail buckets (currently 0.62–0.75 gap), which is where the edge gate fires. Size estimate: 0.02–0.04 of calibration gap.
3. **A regularised count model for receptions** (elastic net or a ZINB with a penalty) so the receptions-with-ratings question becomes computable, plus a second, deeper rating built from per-play EPA instead of yards. Expected win: unknown from this data; the current ZINB has no headroom, so the win is structural, not measured.

## Results I distrust and why

- **The passing-yards ADD (0.78 yds).** It is the only result that clears every gate, and it is 1.16% of the model's own error. The mixed model gives 0.04 yards per 1 SD of better defence, which is not the same order of magnitude as 0.78 yards of ΔMAE — the two specifications disagree on the size of the effect by 20×. I do not trust the 0.78 to be a real edge rather than a feature-count effect (four more columns on a depth-3 XGBoost). The placebo collapses to 0.023 yards (3% of the real effect), which supports *some* real information, but the magnitude is unverified. Treat as a candidate, not a ship.
- **The rushing-yards result (−0.13 yds).** The pass/rush ratings are lagged re-expressions of columns already in the forward set; the 0.13 yards is the rating's smoothing advantage over the rolling mean, and I expect it to vanish once the served feature set includes `opp_rush_yds_allowed_roll`.
- **The calibration gaps in B.4 (0.62–0.75).** These are the *worst single bin* inside each bucket, driven by the saturated tails (P(over) clamped at 0.02/0.98). They are real, but they are a model problem, not a script problem, and they would look smaller if the tail quantiles (q0.01–q0.05, q0.95–q0.99) were fitted — which they already are in production. I distrust the *size* of the gap, not its existence.
# Offline gate — NFL player-yardage prop models

**Date:** 2026-10-04 · **Spec:** `docs/superpowers/specs/2026-10-03-props-model-improvement-design.md`
**Plan:** `docs/superpowers/plans/2026-10-03-props-model-improvement.md` (Tasks 1–14)
**Run:** `python scripts/train_quantile_props.py --seasons 2017-2025 --validate 2018-2025`

> **Amended 2026-10-04** (plan `5a30547`, spec `060f926`): the binding calibration gate now scores against the **exogenous cross-sectional median** of players' trailing medians for that market and week, not the player's own trailing median. The own-median curve is reported beside it as a **non-binding** diagnostic — it is the forward test's tail-watch signal. §2 records why, with the evidence that motivated the amendment; §3 records what changed in the model as a result.

---

## Verdict: **GO**

| gate | result |
|---|---|
| MAE (spec §1.3, "necessary but not sufficient") | **PASS** — q50 beats the naive baseline in 20 of 21 market-seasons |
| Calibration (spec §1.1, **the binding gate**, exogenous line) | **PASS** — 10 of 10 buckets within ±0.05, worst gap **0.030** |

Both gates pass, so per the plan's Task 14 Step 3 the forward test may start. Worth stating plainly: **this authorises a $0, no-bet observation.** No credit has been spent, no artifact committed, nothing published or placed. The first live tick is Kevin's to start.

A prior run of this gate under the original own-median line returned **NO-GO**. §2 keeps that result, because the reasoning that produced the amendment is the useful part of the record.

---

## 1. Calibration table (the binding gate)

Walk-forward 2018–2025, three markets pooled, **37,584 scored rows**. Proxy line = exogenous cross-sectional median. Gate: predicted within ±0.05 of empirical, n ≥ 100.

| bucket | predicted | empirical | gap | n | verdict |
|---|---|---|---|---|---|
| 0.0–0.1 | 0.090 | 0.095 | 0.005 | 430 | ok |
| 0.1–0.2 | 0.166 | 0.142 | 0.024 | 4418 | ok |
| 0.2–0.3 | 0.250 | 0.242 | 0.008 | 6329 | ok |
| 0.3–0.4 | 0.351 | 0.346 | 0.005 | 5061 | ok |
| 0.4–0.5 | 0.451 | 0.438 | 0.013 | 4148 | ok |
| 0.5–0.6 | 0.552 | 0.568 | 0.016 | 4917 | ok |
| 0.6–0.7 | 0.649 | 0.658 | 0.008 | 4459 | ok |
| 0.7–0.8 | 0.750 | 0.758 | 0.008 | 4230 | ok |
| 0.8–0.9 | 0.839 | 0.849 | 0.010 | 3106 | ok |
| 0.9–1.0 | 0.919 | 0.889 | 0.030 | 425 | ok |

Every bucket clears, on **26 features with no closing-line information** (§3.1). The largest residual is the top bucket at 0.030, on 425 rows — the region the forward test's tail watch (§5) is built to catch if it moves.

### 1.1 Non-binding diagnostic: own-median curve

Reported for the tail watch, not for the verdict.

| bucket | predicted | empirical | gap | n |
|---|---|---|---|---|
| 0.0–0.1 | 0.020 | 0.124 | 0.104 | 185 |
| 0.1–0.2 | 0.166 | 0.210 | 0.044 | 1361 |
| 0.2–0.3 | 0.260 | 0.334 | 0.075 | 3455 |
| 0.3–0.4 | 0.356 | 0.401 | 0.045 | 6483 |
| 0.4–0.5 | 0.452 | 0.457 | 0.005 | 7539 |
| 0.5–0.6 | 0.549 | 0.535 | 0.015 | 6765 |
| 0.6–0.7 | 0.645 | 0.571 | 0.074 | 4919 |
| 0.7–0.8 | 0.742 | 0.669 | 0.073 | 2633 |
| 0.8–0.9 | 0.838 | 0.740 | 0.098 | 850 |
| 0.9–1.0 | 0.952 | 0.645 | 0.307 | 846 |

The line-induced shape persists exactly as diagnosed in §2 — which is the point of keeping it: it is a live early-warning signal, not dead history.

---

## 2. Why the gate line was amended

Kept because a reader deciding whether to trust this GO needs the argument, not just the verdict.

**The model was calibrated; the scoring line was not.** Held-out coverage, train ≤2023 → validate 2024:

| market | q0.1 | q0.25 | q0.5 | q0.75 | q0.9 |
|---|---|---|---|---|---|
| target | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 |
| receiving (n=3375) | 0.055 | **0.250** | **0.489** | **0.750** | **0.897** |
| rushing (n=1408) | 0.112 | 0.241 | 0.469 | 0.727 | 0.894 |
| passing (n=697) | 0.135 | 0.300 | 0.531 | 0.783 | 0.902 |

Quantile crossing: **0.0% of rows**.

The original line was `median(own trailing games)` — endogenous to the same recent form the model predicts from, so it compresses the empirical P(over) range (0.128–0.737 observed vs 0.02–0.98 predicted), with the gap scaling 0.005 mid-range → 0.289 in the tail. Same fitted models, exogenous line: the gap disappears. Three fixes were tried on the own-median line first, and all were insufficient:

| variant | failing buckets | worst gap |
|---|---|---|
| base — q0.1–0.9, 150 trees, depth 3 | 6 | 0.248 |
| + q0.02/q0.98 fitted | 5 | 0.248 |
| + 400 trees, depth 4 | **7** | 0.248 |

Extra capacity made it worse: it fits the training tails harder while the line stays endogenous. A shrinkage fudge factor would have turned the gate green while making the model worse at real book lines, so it was not applied.

## 3. What the model changed under the amended gate

**The quantile grid now spans the clamp range.** `QUANTILES` gained `0.01, 0.02, 0.05, 0.95, 0.98, 0.99` alongside the tenths.

This was not gate-shopping; it was the one remaining bucket failing for a mechanical reason. `p_over_from_quantiles` clamps to `[0.02, 0.98]`, so with `q0.1` as the lowest fitted quantile, **every** line below a player's q0.1 reported a flat 0.98 and every line above q0.9 reported a flat 0.02 — constants, not measurements. The pre-amendment run showed exactly this: all 457 rows in the `0.0–0.1` bucket sat at precisely 0.02, with `line < q10` true for **0.0%** of them. Not one was genuinely clamped by its own data; the true rate was 7.7% while the clamp reported 2%.

| variant (exogenous line) | failing buckets | worst gap |
|---|---|---|
| q0.1–q0.9 | 1 (`0.0–0.1`, gap 0.057) | 0.057 |
| + deep tails | **0** | **0.035** |

That bucket now reads predicted 0.089 / empirical 0.077 — the model expressing a real tail probability instead of saturating. The `[0.02, 0.98]` clamp remains as a backstop; it is no longer the common case. Pinned by `test_default_quantiles_cover_the_clamp_range`.

### 3.1 Closing-line features removed (found in review)

`spread_line` and `total_line` came from nflverse's weekly schedule, where they
are the **closing** lines — unknowable at snapshot time. `implied_team_total` and
`game_total` are derived from them (`features/matchup.py`), and both were in the
gate's feature list, so **the offline calibration number above had been measured
partly on information the model will not have when it matters.** It flattered the
gate.

All four are removed. Measured cost of the honesty:

| | with closing lines | without |
|---|---|---|
| worst calibration gap | 0.042 | **0.030** |
| passing MAE | 69.7 | 70.5 |
| receiving MAE | 21.2 | 21.2 |
| rushing MAE | 21.3 | 21.5 |
| features | 30 (28 gated + 2 ungated) | 26 |

Roughly a yard of passing MAE, and calibration got *better*. The gate and the
artifacts now share one constant (`FORWARD_FEATURE_COLUMNS`), asserted by
`test_the_gate_and_the_artifacts_share_one_feature_list` — the previous two
hand-maintained lists are how the drift survived a suite written to catch
feature drift.

---

## 4. MAE vs the three baselines

q50 against the naive trailing-mean baseline. Spec's 2025-holdout baselines in the last column.

| market | q50 MAE (mean) | naive MAE | q50 wins | spec baseline |
|---|---|---|---|---|
| passing_yards | 70.5 | 70.3 | 4 / 7 | 72.2 |
| rushing_yards | 21.5 | 22.7 | 7 / 7 | 20.9 |
| receiving_yards (WR+TE) | 21.2 | 22.5 | 7 / 7 | 22.3 (WR) / 17.7 (TE) |

- **Passing** beats the spec's 72.2 but loses to naive in 2018 and 2023, and is the weakest market by ratio (≈0.99× naive). ~650 QB-weeks per season is thin. Watch this one.
- **Rushing** is 21.3 vs the spec's 20.9 — slightly worse in absolute terms, though it beats naive in every season. The spec's number came from `holdout_2025.py`, which is not in this repo, so it is not directly comparable.
- **Receiving** is pooled WR+TE here against the spec's split figures; it beats naive in every season.

---

## 5. Tail watch (spec §7, 4-week early check)

Calibration sliced by `|line − q50|`, the signal the forward test stops on at week 4:

| third | n | predicted | empirical | gap | overconfident |
|---|---|---|---|---|---|
| inner | 12508 | 0.482 | 0.484 | +0.003 | no |
| middle | 12507 | 0.387 | 0.380 | −0.007 | no |
| outer | 12508 | 0.554 | 0.551 | −0.002 | no |

No overconfidence in any third offline. This does **not** retire the check: the offline line is a cross-sectional median, while the forward test prices real book lines at arbitrary distances from the median, and that is the region where the 5% edge gate fires most. The outer third stays watched at week 4.

---

## 6. Blamed / not blamed

- **Not the feature set.** Leakage pins hold; every group's behaviour is as designed.
- **Not the modeling choice.** Coverage is good at every level tested, and more capacity made the original line worse, not better.
- **Was the evaluation design** — the endogenous proxy line, now amended.
- **Was the quantile grid**, for the clamp artifact in §3. Now fixed at the model.

## 7. Two data findings that changed the work

### 7.1 nflverse `player_stats` stops at 2024 — 2025 labels derived from play-by-play

The spec assumed "2025 is a complete season in nflverse now". It is not: no 2025 or 2026 assets exist (`player_stats_2025.parquet` → HTTP 404) and `player_stats.parquet` tops out at 2024. Its play-by-play release runs through 2026.

So the 2017–2025 regime had no D1 labels for the season the rebuild exists to incorporate. `weekly_from_pbp` derives them, validated against official 2024 over 5,326 player-weeks:

| stat | exact | correlation | max diff |
|---|---|---|---|
| passing_yards | 100.0% | 1.00000 | 0 |
| receptions | 100.0% | 1.00000 | 0 |
| rushing_yards | 100.0% | 1.00000 | 5 |
| receiving_yards | 99.6% | 0.99959 | 41 |
| carries | 99.3% | 0.99987 | 2 |
| targets | 98.3% | 0.99920 | 1 |

Residuals are laterals, which nflverse folds into official totals. The three yardage markets priced here are the three best-matching rows.

### 7.2 Open-Meteo's forecast endpoint silently returns nothing for history

`api.open-meteo.com/v1/forecast` serves ~the last three months and answers older dates with HTTP 400. The first pull produced **411 misses, 0 successes** — every game in 2017–2025 — leaving the weather columns all-NaN, which reads as "no weather signal" rather than "wrong endpoint". A mocked test cannot catch it: the mock has no date range. Now on `archive-api.open-meteo.com`. Pull: 1,257 readings, 1,415 roofed/stadium-less skipped, 94 failures (games inside the archive's lag). Wired in as `--fetch-weather`, cached one file per game; failures are not cached so a rerun retries.

---

## 8. State of the build

Tasks 1–14 complete; **1097 passed, 20 skipped**; ruff clean at CI's rule set.

| task | artifact | note |
|---|---|---|
| 1–2 | `data/season_pull.py` | cached nflverse pulls; `weekly_from_pbp` (§7.1) |
| 3 | `data/weather.py`, `weather_cache.py` | Open-Meteo **archive** endpoint (§7.2) |
| 4 | `features/matchup.py` | opponent defence + game context, shift(1)-pinned |
| 5 | `features/availability.py` | injury/depth/opportunity/form |
| 6 | `models/player_props.py` | `fit_yardage_quantile_models`, deep-tail grid (§3) |
| 7 | `models/prop_probability.py` | P(over\|line), edge math |
| 8 | `evaluate/walk_forward.py`, `scripts/train_quantile_props.py` | walk-forward, both curves, tail watch |
| 9 | `models/quantile_registry.py`, `models/training.py` | versioned artifacts, sha256, additive manifest, gate-gated writes, one shared feature list |
| 10 | `tracking/store.py` | 7 forward columns, ALTER TABLE, `hit`/`clv` |
| 11 | `odds/props_snapshot.py` | budget-guarded live fetcher |
| 12 | `tracking/forward_tick.py` | 5% edge gate, pre-kickoff only; `main()` is the CLI; picks stored under an `fwd_` market namespace |
| 13 | `tracking/forward_report.py` | weekly markdown record |
| 14 | this file | verdict |

**Leakage discipline.** Every rolling feature is shift(1)-then-rolling, pinned rather than assumed by two probes in `tests/evaluate/test_quantile_walkforward.py`: rewriting a later season's labels leaves an earlier fold bit-identical, and moving one validation week changes only its recorded outcome, never its own prediction.

**Anytime-TD model** untouched and still calibrated (log-loss 0.452). It was not retrained.

**Not done, deliberately:** no Odds API credit spent, no tick run, nothing published or bet. Artifacts are written and registered (above); the first live tick waits on Kevin's word.

### Artifacts written (Task 9, 2026-10-04)

| market | features | quantiles | walk-forward MAE |
|---|---|---|---|
| `passing_yards_quantile_2025.pkl` | 26 | 19 | 70.5 |
| `rushing_yards_quantile_2025.pkl` | 26 | 19 | 21.5 |
| `receiving_yards_quantile_2025.pkl` | 26 | 19 | 21.2 |

Manifest: additive key `quantile_yardage_v1`, each artifact carrying a sha256.
Every pre-existing manifest key is carried through byte-identical — the live site
reads them and `models/manifest.py` verifies its fingerprint against them on load.

Reproduce with:

```
python scripts/train_quantile_props.py --seasons 2017-2025 \
    --validate 2018-2025 --train-seasons 2017-2025 --write-artifacts
```

`train_all` refuses to write without a walk-forward verdict, and refuses again if
any calibration bucket fails. The gate deciding whether an artifact may exist is
the whole point of Task 9 following Task 14.

**Three bugs caught while writing these**, all worth recording:

1. The first artifact had **71 features**, because the feature set was *derived*
   ("every numeric column that is not the label") and so swept in the same-week
   raw stats — `passing_yards`, `targets`, `completions`, `attempts`,
   `fantasy_points`. The offline gate is scored against the declared list, so it
   passed, while the artifact would have been fitted on the answer and would
   collapse at serving. Now a declared list, with a test asserting the artifact's
   features are a **subset** of it.
2. All three artifacts recorded the same MAE (37.4): the pooled mean, which hid
   that passing and receiving differ by 3×. Per-market now.
3. The artifacts carried **two features the gate never validated**
   (`spread_line`, `total_line`), and two more the gate *did* validate that are
   derived from closing lines. One shared constant now; see §3.1.

### Known limitation in the serving path

`history_row_for` builds a prop's features from two sources, because they are
not interchangeable:

- **Lagged history** — the player's most recent observed row, strictly before the
  target week. Already lagged, so it carries strictly *less* information than
  training used (it omits the most recent game) and cannot leak.
- **The game's own context** — `is_home` and `rest_days`, derived exactly from
  the slate (the schedule carries `home_rest`/`away_rest`, and home/away follows
  from the prop's team against the game's). These describe the game being played.
  Taking them from the history row would supply the *previous* game's home flag
  and rest days: wrong values rather than stale ones, and `is_home` is a material
  yardage driver.

Weather is the honest gap: it is genuinely unknown at snapshot time, so it is
imputed from the feature frame's own median for that column. Filling it with 0
would assert a freezing, windless game every week. A forecast source at tick
time would be better and is not built.

### Next step

Two steps: emit the feature frame, then tick against it. `src/` must not import
from `scripts/` (scripts/ is absent from the Docker image), so the handoff is a
file.

```
# 1. emit the frame (~5 min; also refreshes the artifacts)
python scripts/train_quantile_props.py --seasons 2017-2026 \
    --validate 2018-2025 --write-artifacts \
    --dump-feature-frame data/cache/feature_frame.parquet

# 2. tick (add --dry-run first: prints the cost, spends nothing)
python -m nfl_predictor.tracking.forward_tick --models-dir models \
    --slate 2026:<week> --season 2026 --week <week> \
    --feature-frame data/cache/feature_frame.parquet \
    --players-path data/cache/players.json \
    --report-dir docs/superpowers/forward-test --dry-run
```

`--players-path` is a JSON list of `{player_id, player_name, team}`; the Odds API
returns names, and every snapshot row is keyed by `player_id`, so without it the
tick exits 2 and records nothing. A tick costs one props call per pre-kickoff
game; the credit probe is free. Still Kevin's word to give — this build has made
no request and spent no credit.

## 9. Plan deviations

- **Task 1–2 tests stub `nfl_data_py`** rather than hitting the network. `tests/conftest.py` blocks connects suite-wide — the guard exists because this exact mistake already caused an incident here.
- **Task 9's registry tests use `tmp_path`.** Requiring committed production artifacts would make a unit test depend on a multi-minute training run.
- **Tasks 4/5/8 join on real nflverse columns** (`recent_team`/`opponent_team`), as the plan instructed, rather than the `team` its sketches used.
- **`weather` is opt-in** (`--fetch-weather`): ~2.3k free calls and ten minutes of network that a plain training run should not spend silently.
- One pre-existing test, `test_passing_td_record_absence.py`, asserted `scripts/` held exactly one `.py`; Task 8's CLI made it two. It failed identically with all this branch's work stashed. The assertion now checks what the test is about.
- **Tasks 1–3, 4, 7, 12 corrected eight places** where the plan's own code contradicted its own tests or the live nflverse schema — including a `pull_ngs` signature bug its own stub had agreed with. Per-commit reasons in the history.
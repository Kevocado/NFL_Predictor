# NFL block_eval results — walk-forward, 2000–2025

**Sign convention:** every `Δ` is `base − block`, so **POSITIVE means the block is better**
(MAE/Brier are lower-is-better). `Clears` needs the MAE CI, the Brier CI **and** the
gap-difference CI all above zero.

**Method:** seasons are folds. Each fold trains on every season strictly before it and is
scored on the held-out season. Base and with-block runs use the *same* folds, so the held-out
games are identical and the tool raises if game counts diverge. `--candidate ridge`.

## Overall — all seasons 2000–2025

| Block | N | MAE Δ | MAE 95% CI | Brier Δ | Brier 95% CI | Gap base | Gap block | Gap Δ 95% CI | Clears |
|---|---|---|---|---|---|---|---|---|---|
| epa | 6499 | **−0.0506** | [−0.0928, −0.0066] | −0.0005 | [−0.0014, 0.0005] | 0.0280 | 0.0205 | [−0.0163, 0.0206] | No |
| qb | 6499 | **+0.0726** | [0.0198, 0.1256] | **+0.0034** | [0.0020, 0.0047] | 0.0280 | 0.0288 | [−0.0106, 0.0227] | No |
| conditions | 6499 | −0.0150 | [−0.0251, −0.0053] | −0.0002 | [−0.0004, −0.0000] | 0.0280 | 0.0255 | [−0.0105, 0.0079] | No |

*Conditions is claimed for **totals**, so it also gets total MAE:* **+0.0449**
[−0.0042, 0.0949] — positive is better on totals, but the CI straddles zero.

## The two QB runs, same games both sides

The QB block reads *who started*. Training can use the real starter off play-by-play, but a
serving path only has the pre-game depth chart. So the block is evaluated twice — and both runs
must cover the **same** held-out games, or the comparison is confounded by the game range.

**Depth charts only exist from 2001** (the 2000 feed 404s), so the honest comparison range is
**2001–2024**. On that range:

| QB feed | N | MAE Δ | MAE 95% CI | Brier Δ | Brier 95% CI | Gap Δ 95% CI | Clears |
|---|---|---|---|---|---|---|---|
| Actual starter (what training sees) | 5947 | **+0.0871** | [0.0337, 0.1408] | +0.0037 | [0.0024, 0.0051] | [−0.0163, 0.0246] | No |
| **Expected starter (what serving sees)** | 5947 | **+0.0185** | [−0.0093, 0.0484] | **+0.0011** | [0.0003, 0.0019] | [−0.0175, 0.0101] | No |

Same folds, same N. The MAE point estimate survives in sign but **shrinks 4.7×** and its
confidence interval drops below zero: `+0.0871 [0.0337, 0.1408]` → `+0.0185 [−0.0093, 0.0484]`. The
Brier gain survives and shrinks 3.4×. Neither clears, because the calibration gap does not narrow
on either feed. So the honest verdict on the QB block is: *a real but small Brier gain, no
demonstrated MAE gain once you feed it the starter serving could actually name.*

### What the serving run gives up — measured, not assumed

`--serving-realistic-qb` prints the coverage cost, because slicing `qb_games` to the expected
starter discards that QB's history for any game where he has no prior rows:

```
SERVING-REALISTIC QB: 6281 games carry an expected starter from the depth chart;
  6081 of them (96.8% of the named games) have that QB's history in qb_games
```

On the wider 2000–2025 range the drop-off is concentrated exactly where you would expect:

| team-game slots | count | share |
|---|---|---|
| Total in the training frame | 14034 | |
| With a depth-chart expected starter | 12528 | 89.3% |
| With **no** expected starter | 1506 | 10.7% |

Of those 1506: **518 are every 2000 game** (the feed does not exist), **570 are 2025** (season in
progress, so no completed chart resolves), and the rest are 14–32 a year — byes and teams whose
chart is missing for that week. Separately, of the games that *do* name a starter, **7.7%** name a
QB with no prior `qb_games` row, so serving has no history for him and the block sees a cold start.

So the serving-realistic number is a lower bound in two directions: it excludes games serving
genuinely cannot answer, and for 7.7% of the rest it hands the model *less* than a real serving
system would have.

## EPA — localising the damage (fold depth, not season)

EPA's overall MAE Δ is −0.0506. Splitting by calendar year looks like a season effect, but it
isn't. The efficiency input exists for **every** season 2000–2025 (534 rows / 267 games a year).
What actually tracks the damage is **how many seasons the fold trained on**:

| Era | N | MAE Δ | MAE 95% CI | Brier Δ | Brier 95% CI | Gap base | Gap block | Clears |
|---|---|---|---|---|---|---|---|---|
| 2010–2025 | 3829 | +0.0042 | [−0.0493, 0.0571] | +0.0006 | [−0.0008, 0.0019] | 0.0547 | 0.0199 | No |
| 2000–2009 | 2136 | **−0.1473** | [−0.2553, −0.0440] | −0.0020 | [−0.0043, 0.0003] | 0.0321 | 0.0334 | No |

Per-fold MAE Δ shows the mechanism. The 2002 fold (trained on **2** seasons) is **−0.62**;
2004 is −0.33; 2006–2011 sit around −0.15; and from 2012 on every fold is inside ±0.1.

| val season | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 … | 20 | 21 | 22 | 23 | 24 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δ MAE | −0.62 | +0.12 | −0.33 | +0.08 | −0.21 | −0.16 | −0.16 | +0.11 | −0.15 | −0.17 | −0.03 | +0.05 | −0.02 | −0.10 | ≈0.04 … | +0.01 | +0.07 | +0.01 | +0.01 | +0.11 | −0.00 |

The EPA features are expanding-EWM rollups, so an early fold has almost no history behind it and
~12 % of its EPA cells are still NaN (the model fills them with 0); by 2020 that is 0 %. So the
pre-2010 damage is **fold depth plus cold-start NaNs**, not "efficiency data is missing for early
seasons". Both go away once enough seasons of history exist.

## QB agreement — 2024 weeks 1–4

The actual starter is taken to be the QB who threw the most passes for that team in the game
(ties broken on carries, then rushing yards), from `player_stats.fetch_weekly_player_stats`.
The expected starter is the depth chart's `depth_team == '1'` QB per club, resolved to the newest
chart at or before that week.

| Week | Expected starters | Agreement rate | Status |
|---|---|---|---|
| 1 | 32 | **1.000** | meets target |
| 2 | 32 | 0.969 | meets target |
| 3 | 32 | 0.875 | below 90 % |
| 4 | 32 | 0.844 | below 90 % |
| **1–4 pooled** | **128** | **0.922** | |

Week 1 is 1.000 legitimately: every one of the 32 chart-starters also started, which is what a
season opener looks like (undisputed starters start, no one is injured yet). The list of 32:
ARI K.Murray, ATL K.Cousins, BAL L.Jackson, BUF J.Allen, CAR B.Young, CHI C.Williams,
CIN J.Burrow, CLE D.Watson, DAL D.Prescott, DEN B.Nix, DET J.Goff, GB J.Love, HOU C.Stroud,
IND A.Richardson, JAX T.Lawrence, KC P.Mahomes, LA M.Stafford, LAC J.Herbert, LV G.Minshew,
MIA T.Tagovailoa, MIN S.Darnold, NE J.Brissett, NO D.Carr, NYG D.Jones, NYJ A.Rodgers,
PHI J.Hurts, PIT J.Fields, SEA G.Smith, SF B.Purdy, TB B.Mayfield, TEN W.Levis, WAS J.Daniels.

Disagreements (all verified against the 2024 player-stats feed, none fabricated):

| Week | Game | Team | Expected | Actual |
|---|---|---|---|---|
| 2 | 2024_02_IND_GB | GB | J.Love | M.Willis |
| 3 | 2024_03_GB_TEN | GB | J.Love | M.Willis |
| 3 | 2024_03_MIA_SEA | MIA | T.Tagovailoa | S.Thompson |
| 3 | 2024_03_LAC_PIT | PIT | R.Wilson | J.Fields |
| 3 | 2024_03_CAR_LV | CAR | B.Young | A.Dalton |
| 4 | 2024_04_TEN_MIA | TEN | W.Levis | M.Rudolph |
| 4 | 2024_04_PIT_IND | IND | A.Richardson | J.Flacco |
| 4 | 2024_04_TEN_MIA | MIA | S.Thompson | T.Huntley |
| 4 | 2024_04_PIT_IND | PIT | R.Wilson | J.Fields |
| 4 | 2024_04_CIN_CAR | CAR | B.Young | A.Dalton |

`tests/test_qb_agreement.py::test_committed_csv_reproduces_the_reported_agreement_rate` reads
`output/qb_agreement_2024_w1-4.csv` and asserts it still yields 128 rows / 118 agree / 0.922, so
the headline number is reproducible from the committed file rather than trusted.

## DEFAULT_BLOCKS: no change, stays `()`

No block clears on all three of MAE, Brier and the paired gap interval:

- **`epa`** — MAE Δ is **−0.0506 [−0.0928, −0.0066]**, a genuine loss, and the damage is
  concentrated in the early folds.
- **`conditions`** — MAE Δ is −0.0150 and its total MAE is +0.0449 [−0.0042, 0.0949], straddling
  zero. Not demonstrated to help either metric.
- **`qb`** — the closest. Brier Δ **is** positive with the CI above zero on both feeds
  (+0.0037 [0.0024, 0.0051] on the actual starter, +0.0011 [0.0003, 0.0019] on the expected
  one). But its MAE CI drops below zero the moment you feed it the starter serving could name,
  and the calibration gap does not narrow on either feed. A one-sided gain in one metric is not
  the bar.

Flipping any of them in would mean serving a block whose MAE advantage is not demonstrated under
serving information. `DEFAULT_BLOCKS` stays `()`.

*Nothing here is flipped without the maintainer's decision — this PR reports, it does not change
`DEFAULT_BLOCKS`.*


## Reproduction

All commands run from the repo root with the project venv:

```bash
VENV="PYTHONPATH=src:.venv/lib/python3.11/site-packages .venv/bin/python"

# Block eval, all three blocks, every season available (the headline table)
$VENV -m nfl_predictor.tools.block_eval \
  --blocks epa qb conditions --start-year 2000 --end-year 2025 | tee output/block_eval_overall.txt

# The two QB runs, on the SAME range and same folds, because depth charts only exist from 2001
$VENV -m nfl_predictor.tools.block_eval \
  --blocks qb --start-year 2001 --end-year 2024 | tee output/block_eval_qb_actual_starter.txt

$VENV -m nfl_predictor.tools.block_eval \
  --blocks qb --start-year 2001 --end-year 2024 --serving-realistic-qb \
  | tee output/block_eval_qb_serving_realistic.txt

# EPA era split, to localise the damage
$VENV -m nfl_predictor.tools.block_eval \
  --blocks epa --start-year 2010 --end-year 2025 | tee output/block_eval_epa_2010plus.txt
$VENV -m nfl_predictor.tools.block_eval \
  --blocks epa --start-year 2000 --end-year 2009  | tee output/block_eval_epa_pre2010.txt

# QB agreement, plus the committed raw per-team-week CSV
$VENV -m nfl_predictor.tools.qb_agreement \
  --season 2024 --weeks 1 2 3 4 --output output/qb_agreement_2024_w1-4.csv \
  | tee output/qb_agreement_2024_w1-4.txt

# Full suite
$VENV -m pytest tests/ -q
# -> 1207 passed, 24 skipped, exit 0
```

`main()` returns a nonzero exit status when any requested block fails, so an automated caller
cannot read `0` as "this table is every block I asked for".

## Files committed as evidence

| File | What it is |
|---|---|
| `output/block_eval_overall.txt` | the three-block, 2000–2025 run |
| `output/block_eval_qb_actual_starter.txt` | QB on the actual starter, 2001–2024 |
| `output/block_eval_qb_serving_realistic.txt` | QB on the depth-chart expected starter, 2001–2024, with its coverage line |
| `output/block_eval_epa_2010plus.txt` | EPA, 2010–2025 |
| `output/block_eval_epa_pre2010.txt` | EPA, 2000–2009 |
| `output/qb_agreement_2024_w1-4.csv` | raw per-team-week expected/actual rows the 0.922 is derived from |
| `output/qb_agreement_2024_w1-4.txt` | the agreement run's own output |

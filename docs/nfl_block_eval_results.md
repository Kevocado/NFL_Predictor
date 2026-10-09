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

## The two agreement runs for QB

The QB block reads *who started*. Training can use the real starter from play-by-play, but a
serving path only has the pre-game depth chart. So the block is evaluated twice:

| QB block feed | N | MAE Δ | MAE 95% CI | Brier Δ | Brier 95% CI | Clears |
|---|---|---|---|---|---|---|
| Actual starter (play-by-play) — what training sees | 6499 | +0.0726 | [0.0198, 0.1256] | +0.0034 | [0.0020, 0.0047] | No |
| **Depth-chart expected starter — what serving sees** | 6499 | **−0.0026** | [−0.0353, 0.0289] | **+0.0008** | [0.0001, 0.0015] | No |

The headline MAE gain (+0.0726) **collapses to −0.0026 inside the noise band** once the block is
fed the starter a serving path could actually name. The Brier gain survives but shrinks ~4×
(0.0034 → 0.0008). The MAE advantage was lookahead: the block knew who really started.

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

Nothing clears on the serving-realistic feed. `epa` and `conditions` damage MAE; `conditions`
also fails on total MAE; and `qb`'s MAE advantage does not survive the switch from the actual to
the expected starter. Flipping any of them into `DEFAULT_BLOCKS` would mean serving a block whose
measured gain only exists with information the serving path does not have.

## Reproduction

```bash
PYTHONPATH=src:.venv/lib/python3.11/site-packages .venv/bin/python -m nfl_predictor.tools.block_eval \
  --blocks epa qb conditions --start-year 2000 --end-year 2025 | tee output/block_eval_overall.txt

PYTHONPATH=src:.venv/lib/python3.11/site-packages .venv/bin/python -m nfl_predictor.tools.block_eval \
  --blocks qb --start-year 2000 --end-year 2025 --serving-realistic-qb \
  | tee output/block_eval_qb_serving_realistic.txt

PYTHONPATH=src:.venv/lib/python3.11/site-packages .venv/bin/python -m nfl_predictor.tools.block_eval \
  --blocks epa --start-year 2010 --end-year 2025 | tee output/block_eval_epa_2010plus.txt

PYTHONPATH=src:.venv/lib/python3.11/site-packages .venv/bin/python -m nfl_predictor.tools.block_eval \
  --blocks epa --start-year 2000 --end-year 2009 | tee output/block_eval_epa_pre2010.txt

PYTHONPATH=src:.venv/lib/python3.11/site-packages .venv/bin/python -m nfl_predictor.tools.qb_agreement \
  --season 2024 --weeks 1 2 3 4 --output output/qb_agreement_2024_w1-4.csv \
  | tee output/qb_agreement_2024_w1-4.txt
```

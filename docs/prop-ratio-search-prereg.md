# NFL player-prop ratio-feature search: pre-registration

Status: written 2026-10-30, BEFORE any candidate ratio was scored. This file is the
frozen procedure. Changing a rule below after results exist requires a new
pre-registration, recorded as a separate dated section, not an edit in place.

## Question

Does any ratio of two lagged, trailing player or team statistics improve the
current production player-prop point model (median, q50) out of sample, beyond the
production feature set, for the markets the props route serves?

## Data and frame

- Frame: `scripts/prop_ratio_search.py` calls the existing
  `scripts/train_quantile_props.py::build_feature_frame` (unchanged) over seasons
  2017-2026, reading the existing nflverse cache (`data/cache/nflverse` in the main
  clone, read-only). No new dependencies. Python: the repo `.venv` (3.11).
- Weather is NOT fetched (the default of `build_feature_frame`), so weather columns
  are NaN, as in the default production training run.
- Team of record is `recent_team`; opponent is `opponent_team`.
- Known gap: nflverse dropped `route` from 2026 play-by-play. No route-based candidate
  (yards per route, route share) is in the pool. Snap counts are not in the cache, so
  `snap_share` (already in the frame, lagged) is used only as a pairwise base, never as
  a numerator/denominator built from raw snaps.

## Markets scored (5)

| market | positions | label column |
|---|---|---|
| passing_yards | QB | passing_yards |
| rushing_yards | RB | rushing_yards |
| receiving_yards | WR, TE | receiving_yards |
| receptions | WR, TE | receptions |
| passing_tds | QB | passing_tds |

`receptions` is a ZINB count model in production. It is scored here by the MAE of
the q50 quantile regressor (the only proxy available without refitting the ZINB per
candidate). This is a deliberate, declared deviation from "the model's own loss".
`passing_tds` is scored with the same proxy; its baseline adds `passing_tds_roll`.
QB rushing is excluded (not named in the task).

## Baseline (the current production feature set)

- Yardage markets: `QUANTILE_FEATURE_COLUMNS` (14 columns, from
  `models/training.py`), minus the market's own label column.
- `passing_tds`: the same 14 plus `passing_tds_roll` (`SERVING_FEATURE_COLUMNS`).

## Scoring model (identical for baseline and candidates)

`fit_yardage_quantile_models(X, y, quantiles=[0.5])` from `models/player_props.py`:
XGBoost `reg:quantileerror`, alpha 0.5, 150 trees, depth 3, lr 0.05, random_state 42,
`fillna(0)`. The metric is the MAE of the q50 prediction against the realised label.
This is the production point-prediction objective, so it is the production quantity.

## Candidate pool

All candidates are ratios built from rolling statistics. Every rolling statistic is
`shift(1)` within the entity (player or team), over the prior 5 games (window 5,
`min_periods=1`), and a ratio is `sum_5(numerator) / sum_5(denominator)` over the same
window. A zero or NaN denominator gives NaN, which is `fillna(0)` at fit time, as in
production. Nothing from the target game enters any feature.

Sources:
- Player statistics: frame `*_roll` columns (official weekly) for targets, receptions,
  receiving_yards, carries, rushing_yards, passing_yards, attempts, completions.
- Play-by-play (`pbp_<season>.parquet`), aggregated per player-game:
  `air_yards` (sum over targeted plays), `rz_targets` and `rz_carries` (yardline_100 <= 20).
  A game with no pbp row counts as 0 before the rolling step.
- Play-by-play, aggregated per team-game (posteam): `team_pass_att`, `team_plays`
  (pass or rush attempts), `team_air_yards`, `team_rush_att`, `team_rz_pass_att`,
  `team_rz_rush_att`. Team rolls are shifted within the team's own game sequence and
  merged onto player rows by (season, week, recent_team).
- Opponent-allowed: the frame's `opp_pass_yds_allowed_roll`, `opp_rush_yds_allowed_roll`,
  `opp_rec_yds_allowed_roll` (already lagged in `features/matchup.py`).

### Named ratios (16)

1. target_share = targets / team_pass_att
2. air_yards_share = air_yards / team_air_yards
3. yards_per_target = receiving_yards / targets
4. yards_per_touch = (receiving_yards + rushing_yards) / (targets + carries)
5. carries_share = carries / team_rush_att
6. yards_per_carry = rushing_yards / carries
7. rz_target_share = rz_targets / team_rz_pass_att
8. rz_carry_share = rz_carries / team_rz_rush_att
9. catch_rate = receptions / targets
10. team_pass_rate = team_pass_att / team_plays
11. passing_vs_opp = passing_yards / opp_pass_yds_allowed_roll
12. rushing_vs_opp = rushing_yards / opp_rush_yds_allowed_roll
13. receiving_vs_opp = receiving_yards / opp_rec_yds_allowed_roll
14. opp_rec_over_pass = opp_rec_yds_allowed_roll / opp_pass_yds_allowed_roll
15. opp_rush_over_pass = opp_rush_yds_allowed_roll / opp_pass_yds_allowed_roll
16. opp_rec_over_rush = opp_rec_yds_allowed_roll / opp_rush_yds_allowed_roll

Items 1-9 and 11-13 are defined on the player; 10 and 14-16 are team or opponent.
Items 3 and 6 duplicate production features in spirit (`yards_per_target_roll`,
`yards_per_carry_roll`, which are mean-of-ratios, not ratio-of-sums). They are kept
because the ratio-of-sums form is a different estimator.

### Pairwise ratios

Base set (16): targets, receptions, receiving_yards, carries, rushing_yards,
passing_yards, attempts, completions, air_yards, rz_targets, rz_carries, team_pass_att,
team_plays, team_air_yards, team_rush_att, snap_share.
All ordered pairs (A, B), A != B: 16 x 15 = 240. Any pair whose definition equals a
named ratio above is dropped (counted once). The pool is capped at 300 (named + pairs
<= 256, so the cap does not bind). The exact count is written to the JSON.

## Selection (SELECTION seasons only: 2019-2024)

Each validation season Y in 2019-2024 is scored by a model fit on all frame rows with
season < Y (walk-forward, the same discipline as `walk_forward_quantile`).

For each market m and candidate c:
1. Per-row paired difference d_i = |e_base,i| - |e_c,i| pooled across all selection
   folds. Test: one-sided paired t-test, H1: mean(d) > 0 (candidate reduces MAE).
2. Holm step-down adjustment across the full family: every (market, candidate) test,
   i.e. 5 x |pool| tests.
3. Stability: mean MAE improvement is positive in at least 5 of the 6 selection
   seasons.
4. A (m, c) pair PASSES selection if Holm-adjusted p < 0.05 AND stability holds.
5. Per market, keep at most 3 passing candidates, ranked by Holm p (ties by name).
   The kept set S is frozen at this point.

If S is empty, the search is a negative result and stops here, with no holdout look.

## Holdout (HOLDOUT seasons: 2025 and 2026, looked at ONCE)

Holdout is a look that happens only after S is written to the selection JSON, and the
script refuses to rerun the holdout if its JSON already exists.

- Each (m, c) in S is refit on all rows with season < holdout season and scored on
  2025 and 2026 pooled (2026 is a partial season, used only as far as the cache goes).
- Effect = MAE_base - MAE_c on holdout rows. Interval: cluster bootstrap by player_id,
  B = 2000, seed 20261030, percentile interval at Bonferroni level alpha/(2k), where
  k = |S| and alpha = 0.05 (two-sided). Pass the (m, c) candidate if the lower
  bound > 0.
- "No other market worse": for each passing (m, c), each of the other 4 markets, with
  feature c added to that market's baseline, has holdout point effect >= 0.

A candidate that passes both is a PASS. The branch then adds the feature behind a flag
that defaults OFF. If no candidate passes, the work is a docs-only negative result.

## Caveat, stated before results

2025 was already part of the offline production gate (`--validate 2019-2025`), so
2025 is not pristine for the baseline features. The only season never used in an
offline gate is 2026, which is partial. The holdout is therefore 2025 + 2026 and the
report says which season the result rests on.

## Outputs

- `docs/prop-ratio-search-selection.json`: every candidate, every market, per-season
  effects, p-values, Holm p, stability, selected flag, pool count.
- `docs/prop-ratio-search-holdout.json`: holdout effects, intervals, cross-market check.
- `docs/prop-ratio-search.md`: a short table of the selected candidates and the verdict.

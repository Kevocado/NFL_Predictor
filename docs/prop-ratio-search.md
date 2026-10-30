# Prop ratio-feature search: result

Procedure: `docs/prop-ratio-search-prereg.md` (frozen 2026-10-30, before scoring).
Scripts: `scripts/prop_ratio_search.py` (`select`, then `holdout`). Raw output:
`docs/prop-ratio-search-selection.json` (all 249 candidates x 5 markets, 1,245 tests)
and `docs/prop-ratio-search-holdout.json` (kept set only).

## Verdict: NEGATIVE (docs-only, no feature flag)

Selection kept 3 candidates, all for `passing_tds`. On the holdout (2025 + 2026) none
passes, so no feature is added to serving.

## Selection (walk-forward 2019-2024, Holm over 1,245 tests)

| market | candidates passing Holm<0.05 + stability | best candidate | best Holm p | mean MAE gain |
|---|---|---|---|---|
| passing_yards | 0 | air_yards_share | 1.00 | -0.015 |
| rushing_yards | 0 | air_yards_share | 1.00 | -0.000 |
| receiving_yards | 0 | snap_share / receiving_yards | 0.067 | +0.032 |
| receptions | 0 | rz_carries / passing_yards | 0.098 | +0.000 |
| passing_tds | 3 (kept) | opp_rush_over_pass | 0.0002 | +0.028 |

Kept for holdout (all five selection seasons positive, Holm p < 0.002):
- `named__opp_rush_over_pass` (opp rush-yards allowed / opp pass-yards allowed)
- `pair__team_air_yards__over__team_rush_att`
- `pair__team_air_yards__over__team_plays`

## Holdout (2025 + 2026, cluster bootstrap by player, Bonferroni alpha/3)

| candidate | holdout n | MAE gain | CI (99.2%) | other markets not worse | passes |
|---|---|---|---|---|---|
| opp_rush_over_pass | 844 | 0.000 | [0.000, 0.000] | no | no |
| team_air_yards / team_rush_att | 844 | 0.000 | [0.000, 0.000] | no | no |
| team_air_yards / team_plays | 844 | 0.000 | [0.000, 0.000] | no | no |

Caveat that matters for reading this: the zero is structural, not a measured null. The
production q50 model is a median regressor on integer pass-TD counts. In the 2025 fit the
candidate does get split on (150 splits), but the leaf medians never change, so the
predictions match the baseline on every holdout row. The selection-phase gains came from
earlier folds (2019, 2023) where predictions did move. The holdout therefore cannot
show a positive effect for these candidates, and it cannot show a negative one either.
The "other markets not worse" check also fails on other markets, so the candidates do not
clear that rule even if the zero were replaced by a real effect.

## Limits

- 2025 was already in the offline production gate, so it is not pristine.
- Receptions is scored by q50 MAE, not the ZINB's own loss (declared in the prereg).
- No route-based candidates (nflverse dropped `route` from 2026 pbp). Weather not fetched.
- Ratios are ratios of 5-game means per entity; team/player denominators come from
  different sources (official weekly for player stats, play-by-play for team and air-yards).

# NFL Predictor: Design Spec

**Date:** 2026-09-04
**Status:** approved, pending implementation plan

## Purpose

A self-hosted NFL prediction dashboard: pre-kickoff win/spread/total
probabilities for every game on the weekly slate, plus player-level
touchdown and yardage predictions for skill-position players — served
through a FastAPI backend and a React dashboard, with every prediction
tracked against what actually happens. Third sibling project alongside
`PL_Predictor` (Premier League) and `F1_Predictor` (Formula 1), reusing
their proven infrastructure patterns (cache-or-fetch data modules,
walk-forward validation with candidate models raced against each other,
snapshot-then-reconcile tracking, Shin-de-vig value-bet comparison)
while building NFL-shaped data, features, and models from scratch.

The 2026 NFL regular season opens within days of this spec being
written, which sets the scope: game outcomes and player props both ship
for week 1; a live in-game win-probability engine (the F1_Predictor
equivalent) is explicitly deferred to a later iteration.

## Why NFL doesn't reuse PL's or F1's model shape

- **PL_Predictor** models goals as a low-count process (Dixon-Coles,
  Bivariate-Poisson) — appropriate for a sport where a typical final
  score is 1-2 goals per side.
- **F1_Predictor** models a field-relative rank/probability via
  Plackett-Luce over ~20 drivers per race — there is no "opponent," just
  a field.
- **NFL** is a two-team, high-count, bursty-scoring game (touchdowns +
  field goals, typical totals 40-50 points) where the standard modeling
  approach in the wider NFL analytics community is a power-rating /
  Elo-style team strength estimate feeding a margin-of-victory
  distribution, not a goal-count model. This spec follows that
  convention rather than forcing PL's or F1's approach onto a
  differently-shaped sport.

## Scope (v1)

**In scope:**
- Game-level predictions: moneyline win probability, point-spread cover
  probability, over/under total-points probability, for every regular-
  season (and, once reached, postseason) game.
- Player props: anytime-touchdown-scorer probability, QB passing
  yards/TD probability, RB rushing yards/TD probability, WR/TE receiving
  yards/TD probability.
- Value-bet detection: model probabilities vs. The Odds API lines,
  de-vigged, surfaced the same way PL_Predictor does (at most one
  qualified recommendation per market, never a parlay).
- Honest tracking: every prediction snapshotted before kickoff, in a
  local SQLite store, reconciled against results as they land.
- React frontend: weekly-slate view, player props view, track record.
- Docker → Render deploy, plus a new card on the existing
  `predictor-hub/index.html`.

**Explicitly out of scope for v1:**
- Live in-game win-probability engine (F1_Predictor's live engine is
  the model for this; NFL would need play-by-play live state — deferred
  given the one-week runway).
- Playoff-seeding / championship Monte Carlo projection (F1_Predictor's
  championship-projection equivalent) — worth adding once the core
  game/player models are proven, not before.
- Any prop market beyond the four listed (e.g. receptions, interceptions,
  kicker props).

## Project layout

New sibling repo at `Documents/Projects/NFL_Predictor` (top-level,
alongside `F1_Predictor` — not nested inside `Prem_Predictor`, whose
name is Premier-League-specific). Own git repo, own Python venv.

```
src/nfl_predictor/
  config.py                 paths, env loading, shared constants
  data/                      nfl_data_py.py, espn.py, odds_api.py
  features/                  build.py, power_ratings.py, rolling_form.py,
                             rest_days.py, weather.py, injuries.py,
                             player_usage.py, matchup.py
  models/                    game_outcome.py (candidates: elo, margin_regression,
                             xgb), player_props.py (anytime_td classifier,
                             yardage regressors), manifest.py
  evaluate/                  walk_forward.py, backtest.py
  tracking/                  store.py, value_bet_ledger.py
  odds/                      value_bets.py
  api/                       main.py, routes.py, schemas.py
frontend/src/
  api/                       client.ts, types.ts
  components/                (game prediction cards, player prop tables,
                             probability heat cells — reusing PL/F1 visual
                             conventions where the data shape matches)
  pages/                     GamesPage, PlayerPropsPage, TrackRecordPage
tests/
docs/
  AI_CONTINUITY.md           maintained handoff doc, same role as PL_Predictor's
  superpowers/specs/         this file
  superpowers/plans/         implementation plan(s)
Dockerfile
pyproject.toml
```

## Data layer

- **Primary source: `nfl_data_py`** (nflverse) — free, keyless, actively
  maintained. Provides schedules, play-by-play back to 1999, team and
  player weekly stats, rosters, injury reports, and historical Vegas
  lines. Cached to `data/cache/` after first fetch, same pattern as the
  other two projects' cache-or-fetch data modules.
- **ESPN's unofficial API** — supplements with closer-to-kickoff
  inactive/injury status, the same role it plays in `PL_Predictor` for
  confirmed lineups.
- **The Odds API** — same account/key already used by `PL_Predictor`.
  Covers NFL `h2h`/`spreads`/`totals` on the existing plan. Whether the
  current tier also carries player-prop markets needs to be checked
  directly against the API during implementation (a data-availability
  check, not a design blocker); if player-prop odds aren't available on
  the current plan, player props ship as model-only probabilities
  without a value-bet comparison for v1, and game-level value bets are
  unaffected either way.
- No new API keys are required to reach a working v1 beyond confirming
  the existing Odds API key's plan coverage.

## Features

**Team-level:** offense/defense power ratings (updated per week), rolling
scoring/yardage/EPA form, rest days since last game (bye weeks are a
bigger swing here than PL's midweek-fixture congestion), home-field
indicator, divisional-game flag, weather for outdoor stadiums,
injury-adjusted starter availability.

**Player-level:** usage share (targets for receivers, carries for
backs), rolling yards-per-game, opponent-defense matchup adjustment
(e.g. yards allowed to the position), red-zone role/share.

All consumers (training, backtest, API, frontend data prep) go through
one `features/build.py` entry point — the same "single feature-
construction entry point" discipline documented as a reused pattern from
FPL_Optimizer/PL_Predictor.

## Models

**Game outcome (moneyline/spread/total):** built as a candidate race,
not a prescribed single architecture — matching how `PL_Predictor` races
Dixon-Coles/Bivariate-Poisson/XGBoost and `F1_Predictor` races
Elo/XGBoost-ranker. Candidates:
1. Elo / power-rating baseline (interpretable, fast-adapting to recent
   form).
2. Ridge or linear margin-of-victory regression on team power ratings +
   situational features.
3. XGBoost regression on the full feature set.

Each candidate is walk-forward validated (train on earlier seasons,
validate on held-out later ones) on a probability-scoring metric
(log-loss, matching the other two projects' selection discipline).
Whichever wins is served; `manifest.py` records which candidate was
selected and why, same as `PL_Predictor`'s `manifest.json` and
`F1_Predictor`'s `manifest.json`. Win probability, spread-cover
probability, and total probability should derive from one coherent
predicted-margin/predicted-total distribution per game, not three
independently-fit numbers that could contradict each other — same
"one coherent simulation" principle F1_Predictor applies to its
Plackett-Luce field simulation.

**Player props:**
- Anytime-TD: binary classifier (mirrors PL_Predictor's goal/assist
  classifier approach) — probability per skill player per game.
- Passing/rushing/receiving yards: regression models (likely XGBoost,
  raced against a simpler baseline the same walk-forward way) producing
  a point estimate plus enough distributional information to answer an
  over/under line question against Odds API player-prop lines when
  available.

## Tracking and value bets

Same immutable snapshot-before-kickoff → reconcile-after-result SQLite
pattern as `PL_Predictor`'s `tracking/store.py`: every game and player
prop prediction is written to the store before kickoff and never
overwritten; results are reconciled in as they land (final scores from
`nfl_data_py`/ESPN). Value-bet detection reuses the Shin de-vig
comparison from `PL_Predictor`'s `odds/value_bets.py` near as-is — model
probability vs. de-vigged market probability, surfacing at most one
qualified single per market, never a parlay.

## API

FastAPI app mirroring the route shape of `PL_Predictor`/`F1_Predictor`:
- `GET /api/games?season=2026&week=1` — weekly slate
- `GET /api/games/{season}/{week}/{game_id}/prediction` — win/spread/
  total probability table for a game
- `GET /api/players/{season}/{week}/props` — player prop predictions for
  the week
- `GET /api/track-record` — honest calibration summary
- `POST /api/retrain` — retrain all models on latest data

## Frontend

React 19 + TypeScript + Vite, following F1_Predictor's tab pattern
(weekly-slate cadence, like F1's race-weekend cadence) rather than
PL_Predictor's continuous-fixture-list-plus-modal pattern:
- **Games** — week-by-week slate, each game showing win/spread/total
  probabilities and value-bet flags.
- **Player Props** — searchable/sortable table of TD and yardage
  predictions for the week.
- **Track Record** — honest calibration, matching the other two
  projects' "judged honestly, not taken on faith" framing.

## Deploy and hub integration

Dockerfile → Render, same as both existing projects. Once live, add a
third card to `Documents/Projects/predictor-hub/index.html` (🏈 icon,
"Live" badge) alongside the existing PL and F1 cards — no structural
change to the hub itself, it's already built to host exactly this.

## Testing

Pytest suite mirroring the other two projects' coverage shape: data
module tests (cache-or-fetch correctness), feature tests (no
lookahead/leakage — historical values must only use prior games), model
tests (walk-forward harness correctness), API tests (route contracts,
auth if applicable), tracking tests (snapshot immutability, reconciliation
correctness).

## Open items to confirm during implementation

- Whether the existing Odds API plan includes NFL player-prop markets
  (affects whether player props get a value-bet comparison in v1, per
  the Data layer section above).
- Exact walk-forward split boundaries for NFL given nflverse's play-by-
  play depth (1999-present) vs. how much is actually useful for current
  team-strength estimation — a research/tuning question, not a blocking
  design decision.

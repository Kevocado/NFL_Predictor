# Step 7a (NFL half): Kalshi pre-game feed — evidence report

- Plan: `docs/superpowers/plans/2026-09-25-predictor-pregame-feed.md` (step 7a, Tasks 1–4)
- Branch: `plan/2026-09-25-kalshi-feed`
- Base: `origin/main` at `4f97944` (post-merge of PRs #1 and #2; the local `main` checkout is still
  at `3c16861` with an unrelated uncommitted `deploy-azure-nfl.yml` edit, which this branch does not
  touch)
- CFB half of the same step, in its own repo and branch of the same name: see the CFB report.

## Baseline

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src \
  /Users/sigey/Documents/Projects.nosync/NFL_Predictor/.venv/bin/python -m pytest -q
156 passed
```

The plan's stated baseline (111 tests at `3c16861`) predates the merges of PRs #1 and #2. The code
anchors the plan's replace-blocks against still matched, so only the counts moved.

## Task 1 — `9c5fb5a`: store the snapshot distribution, add the feed + calibration readers

### RED

```text
# re-captured against the parent commit's src/ (see "How the RED evidence was captured")
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_kalshi_feed_store.py
9 failed, 1 passed in 0.79s
```

### GREEN

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_kalshi_feed_store.py
10 passed in 0.43s
```

- `_connect()` adds the distribution columns through `ALTER TABLE ... ADD COLUMN` (so a database
  written by an earlier container is upgraded in place, no separate migration file):
  `predicted_margin`, `sigma`, `predicted_total`, `total_sigma`, `model_version`.
- `get_feed_predictions()` serves only rows still ahead of kickoff, as
  `{game_id, season, week, home, away, start_utc, p_home, margin_mu, sigma, total_mu, total_sigma,
  home_spread_line, total_line, model_version, snapshotted_at, backfilled}`.
- `get_calibration()` buckets graded pre-game snapshots into 10 equal-width bins over
  `winner` / `spread` / `total` and reports `{lo, hi, n, mean_prob, hit_rate}`, plus `n_buckets`.
- **Deviation — no `backfilled` column, no legacy classification pass.** The plan added a
  `backfilled INTEGER` column plus a one-time pass over existing rows. PR #1 (merged) already
  shipped the better version of that idea as `_snapshotted_after_kickoff(snapshotted_at,
  commence_time)`: pre-game-ness derived live from two columns that always existed, failing closed
  on an unparseable row. Reusing it means the flag cannot drift out of step with the timestamps and
  a database written before this change needs no migration pass at all.
- **Deviation — `n_backfilled` not added.** `/api/track-record` already reports `n_rebuilt`,
  computed from the same predicate and excluding those rows before counting. A second counter over
  the same rows would be a second thing to keep in step.
- The feed still emits `"backfilled": false`: for a row that passed the filter that is the honest
  statement, and `tradehub/sports/feed.py::parse_feed` rejects a truthy value.

## Task 2 — `ee1d3ef`: freeze each snapshot inside a 48 h lead window, record `model_version`

### RED

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_snapshot_window.py
9 failed in 2.03s
```

### GREEN

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_snapshot_window.py
9 passed in 1.58s
```

- `SNAPSHOT_LEAD_HOURS = float(os.getenv("SNAPSHOT_LEAD_HOURS", "48"))`.
- `_games_to_snapshot(season, week, now, lead_hours=None)` concatenates this week's and next week's
  `fetch_upcoming_games` and keeps rows whose kickoff is within the window. `current_season_and_week()`
  rolls over on the UTC date of week 1's first kickoff, so "this week" alone left Thursday-night
  games unsnapshotted until after kickoff (where `record_game_predictions` rejects them, so they were
  never tracked) and froze next week's games on the previous Saturday, before that day's results.
- `model_version()` in `models/manifest.py` is `f"{chosen_candidate}@{trained_at}"`, surfaced through
  `load_models()` and copied into every snapshot by `_predict_game_from_models`.
- **Deviation — each snapshot row carries its own `week`.** The tick now snapshots two weeks, so
  labelling every row with the tick's `week` would put next week's games in the wrong week
  everywhere downstream. `int(game["week"]) if pd.notna(game.get("week")) else week`.
- **Deviation — the model load moved inside `if not games.empty`.** With an empty window there is
  nothing to predict, and the old unconditional load logged a warning every tick when models were
  absent.
- **Stricter than the plan (2 extra tests):** a `NaT` kickoff is dropped rather than snapshotted at
  an unknown distance from kickoff, and the lead window is asserted to be configurable and to default
  to 48 h. One test pins that the window filters on the *upper* bound only, so already-played games
  still reach `record_game_predictions` (which rejects them itself).

## Task 3 — `833093f`: `GET /api/kalshi-feed`

### RED

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_kalshi_feed_api.py
3 failed, 1 warning in 1.51s     # 404: the route does not exist yet
```

### GREEN

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_kalshi_feed_api.py
3 passed, 1 warning in 1.45s
```

- `{"sport", "generated_at", "lead_hours", "games", "calibration"}` — matches
  `tradehub/sports/feed.py::parse_feed` exactly, ISO 8601 with `+00:00`.
- One test monkeypatches `schedules.fetch_upcoming_games`, `_load_models_cached` and `requests` to
  raise, then asserts the feed still returns 200: the feed is served from the tracking database and
  never recomputes. A live forecast is a *different* number from the frozen snapshot the hub graded,
  so recomputing would swap the series out from under the calibration check with no error anywhere.

## Task 4 — `a4d9981`: NaN lines no longer produce NaN probabilities or a `/batch` 500

Found while running the live app in Task 3's smoke, not in the plan.

### RED

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_missing_lines.py
3 failed, 1 warning in 1.79s
```

### GREEN

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q tests/test_missing_lines.py
3 passed, 1 warning in 1.40s
```

- Symptom: `/api/predictions/2026/4/batch` returned **500** in the running app. Snapshots written
  before missing lines were handled contain `NaN`, and Starlette refuses to serialize `NaN`.
- Two causes, both fixed:
  1. nflverse writes a missing spread/total as `NaN`, not `None`, and `_lines_for_game`'s
     `is not None` test let it through, so `margin_to_probabilities` returned NaN cover/over
     probabilities for a game with no line. `game_outcome._line_or_none()` now maps `None` and `NaN`
     to `None`.
  2. Snapshots already in the public snapshot file still carry `NaN`. `routes._json_safe()` now maps
     non-finite floats to `None` recursively at the serving edge, which also covers any other
     non-finite value a stale snapshot might carry.
- Deviation: none — this task is an addition to the plan, made because the plan's own smoke step
  surfaces it.

## Full suite

```text
SUPABASE_SERVICE_ROLE_KEY=dummy-baseline-placeholder PYTHONPATH=$PWD/src ... -m pytest -q
181 passed, 11 warnings in 12.55s      # 156 baseline + 25 new
```

## Lint

```text
uvx ruff@latest check --isolated --select E4,E7,E9,F,I <changed files>
```

Identical findings before and after — three pre-existing ones in `src/nfl_predictor/api/routes.py`
(`I001` import-block order, `F401` unused `..odds.value_bets` import, `F841` unused
`except ... as e` local), all present at `origin/main` and left alone because this branch must not
mix a lint sweep into a behaviour change. No finding in any file or test added by this branch.

## Local smoke (Task 8 step 1)

```text
PUBLIC_MODE=false PYTHONPATH=$PWD/src ... /tmp/smoke_feed.py nfl_predictor
2026 3 feed games: 14 lead: 48.0
[
 {
  "game_id": "2026_03_CAR_CLE",
  "season": 2026,
  "week": 3,
  "home": "CLE",
  "away": "CAR",
  "start_utc": "2026-09-27T17:00:00+00:00",
  "p_home": 0.5322389350707555,
  "margin_mu": 1.0697423219680786,
  "sigma": 13.223153618916728,
  "total_mu": 37.598915100097656,
  "total_sigma": 12.486688413234214,
  "home_spread_line": -2.5,
  "total_line": 42.5,
  "model_version": "ridge@2026-09-04T22:12:49.750941+00:00",
  "snapshotted_at": "2026-09-26T21:59:51.415886+00:00",
  "backfilled": false
 }
]
null sigma: 0 null model_version: 0 not pregame: 0
calibration n: {'winner': 0, 'spread': 0, 'total': 0}
```

Against a fresh worktree-local `data/tracking.db` (gitignored, and the main checkout's
`data/tracking.db` was not touched). `snapshotted_at < start_utc` for all 14 rows. Calibration `n`
is 0 only because the database is empty: buckets are built from *graded* snapshots, and nothing has
finished yet in this fresh database.

## How the RED evidence was captured

Tasks 1–4 were committed in sequence, so the RED output above was re-captured after the fact by
restoring only the task's **source** files from its parent commit and running the task's new test
file against them, then restoring (`git checkout <commit>^ -- src/...` → run → `git checkout HEAD --
src/...`). No test file was modified and the worktree was verified clean afterwards. Tasks 2–4's RED
counts match what was observed at the time of writing.

## Kevin's checklist (not the agent's)

1. Review and merge this PR. Nothing here is merged by the agent.
2. Deploy through the repo workflow's `vps` job to the VPS (`nfl.<domain>`). The first container
   start runs `_connect()`, which `ALTER TABLE`s the new distribution columns into the **persistent**
   tracking database — verify with `/api/track-record` returning normally afterwards.
3. `curl -s https://<nfl host>/api/kalshi-feed | head -c 400` must return JSON with `"sport": "nfl"`
   and `"lead_hours": 48`.
4. The next GitHub snapshot refresh rewrites `public_snapshot.json` without NaN. Until then Task 4's
   serving guard covers it: `/api/predictions/2026/4/batch` should return 200, not 500.
5. `SNAPSHOT_LEAD_HOURS` is optional; set it in the container environment only if 48 h is wrong for
   NFL. The value the feed reports in `lead_hours` is the one the snapshot window actually used.
6. Leave the tracker running at least 48 h before the hub's first live sports run (hub plan step 7b),
   so the feed has games with `snapshotted_at < start_utc` to serve.

# Player-props empty state — fix report

The reader-visible symptom: "No player projection props available for this specific
game yet." shown for weeks that do have a full slate. Original diagnosis is in the
commit message for `4dcf774`; this file records the fix rounds.

## Fix round 1

### Accounting

Every finding from the review, with the evidence. Nothing here is claimed on the
strength of a green suite; each row names the assertion that would fail if the fix
were reverted.

| # | Finding | State | Evidence |
|---|---|---|---|
| **C1** | `facts.py` second consumer of the same route | **DONE** | `src/nfl_predictor/api/facts.py:254` (`except HTTPException`), `:239` (calls `routes.snapshot_props_unavailable`), `:508-510` (short-circuits started games), `:529` (`"players_unavailable"`). Tests: `test_facts.py::test_a_raising_props_route_does_not_take_the_bundle_down`, `test_a_snapshot_week_with_games_but_no_props_flags_the_panel_instead_of_claiming_no_players`, `test_a_recorded_unavailable_status_is_read_by_the_facts_panel`, `test_a_finished_game_never_claims_its_props_are_unavailable`; plus four in `test_player_props_failure.py`. Mutation **M19** CAUGHT, output below. |
| **C2** | 9 live nflverse connections from the new tests | **DONE** | `tests/test_player_props_failure.py:112` stubs `_load_player_history` in the `data_seams` fixture; `tests/test_api_routes.py:157` stubs the roster for the pre-existing 10th. Permanent guard: `tests/conftest.py` (installed at import, `:74`), self-tested by `tests/test_offline_guard.py` (6 tests). Measured: **0** blocked connections. |
| **I3** | `mutation_check.py` has no canary; exit code is the verdict | **DONE** | Baseline gate `:473`; frontend token `FRONTEND_SUCCESS` `:74`; `HARNESS BROKEN` `:534`; 5 canaries (C1–C4, M17). Python arm given the same token discipline — see "found and fixed" below. Proofs below. |
| **I4** | `player_props_status` written, never read; `stale` served as fresh | **DONE** — served with a marker. `snapshot_props_unavailable` (`routes.py:456`) derives from the reason; `stale` → `200` + `X-Player-Props-Stale: true` (`:508`); `unavailable` → 503. The `games` check is retained last, as the pre-key fallback. Frontend renders it (`PlayerPropsPage.tsx:52`). Mutations **M16/M17/M18** all covered. |
| **I5** | 503 detail self-contradictory | **DONE** | `routes.py:606` `skipped.append`, `:631` the guard, `:636`/`:642` the two truthful wordings. Actual log line for the skipped case is in "captured output" below. Test: `test_the_failure_message_does_not_claim_every_player_failed_when_none_did`. Mutation **M23** CAUGHT. |
| **I6** | three comments state things the code does not do | **DONE** | The `mutation_check.py` "tsc runs alongside" claim is gone (grep: absent). `tests.yml:82` now says "the error branch (M9) or the loading branch (M11)"; `props_state_check.mjs:21` already read M9 and M11. Verified by grep: no remaining M-reference misattributes M11 to the error branch. |
| **I7** | `TimeoutError` mechanism claim is wrong | **DONE** — claim corrected, tests kept. `test_player_props_failure.py:19` now states the traced path (`pandas/io/common.py` → `urllib.request.urlopen`, **no timeout kwarg**, so a stalled socket blocks rather than raises), names the failures this path really produces, and says explicitly that the exception type in each test is a stand-in and that no assertion depends on it. |
| **I8** | harness mutates the tracked file in place | **DONE** | `tests/mutation_check.py:13`. The tree is copied to a tempdir; `.venv` and `frontend/node_modules` are symlinked; tracked files are never opened for writing. `atexit` + SIGINT/SIGTERM use `os._exit` after cleanup. Proof below. |
| **m9** | `and not results and failed:` survived | **DONE** | Test `test_the_failure_message_does_not_claim_every_player_failed_when_none_did` builds exactly the shape it survives: history rows for a team not in the week, roster fallback supplying the names, so `failed` is empty and `skipped` is full. Mutation **M15** CAUGHT. |
| **m10** | `except Exception` backstop survives deletion, is dead | **DELETED, and the contract pinned instead** | The route now has only `except PlayerPropsUnavailable` (`routes.py:512`). `_get_player_props_live` wraps its whole body, so any other exception has already become a `PlayerPropsUnavailable`; a second net could only be unreachable code. What makes that safe is asserted by `test_the_live_path_never_leaks_a_raw_exception`, which raises a bare `ValueError` from deep in the call and requires a `PlayerPropsUnavailable` carrying the cause. Mutation **M24** (delete the wrapper) CAUGHT. |
| **m12** | raw `str(exc)` leaks `github.com` to a reader | **DONE** | `PROPS_UNAVAILABLE_DETAIL` (`routes.py:444`) is fixed text; the cause goes to `logger.error`. Test `test_the_public_503_body_does_not_leak_the_upstream_cause` asserts four specific strings absent from the body and *present in the log*. Mutation **M22** CAUGHT. |
| **m13** | zero-row grid renders beside "Loading…" | **DONE** | `PlayerPropsPage.tsx:59` — the table and its sort control are inside `!error && !loading`. Assertions: "a pending fetch does not render an empty table skeleton". Mutation **M10** CAUGHT. |
| **m14** | "for this specific game" on a week-keyed route | **DONE** | `PlayerPropsPage.tsx:57` — "for this week yet". The reader is on a week page; the old copy named a unit the page does not have. |
| **m15** | test named for a "model backend" that is in-process | **DONE** | `test_player_props_failure.py:252` — `test_a_prediction_error_on_every_player_is_not_silently_dropped`, and the injected fault is `ValueError("xgboost model file is corrupt")`, which is what an in-process model failure actually looks like. |
| **m16** | jsdom needs Node `^20.19.0`; CI pins `"20"` | **DONE — both** | `tests.yml:71` pins `20.19`, and `frontend/package.json` gains an `engines` block repeating jsdom's own range. Pinning alone is a comment; the `engines` field is what a contributor's `npm install` reads. |
| **m17** | `props_state_check.mjs` outside `tsconfig.app.json`'s include | **DONE by stating what actually covers it** | `props_state_check.mjs:32` records that it is linted, not typechecked. I tried `tsconfig.check.json` with `allowJs`+`checkJs` first: 11 errors, all of them "implicitly has an `any` type" on parameters, plus `@types/jsdom` missing — that is a JSDoc-types project, not a bug fix. Proved the coverage is real rather than assumed by injecting an unused binding into the file and watching oxlint warn, then restoring it byte-identical. |
| **m18** | `pyproject.toml` claims bare `ruff check` matches CI | **DONE — the claim, not the findings** | `pyproject.toml:47` now says the opposite of what it used to: CI runs `ruff check src tests --select E9,F63,F7,F82` and bare `ruff check` reports ~85 pre-existing style findings in files this work never touched. Adopting the full default set is unrelated cleanup that would bury a bug fix, so the claim is what changed. |

### The `test_facts.py` question, answered

**Yes — there is now a test that fails if `facts.py` stops catching `HTTPException`,
and it does not rely on a stub returning `[]`.**

The old `live` fixture stubbed `facts_mod.routes.get_player_props` with
`lambda season, week: []`, which made the failure unfailable. The two new tests
*override* that stub with something that raises:

- `tests/test_facts.py::test_a_raising_props_route_does_not_take_the_bundle_down`
  (the full HTTP bundle, `PUBLIC_MODE` off) and
- `tests/test_player_props_failure.py::test_a_raising_props_route_does_not_take_the_facts_bundle_down_with_it`
  (the accessor directly, plus a 404 that must *not* be absorbed).

The mutation, and its output:

```
### explicit path pair
  source : src/nfl_predictor/api/facts.py
  copy   : /tmp/opencode/m19b-H6Fx/src/nfl_predictor/api/facts.py
  target : pytest tests/ (in the copy)
M19 applied to the copy
```

```diff
-    try:
-        return routes.get_player_props(season, week), False
-    except HTTPException as exc:
-        if exc.status_code != 503:
-            raise
-        logger.warning(...)
-        return [], True
+    return routes.get_player_props(season, week), False
```

```
=========================== short test summary info ============================
FAILED tests/test_facts.py::test_a_raising_props_route_does_not_take_the_bundle_down
FAILED tests/test_player_props_failure.py::test_a_raising_props_route_does_not_take_the_facts_bundle_down_with_it
2 failed, 55 passed, 20 warnings in 2.34s
```

Restore confirmed by reading the file back — the tracked file is never written, and
its md5 was identical before and after the run.

### Proofs for I3 and I8

**Canary, detector neutered.** `frontend/props_state_check.mjs` `assert(cond, …)`
edited to `assert(true, …)`, harness run, file restored and compared:

```
before md5: 1753bab907bca89f479b513a9fcfb5ed
SURVIVED  M9  frontend error branch deleted
SURVIVED  M10 frontend table shown beside the error
SURVIVED  M11 frontend loading branch deleted
SURVIVED  M12 frontend .catch deleted
SURVIVED  M25 frontend never renders the stale notice
SURVIVED  M26 stale week also claims the empty state
6 mutation(s) SURVIVED:
  ...
0 caught, 6 survived, 0 inconclusive, 2 canary (all correct: True)
restore: byte-identical
after md5 : 1753bab907bca89f479b513a9fcfb5ed
```

Note **M12 now reads SURVIVED, not CAUGHT** — that was the review's exact complaint
(deleting `.catch` used to crash node, and the crash scored as a catch). The runner
now installs an `unhandledRejection` handler that records the rejection *through
`assert()`*, so the mutation is caught by an assertion that looked at the markup.

**Canary, detector cries wolf.** A throwaway driver (no tracked file touched)
replaces the harness's `run_frontend` in memory with a detector that is honest on
the baseline and then reports a failure for everything:

```
SURVIVED  CANARY C1: CANARY: a comment edit ...
SURVIVED  CANARY C2: CANARY: a docstring edit ...
CAUGHT    CANARY C3: CANARY: a frontend JSX label edit ...
CAUGHT    CANARY C4: CANARY: a frontend comment edit ...

HARNESS BROKEN: a canary mutation did not report SURVIVED, so the
detector itself is not working and every other verdict here is void.
  C3 reported CAUGHT, expected SURVIVED
  C4 reported CAUGHT, expected SURVIVED

2 caught, 0 survived, 0 inconclusive, 4 canary (all correct: False)

### harness exit code: 1   (non-zero = it refused to certify a broken detector)
```

**Interrupt mid-run (I8).** `SIGTERM` to the harness while it is mid-mutation:

```
before: a94a28cd8878e1711563354028b64df0
--- sending SIGTERM to pid 56510 (the harness is mid-mutation) ---
harness exit: 130
interrupted; the tracked tree was never written to
after : a94a28cd8878e1711563354028b64df0
UNMODIFIED
--- leftover temp trees in $TMPDIR ---
(none)
```

### Captured output for I5

The skipped-only path, which used to read "every player … failed to predict (0 of 1
failed)":

```
LOG-ONLY detail:
  no props for season 2026 week 3: all 1 player(s) were skipped for want of a
  pregame usage history in the season being predicted, so none could be scored
```

### Offline guard

The guard blocks *connections*, not socket construction, and both directions are
tested (`test_offline_guard.py`). The reason is specific: a mocking transport —
respx and its peers — hands the code under test a real `socket.socket` instance, so
a guard that raised in `socket.__init__` would break every such test while still
looking like it was working.

Attempts are also *recorded*, not just raised. The raise alone is not enough: the
props pipeline wraps its body in `except Exception`, so an attempt there is caught,
logged and turned into a 503 — and a test asserting "503" passes exactly as well
with a blocked connection as with the condition it claims to cover. A
session-scoped fixture fails the run if any attempt survives.

```
[offline guard] blocked connection attempts this session: 0
292 passed, 15 skipped, 268 warnings in 20.60s
```

### Mutation run

`tests/mutation_check.py`, 31 entries, one at a time, each on its own temp copy:

```
baseline (unmutated copy):
  pytest            PASS
  props_state_check PASS
CAUGHT   M1 .. M16, M18 .. M26   (25 real mutations, 0 survived)
SURVIVED CANARY M17, C1, C2, C3, C4

25 caught, 0 survived, 0 inconclusive, 5 canary (all correct: True)
```

Every mutation is a pure find/replace against an explicit `path → find → replace`
triple; `write_mutation` raises `LookupError` naming the file and the anchor it
could not find, and that is now reported as a counted `SKIPPED` rather than a
traceback that discards the verdicts already gathered.

### Found while fixing, not on the list

1. **The harness was testing nothing.** The venv is an *editable* install and its
   `.pth` puts the real `src` on `sys.path`, so mutations applied to a copy of
   `src/` were applied to a file nothing imported. The first full run reported
   **20 mutations SURVIVED** — a catastrophic-looking hole in the tests that was
   entirely an artefact. Fixed by running each target with an explicit
   `PYTHONPATH=<copy>/src` and adding `assert_copy_is_under_test()`, which refuses
   to run unless `nfl_predictor.__file__` resolves inside the copy. That check
   would have caught it on the first run.

2. **`pyproject.toml`'s `pythonpath = ["src"]` is inert.** With pytest 9.1.1 the
   ini does not put `src` on `sys.path` at all — the editable `.pth` is the only
   reason imports work, locally and in CI (`pip install -e .`). Not changed: it is
   harmless, and nothing in this bug's scope depends on it. Worth knowing before
   anyone relies on that ini.

3. **A fourth network leak, pre-existing.** `nfl_data_py.import_schedules` reads
   `http://www.habitatring.com/games.csv` — plain HTTP, a third-party personal
   site — and `nfl_data_py` calls it from the props path through
   `schedules.fetch_schedules`. It was invisible because `_build_week`'s per-game
   `except Exception` swallowed it; the offline guard exposed it. A production
   dependency on an unencrypted third-party endpoint is worth someone's attention
   independently of this PR. Not fixed here.

4. **The Python arm had the review's flaw too.** M19's first form produced a
   `SyntaxError`, and the harness scored that as CAUGHT — a collection error is not
   a test failure. `run_python` now returns `PASS`/`FAIL`/`INCONCLUSIVE` and only
   believes a failure when pytest's own summary says tests failed.

### Not fixed, and why

- **The upstream 2026 player-stat gap.** Not code-fixable, and
  `docs/player-prop-accuracy-blocker.md` owns it. What changed is that the site now
  says so instead of claiming the week has no props.
- **`data/public_snapshot.json` is unchanged.** Weeks 2–7 still hold 14–16 games
  and 0 props. The route now answers 503 for them, which is truthful, and the next
  scheduled refresh will fill them once nflverse publishes.
- **`_build_week`'s carry-forward was unrequested scope.** It is kept because the
  reviewer conditioned it on the status key being read, and it now is (I4). A
  `stale` week is served with `X-Player-Props-Stale: true` and the frontend says
  "these projections come from an earlier build" — a stale pregame projection is
  still a pregame projection, which is all the accuracy rule requires, whereas 503
  would discard exactly the rows the carry-forward exists to preserve.

### Recorded for the replacement PR (not started here)

`frontend/props_state_check.mjs` is a maintainability defect, not a correctness
one, and the agreed replacement shape is:

- extract the page's fetch/derive logic into a **pure function** returning
  `{status: "loading" | "empty" | "error" | "ready", props, error, stale}`;
- the component becomes a thin `useEffect` + render of that state;
- test the pure function with node's built-in **`node --test`** — **zero new
  dependencies**; `jsdom` survives only as long as effect wiring still needs a DOM,
  and drops once the function is tested directly;
- `node --test` then also removes the need for a bespoke runner, a `tsconfig` for a
  `.mjs`, and the Vite-SSR pipeline.

Not started in this PR, per the ruling.

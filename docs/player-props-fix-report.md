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
| **m17** | `props_state_check.mjs` outside `tsconfig.app.json`'s include | **DONE by stating what actually covers it** | `props_state_check.mjs:32` records that it is linted, not typechecked. I tried `tsconfig.check.json` with `allowJs`+`checkJs` first: 11 errors, all of them "implicitly has an `any` type" on parameters, plus `@types/jsdom` missing — that is a JSDoc-types project, not a bug fix. Coverage proved by injecting `eval()` and `debugger;` and watching oxlint report `no-eval` and `no-debugger`, then restoring the file byte-identical. **The evidence I gave in round 1 was wrong and is corrected in round 2 below** — `no-unused-vars` is not in oxlint's default set, so the unused-binding probe I cited reported 0 warnings. The conclusion held; the demonstration did not. |
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

2. **`src` reaches `sys.path` only through the editable install.** *The
   justification I gave here in round 1 was wrong* — see round 2 for the
   correction and for the `git log` evidence.

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

## Fix round 2

Two headline claims from round 1 were not true, and four comments described
behaviour the code does not have. Both claims and all four comments are fixed;
the two coverage holes the review's probes found are closed.

### Blocker 1 — the stale marker could not cross the origin boundary

Round 1 added a header, a reader and a renderer, and the renderer could never
fire. `main.py` configured CORSMiddleware with `allow_origins=["*"]` and no
`expose_headers`, and per the Fetch spec a cross-origin response exposes only the
CORS-safelisted response headers to `Headers.get()`. Reproduced before the fix:

```
status: 200
raw header present          : true
Access-Control-Allow-Origin : *
Access-Control-Expose-Headers: None
=> res.headers.get('X-Player-Props-Stale') = null
```

**Chose the CORS fix over moving the marker into the JSON body**, for the reason
the review anticipated: the CORS change is safe under either serving arrangement.
If the frontend is ever served same-origin, `expose_headers` is a harmless no-op
and the header keeps working. Moving the marker into the body would change the
response shape from a bare array to an object, which is a breaking change for
every existing reader of this endpoint, and would still leave `stale` invisible
to any consumer that ignores the new key.

`src/nfl_predictor/api/main.py` now passes `expose_headers=[PROPS_STALE_HEADER]`,
imported from `routes` so the name is written down once. After:

```
status: 200
raw header present       : true
Access-Control-Expose-Headers: X-Player-Props-Stale
=> res.headers.get('X-Player-Props-Stale') = 'true'
```

and on routes that carry no marker (`/`, `/api/snapshot-meta`) the declaration is
still present, which is what keeps the readable set from changing under a reader
who switches weeks.

**Could CORS be exercised in a test? Not in a DOM test — and it does not need to
be.** CORS is enforced by the browser, and the whole decision is made by the
server: what a cross-origin caller may read is exactly what
`Access-Control-Expose-Headers` lists. So the server response header *is* the
contract, and `tests/test_props_stale_header_cors.py` pins it at the request
level — the header is set, the header is declared readable, the middleware
configuration contains the name, a fresh week does not carry the marker, and the
503 body names the right unit. `props_state_check.mjs` replaces `fetch`
wholesale, so it can never exercise this; that is a property of it, not a gap, and
what it does check is the other two links (that the client reads the header, that
the component renders when the flag is set). A browser-level test would need a
real browser, which this project has no harness for.

**The caveat this repo cannot discharge.** If the frontend is served same-origin
behind a gateway, the whole marker path is inert: the header would be same-origin
and always readable, `expose_headers` would be doing nothing, and nothing in the
suite would notice. The evidence says cross-origin: `main.py` never mounts
`frontend/dist` (no `StaticFiles` anywhere under `src/`), the Dockerfile copies
`dist` into the image without serving it, and the container exposes only uvicorn
on 8001. So the API has no same-origin route to the bundle. **This is recorded as
`test_the_frontend_base_url_is_cross_origin_in_a_deployed_build`, which fails if
that ever stops being true, and it is a thing the deployer must confirm** — if
there is a gateway putting the bundle and the API behind one origin, say so, and
the marker path can be simplified or dropped.

### Blocker 2 — the guard was defeatable, was being defeated, and the zero meant nothing

Three problems, all fixed.

1. **The metric could not see its own blind spot.** `ATTEMPTS` counted only
   *blocked* connects. `_offline_policy` lifts the guard for
   `@pytest.mark.network`, and those tests connect for real and
   `except Exception: pytest.skip`, so the run stayed green and the counter read
   0. "Measured: 0 blocked connections" was true and misleading in the same
   breath. `ATTEMPTS` now records `(kind, target, guarded)` and the summary
   reports blocked, unguarded, remote-DNS and loopback-DNS separately.
2. **The count was invisible under the documented command.** A `print` in a
   teardown fixture is captured; under `pytest -q` it appeared 0 times. The
   summary now goes out through `pytest_terminal_summary`, which the terminal
   reporter writes past the capture, so it shows without anyone having to
   remember `-s`.
3. **`conftest.py:26` was false.** The two `network`-marked tests run on every
   local `pytest`, not only in CI; only the gating CI *job* deselects them. The
   docstring now says so and says the marker means "this test hits the internet",
   not "this test only runs in CI".

**What the guard does not cover, now stated in the docstring instead of implied
away.** `socket.getaddrinfo` is counted, not blocked — blocking it would break
anything that resolves loopback, which is most ASGI clients. `connect_ex` is not
patched. `_socket.socket.connect` called on the C extension type bypasses the
patched attribute. So the honest claim is "connections are blocked and every
attempt is counted", not "the network is unreachable". Patching
`connect`/`create_connection` remains the right design: it is the narrowest point
that covers `requests`, `urllib3`, `http.client` and `pandas.read_csv`, and
`test_the_guard_does_not_break_socket_construction` still guards the
mocking-transport case.

Measured on this machine, the reported numbers are all 0 — including remote DNS —
because `data/cache/` satisfies the two `network`-marked fetches locally. The
reviewer's probe found 4 non-loopback resolutions because it ran where the cache
did not cover them. The counting path is unit-proven instead, against a
`.invalid` host that can never resolve.

### Four comments that described code that does not exist

- **`facts.py:526`** claimed in the present tense that the panel "can then say
  'unavailable'". It cannot: `grep -rn players_unavailable` over `src/`,
  `frontend/src/` and `tests/` found only the producer, my mutation, and test
  assertions. The consumer is the explainer in predictor-hub, outside this tree.
  The comment now says the flag exists for a consumer this repo does not own, and
  the local `Facts` contract model declares the field so it is pinned on this side
  too (it ignores extras, so leaving it undeclared would have meant the contract
  test could not notice the field disappearing — the same "a key nothing reads"
  defect one layer down).

  **The honest consequence, stated plainly: in PUBLIC_MODE a reader still sees
  `players: []` exactly as before. The only reader-visible change from this work
  is that a 500 became a 200.** The flag is groundwork for a consumer that does
  not live here, not a fix to anything a reader can see yet.

- **`routes.py:493`** gave the reason `response: Response = None` exists as
  "because `facts._props` calls this function directly, off-HTTP, to reuse the
  PUBLIC_MODE snapshot rule". False since round 1: `_props` calls
  `get_player_props` only in its **non-PUBLIC_MODE** branch, and its PUBLIC_MODE
  branch calls `snapshot_props_unavailable` directly. It was also a live trap —
  reintroduce a PUBLIC_MODE call from `_props` and `response.headers` is an
  `AttributeError` on `None`, outside the `try`, i.e. Critical 1's exact 500.
  The comment now states the true reason and names the trap.

- **`PROPS_UNAVAILABLE_DETAIL`** still read "unavailable **for this game**" while
  the frontend copy said "for this **week** yet" — and the backend string is
  rendered verbatim by `client.ts` and the component. Now "for this week",
  asserted by `test_the_public_503_body_uses_the_same_unit_as_the_frontend_copy`.
  The runner also stopped retyping the sentence: it reads
  `PROPS_UNAVAILABLE_DETAIL` out of `routes.py`, because feeding its own
  game-scoped wording into the 503 case is how the mismatch passed unnoticed.

- **The report's `pythonpath` claim was wrong.** I wrote that
  `pyproject.toml`'s `pythonpath = ["src"]` was inert. It is not:
  `pythonpath = ["./pp2_marker_dir"]` puts the marker on `sys.path` under pytest
  9.1.1, verified. The real reason imports depend on the editable install is that
  **`pythonpath` has never been in this repo's `pyproject.toml` on any commit**:

  ```
  $ git show origin/main:pyproject.toml | grep -A6 ini_options
  41:[tool.pytest.ini_options]
  42-markers = [
  43:  "network: hits a live upstream API; deselect with '-m \"not network\"'",
  44-]
  $ git log --oneline -S'pythonpath' -- pyproject.toml
  (no output)
  ```

  The task brief's statement that this ini exists is wrong about this repo. The
  **real** reason the harness needs an explicit `PYTHONPATH` is therefore
  stronger than the one I gave: with the key absent, the only thing putting `src`
  on `sys.path` is the editable install, and that points at *this* checkout — so a
  copy's tests would import the original tree. `assert_copy_is_under_test()`
  remains correct and is now backed by the real mechanism.

  The oxlint evidence for Minor 17 was also wrong and is corrected: `no-unused-vars`
  is not in oxlint's default set, so the unused-binding probe I cited reported
  "Found 0 warnings and 0 errors". The conclusion (oxlint does cover the file)
  held and is now demonstrated with rules that are in the default set:

  ```
  appended: const _unusedProbeBinding = 41;
  Found 0 warnings and 0 errors.          <- the probe I cited: useless
  appended: eval() and debugger;
  ! eslint(no-eval): eval can be harmful.
  ! eslint(no-debugger): `debugger` statement is not allowed
  Found 2 warnings and 0 errors.          <- the evidence that holds
  ```

### Two coverage holes the review's probes found

Both SURVIVED, which is the only reason they are here. Neither is behavioural:
they are the two places with a branch and no test looking at it.

- **`routes.py` roster-fallback `except` → `pass`.** Test
  `test_a_roster_fallback_failure_does_not_turn_a_good_week_into_an_empty_one`:
  the week has real history for one of its teams and the roster fetch fails, and
  both the good row and the logged warning must survive. Because that test is
  only meaningful if the fallback is *entered*, it is paired with
  `test_the_roster_fallback_actually_runs_in_the_test_above`, which fails if a
  future change makes `missing_teams` empty in that shape and turns the first test
  into one that passes without exercising anything. Mutation **M27** CAUGHT.
- **The stale-notice copy, pinned by the single substring `'earlier build'`.**
  Reworded into something that no longer says it is stale and nothing noticed.
  The runner now pins three independent fragments — `earlier build`, `rebuild`,
  `current model` — and asserts each in both the stale and stale-empty cases and
  its absence in the fresh case. Mutation **M28** CAUGHT.

Plus the unit mismatch as a mutation: **M29** reverts the backend sentence to
"for this game" and is CAUGHT by both the Python contract test and the runner.

### Optional items, taken

- `assert_copy_is_under_test()` now runs **inside the mutation loop**, not only
  once before it. A mutation that altered import resolution would otherwise point
  the suite back at the real tree and every later verdict would be about code
  nobody edited.
- The frontend arm got the equivalent pre-flight: `run_frontend` checks that the
  copy's `frontend/src/pages/PlayerPropsPage.tsx`, `src/api/client.ts`,
  `props_state_check.mjs` and `node_modules` are all present, and returns
  INCONCLUSIVE naming the missing piece rather than letting vite resolve nothing
  and every frontend mutation come back uninformative.
- `mutation_check.py` block-buffers stdout, so a redirected run shows nothing
  until it exits. Noted in the harness docstring as a known rough edge; not fixed,
  since it changes no verdict.

### Round 2 verification

```
302 passed, 15 skipped          (was 292 / 15)
[offline guard] connection attempts this session:
    blocked while the guard was up : 0
    made with the guard lifted    : 0   <- @pytest.mark.network tests; expected, and not 'clean'
    DNS to a non-loopback host    : 0   <- counted, not blocked; see the module docstring
    DNS to loopback               : 0
```

Mutations: **30 caught, 0 survived, 0 inconclusive, 5 canary (all correct)** — M27
through M31 added this round, all CAUGHT. M30 (`expose_headers` dropped) was also
run by hand so the caught-by is visible rather than asserted:

```
### M30 applied: src/nfl_predictor/api/main.py -> expose_headers removed
FAILED tests/test_props_stale_header_cors.py::test_the_stale_header_is_exposed_to_cross_origin_callers
FAILED tests/test_props_stale_header_cors.py::test_the_marker_header_is_exposed_globally_not_only_on_the_props_route
FAILED tests/test_props_stale_header_cors.py::test_a_fresh_week_declares_the_same_readable_set
3 failed, 302 passed, 15 skipped
```

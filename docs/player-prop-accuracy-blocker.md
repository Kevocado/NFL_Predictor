# Player-prop accuracy is currently unmeasurable, and it is not a code defect

**Status: blocked on an upstream data gap. Re-verified and corrected 2026-09-28.**
Do not spend time re-investigating this before checking the URLs below.

**Correction, 2026-09-28: the pre-kickoff window is retracted.** This document used
to say that a missing pre-kickoff snapshot was "the binding constraint, not the 404".
That was wrong: it named a mechanism that was not doing the work, and it contradicted
this document's own first line. The retraction is recorded in place further down
rather than edited away, so the next reader can see the shape of the mistake. The
404 is the binding constraint. **Nothing about player-prop accuracy is fixed by this
correction** — the props are still unmeasurable and still waiting on nflverse.

## The symptom

`get_track_record()` reports prop accuracy as `n_resolved: 0`. The chain:

1. The tracking tick calls `player_stats.fetch_weekly_player_stats([season])`
   (`api/routes.py`, inside the tick's `try` block).
2. That reads nflverse's weekly player stats for the season being played.
3. nflverse returns **HTTP 404** for the current and previous seasons.
4. `player_stats.fetch_weekly_player_stats` catches it in its own bare
   `except Exception` (`data/player_stats.py:64`) and returns an empty frame
   (`:72`) — deliberately, so the site shows dashes rather than erroring. It logs
   the failure at `logger.info` on the way through (`:65`).
5. `reconcile_player_prop_predictions` returns 0 at its first line
   (`if player_stats_df.empty: return 0`).

So the loop is healthy and nothing is ever resolved. It looks like a tracking bug. It
is a missing upstream file.

### The prop snapshots are not being written either, and it is the same 404

This document also said that props were "snapshotted every tick" and that "the
snapshots are being written". **Both were wrong.** That is not a detail: it moves the
start of the wait, so it is corrected here rather than left in place.

> *Retracted: "`get_track_record()` reports prop accuracy as `n_resolved: 0` even
> though prop predictions are being snapshotted every tick." and "So the loop is
> healthy, the snapshots are being written, and nothing is ever resolved."*

The 404 breaks props in **both** directions, not only the reconciliation direction
listed above:

- **Forward — there is nothing to snapshot.** `_load_player_history(2026)` contains
  no current-season rows, so `latest_players` comes back empty and the roster fallback
  runs. `roster_2026.parquet` is HTTP 200, so the fallback does find every active
  team's roster — and then discards those players one by one, because
  `build_features_for_player` returns `None` for anyone with no usage history in the
  season being predicted (`api/routes.py`). `_get_player_props_live` therefore returns
  `[]`, and `record_player_prop_predictions([])` returns 0 at its first line without
  ever opening the database. **No prop snapshot is written, so none exists to resolve.**

Verified 2026-09-28, against production and against the function the tick actually
calls:

```
GET /api/players/2026/3/props               -> []          (PUBLIC_MODE snapshot)
_get_player_props_live(2026, 3)             -> 0 props     (the tick's own path)
nflverse rosters/roster_2026.parquet        -> HTTP 200    (roster exists;
                                                 usage history does not)
nflverse player_stats/player_stats_2026.parquet -> HTTP 404
```

- **Backward — there is nothing to reconcile.** Unchanged and still true: the tick's
  `fetch_weekly_player_stats([season])` returns an empty frame, so
  `reconcile_player_prop_predictions` is never called.

The practical consequence: this is not "snapshots are piling up, waiting for the
outcome file to arrive". Nothing is piling up. Snapshots only begin once nflverse
publishes `player_stats_2026.parquet`, and from that moment they still have to be
taken pre-kickoff before the first one can resolve. That lengthens step 2 of the
sequence at the end of this document, and it is the one real change to the plan.

## Verified 2026-09-28

```
player_stats_2024.parquet -> HTTP 302   (exists)
player_stats_2025.parquet -> HTTP 404
player_stats_2026.parquet -> HTTP 404
```

And through the production loader, not by poking the URL:

```
season 2024:   5597 rows
season 2025:      0 rows   (no exception raised)
season 2026:      0 rows   (no exception raised)
```

The last part is the part that wastes time. **It fails silently.** A caller cannot
distinguish "nflverse has not published this season" from "there were no stats".

## The two URLs

```
https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2025.parquet
https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_2026.parquet
```

## Why it is not fixed here

One reason, and it is deliberate. (This section previously announced "two separate
reasons"; the second was the pre-kickoff-window claim retracted below.)

**The data does not exist.** This cannot be coded around. A backfill path over
current-season props was considered and withheld: a backfill recomputes predictions
at backfill time from current weights and current data, which is a *reconstruction*,
not a frozen pre-game pick. The repo's tracking rule requires the caller's prediction
to come from strictly pre-game information, and counting reconstructions would inflate
the track record with numbers that were never actually made before kickoff. An
earlier attempt in this session added a `backfilled` column for exactly that purpose
and was reverted when three tests turned out to be pinning the documented decision.

### RETRACTED: "the pre-kickoff window is the binding constraint"

The paragraph and the first list item this replaces said:

> **Even with the data, the numbers would be wrong.** The same pre-kickoff-window
> problem applies: `min-replicas: 0` means a scale-down pushes the next tick past
> kickoff, so no genuine pre-game snapshot exists to resolve. The VPS reported
> `n_resolved 0 / n_rebuilt 33` where an older warm container reported 30 resolved for
> the same 33 games — the missing 30 are the window, not the file.
>
> 1. Fix the pre-kickoff window (an operations decision — see the plan's Task 3,
>    options (a) min-replicas 1, (b) a timer-triggered pre-kickoff tick, (c) accept
>    it). **This is the binding constraint, not the 404.**

**Every load-bearing claim in it is wrong, and the mechanism it names was not doing
the work.** A plausible mechanism written down as a finding is the recurring defect
in this codebase — this document's own opening line said "not a code defect" while
thirty lines later blamed a deployment setting. It is retracted here rather than
quietly deleted, so the next reader can see the shape of it.

**1. The deployment it describes is switched off.** `.github/workflows/deploy-azure-nfl.yml`
is deleted. The only `--min-replicas 0` left in the repository is
`.github/workflows/deploy.yml:91,103`, inside a job gated
`if: vars.DEPLOY_AZURE == 'true'` (deploy.yml:72). `DEPLOY_AZURE` is unset, so that
job has not run since the cutover to the VPS.

**2. The live service cannot scale to zero.** The deployment is a Compose service
(`vps-stack/compose.yml:69-77`) under a shared anchor carrying
`restart: unless-stopped` (compose.yml:21). Docker Compose has no scale-to-zero. The
VPS is rented around the clock whether or not the container is busy, so the
"keep it awake so we are not paying for idle" argument does not apply to this
deployment at all. It was an Azure Container Apps argument, and Azure is off.

**3. The pre-kickoff capture is working — 14 of the 15 resolved games, with one
exception that predates the fix below.** `_tracking_loop` in
`api/main.py` ticks immediately on startup and then every 300s for the life of the
process, so on a continuously supervised container there is no cold-start window in
which a tick can be missed. Production, 2026-09-28:

```
GET /api/current-week       -> {"season":2026,"week":3}
GET /api/track-record games -> n_resolved 14, n_rebuilt 33
                               pct_moneyline 0.786, pct_ats 0.714, pct_totals 0.571
```

Those 14 are graded pre-kickoff picks, and they are pre-kickoff by construction
rather than by claim: `record_game_predictions` hard-rejects any game at or after its
kickoff time, and `_snapshotted_after_kickoff` fails closed. The 33 backfilled rows
are counted and reported separately rather than mixed into the headline. All 16
week-3 games were walked individually; 15 carry a verdict, and coverage spans every
slot — Thu 2026-09-25T00:15Z (20:15 ET), Sun 17:00Z, 20:05Z, 20:25Z, Mon 00:20Z.
Counting `pick_timing` per game makes 15 reconcile with the 14 exactly: 14 are
`pre_kickoff` and one is `rebuilt`, which the track record deliberately excludes. The
one still-unresolved game holds a real frozen row:

```
2026_03_PHI_CHI  kickoff 2026-09-29T00:15:00Z  snapshotted_at 2026-09-25T05:05:18Z
                 backfilled: false
```

A genuine pre-game pick, roughly 91h ahead of kickoff. One honest caveat, so the
number is not read as a current guarantee: 91h exceeds today's 48h
`SNAPSHOT_LEAD_HOURS` window because this row was frozen before that window existed
(commit `ee1d3ef`, 2026-09-26), and `INSERT OR IGNORE` plus the tracking-db backup
volume migrated over from Azure preserve it. Under current code the lead would be at
most 48h.

**4. The Thursday-night question is still open, and this data cannot close it.** The
one backfilled game in week 3 is the Thursday night game itself, `2026_03_ATL_GB`
(`pick_timing: rebuilt`). So the old revision's instinct to worry about Thursday
nights was not baseless — only its explanation was wrong. But that game kicked off
2026-09-25T00:15Z, about 46h *before* `ee1d3ef` (2026-09-26 21:52Z) introduced
next-week inclusion in `_games_to_snapshot`, so it is evidence about the code that ran
before the fix, not about the fix. **Week 4's Thursday game is the first live test of
whether next-week inclusion closed that gap, and that result is not in yet.** Do not
read this document as reporting the Thursday-night gap as closed.

#### Considered and rejected: a scheduled pre-kickoff poke

The plan's option (b), a timer-triggered pre-kickoff tick, was costed and rejected as
**strictly redundant**. The tracking loop already ticks every 300s, so a poke at time
T cannot capture anything the loop did not already have a chance to capture in the
preceding five minutes. On an always-on service it adds operational surface and zero
coverage. Option (a), `min-replicas 1`, is not a live option either: it would change
nothing about what is currently running, and on a VPS the marginal cost of a
container that is already up is effectively zero. Both are recorded here so the next
person does not re-derive this from the plan's Task 3.

#### What is verified, and what is not

**Verified:** the always-on loop captures genuine pre-kickoff *game* snapshots, and 14
week-3 games are graded from them.

**Still open, and worth watching:** whether next-week inclusion in
`_games_to_snapshot` closed the Thursday-night gap. Week 3's Thursday game backfilled,
but it played before that change. Week 4's Thursday game is the first clean test.

**Not verified, and not fixed:** player-prop accuracy is still `n_resolved: 0` in
every market, and no prop snapshot is being written either (see above). It remains
unmeasurable until nflverse publishes `player_stats_2026.parquet`, at which point
snapshots only begin and still need a full pre-kickoff gameweek before the first one
resolves.

So when nflverse publishes, the correct sequence is:

1. Nothing to fix in the deployment. The pre-kickoff window is already covered, so
   the old step 1 is **withdrawn, not deferred**.
2. Let snapshots accumulate for at least one full gameweek — counted from the moment
   the file is published, not from before it, since no prop snapshot exists today to
   accumulate.
3. *Then* prop accuracy becomes measurable. Do not backfill to fill the gap.

## Why this was hard to see

`player_stats.fetch_weekly_player_stats` does log the failure — at `logger.info`
(`data/player_stats.py:65`), which is below the default WARNING threshold, so it does
not appear in normal operation. That is the right level for the fetch itself, which
runs per season per tick and would otherwise flood. The consequence is now logged at
WARNING from the reconcile call site, once per tick, where it is actionable:

```
player prop reconciliation skipped: nflverse has no player stats for season 2026
(player_stats_2026.parquet is 404 upstream); n_resolved will stay 0
```

### Also corrected here: the module that swallows the 404

This section previously read "`hub_cache.cached_frame` does log the failure — at
`logger.info`", and step 4 of the chain above credited the same function with
returning the empty frame. **Both were wrong, and so was the name in the sentence
just rewritten.** `player_stats` does not import `hub_cache`:

```
$ grep -rn hub_cache src/
src/nfl_predictor/api/routes.py:795:      # `hub_cache.cached_frame` catches the upstream 404 and returns an
src/nfl_predictor/data/team_efficiency.py:9:from .hub_cache import cached_frame
src/nfl_predictor/data/player_season.py:8:from .hub_cache import cached_frame
```

Three hits, and **not one of them is `player_stats.py`**. The two imports are the
Data Hub's own pulls. The third is the stale comment quoted in the next paragraph.
`player_stats.py`'s entire import block is `logging`, `pandas` and
`..config.PLAYER_STATS_CACHE_DIR`. The 404 never reaches `hub_cache.cached_frame`;
it is caught by `player_stats`' own bare `except Exception` at
`data/player_stats.py:64`, logged at `:65`, skipped with `continue` at `:66`, and the
empty frame is returned at `:72` by the `if not frames` fallthrough. `hub_cache` does
have its own `logger.info` on the same level (`data/hub_cache.py:27`), which is what
made the misattribution easy to repeat — two unrelated INFO logs, one of them on the
path that actually fails.

**One instance of the same error survives in this repository, in source.** The comment
at `api/routes.py:795`, inside the very `try` block that calls
`player_stats.fetch_weekly_player_stats`, still says `hub_cache.cached_frame` catches
the 404. It is a comment, so it changes no behaviour and no test; it is left here
rather than fixed because this correction is docs-only. Whoever next touches
`background_tracking_tick` should fix that line in the same commit.

**This does not change the diagnosis.** The 404 is real and is the blocker either way;
only the account of *which function* produces the empty frame was wrong. A reader who
follows the corrected path reaches the same conclusion a reader of the old text did,
and now knows where to look.

Commit `6762edb` recorded this same correction in the predictor-hub ledger's
review-round table and did not carry it into this document, so both instances here
survived it. This is the same failure mode as the retracted section above — a
plausible mechanism written down as a finding — and it is corrected in place rather
than quietly deleted.

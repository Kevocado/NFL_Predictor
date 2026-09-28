# Player-prop accuracy is currently unmeasurable, and it is not a code defect

**Status: blocked, on two independent things. Re-verified 2026-09-28.**
Do not spend time re-investigating this before checking the URLs below. Note that the
upstream gap is real but is **not** the binding constraint — see "Which blocker binds
first" at the end, which contradicts the framing in the first two sections.

## The symptom

`get_track_record()` reports prop accuracy as `n_resolved: 0` even though prop
predictions are being snapshotted every tick. The chain:

1. The tracking tick calls `player_stats.fetch_weekly_player_stats([season])`
   (`api/routes.py`, inside the tick's `try` block).
2. That reads nflverse's weekly player stats for the season being played.
3. nflverse returns **HTTP 404** for the current and previous seasons.
4. `player_stats.fetch_weekly_player_stats` catches the failure itself — a bare
   `except Exception` around `_import_weekly_data` that logs at INFO and continues —
   so the function returns an empty frame. (An earlier version of this document
   credited `hub_cache.cached_frame` for this. It is not on this code path and
   `player_stats` does not import it.)
5. `reconcile_player_prop_predictions` returns 0 at its first line
   (`if player_stats_df.empty: return 0`).

So the loop is healthy, the snapshots are being written, and nothing is ever
resolved. It looks like a tracking bug. It is a missing upstream file.

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

Two separate reasons, and both are deliberate.

**The data does not exist.** This cannot be coded around. A backfill path over
current-season props was considered and withheld: a backfill recomputes predictions
at backfill time from current weights and current data, which is a *reconstruction*,
not a frozen pre-game pick. The repo's tracking rule requires the caller's prediction
to come from strictly pre-game information, and counting reconstructions would inflate
the track record with numbers that were never actually made before kickoff. An
earlier attempt in this session added a `backfilled` column for exactly that purpose
and was reverted when three tests turned out to be pinning the documented decision.

**Even with the data, the numbers would be wrong.** The same pre-kickoff-window
problem applies: `min-replicas: 0` means a scale-down pushes the next tick past
kickoff, so no genuine pre-game snapshot exists to resolve. The VPS reported
`n_resolved 0 / n_rebuilt 33` where an older warm container reported 30 resolved for
the same 33 games — the missing 30 are the window, not the file.

### Which blocker binds first

The first two sections say the blocker is the missing file. **That is wrong**, and the
rest of this document contradicts it. The pre-kickoff window is the binding constraint:
even with the data published, `min-replicas: 0` means no genuine pre-game snapshot
exists to resolve, so `n_resolved` would stay 0 and the file's arrival would change
nothing on its own. The 404 is a real and verified gap, but it is the *second* thing to
fix, not the first.

The evidence for the window is inference rather than measurement: the configuration is
real (`deploy-azure-nfl.yml` sets `--min-replicas 0 --max-replicas 3`) and the VPS
reported `n_resolved 0 / n_rebuilt 33` where a warm container reported 30 resolved for
the same 33 games. That is a 33-game sample of **game** data; prop snapshots are written
by a different block in the same tick, and no prop snapshot lost to a scale-down has
been shown. Treat the mechanism as plausible and unverified.

So when nflverse publishes, the correct sequence is:

1. Fix the pre-kickoff window (an operations decision — see the plan's Task 3,
   options (a) min-replicas 1, (b) a timer-triggered pre-kickoff tick, (c) accept
   it). **This is the binding constraint, not the 404.** It is also unverified, in
   the way described above.
2. Let genuine pre-game snapshots accumulate for at least one full gameweek.
3. *Then* prop accuracy becomes measurable. Do not backfill to fill the gap.

## Why this was hard to see

The failure **is** logged — at `logger.info`, from
`player_stats.fetch_weekly_player_stats`, which is below the default WARNING
threshold, so it does not appear in normal operation. The consequence is now logged
at WARNING from the reconcile call site, once per tick, where it is actionable:

```
player prop reconciliation skipped: no player stats for season 2026, so no snapshot
can be resolved and n_resolved stays 0. Most likely nflverse has not published the
season's weekly file (player_stats_2026.parquet currently 404s); it may also be a
rate limit or a schema change. See docs/player-prop-accuracy-blocker.md
```

That message deliberately names **no** cause as certain. The branch is reached by at
least three: a 404 because the season is unpublished (today's case, verified), a 200
carrying no rows for the season, and a rate limit or schema change swallowed by the
same bare `except`. An earlier version of this line asserted "404 upstream" flatly,
which is a guess presented as a diagnosis.

**Two caveats on the WARNING itself, both verified rather than assumed.** It fires only
when `completed` (this season's finished games) is non-empty, which is true from the
first game of the season but false in the offseason — so the condition is silent
exactly when it is expected. And the tracker runs every
`_TRACKING_INTERVAL_SECONDS` (300s), so a warm container emits roughly 288 of these
lines a day. That is the intended trade for a single actionable line, but it is a
number worth knowing.

## A bug this document's own remediation plan would have hit

`fetch_weekly_player_stats` wrote the per-season parquet **unconditionally**,
including when the frame was empty, while the cache check was a bare `path.exists()`
with no TTL. So a single empty pull — the shape this API has *before* a season is
published — was cached and returned forever:

```
1st call: 0 rows, empty=True, cache file written
2nd call (upstream now published): 0 rows, empty=True
```

Prop reconciliation could therefore never recover, even after nflverse published, and
the sequence below would have stalled at step 1. Empty pulls are no longer cached and
an existing empty cache file is discarded. `hub_cache` states the right discipline in
its own docstring — *"Empty or failed pulls are never cached"* — and this module did
not follow it.

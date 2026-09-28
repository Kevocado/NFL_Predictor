# Player-prop accuracy is currently unmeasurable, and it is not a code defect

**Status: blocked on an upstream data gap. Re-verified 2026-09-28.**
Do not spend time re-investigating this before checking the URLs below.

## The symptom

`get_track_record()` reports prop accuracy as `n_resolved: 0` even though prop
predictions are being snapshotted every tick. The chain:

1. The tracking tick calls `player_stats.fetch_weekly_player_stats([season])`
   (`api/routes.py`, inside the tick's `try` block).
2. That reads nflverse's weekly player stats for the season being played.
3. nflverse returns **HTTP 404** for the current and previous seasons.
4. `hub_cache.cached_frame` catches it and returns an empty frame — deliberately,
   so the site shows dashes rather than erroring.
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

So when nflverse publishes, the correct sequence is:

1. Fix the pre-kickoff window (an operations decision — see the plan's Task 3,
   options (a) min-replicas 1, (b) a timer-triggered pre-kickoff tick, (c) accept
   it). **This is the binding constraint, not the 404.**
2. Let genuine pre-game snapshots accumulate for at least one full gameweek.
3. *Then* prop accuracy becomes measurable. Do not backfill to fill the gap.

## Why this was hard to see

`hub_cache.cached_frame` does log the failure — at `logger.info`, which is below the
default WARNING threshold, so it does not appear in normal operation. That is the
right level for the fetch itself, which runs per season per tick and would otherwise
flood. The consequence is now logged at WARNING from the reconcile call site, once
per tick, where it is actionable:

```
player prop reconciliation skipped: nflverse has no player stats for season 2026
(player_stats_2026.parquet is 404 upstream); n_resolved will stay 0
```

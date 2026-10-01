"""public_snapshot.py — precomputes games/predictions/player-props for a
window of weeks so the public deployment never has to run this project's
live feature-building pipeline (schedule/player-stat fetch, feature build,
model predict, the CFBD-style roster fallback) on an actual request.

Run this locally, or from .github/workflows/refresh-public-snapshot.yml:

    python -m nfl_predictor.public_snapshot

Then commit + push data/public_snapshot.json -- the running deployment
picks it up within PUBLIC_SNAPSHOT_POLL_SECONDS via
api/routes.py::refresh_public_snapshot_from_remote, no redeploy needed.
See config.py's PUBLIC_SNAPSHOT_PATH for the rest of that mechanism.
"""

from __future__ import annotations

import json
import math

from fastapi.encoders import jsonable_encoder

from . import config
from .api import routes
from .models import player_props

# How many weeks past the current one get freshly rebuilt every run.
# Everything else reuses the previous snapshot verbatim (if that week was
# already in it) -- a full-season rebuild touches ~270 games plus a full
# player-props pass per week (the slowest part by far, since a team with
# no current-season stats yet falls back to the whole-FBS-equivalent
# roster), most of it for weeks nobody's currently looking at.
REBUILD_WEEKS_AHEAD = 4
REBUILD_WEEKS_BEHIND = 1
MAX_WEEK = 22


def _build_week(season: int, week: int, previous: dict | None = None) -> dict:
    """One week of precomputed data.

    `previous` is that same week from the last snapshot, and it exists so a props
    rebuild that fails keeps the props already computed. The previous `except
    Exception: player_props = []` overwrote real rows with an empty list, which is
    how the committed file came to hold 14-16 games and zero props for weeks 2-7
    -- a transient failure frozen into a tracked artifact and then served as a
    permanent, honest-looking empty state. Serving the last good props with
    status "stale" is strictly better than a blank, and the status key is what
    stops the two from being confused again.
    """
    games = routes._get_games_live(season, week)
    predictions: dict[str, dict] = {}
    for game in games:
        game_id = game["game_id"]
        try:
            predictions[game_id] = routes._get_game_prediction_live(season, week, game_id)
        except Exception as exc:
            print(f"    ! skipped prediction for {game_id}: {exc}")
    # Collected out of `_get_player_props_live` because production is PUBLIC_MODE
    # (`Dockerfile`: ENV PUBLIC_MODE=true): the props a reader sees come from this
    # artifact, so an out list that was not stored here would exist only in the
    # live path and never reach a page. Gating still happens either way -- the
    # rows below are already gated -- but the *reason* a player is missing from
    # them has to be stored alongside them or the frontend cannot attribute it.
    out_players: list[dict] = []
    try:
        player_props = routes._get_player_props_live(season, week, out_players=out_players)
        props_status = "ok"
    except Exception as exc:
        carried = (previous or {}).get("player_props") or []
        # The out entries of a carried-forward week are the PREVIOUS build's, and
        # they describe a report that has since been superseded. Dropping them is
        # the honest direction: the ranking they explain is still being served,
        # and an out line dated to an earlier report would attribute that
        # ranking's absences to facts we can no longer stand behind.
        out_players = []
        if carried:
            print(f"    ! player props for week {week} failed ({exc}); keeping {len(carried)} from the previous snapshot")
            player_props, props_status = carried, "stale"
        else:
            # Nothing to carry and nothing computed. Record that, so the route can
            # answer 503 instead of claiming the week has no props.
            print(f"    ! player props for week {week} failed and there is nothing to carry: {exc}")
            player_props, props_status = [], "unavailable"
    return {
        "games": games,
        "predictions": predictions,
        "player_props": player_props,
        "player_props_status": props_status,
        "player_props_out": out_players,
    }


def _market_keys() -> frozenset[str]:
    """Every key `POSITION_MARKETS` can put on a prop row.

    `predict_props` (models/player_props.py:47-57) emits `anytime_td_prob`
    unconditionally and then one key per market, and only `if market in models`
    -- so the reachable set is exactly the union of the per-position lists, and
    which subset a given row gets is decided inside the position. The callers
    here treat that set as position-specific, which is the safe reading: a
    market is present or absent for reasons that have nothing to do with the
    code's age, so requiring one of these keys says nothing about whether a
    week predates a change.

    `anytime_td_prob` is deliberately NOT in here: `predict_props` sets it on
    every row, whatever the position, so it IS position-invariant and is
    required like any other invariant key.
    """
    return frozenset(
        market for markets in player_props.POSITION_MARKETS.values() for market in markets
    )


def _position_invariant_keys(rows: list[dict]) -> frozenset[str] | None:
    """The prop-row keys the current code emits on EVERY row, whatever the position.

    Prop rows are position-heterogeneous, and deliberately so: `predict_props`
    keys off `POSITION_MARKETS` (models/player_props.py:53), so a QB row carries
    `passing_yards`, an RB row `rushing_yards`, and a WR/TE row
    `receiving_yards`. The committed artifact holds 10,416 prop rows across 12
    prop weeks and three distinct row shapes in every one of them (WR and TE
    share a shape, so four positions give three).

    What a position may carry is wider here than in the sibling CFB repo, and in
    the direction that makes the invariant smaller: `_get_player_props_live`
    does NOT filter its output by position (routes.py:445-452 selects straight
    off the player history; only the season-roster fallback filters, at
    routes.py:463), and `predict_props` reads `POSITION_MARKETS.get(position, [])`
    (models/player_props.py:53). So a K/OL/DL/P player reaches `predict_props`
    and comes back with no market at all -- "a row with no market" IS a shape a
    row here can have, unlike in CFB. Nothing is guaranteed about which of the
    four modelled positions a week contains, and within one position the market
    set is still decided per row by `if market in models`
    (models/player_props.py:54).

    So the invariant key set is computed two ways, and both are needed:

    * every key that `POSITION_MARKETS` can introduce is SUBTRACTED, by name,
      from the sample. That makes the result independent of which positions
      the sample contains, including a single-position sample -- so an all-QB
      week cannot put `passing_yards` in the signature and then demand it of
      every WR row in the season. This is the property the check actually
      needs, and it is unconditional rather than a property of the sample.
    * what is left is the INTERSECTION of the sample's rows, so any *other*
      position-specific key added later without touching `POSITION_MARKETS`
      still cannot leak in by being row 0's shape. Defence in depth, not the
      mechanism.

    Returns None for an empty sample, which the caller reports rather than
    guesses from (see `_prop_key_signature`).
    """
    if not rows:
        return None
    invariant = frozenset(rows[0].keys())
    for row in rows[1:]:
        invariant &= row.keys()
    return invariant - _market_keys()


def _prop_key_signature(
    season: int,
    current_week: int,
    weeks: dict[str, dict],
    reused: list[str],
) -> frozenset[str] | None:
    """The position-invariant prop-row keys the CURRENT code produces, or None if
    undeterminable.

    Every rebuilt week's rows are pooled, because prop rows are not one shape
    (see `_position_invariant_keys`). Pooling costs nothing extra -- those weeks
    are already built. It is defence in depth: the subtraction of
    `POSITION_MARKETS` is what makes the result position-invariant, and it
    holds for a sample of one row, so pooling is not what stands between an
    all-QB week and a season-wide demand for `passing_yards`.

    Only if every rebuilt week is prop-less -- which is the real situation early
    in a season, when the rebuild window holds no games yet -- does it make a
    single live call for the current week, and it intersects that sample the
    same way. Still one call, never one per week.

    Returns None rather than an empty set when it genuinely cannot tell. An
    empty set would compare equal against every prop-less week and report
    "nothing to do", which is the bug being fixed.
    """
    current_rows: list[dict] = []
    for key, week in weeks.items():
        if key in reused:
            continue
        current_rows.extend(week.get("player_props") or [])
    signature = _position_invariant_keys(current_rows)
    if signature is not None:
        return signature

    try:
        sample = routes._get_player_props_live(season, current_week)
    except Exception as exc:  # noqa: BLE001 - reported by the caller
        print(f"  ! live prop probe failed: {exc}")
        return None
    return _position_invariant_keys(sample or [])


def _prop_shape_mismatch(week: dict, required: frozenset[str]) -> bool:
    """True when ANY prop row of a reused week predates the current row shape.

    Every row is checked, not row 0. `required` is position-invariant (see
    `_position_invariant_keys`), so a week whose row 0 happens to be current
    while its other rows are stale is stale -- the case a row-0 subset test
    could not see at all, because a prop week here holds hundreds of rows and it
    only ever asked about one of them.

    A prop-less week is not a mismatch: it has no rows to be stale, and
    rebuilding it would be pure cost.
    """
    props = week.get("player_props") or []
    if not props:
        return False
    return any(not required <= row.keys() for row in props)


def build_snapshot(previous: dict | None = None) -> dict:
    season, current_week = routes.current_season_and_week()
    previous = previous or {}
    previous_weeks = previous.get("weeks", {}) if previous.get("season") == season else {}

    rebuild_from = max(1, current_week - REBUILD_WEEKS_BEHIND)
    rebuild_to = min(MAX_WEEK, current_week + REBUILD_WEEKS_AHEAD)

    print(
        f"Building weeks 1-{MAX_WEEK} (current: {current_week}); "
        f"rebuilding {rebuild_from}-{rebuild_to}, reusing the rest..."
    )
    weeks: dict[str, dict] = {}
    reused: list[str] = []
    for week in range(1, MAX_WEEK + 1):
        key = str(week)
        if rebuild_from <= week <= rebuild_to or key not in previous_weeks:
            print(f"  week {week}")
            weeks[key] = _build_week(season, week, previous=previous_weeks.get(key))
        else:
            weeks[key] = previous_weeks[key]
            reused.append(key)

    # A reused week is a copy, so it can never pick up a field that the code has
    # since started emitting. That is not a hypothetical: `is_starter` and
    # `depth_slot` were added to the prop rows (routes.py:509-510) and every
    # week that actually carries props sits OUTSIDE the rebuild window -- at
    # current_week 3 the window is weeks 2-7, while week 1 and weeks 8-18 are
    # copies, and 2-7 have no games yet. So the new fields reached the artifact
    # nowhere: 0 of its 10,416 prop rows carry them. The symptom is a frontend
    # that looks for a field the API is supposed to serve and finds it missing
    # on the whole season -- and the obvious "it's a serialization bug"
    # conclusion is wrong. It is the copy, not the writer.
    #
    # So: work out the shape the CURRENT code produces, and rebuild any reused
    # week whose props do not match it. Narrow on purpose -- only the weeks
    # whose row shape actually changed are rebuilt, so adding a field costs one
    # build rather than all twenty-two. A week carrying fields the code no
    # longer emits is left alone rather than caught in a rebuild loop.
    #
    # "Shape" has to mean the position-INVARIANT keys and be checked on every
    # row. Both halves are load-bearing and they do different jobs:
    #
    #   * Checking every row is what makes the predicate CONSERVATIVE. A
    #     position-invariant key (`is_starter`, `depth_slot`, `anytime_td_prob`)
    #     is required of each row independently, so a week where row 0 happens to
    #     be current and the other 927 rows are stale is stale. Judging one row
    #     leaves 927 unchecked -- and on this artifact the two verdicts happen to
    #     agree anyway, because all 12 prop weeks are stale on EVERY row and
    #     every one of their row 0s is a QB, which is also what a QB-first sample
    #     demands. That is a coincidence of this artifact, not a property of the
    #     check, and it is what the artifact tests exist to keep from mattering.
    #   * Being position-invariant is what makes it CHEAP. A key that follows the
    #     position is excluded from `required` by construction, so no mix of
    #     rows can demand it of a row that legitimately lacks it. A WR row
    #     without `rushing_yards` is a player whose position moved, not a stale
    #     shape, and rebuilding the week over it costs a full `_build_week` --
    #     real nfl_data_py fetches -- on every scheduled run, forever.
    #
    # What this deliberately does NOT catch, because it cannot: a NEW
    # position-specific field. `POSITION_MARKETS` is per-position, so the most
    # likely next change to this row shape is a market added to one position --
    # and that is not hypothetical here, because `carries` and `receptions` are
    # already listed for RB and WR/TE (models/player_props.py:25-28) and reach
    # the rows only once models are trained for them. The subtraction excludes
    # such a key from `required` on purpose, before any row is looked at, so
    # the day those models exist the new key reaches the rebuilt window weeks
    # and no others, and nothing here will say so. A position-specific field
    # needs its own decision: either a one-off rebuild of the season, or a
    # deliberately widened `required` for that run. Do not "fix" it by making
    # `required` position-dependent again -- that is the defect this signature
    # replaced, and it costs a full `_build_week` per affected week on every
    # scheduled run.
    #
    # Provenance, because the two repos are copies of each other and this one
    # is the CORRECTED one. NFL_Predictor commit 0c4ea1e is the ORIGINAL fix and
    # the row-0 version: it took `props[0]`'s key set as the signature and
    # tested it against `props[0]`'s keys, which is wrong in both directions
    # (a week whose row 0 was current while its other rows were stale was left
    # stale forever; a fully current week was rebuilt whenever row 0's position
    # happened to differ from the sample's). CFB_Predictor commits 13d0e9c,
    # 2076635 and 4668232 are that fix redone, reviewed twice; 4668232 is the
    # subtraction above, and it is the design this file carries. The names
    # differ because `POSITION_MARKETS` and the prop-row fields differ, so do
    # not copy NFL's identifiers into CFB or the reverse -- copy the
    # construction, and in particular keep the subtraction. A plain
    # intersection, or a single row, still depends on the sample happening to
    # be multi-position.
    signature = _prop_key_signature(season, current_week, weeks, reused)
    if signature is None:
        print("  ! could not determine the current prop shape; reused weeks were NOT reconciled")
    else:
        stale = [key for key in reused if _prop_shape_mismatch(weeks[key], signature)]
        for key in stale:
            print(f"  week {key}: prop shape changed, rebuilding")
            weeks[key] = _build_week(season, int(key))
        if stale:
            print(f"  reconciled {len(stale)} reused week(s) onto the new prop shape")

    print("Building season standings projection...")
    try:
        standings = routes._get_standings_live(season)
    except Exception as exc:
        print(f"  ! skipped standings: {exc}")
        standings = previous.get("standings", []) if previous.get("season") == season else []

    print("Building power rankings...")
    try:
        power_rankings = routes._get_power_rankings_live(season)
    except Exception as exc:
        print(f"  ! skipped power rankings: {exc}")
        power_rankings = previous.get("power_rankings", {}) if previous.get("season") == season else {}

    print("Building Data Hub tables...")
    try:
        hub_teams = routes._get_hub_teams_live(season)
    except Exception as exc:
        print(f"  ! skipped hub teams: {exc}")
        hub_teams = previous.get("hub_teams", {}) if previous.get("season") == season else {}
    try:
        hub_players = routes._get_hub_players_live(season)
    except Exception as exc:
        print(f"  ! skipped hub players: {exc}")
        hub_players = previous.get("hub_players", {}) if previous.get("season") == season else {}

    import pandas as pd

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "season": season,
        "current_week": current_week,
        "weeks": weeks,
        "standings": standings,
        "power_rankings": power_rankings,
        "hub_teams": hub_teams,
        "hub_players": hub_players,
    }


def sanitize_floats(value):
    """Replace every non-finite float with None, recursively.

    The feature pipeline produces NaN for "this market has no line" (an
    unplayed game has no spread, a team with no cover model has no cover
    probability). NaN is not valid JSON, and the public deployment serves
    this file verbatim through a starlette JSONResponse, which renders with
    allow_nan=False and turns a NaN into a 500. Null is the honest
    encoding: the UI already renders a dash for a missing number.
    """
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: sanitize_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_floats(v) for v in value]
    return value


def main() -> None:
    previous = json.loads(config.PUBLIC_SNAPSHOT_PATH.read_text()) if config.PUBLIC_SNAPSHOT_PATH.exists() else None
    snapshot = sanitize_floats(jsonable_encoder(build_snapshot(previous)))
    # allow_nan=False is the guard, not the mechanism: sanitize_floats has
    # already replaced every non-finite float, so reaching here means a new
    # one appeared somewhere and the build must fail loudly rather than
    # write invalid JSON again.
    config.PUBLIC_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, allow_nan=False))
    print(f"Wrote {config.PUBLIC_SNAPSHOT_PATH} ({config.PUBLIC_SNAPSHOT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

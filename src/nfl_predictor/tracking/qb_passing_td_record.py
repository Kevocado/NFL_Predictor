"""qb_passing_td_record.py -- grading and reporting for the QB passing-TD call.

**Deliberately not in `tracking/store.py`.** NFL PR #25 is open and unmerged and
owns `tracking/store.py`; it rewrites `_summarize_player_props`, `reconcile_
player_prop_predictions`' neighbourhood and the kickoff helpers. Keeping the
grading here means this branch and #25 have no overlapping hunk in that file, so
both can land in either order. The only thing this module needs from the store is
the resolved rows, read through a connection it is handed.

The counting rule is the one NFL PR #25 implements, read from that branch rather
than re-derived: **one counted pick per `(game_id, player_id, market)`, the
earliest recorded.** `player_id` is in the key -- many players share one market in
one game, so `(game, market)` would collapse a whole prop board into one pick.

Grading
-------
The line is always a half point and the actual passing TDs are always an
integer, so equality is unreachable and **push cannot occur**. `grade_pick`
asserts the line really is a half point rather than trusting it, because that
assertion is the only thing standing between a future whole-number line and a
silently mis-graded record.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from ..models.qb_passing_td import PASSING_TD_MARKET

#: The columns this module needs from a resolved frame. Named so a caller
#: selecting from SQL cannot silently omit one and get a KeyError deep in a
#: pandas reduction instead.
REQUIRED_COLUMNS = (
    "game_id", "player_id", "market", "line", "side", "mu", "call_prob",
    "resolved", "actual_value", "snapshotted_at",
)


def _utc_instant(value) -> datetime | None:
    """An ISO-8601 timestamp as an aware UTC instant, or None if unreadable.

    Offset honoured, so a pick stamped `-05:00` sorts by when it actually
    happened and not by how its string reads. This is the same rule PR #25's
    `_utc_instant` uses; it is repeated here rather than imported because #25 is
    unmerged and importing from it would make this branch unrunnable today.
    """
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def counted_passing_td_picks(resolved: pd.DataFrame) -> pd.DataFrame:
    """The passing-TD rows that count: earliest per `(game_id, player_id, market)`.

    An unparseable timestamp sorts LAST, so it can never displace a row that
    carries a real instant and claim to be the earliest pick.
    """
    if resolved.empty:
        return resolved
    ordered = resolved.assign(_instant=resolved["snapshotted_at"].map(_utc_instant))
    ordered = ordered.sort_values(
        "_instant", na_position="last", kind="stable", ignore_index=True)
    return ordered.drop_duplicates(subset=["game_id", "player_id", "market"], keep="first")


def grade_pick(line: float, side: str, actual: float) -> bool:
    """Whether a call hit, given the line, the side called, and the actual count.

    `actual > line` for over and `actual < line` for under. Push is impossible
    because the line is a half point, which is asserted here rather than assumed.
    """
    line = float(line)
    if not line_is_half_point(line):
        raise AssertionError(
            f"passing-TD line {line} is not a half point, so over and under are not "
            "exhaustive and this pick cannot be graded without a push outcome"
        )
    actual = float(actual)
    return actual > line if side == "over" else actual < line


def summarize_passing_td_calls(resolved: pd.DataFrame) -> dict:
    """The passing-TD track record: n, hit rate, and one row per counted pick."""
    empty = {"n_resolved": 0, "n_called": 0, "hit_rate_when_called": None,
             "by_side": {}, "per_pick": []}
    if resolved is None or resolved.empty:
        return empty

    rows = resolved[resolved["market"] == PASSING_TD_MARKET]
    if rows.empty:
        return empty

    counted = counted_passing_td_picks(rows)
    per_pick = []
    hits = 0
    by_side: dict[str, dict] = {}
    for _, row in counted.iterrows():
        hit = grade_pick(row["line"], row["side"], row["actual_value"])
        hits += int(hit)
        bucket = by_side.setdefault(row["side"], {"n": 0, "hits": 0})
        bucket["n"] += 1
        bucket["hits"] += int(hit)
        per_pick.append({
            "game_id": row["game_id"],
            "player_id": row["player_id"],
            "player_name": row.get("player_name"),
            "line": float(row["line"]),
            "line_source": row.get("line_source", "model_line"),
            "side": row["side"],
            "mu": float(row["mu"]),
            "call_prob": float(row["call_prob"]),
            "actual_passing_tds": float(row["actual_value"]),
            "hit": hit,
        })

    for side, bucket in by_side.items():
        bucket["hit_rate"] = bucket["hits"] / bucket["n"] if bucket["n"] else None

    n = len(per_pick)
    return {
        "n_resolved": n,
        "n_called": n,
        "hit_rate_when_called": hits / n if n else None,
        "by_side": by_side,
        "per_pick": per_pick,
    }


def read_resolved_passing_td_picks(conn) -> pd.DataFrame:
    """Every resolved passing-TD row from the tracking database.

    Reading rather than filtering upstream means the query cannot accidentally
    apply a kickoff rule that differs from the one the rest of the record uses;
    grading is decided by the caller from the same rows everyone else sees.
    """
    query = (
        f"SELECT {', '.join(REQUIRED_COLUMNS)}, player_name, line_source "
        "FROM player_prop_predictions WHERE market = ? AND resolved = 1"
    )
    return pd.read_sql(query, conn, params=(PASSING_TD_MARKET,))


def passing_td_track_record(conn) -> dict:
    """The passing-TD section of the track record, from the live database."""
    try:
        frame = read_resolved_passing_td_picks(conn)
    except Exception:
        # A database without the passing-TD columns yet is an empty record, not
        # a failed one: the migration is additive and runs on connect.
        return {"n_resolved": 0, "n_called": 0, "hit_rate_when_called": None,
                "by_side": {}, "per_pick": []}
    return summarize_passing_td_calls(frame)


def line_is_half_point(line: float) -> bool:
    """Whether `line` is a half point, i.e. push is impossible against it.

    The test is on the FRACTIONAL part, not on `line * 2`: doubling 1.5 gives 3.0,
    which is an integer, so "is `line * 2` an integer" is the question that
    matters and the answer for a half point is yes. A whole-number line doubles to
    an even integer and its fractional part is 0.
    """
    value = float(line)
    if not math.isfinite(value):
        return False
    return abs(value * 2 - round(value * 2)) < 1e-9 and abs(value - round(value)) == 0.5
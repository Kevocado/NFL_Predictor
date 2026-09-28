"""SQLite persistence for the live prediction track record.

Snapshots each game's core-market predictions and each tracked player prop
before kickoff, then reconciles them against actual results once games are
played. Snapshot rows are immutable (``INSERT OR IGNORE``); reconciliation
only fills outcome columns on existing, unresolved rows.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import norm

from ..config import CACHE_DIR, TRACKING_DB_PATH


def _connect() -> sqlite3.Connection:
    """Open the tracking database and ensure its schema exists."""
    conn = _open()
    _ensure_schema(conn)
    return conn


def _open() -> sqlite3.Connection:
    """A connection to the live local tracking file, with SQLite's own timeout
    pragmas set. Schema work is `_ensure_schema`'s job, not this function's, so the
    two can be told apart: `_connect` ensures and discards the result,
    `migrate_tracking_db` ensures and reports it."""
    conn = sqlite3.connect(str(TRACKING_DB_PATH), timeout=15)
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> list[str]:
    """Bring the schema up to date and return the columns that were added (empty if
    it was already current). Split out from `_connect` so `migrate_tracking_db` can
    report what a boot actually changed, rather than discovering it had already been
    applied by the `_connect` it called."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS game_predictions (
            game_id TEXT PRIMARY KEY,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            home_cover_prob REAL,
            away_cover_prob REAL,
            over_prob REAL,
            under_prob REAL,
            home_spread_line REAL,
            total_line REAL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            moneyline_hit INTEGER,
            ats_hit INTEGER,
            total_hit INTEGER
        )
        """
    )
    added = _apply_schema_migrations(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_prop_predictions (
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            market TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            snapshotted_at TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_value REAL,
            PRIMARY KEY (game_id, player_id, market)
        )
        """
    )
    # Phase 9: per-position MAE needs the player's position on each prop
    # row. ALTER TABLE keeps existing rows (position NULL) readable --
    # they stay in overall metrics, just out of position groups.
    prop_cols = {row[1] for row in conn.execute("PRAGMA table_info(player_prop_predictions)")}
    if "position" not in prop_cols:
        conn.execute("ALTER TABLE player_prop_predictions ADD COLUMN position TEXT")
        added.append("player_prop_predictions.position")
    return added


# Every column added to `game_predictions` after the original CREATE TABLE, as
# (name, SQL type), in the order it was added. One declaration, so the migration
# cannot grow a column in one place and forget it in the other.
#
# The last five are the Kalshi feed's predicted distribution, and the first six
# are the lines and grades the track record grades. `predicted_total` and
# `predicted_margin` are in BOTH features' territory: the model has computed them
# for as long as it has computed `over_prob` (`_predict_game_from_models` returns
# them to every caller, and `margin_to_probabilities` turns the total into
# `over_prob`), so the feed could record them and nothing ever summarised them.
_MIGRATION_COLUMNS = (
    ("home_spread_line", "REAL"),
    ("total_line", "REAL"),
    ("ats_hit", "INTEGER"),
    ("total_hit", "INTEGER"),
    ("season", "INTEGER"),
    ("week", "INTEGER"),
    ("predicted_margin", "REAL"),
    ("sigma", "REAL"),
    ("predicted_total", "REAL"),
    ("total_sigma", "REAL"),
    ("model_version", "TEXT"),
)


def _apply_schema_migrations(conn: sqlite3.Connection) -> list[str]:
    """Add whichever declared columns are missing, and return their names.

    Idempotent: it runs on every connection the app opens, and on every cold
    start, so a second pass must add nothing and must not raise.

    One `ALTER TABLE` per column, each its own statement, because that is the
    only form SQLite can run against an existing table.
    """
    # Refreshed after each add rather than cached up front. Caching is what let the
    # original two-loop version below drift: the second loop tested against a
    # `existing_cols` set the first loop had already invalidated, so a column
    # named in both would be attempted twice. One list, one loop, no cache.
    added = []
    for column, sql_type in _MIGRATION_COLUMNS:
        present = {row[1] for row in conn.execute("PRAGMA table_info(game_predictions)")}
        if column not in present:
            conn.execute(f"ALTER TABLE game_predictions ADD COLUMN {column} {sql_type}")
            added.append(column)
    return added


class TrackingDbOnPersistentMount(RuntimeError):
    """Raised instead of migrating. SQLite's locking does not survive the Azure
    Files mount, and the fix for that is not a longer timeout."""


def _is_on_persistent_mount(path) -> bool:
    """Whether `path` is inside the Azure Files cache mount.

    Fails SAFE in the only direction that matters: a path it cannot place is
    reported as not-on-the-mount, because guessing "yes" would refuse a perfectly
    good local database and guessing the other way here is the thing the guard
    exists to prevent.
    """
    try:
        return Path(path).resolve().is_relative_to(Path(CACHE_DIR).resolve())
    except (OSError, ValueError):
        return False


def migrate_tracking_db() -> list[str]:
    """Apply the schema, idempotently, on the LIVE local database. Returns the
    columns added, empty when there was nothing to do.

    **This must never run against the Azure Files mount.** SQLite does not work
    reliably over SMB/Azure Files -- confirmed live: "database is locked" the
    moment `TRACKING_DB_PATH` itself pointed at the mounted cache volume. The fix
    this repo already carries (commit 458bed2) is not a longer `busy_timeout` and
    not a retry loop: the live database stays on local, ephemeral container disk
    and is COPIED to and from the persistent mount at a quiescent point, with no
    connection open at either end (`api/main.py::_restore_tracking_db` /
    `_backup_tracking_db`). Copy files; never open the database across the share.

    So the migration runs on the local file, and this refuses to start at all if
    the path has been pointed at the mount by mistake. Failing loudly here is the
    point: the alternative is a live tracker that dies inside a DDL statement on
    first boot, which is strictly worse than never migrating. Nothing is created
    on the mount when this raises.

    Called from `api/main.py`'s lifespan, immediately after the backup is restored
    and before any task exists -- the same quiescent-point discipline the restore
    and backup already follow, and before the first tick can copy a
    half-migrated file onto the mount.
    """
    if _is_on_persistent_mount(TRACKING_DB_PATH):
        raise TrackingDbOnPersistentMount(
            f"Refusing to migrate {TRACKING_DB_PATH}: it is inside the persistent cache mount "
            f"({CACHE_DIR}), and SQLite's locking does not survive Azure Files/SMB. The live "
            f"database belongs on local disk and is copied to the mount at a quiescent point -- "
            f"see api/main.py::_restore_tracking_db and config.py::TRACKING_DB_BACKUP_PATH."
        )
    # `_open` + `_ensure_schema`, not `_connect`: `_connect` would apply the same DDL
    # and throw away the list of what it added, so a boot could never report a
    # migration. `with conn` commits, then contextlib.closing closes: committed AND
    # closed before this returns, which is the precondition for copying the file.
    with contextlib.closing(_open()) as conn, conn:
        return _ensure_schema(conn)


def _parse_utc(value: str) -> datetime:
    """ISO timestamp -> aware UTC datetime. Naive values are UTC (that is how both
    commence_time and snapshotted_at are written)."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_float(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _require_pre_kickoff(commence_time: str) -> None:
    """Reject a game snapshot requested at or after its kickoff time."""
    kickoff = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    if kickoff <= datetime.now(timezone.utc):
        raise ValueError("Game predictions must be snapshotted before kickoff")


def record_game_predictions(games: list[dict]) -> int:
    """Snapshots only the games that are still pre-kickoff. A single
    already-kicked-off (or malformed) game in the batch used to raise for
    the whole call, silently dropping every other valid game in the same
    tick along with it -- and worse, aborting background_tracking_tick
    before it ever reached reconciliation/backfill, since this call has
    no try/except around it there. Skip just the invalid ones instead."""
    if not games:
        return 0
    valid_games = []
    for game in games:
        try:
            _require_pre_kickoff(game["commence_time"])
            valid_games.append(game)
        except ValueError:
            continue
    if not valid_games:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            game["game_id"], game["home_team"], game["away_team"], game["commence_time"], now,
            float(game["home_win_prob"]), float(game["away_win_prob"]),
            game.get("home_cover_prob"), game.get("away_cover_prob"),
            game.get("over_prob"), game.get("under_prob"),
            game.get("home_spread_line"), game.get("total_line"), game.get("season"), game.get("week"),
            game.get("predicted_margin"), game.get("sigma"), game.get("predicted_total"),
            game.get("total_sigma"), game.get("model_version"),
        )
        for game in valid_games
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob,
                 home_spread_line, total_line, season, week,
                 predicted_margin, sigma, predicted_total, total_sigma, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


def _present(*values) -> bool:
    """Whether every value is a real number rather than missing.

    `None or 0` is the wrong instrument for "this probability was never computed".
    It turns an absent input into a confident zero, and a zero is a *legible*
    probability: on a two-sided market `0.0 >= 0.0` resolves to the home side by the
    accident of `>=`. So a game snapshotted with a spread line but no cover
    probabilities -- which is what the odds feed yields whenever it has a spread
    without a matching market -- was recorded as a graded ATS call the model never
    made, in both directions:

        home 31-24, spread -3.5, cover probs None  -> ats_hit = 1  (fabricated hit)
        home 20-24, spread -3.5, cover probs None  -> ats_hit = 0  (fabricated miss)

    Half a market is equally unusable: `0.6` against `None` is not obviously the home
    side, but it is not a call either, and the same expression reports it as one.

    Presence is therefore required on *every* value the comparison depends on, and a
    missing one yields no grade rather than a wrong one. Identical fix in the CFB
    repo, where the same code shape produced the same defect.
    """
    return all(value is not None and not pd.isna(value) for value in values)


def _compute_hits(
    home_score: float, away_score: float, home_win_prob: float, away_win_prob: float,
    home_spread_line: float | None, home_cover_prob: float | None, away_cover_prob: float | None,
    total_line: float | None, over_prob: float | None, under_prob: float | None,
) -> tuple[int, int | None, int | None]:
    home_win = home_score > away_score
    predicted_home_win = home_win_prob >= away_win_prob
    moneyline_hit = int(predicted_home_win == home_win)

    # Grade a market only when the line *and* both sides of the probability pair are
    # present. See `_present` for why a missing probability must not be coerced to
    # zero rather than merely being unusual.
    ats_hit = None
    if _present(home_spread_line) and _present(home_cover_prob, away_cover_prob):
        home_margin = home_score - away_score
        home_covered = home_margin > home_spread_line
        predicted_home_cover = home_cover_prob >= away_cover_prob
        ats_hit = int(predicted_home_cover == home_covered)

    total_hit = None
    if _present(total_line) and _present(over_prob, under_prob):
        actual_total = home_score + away_score
        went_over = actual_total > total_line
        predicted_over = over_prob >= under_prob
        total_hit = int(predicted_over == went_over)

    return moneyline_hit, ats_hit, total_hit


def reconcile_game_predictions(results_df: pd.DataFrame) -> int:
    """Fill outcomes for existing unresolved game snapshots represented in results."""
    if results_df.empty:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        unresolved = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(results_df, on="game_id", how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            moneyline_hit, ats_hit, total_hit = _compute_hits(
                row["home_score"], row["away_score"], row["home_win_prob"], row["away_win_prob"],
                row.get("home_spread_line"), row.get("home_cover_prob"), row.get("away_cover_prob"),
                row.get("total_line"), row.get("over_prob"), row.get("under_prob"),
            )
            cursor = conn.execute(
                """
                UPDATE game_predictions
                SET resolved = 1, actual_home_score = ?, actual_away_score = ?,
                    moneyline_hit = ?, ats_hit = ?, total_hit = ?
                WHERE game_id = ? AND resolved = 0
                """,
                (int(row["home_score"]), int(row["away_score"]), moneyline_hit, ats_hit, total_hit, row["game_id"]),
            )
            resolved_count += cursor.rowcount
        return resolved_count


def get_untracked_game_ids(game_ids: list[str]) -> set[str]:
    """Which of these game_ids have no row in game_predictions at all yet
    -- never snapshotted before kickoff (distinct from "pending", which
    means snapshotted but not yet resolved). Used to backfill a game the
    tracker missed entirely, e.g. because the tracking loop wasn't
    running yet when it kicked off."""
    if not game_ids:
        return set()
    with contextlib.closing(_connect()) as conn:
        placeholders = ",".join("?" * len(game_ids))
        tracked = pd.read_sql(
            f"SELECT game_id FROM game_predictions WHERE game_id IN ({placeholders})", conn, params=game_ids
        )
    return set(game_ids) - set(tracked["game_id"])


def record_resolved_game_predictions(games: list[dict]) -> int:
    """Backfill a prediction snapshot for an already-finished game that was
    never tracked before kickoff -- records the prediction AND its known
    outcome in one shot, skipping the pre-kickoff requirement since
    there's no live pregame moment left to protect. Each game dict's
    prediction fields must already come from strictly pre-game
    information (the caller's job, e.g. the same _predict_game_from_models
    used for live upcoming games); this function only persists it."""
    if not games:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for game in games:
        moneyline_hit, ats_hit, total_hit = _compute_hits(
            game["actual_home_score"], game["actual_away_score"], game["home_win_prob"], game["away_win_prob"],
            game.get("home_spread_line"), game.get("home_cover_prob"), game.get("away_cover_prob"),
            game.get("total_line"), game.get("over_prob"), game.get("under_prob"),
        )
        rows.append((
            game["game_id"], game["home_team"], game["away_team"], game["commence_time"], now,
            float(game["home_win_prob"]), float(game["away_win_prob"]),
            game.get("home_cover_prob"), game.get("away_cover_prob"),
            game.get("over_prob"), game.get("under_prob"),
            game.get("home_spread_line"), game.get("total_line"), game.get("season"), game.get("week"),
            1, int(game["actual_home_score"]), int(game["actual_away_score"]), moneyline_hit, ats_hit, total_hit,
            game.get("predicted_margin"), game.get("sigma"), game.get("predicted_total"),
            game.get("total_sigma"), game.get("model_version"),
        ))
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob,
                 home_spread_line, total_line, season, week,
                 resolved, actual_home_score, actual_away_score, moneyline_hit, ats_hit, total_hit,
                 predicted_margin, sigma, predicted_total, total_sigma, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


def backfill_unresolved_games(schedules_module) -> int:
    """Targeted backfill: reconciles any still-unresolved snapshot whose
    season isn't CURRENT_SEASON, which the normal tick's
    fetch_current_season_partial() never looks at again. Takes the
    schedules module (not a season list) so it can look up whichever
    season each unresolved row actually belongs to."""
    with contextlib.closing(_connect()) as conn:
        unresolved = pd.read_sql("SELECT season FROM game_predictions WHERE resolved = 0", conn)
    if unresolved.empty:
        return 0

    seasons = sorted({int(s) for s in unresolved["season"].dropna()} | {pd.Timestamp.now().year})
    finished = schedules_module.load_training_data(seasons=seasons)
    if finished.empty:
        return 0
    return reconcile_game_predictions(finished[["game_id", "home_score", "away_score"]])


def get_track_record(current_week: int | None = None, season: int | None = None) -> dict:
    """Aggregate accuracy summary across every reconciled game and player
    prop -- not a per-game list (the frontend already has that in the game
    detail modal's own verdict section; this is the "how good is the model
    overall" view, same shape as PL_Predictor's Data Hub track record).

    `current_week`/`season` bound the `weekly` rows. The route passes both from
    `routes.current_season_and_week()`, which is the only place that knows the
    calendar; a caller that does not (facts.py, which reads two numbers off the
    headline) gets the newest week actually recorded. Either way the list runs
    from week 1 to that bound with no gaps -- see `_weekly_window`.
    """
    with contextlib.closing(_connect()) as conn, conn:
        resolved_games = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 1", conn)
        resolved_props = pd.read_sql("SELECT * FROM player_prop_predictions WHERE resolved = 1", conn)
    # Only picks made before kickoff count; rebuilt ones are reported apart.
    if not resolved_games.empty:
        rebuilt = resolved_games.apply(lambda r: _snapshotted_after_kickoff(r["snapshotted_at"], r["commence_time"]), axis=1).astype(bool)
    else:
        rebuilt = pd.Series(dtype=bool)
    n_rebuilt = int(rebuilt.sum())
    resolved_games = resolved_games[~rebuilt] if not resolved_games.empty else resolved_games
    summary = _summarize_games(resolved_games, current_week=current_week, season=season)
    return {"games": {**summary, "n_rebuilt": n_rebuilt}, "player_props": _summarize_player_props(resolved_props)}


def get_feed_predictions(now: datetime | None = None) -> list[dict]:
    """Frozen pre-game snapshots for games that have not started yet: the rows the trade hub may
    compare against Kalshi prices.

    Two exclusions, and they are the same question asked twice. `resolved = 0` drops games that
    have already been graded; `_snapshotted_after_kickoff` drops rows written at or after kickoff
    (PR #1's `rebuilt` rule), which is the leakage guard from spec 5a. A row that fails either is
    never served, and an unparseable timestamp is treated as rebuilt, so the filter fails closed.
    """
    now = now or datetime.now(timezone.utc)
    with contextlib.closing(_connect()) as conn:
        rows = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 0", conn)
    feed = []
    for _, row in rows.iterrows():
        if _snapshotted_after_kickoff(row["snapshotted_at"], row["commence_time"]):
            continue
        try:
            start = _parse_utc(row["commence_time"])
            snapshotted = _parse_utc(row["snapshotted_at"])
        except (TypeError, ValueError):
            continue
        if start <= now:
            continue
        feed.append({
            "game_id": row["game_id"],
            "season": None if pd.isna(row["season"]) else int(row["season"]),
            "week": None if pd.isna(row["week"]) else int(row["week"]),
            "home": row["home_team"],
            "away": row["away_team"],
            "start_utc": start.isoformat(),
            "p_home": float(row["home_win_prob"]),
            "margin_mu": _optional_float(row["predicted_margin"]),
            "sigma": _optional_float(row["sigma"]),
            "total_mu": _optional_float(row["predicted_total"]),
            "total_sigma": _optional_float(row["total_sigma"]),
            "home_spread_line": _optional_float(row["home_spread_line"]),
            "total_line": _optional_float(row["total_line"]),
            "model_version": row["model_version"] if isinstance(row["model_version"], str) else None,
            "snapshotted_at": snapshotted.isoformat(),
            # Always False here -- that is what the rebuilt filter above guarantees. The key stays
            # because the hub's parser rejects a truthy value, and False is the honest statement.
            "backfilled": False,
        })
    return sorted(feed, key=lambda r: (r["start_utc"], r["game_id"]))


def _calibration_buckets(pairs: list[tuple[float, int]], n_buckets: int) -> list[dict]:
    buckets = []
    for i in range(n_buckets):
        lo, hi = i / n_buckets, (i + 1) / n_buckets
        last = i == n_buckets - 1
        inside = [(p, y) for p, y in pairs if lo <= p < hi or (last and p == 1.0)]
        buckets.append({
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "n": len(inside),
            "mean_prob": sum(p for p, _ in inside) / len(inside) if inside else None,
            "hit_rate": sum(y for _, y in inside) / len(inside) if inside else None,
        })
    return buckets


# 4 buckets, not 10, ruled 2026-09-27.
#
# The trade hub gates an edge on `n >= calibration_min_n` in the bucket the edge's own probability
# falls in, and it does so for EVERY bucket before it will admit anything. So the settled contracts
# needed to open the gate are `n_buckets x calibration_min_n`: at 10 x 20 that is 200, while the
# hub's own reviewer renders a verdict at 100. The product had to be twice as strict about admitting
# an edge as it was about judging one, and on the real distribution (42 settled for CFB winner) the
# winner gate admitted nothing at all at 100 settled.
#
# At 4 x 20 that is 80, under the reviewer's bar, and the same measurement shows 4 buckets admitting
# 95% / 82% / 97% of edge mass at 100 settled where 10 admitted 0% / 46% / 66%.
#
# Fewer buckets also means each one is thicker, which matters because the hub compares each bucket's
# mean probability to its hit rate. Fewer, better-populated buckets are a more honest reliability
# record, not a coarser one.
CALIBRATION_N_BUCKETS = 4


def get_calibration(n_buckets: int = CALIBRATION_N_BUCKETS) -> dict:
    """Reliability buckets over resolved, genuinely pre-game snapshots: home win probability vs
    home won, home cover probability vs covered (at the recorded line), over probability vs went
    over. Ties and pushes are left out.

    Rebuilt rows are excluded by the same predicate the feed uses, so the buckets the hub
    calibrates against and the rows it prices come from the same population.

    The bucket count is the hub's gate arithmetic reaching this side: see CALIBRATION_N_BUCKETS. It
    stays a parameter so a caller can still ask for a finer view without changing the default.
    """
    with contextlib.closing(_connect()) as conn:
        rows = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 1", conn)
    winner: list[tuple[float, int]] = []
    spread: list[tuple[float, int]] = []
    total: list[tuple[float, int]] = []
    for _, row in rows.iterrows():
        if _snapshotted_after_kickoff(row["snapshotted_at"], row["commence_time"]):
            continue
        home, away = row["actual_home_score"], row["actual_away_score"]
        if pd.isna(home) or pd.isna(away):
            continue
        margin, points = float(home - away), float(home + away)
        if margin != 0:
            winner.append((float(row["home_win_prob"]), int(margin > 0)))
        line, prob = row["home_spread_line"], row["home_cover_prob"]
        if pd.notna(line) and pd.notna(prob) and margin != float(line):
            spread.append((float(prob), int(margin > float(line))))
        line, prob = row["total_line"], row["over_prob"]
        if pd.notna(line) and pd.notna(prob) and points != float(line):
            total.append((float(prob), int(points > float(line))))
    return {
        "n_buckets": n_buckets,
        "winner": _calibration_buckets(winner, n_buckets),
        "spread": _calibration_buckets(spread, n_buckets),
        "total": _calibration_buckets(total, n_buckets),
    }


# The three graded markets, as (grade column, hit count key, accuracy key). One list so a
# week row cannot grow an accuracy without also growing the count behind it -- a rate with
# no denominator is not a fact, and a *weekly* rate is where that mistake hides best: 5
# games, 3 ATS grades and 2 totals grades are three different denominators, and a chart
# that shows all three accuracies as bare percentages is asserting they share one.
_GRADED_MARKETS = (
    ("moneyline_hit", "n_moneyline", "pct_moneyline_correct"),
    ("ats_hit", "n_ats", "pct_ats_correct"),
    ("total_hit", "n_totals", "pct_totals_correct"),
)


def _grade(frame: pd.DataFrame, column: str) -> dict:
    """Hit rate and its count for one graded market over `frame`.

    A market is graded per game, and not always: ATS needs a real spread line plus both
    cover probabilities, totals needs a real total line plus both over/under
    probabilities (see `_present`). So the denominator is the graded subset, never the
    row count, and an ungraded market is `None` rather than 0.0 -- 0% is a legible claim
    that every game was missed, which is not what "never measured" means.
    """
    if column not in frame.columns:
        return {"n": 0, "pct": None}
    graded = frame[frame[column].notna()]
    return {"n": int(len(graded)), "pct": float(graded[column].mean()) if not graded.empty else None}


def _weekly_row(week: int, frame: pd.DataFrame) -> dict:
    """One week of the record.

    `n_games` is volume and each `n_*` is the denominator behind the accuracy beside it.
    They are deliberately separate keys. `TrackRecordPage.tsx` builds its bar width as
    `(n_games / max_games) * pct_moneyline_correct`, which multiplies accuracy by a
    volume share: a 1-game perfect week then draws at 0.25 of a 4-game 50% week. The
    backend cannot fix that formula, but it can refuse to offer a single fused number
    for it to consume -- and `n_games` next to an unfused accuracy is what lets the
    page draw the two separately.

    `tracked` is `n_games > 0`, and a week with nothing in it still gets a row. See
    `_weekly_window`.
    """
    row = {"week": int(week), "n_games": int(len(frame)), "tracked": bool(len(frame))}
    for column, count_key, pct_key in _GRADED_MARKETS:
        graded = _grade(frame, column)
        row[count_key], row[pct_key] = graded["n"], graded["pct"]
    return row


def _untracked_week(week: int) -> dict:
    """A week the tracker holds nothing for. Present, marked, and empty of rates.

    `n_games` and the three `n_*` are 0 because 0 is the *true count* of nothing
    tracked. The rates are None because no rate was measured -- 0.0 would claim the
    model was wrong on every game in a week it never picked in.
    """
    return {
        "week": int(week), "n_games": 0, "tracked": False,
        "n_moneyline": 0, "pct_moneyline_correct": None,
        "n_ats": 0, "pct_ats_correct": None,
        "n_totals": 0, "pct_totals_correct": None,
    }


def _weekly_window(resolved: pd.DataFrame, current_week: int | None, season: int | None) -> list[dict]:
    """One row for every elapsed week of `season`, 1..`current_week`.

    Grouping the rows by week and emitting one row per group only ever draws the weeks
    the tracker happened to cover. A week with no picks then does not read as "not
    tracked" -- it reads as *not part of the season*, which is a much stronger claim and
    a false one, since a reader cannot tell a gap from an absence. So the weeks are
    enumerated from the calendar and the empty ones are filled in.

    `season` scopes the window, because week numbers restart every season: an
    unfiltered group-by collapses 2025 week 12 into 2026 week 12 and reports a 2025
    accuracy on this season's chart. Callers that know the calendar pass it; callers
    that do not (facts.py, tests) get the newest season and week actually present in the
    data, which is still a complete, gap-free list.
    """
    by_week = _week_groups(resolved, current_week, season)
    return [
        _weekly_row(week, by_week[week]) if week in by_week else _untracked_week(week)
        for week in range(1, _window_end(current_week, by_week) + 1)
    ]


def _week_groups(
    resolved: pd.DataFrame, current_week: int | None, season: int | None
) -> dict[int, pd.DataFrame]:
    """Week number -> that week's rows, scoped to one season.

    Week numbering restarts every season, so an unfiltered group-by collapses 2025
    week 12 into 2026 week 12 and reports a 2025 accuracy on this season's chart.
    Callers that know the calendar pass `season`; callers that do not (facts.py,
    tests) get the newest season present, which is still a complete, gap-free list.
    """
    if "season" in resolved.columns:
        seasons = resolved["season"].dropna()
        if season is None and not seasons.empty:
            season = int(seasons.max())
    if season is not None and "week" in resolved.columns:
        scoped = resolved[resolved["season"] == season]
    else:
        # Nothing identifies a season here (no rows at all, or a caller assembling a
        # frame by hand). Emit the window unfiltered rather than emitting nothing.
        scoped = resolved
    if "week" not in scoped.columns:
        return {}
    return {int(w): group for w, group in scoped.groupby("week") if pd.notna(w)}


def _point_forecast(frame: pd.DataFrame, column: str, actual: pd.Series) -> dict:
    """Mean absolute error and mean signed error of a points forecast, in points.

    `column` holds the predicted value and `actual` the realised one for the same
    rows, so the error is `predicted - actual` in both cases; only what `actual`
    *is* differs (a margin for `predicted_margin`, a total for `predicted_total`).

    **A NULL forecast is EXCLUDED, never read as 0.0.** Rows written before
    `predicted_total`/`predicted_margin` existed have no points forecast at all --
    every one of the 48 rows in the real tracking database -- and 0.0 is a perfectly
    legible points forecast. Coercing them to zero would put a fabricated
    `predicted_total = 0` error of "however many points the game scored" on every
    legacy row, permanently, and would make the live season's MAE look like a model
    that predicts near-zero football. A row with no forecast has no error, only a
    prediction.

    `signed_error` is the mean of `predicted - actual`: positive means the model
    systematically over-forecast, negative under. It is the repo's existing
    `mean_signed_error` on player props under a shorter name, same definition.
    """
    if column not in frame.columns:
        return {"n": 0, "mae": None, "signed_error": None}
    eligible = frame[frame[column].notna() & actual.notna()]
    if eligible.empty:
        return {"n": 0, "mae": None, "signed_error": None}
    error = eligible[column] - actual[eligible.index]
    return {
        "n": int(len(eligible)),
        "mae": float(error.abs().mean()),
        "signed_error": float(error.mean()),
    }


def _total_points(frame: pd.DataFrame) -> pd.Series:
    if "actual_home_score" not in frame.columns or "actual_away_score" not in frame.columns:
        return pd.Series(dtype="float64", index=frame.index)
    return frame["actual_home_score"] + frame["actual_away_score"]


def _point_spread(frame: pd.DataFrame) -> pd.Series:
    if "actual_home_score" not in frame.columns or "actual_away_score" not in frame.columns:
        return pd.Series(dtype="float64", index=frame.index)
    return frame["actual_home_score"] - frame["actual_away_score"]


# (payload key, forecast column, what the realised value is). Kept as a list so the
# overall block and every weekly row are computed by the same call and cannot drift.
_POINT_FORECASTS = (
    ("totals", "predicted_total", _total_points),
    ("margin", "predicted_margin", _point_spread),
)


# --- B5: the model against the market ---------------------------------------
#
# The closing spread is already stored (`home_spread_line`, nflverse convention:
# the home team's expected margin, so a POSITIVE value means home is favoured;
# home covers when the real margin exceeds it). Turning a line into a probability
# needs one assumption: how wide a game's final margin typically is, in points.
# That is σ_league, and it is the only invented number in this block, which is why
# it is a named constant with a test asserting its value rather than an inline
# 13.5 in three places.
#
# THE SIGN, and it is the fact that makes the whole block correct: this column is
# ALREADY NEGATED relative to how a book words a line. A book prints "BAL -3.5" for
# a game where Baltimore is favoured by 3.5, and nflverse (and therefore this
# column) records the same game as +3.5, the home team's expected margin. So the
# spec's `Φ(spread / σ_league)` -- written in book convention, where a positive
# number is points the favourite gives -- becomes `Φ(-spread / σ_league)` here, and
# the negation happens ONCE, at the conversion in `implied_home_cover_prob`. Every
# other line in this block reads the column as the home team's expected margin and
# must not negate it again. `api/facts.py::_spread_line` negates it a second time,
# but for the opposite reason: the site words a favourite as giving points.
#
# Negating twice is invisible, which is the whole danger: every implied probability
# comes out exactly mirrored around 0.5, so the numbers still look reasonable, and
# the disagreement cohort silently becomes the set of games where the model AGREED
# with the line. `tests/test_track_record_vs_market.py` pins the convention against
# `margin_to_probabilities` and `_compute_hits` -- the two functions that already
# encode it -- rather than restating it, because a test written from the same
# misreading is not a test.
#
# PROVENANCE: NFL 13.5, from the design spec
# `docs/superpowers/specs/2026-09-27-predicted-box-score-and-track-record-design.md`
# ("σ_league a single named, documented constant per sport (NFL 13.5, CFB 14.0,
# to be pinned in the plan and asserted in a test so it cannot drift silently").
# The spec chose the value; this repo pins it. CFB's 14.0 belongs to CFB_Predictor
# and is NOT set here -- a per-sport constant has to live in the repo serving that
# sport, or the two will drift apart.
#
# Changing it silently would reinterpret every implied probability, every edge and
# every disagreement in the record, and nothing else in the codebase would notice.
SIGMA_LEAGUE_NFL = 13.5

# The same disclaimers, in words, in the payload. The page has to be able to
# explain what it is showing; a number nobody can interpret is not a decision aid,
# and a tooltip nobody opens is not an explanation.
_VS_MARKET_METHOD = {
    "sigma_league_points": SIGMA_LEAGUE_NFL,
    "sigma_league_meaning": (
        "NFL final margin is treated as roughly Normal with a standard deviation of "
        f"{SIGMA_LEAGUE_NFL} points. That single number is what turns a closing spread "
        "into a probability."
    ),
    "implied_probability": (
        f"The closing line is a margin -- the home team's expected margin -- so the "
        f"probability the market is asserting for the home side to cover it is "
        f"Φ(-spread / {SIGMA_LEAGUE_NFL}), the Normal cumulative at the NEGATED line "
        f"divided by {SIGMA_LEAGUE_NFL}. The minus sign is not a preference: lines are "
        f"quoted here in the expected-margin convention, where a POSITIVE line means "
        f"the home team is favoured, and the home side covers by beating that line. A "
        f"line of {SIGMA_LEAGUE_NFL} points either way is a 50/50 cover, and a line "
        "favouring the home team is a probability below 50% for the home side to "
        "cover it -- exactly as it is below 50% that a team favoured by seven wins "
        "by more than seven."
    ),
    "edge": (
        "Edge is the model's cover probability minus the probability the closing line "
        "implies, in percentage points, positive when the model likes a side more than "
        "the price does. It measures disagreement with a price, not superiority: a "
        "closing line is the market's best estimate, so a well-calibrated model's average "
        "edge is near zero by design. A large average edge would mean one of the two is "
        "miscalibrated, not that the model is right."
    ),
    "disagreement": (
        "The disagreement cohort is the games where the model backed the side the line "
        "did not favour -- a pick against the price. Its hit rate is how often that side "
        "covered, over exactly those games. Pick'em lines and evenly split model "
        "probabilities are excluded, because neither has a side to disagree with. This is "
        "the number to read for a decision; the mean edge is a calibration check."
    ),
    "not_a_profit_claim": (
        "This is agreement with a price, not a profit claim. No figure here is a return, "
        "a yield, a stake or a cent. We do not publish profit or ROI figures, and this "
        "comparison does not become one by being labelled 'edge'."
    ),
}


def implied_home_cover_prob(spread: float, sigma_league: float = SIGMA_LEAGUE_NFL) -> float:
    """P(home covers) implied by a closing spread, under the league margin sigma.

    `spread` is `home_spread_line`: the home team's EXPECTED margin, positive when
    home is favoured, and the home side covers when the real margin exceeds it. So
    the threshold is the line itself and the quantity wanted is P(margin > spread),
    which under a zero-centred margin of width `sigma_league` is Φ(-spread / σ).
    The negation is applied here, once; every other reader of the column in this
    module reads it un-negated.

    The sign is the whole function, and getting it backwards is invisible: every
    implied probability comes out exactly mirrored around 0.5, so the numbers still
    look reasonable while the mean edge flips sign and the disagreement cohort
    becomes the set of games where the model AGREED with the line. The spec writes
    the same probability as `Φ(spread / σ)` in book convention, where a book prints
    "BAL -3.5" for the game this column stores as +3.5; the spec is right and this
    is where the two conventions meet.
    """
    return float(norm.cdf(-spread / sigma_league))


def _market_row(frame: pd.DataFrame) -> dict:
    """One game (or one week) of the model-vs-market comparison.

    Eligible means: a real closing spread, both cover probabilities, and a real
    margin to grade against. Each missing piece removes the game from the whole
    block rather than contributing a neutral observation -- a game with no line has
    no price to disagree with, and 0.0 edge would be a fabricated agreement.
    """
    summary = {
        "n": 0, "mean_implied_home_cover_prob": None, "mean_model_home_cover_prob": None,
        "mean_edge_points": None, "disagreement_n": 0, "disagreement_hit_rate": None,
        "games": [],
    }
    if frame is None or frame.empty:
        return summary
    eligible = frame[
        frame["home_spread_line"].notna()
        & frame["home_cover_prob"].notna()
        & frame["away_cover_prob"].notna()
        & _actual_home(frame).notna()
        & _actual_away(frame).notna()
    ]
    if eligible.empty:
        return summary

    implied = eligible["home_spread_line"].map(implied_home_cover_prob).astype(float)
    model = eligible["home_cover_prob"].astype(float)
    edge_points = (model - implied) * 100.0

    disagreeing = _disagreeing_games(eligible)
    summary.update({
        "n": int(len(eligible)),
        "mean_implied_home_cover_prob": float(implied.mean()),
        "mean_model_home_cover_prob": float(model.mean()),
        "mean_edge_points": float(edge_points.mean()),
        "disagreement_n": int(len(disagreeing)),
        "disagreement_hit_rate": (
            float(disagreeing["ats_hit"].mean()) if not disagreeing.empty else None
        ),
        "games": sorted(str(g) for g in disagreeing["game_id"]),
    })
    return summary


def _actual_home(frame: pd.DataFrame) -> pd.Series:
    if "actual_home_score" not in frame.columns:
        return pd.Series(dtype="float64", index=frame.index)
    return frame["actual_home_score"]


def _actual_away(frame: pd.DataFrame) -> pd.Series:
    if "actual_away_score" not in frame.columns:
        return pd.Series(dtype="float64", index=frame.index)
    return frame["actual_away_score"]


def _disagreeing_games(eligible: pd.DataFrame) -> pd.DataFrame:
    """Games where the model backed the side the closing line did not favour.

    `home_spread_line` is the home team's expected margin, so `line > 0` is the
    line favouring home -- the same reading as `margin_to_probabilities` and
    `_compute_hits`, and the same reading `implied_home_cover_prob` negates once at
    the boundary. Reading it the other way inverts the entire cohort: the games
    where the model took the seven-point dog disappear from it and the games where
    the model agreed with the line take their place.

    Three exclusions, each for a stated reason:

    - **Pick'em** (`home_spread_line == 0`): the line favours nobody, so there is no
      side to disagree with. Not agreement either -- absence of a position, not a
      position.
    - **Evenly split model** (`home_cover_prob == away_cover_prob`): the model is not
      leaning anywhere. `>=` would hand these to home on a coin-flip tie.
    - **Pushes** (final margin exactly equal to the line): nobody won and nobody lost.
      `get_calibration` already declines to grade these (`margin != float(line)`),
      and this follows it rather than re-deriving a verdict from the scores, which
      would call the away side a winner of a game nobody won.
    """
    if eligible.empty:
        return eligible.iloc[0:0]
    margin = _actual_home(eligible) - _actual_away(eligible)
    line = eligible["home_spread_line"].astype(float)
    home = eligible["home_cover_prob"].astype(float)
    away = eligible["away_cover_prob"].astype(float)
    line_favours_home = line > 0
    model_favours_home = home > away
    return eligible[
        (line != 0)
        & (home != away)
        & (margin != line)
        & (line_favours_home != model_favours_home)
    ]


def _vs_market_weekly(
    resolved: pd.DataFrame, current_week: int | None, season: int | None
) -> dict:
    groups = _week_groups(resolved, current_week, season)
    return {
        "weekly": [
            {**_market_row(groups.get(week)), "week": int(week), "tracked": week in groups}
            for week in range(1, _window_end(current_week, groups) + 1)
        ]
    }


def _point_forecast_weekly(
    resolved: pd.DataFrame, current_week: int | None, season: int | None
) -> dict[str, dict]:
    """The same forecast block, but split by week, on the same window as `weekly`.

    Enumerated from the calendar rather than from the groups, for the same reason
    `weekly` is (B3): a week with no forecast must read as "not forecast", not
    vanish. And it is the per-week numbers that carry the season's shape -- a
    model 10 points high in week 1 and 10 low in week 3 averages to nothing and is
    wrong every single week.
    """
    groups = _week_groups(resolved, current_week, season)
    return {
        key: {
            "weekly": [
                _point_forecast_week(week, groups.get(week), column, realised)
                for week in range(1, _window_end(current_week, groups) + 1)
            ]
        }
        for key, column, realised in _POINT_FORECASTS
    }


def _window_end(current_week: int | None, groups: dict) -> int:
    if current_week is not None:
        return int(current_week)
    return max(groups, default=0)


def _point_forecast_week(week: int, frame: pd.DataFrame | None, column: str, realised) -> dict:
    summary = _point_forecast(frame if frame is not None else pd.DataFrame(), column, realised(frame) if frame is not None else pd.Series(dtype="float64"))
    return {"week": int(week), "tracked": summary["n"] > 0, **summary}


def _summarize_games(
    resolved: pd.DataFrame, current_week: int | None = None, season: int | None = None
) -> dict:
    graded = {column: _grade(resolved, column) for column, _, _ in _GRADED_MARKETS}
    forecasts = {
        key: {**_point_forecast(resolved, column, realised(resolved)), "weekly": []}
        for key, column, realised in _POINT_FORECASTS
    }
    for key, block in _point_forecast_weekly(resolved, current_week, season).items():
        forecasts[key]["weekly"] = block["weekly"]
    market = _market_row(resolved)
    return {
        "n_resolved": int(len(resolved)),
        # The headline counts too. A rate without its denominator is exactly the
        # ambiguity B1 shipped on this page, one level up.
        **{count: graded[column]["n"] for column, count, _ in _GRADED_MARKETS},
        **{pct: graded[column]["pct"] for column, _, pct in _GRADED_MARKETS},
        "weekly": _weekly_window(resolved, current_week, season),
        **forecasts,
        "vs_market": {
            **{k: v for k, v in market.items() if k != "games"},
            # The cohort's game list lives under its own key: `games` at the top
            # level of this block would read as every game compared, which is `n`.
            "disagreement": {
                "n": market["disagreement_n"],
                "hit_rate": market["disagreement_hit_rate"],
                "games": market["games"],
            },
            **_vs_market_weekly(resolved, current_week, season),
            "method": dict(_VS_MARKET_METHOD),
        },
    }


_YARDAGE_MARKETS = ("passing_yards", "rushing_yards", "receiving_yards", "receptions", "carries")

# Anytime-TD confidence buckets: predicted-probability ranges whose
# empirical hit rates calibrate the model's stated confidence. A
# prediction below 0.5 isn't a "call" and lands in no bucket.
_TD_CONFIDENCE_BUCKETS = (
    ("50-60%", 0.50, 0.60),
    ("60-70%", 0.60, 0.70),
    ("70%+", 0.70, 1.01),
)


def _summarize_player_props(resolved: pd.DataFrame) -> dict:
    result: dict[str, dict] = {}

    anytime_td = resolved[resolved["market"] == "anytime_td"]
    if anytime_td.empty:
        result["anytime_td"] = {"n_resolved": 0, "hit_rate_when_called": None, "brier_score": None,
                                "confidence_buckets": []}
    else:
        called = anytime_td[anytime_td["predicted_value"] >= 0.5]
        buckets = []
        for label, lo, hi in _TD_CONFIDENCE_BUCKETS:
            in_bucket = anytime_td[(anytime_td["predicted_value"] >= lo) & (anytime_td["predicted_value"] < hi)]
            buckets.append({
                "label": label,
                "n": int(len(in_bucket)),
                # actual_value is 1.0/0.0 for anytime_td, so the mean is the hit rate,
                # same convention as hit_rate_when_called above.
                "hit_rate": float(in_bucket["actual_value"].mean()) if not in_bucket.empty else None,
            })
        result["anytime_td"] = {
            "n_resolved": int(len(anytime_td)),
            "n_called": int(len(called)),
            "hit_rate_when_called": float(called["actual_value"].mean()) if not called.empty else None,
            "brier_score": float(((anytime_td["predicted_value"] - anytime_td["actual_value"]) ** 2).mean()),
            "confidence_buckets": buckets,
        }

    for market in _YARDAGE_MARKETS:
        rows = resolved[resolved["market"] == market]
        if rows.empty:
            result[market] = {"n_resolved": 0, "mean_absolute_error": None,
                              "mean_signed_error": None, "by_position": []}
        else:
            signed_errors = rows["predicted_value"] - rows["actual_value"]
            # Rows recorded before the position column existed (NULL)
            # stay in the overall metrics; only positioned rows group.
            positioned = rows[rows["position"].notna()] if "position" in rows.columns else rows.iloc[0:0]
            by_position = [
                {
                    "position": position,
                    "n_resolved": int(len(group)),
                    "mean_absolute_error": float((group["predicted_value"] - group["actual_value"]).abs().mean()),
                }
                for position, group in sorted(positioned.groupby("position"))
            ]
            result[market] = {
                "n_resolved": int(len(rows)),
                "mean_absolute_error": float(signed_errors.abs().mean()),
                # mean(predicted - actual): positive = systematic
                # overprediction, negative = systematic underprediction.
                "mean_signed_error": float(signed_errors.mean()),
                "by_position": by_position,
            }
    return result


def get_game_verdict(game_id: str) -> dict | None:
    """Return per-game post-match verdict (moneyline/ATS/totals). Returns None if
    the game is not tracked or not yet resolved."""
    with contextlib.closing(_connect()) as conn:
        rows = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = ?", conn, params=(game_id,))
    if rows.empty or not rows.iloc[0]["resolved"]:
        return None
    row = rows.iloc[0]

    predicted_home_win = row["home_win_prob"] >= row["away_win_prob"]
    actual_home_win = row["actual_home_score"] > row["actual_away_score"]
    verdict = {
        "game_id": game_id,
        "resolved": True,
        "moneyline": {
            "hit": bool(row["moneyline_hit"]),
            "predicted": "home_win" if predicted_home_win else "away_win",
            "actual": "home_win" if actual_home_win else "away_win",
        },
        "ats": None,
        "totals": None,
        "actual_home_score": int(row["actual_home_score"]),
        "actual_away_score": int(row["actual_away_score"]),
        "home_spread_line": float(row["home_spread_line"]) if pd.notna(row["home_spread_line"]) else None,
        "total_line": float(row["total_line"]) if pd.notna(row["total_line"]) else None,
    }
    # `hit` was written by `_compute_hits`, which already refuses to grade a market
    # whose probabilities are missing. Guard the re-derivation the same way, so a row
    # written by an older build cannot report a `predicted` side fabricated here: the
    # two must agree, and a disagreement would be a silent contradiction inside one
    # verdict object.
    if pd.notna(row["ats_hit"]) and _present(row["home_cover_prob"], row["away_cover_prob"]):
        predicted_home_cover = row["home_cover_prob"] >= row["away_cover_prob"]
        verdict["ats"] = {"hit": bool(row["ats_hit"]), "predicted": "home_cover" if predicted_home_cover else "away_cover"}
    if pd.notna(row["total_hit"]) and _present(row["over_prob"], row["under_prob"]):
        predicted_over = row["over_prob"] >= row["under_prob"]
        verdict["totals"] = {"hit": bool(row["total_hit"]), "predicted": "over" if predicted_over else "under"}
    return verdict


def record_player_prop_predictions(props: list[dict]) -> int:
    """Snapshot player props, retaining the first value for each prop market."""
    if not props:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (prop["game_id"], prop["player_id"], prop["player_name"], prop.get("position"),
         prop["market"], float(prop["predicted_value"]), now)
        for prop in props
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, position, market, predicted_value, snapshotted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


_MARKET_TO_STAT_COLUMN = {
    "anytime_td": None,
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "carries": "carries",
}


def reconcile_player_prop_predictions(player_stats_df: pd.DataFrame) -> int:
    """Fill outcomes for existing unresolved player-prop snapshots only."""
    if player_stats_df.empty:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        unresolved = pd.read_sql("SELECT * FROM player_prop_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(player_stats_df, on=["game_id", "player_id"], how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            if row["market"] == "anytime_td":
                actual = float(
                    (row.get("rushing_tds", 0) or 0)
                    + (row.get("receiving_tds", 0) or 0)
                    + (row.get("passing_tds", 0) or 0)
                    > 0
                )
            else:
                stat_col = _MARKET_TO_STAT_COLUMN[row["market"]]
                if stat_col not in row or pd.isna(row[stat_col]):
                    continue
                actual = float(row[stat_col])
            cursor = conn.execute(
                """
                UPDATE player_prop_predictions
                SET resolved = 1, actual_value = ?
                WHERE game_id = ? AND player_id = ? AND market = ? AND resolved = 0
                """,
                (actual, row["game_id"], row["player_id"], row["market"]),
            )
            resolved_count += cursor.rowcount
        return resolved_count


def _snapshotted_after_kickoff(snapshotted_at: str, commence_time: str) -> bool:
    def parse(value: str) -> datetime:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    try:
        return parse(snapshotted_at) >= parse(commence_time)
    except (TypeError, ValueError):
        # Can't prove it was made before kickoff, so it doesn't count.
        return True


def get_predictions_for_week(season: int, week: int, games_df: pd.DataFrame) -> list[dict]:
    """Return prediction status for all games in a given week.

    For each game in games_df, returns a dict with:
    - game_id: the game identifier
    - status: "untracked" (not snapshotted), "pending" (snapshotted but not resolved), or "resolved" (reconciled with final score)
    - home_win_prob/away_win_prob: prediction probabilities (only for tracked games)
    - verdict: full post-match verdict (only for resolved games)
    """
    if games_df.empty:
        return []
    game_ids = tuple(games_df["game_id"])
    placeholders = ",".join("?" * len(game_ids))
    with contextlib.closing(_connect()) as conn:
        tracked = pd.read_sql(
            f"SELECT * FROM game_predictions WHERE game_id IN ({placeholders})", conn, params=game_ids
        )
    tracked_by_id = {row["game_id"]: row for _, row in tracked.iterrows()}

    results = []
    for _, game in games_df.iterrows():
        row = tracked_by_id.get(game["game_id"])
        if row is None:
            results.append({"game_id": game["game_id"], "status": "untracked", "verdict": None})
            continue
        resolved = bool(row["resolved"])
        results.append({
            "game_id": game["game_id"],
            "status": "resolved" if resolved else "pending",
            # Snapshotted at or after kickoff means rebuilt after the fact
            # (record_resolved_game_predictions): shown, never counted.
            "rebuilt": _snapshotted_after_kickoff(row["snapshotted_at"], row["commence_time"]),
            "home_win_prob": row["home_win_prob"],
            "away_win_prob": row["away_win_prob"],
            "verdict": get_game_verdict(game["game_id"]) if resolved else None,
        })
    return results

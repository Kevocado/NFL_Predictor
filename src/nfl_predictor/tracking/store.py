"""SQLite persistence for the live prediction track record.

Snapshots each game's core-market predictions and each tracked player prop
before kickoff, then reconciles them against actual results once games are
played. Snapshot rows are immutable (``INSERT OR IGNORE``); reconciliation
only fills outcome columns on existing, unresolved rows.
"""

from __future__ import annotations

import contextlib
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import norm

from ..config import CACHE_DIR, TRACKING_DB_PATH
from ..features import player_usage


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

    # The QB passing-TD call is an over/under on a line, so a bare
    # `predicted_value` cannot express it: the value is the projection (mu),
    # and the call is the line plus the side. All four are recorded so the track
    # record grades against the line that was actually called rather than
    # re-deriving one from a mu that may since have moved.
    #
    # `line_source` is stored rather than assumed: the line is derived from the
    # projection, so it is a MODEL line and not an edge claim. If a real book
    # line is ever wired in it replaces `line` and sets this to something else,
    # with no change to any other part of the record.
    for column, sql_type in _PASSING_TD_PROP_COLUMNS:
        if column not in prop_cols:
            conn.execute(f"ALTER TABLE player_prop_predictions ADD COLUMN {column} {sql_type}")
            added.append(f"player_prop_predictions.{column}")

    # Which DEFINITION of `anytime_td` produced this row's `actual_value`. See
    # `LABEL_VERSION_COLUMN` for why the column exists and why it is NOT backfilled.
    if LABEL_VERSION_COLUMN not in prop_cols:
        conn.execute(
            f"ALTER TABLE player_prop_predictions ADD COLUMN {LABEL_VERSION_COLUMN} INTEGER"
        )
        added.append(f"player_prop_predictions.{LABEL_VERSION_COLUMN}")

    # Forward-test columns: the book's line as snapshotted, the model's
    # probability and edge against it, and the closing line for CLV. ALTER TABLE
    # again, so rows written before the forward test keep their NULLs and stay
    # readable -- they simply are not forward-test rows.
    for column, sql_type in _FORWARD_PROP_COLUMNS:
        if column not in prop_cols:
            conn.execute(f"ALTER TABLE player_prop_predictions ADD COLUMN {column} {sql_type}")
            added.append(f"player_prop_predictions.{column}")
    return added


#: Columns the passing-TD call adds to `player_prop_predictions`, in the order
#: they are migrated. Declared once, beside the game-side list, so the migration
#: and the writer cannot grow a column in one place and forget it in the other.
#:
#: **All five are now filled in production.** `routes.background_tracking_tick`
#: snapshots `market="anytime_td"`, every market in `POSITION_MARKETS`, and
#: `market="passing_tds"` via `routes._passing_td_prop_row`, and `record_player_
#: prop_predictions` round-trips all five for it. They stayed through the era when
#: nothing wrote the market (`tracking/qb_passing_td_record.py` was deleted in PR
#: #31 for reporting rows nothing wrote) precisely so that wiring the writer up
#: would need no migration.
#:
#: `line_source` is stored rather than assumed, and it is the reason the line can
#: never be misread as a price: the line is derived from the model's own
#: projection (`models/qb_passing_td.model_line`), so it is a MODEL line and not a
#: sportsbook line -- there is no book here to have an edge against.
_PASSING_TD_PROP_COLUMNS = (
    ("line", "REAL"),
    ("line_source", "TEXT"),
    ("side", "TEXT"),
    ("mu", "REAL"),
    ("call_prob", "REAL"),
)

#: Columns the forward test adds to `player_prop_predictions`, in migration
#: order. Declared once beside the passing-TD list so the migration and the
#: writer cannot grow a column in one place and forget it in the other.
#:
#: `line_at_snapshot` is deliberately NOT the existing `line` column. `line` is
#: the MODEL's line for a projection call; the forward test grades against the
#: BOOK's line, which is a different number with a different meaning, and
#: overwriting one with the other would corrupt existing rows.
_FORWARD_PROP_COLUMNS = (
    ("line_at_snapshot", "REAL"),
    ("odds_at_snapshot", "REAL"),
    ("model_p_over", "REAL"),
    ("edge_vs_breakeven", "REAL"),
    ("closing_line", "REAL"),
    ("clv", "REAL"),
    ("hit", "INTEGER"),
)

#: The `anytime_td` DEFINITION that graded a row, stamped at resolve time from
#: `player_usage.ANYTIME_TD_LABEL_VERSION`.
#:
#: `anytime_td_actual` changed on 2026-10-01 (v1 `rushing + receiving + passing
#: > 0`, v2 `rushing + receiving > 0`), and the grader changed with it. Every row
#: resolved before that change was scored against v1 and every row resolved after
#: it against v2 -- and because a recorded pick is immutable, no re-resolution can
#: ever move the old rows across. Without this column the two definitions were
#: folded into one hit rate and one Brier score, silently, with nothing in the
#: payload to say so. `_prop_markets` reads it and never mixes versions.
#:
#: **Named for the label it versions, not `label_version`,** because the prop table
#: carries several markets and only `anytime_td`'s truth has ever changed
#: definition. A yardage market's `actual_value` is a raw nflverse stat column and
#: carries no version; stamping a generic column on every row would put a number
#: on rows it says nothing about.
#:
#: **NULL is a real, third value and means "resolved before this column existed".**
#: It is never read as the current version. It cannot be backfilled from first
#: principles: the table records `snapshotted_at` (when the pick was MADE) but
#: never when it was GRADED, and grading happens at or after kickoff -- so a pick
#: snapshotted on 2026-09-28 may well have been resolved after the definition
#: changed on 2026-10-01. `snapshotted_at` is not a sound proxy for it, and
#: guessing would be the same silent misattribution in a different direction.
#: Rows that migrate in with NULL are reported in their own `by_label_version`
#: bucket under `label_version: null`, counted and visible, and excluded from the
#: current-version headline. Defaulting them to 1 was rejected: the definition
#: changed *after* this column was added, so the pre-existing rows are not
#: uniformly v1, and saying so would understate v2 by exactly the number of rows
#: graded since 2026-10-01.
LABEL_VERSION_COLUMN = "anytime_td_label_version"


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

    **The headline counts every counted pick, whenever it was made, and `pre_kickoff` beside it is
    the subset made before kickoff.** This reverses the pre-2026-10-01 rule, under which the
    headline was the pre-kickoff record and a pick recorded after its own kickoff never counted
    toward it. Kevin's reason, and it is the right one: the models are re-run constantly, so
    under the old rule a re-run on an already-played game stopped counting and the record
    emptied out on every model change.

    Three consequences, stated so a reader is never misled about what the headline is:

    - **A recorded pick is never re-scored.** One counted pick per (game, market) -- the EARLIEST
      recorded one (`_counted_picks`). A later rerun is kept in the table as history; it neither
      displaces the counted pick nor counts a second time, because re-running until the model is
      right would otherwise be free.
    - **`pre_kickoff` is the honest read of live performance**, and it is the figure to quote for
      what the model would have done on the night. It is the same summariser over the counted
      picks whose own timestamps prove they were made before kickoff, so its `n` is the exact size
      of that subset rather than a figure reconciled by hand.
    - **Counting a late pick is a known, accepted cost.** A model fitted on data including the
      result can look better than it would have on the night. That is why the two figures are
      published together and why `n_rebuilt` is still reported: the difference between the
      headline and `pre_kickoff` is exactly the size of the inflation.

    `n_rebuilt` is RETAINED and still counts counted picks made at or after their own kickoff. It
    used to mean "excluded from the headline"; it no longer does, and
    `n_resolved == pre_kickoff.n_resolved + n_rebuilt` holds on the counted frame. **That identity
    is about `games`, and about it only.** It does not transfer to `player_props`: `anytime_td`'s
    `n_resolved` there is one label version's rows out of the counted set, so for that market
    headline == pre_kickoff + n_rebuilt is false by design. See `_summarize_player_props`.

    `all_picks` is retained under its published name and still means "every counted pick" -- which
    is now the same population as the headline. It is not a rename to `pre_kickoff`; nothing a
    site reads today changes meaning or disappears. `per_pick` lists counted picks only, each with
    its own `made_before_kickoff` and `snapshotted_at`.
    """
    with contextlib.closing(_connect()) as conn, conn:
        resolved_games = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 1", conn)
        resolved_props = pd.read_sql("SELECT * FROM player_prop_predictions WHERE resolved = 1", conn)

    # One counted pick per game, the earliest recorded; the losers stay in the table as history.
    counted = _counted_picks(resolved_games)
    # The secondary figure: of the counted picks, the ones whose own timestamps prove they were
    # made before their game's kickoff. Its n is therefore exactly the size of that subset.
    pre_kickoff_games = counted[counted.apply(_made_before_kickoff_row, axis=1)] if not counted.empty else counted
    n_rebuilt = int(len(counted) - len(pre_kickoff_games))

    summary = _summarize_games(counted, current_week=current_week, season=season)
    return {
        "games": {
            **summary,
            # Still reported, still the same rows the secondary figure leaves out. It is the
            # reconciliation between the two figures: headline = pre_kickoff + n_rebuilt.
            "n_rebuilt": n_rebuilt,
            "pre_kickoff": _summarize_games(
                pre_kickoff_games, current_week=current_week, season=season
            ),
            # Kept under its published name; it means every counted pick, as it always did.
            "all_picks": _all_picks_record(counted),
            "per_pick": _per_pick_rows(counted),
        },
        "player_props": _summarize_player_props(resolved_props),
    }


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

    **Unchanged by the 2026-10-01 track-record reversal, deliberately.** A bucket here is not a
    track record, it is a price: the hub compares a live probability against it, and a bucket fed
    with picks made after kickoff would be telling the hub that a probability is better calibrated
    than it is, on evidence that could not have existed when the price was quoted. So the
    pre-kickoff guard stays exactly as strict here, and only the track record was reversed. If this
    ever should follow, that is a pricing change with its own review, not a side effect.

    Rebuilt rows are excluded by the same predicate the feed uses, so the buckets the hub
    calibrates against and the rows it prices come from the same population, and one counted pick
    per game (`_counted_picks`) so a re-keyed history cannot be graded twice.

    The bucket count is the hub's gate arithmetic reaching this side: see CALIBRATION_N_BUCKETS. It
    stays a parameter so a caller can still ask for a finer view without changing the default.
    """
    with contextlib.closing(_connect()) as conn:
        rows = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 1", conn)
    rows = _counted_picks(rows)
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


# The three graded markets, as (grade column, hit count key, accuracy key, probability
# pair the market is only a call if it has). One list so a week row cannot grow an
# accuracy without also growing the count behind it -- a rate with no denominator is not
# a fact, and a *weekly* rate is where that mistake hides best: 5 games, 3 ATS grades and
# 2 totals grades are three different denominators, and a chart that shows all three
# accuracies as bare percentages is asserting they share one.
#
# The fourth field is the other half of the same rule, kept here rather than written out
# a second time beside the count keys. `get_game_verdict` below already declines to
# report an ATS or totals market for a row whose probabilities are missing, because
# `_present` says that is not a call the model made; `_grade` excludes exactly those rows
# so the per-game view and the aggregate cannot give opposite answers about the same game.
# The moneyline carries an EMPTY pair, which is a different statement from "its pair is
# missing": the moneyline has no spread, `_compute_hits` grades it unconditionally, and
# `get_game_verdict` always reports it. Requiring a probability pair there would drop
# real grades to satisfy a rule that market has no part in.
#
# The data migration that would repair the stored flags is separate; until it runs,
# excluding the rows in `_grade` is what keeps the aggregate honest.
_GRADED_MARKETS = (
    ("moneyline_hit", "n_moneyline", "pct_moneyline_correct", ()),
    ("ats_hit", "n_ats", "pct_ats_correct", ("home_cover_prob", "away_cover_prob")),
    ("total_hit", "n_totals", "pct_totals_correct", ("over_prob", "under_prob")),
)


def _pair_present(frame: pd.DataFrame, *columns: str) -> pd.Series:
    """Row-wise `all(_present(...))`, for filtering a frame.

    `_present` is scalar; iterating it with Series arguments would make
    `pd.isna` return a Series and blow up on truthiness. This is the same rule
    expressed over columns.
    """
    mask = pd.Series(True, index=frame.index)
    for column in columns:
        mask &= frame[column].notna()
    return mask


def _grade(frame: pd.DataFrame, column: str, required: tuple[str, ...] = ()) -> dict:
    """Hit rate and its count for one graded market over `frame`.

    A market is graded per game, and not always: ATS needs a real spread line plus both
    cover probabilities, totals needs a real total line plus both over/under
    probabilities (see `_present`). So the denominator is the graded subset, never the
    row count, and an ungraded market is `None` rather than 0.0 -- 0% is a legible claim
    that every game was missed, which is not what "never measured" means.

    `required` is the probability pair `_GRADED_MARKETS` records for this market. A
    stored grade flag is not on its own enough: an older build wrote `ats_hit` for rows
    that had a spread line and no cover probabilities, and `get_game_verdict` refuses
    to report those rows, so counting them would put the per-game view and this
    aggregate in disagreement about the same game. Empty for the moneyline, which
    `_compute_hits` grades with no pair to check.

    This is the ONLY place the two halves meet, deliberately: the headline aggregate
    and every weekly row both go through it, so a week cannot apply a looser rule than
    the season total above it.
    """
    if column not in frame.columns:
        return {"n": 0, "pct": None}
    graded = frame[frame[column].notna()]
    if required and all(probability in frame.columns for probability in required):
        graded = graded[_pair_present(graded, *required)]
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
    for column, count_key, pct_key, required in _GRADED_MARKETS:
        graded = _grade(frame, column, required)
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


def _window_season(resolved: pd.DataFrame, season: int | None) -> int | None:
    """The season a week-numbered window is actually scoped to, resolved once.

    Week numbering restarts every season, so a window has to name one. Callers
    that know the calendar pass `season`; callers that do not (facts.py, tests)
    get the newest season present, which is still a complete, gap-free list.
    Returns None when nothing identifies a season -- no rows, or a caller
    assembling a frame by hand -- and every reader of this deals with that by
    emitting the window unfiltered rather than emitting nothing.

    Split out of `_week_groups` so a payload can LABEL the window it grouped by.
    A label recomputed from the caller's arguments could disagree with the
    grouping that actually happened, which is the one thing a label must not do.
    """
    if "season" in resolved.columns:
        seasons = resolved["season"].dropna()
        if season is None and not seasons.empty:
            season = int(seasons.max())
    return season


def _week_groups(
    resolved: pd.DataFrame, current_week: int | None, season: int | None
) -> dict[int, pd.DataFrame]:
    """Week number -> that week's rows, scoped to one season.

    Week numbering restarts every season, so an unfiltered group-by collapses 2025
    week 12 into 2026 week 12 and reports a 2025 accuracy on this season's chart.
    Which season it resolved to is `_window_season`, so a caller can say so.
    """
    season = _window_season(resolved, season)
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
        f"probability that the home side covers it is Φ(-spread / {SIGMA_LEAGUE_NFL}), "
        f"the Normal cumulative at the NEGATED line divided by {SIGMA_LEAGUE_NFL}. "
        f"That is the market's margin read through the league width stated below, not "
        f"a figure the market published: the line is the market's, the width is this "
        f"model's. The minus sign is not a preference: lines are quoted here in the "
        f"expected-margin convention, where a POSITIVE line means the home team is "
        f"favoured, and the home side covers by beating that line. Only a line of 0.0 "
        f"is a 50/50 cover. The line is a bar to clear, so the further it sits from "
        f"0.0 the further the probability sits from 50%: a line favouring the home "
        f"team is below 50% for the home side to cover it, which is the same "
        f"statement as it being under 50% likely that a team favoured by seven wins "
        f"by more than seven."
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
    "population": (
        "The headline figure covers every game the tracker holds a line, both cover "
        "probabilities and a final score for, in every season, because that is the "
        "question the page is answering. A game missing any one of those is left out "
        "of both this figure and the chart below rather than counted as a neutral "
        "observation, which is why a game with a line can still be missing from both. "
        "The chart covers one season's elapsed weeks only, so the two are over "
        "different populations on purpose. The block's scope states how many of the "
        "games the chart accounts for and how many fall outside it; the two add up to "
        "the headline's count, so no game is in one and silently missing from the other."
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


def _market_eligible(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of `frame` that can be compared with a price at all.

    A real closing spread, both cover probabilities, and a real margin to grade
    against. Each missing piece removes the game from the whole block rather than
    contributing a neutral observation -- a game with no line has no price to
    disagree with, and 0.0 edge would be a fabricated agreement. One predicate,
    read by the summary and by the scope reconciliation, so the two cannot
    disagree about which games were compared.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame[
        frame["home_spread_line"].notna()
        & frame["home_cover_prob"].notna()
        & frame["away_cover_prob"].notna()
        & _actual_home(frame).notna()
        & _actual_away(frame).notna()
    ]


def _market_row(frame: pd.DataFrame) -> dict:
    """One game (or one week) of the model-vs-market comparison.

    Eligible means: see `_market_eligible`, which is the same rule this applies
    and the scope reconciliation counts with.
    """
    summary = {
        "n": 0, "mean_implied_home_cover_prob": None, "mean_model_home_cover_prob": None,
        "mean_edge_points": None, "disagreement_n": 0, "disagreement_hit_rate": None,
        "games": [],
    }
    if frame is None or frame.empty:
        return summary
    eligible = _market_eligible(frame)
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
            _vs_market_week(week, groups.get(week))
            for week in range(1, _window_end(current_week, groups) + 1)
        ]
    }


def _vs_market_week(week: int, frame: pd.DataFrame | None) -> dict:
    """One week of the model-vs-market comparison.

    `tracked` is `n > 0` -- games actually COMPARED with a price -- and not
    "the tracker holds rows for this week". A week whose only rows have no
    spread, no cover probabilities or no scores was never compared with the
    market, and B3 makes this key the page's visible "Not tracked" marker, so
    `week in groups` would tell a visitor a week was tracked when zero games
    were compared. Same rule as `_weekly_row` (`n_games > 0`) and `_point_forecast_week`
    (`n > 0`); the tracker holding a row and the tracker having something to
    compare are two different facts, and only one of them is this key.
    """
    summary = _market_row(frame if frame is not None else pd.DataFrame())
    return {"week": int(week), "tracked": summary["n"] > 0, **summary}


def _vs_market_scope(
    resolved: pd.DataFrame, current_week: int | None, season: int | None, weekly: list[dict]
) -> dict:
    """Say, in the payload, which games the headline covers and which the chart does.

    The headline and the `weekly` rows are computed over DIFFERENT populations and
    always have been -- the headline is the whole resolved record, the chart is one
    season bounded by the calendar, which is what the spec mandates for the chart.
    What is new in B5 is that a second block started doing it, and did it silently:
    a reader could see a headline drawn from two seasons sitting directly above a
    chart drawn from one, with a row at `week > current_week` counted in the
    headline and appearing in no weekly list, and nothing in the payload saying so.

    Two ways to repair that were defensible: scope the headline to the same season
    and window as the chart, or keep it on the whole record and say so. **The
    headline stays on the whole record**, because it sits directly beside
    `n_resolved`, `n_ats` and `pct_ats_correct`, which are all whole-record numbers:
    narrowing only `vs_market` would trade one incoherence for a worse one, with
    `vs_market.n` disagreeing with the ATS count two keys above it. A record page
    should answer "how has this model done against the market", and that is not one
    season's question.

    So the payload carries the reconciliation instead. `n_games_in_weekly` is the sum
    of the `n` on the weekly rows themselves rather than a second count of the same
    games, so it cannot drift from the chart it is describing, and
    `n_games_outside_weekly` is the difference -- the games a reader who adds up the
    chart cannot account for, stated as a number rather than left to be discovered.
    The identity `n_games_total == n_games_in_weekly + n_games_outside_weekly` is
    what makes the block auditable, and it is asserted in a test.
    """
    total = int(len(_market_eligible(resolved)))
    in_weekly = int(sum(row["n"] for row in weekly))
    return {
        "population": "all_seasons",
        "weekly_season": _window_season(resolved, season),
        "weekly_last_week": None if current_week is None else int(current_week),
        "n_games_total": total,
        "n_games_in_weekly": in_weekly,
        "n_games_outside_weekly": total - in_weekly,
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


def _counted_picks(resolved: pd.DataFrame) -> pd.DataFrame:
    """The rows that count toward the record: one per game, the EARLIEST recorded.

    The uniqueness key of a counted pick is (game, market). On the games side all three graded
    markets live as columns on one row, so the key reduces to the game and the three markets are
    read off the same counted row -- which is why `_per_pick_rows` still emits one row per
    (game, market) off it, and why the dedup must not collapse that row to a single market.

    `game_predictions` is keyed on `game_id` and both writers are `INSERT OR IGNORE`, so no writer
    in this repo can today put two rows on one game: the uniqueness is currently a property of the
    schema rather than of this function. It is implemented in the reader anyway, because a rule
    enforced only by a primary key stops being enforced the moment the key changes, and the
    failure when it does is invisible -- the accuracy simply improves, silently, and a rerun of
    the model gets to grade a second time.

    "Earliest" is by UTC instant (`_utc_instant`), so which row wins does not depend on how its
    timestamp happens to be spelled. An unparseable timestamp cannot be proven earliest, so it
    sorts last and never displaces a row carrying a real instant; where no row in a group has one,
    the first row encountered stands and `_made_before_kickoff` fails closed on it.

    Rows that lose here are NOT deleted. They stay in the table -- rule 1, recorded stays recorded
    -- and they are deliberately absent from `per_pick` as well, so that the list a reader tallies
    is the list that produced the headline. A non-counted rerun is in the table, in no figure, and
    in no list: that is what history is.
    """
    if resolved.empty:
        return resolved
    ordered = resolved.assign(_instant=resolved["snapshotted_at"].map(_utc_instant_or_none))
    ordered = ordered.sort_values("_instant", na_position="last", kind="stable", ignore_index=True)
    return ordered.drop_duplicates(subset=["game_id"], keep="first")


def _all_picks_record(resolved: pd.DataFrame) -> dict:
    """The same three accuracies as the headline, over EVERY counted pick.

    **Kept under its published name, and it means the same thing it always did: every pick.**
    The pre-reversal headline excluded picks recorded after their own kickoff, so this block was
    the wider figure. The reversal makes the headline the wider figure instead (see
    `get_track_record`), and this key is what a site that already reads it gets. It is kept rather
    than renamed so no consumer breaks, and it is kept rather than deleted so a consumer reading
    `all_picks` is not silently pointed at a subset it believes is the whole record.

    Its `n_resolved` therefore now equals the headline's rather than exceeding it. The
    pre-kickoff figure a reader wants is `pre_kickoff`.
    """
    if resolved.empty:
        return {"n_resolved": 0, "pct_moneyline_correct": None, "pct_ats_correct": None, "pct_totals_correct": None}
    graded = {column: _grade(resolved, column, required) for column, _, _, required in _GRADED_MARKETS}
    return {
        "n_resolved": int(len(resolved)),
        **{pct: graded[column]["pct"] for column, _, pct, _ in _GRADED_MARKETS},
    }


def _per_pick_rows(resolved: pd.DataFrame) -> list[dict]:
    """One row per counted (game, market) pick, hit and miss alike, never filtered.

    Every row carries `made_before_kickoff` and `snapshotted_at`, which is the disclosure the
    reversal asks for: honesty moves from exclusion to being told. `made_before_kickoff` is
    DERIVED from the row's own two timestamps compared as UTC instants (`_made_before_kickoff`),
    not read from a stored flag, so there is nothing to be stale and nothing to backfill. The
    timestamp is beside it so a reader can check the claim rather than take it.

    `rebuilt` is retained on the row as the exact negation, because it was published and a site
    may read it. It is no longer the word for "not counted" -- under the reversal a rebuilt pick
    IS counted -- so `made_before_kickoff` is the field to read, and the two cannot disagree
    because one is computed from the other.

    This lists COUNTED picks only, so `len(per_pick) / 3` and the headline's `n_resolved` agree and
    a reader tallying the list arrives at the headline. A non-counted rerun is in the table as
    history and in neither figure.

    A market with no grade is omitted rather than emitted as a miss: an ungraded market is not a
    pick the model made, and listing it as a miss would be a fabricated failure.
    """
    rows: list[dict] = []
    for _, game in resolved.iterrows():
        made_before_kickoff = _made_before_kickoff(game["snapshotted_at"], game["commence_time"])
        for market, column in (("moneyline", "moneyline_hit"), ("ats", "ats_hit"), ("totals", "total_hit")):
            hit = game[column]
            if hit is None or pd.isna(hit):
                continue
            rows.append({
                "game_id": game["game_id"],
                "gameday": game["commence_time"],
                "market": market,
                "pick": _pick_words(game, market),
                "actual": _actual_words(game, market),
                "hit": bool(hit),
                # The disclosure the reversal asks for, derived and never stored.
                "made_before_kickoff": bool(made_before_kickoff),
                # The time the pick was made, so the flag above can be checked.
                "snapshotted_at": game["snapshotted_at"],
                # Retained name, now meaning "made at or after kickoff" and NOTHING more: this
                # pick is still counted. Kept so an existing reader does not break; use
                # `made_before_kickoff` for anything a reader is meant to understand.
                "rebuilt": not made_before_kickoff,
            })
    return rows


def _pick_words(game: pd.Series, market: str) -> str:
    """What the model backed, in words a reader can check against the result."""
    if market == "moneyline":
        return game["home_team"] if game["home_win_prob"] >= game["away_win_prob"] else game["away_team"]
    if market == "ats":
        return f"{game['home_team']} {game['home_spread_line']:+.1f}" if game["home_cover_prob"] >= game["away_cover_prob"] else f"{game['away_team']} {-game['home_spread_line']:+.1f}"
    return "Over" if game["over_prob"] >= game["under_prob"] else "Under"


def _actual_words(game: pd.Series, market: str) -> str:
    """What actually happened, in the same words."""
    if market == "moneyline":
        return game["home_team"] if game["actual_home_score"] > game["actual_away_score"] else game["away_team"]
    if market == "ats":
        margin = game["actual_home_score"] - game["actual_away_score"]
        return f"{game['home_team']} {game['home_spread_line']:+.1f}" if margin > game["home_spread_line"] else f"{game['away_team']} {-game['home_spread_line']:+.1f}"
    total = game["actual_home_score"] + game["actual_away_score"]
    return "Over" if total > game["total_line"] else "Under"


def _summarize_games(
    resolved: pd.DataFrame, current_week: int | None = None, season: int | None = None
) -> dict:
    graded = {column: _grade(resolved, column, required) for column, _, _, required in _GRADED_MARKETS}
    forecasts = {
        key: {**_point_forecast(resolved, column, realised(resolved)), "weekly": []}
        for key, column, realised in _POINT_FORECASTS
    }
    for key, block in _point_forecast_weekly(resolved, current_week, season).items():
        forecasts[key]["weekly"] = block["weekly"]
    market = _market_row(resolved)
    vs_market_weekly = _vs_market_weekly(resolved, current_week, season)
    return {
        "n_resolved": int(len(resolved)),
        # The headline counts too. A rate without its denominator is exactly the
        # ambiguity B1 shipped on this page, one level up.
        **{count: graded[column]["n"] for column, count, _, _ in _GRADED_MARKETS},
        **{pct: graded[column]["pct"] for column, _, pct, _ in _GRADED_MARKETS},
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
            # The headline is the whole record and the chart below it is one
            # season's elapsed weeks. The scope block reconciles the two, so the
            # difference is stated rather than discovered. See `_vs_market_scope`.
            "scope": _vs_market_scope(
                resolved, current_week, season, vs_market_weekly["weekly"]
            ),
            **vs_market_weekly,
            "method": dict(_VS_MARKET_METHOD),
        },
    }


#: Markets holding a yardage POINT ESTIMATE, graded by MAE and signed error.
#:
#: Forward picks are excluded by construction: they are stored under an
#: `fwd_`-prefixed market (`tracking.forward_tick.forward_market`) because their
#: `predicted_value` is the book's line, not a model estimate. A `fwd_` row here
#: would be graded as a point estimate and corrupt the published numbers.
_YARDAGE_MARKETS = ("passing_yards", "rushing_yards", "receiving_yards", "receptions", "carries")

# Anytime-TD confidence buckets: predicted-probability ranges whose
# empirical hit rates calibrate the model's stated confidence. A
# prediction below 0.5 isn't a "call" and lands in no bucket.
_TD_CONFIDENCE_BUCKETS = (
    ("50-60%", 0.50, 0.60),
    ("60-70%", 0.60, 0.70),
    ("70%+", 0.70, 1.01),
)


def _counted_prop_picks(resolved: pd.DataFrame) -> pd.DataFrame:
    """The prop rows that count: one per (game, player, market), the EARLIEST recorded.

    The prop table's primary key is already (game_id, player_id, market) and the writers are
    `INSERT OR IGNORE`, so today no writer can produce two rows for one pick -- exactly the
    situation `_counted_picks` handles for games, and the reason the selection is done here in
    the reader rather than relied upon from the schema. A rule enforced only by a primary key
    stops being enforced the moment the key changes, and a re-keyed or restored history is
    exactly the case where double-counting would be invisible.

    "Earliest" is compared as UTC instants, so a pick stamped with a `-05:00` offset sorts by when
    it actually happened rather than by how its string reads. See `_utc_instant`.
    """
    if resolved.empty:
        return resolved
    ordered = resolved.assign(_instant=resolved["snapshotted_at"].map(_utc_instant_or_none))
    # An unparseable timestamp cannot be proven to be the earliest, so it sorts LAST and never
    # displaces a row that carries a real instant. Where NO row in a group has one, the first
    # row encountered stands -- and `_made_before_kickoff` still fails closed on it below, so
    # the pick is counted but never labelled pre-kickoff.
    ordered = ordered.sort_values(
        "_instant", na_position="last", kind="stable", ignore_index=True
    )
    return ordered.drop_duplicates(subset=["game_id", "player_id", "market"], keep="first")


def _summarize_player_props(resolved: pd.DataFrame) -> dict:
    """Per-market metrics over graded props, in two figures.

    **The counted set is every recorded pick**, whenever it was made, and `pre_kickoff` beside it
    is the subset whose own timestamps prove they were made before kickoff. That is the reversal
    Kevin decided (see the 2026-10-01 spec): a re-run model must not make a past game stop
    counting, or the record empties out on every model change. `n_rebuilt` is kept, and still
    means "counted picks made after their own kickoff" -- the same rows the secondary figure
    leaves out, so the arithmetic reconciles.

    The prop table carries no commence_time, so it is joined to game_predictions on game_id (that
    table's primary key, so the join is many-to-one and safe). A row whose game is absent cannot
    be proven pre-kickoff, so it is counted and reported under `n_rebuilt` rather than excluded:
    the label fails closed, but rule 1 (recorded stays recorded) has no exception for a missing
    join.

    **The counted set is not the same thing as a market's `n_resolved`, and the difference is
    deliberate.** Nothing here drops a pick from the counted set, and `n_rebuilt` reconciles
    against `pre_kickoff` exactly as before. But `anytime_td`'s `n_resolved` counts only the rows
    graded under the CURRENT definition -- `_prop_markets` puts the rest into `by_label_version`
    rather than averaging them in, because their `actual_value` was computed from a different
    definition of the market. So `anytime_td.n_resolved + sum(non-current buckets)` is the counted
    set, while `anytime_td.n_resolved` on its own is a single-definition figure. This is the one
    market where the two differ: a yardage market's truth is a raw stat column with one definition
    and its `n_resolved` IS the counted set.
    """
    if resolved.empty:
        return {**_prop_markets(resolved), "n_rebuilt": 0, "pre_kickoff": _prop_markets(resolved)}

    with contextlib.closing(_connect()) as conn:
        kickoff_times = pd.read_sql("SELECT game_id, commence_time FROM game_predictions", conn)
    resolved = resolved.merge(kickoff_times, on="game_id", how="left")

    counted = _counted_prop_picks(resolved)
    pre_kickoff_resolved = counted[counted.apply(_made_before_kickoff_row, axis=1)]
    n_rebuilt = int(len(counted) - len(pre_kickoff_resolved))
    return {**_prop_markets(counted), "n_rebuilt": n_rebuilt,
            "pre_kickoff": _prop_markets(pre_kickoff_resolved)}




def _anytime_td_metrics(anytime_td: pd.DataFrame) -> dict:
    """n, hit rate, Brier and confidence buckets for exactly the rows handed in.

    One block of arithmetic, called once for the current-version headline and once per bucket in
    `_anytime_td_by_label_version`. A bucket is therefore literally the same computation over a
    subset, and cannot drift from the headline's -- which is what makes the reconciliation
    `sum(bucket n_resolved) == all counted anytime_td rows` true by construction.
    """
    if anytime_td.empty:
        return {"n_resolved": 0, "n_called": 0, "hit_rate_when_called": None,
                "brier_score": None, "confidence_buckets": []}
    called = anytime_td[anytime_td["predicted_value"] >= 0.5]
    buckets = []
    for label, lo, hi in _TD_CONFIDENCE_BUCKETS:
        in_bucket = anytime_td[(anytime_td["predicted_value"] >= lo) & (anytime_td["predicted_value"] < hi)]
        buckets.append({
            "label": label,
            "n": int(len(in_bucket)),
            # actual_value is 1.0/0.0 for anytime_td, so the mean is the hit rate,
            # same convention as hit_rate_when_called below.
            "hit_rate": float(in_bucket["actual_value"].mean()) if not in_bucket.empty else None,
        })
    return {
        "n_resolved": int(len(anytime_td)),
        "n_called": int(len(called)),
        "hit_rate_when_called": float(called["actual_value"].mean()) if not called.empty else None,
        "brier_score": float(((anytime_td["predicted_value"] - anytime_td["actual_value"]) ** 2).mean()),
        "confidence_buckets": buckets,
    }


def _label_version_sort_key(value):
    """Numeric versions first and ascending, anything else after, by text.

    `LABEL_VERSION_COLUMN` is `INTEGER`, so this only ever matters for a row nothing in
    `src/` wrote -- which is exactly when it must not raise. Grouping by whether the
    value is a real number and only then comparing means the two groups never compare
    across types, so a `2` and a `'v2'` on the same table cannot raise inside
    `get_track_record`.
    """
    if isinstance(value, bool):
        return (1, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value))


def _anytime_td_by_label_version(anytime_td_all: pd.DataFrame, current_version: int) -> list[dict]:
    """One bucket per distinct `LABEL_VERSION_COLUMN` value present, versions first, null last.

    Every counted `anytime_td` row appears in exactly one bucket, including the current version's
    -- so the buckets are a partition of the whole market rather than a list of leftovers, and
    their `n_resolved` values sum to the total number of counted `anytime_td` rows. That is what
    lets a consumer who wants the old record go and get it, without the headline having to claim
    to be it.

    The null bucket is real and is not folded into any version. A null means "graded before the
    column existed", which is genuinely unknown -- see `LABEL_VERSION_COLUMN` for why it cannot be
    backfilled -- and reading it as the current version is exactly the silent mixing this exists to
    stop. It is sorted last and reported with `label_version: null` so it is visible in JSON and
    cannot be mistaken for a version number.

    **Sorting is type-safe, and has to be.** The column is declared `INTEGER` and the grader writes
    an `int`, but SQLite does not enforce that: a hand-edited row or a future writer could put a
    string in it, and a plain `sorted()` over a mixed `{2, "v2"}` raises `TypeError: '<' not
    supported between instances of 'int' and 'str'` -- which would take down `get_track_record` for
    the whole site, not just this market. Numeric versions sort ascending first and anything else
    follows by its text, so a row this reader cannot interpret is still counted and still reported
    rather than crashing the summary or, worse, comparing loosely enough to fall into the current
    headline.
    """
    present = anytime_td_all[LABEL_VERSION_COLUMN].dropna().tolist()
    ordered = sorted(set(present), key=_label_version_sort_key)
    buckets = []
    for version in ordered:
        rows = anytime_td_all[anytime_td_all[LABEL_VERSION_COLUMN] == version]
        buckets.append({
            "label_version": version,
            "is_current": version == current_version,
            **_anytime_td_metrics(rows),
        })
    unlabelled = anytime_td_all[anytime_td_all[LABEL_VERSION_COLUMN].isna()]
    if not unlabelled.empty:
        buckets.append({
            "label_version": None,
            "is_current": False,
            "unknown_reason": (
                "resolved before the label version was stamped; the row carries no way to tell "
                "which anytime_td definition graded it, so it is reported apart rather than "
                "folded into a version"
            ),
            **_anytime_td_metrics(unlabelled),
        })
    return buckets


#: The market the QB passing-TD call is recorded under. Spelled as a literal
#: rather than imported from `models.qb_passing_td` so the tracking layer does not
#: pull the models package (and xgboost) in for one constant -- and
#: `_MARKET_TO_STAT_COLUMN` keys the same market the same way.
PASSING_TD_MARKET = "passing_tds"


def _line_is_half_point(line: float) -> bool:
    """Whether `line` ends in .5, which is what makes a push impossible against it.

    The actual count is an integer and the line a half point, so `actual == line`
    is unreachable and over/under is exhaustive over the outcomes.
    `models.qb_passing_td.model_line` only ever produces x.5, so this is a backstop
    on a STORED row rather than the rule that makes it true: a whole-number line
    means a push is reachable, and `_passing_td_metrics` reports such a row as
    ungradeable instead of guessing a side for it.
    """
    value = float(line)
    return math.isfinite(value) and abs(value - math.floor(value) - 0.5) < 1e-9


def _passing_td_hit(row: pd.Series) -> bool | None:
    """Whether the recorded call hit, or None when the row cannot be graded.

    `actual > line` for over and `actual < line` for under -- the stored line and
    side, never a re-derived one, so the record grades the call that was actually
    made. Three ways to be ungradeable, all of them defects rather than outcomes:
    no line, no actual, a side that is neither over nor under, or a line a push is
    reachable against. None is reported, never guessed and never raised: a raise
    here would take down `get_track_record` for the whole site over one bad row,
    which is the lesson `_label_version_sort_key` exists for.
    """
    line, side, actual = row.get("line"), row.get("side"), row.get("actual_value")
    if line is None or pd.isna(line) or actual is None or pd.isna(actual):
        return None
    if side not in ("over", "under") or not _line_is_half_point(line):
        return None
    return float(actual) > float(line) if side == "over" else float(actual) < float(line)


def _qb_pick_rows(resolved: pd.DataFrame) -> pd.DataFrame:
    """Every counted QB pick in the frame -- the denominator for the passing-TD record.

    A QB the tick recorded gets an `anytime_td` row whether or not a passing-TD
    call could be produced for it, so this frame is "the QBs we picked" read off
    the table rather than from a counter somebody has to remember to increment.
    That is what lets `_passing_td_metrics` name a QB whose call is missing instead
    of reporting a short record as if it were complete.

    A row whose `position` is NULL -- recorded before that column existed -- is not
    counted, because "is this a QB" is exactly what the NULL leaves unknown.
    """
    if resolved.empty or "position" not in resolved.columns:
        return resolved.iloc[0:0]
    return resolved[(resolved["market"] == "anytime_td") & (resolved["position"] == "QB")]


def _pairs(frame: pd.DataFrame) -> set[tuple]:
    """The `(game_id, player_id)` keys of a frame, as a set for set arithmetic."""
    if frame.empty:
        return set()
    return set(zip(frame["game_id"].tolist(), frame["player_id"].tolist()))


def _passing_td_metrics(rows: pd.DataFrame, qb_picks: pd.DataFrame) -> dict:
    """The passing-TD call's own record: how many calls hit, and how wrong they were.

    `rows` are the counted `passing_tds` rows (`_counted_prop_picks` has already
    kept the earliest per `(game_id, player_id, market)`), and `qb_picks` the
    counted QB `anytime_td` rows of the same frame. Both come from one frame, so
    `n_served` and `n_resolved` are measured over the same picks and the gap
    between them means something.

    **Absence is reported, not absorbed.** `n_served_without_a_graded_call` is the
    set difference, so a QB the tick picked with no passing-TD call written for it
    shows up as a named gap instead of silently not being in the record. The writer
    logs the same gap at WARNING (see `routes.background_tracking_tick`); this is
    the half a reader of the track record sees.

    `n_resolved` counts the rows this block was handed, and every one of them was
    written by the grader with the real `passing_tds` stat column as `actual_value`
    (`_MARKET_TO_STAT_COLUMN`) -- so every row here is graded against an outcome that
    exists. `n_ungradeable` is separate and non-zero only for a corrupt row; it is
    never folded into the hit rate, so the rate is always a rate over gradeable
    calls and its denominator is published beside it.

    **The line is a MODEL line.** It comes from `qb_passing_td.model_line(mu)` --
    the nearest half point to the model's own expectation -- so there is no
    sportsbook price in this block and nothing to have an edge against.
    `line_sources` publishes the provenance found on the stored rows as a list
    rather than a constant, so a real book line wired in later shows up here
    instead of being read as a model line. A row with no provenance at all reads
    `unstated`, which is the one value that must never be mistaken for a model line.
    """
    served, recorded = _pairs(qb_picks), _pairs(rows)
    graded: list[bool] = []
    per_pick: list[dict] = []
    ungradeable = 0
    for _, row in rows.iterrows():
        hit = _passing_td_hit(row)
        if hit is None:
            ungradeable += 1
            continue
        graded.append(hit)
        per_pick.append({
            "game_id": row["game_id"],
            "player_name": row.get("player_name"),
            "line": float(row["line"]),
            "line_source": row.get("line_source"),
            "side": row["side"],
            "mu": _optional_float(row.get("mu")),
            "call_prob": _optional_float(row.get("call_prob")),
            "actual_passing_tds": float(row["actual_value"]),
            "hit": hit,
        })

    by_side: dict[str, dict] = {}
    for entry in per_pick:
        bucket = by_side.setdefault(entry["side"], {"n": 0, "hits": 0})
        bucket["n"] += 1
        bucket["hits"] += int(entry["hit"])
    for bucket in by_side.values():
        bucket["hit_rate"] = bucket["hits"] / bucket["n"]

    # Brier on the binary outcome the call was about: p = the called side's
    # probability, y = whether it hit. Rows with no stored probability are left out
    # rather than scored at 0, which would be a fabricated confidence.
    brier = [(e["call_prob"] - float(e["hit"])) ** 2
             for e in per_pick if e["call_prob"] is not None]
    n = len(graded)
    return {
        "line_sources": sorted({"unstated" if e["line_source"] is None or pd.isna(e["line_source"])
                                else str(e["line_source"]) for e in per_pick}),
        "n_served": len(served),
        "n_resolved": int(len(rows)),
        "n_served_without_a_graded_call": len(served - recorded),
        "n_gradeable": n,
        "n_ungradeable": ungradeable,
        "n_called": n,
        "hit_rate_when_called": (sum(graded) / n) if n else None,
        "brier_score": (sum(brier) / len(brier)) if brier else None,
        "by_side": by_side,
        "per_pick": per_pick,
    }


def _prop_markets(resolved: pd.DataFrame) -> dict:
    """Every prop market's own metrics over exactly the rows handed in, and nothing else.

    Split from `_summarize_player_props` so the headline (every counted pick) and the secondary
    figure (the made-before-kickoff subset) are the SAME summariser over two frames, rather than
    two blocks of arithmetic that can drift apart. The secondary's `n` is then equal to the count
    of counted picks whose own timestamps prove they were made before kickoff, by construction
    rather than by agreement.

    **`anytime_td` is summarised per label version and never across versions.** The market's truth
    changed on 2026-10-01 (v1 included passing TDs, v2 does not), so a row's `actual_value` means
    one of two things depending on when it was graded, and recorded picks are immutable so the old
    rows can never be moved. The headline figures here are therefore the CURRENT definition's rows
    and only those, and they say so: `label_version` names the definition, `n_resolved` counts
    that definition's rows alone. Nothing is discarded -- `by_label_version` carries every other
    row in its own bucket, including the `label_version: null` bucket for rows resolved before the
    column existed -- so the buckets reconcile to the full counted set and a consumer can always
    get at a version it did not get as a headline.

    The yardage markets are untouched and carry no `label_version`: their `actual_value` is a raw
    nflverse stat column read straight off the box score, so there is no definition to have moved.

    **`passing_tds` is graded against that same kind of raw stat column** -- the real `passing_tds`
    count from the box score -- so it carries no `label_version` either, and it is NOT folded into
    `_YARDAGE_MARKETS`, which grades a yardage point estimate and has no line or side to grade. It
    is an over/under, so it gets its own block, and it carries both the line's provenance and the
    count of QB picks that never got a call recorded. See `_passing_td_metrics`.
    """
    result: dict[str, dict] = {}

    current_version = player_usage.ANYTIME_TD_LABEL_VERSION
    anytime_td_all = resolved[resolved["market"] == "anytime_td"] if not resolved.empty else resolved
    # A frame with no `LABEL_VERSION_COLUMN` at all -- an empty frame, or a caller
    # handing in columns the table no longer has -- is treated as "every row is
    # unlabelled", which puts them all in the null bucket and leaves the headline
    # empty. Reading a missing column as the current version would be the one
    # answer that could silently mix definitions.
    if anytime_td_all.empty or LABEL_VERSION_COLUMN not in anytime_td_all.columns:
        anytime_td = anytime_td_all.iloc[0:0]
        by_version = []
    else:
        anytime_td = anytime_td_all[anytime_td_all[LABEL_VERSION_COLUMN] == current_version]
        by_version = _anytime_td_by_label_version(anytime_td_all, current_version)

    result["anytime_td"] = {
        # Which definition these figures are about. A consumer must be able to
        # tell without inferring it from the code that produced them.
        "label_version": current_version,
        **_anytime_td_metrics(anytime_td),
        "by_label_version": by_version,
    }

    td_rows = resolved[resolved["market"] == PASSING_TD_MARKET] if not resolved.empty else resolved
    result[PASSING_TD_MARKET] = _passing_td_metrics(td_rows, _qb_pick_rows(resolved))

    for market in _YARDAGE_MARKETS:
        rows = resolved[resolved["market"] == market] if not resolved.empty else resolved
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
         prop["market"], float(prop["predicted_value"]), now,
         _optional_float(prop.get("line")), prop.get("line_source"),
         prop.get("side"), _optional_float(prop.get("mu")), _optional_float(prop.get("call_prob")),
         _optional_float(prop.get("line_at_snapshot")), _optional_float(prop.get("odds_at_snapshot")),
         _optional_float(prop.get("model_p_over")), _optional_float(prop.get("edge_vs_breakeven")))
        for prop in props
    ]
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, position, market, predicted_value,
                 snapshotted_at, line, line_source, side, mu, call_prob,
                 line_at_snapshot, odds_at_snapshot, model_p_over, edge_vs_breakeven)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cursor.rowcount


def record_closing_lines(updates: list[dict]) -> int:
    """Write each row's closing line, keyed by (game_id, player_id, market).

    Separate from the snapshot because the close is not knowable pre-kickoff:
    writing it at snapshot time would mean recording a price as though it were
    bettable. Rows that do not exist are skipped rather than inserted -- a close
    with no snapshot is not a pick.
    """
    if not updates:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        cursor = conn.executemany(
            """
            UPDATE player_prop_predictions SET closing_line = ?
            WHERE game_id = ? AND player_id = ? AND market = ?
            """,
            [(float(u["closing_line"]), u["game_id"], u["player_id"], u["market"])
             for u in updates if u.get("closing_line") is not None],
        )
        return cursor.rowcount


def _forward_verdict(side, line_at_snapshot, closing_line, actual_value):
    """(hit, clv) for a snapshot row, or (None, None) when it cannot be graded.

    CLV's sign follows the side taken: an over gains when the line closes
    higher, an under when it closes lower. Both are "the line moved my way",
    and reporting them with one shared sign would make half the track record
    read backwards.

    `clv` needs a closing line, `hit` needs a side and a snapshot line. A yardage
    projection with no over/under call is not a pick, so it gets neither.
    """
    # `_present`, not `is None`: these values come from `pd.read_sql`, so a SQL
    # NULL arrives as NaN, and every comparison against NaN is False. Testing
    # `is None` therefore graded a snapshot-less row as a MISS.
    if side not in ("over", "under") or not _present(line_at_snapshot, actual_value):
        return None, None

    line = float(line_at_snapshot)
    value = float(actual_value)
    covered = value > line if side == "over" else value < line

    clv = None
    if _present(closing_line):
        close = float(closing_line)
        clv = close - line if side == "over" else line - close
    return int(covered), clv


def _strip_forward_prefix(market: str) -> str:
    """`fwd_rushing_yards` -> `rushing_yards`; anything else unchanged.

    Imported lazily because `forward_tick` imports this module, so a top-level
    import here would be a cycle.
    """
    from .forward_tick import FORWARD_MARKET_PREFIX

    return market[len(FORWARD_MARKET_PREFIX):] \
        if market.startswith(FORWARD_MARKET_PREFIX) else market


_MARKET_TO_STAT_COLUMN = {
    "anytime_td": None,
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "carries": "carries",
    # The QB passing-TD call grades over/under against the ACTUAL passing TDs,
    # which is this column directly -- not the anytime-TD roll-up, which also
    # counts rushing and receiving touchdowns. So this market routes through the
    # ordinary stat-column path: `actual_value` becomes the real passing TD count,
    # ready for a line-and-side comparison against the recorded `line`, which
    # `_passing_td_metrics` does.
    #
    # **This market is now written, graded and reported.** `routes.background_
    # tracking_tick` emits it through `routes._passing_td_prop_row`, and
    # `_prop_markets` reports it. It was unreachable from serving until that
    # writer landed; the report that used to read these rows,
    # `tracking/qb_passing_td_record.py`, had already been DELETED in PR #31
    # because it read rows nothing wrote and swallowed every error, which is
    # exactly the "permanently empty record that looks like no picks yet" failure.
    # The rewrite keeps that module deleted -- the report is a block in
    # `_prop_markets` instead, where no bare `except` can hide it -- and
    # `tests/test_passing_td_record_absence.py` still pins the module's absence
    # and now also pins that the writer and the slot both exist.
    "passing_tds": "passing_tds",
}


def reconcile_player_prop_predictions(player_stats_df: pd.DataFrame) -> int:
    """Fill outcomes for existing unresolved player-prop snapshots only.

    A row snapshotted at or after its game's kickoff is a reconstruction, not
    a pick, and is never graded -- the same guard the games path applies in
    `get_track_record`/`get_feed_predictions`. The prop table carries no
    commence_time, so it is joined to game_predictions on game_id (that
    table's primary key, so the join is many-to-one and safe). An unparseable
    timestamp fails closed, and a prop row whose game is absent from
    game_predictions drops out of the inner join -- also fail closed.

    **The only columns written here are `resolved`, `actual_value`, `hit`, `clv`
    and `LABEL_VERSION_COLUMN`, and only on rows that are still `resolved = 0`.**
    That is the immutability rule: a recorded pick's prediction, snapshot line,
    odds, probability and edge are never rewritten, and a row already graded is
    never re-graded -- which is precisely why the label version has to be stamped
    here, in the same UPDATE that writes the verdict, instead of being backfilled
    over the history afterwards.

    `hit` and `clv` are derived from immutable inputs, so computing them at
    resolve time is safe; they are written here rather than by a second pass so
    a row's verdict and its outcome can never disagree. They stay NULL when the
    row is not an over/under call, which every existing yardage projection is.
    """
    if player_stats_df.empty:
        return 0
    with contextlib.closing(_connect()) as conn, conn:
        unresolved = pd.read_sql("SELECT * FROM player_prop_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        kickoff_times = pd.read_sql("SELECT game_id, commence_time FROM game_predictions", conn)
        merged = unresolved.merge(kickoff_times, on="game_id", how="inner")
        merged = merged.merge(player_stats_df, on=["game_id", "player_id"], how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            if _snapshotted_after_kickoff(row["snapshotted_at"], row["commence_time"]):
                continue
            if row["market"] == "anytime_td":
                # Rushing + receiving only; `passing_tds` was dropped from the
                # definition on 2026-10-01. The grader used to carry its own
                # inline sum of all three, so it would have kept resolving the
                # market against the OLD truth after the classifier's label
                # changed -- scoring every QB pick against a definition the model
                # was never fitted on. One definition, one function:
                # `player_usage.anytime_td_actual`.
                actual = player_usage.anytime_td_actual(
                    row.get("rushing_tds", 0), row.get("receiving_tds", 0)
                )
                # Stamp WHICH definition produced `actual_value`, in the same
                # UPDATE that writes it, so the two can never disagree. Read from
                # the constant at resolve time rather than written at record time:
                # the version that matters is the one the grader used, and a pick
                # recorded before the definition changed and graded after it was
                # still graded under the new one. Stamped only for `anytime_td`,
                # because no other prop market's truth is a definition that moves.
                label_version = player_usage.ANYTIME_TD_LABEL_VERSION
            else:
                # A forward pick is namespaced `fwd_<market>` so its
                # `predicted_value` (the book's line) is never read as a model
                # point estimate. That namespace made EVERY forward row
                # ungradeable: the lookup below raised KeyError on
                # 'fwd_passing_yards' and the reconciler died on the first one,
                # so no forward pick could ever produce a hit or a CLV. The
                # prefix is a storage detail -- the column behind
                # `fwd_rushing_yards` is still `rushing_yards` -- so it is
                # stripped here, at the one lookup that cares.
                stat_col = _MARKET_TO_STAT_COLUMN[_strip_forward_prefix(row["market"])]
                if stat_col not in row or pd.isna(row[stat_col]):
                    continue
                actual = float(row[stat_col])
                label_version = None

            hit, clv = _forward_verdict(
                row.get("side"), row.get("line_at_snapshot"),
                row.get("closing_line"), actual,
            )
            cursor = conn.execute(
                f"""
                UPDATE player_prop_predictions
                SET resolved = 1, actual_value = ?, {LABEL_VERSION_COLUMN} = ?,
                    hit = ?, clv = ?
                WHERE game_id = ? AND player_id = ? AND market = ? AND resolved = 0
                """,
                (actual, label_version, hit, clv,
                 row["game_id"], row["player_id"], row["market"]),
            )
            resolved_count += cursor.rowcount
        return resolved_count


def _utc_instant(value) -> datetime | None:
    """An ISO-8601 timestamp as an aware UTC instant, or None if it cannot be read as one.

    **The offset is honoured, and that is the whole point.** `2026-01-15T20:00:00` and
    `2026-01-14T19:00:00-05:00` are the same wall clock in different notations of the same
    evening, and a string comparison of the two against a kickoff written with an offset gives the
    OPPOSITE answer to the truth for a large part of any evening. So the parse goes through
    `fromisoformat` (which reads the offset) and then to UTC, and every comparison in the
    track-record path is between two of these -- never between two strings.

    Naive values are read as UTC, which is how both `commence_time` and `snapshotted_at` are
    written everywhere in this repo.
    """
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_instant_or_none(value) -> datetime | None:
    """`_utc_instant` under a name that says what it returns when the value is unusable."""
    return _utc_instant(value)


def _made_before_kickoff(snapshotted_at: str, commence_time: str) -> bool:
    """Whether this pick was made before this game's kickoff, from its own two timestamps.

    Derived, never stored: there is no `made_before_kickoff` column to be wrong, and nothing to
    backfill. Both sides are compared as UTC instants (`_utc_instant`), and a pick stamped exactly
    at kickoff is not before it.

    **Fails closed.** An unparseable timestamp on either side, or a missing kickoff, gives False:
    "cannot prove it was made before kickoff" is the only honest answer, and it is never a `true`
    the timestamps do not support. Note what failing closed does and does not mean now -- the
    label is withheld, but the pick is still COUNTED. Under the pre-reversal rule this function's
    answer decided whether a pick counted at all; it no longer does.
    """
    picked, kickoff = _utc_instant(snapshotted_at), _utc_instant(commence_time)
    if picked is None or kickoff is None:
        return False
    return picked < kickoff


def _made_before_kickoff_row(row: pd.Series) -> bool:
    """`_made_before_kickoff` over one row, for a frame filter."""
    return _made_before_kickoff(row["snapshotted_at"], row["commence_time"])


def _snapshotted_after_kickoff(snapshotted_at: str, commence_time: str) -> bool:
    """Whether a pick was recorded at or after its own kickoff -- i.e. rebuilt after the fact.

    Exactly the negation of `_made_before_kickoff`, and it still fails closed the same way. This
    is the predicate the feed and the calibration buckets keep using, where a post-kickoff row
    genuinely must not be served; it is no longer what decides whether a pick is counted.
    """
    return not _made_before_kickoff(snapshotted_at, commence_time)


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

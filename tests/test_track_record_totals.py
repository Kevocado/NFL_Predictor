"""B4 — total points and margin: the forecast that was being dropped on the floor.

`_predict_game_from_models` returns `predicted_margin` and `predicted_total` to
every caller (routes.py:185-186), and `margin_to_probabilities` turns the second
one into `over_prob`, so the model genuinely computes both on every call. The
track record could not show either of them, because nothing summarised them.

**The premise in the plan is half wrong and the half that matters is the data,
not the schema.** `predicted_total` and `predicted_margin` are NOT new columns:
they already exist on `game_predictions`, added for the Kalshi feed, and
`record_game_predictions` already writes them. What is true is that every row
written before that change has them NULL -- checked against the real tracking
database: 48 of 48 rows, both columns, all NULL. So "the tracker drops them" is
a statement about the history, not about the code path.

That is exactly why these tests are shaped the way they are. A NULL there means
*this pick was never forecast in points*, and a metric that reads it as 0.0 does
not report a small error -- it reports a false one, on every legacy row, forever.
The next session's first mistake will be to `fillna(0)` and see a number appear.

Also here: the migration, which the plan flags as running on a live SQLite file
on the Azure Files mount. It does not, and must not -- see the guard test.
"""
import contextlib
import sqlite3

import pandas as pd
import pytest

from nfl_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _resolve(game_id, week, *, home_score, away_score, **forecast):
    """A finished, genuinely pre-kickoff pick: snapshot it, then grade it.

    Any key left out of `forecast` is written as NULL, which is how a pre-feed
    legacy row is reproduced exactly.
    """
    game = {
        "game_id": game_id, "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "season": 2026, "week": week,
    } | forecast
    store.record_game_predictions([game])
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": game_id, "home_score": home_score, "away_score": away_score}])
    )


def _totals(current_week=1, season=2026):
    return store.get_track_record(current_week=current_week, season=season)["games"]["totals"]


def _margin(current_week=1, season=2026):
    return store.get_track_record(current_week=current_week, season=season)["games"]["margin"]


# --- the metrics, on a fixture whose answers are arithmetic ---------------

def test_total_mae_and_signed_error_on_a_known_fixture():
    """Three games. Actual totals 44, 44 and 40.

        predicted 50 -> error  +6
        predicted 45 -> error  +1
        predicted 30 -> error -10

    MAE   = (6 + 1 + 10) / 3 = 17/3
    bias  = (6 + 1 - 10) / 3 = -1

    Signed and absolute are deliberately different numbers. A test whose expected
    values happen to be equal cannot tell `error.abs().mean()` from `error.mean()`,
    and that confusion is the whole difference between "how far off" and "which way".
    """
    _resolve("g1", 1, home_score=24, away_score=20, predicted_total=50.0)
    _resolve("g2", 1, home_score=20, away_score=24, predicted_total=45.0)
    _resolve("g3", 1, home_score=30, away_score=10, predicted_total=30.0)

    totals = _totals()

    assert totals["n"] == 3
    assert totals["mae"] == pytest.approx(17 / 3)
    assert totals["signed_error"] == pytest.approx(-1.0)


def test_margin_mae_and_signed_error_on_a_known_fixture():
    """Same three games as actual margins +4, -4 and +20.

        predicted   4 -> error   0
        predicted   6 -> error  10
        predicted   2 -> error -18

    MAE  = (0 + 10 + 18) / 3 = 28/3
    bias = (0 + 10 - 18) / 3 = -8/3
    """
    _resolve("g1", 1, home_score=24, away_score=20, predicted_margin=4.0)
    _resolve("g2", 1, home_score=20, away_score=24, predicted_margin=6.0)
    _resolve("g3", 1, home_score=30, away_score=10, predicted_margin=2.0)

    margin = _margin()

    assert margin["n"] == 3
    assert margin["mae"] == pytest.approx(28 / 3)
    assert margin["signed_error"] == pytest.approx(-8 / 3)


# --- the one that matters most -------------------------------------------

def test_a_null_forecast_is_excluded_never_counted_as_zero():
    """A row written before the forecast columns existed is excluded, not zeroed.

    The fourth game below has no `predicted_total` at all -- exactly the shape of
    the 48 legacy rows in the real database. Read as 0.0 it would be a 44-point
    error on a game nobody ever forecast in points, and every headline number would
    move: n would read 4 instead of 3, MAE 4.25 instead of 17/3, bias -0.75
    instead of -1. All four are asserted, because any one of them changing means
    the row was counted.
    """
    # Each game gets a total forecast but no margin forecast, so the two blocks are
    # exercised independently: a fix that excluded NULL for one and not the other
    # would pass a fixture where every row has both.
    _resolve("g1", 1, home_score=24, away_score=20, predicted_total=50.0, predicted_margin=4.0)
    _resolve("g2", 1, home_score=20, away_score=24, predicted_total=45.0, predicted_margin=6.0)
    _resolve("g3", 1, home_score=30, away_score=10, predicted_total=30.0, predicted_margin=2.0)
    _resolve("legacy", 1, home_score=28, away_score=16)  # no predicted_total/margin

    totals = _totals()
    margin = _margin()

    assert totals["n"] == 3, "a row with no predicted_total must not enter the denominator"
    assert totals["mae"] == pytest.approx(17 / 3)
    assert totals["signed_error"] == pytest.approx(-1.0)
    assert margin["n"] == 3, "a row with no predicted_margin must not enter the denominator"
    assert margin["mae"] == pytest.approx(28 / 3)


def test_the_legacy_row_still_counts_in_the_graded_moneyline():
    """Excluding a NULL forecast must not cost the game its place in the record it
    does belong to. The pick was made pre-kickoff and graded; only the points
    forecast is missing."""
    _resolve("g1", 1, home_score=24, away_score=20, predicted_total=50.0)
    _resolve("legacy", 1, home_score=28, away_score=16)

    games = store.get_track_record(current_week=1, season=2026)["games"]

    assert games["n_resolved"] == 2
    assert games["n_moneyline"] == 2
    assert games["totals"]["n"] == 1


def test_a_metric_over_no_eligible_rows_is_null_not_zero():
    """Zero means 'measured and got none of them'. Nothing was measured here: 0.0
    points of average error on a season with no points forecast is a claim, and a
    false one."""
    for block in (_totals(), _margin()):
        assert block["n"] == 0
        assert block["mae"] is None
        assert block["signed_error"] is None
        # And the by-week view says the same thing, week by week, rather than
        # omitting the week outright: B3's rule applies to points forecasts too.
        assert [(row["week"], row["n"], row["mae"]) for row in block["weekly"]] == [(1, 0, None)]


def test_a_forecast_with_no_recorded_score_is_excluded_too():
    """The other end of the join: a forecast with no actual has no error, only a
    prediction. Counting it as zero error would be the mirror-image mistake."""
    store.record_game_predictions([{
        "game_id": "pending", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "season": 2026, "week": 1, "predicted_total": 50.0,
    }])

    block = _totals()

    assert block["n"] == 0
    assert block["mae"] is None
    assert block["signed_error"] is None


# --- by week ---------------------------------------------------------------

def test_totals_and_margin_are_reported_by_week():
    """The season's drift is the reason for the by-week view: a model whose total
    forecast is 10 points high in week 1 and 10 low in week 3 averages to nothing
    and is wrong every single week."""
    _resolve("w1a", 1, home_score=24, away_score=20, predicted_total=50.0)   # +6
    _resolve("w1b", 1, home_score=20, away_score=24, predicted_total=30.0)   # -14
    _resolve("w2a", 2, home_score=30, away_score=10, predicted_total=40.0)   # 0

    totals = _totals(current_week=3)
    by_week = {row["week"]: row for row in totals["weekly"]}

    assert [row["week"] for row in totals["weekly"]] == [1, 2, 3]
    assert by_week[1] == {"week": 1, "tracked": True, "n": 2, "mae": 10.0, "signed_error": -4.0}
    assert by_week[2] == {"week": 2, "tracked": True, "n": 1, "mae": 0.0, "signed_error": 0.0}
    assert by_week[3] == {"week": 3, "tracked": False, "n": 0, "mae": None, "signed_error": None}
    # And the overall figure is over the same three games the weeks hold: 3.
    # errors +6, -14, 0 -> MAE (6 + 14 + 0) / 3 = 20/3, bias (6 - 14) / 3 = -8/3.
    assert totals["n"] == 3
    assert totals["mae"] == pytest.approx(20 / 3)
    assert totals["signed_error"] == pytest.approx(-8 / 3)


def test_the_weekly_row_of_a_tracked_week_can_be_exactly_zero():
    """A perfectly forecast game is a real measurement of 0.0 error. This is the
    other side of the null rule: an eligible row with no error must not be confused
    with a row that was never eligible."""
    _resolve("w2a", 2, home_score=30, away_score=10, predicted_total=40.0)

    by_week = {row["week"]: row for row in _totals(current_week=2)["weekly"]}

    assert by_week[2]["n"] == 1
    assert by_week[2]["mae"] == 0.0
    assert by_week[2]["signed_error"] == 0.0


def test_a_rebuilt_pick_stays_out_of_the_points_metrics():
    """The headline is the pre-kickoff record. A pick backfilled after kickoff is
    shown, never judged, and that now covers the points forecasts too."""
    _resolve("pregame", 1, home_score=24, away_score=20, predicted_total=50.0)
    store.record_resolved_game_predictions([{
        "game_id": "rebuilt", "home_team": "NYJ", "away_team": "BUF",
        "commence_time": "2026-09-07T17:00:00+00:00",
        "home_win_prob": 0.4, "away_win_prob": 0.6,
        "actual_home_score": 10, "actual_away_score": 24,
        "season": 2026, "week": 1, "predicted_total": 50.0,
    }])

    games = store.get_track_record(current_week=1, season=2026)["games"]

    assert games["n_rebuilt"] == 1
    assert games["totals"]["n"] == 1, "a rebuilt forecast is not a pre-kickoff forecast"


# --- the migration ---------------------------------------------------------

_LEGACY_DDL = """CREATE TABLE game_predictions (
    game_id TEXT PRIMARY KEY, home_team TEXT NOT NULL, away_team TEXT NOT NULL,
    commence_time TEXT NOT NULL, snapshotted_at TEXT NOT NULL,
    home_win_prob REAL NOT NULL, away_win_prob REAL NOT NULL,
    home_cover_prob REAL, away_cover_prob REAL, over_prob REAL, under_prob REAL,
    resolved INTEGER NOT NULL DEFAULT 0, actual_home_score INTEGER, actual_away_score INTEGER,
    moneyline_hit INTEGER)"""


def test_the_migration_adds_the_forecast_columns_to_a_legacy_table():
    conn = sqlite3.connect(str(store.TRACKING_DB_PATH))
    conn.execute(_LEGACY_DDL)
    conn.commit()
    conn.close()

    added = store.migrate_tracking_db()

    with contextlib.closing(store._connect()) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(game_predictions)")}
    assert {"predicted_total", "predicted_margin", "week", "season", "ats_hit"} <= columns
    assert "predicted_total" in added and "predicted_margin" in added


def test_the_migrated_columns_stay_null_for_rows_that_predate_them():
    """A migration that backfilled 0.0 would be worse than no migration: 0.0 is a
    real points forecast, and it would put a fabricated error on every row in the
    live database. There is no value to backfill, so there is none."""
    conn = sqlite3.connect(str(store.TRACKING_DB_PATH))
    conn.execute(_LEGACY_DDL)
    conn.execute(
        "INSERT INTO game_predictions (game_id, home_team, away_team, commence_time,"
        " snapshotted_at, home_win_prob, away_win_prob, resolved, actual_home_score,"
        " actual_away_score) VALUES ('old', 'BAL', 'KC', '2026-09-13 20:20:00',"
        " '2026-09-12T15:00:00+00:00', 0.6, 0.4, 1, 24, 20)"
    )
    conn.commit()
    conn.close()

    store.migrate_tracking_db()

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions", conn).iloc[0]
    assert pd.isna(row["predicted_total"])
    assert pd.isna(row["predicted_margin"])


def test_migrating_is_idempotent():
    """It runs on every cold start, so a second run must add nothing and must not
    raise. A migration that is not idempotent eventually fails the tracker.

    The first call legitimately reports a fresh database's worth of columns -- the
    schema has never been written -- so the assertion is on the SECOND call, which
    is the one that has to find nothing left to do.
    """
    store.migrate_tracking_db()

    assert store.migrate_tracking_db() == []


def test_migrating_a_database_on_the_persistent_mount_is_refused(tmp_path, monkeypatch):
    """The trap this whole migration is wrapped in.

    SQLite does not work reliably over the Azure Files/SMB mount -- confirmed live
    as "database is locked" the moment TRACKING_DB_PATH pointed there. The repo
    already carries the fix (commit 458bed2): the live database stays on local
    ephemeral disk and is COPIED to and from the mount at a quiescent point, with
    no connection open at either end. So the migration must never run there.

    This asserts the refusal rather than the happy path, because the happy path is
    untestable here and the refusal is the part that is new.
    """
    from nfl_predictor import config

    mount = tmp_path / "cache"  # stands in for the Azure Files volume
    mount.mkdir()
    on_mount = mount / "tracking.db"
    monkeypatch.setattr(config, "CACHE_DIR", mount)
    monkeypatch.setattr(store, "CACHE_DIR", mount)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", on_mount)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", on_mount)

    with pytest.raises(store.TrackingDbOnPersistentMount, match="persistent"):
        store.migrate_tracking_db()

    # It left nothing behind: not even an empty database. A refusal that had already
    # created the file would have put the artifact on the mount that it is refusing
    # to touch, which is the failure this guard is for.
    assert not on_mount.exists()


def test_a_local_database_outside_the_mount_is_migrated_normally(tmp_path, monkeypatch):
    """The guard's other half: it must not refuse a legitimate local path, or the
    app simply never starts. `TRACKING_DB_PATH` is a sibling of the mount here, which
    is the real layout -- local data dir, separate mounted cache dir."""
    from nfl_predictor import config

    mount = tmp_path / "cache"
    mount.mkdir()
    local = tmp_path / "data" / "tracking.db"
    local.parent.mkdir()
    monkeypatch.setattr(config, "CACHE_DIR", mount)
    monkeypatch.setattr(store, "CACHE_DIR", mount)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", local)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", local)

    assert "predicted_total" in store.migrate_tracking_db()
    assert store.migrate_tracking_db() == []

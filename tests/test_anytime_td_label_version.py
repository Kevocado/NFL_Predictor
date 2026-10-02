"""tests/test_anytime_td_label_version.py -- picks graded under two different
definitions of one market are never averaged into one figure.

The defect
----------
`player_usage.anytime_td_actual` changed on 2026-10-01. v1 was
`rushing_tds + receiving_tds + passing_tds > 0`; v2 is `rushing_tds +
receiving_tds > 0`. The grader changed with it, so the `player_prop_predictions`
table holds rows whose `actual_value` means one of two things -- and because a
recorded pick is immutable and nothing re-resolves an already-graded row, the two
populations can never be merged or converted.

Before this change `_prop_markets` selected every resolved `anytime_td` row and
folded them into one `n_resolved`, one `hit_rate_when_called` and one
`brier_score`, with no version column anywhere on the row and nothing in the
payload to tell a consumer which definition produced the number it was looking at.
A QB's pre-2026-10-01 pick scored against a truth the classifier was never
fitted on, and it moved the headline of a market fitted on the new one.

The fix
-------
`player_prop_predictions.anytime_td_label_version` is stamped at resolve time from
`player_usage.ANYTIME_TD_LABEL_VERSION`, in the same UPDATE that writes
`actual_value` so the two cannot disagree. `_prop_markets` reports the CURRENT
definition as the headline and every other version -- including the
`label_version: null` bucket for rows resolved before the column existed -- in
`by_label_version`.

Why "report apart" and not "exclude the old ones"
-------------------------------------------------
`by_label_version` is a *partition*, not a list of leftovers: the buckets sum to
the whole counted `anytime_td` set. Dropping v1 rows from the payload entirely
would have been the smaller change and it was rejected, because the standing rule
is that every recorded pick counts. Exclusion is not available as a way to be
quiet about them either -- the v1 rows are a real record of what the model did
against the truth it was fitted on, and anyone asking for it must be able to get
it without going to the database. Bucketing keeps both properties: no two
definitions are ever summed, and no pick is invisible.

Why NULL is a bucket and not a backfilled 1
--------------------------------------------
It cannot be backfilled from first principles. The table records
`snapshotted_at` -- when the pick was MADE -- but never when it was GRADED, and
grading happens at or after kickoff. A pick snapshotted on 2026-09-28 can easily
have been graded after the definition changed on 2026-10-01, so `snapshotted_at`
is not a sound proxy and guessing would repeat the original bug in the opposite
direction. Defaulting the NULLs to 1 was rejected for the same reason: the
definition changed *after* this column was added, so the pre-existing rows are
not uniformly v1, and calling them v1 would understate v2 by exactly the number
graded since. `label_version: null` asserts only what is known.
"""
from __future__ import annotations

import contextlib

import pandas as pd
import pytest

from nfl_predictor.features import player_usage
from nfl_predictor.tracking import store

CURRENT = player_usage.ANYTIME_TD_LABEL_VERSION
OLD = CURRENT - 1


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _game(**overrides):
    game = {
        "game_id": "2025_01_BAL_KC", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
    }
    game.update(overrides)
    return game


def _prop(player_id, predicted=0.6):
    return {
        "game_id": "2025_01_BAL_KC", "player_id": player_id, "player_name": f"Player {player_id}",
        "position": "QB", "market": "anytime_td", "predicted_value": predicted,
    }


def _stats(*player_tds):
    """(player_id, rushing_tds, receiving_tds, passing_tds) per player."""
    return pd.DataFrame(
        [{"game_id": "2025_01_BAL_KC", "player_id": pid,
          "rushing_tds": ru, "receiving_tds": re, "passing_tds": pa}
         for pid, ru, re, pa in player_tds]
    )


def _as_graded_under(player_ids, version, actual=1.0):
    """Rewrite a resolved row into the state a grader of `version` would have left.

    Two columns, together, because that is what a row resolved before this change
    actually is: `actual_value` from the definition that was live at the time, and
    no version stamp at all. Written directly rather than by running the grader,
    because the grader only ever writes the CURRENT definition -- which is the
    entire defect, and the reason the old rows cannot be reproduced by running the
    code again. Passing `version=None` reproduces the pre-column rows exactly:
    the value is there, the stamp is not.
    """
    with contextlib.closing(store._connect()) as conn, conn:
        for pid in player_ids:
            conn.execute(
                f"UPDATE player_prop_predictions SET {store.LABEL_VERSION_COLUMN} = ?, "
                "actual_value = ? WHERE player_id = ? AND market = 'anytime_td'",
                (version, actual, pid),
            )


def _td(*pids):
    return store.get_track_record()["player_props"]["anytime_td"]


def _bucket(td, version):
    return next(b for b in td["by_label_version"] if b["label_version"] == version)


# --- the stamp ---------------------------------------------------------------


def test_resolving_an_anytime_td_pick_stamps_the_current_version():
    """The whole fix starts here: the version has to be written by the grader.

    Read from the constant rather than hardcoded, so bumping the definition in
    `features/player_usage.py` moves the stamp with it. Before this column existed
    nothing wrote a version anywhere, which is exactly how two definitions ended
    up in one aggregate.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("p1")])
    assert store.reconcile_player_prop_predictions(_stats(("p1", 1, 0, 0))) == 1

    with contextlib.closing(store._connect()) as conn:
        stamped = pd.read_sql(
            f"SELECT {store.LABEL_VERSION_COLUMN} AS v FROM player_prop_predictions "
            "WHERE player_id = 'p1'", conn,
        )["v"].iloc[0]

    assert stamped == CURRENT == player_usage.ANYTIME_TD_LABEL_VERSION


def test_the_stamp_reads_the_constant_at_resolve_time(monkeypatch):
    """Not a hardcoded 2. Move the definition and the stamp follows it.

    This is what makes `ANYTIME_TD_LABEL_VERSION` load-bearing in the grading path
    rather than decoration beside it: it is the single source for what the grader
    says it used. A third definition would be stamped 3, and everything graded
    before would fall out of the headline on its own.
    """
    monkeypatch.setattr(player_usage, "ANYTIME_TD_LABEL_VERSION", 3)
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("p1")])
    assert store.reconcile_player_prop_predictions(_stats(("p1", 1, 0, 0))) == 1

    with contextlib.closing(store._connect()) as conn:
        stamped = pd.read_sql(
            f"SELECT {store.LABEL_VERSION_COLUMN} AS v FROM player_prop_predictions "
            "WHERE player_id = 'p1'", conn,
        )["v"].iloc[0]

    assert stamped == 3


def test_a_resolved_row_is_never_restamped_by_a_second_pass(monkeypatch):
    """Immutability, stated as a test rather than trusted.

    `reconcile_player_prop_predictions` only selects `resolved = 0`, so a graded
    row keeps the version it was graded under forever -- even after the definition
    moves. Re-running the grader must not quietly re-date the record, and this is
    the assertion that would fail if the WHERE clause ever widened to `resolved IN
    (0, 1)`.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("p1")])
    assert store.reconcile_player_prop_predictions(_stats(("p1", 1, 0, 0))) == 1

    monkeypatch.setattr(player_usage, "ANYTIME_TD_LABEL_VERSION", 7)
    assert store.reconcile_player_prop_predictions(_stats(("p1", 1, 0, 0))) == 0

    with contextlib.closing(store._connect()) as conn:
        stamped = pd.read_sql(
            f"SELECT {store.LABEL_VERSION_COLUMN} AS v FROM player_prop_predictions "
            "WHERE player_id = 'p1'", conn,
        )["v"].iloc[0]
    assert stamped == CURRENT, "a second grader pass re-dated an immutable row"


def test_a_yardage_pick_is_not_given_a_label_version():
    """`anytime_td`'s truth is a definition that can change; a yardage pick's is a
    raw nflverse stat column read off the box score, and there is nothing to have
    moved. Stamping 2 on a `rushing_yards` row would claim a version for a quantity
    that has exactly one definition, which is a different kind of lie.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([
        {**_prop("p1"), "market": "rushing_yards", "predicted_value": 90.0},
    ])
    stats = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "player_id": "p1",
                           "rushing_yards": 80.0}])
    assert store.reconcile_player_prop_predictions(stats) == 1

    with contextlib.closing(store._connect()) as conn:
        stamped = pd.read_sql(
            f"SELECT {store.LABEL_VERSION_COLUMN} AS v FROM player_prop_predictions "
            "WHERE player_id = 'p1'", conn,
        )["v"].iloc[0]
    assert stamped is None or pd.isna(stamped), "a yardage row was stamped with a label version"


# --- the payload says which --------------------------------------------------


def test_the_payload_carries_the_label_version_it_is_about():
    """A consumer must be able to tell what it is looking at without reading code.

    `label_version` is on the `anytime_td` object itself, beside the numbers it
    describes, so it cannot be read against the wrong figure.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("p1")])
    store.reconcile_player_prop_predictions(_stats(("p1", 1, 0, 0)))

    td = _td()
    assert td["label_version"] == CURRENT
    assert td["n_resolved"] == 1


def test_the_yardage_markets_carry_no_label_version():
    """Their truth has one definition, so there is nothing to name. Absent rather
    than null: the key does not exist, which is the honest shape for 'not
    applicable' in a payload where a real version would be a number."""
    assert "label_version" not in store.get_track_record()["player_props"]["rushing_yards"]


# --- versions are never summed ------------------------------------------------


def _mixed_record():
    """Three rows on one market, graded under three different states.

    `v2_hit`    - graded now, and an anytime TD under the definition in force.
    `v1_miss`   - the row that makes the defect visible. A passing-TD-only game.
                  v1 scored it 1.0 (it summed passing TDs); v2 scores it 0.0. The
                  two definitions genuinely disagree about this pick, so summing
                  their hit rates is not a rounding matter, it is two different
                  markets called one.
    `unlabelled`- resolved before the column existed; genuinely unknown. Also a
                  receiving-TD game, so its 1.0 is the one value both definitions
                  agree on -- which makes it a clean control: it lands in its own
                  bucket and changes nothing about the others.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("v2_hit"), _prop("v1_miss"), _prop("unlabelled")])
    store.reconcile_player_prop_predictions(_stats(
        ("v2_hit", 1, 0, 0),     # rushing TD: 1.0 under either definition
        ("v1_miss", 0, 0, 2),     # passing TDs only: 1.0 under v1, 0.0 under v2
        ("unlabelled", 0, 1, 0),  # receiving TD: 1.0 under either definition
    ))
    # The v1 row carries v1's verdict; the pre-column row carries its verdict and no
    # stamp. These are the two states the live database is actually in.
    _as_graded_under(["v1_miss"], OLD, actual=1.0)
    _as_graded_under(["unlabelled"], None, actual=1.0)
    return _td()


def test_two_definitions_are_never_averaged_into_one_hit_rate():
    """The defect itself, as an assertion.

    The headline `hit_rate_when_called` is computed over the current definition's
    rows alone. If the two definitions were summed, this row's `n_resolved` would
    be 3 and its hit rate 1.0 (all three rows scored 1.0) -- which is the number
    the track record published while quietly mixing a QB's passing-TD pick into a
    rushing-or-receiving record.
    """
    td = _mixed_record()

    assert td["n_resolved"] == 1, "the headline counted rows graded under another definition"
    assert td["hit_rate_when_called"] == 1.0
    assert _bucket(td, CURRENT)["n_resolved"] == 1


def test_a_mixed_fixture_produces_three_buckets_not_one_figure():
    """The chosen mechanism: report apart, never together.

    Three buckets for three states, each carrying its own n, hit rate and Brier.
    `is_current` is on each so a consumer can find the headline's own bucket
    without hardcoding the version number.
    """
    td = _mixed_record()

    versions = [b["label_version"] for b in td["by_label_version"]]
    assert versions == [OLD, CURRENT, None], "buckets must be versions ascending, null last"

    for bucket in td["by_label_version"]:
        assert set(bucket) >= {"label_version", "is_current", "n_resolved",
                               "n_called", "hit_rate_when_called", "brier_score"}
    assert [b["is_current"] for b in td["by_label_version"]] == [False, True, False]


def test_the_old_definition_keeps_its_own_number_and_the_two_differ():
    """The disagreement is real and the payload shows it.

    `v1_miss` scored 1.0 under the definition that counted passing TDs and is not
    in the current record at all. Both numbers are visible: the old bucket carries
    the old hit rate, the headline carries the new one, and neither contains the
    other's row.
    """
    td = _mixed_record()

    old = _bucket(td, OLD)
    assert old["n_resolved"] == 1
    assert old["hit_rate_when_called"] == 1.0, "the v1 row's own definition scored it a hit"
    assert td["hit_rate_when_called"] == 1.0, "and it is not in the current headline's n"

    # The same pick, under the definition in force now, is not an anytime TD at
    # all. That is the whole reason the two figures must not be added.
    assert player_usage.anytime_td_actual(0, 0) == 0.0
    assert _v1_actual(0, 0, 2) == 1.0


def _v1_actual(rushing, receiving, passing):
    """The superseded v1 definition, written out here and nowhere else in src/.

    Reproduced locally rather than imported because it is gone on purpose: nothing
    in `src/` may still compute it, or the grader could drift back onto it. It
    exists in this test only to prove the two definitions genuinely disagree.
    """
    return float((int(rushing) + int(receiving) + int(passing)) > 0)


def test_the_brier_score_is_single_definition_too():
    """Brier is a mean over rows, so it is exactly as wrong as the hit rate when
    versions are mixed. `v1_miss` was predicted at 0.6 and v1 scored it 1.0, so it
    contributes 0.16 to a Brier it has no business contributing to."""
    td = _mixed_record()

    assert td["brier_score"] == pytest.approx((0.6 - 1.0) ** 2)
    assert _bucket(td, OLD)["brier_score"] == pytest.approx((0.6 - 1.0) ** 2)
    assert _bucket(td, None)["brier_score"] == pytest.approx((0.6 - 1.0) ** 2)
    # Three versions summed would be the mean of three identical squares, which
    # happens to equal this one -- so assert on n, not only on the score.
    assert td["n_resolved"] == 1


def test_the_confidence_buckets_are_single_definition_too():
    """Same arithmetic, one level down: a bucket is a mean over a subset, so a
    bucket that mixed versions would calibrate the model against a truth it was
    not fitted on."""
    td = _mixed_record()

    called = {b["label"]: b for b in td["confidence_buckets"]}
    assert called["60-70%"]["n"] == 1, "the v1 and unlabelled rows leaked into a bucket"
    assert called["60-70%"]["hit_rate"] == 1.0


def test_the_buckets_partition_the_whole_counted_market():
    """Nothing is discarded, which is why bucketing was chosen over exclusion.

    Every counted `anytime_td` row is in exactly one bucket, so the buckets sum to
    the total. A consumer can therefore get any version they did not get as a
    headline, and the payload accounts for every pick on the record.
    """
    td = _mixed_record()

    assert sum(b["n_resolved"] for b in td["by_label_version"]) == 3
    assert td["n_resolved"] + sum(
        b["n_resolved"] for b in td["by_label_version"] if not b["is_current"]
    ) == 3


def test_an_unlabelled_row_is_in_its_own_bucket_and_not_in_the_headline():
    """NULL is the honest answer for a row graded before the column existed, and
    it is kept out of the current headline rather than assumed into it."""
    td = _mixed_record()

    unknown = _bucket(td, None)
    assert unknown["n_resolved"] == 1
    assert unknown["is_current"] is False
    assert "unknown_reason" in unknown
    assert td["n_resolved"] == 1, "a row of unknown definition entered the current headline"


def test_versions_are_separated_even_when_they_produce_identical_values():
    """The attack that a value-based check would fall to.

    A receiving-TD game is 1.0 under v1 and 1.0 under v2, so these two rows carry the
    SAME `actual_value`. A naive fix -- bucket by the value, or dedupe rows whose
    `actual_value` agrees -- would merge them and report one market. Separation is
    keyed on the STAMP, never on the number, precisely so that two rows which happen
    to agree still count as two different markets' worth of history.

    They are also both "called" (0.6 >= 0.5), so a hit rate averaged over the pair is
    1.0 exactly as it is over either one. The only figure that exposes the difference is
    `n_resolved`, which is why it is asserted directly.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("old"), _prop("new")])
    store.reconcile_player_prop_predictions(_stats(("old", 0, 1, 0), ("new", 0, 1, 0)))
    _as_graded_under(["old"], OLD, actual=1.0)   # identical verdict, older definition

    td = _td()

    assert td["n_resolved"] == 1, "two versions merged because their values happened to agree"
    assert td["hit_rate_when_called"] == 1.0
    assert _bucket(td, OLD)["n_resolved"] == 1
    assert _bucket(td, OLD)["hit_rate_when_called"] == 1.0
    assert sum(b["n_resolved"] for b in td["by_label_version"]) == 2


def test_a_version_value_the_reader_has_never_seen_still_gets_its_own_bucket():
    """Forward compatibility, and the other way a filter-by-current-version can go wrong.

    A future `ANYTIME_TD_LABEL_VERSION = 3` will produce rows stamped 3 while the record
    still holds 1s and 2s. Bucketing iterates the values actually present and sorts them,
    so 3 appears without anyone editing `_prop_markets` -- and a hardcoded two-bucket dict
    of {1: ..., 2: ...} would have silently dropped it, which is the failure mode a
    record nobody can fully account for produces.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("a"), _prop("b")])
    store.reconcile_player_prop_predictions(_stats(("a", 1, 0, 0), ("b", 0, 0, 0)))
    _as_graded_under(["a"], CURRENT + 1, actual=1.0)

    td = _td()

    # Ascending, so the payload reads in version order rather than insertion order.
    assert [b["label_version"] for b in td["by_label_version"]] == [CURRENT, CURRENT + 1]
    assert td["n_resolved"] == 1, "a row graded under a different definition entered the headline"
    assert _bucket(td, CURRENT + 1)["n_resolved"] == 1
    assert sum(b["n_resolved"] for b in td["by_label_version"]) == 2


def test_a_non_integer_version_on_a_row_does_not_crash_the_summary():
    """Attack the reader, not the writer.

    `LABEL_VERSION_COLUMN` is `INTEGER` and the grader writes an int, but SQLite has
    no enforcement: a hand-edited row or a future writer could put a string there.
    The filter must then simply not match, which leaves the row in its own bucket
    rather than raising or -- worse -- being compared loosely enough to join the
    current headline.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("p1"), _prop("p2")])
    store.reconcile_player_prop_predictions(_stats(("p1", 1, 0, 0), ("p2", 0, 0, 0)))
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            f"UPDATE player_prop_predictions SET {store.LABEL_VERSION_COLUMN} = 'v2' "
            "WHERE player_id = 'p2'"
        )

    td = _td()

    assert td["n_resolved"] == 1, "a string version leaked into the current headline"
    assert sum(b["n_resolved"] for b in td["by_label_version"]) == 2, "a row went missing"
    # Numeric first, then the unreadable one -- and the unreadable one is still its own
    # bucket rather than being dropped or folded into v2.
    assert [b["label_version"] for b in td["by_label_version"]] == [CURRENT, "v2"]


def test_an_all_unlabelled_record_reports_an_empty_headline_rather_than_a_mixed_one():
    """The worst case: a database where every row predates the column.

    The only two answers here are "an empty headline plus the rows in the null
    bucket" and "a headline mixing unknown definitions into v2". Only the first
    is honest, and it is what a straight `== current_version` filter produces.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("p1"), _prop("p2")])
    store.reconcile_player_prop_predictions(_stats(("p1", 1, 0, 0), ("p2", 0, 0, 0)))
    _as_graded_under(["p1", "p2"], None, actual=1.0)

    td = _td()
    assert td["n_resolved"] == 0
    assert td["hit_rate_when_called"] is None
    assert td["brier_score"] is None
    assert _bucket(td, None)["n_resolved"] == 2


def test_bumping_the_definition_empties_the_headline_and_keeps_every_row(monkeypatch):
    """Why the version is load-bearing rather than informational.

    Nothing re-resolves and nothing is deleted when the definition moves; the
    headline simply stops matching, and `by_label_version` still accounts for
    every pick. Without the stamp the same change would have blended v2's rows
    into a v3 headline and reported one number as three things at once.
    """
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("p1"), _prop("p2")])
    store.reconcile_player_prop_predictions(_stats(("p1", 1, 0, 0), ("p2", 0, 0, 0)))

    monkeypatch.setattr(player_usage, "ANYTIME_TD_LABEL_VERSION", CURRENT + 1)
    td = _td()

    assert td["label_version"] == CURRENT + 1
    assert td["n_resolved"] == 0, "rows graded under v2 entered a v3 headline"
    assert _bucket(td, CURRENT)["n_resolved"] == 2
    assert sum(b["n_resolved"] for b in td["by_label_version"]) == 2


# --- the secondary figure behaves the same ------------------------------------


def test_the_pre_kickoff_figure_is_split_by_version_the_same_way():
    """`pre_kickoff` is the same summariser over a subset of the same rows, so it
    has to apply the same rule. A headline split by version beside a secondary
    figure that quietly mixed them would let the two disagree about what the
    current definition looks like."""
    store.record_game_predictions([_game()])
    store.record_player_prop_predictions([_prop("p1")])
    store.reconcile_player_prop_predictions(_stats(("p1", 1, 0, 0)))
    _as_graded_under(["p1"], OLD, actual=1.0)

    props = store.get_track_record()["player_props"]

    assert props["anytime_td"]["n_resolved"] == 0
    assert props["anytime_td"]["by_label_version"][0]["label_version"] == OLD
    assert props["pre_kickoff"]["anytime_td"]["n_resolved"] == 0
    assert props["pre_kickoff"]["anytime_td"]["by_label_version"][0]["label_version"] == OLD
    assert props["pre_kickoff"]["anytime_td"]["by_label_version"][0]["n_resolved"] == 1


# --- the migration ------------------------------------------------------------


def test_the_migration_adds_the_column_without_touching_existing_rows(tmp_path):
    """Additive, and it must be: the ALTER TABLE is the one piece of schema work
    that runs against a live database of immutable picks. If it rewrote a row's
    `actual_value` -- or backfilled a version onto it -- the record would have been
    edited after the fact by a migration, which is precisely what the standing
    rule forbids.

    Built by hand from the pre-column schema rather than by migrating a current
    database, because the column already exists in this build and an idempotent
    second pass adds nothing.
    """
    import sqlite3

    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.execute(
        """
        CREATE TABLE player_prop_predictions (
            game_id TEXT NOT NULL, player_id TEXT NOT NULL, player_name TEXT NOT NULL,
            market TEXT NOT NULL, predicted_value REAL NOT NULL, snapshotted_at TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0, actual_value REAL,
            PRIMARY KEY (game_id, player_id, market)
        )
        """
    )
    conn.execute(
        "INSERT INTO player_prop_predictions VALUES ('g1', 'p1', 'P', 'anytime_td', 0.6, "
        "'2026-09-01T00:00:00', 1, 1.0)"
    )
    conn.commit()
    added = store._ensure_schema(conn)
    conn.commit()
    row = conn.execute(
        f"SELECT predicted_value, actual_value, resolved, {store.LABEL_VERSION_COLUMN} "
        "FROM player_prop_predictions WHERE player_id = 'p1'"
    ).fetchone()
    conn.close()

    assert f"player_prop_predictions.{store.LABEL_VERSION_COLUMN}" in added
    assert row == (0.6, 1.0, 1, None), (
        f"the migration edited an existing pick: {row}"
    )


def test_the_migration_is_idempotent():
    """It runs on every connection, so a second pass must add nothing."""
    first = store.migrate_tracking_db()
    second = store.migrate_tracking_db()
    assert f"player_prop_predictions.{store.LABEL_VERSION_COLUMN}" in first
    assert f"player_prop_predictions.{store.LABEL_VERSION_COLUMN}" not in second
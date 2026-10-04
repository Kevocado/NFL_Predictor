"""tests/test_passing_td_record.py -- the QB passing-TD pick, end to end.

The gap PR #31 found: `routes.background_tracking_tick` snapshotted
`market="anytime_td"` and every market in `player_props.POSITION_MARKETS`, and
nothing else. The site served a passing-TD call (`Over 2.5 · 64%`) and nothing
ever wrote a row for it, so no pick was graded against actual passing TDs and no
record could accumulate. PR #31 deleted the report that read those rows, correctly:
a report nothing writes is worse than no report, because it publishes a clean
permanently-empty record that reads as "no picks yet".

This file is the other half. The writer now exists
(`routes._passing_td_prop_row`, called from the tick), the report exists
(`store._passing_td_metrics`, inside `_prop_markets` beside every other market's),
and these tests hold the four properties that make the record mean something.

**The schema, since it is the first question.** `player_prop_predictions`' primary
key is `(game_id, player_id, market)` -- `market` is part of the key, not a column
of the key. So a `passing_tds` row fits as an ordinary row on an ordinary key: no
migration, no new table, and nothing to weaken. PR #31's surviving columns
(`line`, `line_source`, `side`, `mu`, `call_prob`) are exactly what a writer
needed and were kept for it. Nothing here relaxes a uniqueness constraint.

**The grading rule**, which is the anytime-TD one applied to an over/under: hit
iff the ACTUAL passing TD count is above the recorded line for an over call and
below it for an under call. The count is an integer and `model_line` only ever
produces x.5, so a push is unreachable rather than merely unlikely; a stored
whole-number line is reported `n_ungradeable` instead of being scored a side it
does not have. The stored line and side are what get graded -- never a re-derived
one, so the record grades the call a reader could actually have made.

**Immutability**, unchanged and shared with every other market: the writer is
`INSERT OR IGNORE`, so the first snapshot of a key is the one that survives, and
the grader's UPDATE selects only `resolved = 0`, so a recorded pick is never
re-resolved. `_counted_prop_picks` additionally keeps the EARLIEST row per key in
the reader, so a re-keyed or restored history still counts one pick per key.

**Provenance.** The line is derived from the model's own expectation
(`qb_passing_td.model_line`), so it is a MODEL line and not a sportsbook price --
there is no book here to have an edge against. Kevin's 2026-10-01 decision is
explicit and the site already labels it "model line"; what is pinned here is that
the provenance travels with the number into the database and back out onto the
record, so nothing stored or surfaced invites the other reading.
"""
from __future__ import annotations

import contextlib
import logging
import sqlite3

import pandas as pd
import pytest

from nfl_predictor.api import routes
from nfl_predictor.data import schedules
from nfl_predictor.models import player_props
from nfl_predictor.models.qb_passing_td import MODEL_LINE_SOURCE, PASSING_TD_MARKET
from nfl_predictor.tracking import store

GAME_ID = "2099_01_AKC_BAL"
KICKOFF = "2099-01-01T18:00:00+00:00"
#: Snapshotted before `KICKOFF`, so `_snapshotted_after_kickoff` is False and the
#: grader will grade the row rather than treat it as a reconstruction.
SNAPSHOT = "2098-12-01T12:00:00+00:00"


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A tracking database of this test's own, migrated on first use."""
    from nfl_predictor import config

    path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", path)
    return path


def _qb_call(player_id="00-1", line=2.5, side="under", mu=2.3, prob=0.61):
    """A served QB passing-TD payload, exactly the keys `predict_props` flattens."""
    return {
        "player_id": player_id, "player_name": "Pat QB", "position": "QB",
        "recent_team": "KC", "anytime_td_prob": 0.44, "passing_yards": 251.0,
        "passing_td_line": line, "passing_td_line_source": MODEL_LINE_SOURCE,
        "passing_td_side": side, "passing_td_mu": mu, "passing_td_prob": prob,
        "passing_td_distribution": "poisson",
    }


def _row(payload, market, value, **extra):
    return {
        "game_id": GAME_ID, "player_id": payload["player_id"],
        "player_name": payload["player_name"], "position": payload["position"],
        "market": market, "predicted_value": value, **extra,
    }


def _record_served_call(db, payload=None, **overrides):
    """What the tick writes for one QB: the anytime-TD pick, the yards, and the call."""
    payload = payload or _qb_call(**overrides)
    return store.record_player_prop_predictions([
        _row(payload, "anytime_td", payload["anytime_td_prob"]),
        _row(payload, "passing_yards", payload["passing_yards"]),
        _row(payload, PASSING_TD_MARKET, payload["passing_td_prob"],
             line=payload["passing_td_line"], line_source=payload["passing_td_line_source"],
             side=payload["passing_td_side"], mu=payload["passing_td_mu"],
             call_prob=payload["passing_td_prob"]),
    ])


def _record_game(db, kickoff=KICKOFF):
    store.record_game_predictions([{
        "game_id": GAME_ID, "home_team": "BAL", "away_team": "KC",
        "commence_time": kickoff, "home_win_prob": 0.5, "away_win_prob": 0.5,
    }])


def _actual(passing_tds=2, player_id="00-1"):
    return pd.DataFrame([{"game_id": GAME_ID, "player_id": player_id,
                          "passing_tds": passing_tds}])


def _stored(db, market=PASSING_TD_MARKET, player_id="00-1"):
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM player_prop_predictions WHERE game_id = ? AND player_id = ? "
            "AND market = ?", (GAME_ID, player_id, market),
        ).fetchall()
    return [dict(r) for r in rows]


# --- 1. the writer ------------------------------------------------------------


def _run_tick(monkeypatch, payloads):
    """One `background_tracking_tick` against a real store, nothing fetched."""
    games = pd.DataFrame([{
        "game_id": GAME_ID, "season": 2099, "week": 1, "home_team": "BAL",
        "away_team": "KC", "home_score": None, "away_score": None,
        "gameday": KICKOFF, "commence_time": KICKOFF,
        "spread_line": None, "total_line": None,
    }])
    monkeypatch.setattr(schedules, "fetch_upcoming_games", lambda season, week: games)
    monkeypatch.setattr(schedules, "fetch_schedules",
                        lambda seasons, force_refresh=False, max_age_seconds=None: games)
    # Nothing has finished, so the tick stops after reconciliation and the
    # backfill path (which would fetch weekly player stats) is never entered.
    monkeypatch.setattr(
        schedules, "fetch_current_season_partial",
        lambda: pd.DataFrame(columns=["game_id", "home_score", "away_score"]),
    )
    # Wide enough that a far-future kickoff is inside the snapshot window. The
    # window filters on `now + lead`, and this game's kickoff is in 2099 so the row
    # it writes is provably pre-kickoff and the grader will grade it.
    monkeypatch.setattr(routes, "SNAPSHOT_LEAD_HOURS", 24 * 365 * 100)
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {})
    monkeypatch.setattr(routes, "_load_game_history", lambda season: pd.DataFrame())
    monkeypatch.setattr(routes, "_get_player_props_live", lambda season, week: payloads)
    routes.background_tracking_tick(2099, 1)


def test_the_tracking_tick_stores_the_passing_td_call_it_served(db, monkeypatch):
    """The gap itself, closed. A served call becomes a row, on the production path.

    Goes through `background_tracking_tick` rather than
    `store.record_player_prop_predictions` because that tick is the only writer in
    `src/` -- a row written by a test fixture would prove the writer accepts a row
    and nothing about whether the record can ever acquire one.
    """
    _run_tick(monkeypatch, [_qb_call()])

    stored = _stored(db)
    assert len(stored) == 1, (
        f"the tick wrote {len(stored)} passing_tds row(s) for a QB it was handed a "
        "call for; the market is served and reported but nothing records it, which "
        "is the defect PR #31 found"
    )
    row = stored[0]
    assert row["line"] == 2.5, "the line that was served is not the line stored"
    assert row["side"] == "under"
    # The provenance travels with the number, through the WRITER rather than around
    # it: `test_the_stored_shape_carries_the_model_line_provenance` below writes its
    # rows through the store directly, so without this assertion a writer that
    # dropped `line_source` on the floor would still pass every provenance test.
    assert row["line_source"] == MODEL_LINE_SOURCE, (
        "the tick stored the line without its provenance, so the number in the "
        "database can be read as a sportsbook price"
    )
    assert row["mu"] == pytest.approx(2.3)
    assert row["call_prob"] == pytest.approx(0.61)
    # `predicted_value` is the called side's probability -- the same meaning it has
    # on the `anytime_td` row -- so both markets' confidence grade one way.
    assert row["predicted_value"] == pytest.approx(0.61)
    # And the markets the tick already wrote are still written: adding a market
    # must not cost one.
    assert {r["market"] for r in _stored(db, market="anytime_td")} == {"anytime_td"}
    assert {r["market"] for r in _stored(db, market="passing_yards")} == {"passing_yards"}


def test_the_prop_table_key_takes_a_passing_td_row_unchanged(db):
    """The schema question, answered by the schema rather than by a migration.

    `market` is part of the primary key, so the passing-TD call is an ordinary row
    on an ordinary key and needed no new table and no relaxed constraint. Asserted
    from `sqlite_master` so the claim cannot rot into a comment: if the key ever
    loses `market`, this fails and the writer has to be re-checked, not assumed.
    """
    _record_served_call(db)
    with sqlite3.connect(db) as conn:
        key = [r for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'player_prop_predictions'"
        )][0][0]
    assert "PRIMARY KEY (game_id, player_id, market)" in key, (
        "player_prop_predictions is no longer keyed per market, so a passing_tds row "
        "would collide with the same player's other markets and the writer would "
        "need a schema change it has not had"
    )


# --- 2. the grader -----------------------------------------------------------


def test_a_pick_is_graded_against_the_actual_passing_td_count(db):
    """The outcome exists and is the real one: nflverse's `passing_tds` column.

    Not the anytime-TD roll-up, which also counts rushing and receiving touchdowns
    -- a model fitted on passing TDs graded against a different count would be a
    record of nothing.
    """
    _record_game(db)
    _record_served_call(db, line=2.5, side="over", prob=0.64)
    # Two rows graded: the `anytime_td` pick and this one. The `passing_yards` row is
    # left alone because `_actual` carries no `passing_yards` column, which is the
    # grader refusing to invent a stat the box score did not report.
    assert store.reconcile_player_prop_predictions(_actual(passing_tds=3)) == 2

    row = _stored(db)[0]
    assert row["resolved"] == 1
    assert row["actual_value"] == 3.0

    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["n_resolved"] == 1
    assert td["per_pick"][0]["actual_passing_tds"] == 3.0


def test_a_wrong_call_is_recorded_as_wrong(db):
    """The over half. 1 passing TD against a call of over 2.5 is a miss, and the
    record says so rather than dropping the pick or crediting it."""
    _record_game(db)
    _record_served_call(db, line=2.5, side="over", prob=0.64)
    store.reconcile_player_prop_predictions(_actual(passing_tds=1))

    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["hit_rate_when_called"] == 0.0
    assert td["n_called"] == 1
    assert td["by_side"]["over"] == {"n": 1, "hits": 0, "hit_rate": 0.0}
    assert td["per_pick"][0]["hit"] is False
    # Confidence 0.64 against a miss is the worst possible outcome, so the Brier
    # score on that binary event is (0.64 - 0) ** 2. A grader that scored the miss
    # as a hit would report something near zero here instead.
    assert td["brier_score"] == pytest.approx(0.64 ** 2)


def test_the_under_half_is_graded_in_the_other_direction(db):
    """Both sides, because `actual > line` alone would pass the over test above."""
    _record_game(db)
    _record_served_call(db, player_id="00-2", line=1.5, side="under", prob=0.7)
    # 1 passing TD is under a 1.5 line, so the under call hit. A grader written only
    # as `actual > line` would score this miss.
    store.reconcile_player_prop_predictions(_actual(passing_tds=1, player_id="00-2"))

    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["hit_rate_when_called"] == 1.0
    assert td["by_side"]["under"] == {"n": 1, "hits": 1, "hit_rate": 1.0}
    assert td["brier_score"] == pytest.approx((0.7 - 1.0) ** 2)


def test_a_whole_number_line_is_reported_ungradeable_rather_than_scored(db):
    """The push case, which `model_line` makes unreachable and a stored row could not.

    A whole-number line means `actual == line` has a third outcome, so there is no
    honest hit/miss. Scoring it anyway would invent a side; raising would take down
    `get_track_record` for every market on the site. So it is counted, named, and
    left out of the hit rate, whose denominator is published beside it.
    """
    _record_game(db)
    _record_served_call(db, line=3.0, side="over", prob=0.6)
    store.reconcile_player_prop_predictions(_actual(passing_tds=3))

    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["n_ungradeable"] == 1
    assert td["n_gradeable"] == 0
    assert td["hit_rate_when_called"] is None, (
        "a push was scored as if it were over or under"
    )
    assert td["per_pick"] == []


# --- 3. immutability ---------------------------------------------------------


def test_the_earliest_recorded_pick_per_key_is_the_one_counted(db):
    """Two rows on one key -- a re-keyed or restored history -- and the earlier wins.

    `INSERT OR IGNORE` plus the primary key makes this unreachable from any writer in
    this repo, which is exactly why the READER implements the rule too: a rule
    enforced only by a primary key stops being enforced the moment the key changes,
    and double-counting is invisible when it happens -- the hit rate simply improves,
    silently, and a rerun of the model gets to grade a second time.

    So the table here is created with `snapshotted_at` in its key -- a history that
    lost the per-market uniqueness -- and two snapshots of one key go in. No
    constraint is weakened anywhere in `src/`; this is a fixture standing in for a
    database nobody in this repo builds.
    """
    with contextlib.closing(sqlite3.connect(db)) as conn:
        conn.execute(
            """
            CREATE TABLE player_prop_predictions (
                game_id TEXT NOT NULL, player_id TEXT NOT NULL, player_name TEXT NOT NULL,
                market TEXT NOT NULL, predicted_value REAL NOT NULL,
                snapshotted_at TEXT NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0, actual_value REAL,
                PRIMARY KEY (game_id, player_id, market, snapshotted_at)
            )
            """
        )
        conn.commit()
    # Then the ordinary migration, which adds `position` and the five passing-TD
    # columns as ALTERs -- the same path a live database takes.
    with contextlib.closing(store._connect()) as conn, conn:
        pass

    with contextlib.closing(sqlite3.connect(db)) as conn:
        for snapshot, line, side, prob in (
            ("2098-12-01T12:00:00+00:00", 2.5, "under", 0.61),
            ("2098-12-05T12:00:00+00:00", 3.5, "over", 0.55),
        ):
            conn.execute(
                "INSERT INTO player_prop_predictions (game_id, player_id, player_name, "
                "position, market, predicted_value, snapshotted_at, line, line_source, "
                "side, mu, call_prob, resolved, actual_value) "
                "VALUES (?,?,?,'QB',?,?,?,?,?,?,?,?,1,?)",
                (GAME_ID, "00-1", "Pat QB", PASSING_TD_MARKET, prob, snapshot,
                 line, MODEL_LINE_SOURCE, side, 2.3, prob, 1.0),
            )
            conn.execute(
                "INSERT INTO player_prop_predictions (game_id, player_id, player_name, "
                "position, market, predicted_value, snapshotted_at, resolved, actual_value) "
                "VALUES (?,?,?,'QB','anytime_td',?,?,1,?)",
                (GAME_ID, "00-1", "Pat QB", 0.44, snapshot, 1.0),
            )
        conn.commit()

    assert len(_stored(db)) == 2, "the fixture failed to build the re-keyed history"
    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert len(td["per_pick"]) == 1, "the later rerun was counted as a second pick"
    assert td["n_resolved"] == 1
    # The EARLIEST row's call is the counted one, so the reader is grading the pick
    # that was made first and not the one a later rerun produced.
    assert td["per_pick"][0]["line"] == 2.5, "the later rerun displaced the earliest pick"
    assert td["per_pick"][0]["side"] == "under"


def test_a_later_snapshot_does_not_rewrite_a_recorded_pick(db):
    """The writer's half of the same rule: `INSERT OR IGNORE` keeps the first.

    Otherwise re-running the model after a refit would overwrite the pick a reader
    acted on with one fitted on the game's own result, for free.
    """
    _record_served_call(db, line=2.5, side="under", prob=0.61)
    assert _record_served_call(db, line=3.5, side="over", prob=0.55) == 0, (
        "the second snapshot was written rather than ignored"
    )
    row = _stored(db)[0]
    assert (row["line"], row["side"], row["call_prob"]) == (2.5, "under", pytest.approx(0.61))


def test_the_immutability_rule_is_in_the_sql_and_not_only_in_what_it_does_today(db):
    """Both guards, read out of the statements rather than inferred from behaviour.

    The two tests above would pass with either guard removed, and that is the whole
    point of this one. `record_player_prop_predictions` reads nothing, so dropping
    `OR IGNORE` would raise `IntegrityError` rather than rewrite -- caught, but not by
    the assertion written to say "a later snapshot does not rewrite a pick". And
    `reconcile_player_prop_predictions` already selects only `resolved = 0` rows, so
    dropping `AND resolved = 0` from its UPDATE changes no result at all while the
    rule it protects becomes one statement wide. A rule that only holds because of
    the statement next to it is not yet a rule.
    """
    import inspect

    writer = inspect.getsource(store.record_player_prop_predictions)
    assert "INSERT OR IGNORE INTO player_prop_predictions" in writer, (
        "the writer is no longer INSERT OR IGNORE, so a rerun after a refit would "
        "raise instead of leaving the first snapshot standing"
    )
    grader = inspect.getsource(store.reconcile_player_prop_predictions)
    assert "AND resolved = 0" in grader, (
        "the grader's UPDATE no longer restricts itself to unresolved rows, so a "
        "recorded pick could be re-resolved -- which the read filter alone does not "
        "prevent if that filter is ever widened"
    )


def test_a_recorded_pick_is_never_re_resolved(db):
    """The grader's half: the UPDATE selects only `resolved = 0`.

    A pick graded against one actual count stays graded against it. A grader that
    rewrote `actual_value` would let a corrected box score silently change a
    published record -- and, with the line frozen, would then grade the original
    call against an outcome it was not made for.
    """
    _record_game(db)
    _record_served_call(db, line=2.5, side="over", prob=0.64)
    assert store.reconcile_player_prop_predictions(_actual(passing_tds=3)) == 2
    # Nothing left unresolved, so a second pass with a different count grades nothing.
    assert store.reconcile_player_prop_predictions(_actual(passing_tds=0)) == 0

    row = _stored(db)[0]
    assert row["actual_value"] == 3.0
    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["hit_rate_when_called"] == 1.0


# --- 4. absence --------------------------------------------------------------


def test_a_qb_with_no_call_is_visible_as_absent(db, monkeypatch, caplog):
    """Requirement: a pick that cannot be recorded must be VISIBLE, not dropped.

    Two halves, both asserted, because either alone is a silent short record. The
    tick logs the QB at WARNING -- the same treatment the empty-`actual_stats` case
    below it gets, and for the same reason: an absent row is otherwise
    indistinguishable from a QB nobody picked. And the record names the gap
    numerically, so a reader of the track record sees `n_served_without_a_graded_call`
    rather than a passing-TD record that is quietly short.
    """
    without_call = _qb_call()
    del without_call["passing_td_line"]
    with caplog.at_level(logging.WARNING, logger=routes.__name__):
        _run_tick(monkeypatch, [without_call])

    assert _stored(db) == [], "a call that was never produced must not become a row"
    assert any("no passing-TD call for 1 QB prop" in r.getMessage()
               for r in caplog.records), (
        "the tick dropped a QB with no passing-TD call without saying so: "
        f"{[r.getMessage() for r in caplog.records]}"
    )

    # Grade the `anytime_td` pick the tick did write, so the QB is a RESOLVED served
    # pick and the gap is visible in the record rather than in a pending row. The
    # game row is written here because this fixture's fake slate cannot be predicted
    # (no game history), and the grader joins to `game_predictions` for the kickoff.
    _record_game(db)
    assert store.reconcile_player_prop_predictions(_actual(passing_tds=1)) == 1

    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["n_served"] == 1, "the served QB is not in the denominator"
    assert td["n_resolved"] == 0
    assert td["n_served_without_a_graded_call"] == 1, (
        "a served QB with no recorded call does not show up as a gap, so this "
        "record reads as complete while being short"
    )


def test_one_malformed_qb_payload_costs_that_qb_and_nothing_else(db, monkeypatch, caplog):
    """A failure mid-tick is visible AND bounded.

    A payload carrying a line and no side is a broken contract, so the writer raises
    rather than storing half a call. Where it is caught is the point: caught at the
    block's own `except`, one bad QB would take down the `anytime_td` and yardage
    rows as well -- forty-odd picks a week that have nothing to do with this market --
    and the prop record would be silently short for a reason nobody logged. So it is
    caught per player: the exception is logged with a traceback, that QB joins the
    uncalled list, and every other market is still recorded.
    """
    broken = _qb_call(player_id="00-bad")
    del broken["passing_td_side"]
    with caplog.at_level(logging.WARNING, logger=routes.__name__):
        _run_tick(monkeypatch, [broken, _qb_call(player_id="00-good")])

    assert _stored(db, player_id="00-bad", market="anytime_td"), (
        "one malformed QB payload took down the anytime-TD record with it"
    )
    assert _stored(db, player_id="00-good", market=PASSING_TD_MARKET), (
        "one malformed QB payload took down every other QB's passing-TD pick"
    )
    assert _stored(db, player_id="00-bad") == []
    assert any("malformed passing-TD payload for player_id=00-bad" in r.getMessage()
               for r in caplog.records), (
        f"the malformed payload was swallowed: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    assert any("no passing-TD call for 1 QB prop" in r.getMessage()
               for r in caplog.records), (
        "the malformed QB is not in the uncalled count, so the warning understates it"
    )


def test_a_record_where_every_qb_got_a_call_reports_no_gap(db):
    """The other direction, so the gap counter cannot be green by construction.

    Without this, `n_served_without_a_graded_call` being always 0 -- or the writer
    silently writing nothing -- would look like a healthy record.
    """
    _record_game(db)
    _record_served_call(db)
    store.reconcile_player_prop_predictions(_actual(passing_tds=3))
    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["n_served"] == 1
    assert td["n_served_without_a_graded_call"] == 0


def test_a_non_qb_never_counts_as_a_served_qb(db, monkeypatch):
    """The denominator is QBs. A WR has no passing-TD call, so a week of receivers
    is not a week of gaps -- otherwise the counter would be permanently red and
    therefore permanently ignored."""
    payload = {"player_id": "00-9", "player_name": "A Receiver", "position": "WR",
               "recent_team": "KC", "anytime_td_prob": 0.4, "receiving_yards": 60.0}
    _run_tick(monkeypatch, [payload])
    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["n_served"] == 0
    assert td["n_served_without_a_graded_call"] == 0


# --- 5. provenance -----------------------------------------------------------


def test_the_stored_shape_carries_the_model_line_provenance(db):
    """A stored line must never be mistakable for a price someone could bet at.

    `model_line(mu)` is the nearest half point to the model's OWN expectation, so
    there is no book here to have an edge against. The `line_source` column carries
    that next to the number in the database, and `line_sources` publishes what was
    found on the rows -- as a list, so a real book line wired in later shows up
    here instead of being read as a model line.
    """
    assert MODEL_LINE_SOURCE == "model_line"
    _record_game(db)
    _record_served_call(db, line=2.5, side="over", prob=0.64)
    store.reconcile_player_prop_predictions(_actual(passing_tds=3))

    assert _stored(db)[0]["line_source"] == MODEL_LINE_SOURCE
    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["line_sources"] == ["model_line"]
    assert [p["line_source"] for p in td["per_pick"]] == ["model_line"]


def test_a_row_with_an_unstated_provenance_is_reported_as_unstated(db):
    """The negative case, so `line_sources` cannot be a constant.

    A hand-written or migrated row with a NULL `line_source` is the one way the
    block could invite the wrong reading, so it is surfaced as `unstated` rather
    than dropped from the list or, worse, read as a model line.
    """
    _record_game(db)
    _record_served_call(db, line=2.5, side="over", prob=0.64)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE player_prop_predictions SET line_source = NULL "
                     "WHERE market = ?", (PASSING_TD_MARKET,))
    store.reconcile_player_prop_predictions(_actual(passing_tds=3))
    td = store.get_track_record()["player_props"][PASSING_TD_MARKET]
    assert td["line_sources"] == ["unstated"]
import contextlib

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


def _game():
    return {
        "game_id": "2025_01_BAL_KC", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00", "home_win_prob": 0.58, "away_win_prob": 0.42,
        "home_cover_prob": 0.52, "away_cover_prob": 0.48, "over_prob": 0.55, "under_prob": 0.45,
    }


def test_record_game_predictions_is_idempotent():
    n1 = store.record_game_predictions([_game()])
    n2 = store.record_game_predictions([_game()])

    assert n1 == 1
    assert n2 == 0  # already logged, INSERT OR IGNORE


def test_record_game_predictions_skips_snapshots_after_kickoff():
    already_kicked_off = _game() | {"commence_time": "2000-09-04T20:20:00"}
    still_upcoming = _game() | {"game_id": "2025_01_SF_LA", "home_team": "LA", "away_team": "SF"}

    n = store.record_game_predictions([already_kicked_off, still_upcoming])

    assert n == 1  # only the still-upcoming game got snapshotted
    assert store.get_track_record()["games"]["n_resolved"] == 0


def test_reconcile_game_predictions_fills_actual_outcome():
    store.record_game_predictions([_game()])
    results = pd.DataFrame(
        [{"game_id": "2025_01_BAL_KC", "home_score": 27, "away_score": 20}]
    )

    n = store.reconcile_game_predictions(results)

    assert n == 1
    record = store.get_track_record()["games"]
    assert record["n_resolved"] == 1
    assert record["pct_moneyline_correct"] == 1.0  # predicted home win, home won


def test_reconcile_game_predictions_counts_duplicate_results_once():
    store.record_game_predictions([_game()])
    results = pd.DataFrame(
        [
            {"game_id": "2025_01_BAL_KC", "home_score": 27, "away_score": 20},
            {"game_id": "2025_01_BAL_KC", "home_score": 20, "away_score": 27},
        ]
    )

    assert store.reconcile_game_predictions(results) == 1
    assert store.get_track_record()["games"]["n_resolved"] == 1


def test_record_and_reconcile_player_prop_predictions():
    store.record_game_predictions([_future_game()])
    prop = {"game_id": "2025_01_BAL_KC", "player_id": "p1", "player_name": "Runner",
            "market": "rushing_yards", "predicted_value": 85.0}
    store.record_player_prop_predictions([prop])

    player_stats_df = pd.DataFrame(
        [{"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 92, "receiving_yards": 5,
          "passing_yards": 0, "rushing_tds": 1, "receiving_tds": 0, "passing_tds": 0}]
    )
    n = store.reconcile_player_prop_predictions(player_stats_df)

    assert n == 1


def test_reconcile_player_prop_predictions_counts_duplicate_stats_once():
    store.record_game_predictions([_future_game()])
    prop = {"game_id": "2025_01_BAL_KC", "player_id": "p1", "player_name": "Runner",
            "market": "rushing_yards", "predicted_value": 85.0}
    store.record_player_prop_predictions([prop])
    player_stats_df = pd.DataFrame(
        [
            {"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 92},
            {"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 10},
        ]
    )

    assert store.reconcile_player_prop_predictions(player_stats_df) == 1


def _future_game(**overrides):
    game = {
        "game_id": "2025_01_BAL_KC", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": -3.5, "total_line": 51.5,
    }
    game.update(overrides)
    return game


def test_record_game_predictions_persists_spread_and_total_lines():
    import contextlib

    store.record_game_predictions([_future_game()])

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn).iloc[0]

    assert row["home_spread_line"] == -3.5
    assert row["total_line"] == 51.5


def _insert_legacy_row(conn, *, game_id, ats_hit, home_cover_prob, away_cover_prob,
                       total_hit=None, over_prob=None, under_prob=None,
                       season=None, week=None):
    """Write a row in the shape the pre-`_present` build produced.

    The current write path cannot produce this -- `_compute_hits` leaves `ats_hit`
    and `total_hit` NULL when the probabilities are missing -- so the only way to
    test the read path is to insert the row directly. The deployed database holds
    these.

    `season`/`week` default to NULL, which is what the deployed legacy rows hold;
    a test that needs the row to appear on the weekly chart has to say so.
    """
    conn.execute(
        """
        INSERT INTO game_predictions (
            game_id, home_team, away_team, commence_time, snapshotted_at,
            home_win_prob, away_win_prob, home_cover_prob, away_cover_prob,
            over_prob, under_prob, home_spread_line, total_line,
            season, week,
            resolved, actual_home_score, actual_away_score, moneyline_hit,
            ats_hit, total_hit
        ) VALUES (?, 'SF', 'DAL', '2025-09-14T20:20:00', '2025-09-14T17:00:00',
                  0.55, 0.45, ?, ?, ?, ?, -3.5, 51.5, ?, ?,
                  1, 20, 24, 1, ?, ?)
        """,
        (game_id, home_cover_prob, away_cover_prob, over_prob, under_prob,
         season, week, ats_hit, total_hit),
    )
    conn.commit()


def _record_one_genuine_hit(**overrides):
    """One real graded game, a HIT on *both* markets.

    30-30 with a -3.5 spread and a 51.5 total: home covers (-3.5) and the game
    goes over, and the model predicted both. Deliberately chosen so the genuine row
    is a HIT on every market -- a fabricated MISS is then the only thing that can
    move an aggregate off 1.0, so a filter that fails to exclude it is visible on
    each market independently.

    Its moneyline is a MISS: a 30-30 tie is an away win, the model picked home, so
    a test that needs a known moneyline rate can read it off the grade directly
    instead of recomputing it. `overrides` reach `_future_game`, so a test that
    needs the row on the weekly chart can give it a week.
    """
    store.record_game_predictions([_future_game(**overrides)])
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 30, "away_score": 30}])
    store.reconcile_game_predictions(results)


def test_track_record_aggregate_excludes_a_legacy_ats_row_with_no_cover_probabilities():
    """`get_game_verdict` already refuses to show an ATS market for a row whose cover
    probabilities are missing. The aggregate must agree: `pct_ats_correct` counted
    those same rows, so one game produced two opposite answers -- a per-game view
    with no ATS market beside an ATS percentage that included it.

    The fabricated row is a MISS and the genuine row is a HIT, so including the
    fabricated one moves the aggregate to 0.5 and the two cases cannot be confused.
    """
    import contextlib

    _record_one_genuine_hit()
    with contextlib.closing(store._connect()) as conn:
        _insert_legacy_row(
            conn, game_id="2025_02_SF_DAL", ats_hit=0,
            home_cover_prob=None, away_cover_prob=None,
        )

    track = store.get_track_record()

    # the per-game view already declines to report an ATS market for it
    # (`ats` is present but None -- the guard at store.py:593 leaves it unset)
    assert store.get_game_verdict("2025_02_SF_DAL")["ats"] is None
    # and the aggregate must not count it either
    assert track["games"]["pct_ats_correct"] == 1.0


def test_track_record_aggregate_excludes_a_legacy_ats_row_with_half_a_market():
    """One probability present and one missing is not a call either.

    `0.6` against `None` is not obviously the home side, but it is not a two-sided
    market, so the comparison never had two things to compare. A filter that checks
    only the first column of the pair would let this row through, which is why this
    case is separate from the both-missing one above.
    """
    import contextlib

    _record_one_genuine_hit()
    with contextlib.closing(store._connect()) as conn:
        _insert_legacy_row(
            conn, game_id="2025_03_SF_DAL", ats_hit=0,
            home_cover_prob=0.6, away_cover_prob=None,
        )

    assert store.get_game_verdict("2025_03_SF_DAL")["ats"] is None
    assert store.get_track_record()["games"]["pct_ats_correct"] == 1.0


def test_track_record_aggregate_excludes_a_legacy_totals_row_with_no_probabilities():
    """Same defect on the totals market, which is why the filter is applied twice.

    A total line recorded with no over/under probabilities produced the same
    fabricated `total_hit` the ATS fix removed.
    """
    import contextlib

    _record_one_genuine_hit()
    with contextlib.closing(store._connect()) as conn:
        _insert_legacy_row(
            conn, game_id="2025_04_SF_DAL", ats_hit=None,
            home_cover_prob=None, away_cover_prob=None,
            total_hit=0, over_prob=None, under_prob=None,
        )

    assert store.get_track_record()["games"]["pct_totals_correct"] == 1.0


def test_the_weekly_ats_figure_excludes_a_legacy_null_probability_row_too():
    """The merge of #15 and B3, pinned. Neither change's own tests can hold both.

    #15 taught the ATS/totals aggregate to refuse a row whose probability pair is
    missing, because `get_game_verdict` already refuses it. B3 (the weekly rows)
    later re-derived the same aggregate from `_grade`, which filtered on the grade
    column alone -- so the rule had to survive being rebuilt, or it silently did
    not. B3's weekly tests and #15's aggregate tests each passed against a version
    that broke the other; only a test asserting both figures at once fails if
    either half is dropped.

    Three rows in one week, so the week is the denominator that matters:

    * a genuine ATS/totals HIT, both pairs present (the only real grade there is);
    * a legacy row with a fabricated ATS and totals MISS and no probabilities --
      the per-game view refuses it (`get_game_verdict(...)["ats"] is None`), so
      every aggregate has to;
    * a row the current build recorded with a moneyline only: no line and no
      probability but the two required win probabilities, so both flags are NULL.
      It is the canary for the *other* half -- the moneyline has no pair to check,
      so any pair requirement applied to it drops a real grade here.

    The fabricated row is a MISS and the genuine one a HIT, and the ungraded row
    carries no flag, so leaking either into a count is visible in the number
    itself rather than in a rate that could round to the right thing.
    """
    import contextlib

    # (a) fully graded. `season`/`week` put it on the weekly chart.
    _record_one_genuine_hit(season=2099, week=3)
    # (b) `ats_hit`/`total_hit` set by a build that graded without the pairs.
    with contextlib.closing(store._connect()) as conn:
        _insert_legacy_row(
            conn, game_id="2025_05_SF_DAL", ats_hit=0,
            home_cover_prob=None, away_cover_prob=None,
            total_hit=0, over_prob=None, under_prob=None,
            season=2099, week=3,
        )
    # (c) never graded: the current build leaves both flags NULL without a line,
    # and the odds feed yields games with no spread market at all, so every
    # graded probability except the two the schema requires is absent.
    store.record_game_predictions([_future_game(
        game_id="2025_06_SF_DAL", home_spread_line=None, total_line=None,
        home_cover_prob=None, away_cover_prob=None, over_prob=None, under_prob=None,
        season=2099, week=3,
    )])
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": "2025_06_SF_DAL", "home_score": 24, "away_score": 20}])
    )

    # The per-game view's answer, which is the rule the aggregate has to match.
    assert store.get_game_verdict("2025_05_SF_DAL")["ats"] is None
    assert store.get_game_verdict("2025_05_SF_DAL")["totals"] is None

    games = store.get_track_record()["games"]

    # Headline: (b) is out of the ATS count and the ATS rate. Counting it gives
    # n_ats 2 and 0.5, which are both legible wrong answers.
    assert games["n_resolved"] == 3
    assert games["n_ats"] == 1
    assert games["pct_ats_correct"] == pytest.approx(1.0)
    assert games["n_totals"] == 1
    assert games["pct_totals_correct"] == pytest.approx(1.0)
    # ...and the moneyline is untouched by any of it. (a) is a moneyline MISS
    # (a 30-30 tie went to the away side), (b) and (c) are both hits, so the rate
    # is 2/3 over 3. Row (c) has no cover or over/under probability at all, so
    # borrowing either pair for the moneyline reports 1/2 over 2 -- the
    # over-filter, and the one that loses real grades.
    assert games["n_moneyline"] == 3
    assert games["pct_moneyline_correct"] == pytest.approx(2 / 3)

    week3 = {row["week"]: row for row in games["weekly"]}[3]

    # The weekly ATS figure is the same aggregate over one week, and B3's rule is
    # that its rate ships with its own denominator. Both must exclude (b).
    assert week3["n_ats"] == 1
    assert week3["pct_ats_correct"] == pytest.approx(1.0)
    assert week3["n_totals"] == 1
    assert week3["pct_totals_correct"] == pytest.approx(1.0)
    assert week3["n_moneyline"] == 3
    assert week3["pct_moneyline_correct"] == pytest.approx(2 / 3)
    # `n_games` stays 3, and this is the point rather than an oversight. It is
    # volume -- how many resolved games the tracker holds for the week -- and (b)
    # IS one: the game was picked and graded, it just has no ATS call. The ATS
    # figure's denominator is `n_ats`, which excludes it. Shrinking `n_games`
    # instead would also shrink the moneyline denominator, i.e. over-filter a
    # market that never needed the rule.
    assert week3["n_games"] == 3
    assert week3["tracked"] is True


def test_reconcile_grades_moneyline_ats_and_totals():
    import contextlib

    store.record_game_predictions([_future_game()])
    # home favored by -3.5 and predicted to cover (home_cover_prob=0.55 > away);
    # over predicted (over_prob=0.5 == under_prob=0.5, home_win predicted).
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 30, "away_score": 20}])

    resolved = store.reconcile_game_predictions(results)

    assert resolved == 1
    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn).iloc[0]

    assert row["moneyline_hit"] == 1  # home won, home was favored
    # home won by 10, spread was -3.5 => home covered; predicted_home_cover=True
    assert row["ats_hit"] == 1
    # total = 50, line = 51.5 => actual under; predicted was a coin flip (over_prob==under_prob)
    # tie-break must not crash -- assert it resolved to 0 or 1, not None
    assert row["total_hit"] in (0, 1)


def test_reconcile_grades_ats_correctly_when_margin_is_smaller_than_the_spread():
    """Discriminating case for the ATS sign convention (this plan's final
    review, finding B1): home_spread_line means "home expected margin"
    (positive = home favored). Home favored by 6, final margin only 3 --
    home did NOT cover. The old buggy formula (`margin + spread > 0`, i.e.
    3 + 6 > 0) claimed home covered; the fixture in
    test_reconcile_grades_moneyline_ats_and_totals above can't catch this
    because it happens to agree under both conventions."""
    import contextlib

    store.record_game_predictions([_future_game(
        home_spread_line=6.0, home_cover_prob=0.4087, away_cover_prob=0.5913,
    )])
    # home wins 24-21 -> home_margin = 3 < spread_line = 6 -> home did not cover.
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 24, "away_score": 21}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn).iloc[0]

    # model predicted away to cover (away_cover_prob > home_cover_prob) and
    # home indeed did not cover -> the model's call was correct.
    assert row["ats_hit"] == 1


def test_reconcile_leaves_ats_and_total_hit_null_when_lines_were_never_recorded():
    import contextlib

    store.record_game_predictions([_future_game(home_spread_line=None, total_line=None)])
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 30, "away_score": 20}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn).iloc[0]

    assert row["moneyline_hit"] == 1
    assert pd.isna(row["ats_hit"])
    assert pd.isna(row["total_hit"])


def test_reconcile_catches_a_game_missed_by_a_prior_tick():
    """Simulates a deploy/restart: the game was snapshotted, its results
    became available, but no tick ran to reconcile it until now."""
    store.record_game_predictions([_future_game(game_id="g1", commence_time="2099-01-01T00:00:00Z")])
    results = pd.DataFrame([{"game_id": "g1", "home_score": 21, "away_score": 14}])

    resolved = store.reconcile_game_predictions(results)

    assert resolved == 1


def test_backfill_catches_a_prior_season_row_current_season_partial_would_miss(monkeypatch):
    store.record_game_predictions([_future_game(game_id="g_old", season=2024, commence_time="2099-01-01T00:00:00Z")])
    from nfl_predictor.data import schedules
    monkeypatch.setattr(
        schedules, "load_training_data",
        lambda seasons: pd.DataFrame([{"game_id": "g_old", "home_score": 10, "away_score": 24}]),
    )

    resolved = store.backfill_unresolved_games(schedules)

    assert resolved == 1


def test_get_game_verdict_returns_none_for_unresolved_game():
    store.record_game_predictions([_future_game()])

    verdict = store.get_game_verdict("2025_01_BAL_KC")

    assert verdict is None


def test_get_game_verdict_summarizes_all_three_markets():
    store.record_game_predictions([_future_game()])
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 30, "away_score": 20}])
    store.reconcile_game_predictions(results)

    verdict = store.get_game_verdict("2025_01_BAL_KC")

    assert verdict is not None
    assert verdict["resolved"] is True
    assert verdict["game_id"] == "2025_01_BAL_KC"
    assert verdict["moneyline"]["hit"] == 1
    assert verdict["moneyline"]["predicted"] == "home_win"
    assert verdict["moneyline"]["actual"] == "home_win"
    assert verdict["ats"]["hit"] == 1
    assert verdict["ats"]["predicted"] == "home_cover"
    assert verdict["totals"]["hit"] in (0, 1)
    assert verdict["totals"]["predicted"] == "over"
    assert verdict["actual_home_score"] == 30
    assert verdict["actual_away_score"] == 20
    assert verdict["home_spread_line"] == -3.5
    assert verdict["total_line"] == 51.5


def test_get_game_verdict_reports_null_lines_when_never_recorded():
    store.record_game_predictions([_future_game(home_spread_line=None, total_line=None)])
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 30, "away_score": 20}])
    store.reconcile_game_predictions(results)

    verdict = store.get_game_verdict("2025_01_BAL_KC")

    assert verdict["home_spread_line"] is None
    assert verdict["total_line"] is None
    assert verdict["actual_home_score"] == 30
    assert verdict["actual_away_score"] == 20


def test_get_predictions_for_week_returns_pending_for_unresolved_and_verdict_for_resolved():
    store.record_game_predictions([_future_game(game_id="g1"), _future_game(game_id="g2", home_team="DEN", away_team="LAC")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))
    games_df = pd.DataFrame([
        {"game_id": "g1", "home_team": "BAL", "away_team": "KC"},
        {"game_id": "g2", "home_team": "DEN", "away_team": "LAC"},
    ])

    week = store.get_predictions_for_week(2025, 1, games_df)

    by_id = {row["game_id"]: row for row in week}
    assert by_id["g1"]["status"] == "resolved"
    assert by_id["g1"]["verdict"]["moneyline"]["hit"] is True
    assert by_id["g2"]["status"] == "pending"
    assert by_id["g2"]["verdict"] is None


def _prop(player_id="p1", market="rushing_yards", predicted=85.0, position=None,
          game_id="2025_01_BAL_KC"):
    prop = {"game_id": game_id, "player_id": player_id, "player_name": f"Player {player_id}",
            "market": market, "predicted_value": predicted}
    if position is not None:
        prop["position"] = position
    return prop


def _td_stats(*player_tds):
    """player_tds: (player_id, rushing_tds, receiving_tds, passing_tds)."""
    return pd.DataFrame(
        [{"game_id": "2025_01_BAL_KC", "player_id": pid,
          "rushing_tds": ru, "receiving_tds": re, "passing_tds": pa}
         for pid, ru, re, pa in player_tds]
    )


def _actual_for(player_id):
    """The `actual_value` the grader stored for one resolved anytime_td row."""
    row = _resolved_row(player_id)
    return row["actual_value"]


def _resolved_row(player_id):
    with contextlib.closing(store._connect()) as conn:
        return pd.read_sql(
            "SELECT actual_value FROM player_prop_predictions "
            "WHERE player_id = ? AND market = 'anytime_td'",
            conn, params=[player_id],
        ).iloc[0]


def test_the_grader_excludes_passing_tds_from_anytime_td():
    """The grader's truth must be the definition the classifier was fitted on.

    This is the counterpart to `test_player_usage.py::test_a_qb_game_with_passing_tds_only_is_not_an_anytime_td`.
    `reconcile_player_prop_predictions` used to carry its own inline sum of
    rushing + receiving + passing, so changing the model's label alone would have
    left every QB's pick graded against the OLD truth -- the model predicting one
    thing and the scorecard calling it wrong. Both now call
    `player_usage.anytime_td_actual`.

    Four players, one game: passing only, rushing only, receiving only, and
    passing plus receiving. Only the last two are anytime TDs.
    """
    store.record_game_predictions([_future_game()])
    store.record_player_prop_predictions([
        _prop("qb_pass", "anytime_td", 0.60),
        _prop("qb_rush", "anytime_td", 0.60),
        _prop("wr_rec", "anytime_td", 0.60),
        _prop("qb_both", "anytime_td", 0.60),
    ])
    assert store.reconcile_player_prop_predictions(_td_stats(
        ("qb_pass", 0, 0, 3),   # passing alone -> not an anytime TD
        ("qb_rush", 1, 0, 2),   # rushing, plus passing -> yes
        ("wr_rec", 0, 1, 0),    # receiving -> yes
        ("qb_both", 0, 1, 1),   # receiving, plus passing -> yes
    )) == 4

    assert _actual_for("qb_pass") == 0.0
    assert _actual_for("qb_rush") == 1.0
    assert _actual_for("wr_rec") == 1.0
    assert _actual_for("qb_both") == 1.0


def test_the_grader_and_the_training_frame_agree_on_the_definition():
    """The definition cannot drift, because both call one function.

    A tautology on its own -- which is the point. It fails if someone re-inlines
    the sum in either call site, which is exactly the regression this asserts
    against: the two sites were written separately and that is how they came to
    disagree.
    """
    from nfl_predictor.features import player_usage

    frame = pd.DataFrame([
        {"player_id": "a", "player_name": "A", "position": "QB", "recent_team": "BAL",
         "season": 2025, "week": 1, "passing_yards": 300, "passing_tds": 4,
         "rushing_yards": 0, "rushing_tds": 0, "receiving_yards": 0, "receiving_tds": 0,
         "receptions": 0, "targets": 0, "carries": 0},
        {"player_id": "b", "player_name": "B", "position": "WR", "recent_team": "BAL",
         "season": 2025, "week": 1, "passing_yards": 0, "passing_tds": 0,
         "rushing_yards": 0, "rushing_tds": 0, "receiving_yards": 70, "receiving_tds": 1,
         "receptions": 5, "targets": 8, "carries": 0},
    ])
    labelled, _ = player_usage.build_player_training_frame(frame)
    assert labelled["anytime_td"].tolist() == [
        player_usage.anytime_td_actual(0, 0),   # QB, passing only
        player_usage.anytime_td_actual(0, 1),   # WR, receiving
    ]
    assert labelled["anytime_td"].tolist() == [0, 1]


def test_anytime_td_actual_handles_missing_td_columns():
    """The grader's row comes from a join that can leave a TD column absent or
    NaN, and `NaN + 0` is NaN, which is `> 0` False. Rushing must still carry."""
    from nfl_predictor.features import player_usage

    assert player_usage.anytime_td_actual(None, 0) == 0.0
    assert player_usage.anytime_td_actual(float("nan"), 0) == 0.0
    assert player_usage.anytime_td_actual(float("nan"), 1) == 1.0
    assert player_usage.anytime_td_actual(1, float("nan")) == 1.0
    assert player_usage.anytime_td_actual(0, 0) == 0.0


def test_anytime_td_confidence_buckets_group_by_predicted_probability():
    store.record_game_predictions([_future_game()])
    store.record_player_prop_predictions([
        _prop("p1", "anytime_td", 0.55),
        _prop("p2", "anytime_td", 0.65),
        _prop("p3", "anytime_td", 0.75),
    ])
    assert store.reconcile_player_prop_predictions(_td_stats(
        ("p1", 1, 0, 0), ("p2", 0, 0, 0), ("p3", 0, 1, 0),
    )) == 3

    buckets = {
        b["label"]: b
        for b in store.get_track_record()["player_props"]["anytime_td"]["confidence_buckets"]
    }
    assert buckets["50-60%"]["n"] == 1
    assert buckets["50-60%"]["hit_rate"] == 1.0
    assert buckets["60-70%"]["n"] == 1
    assert buckets["60-70%"]["hit_rate"] == 0.0
    assert buckets["70%+"]["n"] == 1
    assert buckets["70%+"]["hit_rate"] == 1.0


def test_confidence_buckets_have_agreed_keys():
    """Contract test: confidence bucket objects must carry exactly
    {label, n, hit_rate}. Renaming any key breaks the frontend
    guard b.n > 0 and the section silently never renders."""
    store.record_game_predictions([_future_game()])
    store.record_player_prop_predictions([
        _prop("p1", "anytime_td", 0.55),
        _prop("p2", "anytime_td", 0.65),
        _prop("p3", "anytime_td", 0.75),
    ])
    assert store.reconcile_player_prop_predictions(_td_stats(
        ("p1", 1, 0, 0), ("p2", 0, 0, 0), ("p3", 0, 1, 0),
    )) == 3
    buckets = store.get_track_record()["player_props"]["anytime_td"]["confidence_buckets"]
    agreed_keys = {"label", "n", "hit_rate"}
    for bucket in buckets:
        assert set(bucket.keys()) == agreed_keys, (
            f"bucket {bucket} has keys {set(bucket.keys())}, "
            f"expected exactly {agreed_keys}"
        )


def test_yardage_markets_report_signed_bias_as_predicted_minus_actual():
    """Consistent overprediction must show positive signed bias (bias is
    mean(predicted - actual)), while unsigned MAE stays positive."""
    store.record_game_predictions([_future_game()])
    store.record_player_prop_predictions([
        _prop("p1", "rushing_yards", 100.0),
        _prop("p2", "rushing_yards", 100.0),
    ])
    stats = pd.DataFrame([
        {"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 90},
        {"game_id": "2025_01_BAL_KC", "player_id": "p2", "rushing_yards": 80},
    ])
    assert store.reconcile_player_prop_predictions(stats) == 2

    summary = store.get_track_record()["player_props"]["rushing_yards"]
    assert summary["mean_absolute_error"] == pytest.approx(15.0)
    assert summary["mean_signed_error"] == pytest.approx(15.0)


def test_signed_bias_cancels_when_over_and_under_predictions_balance_out():
    """Signed bias distinguishes systematic skew from noise: symmetric
    errors give ~zero bias even though MAE is clearly nonzero."""
    store.record_game_predictions([_future_game()])
    store.record_player_prop_predictions([
        _prop("p1", "rushing_yards", 100.0),
        _prop("p2", "rushing_yards", 80.0),
    ])
    stats = pd.DataFrame([
        {"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 90},
        {"game_id": "2025_01_BAL_KC", "player_id": "p2", "rushing_yards": 90},
    ])
    assert store.reconcile_player_prop_predictions(stats) == 2

    summary = store.get_track_record()["player_props"]["rushing_yards"]
    assert summary["mean_absolute_error"] == pytest.approx(10.0)
    assert summary["mean_signed_error"] == pytest.approx(0.0)


def test_yardage_summary_breaks_down_mae_by_position():
    store.record_game_predictions([_future_game()])
    store.record_player_prop_predictions([
        _prop("p1", "rushing_yards", 100.0, position="RB"),
        _prop("p2", "rushing_yards", 100.0, position="RB"),
        _prop("p3", "rushing_yards", 100.0, position="WR"),
    ])
    stats = pd.DataFrame([
        {"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 90},
        {"game_id": "2025_01_BAL_KC", "player_id": "p2", "rushing_yards": 80},
        {"game_id": "2025_01_BAL_KC", "player_id": "p3", "rushing_yards": 95},
    ])
    assert store.reconcile_player_prop_predictions(stats) == 3

    by_position = {
        g["position"]: g
        for g in store.get_track_record()["player_props"]["rushing_yards"]["by_position"]
    }
    assert by_position["RB"]["n_resolved"] == 2
    assert by_position["RB"]["mean_absolute_error"] == pytest.approx(15.0)
    assert by_position["WR"]["n_resolved"] == 1
    assert by_position["WR"]["mean_absolute_error"] == pytest.approx(5.0)


def test_record_player_prop_predictions_persists_position():
    import contextlib

    store.record_player_prop_predictions([_prop("p1", "rushing_yards", 85.0, position="RB")])

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql("SELECT position FROM player_prop_predictions", conn).iloc[0]

    assert row["position"] == "RB"


def test_rows_recorded_without_position_stay_in_overall_metrics_only():
    """Legacy rows (recorded before the position column existed) keep
    counting toward overall MAE but are excluded from per-position groups."""
    store.record_game_predictions([_future_game()])
    store.record_player_prop_predictions([_prop("p1", "rushing_yards", 100.0)])
    stats = pd.DataFrame([
        {"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 90},
    ])
    assert store.reconcile_player_prop_predictions(stats) == 1

    summary = store.get_track_record()["player_props"]["rushing_yards"]
    assert summary["n_resolved"] == 1
    assert summary["mean_absolute_error"] == pytest.approx(10.0)
    assert summary["by_position"] == []


def test_receptions_and_carries_appear_in_track_record():
    """Phase 8 added receptions/carries as recorded markets; the track
    record must summarize them like the other yardage-style markets."""
    store.record_game_predictions([_future_game()])
    store.record_player_prop_predictions([
        _prop("p1", "receptions", 5.0, position="WR"),
        _prop("p2", "carries", 18.0, position="RB"),
    ])
    stats = pd.DataFrame([
        {"game_id": "2025_01_BAL_KC", "player_id": "p1", "receptions": 6},
        {"game_id": "2025_01_BAL_KC", "player_id": "p2", "carries": 20},
    ])
    assert store.reconcile_player_prop_predictions(stats) == 2

    record = store.get_track_record()["player_props"]
    assert record["receptions"]["n_resolved"] == 1
    assert record["receptions"]["mean_absolute_error"] == pytest.approx(1.0)
    assert record["receptions"]["mean_signed_error"] == pytest.approx(-1.0)
    assert record["carries"]["n_resolved"] == 1
    assert record["carries"]["mean_absolute_error"] == pytest.approx(2.0)


def test_empty_track_record_includes_phase9_metric_keys():
    record = store.get_track_record()["player_props"]

    assert record["anytime_td"]["confidence_buckets"] == []
    assert record["rushing_yards"]["mean_signed_error"] is None
    assert record["rushing_yards"]["by_position"] == []


def test_get_predictions_for_week_marks_picks_rebuilt_after_kickoff():
    """A pick backfilled after the game (record_resolved_game_predictions) is
    not a pre-kickoff call: the week view must say so, so the site can label
    it and leave it out of the record."""
    store.record_game_predictions([_future_game(game_id="g1")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 30, "away_score": 20}]))
    store.record_resolved_game_predictions([{
        "game_id": "g3", "home_team": "NYJ", "away_team": "BUF", "commence_time": "2025-09-07T17:00:00+00:00",
        "home_win_prob": 0.4, "away_win_prob": 0.6, "actual_home_score": 10, "actual_away_score": 24,
    }])
    games_df = pd.DataFrame([
        {"game_id": "g1", "home_team": "BAL", "away_team": "KC"},
        {"game_id": "g3", "home_team": "NYJ", "away_team": "BUF"},
    ])

    by_id = {row["game_id"]: row for row in store.get_predictions_for_week(2025, 1, games_df)}

    assert by_id["g1"]["rebuilt"] is False
    assert by_id["g3"]["rebuilt"] is True
    assert by_id["g3"]["status"] == "resolved"


def test_the_track_record_counts_a_backfilled_pick_and_reports_it_apart():
    """Kevin's 2026-10-01 reversal: a pick recorded after its own kickoff is still a recorded
    pick, so it counts toward the headline. It used to be excluded outright, which meant a re-run
    of the models made a past game stop counting and the record emptied out on every change.

    `n_rebuilt` is retained and still reports those rows; it no longer means "not counted", and the
    pre-kickoff figure beside the headline is the one that leaves them out."""
    store.record_game_predictions([_future_game(game_id="g1")])
    store.reconcile_game_predictions(pd.DataFrame([{"game_id": "g1", "home_score": 10, "away_score": 20}]))
    store.record_resolved_game_predictions([{
        "game_id": "g3", "home_team": "H", "away_team": "A", "commence_time": "2025-09-07T17:00:00+00:00",
        "home_win_prob": 0.4, "away_win_prob": 0.6, "actual_home_score": 10, "actual_away_score": 24,
    }])

    games = store.get_track_record()["games"]

    assert games["n_resolved"] == 2
    assert games["n_rebuilt"] == 1
    assert games["pre_kickoff"]["n_resolved"] == 1
    assert games["n_resolved"] == games["pre_kickoff"]["n_resolved"] + games["n_rebuilt"]


def test_an_unreadable_snapshot_time_is_never_labelled_pre_kickoff():
    """Fails closed on the LABEL, which is what the flag is for. Note what it no longer means: the
    pick is still counted, because 'cannot prove it was made before kickoff' is a reason to withhold
    a claim, not a reason to drop a recorded prediction from the record."""
    assert store._snapshotted_after_kickoff("not a time", "2025-09-07T17:00:00+00:00") is True
    assert store._made_before_kickoff("not a time", "2025-09-07T17:00:00+00:00") is False


# --- a missing probability must not become a recorded verdict ------------------
#
# The same defect shape as CFB's `tracking/store.py`, at the same line numbers, and
# the two files are evidently the same lineage. What makes it serious here is the
# destination: these values are the **track record**, the page a reader uses to
# judge the model.
#
# `_compute_hits` guarded on the LINE being present and then wrote
# `(home_cover_prob or 0) >= (away_cover_prob or 0)`. The probabilities were never
# checked, so a game with a spread but no cover probabilities -- what the odds feed
# produces when it has a spread without a matching market -- was graded as though
# the model had called `0.0 >= 0.0`, which `>=` resolves to the home side.

def test_reconcile_leaves_ats_null_when_the_line_is_present_but_the_cover_probabilities_are_not():
    import pandas as pd

    from nfl_predictor.tracking import store

    store.record_game_predictions([_future_game(
        home_spread_line=-3.5, home_cover_prob=None, away_cover_prob=None,
    )])
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 31, "away_score": 24}])

    store.reconcile_game_predictions(results)

    import contextlib
    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql(
            "SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn
        ).iloc[0]

    # home did cover, so a fabricated call reads as ats_hit == 1. There was no call.
    assert pd.isna(row["ats_hit"]), (
        f"a missing cover probability must not be recorded as a verdict, got {row['ats_hit']}"
    )
    assert row["moneyline_hit"] == 1, "the moneyline call was real and is still gradable"


def test_reconcile_leaves_total_null_when_the_line_is_present_but_the_total_probabilities_are_not():
    import contextlib

    import pandas as pd

    from nfl_predictor.tracking import store

    store.record_game_predictions([_future_game(
        total_line=51.5, over_prob=None, under_prob=None,
    )])
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 31, "away_score": 24}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql(
            "SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn
        ).iloc[0]

    assert pd.isna(row["total_hit"]), (
        f"a missing over/under probability must not be recorded as a verdict, got {row['total_hit']}"
    )
    assert row["ats_hit"] == 1, "the ATS call was real (probs present) and is unaffected"


def test_reconcile_does_not_grade_a_one_sided_market():
    import contextlib

    import pandas as pd

    from nfl_predictor.tracking import store

    store.record_game_predictions([_future_game(
        home_spread_line=-3.5, home_cover_prob=0.6, away_cover_prob=None,
    )])
    results = pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 31, "away_score": 24}])

    store.reconcile_game_predictions(results)

    with contextlib.closing(store._connect()) as conn:
        row = pd.read_sql(
            "SELECT * FROM game_predictions WHERE game_id = '2025_01_BAL_KC'", conn
        ).iloc[0]

    assert pd.isna(row["ats_hit"])


def test_get_game_verdict_does_not_re_derive_a_prediction_from_missing_probabilities():
    """The read path, for a row an older build already wrote.

    New rows never reach this state, but the deployed database already holds them:
    every game the old code snapshotted with a spread line and no cover
    probabilities was written with a fabricated `ats_hit`. Those rows are permanent
    unless something backfills them, so `get_game_verdict` must not manufacture a
    `predicted` side next to a stored `hit` flag.
    """
    import contextlib

    import pandas as pd

    from nfl_predictor.tracking import store

    store.record_game_predictions([_future_game(
        home_spread_line=-3.5, home_cover_prob=None, away_cover_prob=None,
    )])
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            "UPDATE game_predictions SET resolved = 1, ats_hit = 1, moneyline_hit = 1, "
            "actual_home_score = 30, actual_away_score = 20 WHERE game_id = '2025_01_BAL_KC'"
        )

    verdict = store.get_game_verdict("2025_01_BAL_KC")

    assert verdict is not None
    assert verdict["ats"] is None, (
        f"a null-probability row must not report a predicted side, got {verdict['ats']}"
    )
    assert verdict["moneyline"]["hit"] is True
    assert verdict["actual_home_score"] == 30
    assert verdict["home_spread_line"] == -3.5


def test_get_game_verdict_keeps_reconciling_markets_when_probabilities_are_present():
    import pandas as pd

    from nfl_predictor.tracking import store

    store.record_game_predictions([_future_game()])
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": "2025_01_BAL_KC", "home_score": 30, "away_score": 20}])
    )

    verdict = store.get_game_verdict("2025_01_BAL_KC")

    assert verdict["ats"] is not None
    assert verdict["ats"]["predicted"] == "home_cover"
    assert verdict["ats"]["hit"] is True
    assert verdict["totals"] is not None


# --- the pre-kickoff guard on the prop path ------------------------------------
#
# The games path refuses to count a pick snapshotted at/after kickoff
# (get_track_record / get_feed_predictions / get_calibration all apply
# `_snapshotted_after_kickoff`). The prop path grades in
# `reconcile_player_prop_predictions` and summarizes in
# `_summarize_player_props`, and neither carried the guard: every unresolved
# row that joined to stats was graded, and every resolved row counted. The
# prop table has no commence_time, so both guard it by joining to
# game_predictions on game_id (that table's primary key, so the join is
# many-to-one and safe) and failing closed, exactly like the games path.


def _insert_game(game_id, commence_time):
    """Insert a game row directly. `record_game_predictions` refuses
    past-kickoff games, but the guard's post-kickoff cases need one."""
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob)
            VALUES (?, 'H', 'A', ?, '2026-01-01T00:00:00+00:00', 0.5, 0.5)
            """,
            (game_id, commence_time),
        )


def _insert_prop(game_id, player_id, snapshotted_at, market="rushing_yards",
                 predicted=85.0, resolved=False, actual=None):
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value, snapshotted_at,
                 resolved, actual_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, player_id, f"Player {player_id}", market, predicted,
             snapshotted_at, 1 if resolved else 0, actual),
        )


def _resolved_flags(game_id):
    with contextlib.closing(store._connect()) as conn:
        rows = pd.read_sql(
            "SELECT player_id, resolved FROM player_prop_predictions WHERE game_id = ?",
            conn, params=(game_id,),
        )
    return dict(zip(rows["player_id"], rows["resolved"]))


def test_reconcile_skips_prop_rows_snapshotted_after_kickoff():
    """A prop snapshot taken at/after its game's kickoff is a reconstruction,
    not a pick: the same rule the games path applies. The prop table carries
    no commence_time, so the guard joins to game_predictions on game_id."""
    _insert_game("g_post", "2000-09-04T20:20:00+00:00")  # long past
    store.record_player_prop_predictions([_prop(game_id="g_post")])
    stats = pd.DataFrame([{"game_id": "g_post", "player_id": "p1", "rushing_yards": 92}])

    assert store.reconcile_player_prop_predictions(stats) == 0
    assert _resolved_flags("g_post")["p1"] == 0


def test_reconcile_grades_prop_rows_snapshotted_before_kickoff():
    """The guard is a filter, not a block: a genuine pre-kickoff snapshot is
    still graded."""
    store.record_game_predictions([_future_game(game_id="g_pre")])
    store.record_player_prop_predictions([_prop(game_id="g_pre")])
    stats = pd.DataFrame([{"game_id": "g_pre", "player_id": "p1", "rushing_yards": 92}])

    assert store.reconcile_player_prop_predictions(stats) == 1
    assert _resolved_flags("g_pre")["p1"] == 1


def test_reconcile_kickoff_boundary_is_exact():
    """The boundary is the games path's `>=`: a row exactly at kickoff is
    excluded, one second before is included. Pinned so the boundary cannot
    drift -- a test that only checked 'some post-kickoff row is skipped' would
    pass against an off-by-one guard."""
    _insert_game("g_boundary", "2099-09-04T20:20:00+00:00")
    _insert_prop("g_boundary", "p_at", "2099-09-04T20:20:00+00:00")      # exactly at kickoff
    _insert_prop("g_boundary", "p_before", "2099-09-04T20:19:59+00:00")  # one second before
    stats = pd.DataFrame([
        {"game_id": "g_boundary", "player_id": "p_at", "rushing_yards": 92},
        {"game_id": "g_boundary", "player_id": "p_before", "rushing_yards": 92},
    ])

    assert store.reconcile_player_prop_predictions(stats) == 1

    resolved = _resolved_flags("g_boundary")
    assert resolved["p_at"] == 0
    assert resolved["p_before"] == 1


def test_reconcile_fails_closed_on_unparseable_snapshot_time():
    """When the timing can't be proven, the honest default is 'after
    kickoff' -- the same fail-closed rule as the games path
    (test_an_unreadable_snapshot_time_counts_as_rebuilt)."""
    store.record_game_predictions([_future_game(game_id="g_bad")])
    _insert_prop("g_bad", "p1", "not a time")
    stats = pd.DataFrame([{"game_id": "g_bad", "player_id": "p1", "rushing_yards": 92}])

    assert store.reconcile_player_prop_predictions(stats) == 0
    assert _resolved_flags("g_bad")["p1"] == 0


def test_reconcile_skips_orphan_prop_rows_with_no_game():
    """A prop row whose game is absent from game_predictions cannot be proven
    pre-kickoff, so it is not graded -- fail closed on the missing join, not
    silently graded on the strength of a timestamp alone."""
    store.record_player_prop_predictions([_prop(game_id="g_orphan")])
    stats = pd.DataFrame([{"game_id": "g_orphan", "player_id": "p1", "rushing_yards": 92}])

    assert store.reconcile_player_prop_predictions(stats) == 0
    assert _resolved_flags("g_orphan")["p1"] == 0


def test_the_prop_summary_counts_post_kickoff_rows_and_reports_the_pre_kickoff_subset_apart():
    """The 2026-10-01 reversal applied to the prop markets, and this is where it matters most: in
    the real database every prop row was recorded after its game's kickoff, so under the old rule
    the anytime_td market had no track record at all (n_resolved 0, brier None) -- a market the
    model prices every day, reported as unmeasured.

    Now the headline counts all three rows and `pre_kickoff` carries the one pre-kickoff row. Both
    market families are planted, so a swap applied to only one of them cannot pass. `n_rebuilt` is
    retained and reconciles the two figures."""
    _insert_game("g_pre", "2099-09-04T20:20:00+00:00")
    _insert_game("g_post", "2000-09-04T20:20:00+00:00")
    _insert_prop("g_pre", "p_pre", "2099-09-01T00:00:00+00:00",
                 market="rushing_yards", predicted=90.0, resolved=True, actual=80.0)
    _insert_prop("g_post", "p_post_yards", "2001-01-01T00:00:00+00:00",
                 market="rushing_yards", predicted=10.0, resolved=True, actual=80.0)
    _insert_prop("g_post", "p_post_td", "2001-01-01T00:00:00+00:00",
                 market="anytime_td", predicted=0.9, resolved=True, actual=1.0)

    props = store.get_track_record()["player_props"]

    assert props["n_rebuilt"] == 2
    # Headline: all three rows count. Yardage MAE is mean(|90-80|, |10-80|) = 40.
    assert props["rushing_yards"]["n_resolved"] == 2
    assert props["rushing_yards"]["mean_absolute_error"] == pytest.approx(40.0)
    # The anytime_td market is no longer empty -- the post-kickoff row counts.
    assert props["anytime_td"]["n_resolved"] == 1
    assert props["anytime_td"]["brier_score"] == pytest.approx(0.01)
    # The pre-kickoff figure beside it is exactly the one pre-kickoff row, in both families.
    assert props["pre_kickoff"]["rushing_yards"]["n_resolved"] == 1
    assert props["pre_kickoff"]["rushing_yards"]["mean_absolute_error"] == pytest.approx(10.0)
    assert props["pre_kickoff"]["anytime_td"]["n_resolved"] == 0
    assert props["pre_kickoff"]["anytime_td"]["brier_score"] is None
    # One post-kickoff row per family, so each market's headline is its pre-kickoff figure plus
    # one -- and n_rebuilt is the sum of those, never per-market.
    for market in ("rushing_yards", "anytime_td"):
        assert props[market]["n_resolved"] == props["pre_kickoff"][market]["n_resolved"] + 1
    assert props["n_rebuilt"] == 2

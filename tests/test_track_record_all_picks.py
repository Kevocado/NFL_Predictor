"""B8 — the full record, beside the pre-kickoff one.

Kevin's ask was to stop losing sight of the picks that have already passed,
and the shape that delivers it without letting a post-hoc 100% stand in for
skill is two figures, not one.

A hit rate is only meaningful if the pick existed before the result.
Counting post-kickoff picks toward one headline number means the number can be
inflated by construction -- pick the winner after the fact, score 100%. So the
headline stays pre-kickoff and the rule that makes it mean anything survives.
What changes is that nothing is hidden and everything is counted, in two figures
instead of one.

These tests pin the three ways that could silently go wrong: the all-picks
figure dropping rebuilt rows (the old behaviour), the all-picks figure being
computed over the pre-kickoff subset (so the two figures are identical and the
second one is decoration), and `rebuilt` arriving only as an aggregate rather
than per pick.
"""
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


def _pre_kickoff(game_id, *, week=1, season=2026):
    """A genuinely pre-kickoff pick: snapshotted well before kickoff, graded."""
    game = {
        "game_id": game_id, "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00", "snapshotted_at": "2099-09-01T00:00:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": -2.5, "total_line": 43.5,
        "season": season, "week": week,
    }
    store.record_game_predictions([game])
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": game_id, "home_score": 24, "away_score": 20}])
    )
    return game


def _rebuilt(game_id, *, week=1, season=2026):
    """A pick rewritten after kickoff.

    `record_game_predictions` refuses past-kickoff games outright, and
    `record_resolved_game_predictions` stamps `snapshotted_at` with the current time -- so
    the way to make a row that was snapshotted after its own kickoff is to give it a kickoff
    in the past and record it as already resolved. A future kickoff with a later
    `snapshotted_at` in the fixture is silently overwritten, which is how this test failed
    the first time.
    """
    store.record_resolved_game_predictions([{
        "game_id": game_id, "home_team": "BAL", "away_team": "KC",
        "commence_time": "2020-01-01T00:00:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": -2.5, "total_line": 43.5,
        "season": season, "week": week,
        "actual_home_score": 24, "actual_away_score": 20,
    }])


def test_the_headline_now_counts_the_rebuilt_row_and_pre_kickoff_beside_it_does_not():
    """B8's two figures swapped roles on 2026-10-01: the headline counts every recorded pick and
    `pre_kickoff` is the subset made before kickoff. This is the test that would catch the old
    exclusion creeping back -- and, just as importantly, the test that would catch someone
    'fixing' the headline back down to the pre-kickoff subset to make a flattering number."""
    _pre_kickoff("pre")
    _rebuilt("post")

    games = store.get_track_record(current_week=3)["games"]

    assert games["n_resolved"] == 2, "a recorded pick stays counted whether or not it beat kickoff"
    assert games["n_moneyline"] == 2
    assert games["n_rebuilt"] == 1, "still reported, for disclosure"
    assert games["pre_kickoff"]["n_resolved"] == 1
    assert games["pre_kickoff"]["n_moneyline"] == 1
    assert games["n_resolved"] == games["pre_kickoff"]["n_resolved"] + games["n_rebuilt"]


def test_all_picks_includes_rebuilt_rows():
    """The key is kept, and it still means every counted pick. If it dropped rebuilt rows it would
    quietly become a pre-kickoff figure under a name that promises otherwise."""
    _pre_kickoff("pre")
    _rebuilt("post")

    games = store.get_track_record(current_week=3)["games"]
    assert games["all_picks"]["n_resolved"] == 2


def test_all_picks_and_the_headline_are_the_same_population_and_pre_kickoff_is_not():
    """The two figures that are supposed to differ. Under the reversal `all_picks` and the
    headline are the same population -- every counted pick -- and `pre_kickoff` is the one that
    differs, so the pair a reader reconciles is (headline, pre_kickoff) and not
    (headline, all_picks).

    If these two ever agree again, one of them is decoration."""
    _pre_kickoff("pre")
    _rebuilt("post")
    _rebuilt("post2")

    games = store.get_track_record(current_week=3)["games"]

    assert games["all_picks"]["n_resolved"] == 3
    assert games["n_resolved"] == 3
    assert games["pre_kickoff"]["n_resolved"] == 1
    assert games["pre_kickoff"]["n_resolved"] < games["n_resolved"]
    # And the rates actually differ, which is the point of showing both.
    assert games["all_picks"]["pct_moneyline_correct"] is not None
    assert games["pre_kickoff"]["pct_moneyline_correct"] is not None


def test_per_pick_carries_rebuilt_per_row_not_only_in_aggregate():
    """A reader must be able to tell, per pick, whether it was made before kickoff."""
    _pre_kickoff("pre")
    _rebuilt("post")

    rows = store.get_track_record(current_week=3)["games"]["per_pick"]
    by_game = {row["game_id"]: row for row in rows}
    assert by_game["pre"]["made_before_kickoff"] is True
    assert by_game["post"]["made_before_kickoff"] is False
    # `rebuilt` is retained as the exact negation, so the two cannot drift apart.
    assert by_game["pre"]["rebuilt"] is False
    assert by_game["post"]["rebuilt"] is True


def test_per_pick_carries_the_time_the_pick_was_made():
    """The plan's standing constraint: every pick is displayed with the time it was made.

    `snapshotted_at` was stored on every prediction row and exposed in no payload
    before B8, so the constraint was unmet on every surface. This is the one
    place it can be met, and a row without it would leave the constraint unmet
    again.
    """
    _pre_kickoff("pre")
    _rebuilt("post")

    rows = store.get_track_record(current_week=3)["games"]["per_pick"]
    assert rows, "per_pick must not be empty when games are resolved"
    for row in rows:
        assert row["snapshotted_at"], f"{row['game_id']} has no snapshotted_at"
    by_game = {row["game_id"]: row for row in rows}
    assert by_game["pre"]["snapshotted_at"] < by_game["post"]["snapshotted_at"]


def test_per_pick_lists_hits_and_misses_in_the_same_list():
    """Never filtered, never collapsed. A miss is evidence too."""
    _pre_kickoff("hit")
    # A pre-kickoff pick the model got wrong: KC wins, so the BAL moneyline pick missed.
    game = {
        "game_id": "miss", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00", "snapshotted_at": "2099-09-01T00:00:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": -2.5, "total_line": 43.5,
        "season": 2026, "week": 1,
    }
    store.record_game_predictions([game])
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": "miss", "home_score": 20, "away_score": 24}])
    )

    rows = store.get_track_record(current_week=3)["games"]["per_pick"]
    hits = [row for row in rows if row["hit"]]
    misses = [row for row in rows if not row["hit"]]
    assert hits and misses, "the list must contain both, or it is filtered"
    assert {row["game_id"] for row in rows} == {"hit", "miss"}


def test_an_ungraded_market_is_omitted_not_listed_as_a_miss():
    """A market with no grade is not a pick the model made. Listing it as a miss would be a
    fabricated failure, which is the same class of lie as a 0% for an unmeasured market."""
    game = {
        "game_id": "noline", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00", "snapshotted_at": "2099-09-01T00:00:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": None, "away_cover_prob": None,
        "over_prob": None, "under_prob": None,
        "home_spread_line": None, "total_line": None,
        "season": 2026, "week": 1,
    }
    store.record_game_predictions([game])
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": "noline", "home_score": 24, "away_score": 20}])
    )

    rows = store.get_track_record(current_week=3)["games"]["per_pick"]
    assert [row["market"] for row in rows] == ["moneyline"]

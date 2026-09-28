"""B3 — a week with no tracked games must still appear, marked as untracked.

The record this page shows is the season's shape. Grouping by week and emitting
a row per group only ever draws the weeks the tracker happened to cover, so a
week with no picks does not read as "not tracked" -- it reads as *not part of the
season*, which is a different and much stronger claim. A reader cannot tell a
gap from an absence.

So: one row for every elapsed week, `tracked: false` where there was nothing,
and every accuracy `None` there. Never omitted, never 0.0.

Also, the per-week rows move from `weekly_trend` to `weekly`, and the old key
goes away rather than lingering. Two keys for one fact is how this page ended up
rendering nothing at all in B1.
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


def _resolve(game_id, week, *, season=2026, home_wins=True):
    """A finished, genuinely pre-kickoff pick: snapshot it, then grade it."""
    game = {
        "game_id": game_id, "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": -2.5, "total_line": 43.5,
        "season": season, "week": week,
    }
    store.record_game_predictions([game])
    home_score, away_score = (24, 20) if home_wins else (20, 24)
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": game_id, "home_score": home_score, "away_score": away_score}])
    )
    return game


def _weekly(current_week, season=2026):
    games = store.get_track_record(current_week=current_week, season=season)["games"]
    return {row["week"]: row for row in games["weekly"]}


def test_a_week_with_no_picks_is_present_and_marked():
    """Weeks 1 and 3 were tracked, week 2 was not, and week 4 has not been played.
    All four get a row. The two with nothing are marked, not dropped."""
    _resolve("w1g", 1)
    _resolve("w3g", 3)

    weekly = _weekly(current_week=4)

    assert sorted(weekly) == [1, 2, 3, 4]
    assert weekly[1]["tracked"] is True and weekly[1]["n_games"] == 1
    assert weekly[3]["tracked"] is True and weekly[3]["n_games"] == 1
    assert weekly[2]["tracked"] is False
    assert weekly[4]["tracked"] is False


def test_an_untracked_week_is_marked_rather_than_zeroed():
    """Zero means 'measured and got none of them'. A week nobody bet on measures
    nothing, so every rate is None -- not 0.0, which is a legible claim that the
    model was wrong all week."""
    _resolve("w1g", 1)

    week2 = _weekly(current_week=2)[2]

    assert week2["tracked"] is False
    assert week2["n_games"] == 0
    for key in ("pct_moneyline_correct", "pct_ats_correct", "pct_totals_correct"):
        assert week2[key] is None, f"{key} must be None on an untracked week, never 0.0"


def test_an_untracked_week_still_carries_all_three_counts_at_zero():
    """The counts are facts even when the rates are not: 0 ATS graded in week 2 is a
    real measurement of how many games were graded, and the page needs the field
    present to draw a dash rather than a blank."""
    _resolve("w1g", 1)

    week2 = _weekly(current_week=2)[2]

    assert week2["n_moneyline"] == 0
    assert week2["n_ats"] == 0
    assert week2["n_totals"] == 0


def test_a_record_with_no_games_at_all_still_lists_every_elapsed_week():
    """The early-return path used to hand back `[]`, so a page opened before the
    first pick resolved showed an empty section rather than five not-tracked weeks."""
    weekly = _weekly(current_week=5)

    assert sorted(weekly) == [1, 2, 3, 4, 5]
    assert all(row["tracked"] is False and row["n_games"] == 0 for row in weekly.values())


def test_week_one_appears_even_with_nothing_tracked():
    """Off-by-one guard: the range starts at 1, not 0, and a single elapsed week is
    one row."""
    weekly = _weekly(current_week=1)

    assert list(weekly) == [1]
    assert weekly[1]["tracked"] is False


def test_every_elapsed_week_carries_all_three_markets_in_one_row():
    """'One request, not three' -- the week view needs moneyline, ATS and totals for
    the same week, so they travel together. Each with its own count, because the
    three denominators differ."""
    _resolve("w2g", 2)

    row = _weekly(current_week=3)[2]

    for grade, count in (
        ("pct_moneyline_correct", "n_moneyline"),
        ("pct_ats_correct", "n_ats"),
        ("pct_totals_correct", "n_totals"),
    ):
        assert grade in row and count in row
        assert row[count] == 1


def test_a_week_number_from_another_season_is_not_mixed_into_this_one():
    """Week numbering restarts every season, so an unfiltered group-by collapses
    2025 week 12 into 2026 week 12 and reports a 2025 accuracy on a 2026 chart.
    The weekly rows are scoped to one season."""
    _resolve("old", 12, season=2025)
    _resolve("new", 2, season=2026)

    weekly = _weekly(current_week=3, season=2026)

    assert sorted(weekly) == [1, 2, 3]
    assert weekly[2]["n_games"] == 1
    assert weekly[1]["tracked"] is False


def test_the_window_defaults_to_the_latest_week_in_the_data():
    """Callers that do not know the calendar (facts.py, a test) still get a complete,
    gap-free list -- bounded by the newest week actually recorded rather than by
    nothing at all."""
    _resolve("w1g", 1)
    _resolve("w3g", 3)

    games = store.get_track_record()["games"]

    assert [row["week"] for row in games["weekly"]] == [1, 2, 3]


def test_weekly_trend_is_gone_rather_than_kept_alongside():
    """One shape for one fact. B1 is what a second, stale key on this payload costs:
    `types.ts` read `{label, n}` while the API emitted `{bucket, n_resolved}`, the
    guard `n > 0` was permanently false, and the calibration section had never
    rendered on this site in any deploy."""
    _resolve("w1g", 1)

    games = store.get_track_record(current_week=2)["games"]

    assert "weekly" in games
    assert "weekly_trend" not in games

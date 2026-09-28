"""B2 — the weekly rows must carry volume and accuracy as TWO numbers.

The defect this file exists for is in the frontend, `TrackRecordPage.tsx`:

    width = (n_games / max_games) * pct_moneyline_correct * 100

One fused number. Volume and accuracy are separate questions, and multiplying
them says that a 16-game week at 50% is a better week than a 3-game week at
90% -- which is not a claim anyone can make, because the 3-game week's 90% is
one game different from 50%.

The visual fix is frontend work and is not done here. What IS done here is the
half a chart needs and cannot invent: every weekly row reports the number of
games in that week next to the accuracies, so volume can be drawn as volume and
accuracy on an accuracy scale.

`n_games` already existed on the row. What did not exist is the count *behind
each accuracy*: `pct_ats_correct` over an unknown denominator is not a fact, and
this page is exactly where that mistake got made (B1, where `n` was read from
the wrong key and a whole section never rendered).
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


def _snapshot(game_id, week, *, home_wins, spread=None, total=None):
    """Snapshot a game pre-kickoff, then reconcile it with a final score.

    `home_wins` drives the moneyline grade: the model's side is fixed (home
    favoured at 0.6/0.4), so the home team winning is a hit and losing a miss.
    """
    game = {
        "game_id": game_id, "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.55, "away_cover_prob": 0.45,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": spread, "total_line": total,
        "season": 2099, "week": week,
    }
    store.record_game_predictions([game])
    home_score, away_score = (24, 20) if home_wins else (20, 24)
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": game_id, "home_score": home_score, "away_score": away_score}])
    )
    return game


def test_weekly_row_reports_games_and_accuracy_as_separate_numbers():
    """The exact shape the fused width formula gets wrong.

    Week 1 is four games at 50%. Week 2 is ONE game, won. A width built as
    `(n_games / max_games) * accuracy` draws week 2 at 0.25 -- a quarter of the
    track for a perfect week. The payload has to be able to say otherwise, so
    `n_games` is the count and `pct_moneyline_correct` is the arithmetic mean of
    that week's grades, with nothing multiplied into either.
    """
    for i in range(4):
        _snapshot(f"w1g{i}", 1, home_wins=(i < 2))
    _snapshot("w2g0", 2, home_wins=True)

    weekly = {row["week"]: row for row in store.get_track_record()["games"]["weekly_trend"]}

    assert weekly[1]["n_games"] == 4
    assert weekly[1]["pct_moneyline_correct"] == pytest.approx(0.5)
    assert weekly[2]["n_games"] == 1
    # 1.0, not 0.25. This is the assertion that fails if anyone ever re-fuses
    # the two quantities inside the backend as well as the frontend.
    assert weekly[2]["pct_moneyline_correct"] == pytest.approx(1.0)


def test_weekly_row_carries_the_count_behind_every_accuracy():
    """A rate with no denominator is not a fact -- and the denominators differ.

    A game's ATS is only graded when the spread line and both cover
    probabilities are present, and its total only when the line and both
    over/under probabilities are. So one week can have 5 games, 3 ATS grades
    and 2 totals grades, and every one of those three numbers is different.
    """
    _snapshot("g1", 1, home_wins=True, spread=-2.5, total=43.5)
    _snapshot("g2", 1, home_wins=True, spread=-2.5, total=45.5)
    _snapshot("g3", 1, home_wins=False, spread=None, total=None)
    _snapshot("g4", 1, home_wins=False, spread=None, total=45.5)
    _snapshot("g5", 1, home_wins=True, spread=None, total=None)

    week1 = next(r for r in store.get_track_record()["games"]["weekly_trend"] if r["week"] == 1)

    assert week1["n_games"] == 5
    assert week1["n_moneyline"] == 5
    assert week1["n_ats"] == 2
    assert week1["n_totals"] == 3
    # And the rates are over exactly those subsets, not over all five games. The model's
    # over probability is 0.5/0.5, so it always picks over; only g1's 44 beats 43.5.
    assert week1["pct_ats_correct"] == pytest.approx(1.0)  # 2/2, both home-favoured and home won
    assert week1["pct_totals_correct"] == pytest.approx(1 / 3)


def test_a_market_with_no_grades_in_a_week_is_null_not_zero():
    """Zero means 'measured and got none of them'. No ATS line in the week means
    the ATS was never measured, and 0% would be a claim about games that were
    never graded."""
    _snapshot("g1", 1, home_wins=True, spread=None, total=None)

    week1 = next(r for r in store.get_track_record()["games"]["weekly_trend"] if r["week"] == 1)

    assert week1["n_ats"] == 0
    assert week1["pct_ats_correct"] is None
    assert week1["n_totals"] == 0
    assert week1["pct_totals_correct"] is None

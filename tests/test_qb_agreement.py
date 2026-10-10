from nfl_predictor.tools.qb_agreement import agreement


def test_agreement_counts_expected_equals_actual():
    actual = {("g1", "A"): "q1", ("g1", "B"): "q2", ("g2", "A"): "q1"}
    expected = {("g1", "A"): "q1", ("g1", "B"): "q9", ("g2", "A"): "q1"}
    out = agreement(actual, expected)
    assert out["n"] == 3 and out["agree"] == 2 and abs(out["rate"] - 2 / 3) < 1e-9


def test_games_without_an_expected_starter_are_reported_not_counted_as_agreement():
    out = agreement({("g1", "A"): "q1"}, {})
    assert out["n"] == 1 and out["agree"] == 0 and out["no_expectation"] == 1


def test_missing_values_on_either_side_never_count_as_agreement():
    """None actual with None expected is "we know nothing", not agreement.

    Failing-first for the review: `expected.get(k) == q` counts None == None as
    agreement, so a week with no starters known anywhere reported rate 1.0.
    """
    out = agreement({("g1", "A"): None}, {})
    assert out["agree"] == 0 and out["rate"] == 0.0
    out = agreement({("g1", "A"): None}, {("g1", "A"): None})
    assert out["agree"] == 0 and out["rate"] == 0.0
    out = agreement({("g1", "A"): "q1"}, {("g1", "A"): None})
    assert out["agree"] == 0

# --- added by the block-eval results PR ---------------------------------------

import csv as _csv
from pathlib import Path as _Path

import pytest

import nfl_predictor
from nfl_predictor.tools.qb_agreement import (
    expected_starters_serving_view,
    get_actual_starters,
    get_expected_starters,
)

# The committed raw output of the exact command in docs/nfl_block_eval_results.md, so the
# headline rate is reproducible from the CSV alone rather than trusted from memory.
# Resolved through the package, not __file__: the mutation harness copies tests into a
# temp dir, where __file__ no longer sits under the repo root.
QB_AGREEMENT_CSV = _Path(nfl_predictor.__file__).resolve().parents[2] / "output" / "qb_agreement_2024_w1-4.csv"


def test_committed_csv_reproduces_the_reported_agreement_rate():
    """The 0.922 headline for 2024 weeks 1-4 must fall out of the committed CSV."""
    assert QB_AGREEMENT_CSV.exists(), f"missing {QB_AGREEMENT_CSV}"
    with open(QB_AGREEMENT_CSV, newline="") as f:
        rows = list(_csv.DictReader(f))
    assert rows, "CSV is empty"
    n = len(rows)
    agree = sum(1 for r in rows if r["match"] == "True")
    assert (n, agree) == (128, 118), f"CSV no longer backs the reported numbers: {n} rows, {agree} agree"
    assert abs(agree / n - 0.922) < 1e-3


@pytest.mark.network
def test_committed_csv_has_no_fabricated_disagreements():
    """The review flagged ATL 'Matt Ryan' in 2024, which cannot be right -- he last
    played for ATL in 2021. Every id in the CSV must be a real 2024 QB."""
    from nfl_predictor.data import player_stats
    stats = player_stats.fetch_weekly_player_stats([2024])
    known = set(stats[stats["position"] == "QB"]["player_id"])
    with open(QB_AGREEMENT_CSV, newline="") as f:
        rows = list(_csv.DictReader(f))
    for r in rows:
        assert r["expected_qb_id"] in known, f"{r['team']} w{r['week']}: unknown expected id"
        assert r["actual_qb_id"] in known, f"{r['team']} w{r['week']}: unknown actual id"


@pytest.mark.network
def test_expected_starters_serving_view_covers_every_named_game_once_per_team():
    view = expected_starters_serving_view(2024, [1])
    assert len(view) >= 32
    teams = [t for _, t in view]
    assert len(teams) == len(set(teams)), "a team appears twice in one week"


@pytest.mark.network
def test_the_serving_view_is_built_from_the_depth_chart_not_the_box_score():
    """Both sides exist, but serving may only read the chart side.

    Failure-first for the review: if get_expected_starters were secretly calling
    get_actual_starters, the serving view would be identical to the box score by
    construction and the 0.922 agreement number would be meaningless.
    """
    expected = get_expected_starters(2024, 1)
    actual = get_actual_starters(2024, 1)
    assert set(expected) == set(actual)
    # Week 3 2024 is where they break, so agreement is truly < 1 vs the box score.
    exp_w3 = get_expected_starters(2024, 3)
    act_w3 = get_actual_starters(2024, 3)
    assert exp_w3 != act_w3, "week 3 has known disagreements; the two paths must differ"


# --- pre-kickoff only: no depth chart dated after the game --------------------

import pandas as pd


def _qb_chart(season, week, gsis, club="MIA"):
    # Real feed dtypes: week is float64 (NaN for offseason rows), depth_team is a STRING.
    return pd.DataFrame({
        "season": [season, season], "club_code": [club, club],
        "week": [float(week), float("nan")], "game_type": ["REG", "REG"],
        "depth_team": ["1", "1"], "depth_position": ["QB", "QB"], "gsis_id": [gsis, gsis],
    })


def _patch_feed(monkeypatch, charts_by_season):
    from nfl_predictor.data import depth_charts, schedules
    monkeypatch.setattr(depth_charts, "load_depth_charts", lambda s, d, **k: charts_by_season.get(s))
    games = pd.DataFrame({"game_id": ["g2"], "week": [2], "home_team": ["MIA"], "away_team": ["BUF"]})
    monkeypatch.setattr(schedules, "fetch_week_games", lambda s, w: games)


def test_a_chart_dated_after_the_game_is_never_the_expected_starter(monkeypatch):
    """Week 2 must not be answered from a week-5 chart (it did: resolve_chart's "earliest" fallback)."""
    _patch_feed(monkeypatch, {2024: _qb_chart(2024, 5, "future_qb")})
    assert get_expected_starters(2024, 2) == {}


def test_a_game_before_the_first_chart_uses_last_seasons_final_chart(monkeypatch):
    _patch_feed(monkeypatch, {2024: _qb_chart(2024, 5, "future_qb"), 2023: _qb_chart(2023, 18, "last_year_qb")})
    assert get_expected_starters(2024, 2) == {("g2", "MIA"): "last_year_qb"}

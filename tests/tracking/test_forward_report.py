"""the weekly markdown forward-test report.

This is the artifact Kevin reads instead of a dashboard, so its job is to state
the record without flattering it: hit rate against the 52.4% breakeven, mean CLV,
and the P(over) calibration buckets side by side.

The cases that matter are the empty and the thin ones. A report that reads well
on 60 picks and is silent about having 3 is worse than no report.
"""
from __future__ import annotations

import pytest

from nfl_predictor.tracking.forward_report import write_weekly_report


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    """Without this every test reads the developer's live tracking database, so
    the pick counts reflect whatever the real site recorded."""
    from nfl_predictor import config
    from nfl_predictor.tracking import store

    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    yield

BREAKEVEN = 0.524


def _graded_rows(n=60, hit_rate=0.55, mean_clv=1.2):
    """`clv` is set so the MEAN over the rows is exactly `mean_clv`: half the
    rows at +v and half at -v. The report's mean CLV must land on the number the
    test asked for, not on an average of a lumpy list."""
    rows = []
    hits = int(round(n * hit_rate))
    for i in range(n):
        rows.append({
            "game_id": f"G{i}", "player_id": f"00-{i}", "player_name": f"P{i}",
            "market": "receiving_yards", "side": "over" if i % 2 else "under",
            "predicted_value": 50.0, "snapshotted_at": "2026-09-05T00:00:00+00:00",
            "resolved": 1, "actual_value": 55.0,
            "line_at_snapshot": 52.5, "odds_at_snapshot": -110.0,
            "model_p_over": 0.60, "edge_vs_breakeven": 0.076,
            "closing_line": 52.5 + mean_clv,
            "clv": mean_clv, "hit": 1 if i < hits else 0,
        })
    return rows


def _seed(rows):
    from nfl_predictor.tracking import store

    # Reconciliation inner-joins game_predictions for the kickoff time and drops
    # any prop row whose game is absent, so the games have to exist. Its test
    # fixture had omitted this and every row silently stayed unresolved.
    store.record_game_predictions([{
        "game_id": row["game_id"], "home_team": "ALB", "away_team": "DEN",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.5, "away_win_prob": 0.5,
        "home_cover_prob": 0.5, "away_cover_prob": 0.5,
        "over_prob": 0.5, "under_prob": 0.5,
    } for row in rows])

    for row in rows:
        store.record_player_prop_predictions([{
            "game_id": row["game_id"], "player_id": row["player_id"],
            "player_name": row["player_name"], "market": row["market"],
            "position": "WR", "predicted_value": row["predicted_value"],
            "side": row["side"], "line_at_snapshot": row["line_at_snapshot"],
            "odds_at_snapshot": row["odds_at_snapshot"],
            "model_p_over": row["model_p_over"],
            "edge_vs_breakeven": row["edge_vs_breakeven"],
        }])
    store.record_closing_lines([
        {"game_id": r["game_id"], "player_id": r["player_id"],
         "market": r["market"], "closing_line": r["closing_line"]} for r in rows])
    for row in rows:
        store.reconcile_player_prop_predictions(_stats(row))


def _stats(row):
    import pandas as pd

    column = {"receiving_yards": "receiving_yards"}[row["market"]]
    return pd.DataFrame([{"game_id": row["game_id"], "player_id": row["player_id"],
                          column: row["actual_value"]}])


def test_report_contains_gate_metrics(tmp_path):
    _seed(_graded_rows(60, hit_rate=0.55, mean_clv=1.2))

    path = write_weekly_report(season=2026, week=4, out_dir=tmp_path)
    text = path.read_text()

    assert path.name == "2026-W4.md"
    assert "hit rate" in text.lower()
    assert "52.4" in text
    assert "clv" in text.lower()
    assert "2026" in text
    assert "week 4" in text.lower()


def test_report_states_the_pick_count(tmp_path):
    _seed(_graded_rows(60))

    text = write_weekly_report(season=2026, week=4, out_dir=tmp_path).read_text()

    assert "60" in text


def test_report_shows_mean_clv_with_its_sign(tmp_path):
    _seed(_graded_rows(60, mean_clv=1.2))

    text = write_weekly_report(season=2026, week=4, out_dir=tmp_path).read_text()

    assert "+1.2" in text or "+1.20" in text


def test_report_shows_negative_clv_as_negative(tmp_path):
    _seed(_graded_rows(60, mean_clv=-0.8))

    text = write_weekly_report(season=2026, week=4, out_dir=tmp_path).read_text()

    assert "-0.8" in text or "-0.80" in text


def test_report_includes_calibration_buckets(tmp_path):
    _seed(_graded_rows(60))

    text = write_weekly_report(season=2026, week=4, out_dir=tmp_path).read_text()

    assert "0.5" in text and "0.6" in text, "a P(over) bucket should be visible"


def test_empty_week_says_so_rather_than_reporting_zero_percent(tmp_path):
    """No rows seeded in this test, thanks to the per-test database."""
    path = write_weekly_report(season=2026, week=9, out_dir=tmp_path)
    text = path.read_text()

    assert "no picks" in text.lower()
    assert "0.0%" not in text, "an empty week must not render as a 0% hit rate"


def test_thin_sample_is_labelled_as_thin(tmp_path):
    _seed(_graded_rows(3, hit_rate=1.0, mean_clv=2.0))

    text = write_weekly_report(season=2026, week=5, out_dir=tmp_path).read_text()

    assert "3" in text
    assert any(word in text.lower() for word in ("thin", "small sample", "not yet"))


def test_report_gives_a_one_line_verdict(tmp_path):
    _seed(_graded_rows(60, hit_rate=0.55))

    text = write_weekly_report(season=2026, week=4, out_dir=tmp_path).read_text()

    assert any(word in text.lower() for word in ("verdict", "gate", "edge"))


def test_hit_rate_is_measured_against_the_actual_picks_not_the_graded_ones(tmp_path):
    """A row snapshotted but not yet resolved is a pick in the log and not yet a
    result. Counting it as a miss would understate the hit rate."""
    from nfl_predictor.tracking import store

    _seed(_graded_rows(60, hit_rate=0.55))
    before = write_weekly_report(season=2026, week=4, out_dir=tmp_path).read_text()

    store.record_player_prop_predictions([{
        "game_id": "G_NEW", "player_id": "00-new", "player_name": "Pending",
        "market": "receiving_yards", "position": "WR", "predicted_value": 50.0,
        "side": "over", "line_at_snapshot": 52.5, "odds_at_snapshot": -110.0,
        "model_p_over": 0.60, "edge_vs_breakeven": 0.076,
    }])

    after = write_weekly_report(season=2026, week=4, out_dir=tmp_path).read_text()

    assert "61" in after, "the new pick is counted"
    assert _hit_rate(before) == _hit_rate(after), (
        "an unresolved pick must not move the hit rate")


def test_report_creates_the_directory(tmp_path):
    out_dir = tmp_path / "nested" / "forward"

    path = write_weekly_report(season=2026, week=1, out_dir=out_dir)

    assert path.exists()


def _hit_rate(text: str) -> str:
    for line in text.splitlines():
        if "hit rate" in line.lower():
            return line
    raise AssertionError("no hit-rate line in the report")


def test_both_sides_are_reported_separately(tmp_path):
    """An under and an over are different bets; averaging them hides a side."""
    _seed(_graded_rows(60))

    text = write_weekly_report(season=2026, week=4, out_dir=tmp_path).read_text()

    assert "over" in text.lower() and "under" in text.lower()
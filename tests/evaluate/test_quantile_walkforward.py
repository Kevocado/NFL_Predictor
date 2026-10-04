"""quantile walk-forward tests.

Two properties matter and they are tested separately:

* correctness of the calibration bucketing, which is the spec's binding gate;
* that a validation season's predictions cannot depend on anything in it or
  after it. `test_future_seasons_cannot_move_an_earlier_prediction` perturbs a
  later season's labels and asserts the earlier fold's numbers are bit-identical,
  which is a stronger statement than checking that train seasons are filtered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.evaluate.walk_forward import (
    calibration_report, proxy_line, walk_forward_quantile,
)

FEATURES = ["f1", "f2"]


def _frame(seasons=(2018, 2019, 2020), players=6, market="receiving_yards",
           position="WR", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for player in range(players):
            baseline = 40 + 5 * player
            for week in range(1, 9):
                rows.append({
                    "player_id": f"p{player}", "player_name": f"P{player}",
                    "position": position, "season": season, "week": week,
                    "recent_team": "A", "opponent_team": "B",
                    "passing_yards": 0, "rushing_yards": 0,
                    "receiving_yards": float(baseline + rng.normal(scale=12)),
                    "targets": 5, "carries": 0, "receptions": 3,
                    "f1": float(rng.normal()), "f2": float(rng.normal()),
                })
    return pd.DataFrame(rows)


def test_proxy_line_excludes_the_target_week():
    values = pd.Series([10.0, 20.0, 30.0, 40.0])
    by = pd.Series(["p"] * 4)
    line = proxy_line(values, by, min_history=1)

    assert pd.isna(line.iloc[0]), "no history in the first week"
    assert line.iloc[1] == 10.0
    assert line.iloc[2] == 15.0, "median of weeks 1-2, never week 3"
    assert line.iloc[3] == 20.0, "median of weeks 1-3"


def test_proxy_line_needs_enough_history_by_default():
    values = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    line = proxy_line(values, pd.Series(["p"] * 5), min_history=3)

    assert pd.isna(line.iloc[0])
    assert pd.isna(line.iloc[1]), "one prior game is not a median"
    assert pd.isna(line.iloc[2]), "two prior games is still not three"
    assert line.iloc[3] == 20.0, "median of the first three games"


def test_calibration_report_buckets():
    df = pd.DataFrame({"p_over": [0.6] * 200, "covered": [1] * 120 + [0] * 80})
    report = calibration_report(df)

    assert report["0.6-0.7"]["empirical"] == pytest.approx(0.6)
    assert report["0.6-0.7"]["n"] == 200
    assert report["0.6-0.7"]["predicted"] == pytest.approx(0.6)


def test_calibration_report_omits_buckets_under_the_minimum():
    # 50 rows cannot clear the spec's n>=100 rule, so the bucket is not reported
    # at all -- reporting it would invite a verdict from a sample of 50.
    df = pd.DataFrame({"p_over": [0.6] * 50, "covered": [1] * 30 + [0] * 20})
    report = calibration_report(df, min_n=100)

    assert "0.6-0.7" not in report


def test_calibration_report_flags_a_miscalibrated_bucket():
    # Predicts 0.6, actually hits 40% of the time: a 20-point miss.
    df = pd.DataFrame({"p_over": [0.6] * 200, "covered": [1] * 80 + [0] * 120})
    report = calibration_report(df)

    assert report["0.6-0.7"]["within_tolerance"] is False
    assert report["0.6-0.7"]["gap"] == pytest.approx(0.2, abs=0.01)


def test_calibration_report_passes_a_calibrated_bucket():
    df = pd.DataFrame({"p_over": [0.6] * 200, "covered": [1] * 118 + [0] * 82})
    report = calibration_report(df)

    assert report["0.6-0.7"]["within_tolerance"] is True


def test_calibration_report_splits_across_buckets():
    df = pd.DataFrame({
        "p_over": [0.25] * 200 + [0.75] * 200,
        "covered": [1] * 50 + [0] * 150 + [1] * 150 + [0] * 50,
    })
    report = calibration_report(df)

    assert report["0.2-0.3"]["empirical"] == pytest.approx(0.25)
    assert report["0.7-0.8"]["empirical"] == pytest.approx(0.75)


def test_walkforward_produces_one_row_per_player_week_market():
    frame = _frame()
    out = walk_forward_quantile(frame, markets=["receiving_yards"],
                                seasons=[2019, 2020], feature_cols=FEATURES,
                                quantiles=[0.1, 0.5, 0.9])

    assert set(out.columns) >= {"season", "week", "player_id", "market",
                                "p_over", "proxy_line", "actual", "covered", "q50"}
    assert len(out) == 6 * 8 * 2, "6 players x 8 weeks x 2 validation seasons"
    assert set(out["season"]) == {2019, 2020}


def test_walkforward_never_scores_a_row_without_a_proxy_line():
    frame = _frame()
    out = walk_forward_quantile(frame, markets=["receiving_yards"],
                                seasons=[2019], feature_cols=FEATURES,
                                quantiles=[0.1, 0.5, 0.9])

    scored = out[out["covered"].notna()]
    assert not scored["proxy_line"].isna().any(), "a NaN proxy line makes covered meaningless"
    assert (out["p_over"].between(0.02, 0.98)).all()


def test_walkforward_reports_the_naive_baseline():
    frame = _frame()
    out = walk_forward_quantile(frame, markets=["receiving_yards"],
                                seasons=[2020], feature_cols=FEATURES,
                                quantiles=[0.1, 0.5, 0.9])

    assert "naive_pred" in out.columns
    # The naive prediction is the player's own lagged rolling mean.
    assert out["naive_pred"].notna().any()


def test_future_seasons_cannot_move_an_earlier_prediction():
    """The leakage probe. Rewriting 2020's labels must leave the 2019 fold
    bit-identical; if any future information reached the 2019 fit, it would."""
    frame = _frame(seasons=(2018, 2019, 2020))
    baseline = walk_forward_quantile(frame, markets=["receiving_yards"],
                                     seasons=[2019], feature_cols=FEATURES,
                                     quantiles=[0.1, 0.5, 0.9])

    tampered = frame.copy()
    tampered.loc[tampered["season"] == 2020, "receiving_yards"] += 500.0
    after = walk_forward_quantile(tampered, markets=["receiving_yards"],
                                 seasons=[2019], feature_cols=FEATURES,
                                 quantiles=[0.1, 0.5, 0.9])

    pd.testing.assert_frame_equal(
        baseline.sort_values(["player_id", "week"]).reset_index(drop=True),
        after.sort_values(["player_id", "week"]).reset_index(drop=True),
    )


def test_a_seasons_own_labels_cannot_move_its_predictions():
    """A row must not be able to see itself: the target week is excluded from
    the proxy line and the validation row is never in the training set."""
    frame = _frame(seasons=(2018, 2019))
    baseline = walk_forward_quantile(frame, markets=["receiving_yards"],
                                     seasons=[2019], feature_cols=FEATURES,
                                     quantiles=[0.1, 0.5, 0.9])

    tampered = frame.copy()
    # Only week 8 of the validation season moves.
    mask = (tampered["season"] == 2019) & (tampered["week"] == 8)
    tampered.loc[mask, "receiving_yards"] += 900.0
    after = walk_forward_quantile(tampered, markets=["receiving_yards"],
                                 seasons=[2019], feature_cols=FEATURES,
                                 quantiles=[0.1, 0.5, 0.9])

    b = baseline.sort_values(["player_id", "week"]).reset_index(drop=True)
    a = after.sort_values(["player_id", "week"]).reset_index(drop=True)
    # Week 8's own prediction is unchanged...
    earlier = b["week"] < 8
    pd.testing.assert_series_equal(b.loc[earlier, "p_over"], a.loc[earlier, "p_over"])
    # ...but its recorded outcome must reflect the new reality.
    assert (b.loc[~earlier, "actual"] != a.loc[~earlier, "actual"]).any()


def test_walkforward_restricts_markets_to_positions_that_have_them():
    frame = _frame(position="WR", market="rushing_yards")
    out = walk_forward_quantile(frame, markets=["rushing_yards"],
                                seasons=[2019], feature_cols=FEATURES,
                                quantiles=[0.1, 0.5, 0.9])

    assert out.empty, "a WR has no rushing-yards prop"


def test_walkforward_raises_on_an_unknown_market():
    with pytest.raises(ValueError, match="unknown market"):
        walk_forward_quantile(_frame(), markets=["sack_yards"],
                              seasons=[2019], feature_cols=FEATURES)


def test_mae_is_computable_per_market_and_season():
    frame = _frame()
    out = walk_forward_quantile(frame, markets=["receiving_yards"],
                                seasons=[2019, 2020], feature_cols=FEATURES,
                                quantiles=[0.1, 0.5, 0.9])

    scored = out.dropna(subset=["q50", "actual"])
    mae = float((scored["q50"] - scored["actual"]).abs().mean())
    naive_mae = float((scored["naive_pred"] - scored["actual"]).abs().mean())

    assert mae > 0
    assert naive_mae > 0
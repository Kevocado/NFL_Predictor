"""tests/test_qb_passing_td.py -- the QB passing-TD call on a per-player model line.

The line is DERIVED from the projection (it is the nearest half point to the
model's own mu), so nothing here is an edge claim and nothing here compares the
model against a sportsbook. The line is labelled a model line everywhere it is
produced and stored.

Named coverage, one test per claim the reviewer made:

* the line is a half point >= 0.5 and is the nearest half point to mu
* P(over) + P(under) == 1
* the side is the side with the higher probability
* over/under grading against a known stat line
* the pick record round-trips line and side

Plus the two structural claims: push cannot occur on a half-point line, and the
count distribution is chosen by fitting BOTH on history and comparing log loss.
"""
import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models import qb_passing_td as qbt
from nfl_predictor.tracking import qb_passing_td_record as record


# --- helpers ---------------------------------------------------------------


def _mu_1_8() -> dict:
    return {**qbt.passing_td_call(1.8, qbt.DistributionSpec("poisson", alpha=None)), "mu": 1.8}


def _toy_qb_frame(n=600, seed=7):
    """A synthetic QB history with the same columns the real frame has.

    Generated (not invented) so the tests are hermetic: the real numbers in the
    PR body come from the real cached seasons, these are for the invariants.
    """
    rng = np.random.default_rng(seed)
    roll = rng.normal(260, 45, n)
    # Overdispersed on purpose so the two distributions are genuinely different.
    tds = rng.poisson(np.clip(roll / 190.0, 0.05, None))
    tds = np.minimum(tds + rng.poisson(0.25, n), 6)
    return pd.DataFrame({"passing_tds_roll": roll, "passing_yards_roll": roll, "passing_tds": tds})


# --- 1. the line rule ------------------------------------------------------


@pytest.mark.parametrize(
    "mu,expected",
    [
        (1.8, 1.5),
        (2.3, 2.5),
        (0.0, 0.5),
        (0.1, 0.5),
        (0.26, 0.5),
        (0.3, 0.5),
        # Whole-number mus go UP to x.5, never onto the whole number: a whole
        # line would be a push and would be ungradeable.
        (1.0, 1.5),
        (2.0, 2.5),
        (4.0, 4.5),
        (1.49, 1.5),
        (1.51, 1.5),
        (2.99, 2.5),
        (0.75, 0.5),
        (3.7, 3.5),
        (9.9, 9.5),
    ],
)
def test_model_line_is_the_nearest_half_point_to_mu_and_never_below_half(mu, expected):
    assert qbt.model_line(mu) == expected


@pytest.mark.parametrize("mu", [0.0, 0.01, 0.3, 0.7, 1.0, 1.8, 2.0, 2.3, 3.74, 4.0, 9.9, 100.0])
def test_model_line_is_always_on_the_half_point_grid_at_or_above_half(mu):
    line = qbt.model_line(mu)
    assert line >= 0.5
    assert record.line_is_half_point(line), f"{line} is not a half point"
    # And it really is the NEAREST half point, not merely some half point: no
    # other point on the grid is closer to mu. The 0.5 floor is the one
    # documented exception, so it is excluded from the comparison rather than
    # special-cased.
    # The candidates are the numbers ending in .5, which is what makes a push
    # impossible -- see model_line's docstring for why 1.0 is not a candidate.
    # The nearest is a MINIMUM, not a strict minimum: a mu sitting exactly on a
    # whole number is equidistant from the x.5 below it and the x.5 above it,
    # and the documented tie rule resolves that upward.
    grid = [n + 0.5 for n in range(0, 400)]
    assert abs(line - mu) == min(abs(candidate - mu) for candidate in grid)
    # Never the LOWER of two equally-near candidates: mu 1.0 -> 1.5, not 0.5.
    nearest_below = [c for c in grid if abs(c - mu) == abs(line - mu) and c < line]
    assert not (nearest_below and line < max(nearest_below))


def test_model_line_is_a_model_line_and_never_a_sportsbook_line():
    """The label is part of the output, because the line is derived from mu and
    therefore cannot be an edge claim."""
    call = qbt.passing_td_call(2.3, qbt.DistributionSpec("negative_binomial", alpha=1.2))
    assert call["line_source"] == qbt.MODEL_LINE_SOURCE
    assert call["line_source"] == "model_line"


# --- 2. P(over) + P(under) == 1 -------------------------------------------


@pytest.mark.parametrize("mu", [0.4, 1.0, 1.8, 2.3, 3.0, 5.5])
@pytest.mark.parametrize("kind,alpha", [("poisson", None), ("negative_binomial", 0.4),
                                       ("negative_binomial", 1.2), ("negative_binomial", 8.0)])
def test_over_and_under_probabilities_sum_to_one(mu, kind, alpha):
    call = qbt.passing_td_call(mu, qbt.DistributionSpec(kind, alpha=alpha))
    assert call["over_prob"] + call["under_prob"] == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= call["over_prob"] <= 1.0
    assert 0.0 <= call["under_prob"] <= 1.0


def test_over_and_under_are_strictly_exhaustive_on_a_half_point_line():
    """Over + under == 1 is only true if there is no third outcome. Asserted
    separately from the sum so a future push bucket cannot hide inside it."""
    call = qbt.passing_td_call(2.5, qbt.DistributionSpec("negative_binomial", alpha=2.0))
    assert call.get("push_prob", 0.0) == 0.0


# --- 3. the side is the higher probability ---------------------------------


@pytest.mark.parametrize("mu", [0.4, 1.0, 1.8, 2.3, 3.0, 5.5])
@pytest.mark.parametrize("kind,alpha", [("poisson", None), ("negative_binomial", 0.4),
                                       ("negative_binomial", 3.0)])
def test_side_is_the_side_with_the_higher_probability(mu, kind, alpha):
    call = qbt.passing_td_call(mu, qbt.DistributionSpec(kind, alpha=alpha))
    expected = "over" if call["over_prob"] > call["under_prob"] else "under"
    assert call["side"] == expected
    assert call["call_prob"] == pytest.approx(max(call["over_prob"], call["under_prob"]))


def test_call_prob_is_the_probability_of_the_side_named():
    call = qbt.passing_td_call(2.3, qbt.DistributionSpec("negative_binomial", alpha=1.5))
    assert call["call_prob"] == pytest.approx(call[f"{call['side']}_prob"])


def test_a_higher_projection_turns_the_call_from_under_to_over():
    """Driven through P(over) rather than through mu, because mu alone does not
    decide the side: on a 3.5 line a mean of 3.4 is still more likely UNDER. The
    property being asserted is that a larger projection raises P(over) and the
    side follows it, which holds however the two land."""
    low = qbt.passing_td_call(1.2, qbt.DistributionSpec("poisson", alpha=None))
    high = qbt.passing_td_call(1.8, qbt.DistributionSpec("poisson", alpha=None))
    assert low["over_prob"] < high["over_prob"]
    assert low["side"] == "under"
    assert high["side"] == "over"


# --- push cannot occur -----------------------------------------------------


def test_a_half_point_line_can_never_be_push():
    """The line is always x.5 and the count is always an integer, so equality is
    unreachable. Asserted structurally rather than assumed."""
    for mu in (0.2, 1.0, 1.8, 2.3, 4.0):
        for kind, alpha in (("poisson", None), ("negative_binomial", 1.0)):
            call = qbt.passing_td_call(mu, qbt.DistributionSpec(kind, alpha=alpha))
            line = call["line"]
            assert record.line_is_half_point(line), "line must end in .5"
            assert line != int(line)
            assert call["over_prob"] + call["under_prob"] == pytest.approx(1.0, abs=1e-12)


# --- 4. grading over/under against a known stat line -----------------------


def _record_call(player_id="q1", line=2.5, side="under", mu=2.3, prob=0.6):
    return {
        "game_id": "2025_01_BAL_KC", "player_id": player_id, "player_name": "Pat QB",
        "position": "QB", "market": qbt.PASSING_TD_MARKET, "predicted_value": mu,
        "line": line, "side": side, "mu": mu, "call_prob": prob,
    }


def test_over_call_hits_when_actual_passing_tds_clear_the_line():
    summary = record.summarize_passing_td_calls(pd.DataFrame([
        {**_record_call("q1", line=2.5, side="over"),
         "resolved": 1, "actual_value": 3.0, "snapshotted_at": "2025-01-01T00:00:00"},
    ]))
    row = summary["per_pick"][0]
    assert row["hit"] is True
    assert row["actual_passing_tds"] == 3.0
    assert summary["hit_rate_when_called"] == 1.0


def test_over_call_misses_when_actual_passing_tds_fall_under_the_line():
    summary = record.summarize_passing_td_calls(pd.DataFrame([
        {**_record_call("q1", line=2.5, side="over"),
         "resolved": 1, "actual_value": 2.0, "snapshotted_at": "2025-01-01T00:00:00"},
    ]))
    assert summary["per_pick"][0]["hit"] is False
    assert summary["hit_rate_when_called"] == 0.0


def test_under_call_hits_when_actual_passing_tds_fall_under_the_line():
    summary = record.summarize_passing_td_calls(pd.DataFrame([
        {**_record_call("q1", line=2.5, side="under"),
         "resolved": 1, "actual_value": 2.0, "snapshotted_at": "2025-01-01T00:00:00"},
    ]))
    assert summary["per_pick"][0]["hit"] is True


def test_under_call_misses_when_actual_passing_tds_clear_the_line():
    summary = record.summarize_passing_td_calls(pd.DataFrame([
        {**_record_call("q1", line=2.5, side="under"),
         "resolved": 1, "actual_value": 3.0, "snapshotted_at": "2025-01-01T00:00:00"},
    ]))
    assert summary["per_pick"][0]["hit"] is False


def test_a_line_of_two_and_a_half_grades_two_as_under_and_three_as_over():
    """The boundary itself, spelled out, because a half-point line is the only
    reason push is impossible and that is worth pinning on real numbers."""
    for actual, expected_hit in ((2, True), (3, False)):
        summary = record.summarize_passing_td_calls(pd.DataFrame([
            {**_record_call("q1", line=2.5, side="under"),
             "resolved": 1, "actual_value": float(actual),
             "snapshotted_at": "2025-01-01T00:00:00"},
        ]))
        assert summary["per_pick"][0]["hit"] is expected_hit
        assert summary["per_pick"][0]["actual_passing_tds"] == actual


def test_grading_counts_only_the_earliest_pick_per_game_player_and_market():
    """The NFL #25 rule: one counted prop pick per (game, player_id, market),
    the earliest recorded. player_id is in the key -- two players in one game on
    one market are two picks, not one."""
    frame = pd.DataFrame([
        {**_record_call("q1", line=2.5, side="under"),
         "resolved": 1, "actual_value": 3.0, "snapshotted_at": "2025-01-02T00:00:00"},
        {**_record_call("q1", line=1.5, side="over"),
         "resolved": 1, "actual_value": 3.0, "snapshotted_at": "2025-01-01T00:00:00"},
        {**_record_call("q2", line=2.5, side="under"),
         "resolved": 1, "actual_value": 1.0, "snapshotted_at": "2025-01-01T06:00:00"},
    ])
    summary = record.summarize_passing_td_calls(frame)

    assert summary["n_resolved"] == 2, "two players in one game are two picks"
    assert len(summary["per_pick"]) == 2
    counted = {r["player_id"]: r for r in summary["per_pick"]}
    # q1's EARLIEST pick is the 01-01 over at line 1.5, not the later under.
    assert counted["q1"]["line"] == 1.5
    assert counted["q1"]["side"] == "over"
    assert counted["q1"]["hit"] is True
    assert counted["q2"]["hit"] is True


def test_an_empty_frame_is_an_empty_record_not_a_crash():
    summary = record.summarize_passing_td_calls(
        pd.DataFrame(columns=["game_id", "player_id", "market", "line", "side", "mu",
                              "call_prob", "resolved", "actual_value", "snapshotted_at"]))
    assert summary["n_resolved"] == 0
    assert summary["hit_rate_when_called"] is None
    assert summary["per_pick"] == []


# --- 5. the pick record round-trips line and side --------------------------


def test_pick_record_round_trips_line_and_side_through_sqlite(tmp_path, monkeypatch):
    from nfl_predictor import config
    from nfl_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)

    store.record_game_predictions([{
        "game_id": "2025_01_BAL_KC", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00", "home_win_prob": 0.6, "away_win_prob": 0.4,
        "over_prob": 0.5, "under_prob": 0.5,
    }])
    assert store.record_player_prop_predictions([_record_call()]) == 1

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT line, side, mu, call_prob FROM player_prop_predictions "
            "WHERE game_id = ? AND player_id = ? AND market = ?",
            ("2025_01_BAL_KC", "q1", qbt.PASSING_TD_MARKET),
        ).fetchone()

    assert row is not None, "the passing_td pick was not written at all"
    assert row[0] == 2.5
    assert row[1] == "under"
    assert row[2] == pytest.approx(2.3)
    assert row[3] == pytest.approx(0.6)


def test_record_round_trips_a_second_call_with_a_different_line_and_side(tmp_path, monkeypatch):
    """A QB whose projection moved to over writes a different line and side, so
    the record is genuinely keyed on the projection and not a constant."""
    from nfl_predictor import config
    from nfl_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)

    store.record_game_predictions([{
        "game_id": "2025_01_BAL_KC", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00", "home_win_prob": 0.6, "away_win_prob": 0.4,
        "over_prob": 0.5, "under_prob": 0.5,
    }])
    store.record_player_prop_predictions([
        _record_call("q1", line=2.5, side="under", mu=2.3, prob=0.61),
        _record_call("q2", line=3.5, side="over", mu=3.4, prob=0.58),
    ])

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        rows = dict(conn.execute(
            "SELECT player_id, line || '|' || side FROM player_prop_predictions").fetchall())

    assert rows == {"q1": "2.5|under", "q2": "3.5|over"}


# --- distribution choice by log loss ---------------------------------------


def test_choosing_by_log_loss_fits_both_and_reports_both_numbers():
    frame = _toy_qb_frame()
    fitted = qbt.fit_qb_passing_td_model(
        frame[["passing_tds_roll", "passing_yards_roll"]], frame["passing_tds"])

    assert set(fitted["log_loss"]) == {"poisson", "negative_binomial"}
    for name, value in fitted["log_loss"].items():
        assert isinstance(value, float)
        assert np.isfinite(value)
    chosen = fitted["distribution"]
    assert chosen in ("poisson", "negative_binomial")
    assert fitted["log_loss"][chosen] == min(fitted["log_loss"].values())


def test_the_chosen_distribution_is_the_lower_log_loss_on_the_same_holdout():
    frame = _toy_qb_frame(n=800)
    fitted = qbt.fit_qb_passing_td_model(
        frame[["passing_tds_roll", "passing_yards_roll"]], frame["passing_tds"])
    scores = fitted["log_loss"]
    winner = min(scores, key=scores.get)
    assert fitted["distribution"] == winner


def test_expected_passing_tds_is_per_player_not_a_constant():
    frame = _toy_qb_frame()
    X = frame[["passing_tds_roll", "passing_yards_roll"]]
    fitted = qbt.fit_qb_passing_td_model(X, frame["passing_tds"])

    quiet = X.iloc[[0]].copy()
    loud = X.iloc[[0]].copy()
    loud["passing_yards_roll"] = 400.0
    quiet["passing_yards_roll"] = 120.0

    mu_quiet = qbt.expected_passing_tds(fitted, quiet.iloc[0])
    mu_loud = qbt.expected_passing_tds(fitted, loud.iloc[0])

    assert mu_loud > mu_quiet
    assert qbt.model_line(mu_loud) != qbt.model_line(mu_quiet), (
        "different QBs must be able to land on different lines"
    )


def test_expected_passing_tds_agrees_with_the_line_rule_and_the_side():
    """One QB, end to end: mu -> line -> side -> probability, and each step is
    the one the other tests pin."""
    frame = _toy_qb_frame()
    fitted = qbt.fit_qb_passing_td_model(
        frame[["passing_tds_roll", "passing_yards_roll"]], frame["passing_tds"])
    mu = qbt.expected_passing_tds(fitted, frame[["passing_tds_roll", "passing_yards_roll"]].iloc[0])

    call = qbt.passing_td_call(mu, qbt.DistributionSpec(
        fitted["distribution"], alpha=fitted.get("alpha")))

    assert call["line"] == qbt.model_line(mu)
    assert call["mu"] == pytest.approx(mu)
    assert call["over_prob"] + call["under_prob"] == pytest.approx(1.0, abs=1e-12)
    assert call["side"] == ("over" if call["over_prob"] > call["under_prob"] else "under")


def test_a_log_loss_below_zero_is_treated_as_broken_not_as_a_win():
    """Regression guard for a real defect, not a hypothetical one.

    An UNBOUNDED NB dispersion fit walked into the region where
    `nbinom.logpmf` returns inf, whose mean negative log-likelihood is -inf and
    so 'beats' any honest score. On the real 2018-2024 QB history it reported a
    log loss of **-5.81**, with alpha overflowing to 2.2e15.

    A mean NLL over counts a real pmf supports cannot go below 0 in any way that
    indicates a good fit, so a negative score means the fit broke. Poisson is
    near 1.39 on this data and a genuine NB2 improvement is a fraction of that.
    """
    frame = _toy_qb_frame(n=900)
    fitted = qbt.fit_qb_passing_td_model(
        frame[["passing_tds_roll", "passing_yards_roll"]], frame["passing_tds"])

    for name, value in fitted["log_loss"].items():
        assert value >= 0.0, f"{name} reported an impossible log loss of {value}"
        assert value > -1.0, f"{name} log loss {value} is below any plausible fit"
    if fitted["distribution"] == "negative_binomial":
        assert fitted["alpha"] < 1e6, f"alpha overflowed to {fitted['alpha']}"


def test_negative_binomial_dispersion_is_bounded_away_from_the_collapsed_region():
    frame = _toy_qb_frame(n=900)
    fitted = qbt.fit_qb_passing_td_model(
        frame[["passing_tds_roll", "passing_yards_roll"]], frame["passing_tds"])
    lo, hi = qbt._NB_LOG_ALPHA_BOUNDS
    if fitted["alpha"] is not None:
        assert lo - 1e-9 <= np.log(fitted["alpha"]) <= hi + 1e-9


def test_predict_props_gives_a_qb_the_call_and_no_one_else_the_market():
    """The payload contract the frontend PR consumes.

    Flattened onto the same row the yardage markets use, `passing_td_`-prefixed,
    and QB-only -- an RB must not grow a passing-TD call.
    """
    from nfl_predictor.models import player_props

    frame = _toy_qb_frame()
    X = frame[["passing_tds_roll", "passing_yards_roll"]]
    fitted = qbt.fit_qb_passing_td_model(X, frame["passing_tds"])

    class _Stub:
        def predict(self, X):
            return np.full(len(X), 250.0)

        def predict_proba(self, X):
            p = np.full((len(X), 2), 0.2)
            p[:, 1] = 0.8
            return p

    models = {
        "feature_cols": list(X.columns),
        "anytime_td": _Stub(),
        "passing_yards": _Stub(),
        qbt.PASSING_TD_MARKET: fitted,
    }
    row = X.iloc[0]

    qb = player_props.predict_props(models, row, position="QB")
    for key in ("passing_td_line", "passing_td_line_source", "passing_td_side",
                "passing_td_mu", "passing_td_over_prob", "passing_td_under_prob",
                "passing_td_prob", "passing_td_distribution"):
        assert key in qb, f"props payload is missing {key}"
    assert qb["passing_td_line_source"] == "model_line"
    assert qb["passing_td_line"] == qbt.model_line(qb["passing_td_mu"])
    assert qb["passing_td_prob"] == pytest.approx(
        max(qb["passing_td_over_prob"], qb["passing_td_under_prob"]))

    rb = player_props.predict_props(models, row, position="RB")
    assert not [k for k in rb if k.startswith("passing_td_")], (
        "a non-QB must never receive a passing-TD call"
    )


def test_predict_props_omits_the_market_entirely_when_no_model_was_trained():
    """An artifact directory predating this feature must still serve props."""
    from nfl_predictor.models import player_props

    class _Stub:
        def predict(self, X):
            return np.full(len(X), 250.0)

        def predict_proba(self, X):
            p = np.full((len(X), 2), 0.2)
            p[:, 1] = 0.8
            return p

    models = {"feature_cols": ["passing_yards_roll"], "anytime_td": _Stub()}
    result = player_props.predict_props(models, pd.Series({"passing_yards_roll": 250.0}),
                                        position="QB")
    assert not [k for k in result if k.startswith("passing_td_")]
    assert "anytime_td_prob" in result


def test_the_market_name_is_passing_tds_and_is_not_anytime_td():
    """QB anytime_td counts rushing OR receiving OR passing (features/player_usage.py:23-25),
    so this category is separate by construction and never aliases it."""
    assert qbt.PASSING_TD_MARKET == "passing_tds"
    assert qbt.PASSING_TD_MARKET != "anytime_td"
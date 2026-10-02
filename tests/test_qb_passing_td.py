"""tests/test_qb_passing_td.py -- the QB passing-TD call on a per-player model line.

The line is DERIVED from the projection (it is the nearest half point to the
model's own mu), so nothing here is an edge claim and nothing here compares the
model against a sportsbook. The line is labelled a model line everywhere it is
produced and stored.

Named coverage, one test per claim the reviewer made:

* the line is a half point >= 0.5 and is the nearest half point to mu
* P(over) + P(under) == 1
* the side is the side with the higher probability
* push cannot occur on a half-point line
* the pick record round-trips line and side through SQLite

Plus the count distribution chosen by fitting BOTH on history and comparing log
loss, and the train/serve feature parity.

**What is deliberately absent.** The report that graded these picks into a track
record, `nfl_predictor.tracking.qb_passing_td_record`, was deleted: nothing
wrote a `market="passing_tds"` row and nothing read the report, so it was a
feature-shaped module with no feature behind it. Its grading and counting tests
went with it. `tests/test_passing_td_record_absence.py` pins that the module is
gone and unreferenced, so putting it back has to be a decision rather than an
accident. What is left here is only what the model and `store` really do.
"""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor

from nfl_predictor.models import manifest as model_manifest
from nfl_predictor.models import qb_passing_td as qbt


def _is_half_point(line) -> bool:
    """Whether `line` ends in .5, spelled out locally.

    This used to be `qb_passing_td_record.line_is_half_point`, imported from the
    module that recorded the picks. It is four lines of arithmetic about the line
    grid, and the claim it supports is `model_line` landing on that grid -- so it
    belongs beside the test. It also keeps the deleted module's push-impossibility
    argument alive: see `test_a_whole_number_line_would_be_a_push` below.
    """
    value = float(line)
    doubled = value * 2
    return abs(doubled - round(doubled)) < 1e-9 and abs(value - round(value)) == 0.5


def test_the_fit_docstring_quotes_only_the_committed_artefact():
    """A docstring that lies about the artefact it describes is a defect.

    `fit_qb_passing_td_model`'s docstring claimed 4,501 player-weeks over 2018-2024
    at a variance-to-mean ratio of 1.084. The committed artefact is 5,179
    player-weeks over 2018-2025 at 1.0888, so the paragraph described the model
    *before* the 2026-10-02 retrain onto the anytime-TD label v2 features -- and
    nothing in the repository noticed, because prose is not asserted by default.

    Every number this docstring states about the fit is read back out of
    `models/manifest.json` here. A number the manifest does not record is a
    number no test can keep honest, which is why the docstring no longer states
    one: the unconditional mean and variance, the conditional Pearson dispersion
    and the 2024 holdout block are all gone rather than left to rot.
    """
    recorded = model_manifest.load_manifest()
    entry = recorded["qb_passing_td"]
    doc = qbt.fit_qb_passing_td_model.__doc__

    # Seasons: the manifest's own list, as the range the docstring states it.
    seasons = [int(s) for s in recorded["seasons"]]
    assert seasons == list(range(min(seasons), max(seasons) + 1)), (
        "the docstring states a season RANGE, so a non-contiguous training window "
        f"would make its range string wrong: {seasons}"
    )
    assert f"{min(seasons)}-{max(seasons)}" in doc

    # Size, overdispersion and the chosen distribution, all verbatim.
    assert f"{entry['n_train']:,}" in doc, f"n_train={entry['n_train']} is not quoted in the docstring"
    assert f"{entry['variance_ratio']:.4f}" in doc
    assert entry["distribution"] in doc.lower()

    # Both log losses are quoted, because the choice between the two is only
    # meaningful if the loser is inspectable. An infinite score is the "this one
    # broke" sentinel and is deliberately not quoted as a number.
    finite = {name: v for name, v in entry["log_loss"].items() if math.isfinite(v)}
    assert len(finite) >= 2, f"expected two finite log losses to check, got {entry['log_loss']}"
    for name, score in finite.items():
        assert f"{score:.6f}" in doc, f"log_loss[{name}]={score!r} is not quoted in the docstring"

    # And the superseded numbers, named, so a revert fails here rather than
    # passing quietly. These are exactly what the docstring used to claim.
    for stale in ("4,501", "2018-2024", "1.084"):
        assert stale not in doc, f"the docstring still quotes the pre-retrain value {stale!r}"


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
    assert _is_half_point(line), f"{line} is not a half point"
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
            assert _is_half_point(line), "line must end in .5"
            assert line != int(line)
            assert call["over_prob"] + call["under_prob"] == pytest.approx(1.0, abs=1e-12)


def test_a_whole_number_line_would_be_a_push():
    """The counterpart to the test above, and the reason it is structural.

    A whole-number line has an integer sitting on it. On a line of 1.0, a
    quarterback who throws exactly one passing TD makes `actual == line`, so
    neither `actual > line` nor `actual < line` holds and the pick has no verdict:
    a real push, and a record that cannot grade it. `model_line`'s whole job is to
    make that unreachable, so this asserts the grid it lands on is the one that
    excludes the case -- not that the grid is merely nearby. Spelled out because
    the deleted record module used to assert this at grade time, and that
    assertion went with it; the property has to outlive the module.
    """
    whole = 1.0
    exact = 1.0
    assert not _is_half_point(whole)
    assert not (exact > whole or exact < whole), (
        "a whole-number line admits the tie that makes a pick ungradeable"
    )

    # And every line the model can produce excludes it: all of them end in .5, and
    # no integer equals a number ending in .5.
    for mu in (0.2, 1.0, 1.8, 2.3, 4.0, 7.0, 9.9):
        line = qbt.model_line(mu)
        assert _is_half_point(line), f"model_line({mu}) = {line} is gradeable only by luck"
        assert all(count != line for count in range(0, 30)), (
            f"an integer count equals the line {line}, so a push is reachable"
        )


# --- 4. the pick record round-trips line and side -----------------------------
#
# These two moved here from the deleted module's section and are kept, because
# what they assert is `store.record_player_prop_predictions` -- real, reachable
# code that round-trips the optional `line`/`line_source`/`side`/`mu`/`call_prob`
# columns. What they do NOT assert is that anything in serving fills those
# columns: `routes.background_tracking_tick` writes no `passing_tds` row, so on
# the live database all five are NULL. The columns are a format, ready for a
# writer; see the note on `_PASSING_TD_PROP_COLUMNS` in `tracking/store.py`.


def _record_call(player_id="q1", line=2.5, side="under", mu=2.3, prob=0.6):
    return {
        "game_id": "2025_01_BAL_KC", "player_id": player_id, "player_name": "Pat QB",
        "position": "QB", "market": qbt.PASSING_TD_MARKET, "predicted_value": mu,
        "line": line, "side": side, "mu": mu, "call_prob": prob,
    }


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
    """QB anytime_td counts rushing OR receiving only, so this category is
    separate by construction and never aliases it.

    This docstring previously read "rushing OR receiving OR passing
    (features/player_usage.py:23-25)". Passing TDs were dropped from the anytime-TD
    label on 2026-10-01, so that parenthetical is now wrong. The behavioural pin on
    the new definition lives in `test_player_usage.py::test_anytime_td_label_*`; this
    test stays about the market *name*, which is what stops the two from aliasing.
    """
    assert qbt.PASSING_TD_MARKET == "passing_tds"
    assert qbt.PASSING_TD_MARKET != "anytime_td"


# --- train/serve feature parity (the reviewer's defect) --------------------
#
# `_fit_qb_passing_td` fitted the model on `passing_tds_roll` FIRST, and
# `build_features_for_player` did not emit it. `expected_passing_tds` then did
# `feature_row.reindex(cols).fillna(0)`, so every QB was projected from a
# constant-zero rolling TD rate -- every line and every probability wrong, with
# the suite green. These tests are the assertion that was missing: fit on a real
# QB history, serve one QB's pregame row, and require every fitted column to be
# present and non-null.


#: The QB these tests fit and then serve. `_qb_panel` builds his history too, so
#: the served row is a real pregame view of the same frame the model was fitted on.
QB_ID = "00-001"
QB_SEASON = 2024


def _qb_panel(n_qbs: int = 12, seasons: tuple[int, ...] = (2022, 2023, 2024),
              weeks: int = 12, seed: int = 11) -> pd.DataFrame:
    """Weekly QB stat lines in the column names `player_stats.KEEP_COLUMNS` uses.

    Generated rather than real, so the suite is hermetic and needs no network.
    The volume-to-TD relationship (`tds ~ Poisson(yards / 175)`) is the real
    shape -- passing TDs are a rate on passing volume -- which is what makes the
    fitted `passing_tds_roll` coefficient carry a usable sign. A panel rather
    than one QB, because 13 player-weeks is too few to fit four columns on and
    the coefficients come out noise-dominated.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for qb in range(n_qbs):
        # A per-QB arm, so the panel has QB-level variation and not one line
        # repeated, and the fit has something to learn.
        arm = 0.75 + 0.5 * (qb / max(n_qbs - 1, 1))
        player_id = f"00-{qb:03d}"
        for season in seasons:
            for week in range(1, weeks + 1):
                yards = float(np.clip(rng.normal(255 * arm, 45), 40, None))
                rows.append({
                    "player_id": player_id, "player_name": f"QB {qb}", "position": "QB",
                    "recent_team": "KC", "season": season, "week": week,
                    "passing_yards": yards,
                    "passing_tds": int(np.clip(rng.poisson(yards / 175.0), 0, 7)),
                    "rushing_yards": float(rng.normal(6, 8)), "rushing_tds": 0,
                    "receiving_yards": 0.0, "receiving_tds": 0,
                    "receptions": 0, "targets": 0, "carries": float(rng.normal(1.5, 2)),
                })
    return pd.DataFrame(rows)


def _qb_history(n_weeks: int = 14, player_id: str = QB_ID, season: int = QB_SEASON) -> pd.DataFrame:
    """One QB's history. Kept as a separate helper because these tests set his
    passing_tds directly to pin the shift(1) behaviour."""
    rng = np.random.default_rng(11)
    rows = []
    for week in range(1, n_weeks + 1):
        yards = float(rng.normal(255, 45))
        rows.append({
            "player_id": player_id, "player_name": "A. QB", "position": "QB",
            "recent_team": "KC", "season": season, "week": week,
            "passing_yards": yards,
            "passing_tds": int(np.clip(rng.poisson(yards / 175.0), 0, 6)),
            "rushing_yards": float(rng.normal(6, 8)), "rushing_tds": 0,
            "receiving_yards": 0.0, "receiving_tds": 0,
            "receptions": 0, "targets": 0, "carries": float(rng.normal(1.5, 2)),
        })
    return pd.DataFrame(rows)


def _fitted_qb_model(panel: pd.DataFrame | None = None) -> dict:
    """The model exactly as `manifest.train_all` fits it, from a real frame."""
    from nfl_predictor.features import player_usage
    from nfl_predictor.models import manifest

    frame, _ = player_usage.build_player_training_frame(
        _qb_panel() if panel is None else panel)
    fitted = manifest._fit_qb_passing_td(frame)
    assert fitted is not None, "the fixture must produce a QB to fit on"
    return fitted


def test_every_fitted_column_is_present_and_non_null_in_a_served_row():
    """The assertion the suite was missing, and the exact one the reviewer named.

    Build a QB's pregame row through `build_features_for_player` -- the only path
    serving uses -- and require every column the model was fitted on to be there
    with a real value. `passing_tds_roll` was the column missing; the loop is over
    `fitted["feature_cols"]` rather than a hardcoded list so the same test covers
    whatever the model is fitted on next.
    """
    from nfl_predictor.features import player_usage

    panel = _qb_panel()
    fitted = _fitted_qb_model(panel)
    assert "passing_tds_roll" in fitted["feature_cols"], (
        "this fix keeps the column in the model; a test that quietly dropped it "
        "would pass against the broken serving path"
    )

    history = panel[panel["player_id"] == QB_ID]
    served = player_usage.build_features_for_player(
        QB_ID, history, season=QB_SEASON, week=history["week"].max() + 1)
    assert served is not None

    missing = [c for c in fitted["feature_cols"] if c not in served.index]
    assert not missing, (
        f"serving does not emit {missing}; `expected_passing_tds` reindexes and "
        "fillna(0)s them, so every QB is projected from a constant zero"
    )
    nulls = [c for c in fitted["feature_cols"] if pd.isna(served[c])]
    assert not nulls, f"served row has null fitted features {nulls} for a QB with history"
    assert (served[list(fitted["feature_cols"])] != 0).any(), (
        "every fitted feature is exactly zero, which is the served-a-zero shape of the defect"
    )


def test_served_passing_tds_roll_equals_the_training_row_for_the_same_week():
    """The column's `shift(1)` discipline, checked against the training rows.

    Not a re-derivation of the arithmetic: `with_passing_tds_roll` is the same
    function training uses, so this compares serving against the actual fitted
    frame. For target week W it must equal that frame's own `passing_tds_roll`
    at W -- the mean of the prior games, excluding W.
    """
    from nfl_predictor.features import player_usage

    history = _qb_history(n_weeks=12)
    training = player_usage.with_passing_tds_roll(history)

    for target_week in (4, 8, 12):
        served = player_usage.build_features_for_player(
            "00-001", history, season=2024, week=target_week)
        trained = training[training["week"] == target_week][
            player_usage.PASSING_TDS_ROLL_COLUMN].iloc[0]
        assert served[player_usage.PASSING_TDS_ROLL_COLUMN] == pytest.approx(trained), (
            f"week {target_week}: serving and training disagree on passing_tds_roll"
        )

    # And the shift is load-bearing: the mean must exclude the target week's own
    # row. With `week` bounded, the window is weeks 1..W-1, so it can never
    # contain W -- asserting the value differs from an unbounded mean catches a
    # future change that drops the bound.
    weeks = _qb_history(n_weeks=6, player_id="00-002")
    weeks["passing_tds"] = [0, 0, 0, 0, 0, 5]
    served = player_usage.build_features_for_player("00-002", weeks, season=2024, week=6)
    assert served[player_usage.PASSING_TDS_ROLL_COLUMN] == pytest.approx(0.0), (
        "the target week's own 5 TDs leaked into the pregame feature"
    )
    # Unbounded, the window is the last five rows (weeks 2-6) -> mean 1.0, so the
    # two views are distinguishable and a dropped `week` bound is visible.
    assert player_usage.build_features_for_player(
        "00-002", weeks, season=2024)[player_usage.PASSING_TDS_ROLL_COLUMN] == pytest.approx(1.0)


def test_the_fitted_feature_list_is_a_subset_of_what_serving_emits():
    """The whole class, for this model and the player models around it.

    Cheap because it is set arithmetic plus one call to the builder -- no fitting,
    no network. `load_models` runs the same check for real on every artefact in the
    payload (`manifest._assert_artefact_columns_are_served`), and
    `tests/test_fitted_vs_served_columns.py` is the generic test for it.

    **The serving set here is read by CALLING `build_features_for_player`, not by
    reading `player_usage.SERVING_FEATURE_COLUMNS`.** That is a change, and it is
    the substance of this fix: the old version compared two hand-written lists --
    the constant against `MU_FEATURE_COLUMNS`/`PLAYER_FEATURE_COLUMNS` -- and
    `load_models` then compared a third hand-written list against the same
    constant, which is defined as `PLAYER_FEATURE_COLUMNS` plus
    `passing_tds_roll`. Every one of those comparisons was a tautology over the
    same declared data and none of them consulted the builder or a pickle. Asking
    the builder what it emits is the first version of this assertion that can
    fail, and the constant is now only cross-checked against it for agreement.

    The audit this came out of: `passing_tds_roll` was the only QB-model column
    missing from the serving builder, and nothing else in the manifest was
    missing or extra -- the game side is audited the same way, by
    `manifest._game_serving_columns`.
    """
    from nfl_predictor.features import player_usage
    from nfl_predictor.models import manifest

    served = set(manifest._player_serving_columns())
    assert set(qbt.MU_FEATURE_COLUMNS) | {player_usage.PASSING_TDS_ROLL_COLUMN} <= served
    assert set(player_usage.PLAYER_FEATURE_COLUMNS) <= served
    # Nothing fitted is served, nothing served is unfitted-but-modelled: the
    # serving row is exactly the fitted columns plus `passing_tds_roll`.
    assert served - set(player_usage.PLAYER_FEATURE_COLUMNS) == {
        player_usage.PASSING_TDS_ROLL_COLUMN}

    # And the declared constant still agrees with what the builder emits. If this
    # ever fails, `SERVING_FEATURE_COLUMNS` has drifted -- which is worth knowing,
    # because it is a public name in `player_usage`, but it is no longer what the
    # load-time audit trusts.
    assert set(player_usage.SERVING_FEATURE_COLUMNS) == served

    with pytest.raises(ValueError, match="does not emit"):
        manifest._assert_servable_columns(
            ["passing_tds_roll", "not_a_real_column"], "a test model", served,
            "player_usage.build_features_for_player")


def test_expected_passing_tds_refuses_a_row_missing_a_fitted_column():
    """The per-row backstop for the same defect.

    `reindex(cols).fillna(0)` cannot distinguish a null value from an absent
    column, and the absent-column half is what served every QB from a zero
    `passing_tds_roll`. A row missing a fitted column now raises instead of
    being scored on a fabricated zero.
    """
    from nfl_predictor.features import player_usage

    fitted = _fitted_qb_model()
    served = player_usage.build_features_for_player(
        QB_ID, _qb_history(n_weeks=6), season=QB_SEASON, week=6)

    without = served.drop(labels=[player_usage.PASSING_TDS_ROLL_COLUMN])
    with pytest.raises(KeyError, match="constant-zero"):
        qbt.expected_passing_tds(fitted, without)

    # A null *value* on a column that IS present still fills -- that is a player
    # with no prior games, which `routes` skips before scoring.
    nulled = served.copy()
    nulled[player_usage.PASSING_TDS_ROLL_COLUMN] = float("nan")
    assert qbt.expected_passing_tds(fitted, nulled) >= 0.0


def test_serving_emits_exactly_the_declared_serving_columns():
    """No more, no fewer -- an emitter that silently grew or lost a column is a
    drift this repo has already been bitten by twice."""
    from nfl_predictor.features import player_usage

    history = _qb_history(n_weeks=6)
    served = player_usage.build_features_for_player(QB_ID, history, season=QB_SEASON, week=6)
    assert set(served.index) == set(player_usage.SERVING_FEATURE_COLUMNS)
    assert player_usage.SERVING_FEATURE_COLUMNS == [
        *player_usage.PLAYER_FEATURE_COLUMNS, player_usage.PASSING_TDS_ROLL_COLUMN
    ]


def test_a_qb_whose_roll_is_known_is_served_that_roll():
    """End to end through the real serving builder: the projection must actually
    move with the QB's own rolling TD rate, which a `fillna(0)` column cannot do.

    Fitted on the panel, served on one QB's last five weeks with the passing_tds
    column rewritten high and low. Volume is held constant across the two arms so
    the only thing that differs is `passing_tds_roll`.
    """
    from nfl_predictor.features import player_usage

    fitted = _fitted_qb_model()
    panel = _qb_panel()

    def _mu(tds: float) -> float:
        history = panel[panel["player_id"] == QB_ID].copy()
        last_season = history[history["season"] == QB_SEASON]["week"].max()
        history = history[
            (history["season"] != QB_SEASON) | (history["week"] <= last_season)]
        history.loc[history["week"] > 5, "passing_tds"] = tds
        served = player_usage.build_features_for_player(
            QB_ID, history, season=QB_SEASON, week=last_season + 1)
        assert served is not None
        return qbt.expected_passing_tds(fitted, served)

    mu_high, mu_low = _mu(4.0), _mu(0.0)
    assert mu_high > mu_low, (
        "a QB who threw 4 TDs a game for five games and one who threw none must "
        "not project the same mu"
    )
    # And it has to move enough to matter, not by a rounding error.
    assert mu_high - mu_low > 0.2, f"the served roll moved mu by only {mu_high - mu_low}"


# --- the optional artefact read ---------------------------------------------


def test_a_manifest_that_declares_a_passing_td_model_requires_the_artifact(monkeypatch, tmp_path):
    """Declared in the manifest but missing on disk is a corrupt deployment, and
    must raise rather than quietly serve a payload with the market absent.

    The old read asked only whether the file existed, so this state -- a manifest
    written by a retrain plus a directory that lost the artifact -- served as "no
    model this time" and the feature stayed dormant with nothing logged.
    """
    from nfl_predictor.models import manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest_mod, "_artifact_path", lambda name: tmp_path / name)
    monkeypatch.setattr(manifest_mod, "_load_pickle", lambda path: object())

    with pytest.raises(FileNotFoundError, match="missing"):
        manifest_mod._load_passing_td_model(
            {"qb_passing_td": {"distribution": "poisson", "n_train": 100}},
            manifest_mod._player_serving_columns())


@pytest.mark.parametrize("manifest_dict", [
    {"qb_passing_td": None},
    {},  # a manifest written before this model existed
])
def test_a_manifest_with_no_passing_td_model_serves_without_it(monkeypatch, tmp_path, manifest_dict):
    """Still optional in the honest case: no model was trained, so the market is
    omitted, exactly as a yardage market with no model is."""
    from nfl_predictor.models import manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "_artifact_path", lambda name: tmp_path / name)
    monkeypatch.setattr(manifest_mod, "_load_pickle", lambda path: pytest.fail("must not load"))

    assert manifest_mod._load_passing_td_model(
        manifest_dict, manifest_mod._player_serving_columns()) is None


def test_a_present_artifact_is_loaded_and_checked(monkeypatch, tmp_path):
    """When it is there, the fitted columns are still audited -- so a saved model
    fitted on a column serving cannot produce fails at load, not silently."""
    from nfl_predictor.models import manifest as manifest_mod

    (tmp_path / "qb_passing_td_model.pkl").write_bytes(b"stub")
    monkeypatch.setattr(manifest_mod, "_artifact_path", lambda name: tmp_path / name)

    # A real Poisson regresser fitted on two columns, which is what the audit now
    # requires of a mapping payload. `"model": object()` was enough before, because
    # nothing read the inner estimator: the audit corroborated a self-reported
    # `feature_cols` against nothing and accepted it. It now corroborates against
    # the estimator inside the same pickle -- by column names where the inner
    # estimator has them, and otherwise by `n_features_in_`, which sklearn records
    # even on a NumPy fit -- so the inner model has to be a model.
    serving = manifest_mod._player_serving_columns()

    def _payload(cols, model):
        return {"feature_cols": list(cols), "model": model}

    def _poisson(n_cols):
        return PoissonRegressor(alpha=1e-8, max_iter=1000).fit(
            np.ones((8, n_cols)), np.ones(8))

    monkeypatch.setattr(manifest_mod, "_load_pickle", lambda path: _payload(
        ["passing_tds_roll", "passing_yards_roll"], _poisson(2)))

    fitted = manifest_mod._load_passing_td_model(
        {"qb_passing_td": {"distribution": "poisson", "n_train": 100}}, serving)
    assert fitted["feature_cols"][0] == "passing_tds_roll"

    # A fitted column serving cannot produce.
    monkeypatch.setattr(manifest_mod, "_load_pickle", lambda path: _payload(
        ["passing_tds_roll", "a_column_serving_lacks"], _poisson(2)))
    with pytest.raises(ValueError, match="does not emit"):
        manifest_mod._load_passing_td_model(
            {"qb_passing_td": {"distribution": "poisson", "n_train": 100}}, serving)

    # And a claim the wrapped model cannot corroborate -- here by COUNT, since a
    # NumPy fit records no names. This is the case that used to reach
    # `expected_passing_tds` and raise once per QB per request instead.
    monkeypatch.setattr(manifest_mod, "_load_pickle", lambda path: _payload(
        ["passing_tds_roll"], _poisson(2)))
    with pytest.raises(ValueError, match="n_features_in_"):
        manifest_mod._load_passing_td_model(
            {"qb_passing_td": {"distribution": "poisson", "n_train": 100}}, serving)


@pytest.mark.parametrize("record", [
    pytest.param({}, id="no_feature_cols_key"),
    pytest.param({"feature_cols": []}, id="empty_feature_cols"),
])
def test_a_passing_td_payload_that_records_no_fitted_columns_is_refused(
    monkeypatch, tmp_path, record
):
    """A payload whose fitted-column record is missing or empty must not serve.

    The previous line here was `list(fitted.get("feature_cols") or [])`, which turns
    both of these into an empty fitted list -- and an empty list is a subset of
    anything, so the one artefact that actually CARRIES its own fitted columns was
    exempted exactly when its record was unreadable. That is the same vacuous shape
    the player-side guard had, and it is now a refusal.
    """
    from nfl_predictor.models import manifest as manifest_mod

    (tmp_path / "qb_passing_td_model.pkl").write_bytes(b"stub")
    monkeypatch.setattr(manifest_mod, "_artifact_path", lambda name: tmp_path / name)
    monkeypatch.setattr(manifest_mod, "_load_pickle", lambda path: dict(record))

    with pytest.raises(ValueError, match="records no fitted feature columns"):
        manifest_mod._load_passing_td_model(
            {"qb_passing_td": {"distribution": "poisson", "n_train": 100}},
            manifest_mod._player_serving_columns())
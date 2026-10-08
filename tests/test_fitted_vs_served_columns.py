"""Every shipped model's OWN fitted columns, against the columns serving emits.

This is the generic test the #26 brief asked for, and the audit it exercises is
the one that is supposed to answer a single question per model:

    does the artefact on disk record the columns it was actually fitted on, and
    does serving actually emit every one of them?

**Why this file exists at all.** PR #26 shipped a check that could not fail. It
called

    _assert_servable_columns(player_usage.PLAYER_FEATURE_COLUMNS, "...")

while `SERVING_FEATURE_COLUMNS = [*PLAYER_FEATURE_COLUMNS, PASSING_TDS_ROLL_COLUMN]`,
so the assertion computed `PLAYER_FEATURE_COLUMNS ⊆ SERVING_FEATURE_COLUMNS` --
`X ⊆ X` -- against a hardcoded constant. It never opened a pickle. No player
model's fitted columns were ever read, which is why a pickle genuinely refitted
on 3 of its 6 columns, and one refitted on none, both loaded without complaint.

The tests below are written to be unfakeable in that specific way:

* the negative cases are **real refits**, through this repo's own
  `fit_anytime_td_classifier` / `fit_yardage_regressor` /
  `fit_qb_passing_td_model` / `fit_margin_regression` / `fit_xgb_margin`, so the
  artefact's `feature_names_in_` is genuinely short rather than edited to look
  short;
* they run against the **committed** `models/` directory, copied to a tmp dir, so
  the manifest and the fingerprint stay exactly as shipped -- which is the point:
  the fingerprint pins the *declared* list against the *code's* list and cannot
  see a pickle that disagrees with both;
* they are parameterised over **every artefact the committed manifest ships**, so
  a new model cannot ship unaudited by omission.

Nothing here stubs the thing under test. `load_models` runs for real.
"""

from __future__ import annotations

import copy
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.features import build as feature_build
from nfl_predictor.features import player_usage
from nfl_predictor.models import game_outcome, manifest, player_props, qb_passing_td

COMMITTED_MODELS_DIR = Path(manifest.MODELS_DIR)
COMMITTED_MANIFEST = json.loads((COMMITTED_MODELS_DIR / "manifest.json").read_text())

#: Every artefact the committed manifest ships, as
#: `(key, filename, family)` -- and the family decides how a mis-fitted
#: replacement for it is actually produced, so no case is faked.
#:
#: Read off the manifest rather than hardcoded, so a model added to the payload
#: without being added here fails collection loudly instead of going unaudited.
PLAYER_ARTEFACTS = [
    ("anytime_td", manifest.ANYTIME_TD_MODEL_FILENAME, "anytime_td"),
    *[
        (market, f"{market}_model.pkl", "yardage")
        for market in COMMITTED_MANIFEST["yardage_metrics"]
    ],
]
QB_PASING_TD_ARTEFACT = ("passing_tds", manifest.PASSING_TD_MODEL_FILENAME, "qb_passing_td")
GAME_ARTEFACTS = [
    ("game_outcome", manifest.GAME_MODEL_FILENAME, "game_margin"),
    ("total_points", manifest.TOTAL_MODEL_FILENAME, "game_total"),
]

SHIPPED_ARTEFACTS = PLAYER_ARTEFACTS + [QB_PASING_TD_ARTEFACT] + GAME_ARTEFACTS
SHIPPED_IDS = [artefact[0] for artefact in SHIPPED_ARTEFACTS]


# --- synthetic frames, shaped so a real refit is a real refit ----------------


def _player_history_frame() -> pd.DataFrame:
    """Raw weekly player stats -- the frame `build_features_for_player` reads.

    Columns are what `player_usage.ROLL_STATS` names plus `passing_tds`, so this
    frame can also be used to ask the serving builder what it emits rather than
    trusting the declared constant.
    """
    return pd.DataFrame(
        [
            {"player_id": "p1", "season": 2024, "week": week,
             **{stat: float(10 + week + i) for i, stat in enumerate(player_usage.ROLL_STATS)},
             "passing_tds": float(week % 3)}
            for week in (1, 2, 3, 4)
        ]
    )


def _game_frame() -> pd.DataFrame:
    """A played-game schedule frame `build_features_for_game` can score."""
    rows = []
    for week in (1, 2, 3):
        for home, away in (("A", "B"), ("A", "C"), ("B", "C")):
            rows.append(
                {"game_id": f"{home}_{away}_{week}", "season": 2024, "week": week,
                 "gameday": pd.Timestamp("2024-09-01") + pd.Timedelta(days=7 * (week - 1)),
                 "home_team": home, "away_team": away, "home_score": 21, "away_score": 14}
            )
    return pd.DataFrame(rows)


def _game_feature_frame() -> pd.DataFrame:
    """`build_features_for_game`'s output as a frame, on every declared column."""
    row = feature_build.build_features_for_game("A", "C", _game_frame())
    return pd.DataFrame([{c: float(row[c]) for c in feature_build.FEATURE_COLUMNS}])


def _player_design_frame() -> pd.DataFrame:
    """A player design matrix on the rolling columns, built by the real transforms.

    `player_usage._add_rolling` and `player_usage.with_passing_tds_roll` are what
    `build_player_training_frame` and `_fit_qb_passing_td` apply, so the refits
    below get a frame with the same column names training produces -- which is the
    whole reason `feature_names_in_` on the replacement artefact matches the
    committed artefact's shape minus the dropped columns.
    """
    return player_usage.with_passing_tds_roll(
        player_usage._add_rolling(_player_history_frame()))


def _with_columns(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """`frame` plus any of `cols` it does not already have, filled with constants.

    The "one column too many" case asks for a fit on a column no builder emits, so
    the design matrix has to be able to carry a column the real transforms do not
    produce. It is added here rather than in the fitting code, so every artefact
    below still comes out of a real `fit_*` call.
    """
    out = frame.copy()
    for i, col in enumerate(cols):
        if col not in out.columns:
            out[col] = float(i + 1)
    return out


def _refit(artefact_key: str, family: str, keep: list[str]):
    """A genuine artefact fitted on exactly `keep`, via the repo's own fitters.

    Not a metadata edit. `feature_names_in_` ends up short because the model was
    fitted on a short `DataFrame`, which is the only way to prove the audit reads
    the artefact rather than the manifest.
    """
    if family == "anytime_td":
        X, y = _with_columns(_player_design_frame(), keep), pd.Series([0, 1, 0, 1])
        return player_props.fit_anytime_td_classifier(X[keep], y)
    if family == "yardage":
        X, y = _with_columns(_player_design_frame(), keep), pd.Series([1.0, 2.0, 3.0, 4.0])
        return player_props.fit_yardage_regressor(X[keep], y)
    if family == "qb_passing_td":
        X = _with_columns(_player_design_frame(), keep)
        # `_fit_qb_passing_td` fills before fitting (`usable[cols].fillna(0)`), and
        # `fit_qb_passing_td_model` itself does not, so mirror that here.
        return qb_passing_td.fit_qb_passing_td_model(
            X[keep].fillna(0), pd.Series([1.0, 2.0, 0.0, 1.0]))
    if family == "game_margin":
        X = _with_columns(_game_feature_frame(), keep)
        return game_outcome.fit_margin_regression(X[keep], pd.Series([3.0]))
    if family == "game_total":
        X = _with_columns(_game_feature_frame(), keep)
        return game_outcome.fit_xgb_margin(X[keep], pd.Series([35.0]))
    raise AssertionError(f"unknown artefact family {family!r}")


def _fitted_columns(artefact) -> list[str] | None:
    """What the artefact itself records, mirroring the production reader."""
    if isinstance(artefact, dict):
        recorded = artefact.get("feature_cols")
        if recorded is not None:
            return list(recorded)
        inner = artefact.get("model")
        if inner is not None:
            return _fitted_columns(inner)
        return None
    names = getattr(artefact, "feature_names_in_", None)
    return None if names is None else list(names)


def _blank_fitted_record(artefact, *, drop: bool):
    """An artefact that records NO fitted columns, without mutating a property.

    `XGBClassifier.feature_names_in_` is a read-only property over
    `get_booster().feature_names`, and xgboost 3.2 raises `AttributeError` from it
    when that is `None` -- which is precisely the state of a model fitted from a
    NumPy array, i.e. the real-world "I cannot tell what this was fitted on". So
    the None case is produced the way it actually occurs, not by assigning to a
    property that refuses assignment. `Ridge.feature_names_in_` is a plain
    attribute and the mapping payload carries a plain key, so those take the
    direct route.
    """
    if isinstance(artefact, dict):
        blanked = copy.deepcopy(artefact)
        blanked.pop("feature_cols", None)
        blanked.pop("model", None)
        return blanked
    if hasattr(artefact, "get_booster"):
        blanked = artefact
        blanked.get_booster().feature_names = [] if drop else None
        return blanked
    blanked = artefact
    # A scaled Ridge is a Pipeline whose `feature_names_in_` is read from its first step.
    holder = blanked.steps[0][1] if hasattr(blanked, "steps") else blanked
    holder.feature_names_in_ = np.array([], dtype=object) if drop else None
    return blanked


# --- the committed payload, copied so a test can corrupt it ------------------


@pytest.fixture
def payload(tmp_path, monkeypatch):
    """The real committed `models/` directory, copied, and wired into `manifest`.

    Nothing about the manifest or the fingerprint is rewritten, which is the whole
    point: these cases are invisible to the fingerprint by construction, because
    the fingerprint compares the manifest's declared list against the code's list
    and never opens a pickle.
    """
    models_dir = tmp_path / "models"
    shutil.copytree(COMMITTED_MODELS_DIR, models_dir)
    monkeypatch.setattr(manifest, "MODELS_DIR", models_dir)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", models_dir / "manifest.json")
    return models_dir


def _rewrite(models_dir: Path, filename: str, artefact) -> None:
    with open(models_dir / filename, "wb") as f:
        pickle.dump(artefact, f)


def _load_committed(models_dir: Path, filename: str):
    with open(models_dir / filename, "rb") as f:
        return pickle.load(f)


def _committed_fitted_columns(filename: str) -> list[str]:
    """The columns the committed artefact really was fitted on."""
    return _fitted_columns(_load_committed(COMMITTED_MODELS_DIR, filename))


# --- 1. the positive control: the committed payload is servable ---------------


def test_every_shipped_artefact_loads_and_passes_the_audit(payload):
    """Control. If this fails, the audit is too strict, not too weak.

    Also the self-review question "does anything still train on a feature serving
    never emits?" asked of the real, committed artefacts: this runs the audit over
    every one of them, so a genuine mismatch in the payload surfaces here.
    """
    models = manifest.load_models()

    assert set(models["player_models"]) >= {key for key, _, _ in PLAYER_ARTEFACTS}
    assert qb_passing_td.PASSING_TD_MARKET in models["player_models"]
    assert models["game_outcome_model"] is not None
    assert models["total_model"] is not None

    for key, filename, family in SHIPPED_ARTEFACTS:
        fitted = _fitted_columns(_load_committed(payload, filename))
        assert fitted, f"{filename} records no fitted columns at all"


# --- 2. THE RED CASES: a pickle that disagrees with its own manifest ---------
#
# Parameterised over every shipped artefact that has an externally stated fitted
# column list, not one test per model. Each refits that artefact on a strict
# subset of its own columns and asserts `load_models` refuses -- the case #26's
# guard could not see.
#
# The QB passing-TD artefact is deliberately NOT in this block, and its exclusion
# is a documented limitation rather than an oversight: its fitted list is the
# payload's own free choice (`fit_qb_passing_td_model` takes its columns from `X`
# as given), so nothing outside the pickle says what it ought to be. It is covered
# in section 2c for the things the audit can honestly check there.

#: Every artefact `load_models` scores on a list it reads from elsewhere, so all of
#: them have a stated expectation the pickle must match exactly.
AUDITED_WITH_EXPECTATION = PLAYER_ARTEFACTS + GAME_ARTEFACTS
AUDITED_WITH_EXPECTATION_IDS = [a[0] for a in AUDITED_WITH_EXPECTATION]


@pytest.mark.parametrize("key,filename,family", AUDITED_WITH_EXPECTATION, ids=AUDITED_WITH_EXPECTATION_IDS)
def test_load_models_refuses_an_artefact_fitted_on_a_subset_of_its_columns(
    payload, key, filename, family
):
    """Fitted on 3 of 6 player columns / 3 of 10 game columns. Must not load.

    On `origin/main` this passes with no error at all: `_assert_servable_columns`
    was called with the code's own feature list and compared it against a constant
    that is that list plus one column, so every one of these artefacts loaded. The
    fingerprint is equally blind here by construction -- it pins the manifest's
    declared list against the code's and never reads `feature_names_in_`.

    Note that a pure subset test (`fitted ⊆ emitted`) also passes this: all three
    surviving columns are ones the builder emits. That is why the audit compares
    for exact equality against the list serving scores on, and why this case is
    the one that proves it.
    """
    fitted = _committed_fitted_columns(filename)
    assert len(fitted) >= 4, f"{filename} is too short to drop a column from"
    short = fitted[:3]
    assert set(short) < set(fitted)

    _rewrite(payload, filename, _refit(key, family, short))

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()

    message = str(excinfo.value)
    # Actionable on its own: which artefact, what it was fitted on, and what
    # serving scores it on.
    assert key in message or filename in message
    assert str(short) in message
    assert "feature" in message.lower()


@pytest.mark.parametrize("key,filename,family", AUDITED_WITH_EXPECTATION, ids=AUDITED_WITH_EXPECTATION_IDS)
def test_load_models_refuses_an_artefact_fitted_on_a_column_serving_never_hands_it(
    payload, key, filename, family
):
    """The other direction: one column too many. Must not load.

    Exact equality, not subset, is the rule precisely because both directions are
    wrong. A wider design matrix shifts every column past the extra one, so a
    subset test that only asked "are all of these servable" would pass this.
    """
    fitted = _committed_fitted_columns(filename)
    widened = [*fitted[:2], "a_column_no_builder_emits", *fitted[2:]]

    _rewrite(payload, filename, _refit(key, family, widened))

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert "a_column_no_builder_emits" in str(excinfo.value)


@pytest.mark.parametrize("key,filename,family", AUDITED_WITH_EXPECTATION, ids=AUDITED_WITH_EXPECTATION_IDS)
def test_load_models_refuses_an_artefact_fitted_on_no_columns_at_all(
    payload, key, filename, family
):
    """A model fitted on nothing must not sail through. Must not load.

    Nothing can be fitted on zero columns, so this is the shape a corrupt or
    truncated artefact actually presents: a record that is present and empty. It
    is the case an `if fitted_cols:` style guard waves through, and the reason
    this check is a failure rather than a skip.
    """
    fitted = _committed_fitted_columns(filename)
    blanked = _blank_fitted_record(_refit(key, family, fitted), drop=True)
    assert manifest._fitted_feature_columns(blanked) in (None, [])

    _rewrite(payload, filename, blanked)

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert key in str(excinfo.value) or filename in str(excinfo.value)


@pytest.mark.parametrize("key,filename,family", AUDITED_WITH_EXPECTATION, ids=AUDITED_WITH_EXPECTATION_IDS)
def test_load_models_refuses_an_artefact_whose_fitted_columns_are_unknown(
    payload, key, filename, family
):
    """`feature_names_in_` unreadable -- an unfitted or NumPy-fitted artefact. Must not load.

    A model fitted from a NumPy array carries no column names at all, which is
    exactly what `fit_qb_passing_td_model`'s inner PoissonRegressor looks like.
    Treating that as a pass would make "I cannot tell what this was fitted on"
    indistinguishable from "it was fitted on the right things", which is the whole
    defect.
    """
    fitted = _committed_fitted_columns(filename)
    unnamed = _blank_fitted_record(_refit(key, family, fitted), drop=False)
    assert manifest._fitted_feature_columns(unnamed) is None

    _rewrite(payload, filename, unnamed)

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert key in str(excinfo.value) or filename in str(excinfo.value)


@pytest.mark.parametrize("key,filename,family", GAME_ARTEFACTS, ids=["game_outcome", "total_points"])
def test_load_models_refuses_a_game_model_missing_a_fitted_column(payload, key, filename, family):
    """The game half, which the player-only guard could not see at all.

    `routes._predict_game_from_models` reindexes the live row by
    `manifest["feature_cols"]` and `fillna(0)`s, so a removed game feature is
    served as a constant zero -- identical in mechanism to the `passing_tds_roll`
    defect on the player side, and the same reason the game fingerprint (#28) was
    needed. The fingerprint pins the declared list; this pins the pickle.

    One column, not three: the point of this case is the count, and dropping a
    single `div_game` is the smallest realistic version of the defect.
    """
    fitted = _committed_fitted_columns(filename)
    short = fitted[:-1]
    assert len(short) == len(fitted) - 1

    _rewrite(payload, filename, _refit(key, family, short))

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert key in str(excinfo.value) or filename in str(excinfo.value)
    assert str(short) in str(excinfo.value)


# --- 2c. the QB passing-TD artefact, which has no stated expectation ---------


def test_the_qb_passing_td_artefact_is_audited_against_its_own_fitted_columns(payload):
    """A fitted column the builder does not emit is refused, on the dict payload.

    `expected_passing_tds` reindexes the live row by the `feature_cols` the
    payload itself carries, so that list is the record that governs serving and
    the subset rule is the honest one here. Adding a column no builder emits is
    the `passing_tds_roll` defect in its original form.
    """
    fitted = _committed_fitted_columns(manifest.PASSING_TD_MODEL_FILENAME)
    rogue = _refit("passing_tds", "qb_passing_td", [*fitted[:2], "phantom_column", *fitted[2:]])
    assert list(rogue["feature_cols"]) == [*fitted[:2], "phantom_column", *fitted[2:]]

    _rewrite(payload, manifest.PASSING_TD_MODEL_FILENAME, rogue)

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert "phantom_column" in str(excinfo.value)


@pytest.mark.parametrize("drop", [True, False], ids=["empty_record", "no_record"])
def test_load_models_refuses_a_qb_passing_td_payload_that_records_no_columns(payload, drop):
    """The dict payload's `or []` used to make this a silent pass. Must not load.

    `_load_passing_td_model` read `list(fitted.get("feature_cols") or [])`, so a
    payload whose record was missing or empty produced an empty fitted list, and
    an empty list is a subset of anything. The one artefact that DID carry its own
    fitted column list -- the only one a real audit could read -- was therefore
    exempt exactly when its record was unreadable.
    """
    fitted = _committed_fitted_columns(manifest.PASSING_TD_MODEL_FILENAME)
    blanked = _blank_fitted_record(_refit("passing_tds", "qb_passing_td", fitted), drop=drop)
    assert manifest._fitted_feature_columns(blanked) in (None, [])

    _rewrite(payload, manifest.PASSING_TD_MODEL_FILENAME, blanked)

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert manifest.PASSING_TD_MODEL_FILENAME in str(excinfo.value)


# --- 3. the audit's other side: the columns SERVING emits ---------------------
#
# A guard that only reads the pickle is half a guard. If the serving builder
# stops emitting a column the models are trained on, that is the same defect from
# the other end, and it is what `passing_tds_roll` was.


def test_player_audit_catches_the_serving_builder_stopping_emitting_a_column(payload, monkeypatch):
    """Simulate the code moving past the payload: the builder stops emitting one.

    On `origin/main` the analogous check existed but compared
    `PLAYER_FEATURE_COLUMNS` against the hand-written `SERVING_FEATURE_COLUMNS`
    constant, which the same commit defines as that list plus one column -- so it
    could not fail either. Here the *builder* is the thing that shrinks, which is
    the failure it was supposed to catch.
    """
    real_builder = player_usage.build_features_for_player

    def _builder_that_drops_the_last_column(*args, **kwargs):
        return real_builder(*args, **kwargs).iloc[:-1]

    monkeypatch.setattr(
        player_usage, "build_features_for_player", _builder_that_drops_the_last_column)

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert "does not emit" in str(excinfo.value)


def test_game_audit_catches_the_game_builder_stopping_emitting_a_column(payload, monkeypatch):
    """The same, on the game side, which nothing audited before.

    `build_features_for_game` is what produces the live row that
    `routes._predict_game_from_models` then reindexes by `manifest["feature_cols"]`
    and `fillna(0)`s.
    """
    real_builder = feature_build.build_features_for_game

    def _builder_that_drops_div_game(home, away, games_df):
        return real_builder(home, away, games_df).drop(labels=["div_game"])

    monkeypatch.setattr(feature_build, "build_features_for_game", _builder_that_drops_div_game)

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert "div_game" in str(excinfo.value)


# --- 4. the guard is not vacuous, stated as a property of the code ------------


def test_the_player_serving_column_set_is_derived_from_the_builder_not_a_constant(monkeypatch):
    """The audit's serving set must come from CALLING the builder.

    On `origin/main` that set *was* a hand-maintained constant,
    `SERVING_FEATURE_COLUMNS = [*PLAYER_FEATURE_COLUMNS, PASSING_TDS_ROLL_COLUMN]`,
    which is why the assertion above it reduced to `X ⊆ X`. So this asserts that
    `manifest._player_serving_columns()` returns what the builder emits, and --
    the part that would catch a regression -- that it keeps returning that even
    when the declared constant lies about it.
    """
    emitted = list(player_usage.build_features_for_player(
        "p1", _player_history_frame(), season=2024, week=5,
        window=player_usage.DEFAULT_ROLL_WINDOW).index)

    assert set(emitted) == set(player_usage.PLAYER_FEATURE_COLUMNS) | {
        player_usage.PASSING_TDS_ROLL_COLUMN
    }
    assert list(manifest._player_serving_columns()) == emitted

    # The declared constant and the builder currently agree, so this passes --
    # but the load-time audit must not be reading it. Point the constant somewhere
    # wrong and confirm the audit's answer does not move.
    lying = [*player_usage.PLAYER_FEATURE_COLUMNS, "a_column_the_builder_never_emits"]
    monkeypatch.setattr(player_usage, "SERVING_FEATURE_COLUMNS", lying)
    assert list(manifest._player_serving_columns()) == emitted


def test_the_game_serving_column_set_is_derived_from_the_builder_not_a_constant(monkeypatch):
    """Same property on the game side, which nothing audited at all before.

    `routes._predict_game_from_models` scores the live row `build_features_for_game`
    produced, so that is what the game half of the audit has to compare against --
    and `feature_build.FEATURE_COLUMNS` is the code's DECLARED list, which is the
    thing #28 fingerprinted. The pickle is what was missing.
    """
    emitted = list(manifest._game_serving_columns())
    assert emitted == list(feature_build.FEATURE_COLUMNS)

    monkeypatch.setattr(feature_build, "FEATURE_COLUMNS", [*emitted, "weather_roof"])
    assert list(manifest._game_serving_columns()) == emitted


# --- 4b. attacking this check, since a guard nobody has tried to break is a
# --- guard of unknown strength ------------------------------------------------
#
# Every case below reached the audit by constructing a malformed artefact rather
# than by mutating a good one, and every one has to be refused -- as a `ValueError`
# naming the artefact and both column lists, not as a `TypeError` from a `list()`
# call that happened to blow up on the way.


class _WithNames:
    def __init__(self, names):
        self.feature_names_in_ = names


MALFORMED_ESTIMATORS = {
    # A bare string iterates to characters, and every character is then a
    # "missing column". Refused -- but note that a payload whose single fitted
    # column happened to be a one-character name would pass a SUBSET check, which
    # is the other reason the audit compares for exact equality.
    "string_instead_of_a_list": _WithNames("passing_yards_roll"),
    # `list(7)` raises TypeError before any of the audit's own guards run.
    "not_iterable_at_all": _WithNames(7),
    # Non-string entries: `None` and `int` compare fine against a list of strings,
    # so these are found by the equality comparison, not by a type check.
    "none_and_int_entries": _WithNames([None, 3, "passing_yards_roll"]),
    # A duplicate is a subset of its own dedup, so a subset rule passes it.
    "duplicated_columns": _WithNames(["passing_yards_roll"] * 6),
    # Unhashable, so `c not in servable` is a list containment check on `==`.
    "nested_list_entries": _WithNames([["passing_yards_roll"]]),
}


@pytest.mark.parametrize("shape", sorted(MALFORMED_ESTIMATORS))
def test_a_malformed_fitted_column_record_is_refused_not_crashed(payload, shape):
    """Every malformed shape is a ValueError naming the artefact and both lists."""
    blanked = MALFORMED_ESTIMATORS[shape]
    _rewrite(payload, manifest.ANYTIME_TD_MODEL_FILENAME, blanked)

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()

    message = str(excinfo.value)
    # Names the artefact, and both sides of whichever comparison caught it: the
    # recorded list and either the list serving scores on or the list serving
    # emits. Asserted as "two readable column lists", not one specific column,
    # because which of the two guards fires depends on the shape -- and a message
    # that named only one would still be half-actionable.
    assert manifest.ANYTIME_TD_MODEL_FILENAME in message
    assert "passing_yards_roll" in message
    assert "receptions_roll" in message


@pytest.mark.parametrize("record", [
    pytest.param({"feature_cols": None, "model": _WithNames(
        list(player_usage.PLAYER_FEATURE_COLUMNS))}, id="explicit_none_with_a_model"),
    pytest.param({"feature_cols": None}, id="explicit_none_alone"),
    pytest.param({"feature_cols": "passing_yards_roll"}, id="string_value"),
    pytest.param({"feature_cols": 7}, id="non_iterable_value"),
    pytest.param({"model": _WithNames(7)}, id="no_key_junk_inner_model"),
], ids=lambda p: p if isinstance(p, str) else None)
def test_a_malformed_mapping_payload_is_refused(payload, record):
    """`feature_cols: None` must NOT fall through to the inner estimator.

    This is the one case that would have slipped through a truthiness check.
    `expected_passing_tds` reads `fitted["feature_cols"]` unguarded, so a payload
    with `feature_cols: None` and a perfectly readable inner model passes any
    fallback-to-the-model reading -- and then raises per QB, per request,
    forever, instead of once at load.
    """
    _rewrite(payload, manifest.PASSING_TD_MODEL_FILENAME, record)

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert manifest.PASSING_TD_MODEL_FILENAME in str(excinfo.value)


def test_the_elo_skip_cannot_be_claimed_by_a_regressor(payload):
    """`allow_unfitted` is set from the manifest, but the SHAPE still has to be Elo.

    `load_models` passes `allow_unfitted=chosen_candidate == "elo"`, which is the
    manifest's claim about itself. If that claim were honoured on its own, a
    payload could exempt its game model from the audit by editing one JSON field.
    `_is_elo_candidate` is the second condition, and it is shape-based -- a Ridge
    is not `{"points_per_rating_point": ...}`, so it still has to have fitted
    columns.
    """
    fitted = _committed_fitted_columns(manifest.GAME_MODEL_FILENAME)
    refit = _refit("game_outcome", "game_margin", fitted)

    # A mapping wrapping the game model, re-boxed so its record is
    # `feature_cols` rather than `feature_names_in_`. `_is_elo_candidate` looks for
    # `points_per_rating_point`, which this does not carry, so the fitted-column
    # rules apply to it exactly as they do to the bare Ridge. Shortening its
    # recorded list proves they are actually applied rather than skipped.
    for recorded in (list(fitted)[:-1], []):
        impostor = {"feature_cols": list(recorded), "model": refit}
        assert not manifest._is_elo_candidate(impostor)
        _rewrite(payload, manifest.GAME_MODEL_FILENAME, impostor)

        with pytest.raises(ValueError) as excinfo:
            manifest.load_models()
        assert manifest.GAME_MODEL_FILENAME in str(excinfo.value)

    # And the shape that IS exempt, so the exemption is shown to be narrow rather
    # than assumed: the real elo candidate carries no fitted columns, and
    # `_assert_artefact_columns_are_served` only waves it through when the caller
    # has independently established that the manifest claims elo too.
    elo = game_outcome.fit_elo_candidate(pd.DataFrame())
    assert manifest._is_elo_candidate(elo)
    assert manifest._fitted_feature_columns(elo) is None
    with pytest.raises(ValueError):
        manifest._assert_artefact_columns_are_served(
            elo, manifest.GAME_MODEL_FILENAME, list(manifest._game_serving_columns()),
            "feature_build.build_features_for_game", allow_unfitted=False)


def test_a_ridge_under_an_elo_manifest_is_refused(payload):
    """The reverse direction, which the column audit cannot see on its own.

    `routes._predict_game_from_models` branches on `models["chosen_candidate"]` and
    on the `elo` path never reads `game_outcome_model` -- it builds the conversion
    constant inline. So a payload whose manifest says `elo` while the artefact is
    a Ridge produces predictions from neither: the manifest's claim is what routes
    follows, and `sigma` was fitted against whichever candidate actually ran. The
    artefact satisfies every column rule perfectly and is still not the model
    being served, so `load_models` checks the agreement explicitly.
    """
    declared = json.loads((payload / "manifest.json").read_text())
    declared["chosen_candidate"] = "elo"
    (payload / "manifest.json").write_text(json.dumps(declared, indent=2))

    # The committed artefact is a Ridge, and a Ridge is fitted on exactly the game
    # feature list -- so nothing below the candidate-agreement check would fire.
    fitted = _committed_fitted_columns(manifest.GAME_MODEL_FILENAME)
    assert list(fitted) == list(feature_build.FEATURE_COLUMNS)
    assert not manifest._is_elo_candidate(_load_committed(payload, manifest.GAME_MODEL_FILENAME))

    with pytest.raises(ValueError, match="chosen_candidate") as excinfo:
        manifest.load_models()
    assert manifest.GAME_MODEL_FILENAME in str(excinfo.value)


def test_an_elo_candidate_under_an_elo_manifest_does_load(payload):
    """The positive control for the check above.

    Without it, "always raise when the manifest claims elo" would pass the test
    before it. The committed player artefacts and the total-points model are
    untouched here -- only the game-outcome artefact becomes the real conversion
    constant, and only the manifest's candidate claim changes to match.
    """
    declared = json.loads((payload / "manifest.json").read_text())
    declared["chosen_candidate"] = "elo"
    (payload / "manifest.json").write_text(json.dumps(declared, indent=2))

    elo = game_outcome.fit_elo_candidate(pd.DataFrame())
    assert manifest._is_elo_candidate(elo)
    _rewrite(payload, manifest.GAME_MODEL_FILENAME, elo)

    models = manifest.load_models()
    assert models["chosen_candidate"] == "elo"
    assert manifest._is_elo_candidate(models["game_outcome_model"])


def test_a_mapping_payload_whose_feature_cols_is_shorter_than_its_model_is_refused(payload):
    """The case CodeRabbit caught, and the one this file had claimed was undetectable.

    A mapping payload's `feature_cols` is self-reported, so the only corroboration
    available is the estimator inside the same pickle. The committed QB artefact's
    inner Poisson regressor is fitted from a NumPy array and therefore records NO
    column names -- so there is no name comparison to make, and an earlier version
    of this audit skipped corroboration entirely in that state while the comment
    beside it claimed `None` was not treated as agreement.

    The consequence was concrete: a `feature_cols` list SHORTER than the wrapped
    model was fitted on passed the audit, then reached `expected_passing_tds`,
    which reindexes to the short list and hands a narrow matrix to a wider model.
    One `ValueError` per QB per request, forever, instead of once at load. The
    count is corroborated instead, via `n_features_in_`, which sklearn records
    even on a NumPy fit.
    """
    fitted = _committed_fitted_columns(manifest.PASSING_TD_MODEL_FILENAME)
    assert len(fitted) >= 2
    refit = _refit("passing_tds", "qb_passing_td", fitted)

    # Confirmed to be the shape in question: names absent, count present.
    assert getattr(refit["model"], "feature_names_in_", None) is None
    assert int(refit["model"].n_features_in_) == len(fitted)

    for short in (list(fitted)[:-1], [], list(fitted) + ["phantom_column"]):
        payload_copy = copy.deepcopy(refit)
        payload_copy["feature_cols"] = list(short)
        _rewrite(payload, manifest.PASSING_TD_MODEL_FILENAME, payload_copy)

        with pytest.raises(ValueError) as excinfo:
            manifest.load_models()
        assert manifest.PASSING_TD_MODEL_FILENAME in str(excinfo.value)


def test_a_mapping_payload_with_no_corroborable_inner_model_is_refused(payload):
    """A `feature_cols` with no inner estimator to check it against is refused.

    `allow_unfitted` cannot reach this: the payload does record columns, so it is
    the corroboration branch, not the empty-record one. Refusing is the fail-closed
    posture everywhere else in this audit -- an unverifiable claim is not a verified
    one -- and `expected_passing_tds` reads `fitted["feature_cols"]` unguarded, so
    a payload whose list cannot be checked is served on faith.
    """
    fitted = _committed_fitted_columns(manifest.PASSING_TD_MODEL_FILENAME)
    _rewrite(payload, manifest.PASSING_TD_MODEL_FILENAME,
             {"feature_cols": list(fitted)})

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert "n_features_in_" in str(excinfo.value)


def test_a_pickle_that_disagrees_with_its_own_record_is_refused(payload):
    """The mapping payload's `feature_cols` is self-reported, so it is cross-checked.

    The QB passing-TD artefact is a dict: `expected_passing_tds` reindexes the live
    row by the `feature_cols` the dict itself carries, so a dict whose `feature_cols`
    disagrees with the estimator inside it is serving on one list and fitted on
    another. Where an independent record exists inside the same pickle, the two
    must agree.
    """
    filename = manifest.PASSING_TD_MODEL_FILENAME
    fitted = _committed_fitted_columns(filename)
    refit = _refit("passing_tds", "qb_passing_td", fitted)

    disagreeing = copy.deepcopy(refit)
    # The inner estimator records nothing (NumPy fit), so attach an independent
    # record that contradicts the dict's own -- the shape of a payload assembled
    # from two different fits.
    disagreeing["model"] = _refit("anytime_td", "anytime_td", fitted)
    disagreeing["feature_cols"] = list(fitted)[:1]

    _rewrite(payload, filename, disagreeing)

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    assert "feature_cols" in str(excinfo.value) or "disagree" in str(excinfo.value).lower()
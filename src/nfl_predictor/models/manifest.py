"""Train, save, and load the game-outcome and player-prop models."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from datetime import datetime, timezone

import pandas as pd

from ..config import MODELS_DIR
from ..data import player_stats, schedules
from ..evaluate import walk_forward
from ..features import build as feature_build
from ..features import player_usage
from . import game_outcome, player_props, qb_passing_td

MANIFEST_PATH = MODELS_DIR / "manifest.json"
GAME_MODEL_FILENAME = "game_outcome_model.pkl"
TOTAL_MODEL_FILENAME = "total_points_model.pkl"
ANYTIME_TD_MODEL_FILENAME = "anytime_td_model.pkl"
PASSING_TD_MODEL_FILENAME = "qb_passing_td_model.pkl"

DEFAULT_TRAIN_SEASONS = 8


def _save_pickle(obj, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _artifact_path(filename: str):
    """Resolve an artifact path at use time for configurable model storage."""
    return MODELS_DIR / filename


def _yardage_model_path(market: str):
    return _artifact_path(f"{market}_model.pkl")


def _fit_qb_passing_td(player_train_df: pd.DataFrame) -> dict | None:
    """Fit the QB passing-TD count model, or None when there is no QB history.

    The rolling TD column is built by `player_usage.with_passing_tds_roll`, not by
    a second hand-rolled transform here. The reason it is not simply appended to
    `PLAYER_FEATURE_COLUMNS` is unchanged -- that list is what the anytime-TD
    classifier and every yardage regressor are fitted on and what `predict_props`
    indexes by name, so widening it would silently change the feature count of
    every model already committed -- but the column itself now has exactly one
    implementation, shared with `build_features_for_player`. It previously had
    two: one here, and none at serving, so every QB was projected from
    `fillna(0)` on a feature the model had a fitted coefficient for.
    """
    qb = player_train_df[player_train_df["position"] == "QB"].copy()
    if qb.empty:
        return None
    # Same shift(1)-then-rolling discipline as `_add_rolling`, for the same
    # reason: a pregame feature cannot know the game being predicted. Built
    # through `player_usage` so it cannot drift from the column
    # `build_features_for_player` emits at serving time.
    qb = player_usage.with_passing_tds_roll(qb)

    cols = [player_usage.PASSING_TDS_ROLL_COLUMN, *qb_passing_td.MU_FEATURE_COLUMNS]
    # Asserted at fit time as well as at load time: a model that cannot be served
    # should not be written to disk in the first place. The serving set is passed
    # in rather than read from a constant inside the assertion -- see
    # `_assert_servable_columns`, which is where the old signature turned every
    # call into a self-comparison.
    _assert_servable_columns(
        cols, "the QB passing-TD model", _player_serving_columns(),
        "player_usage.build_features_for_player")
    usable = qb.dropna(subset=cols, how="all")
    if usable.empty:
        return None
    fitted = qb_passing_td.fit_qb_passing_td_model(
        usable[cols].fillna(0), usable["passing_tds"])
    return fitted


def _passing_td_manifest_entry(fitted: dict | None) -> dict | None:
    """The passing-TD model's provenance for the manifest, or None.

    Both log losses are written, not just the winner's: the choice between
    Poisson and negative binomial is only meaningful if the losing number is
    inspectable, and a manifest that records one number cannot be re-checked
    against the history later.
    """
    if fitted is None:
        return None
    return {
        "distribution": fitted["distribution"],
        "log_loss": fitted["log_loss"],
        "alpha": fitted["alpha"],
        "n_train": fitted["n_train"],
        "variance_ratio": fitted["variance_ratio"],
    }


def train_all(seasons: list[int] | None = None) -> dict:
    """Fit all models, persist their artifacts, and return their manifest."""
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    seasons = seasons or schedules.default_completed_seasons(n=DEFAULT_TRAIN_SEASONS)

    games_df = schedules.load_training_data(seasons)
    train_df, feature_cols = feature_build.build_training_frame(games_df)

    folds = walk_forward.prepare_folds(games_df, min_train_seasons=max(1, len(seasons) - 2))
    if not folds:
        raise ValueError("Training requires at least one walk-forward validation fold.")
    candidate_scores = {}
    for candidate in ("elo", "ridge", "xgb"):
        scored = walk_forward.evaluate_candidate(folds, candidate) if folds else pd.DataFrame()
        candidate_scores[candidate] = float(scored["log_loss"].mean()) if not scored.empty else float("inf")
    chosen = min(candidate_scores, key=candidate_scores.get)

    X_train = train_df[feature_cols]
    y_margin = train_df["margin"]
    y_total = train_df["total_points"]

    if chosen == "elo":
        game_model = game_outcome.fit_elo_candidate(train_df)
    elif chosen == "ridge":
        game_model = game_outcome.fit_margin_regression(X_train, y_margin)
    else:
        game_model = game_outcome.fit_xgb_margin(X_train, y_margin)

    if chosen == "elo":
        margin_preds = train_df.apply(
            lambda r: game_outcome.predict_margin_elo(
                game_model, r["rating_diff"], r["home_rest_days"], r["away_rest_days"]
            ),
            axis=1,
        )
        sigma_model = type("_", (), {"predict": lambda self, X: margin_preds.to_numpy()})()
        sigma = game_outcome.residual_sigma(sigma_model, X_train, y_margin)
    else:
        sigma = game_outcome.residual_sigma(game_model, X_train, y_margin)

    total_model = game_outcome.fit_xgb_margin(X_train, y_total)
    total_sigma = game_outcome.residual_sigma(total_model, X_train, y_total)

    _save_pickle(game_model, _artifact_path(GAME_MODEL_FILENAME))
    _save_pickle(total_model, _artifact_path(TOTAL_MODEL_FILENAME))

    player_df_raw = player_stats.fetch_weekly_player_stats(seasons)
    player_train_df, player_feature_cols = player_usage.build_player_training_frame(player_df_raw)
    player_train_df = player_train_df.dropna(subset=player_feature_cols, how="all")
    X_player = player_train_df[player_feature_cols].fillna(0)

    anytime_td_model = player_props.fit_anytime_td_classifier(X_player, player_train_df["anytime_td"])
    _save_pickle(anytime_td_model, _artifact_path(ANYTIME_TD_MODEL_FILENAME))

    yardage_metrics = {}
    for market, target_col in player_props.YARDAGE_TARGETS.items():
        path = _yardage_model_path(market)
        subset = player_train_df[player_train_df[target_col] > 0]
        if subset.empty:
            path.unlink(missing_ok=True)
            continue
        model = player_props.fit_yardage_regressor(subset[player_feature_cols].fillna(0), subset[target_col])
        _save_pickle(model, path)
        yardage_metrics[market] = {"n_train": int(len(subset))}

    # QB passing-TD projection. QBs only, on the rolling features a pregame
    # feature row carries. `passing_tds_roll` IS in the model: it is emitted by
    # `player_usage.build_features_for_player` on the same shift(1)-then-rolling
    # discipline it is trained on here, and `_assert_servable_columns` inside
    # `_fit_qb_passing_td` refuses to save a model whose fitted columns the
    # serving builder cannot produce -- passing the columns
    # `_player_serving_columns()` read by CALLING that builder. This comment
    # previously said the opposite -- that the column was "deliberately absent"
    # from the model because serving did not emit it -- and the code beside it
    # fitted on it anyway, so every QB was served a zero for a column the model
    # had a fitted coefficient for.
    passing_td = _fit_qb_passing_td(player_train_df)
    if passing_td is None:
        _artifact_path(PASSING_TD_MODEL_FILENAME).unlink(missing_ok=True)
    else:
        _save_pickle(passing_td, _artifact_path(PASSING_TD_MODEL_FILENAME))

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(int(s) for s in seasons),
        "n_train": int(len(train_df)),
        "feature_cols": feature_cols,
        "player_feature_cols": player_feature_cols,
        "chosen_candidate": chosen,
        "candidate_scores": candidate_scores,
        "sigma": sigma,
        "total_sigma": total_sigma,
        "yardage_metrics": yardage_metrics,
        "qb_passing_td": _passing_td_manifest_entry(passing_td),
        "artifact_fingerprint": artifact_fingerprint(player_feature_cols, feature_cols),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def artifact_fingerprint(player_feature_cols: list[str], feature_cols: list[str]) -> dict:
    """What an artefact must agree with to be servable, recorded at fit time.

    The silent-degradation case this exists to close, measured on the artefact
    this branch started from: `models/manifest.json` listed 5
    `player_feature_cols` while the code had 6 (the 6th, `receptions_roll`, had
    been added), and `anytime_td_model.pkl` was fitted on 5. `load_models()`
    returned that payload without complaint and every prediction came out of a
    model that no longer matched the code.

    Two reasons that was invisible, and this checks both:

    * `_assert_servable_columns` tests `fitted ⊆ SERVING`. The stale 5 columns
      ARE a subset of the 7 servable columns, so the one guard that existed
      passed. A subset test cannot see an artefact that is missing a feature. (It
      is worth being precise about how weak that guard was, since this paragraph
      reads as though it described a working one: it was called with
      `player_usage.PLAYER_FEATURE_COLUMNS` and compared against a constant
      DEFINED as that list plus one column, so it evaluated `X ⊆ X + 1`, opened no
      pickle, and could not fail. See `_assert_servable_columns`.)
    * The label is never recomputed at serving. `predict_props` scores the
      committed pickle from features alone, so an artefact fitted against the
      OLD `anytime_td` definition serves exactly like one fitted against the
      new one -- right shape, wrong meaning, no error anywhere.

    So the fingerprint pins the *meaning*, not just the shape:
    `player_feature_cols` (exact list, not a subset),
    `feature_cols` -- the GAME-level list from `features.build.FEATURE_COLUMNS`,
    the same exact-list-not-a-subset rule for the same reason -- and
    `anytime_td_label_version` from `player_usage`. Bump the version whenever the
    definition changes, and every artefact fitted before that bump stops loading
    loudly instead of quietly.

    Both feature lists are recorded, because they feed two disjoint sets of
    models and drift in either is equally silent: `player_feature_cols` covers
    `anytime_td_model.pkl` and every `*_yards`/`carries`/`receptions` regressor,
    and `feature_cols` covers `game_outcome_model.pkl` (when the chosen candidate
    is ridge or xgb) and `total_points_model.pkl` plus both residual sigmas.
    Fingerprinting only the player half left the game half to the subset test,
    which cannot see a missing feature -- add a column to `FEATURE_COLUMNS`,
    retrain nothing, and the committed total-points model keeps serving fitted
    against the old list while nothing complains.

    `feature_cols` is a REQUIRED argument rather than defaulting to
    `feature_build.FEATURE_COLUMNS`. A default would let a caller record the
    code's current list for a fit that used a different one -- reintroducing the
    manifest-compared-with-itself defect that `_verify_artifact_fingerprint`'s
    docstring describes, one level down.
    """
    return {
        "player_feature_cols": list(player_feature_cols),
        "feature_cols": list(feature_cols),
        "anytime_td_label_version": player_usage.ANYTIME_TD_LABEL_VERSION,
    }


def _verify_artifact_fingerprint(manifest: dict) -> None:
    """Raise when the committed artefacts disagree with the code they serve under.

    Called from `load_models`. A manifest written before this check existed has
    no `artifact_fingerprint` key at all, and that is treated as a FAILURE, not a
    pass -- the absence of a fingerprint is exactly the state this is detecting,
    so skipping the check when it is missing would defeat it.

    The error names both sides, because "your model is stale" is not actionable
    on its own and the whole cost of this bug class was that nothing said
    anything at all.

    **The feature half is compared against the CODE, not against the manifest.**
    The first version built its expectation as
    `artifact_fingerprint(manifest.get("player_feature_cols") or [])`, so
    `expected_features` was a copy of the manifest's own
    `player_feature_cols` while `fitted_features` came out of the manifest's
    `artifact_fingerprint`: the manifest was compared with itself. It passed for
    exactly the artefact it exists to catch -- add a column to `ROLL_STATS`
    (hence to `PLAYER_FEATURE_COLUMNS`), do not retrain, and the committed
    manifest and its fingerprint still agree with each other on the old shorter
    list, so `load_models` served models fitted on features the code has moved
    past. It read green only because the committed artefact happened to be up to
    date; a guard that is right by coincidence is not a guard.

    `player_usage.PLAYER_FEATURE_COLUMNS` is the authority instead, because it is
    the one thing training and serving both derive from: `train_all` fits the
    player models on the list `build_player_training_frame` returns (that
    constant) and `player_props.predict_props` reindexes a live row by
    `manifest["player_feature_cols"]`, which the fingerprint check below now
    pins to it. The label-version half below was always read from the code and is
    left exactly as it was.

    **The same defect existed on the GAME-level list and was not fixed by that.**
    The manifest records a second, unrelated feature list as `feature_cols` --
    `features.build.FEATURE_COLUMNS`: the per-game columns
    `home_pregame_rating`, `away_pregame_rating`, `rating_diff`, the four rolling
    scoring/conceding averages, both rest-day counts and `div_game`. It is built
    by `build_training_frame` and emitted by `build_features_for_game`, so it is
    game-level and not a restatement of the player list. It feeds a disjoint set
    of models: `total_points_model.pkl` always, `game_outcome_model.pkl` whenever
    the chosen candidate is ridge or xgb, and both residual sigmas. Nothing
    pinned it. Add a column to `FEATURE_COLUMNS`, retrain nothing, and
    `routes._predict_game_from_models` keeps reindexing the live row by the
    manifest's old list and scoring the committed models on it: the new feature
    is silently never used. Nothing here can see it either: as of this commit the
    ONLY fitted-vs-served guard, `_assert_servable_columns`, was called once, with
    `player_usage.PLAYER_FEATURE_COLUMNS`, from a body that read
    `player_usage.SERVING_FEATURE_COLUMNS` -- a player-side constant defined as
    that same list plus `passing_tds_roll`, so it evaluated `X ⊆ X + 1` and
    inspected no model. Its verdict would have been about the wrong models even if
    it had worked. So the check below is the exact same comparison against the
    exact same kind of authority -- `feature_build.FEATURE_COLUMNS`, the constant
    `build_training_frame` returns to `train_all` -- by the same exact-list-not-
    subset rule, for the same reason a subset cannot detect a missing feature.
    The player-side checks above are untouched by this.

    **What this docstring deliberately does not claim:** that the fingerprint
    covers the game MODELS. It does not, and it cannot. It compares the manifest's
    declared lists against the code's; the pickles are opened by
    `_assert_artefact_columns_are_served`, which is a separate check with a
    separate failure mode. Together they are three-way agreement between the
    manifest, the code and the artefact, and a payload needs all three. Read as
    "the game models are now verified", this paragraph would be the same kind of
    overstatement that the removed `_assert_servable_columns` docstring made.
    """
    recorded = manifest.get("artifact_fingerprint")
    if recorded is None:
        raise ValueError(
            f"{MANIFEST_PATH} has no 'artifact_fingerprint' key, so the committed "
            "models cannot be verified against the code that serves them. It was "
            "written before fingerprinting existed, and an unverified artefact is "
            "the case this check exists to catch. Re-run training "
            "(`python -m nfl_predictor.models.manifest`) to write one."
        )

    # The code's own fingerprint: `PLAYER_FEATURE_COLUMNS` and
    # `feature_build.FEATURE_COLUMNS` from the source, and the label version from
    # the source. Nothing here is read out of the manifest, because a check whose
    # expectation comes from the artefact it is checking cannot detect the
    # artefact drifting from the code -- see the docstring.
    current = artifact_fingerprint(
        player_usage.PLAYER_FEATURE_COLUMNS, feature_build.FEATURE_COLUMNS)
    expected_features = current["player_feature_cols"]
    fitted_features = list(recorded.get("player_feature_cols") or [])
    if fitted_features != expected_features:
        raise ValueError(
            f"{MANIFEST_PATH} records models fitted on {fitted_features} but the "
            f"code's player features are now {expected_features} "
            f"({player_usage.__name__}.PLAYER_FEATURE_COLUMNS). The committed "
            "pickles were fitted on a different feature set, so every prediction "
            "would come from a stale model. Re-run training "
            "(`python -m nfl_predictor.models.manifest`)."
        )

    # The fingerprint and the manifest's own `player_feature_cols` are written
    # from one value at fit time, so they must still agree -- and this is not
    # belt-and-braces. `load_models` hands `player_models["feature_cols"]` to
    # `player_props.predict_props`, which reindexes every live row by the
    # manifest's list, not by the fingerprint's. A manifest whose top-level list
    # had drifted would therefore score a 6-column model on a narrower feature
    # set while the check above stayed green, which is the same silent
    # degradation from the other direction.
    manifest_features = list(manifest.get("player_feature_cols") or [])
    if manifest_features != fitted_features:
        raise ValueError(
            f"{MANIFEST_PATH} records an artefact fingerprint fitted on "
            f"{fitted_features} but its own 'player_feature_cols' is "
            f"{manifest_features}. Serving reindexes every player row by the "
            "manifest's list, so the two disagreeing means the models would be "
            "scored on features they were not fitted on. Re-run training "
            "(`python -m nfl_predictor.models.manifest`)."
        )

    # --- the game-level half, same rule, same fail-closed posture -------------
    #
    # `recorded.get("feature_cols")` is checked for presence SEPARATELY rather
    # than folded into the comparison below, because a fingerprint written before
    # this half existed has no such key, and `list(None or [])` would then report
    # it as "fitted on []" -- technically a mismatch, but an error message naming
    # an empty list nobody fitted on. Absence is the state being detected here
    # (it is every artefact in the wild right now), so it is refused in its own
    # words and never skipped.
    if recorded.get("feature_cols") is None:
        raise ValueError(
            f"{MANIFEST_PATH} has no game-level 'feature_cols' in its "
            "'artifact_fingerprint', so the committed game models cannot be "
            "verified against the code that serves them. A fingerprint written "
            "before the game-level list was fingerprinted cannot vouch for "
            "`total_points_model.pkl` or a ridge/xgb `game_outcome_model.pkl`, "
            "and an unverified artefact is exactly the case this check exists to "
            "catch. Re-run training "
            "(`python -m nfl_predictor.models.manifest`) to write one."
        )

    expected_game_features = current["feature_cols"]
    fitted_game_features = list(recorded["feature_cols"])
    if fitted_game_features != expected_game_features:
        raise ValueError(
            f"{MANIFEST_PATH} records the game models fitted on "
            f"{fitted_game_features} but the code's game features are now "
            f"{expected_game_features} "
            f"({feature_build.__name__}.FEATURE_COLUMNS). The committed pickles "
            "were fitted on a different feature set, so every game prediction "
            "would come from a stale model. Re-run training "
            "(`python -m nfl_predictor.models.manifest`)."
        )

    # The fingerprint and the manifest's own top-level `feature_cols` are written
    # from one value at fit time, so they must still agree -- the exact reason the
    # player half has this check, and it transfers unchanged. `load_models`
    # returns `manifest["feature_cols"]` and
    # `routes._predict_game_from_models` reindexes the live feature row by it, so
    # a manifest with a current fingerprint but a narrowed top-level list would
    # score a full-width total-points model on a single column while the check
    # above stayed green.
    manifest_game_features = manifest.get("feature_cols")
    if manifest_game_features is None or list(manifest_game_features) != fitted_game_features:
        raise ValueError(
            f"{MANIFEST_PATH} records an artefact fingerprint fitted on "
            f"{fitted_game_features} but its own 'feature_cols' is "
            f"{manifest_game_features}. Serving reindexes every game row by the "
            "manifest's list, so the two disagreeing means the models would be "
            "scored on features they were not fitted on. Re-run training "
            "(`python -m nfl_predictor.models.manifest`)."
        )

    expected_version = current["anytime_td_label_version"]
    fitted_version = recorded.get("anytime_td_label_version")
    if fitted_version != expected_version:
        raise ValueError(
            f"{MANIFEST_PATH} records the anytime-TD model fitted against label "
            f"definition v{fitted_version}, but the code now defines v{expected_version} "
            f"({player_usage.__name__}.ANYTIME_TD_LABEL_VERSION). The committed model "
            "predicts a different market than the one being served or graded: "
            "`anytime_td` is now rushing + receiving TDs only, excluding passing "
            "TDs. Serving it would score every prediction against a definition it "
            "was never fitted on. Re-run training "
            "(`python -m nfl_predictor.models.manifest`)."
        )


def load_manifest() -> dict:
    """Load the metadata produced by :func:`train_all`."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No trained models found. Run `python -m nfl_predictor.models.manifest` first.")
    return json.loads(MANIFEST_PATH.read_text())


def model_version(manifest: dict) -> str:
    """Identifies the trained model behind a prediction: candidate + training time.

    Stored on every frozen snapshot, because the trade hub's gate is keyed on
    (engine, engine_version) and a prediction whose provenance is unknown cannot be graded against
    the same population as the ones that were.
    """
    return f"{manifest['chosen_candidate']}@{manifest['trained_at']}"


def _assert_servable_columns(
    fitted_cols: list[str],
    model_name: str,
    servable_columns: list[str],
    servable_source: str,
) -> None:
    """Refuse to serve a model whose fitted features the serving builder lacks.

    `player_props.predict_props` and `qb_passing_td.expected_passing_tds` both
    score a live row with `reindex(cols).fillna(0)`, so a column the model was
    fitted on and the pregame builder does not emit is served as a **constant
    zero** -- every prediction from that model is then wrong, and nothing in the
    payload says so. That is not hypothetical: `passing_tds_roll` was fitted on
    (see `_fit_qb_passing_td`) and never emitted by
    `player_usage.build_features_for_player`, so every QB was projected from a
    zero rolling TD rate. `fillna(0)` is still right for a genuinely absent
    *value* on a present column, which is what it was written for.

    `servable_columns` is a REQUIRED argument, and it is the reason this function
    is not the vacuous comparison it used to be. The first version of this guard
    had no such parameter and read `player_usage.SERVING_FEATURE_COLUMNS` from
    inside the body; the sole call site in `load_models` passed
    `player_usage.PLAYER_FEATURE_COLUMNS`, and that constant is defined as
    `[*PLAYER_FEATURE_COLUMNS, PASSING_TDS_ROLL_COLUMN]` -- so the guard evaluated
    `X ⊆ X + 1`, a tautology, and no player pickle's fitted columns were ever
    read. Forcing the caller to state the serving set makes that shape impossible
    to write again: the two sides now have to come from different places.

    The other thing this version does NOT claim. It is a subset test, so on its
    own it still cannot see an artefact that is missing a feature; that is what
    `_assert_artefact_columns_are_served` is for, and it is why both exist. Every
    artefact that has a stated expectation is audited through THAT function
    rather than through this one, so the two are not redundant -- this is the
    narrower rule, applied to a named column list.

    Two call sites, and both pass a list that is not the artefact's own: the
    fit-time check inside `_fit_qb_passing_td`, on the columns it is about to fit,
    and `load_models` on `player_usage.PLAYER_FEATURE_COLUMNS`. The QB passing-TD
    artefact's own recorded columns go through
    `_assert_artefact_columns_are_served` in `_load_passing_td_model` instead.
    """
    missing = [c for c in fitted_cols if c not in servable_columns]
    if missing:
        raise ValueError(
            f"{model_name} was fitted on columns the serving feature builder does not emit: "
            f"{missing}. {servable_source} emits {servable_columns}. A model fitted on a "
            "column serving cannot produce is scored on fillna(0) for every prediction."
        )


def _fitted_feature_columns(artefact) -> list[str] | None:
    """The columns `artefact` was fitted on, read out of `artefact` itself.

    Returns `None` when the artefact records nothing it can be read from. That is
    a FAILURE at every call site, not a pass -- see
    `_assert_artefact_columns_are_served`.

    Three shapes are read, and they are the three the committed payload contains:

    * an **estimator** fitted from a `DataFrame` records its own input columns as
      `feature_names_in_`. Every XGB artefact is this, and so is the ridge
      `game_outcome_model.pkl`, because `game_outcome.fit_margin_regression`
      hands its `X_train` DataFrame straight to `Ridge.fit`. It is read here
      rather than assumed, so a future artefact fitted from a NumPy array -- which
      records no names at all -- is caught rather than waved through.
    * a **mapping** records them under `feature_cols`. That is the QB passing-TD
      artefact's shape, and its `feature_cols` is also the list
      `qb_passing_td.expected_passing_tds` reindexes the live row by, so it is
      the record that actually governs serving.
    * a mapping with no `feature_cols` key at all falls through to its inner
      `model`, for the same reason.

    A `feature_cols` that is present but `None` is a FAILURE and does NOT fall
    through, which is why the presence test below is `"feature_cols" not in ...`
    rather than a truthiness check. Falling through would read the inner estimator
    instead and could pass a payload whose own record is unusable --
    `expected_passing_tds` does `fitted["feature_cols"]` unguarded, so a
    `feature_cols: None` payload serves fine through the audit and then raises
    per QB, per request, forever.
    """
    names = getattr(artefact, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(artefact, Mapping):
        # `in`, not a truthiness or `is not None` test. An explicit `None` is the
        # payload stating it has no column record, which is a failure; only an
        # absent key falls through to the inner estimator.
        if "feature_cols" in artefact:
            recorded = artefact["feature_cols"]
            return None if recorded is None else list(recorded)
        inner = artefact.get("model")
        if inner is not None:
            return _fitted_feature_columns(inner)
    return None


def _is_elo_candidate(artefact) -> bool:
    """Is this the Elo "model", which has no fitted feature matrix to audit?

    `game_outcome.fit_elo_candidate` returns a fixed conversion constant, and
    `game_outcome.predict_margin_elo` reads three named columns off the live
    feature row directly -- there is no design matrix and no fitted column list,
    so there is nothing for the audit to compare. This is the ONE artefact the
    audit does not apply to, and recognising it by shape rather than by the
    manifest's `chosen_candidate` is deliberate: a payload whose manifest claims
    `elo` while shipping a Ridge still fails, because a Ridge is not this shape.

    **This function is not reached on the committed payload**, which chose `ridge`
    -- `_assert_artefact_columns_are_served` only asks the question once the
    artefact has already failed to report any fitted columns, and on a Ridge that
    check is answered first. So its behaviour is covered by
    `tests/test_fitted_vs_served_columns.py::test_the_elo_skip_cannot_be_claimed_by_a_regressor`
    rather than by the load path, and that test calls it directly. Stated here
    because a reader tracing `load_models` will not find it, and should not
    conclude it is dead code.
    """
    return isinstance(artefact, Mapping) and "points_per_rating_point" in artefact


def _assert_artefact_columns_are_served(
    artefact,
    model_name: str,
    servable_columns: list[str],
    servable_source: str,
    expected_cols: list[str] | None = None,
    allow_unfitted: bool = False,
) -> list[str]:
    """Audit one loaded artefact against what serving emits. The real audit.

    This is the check #26 said it had and did not have. It reads the artefact's
    OWN recorded fitted columns -- `_fitted_feature_columns`, which opens the
    model's `feature_names_in_` -- and holds them against two separate things.

    **1. `expected_cols`, the columns serving will actually score this model on,
    compared for EXACT equality.** Both player and game serving reindex a live row
    by a fixed list before predicting: `player_props.predict_props` by
    `models["feature_cols"]`, `routes._predict_game_from_models` by
    `models["feature_cols"]`. So the requirement on the artefact is not "its
    columns are among the ones we hand it" but "its columns are precisely the ones
    we hand it" -- a model fitted on fewer gets a narrowed design matrix, in which
    every column past the gap is scored against the wrong number, and a model
    fitted on more gets a wider one.

    This leg is what a subset test structurally cannot do, and it is the case
    proven against `origin/main`: refit `anytime_td_model.pkl` on 3 of its 6
    columns, leave the manifest and the fingerprint untouched, and `load_models()`
    returns. `fitted ⊆ serving` is satisfied -- all 3 of those columns are emitted
    -- and the fingerprint is satisfied too, because it pins the manifest's
    DECLARED list against the CODE's list and never opens a pickle. Three-way
    agreement is what is needed and this is the missing third leg.

    `expected_cols` is the CODE's constant (`player_usage.PLAYER_FEATURE_COLUMNS`
    / `feature_build.FEATURE_COLUMNS`), not the manifest's copy, for the same
    reason `_verify_artifact_fingerprint` reads the code: it has already pinned
    the manifest's copy to that constant, so reading the manifest here would only
    re-prove it.

    **2. `servable_columns`, what the feature builder ACTUALLY emits**, as a
    subset. That is the `passing_tds_roll` direction: a fitted column the pregame
    builder does not emit is served as a constant zero.

    **Neither `None` nor an empty fitted list is ever a pass.** A model fitted from
    a NumPy array records no column names at all, and an unreadable artefact
    records none either; treating either as fine makes "I cannot tell what this was
    fitted on" indistinguishable from "it was fitted on the right things", which is
    the defect itself.

    **What this does NOT claim.** `expected_cols` is `None` for the QB passing-TD
    artefact, because its fitted list is the payload's own free choice --
    `fit_qb_passing_td_model` takes its columns from `X` as given -- and nothing
    outside that pickle states what it ought to be. So the QB artefact is audited
    for recording something and for every recorded column being emitted, and NOT
    for matching any particular list. A QB payload that is self-consistently
    fitted on fewer columns than the committed one is not detectable here; it
    fails later, inside scikit-learn, on the feature count. No guarantee is being
    stated about that case.

    `allow_unfitted` is set only for the Elo candidate, and only
    `_is_elo_candidate` can satisfy it.

    Returns the fitted columns so callers can report them.
    """
    # Read defensively. `list()` on a malformed record can raise before any of
    # this function's own guards run -- a non-iterable `feature_names_in_` is a
    # `TypeError`, not a `ValueError` -- and a load-time audit that dies with an
    # unhandled TypeError still refuses the payload, so the behaviour is safe but
    # the message is not actionable. So the shape is checked first and everything
    # malformed is reported in the same words as everything else, naming the
    # artefact and both column lists.
    try:
        fitted = _fitted_feature_columns(artefact)
    except TypeError as exc:
        fitted = None
        malformed = f"{exc}"
    else:
        malformed = None
    if malformed is not None:
        raise ValueError(
            f"{model_name} has a malformed fitted-column record ({malformed}), so it "
            f"cannot be shown to be servable. Serving emits {servable_columns} and "
            f"{model_name} does not say what it was fitted on. Re-run training "
            "(`python -m nfl_predictor.models.manifest`)."
        )

    if not fitted:
        if allow_unfitted and _is_elo_candidate(artefact):
            return []
        raise ValueError(
            f"{model_name} records no fitted feature columns "
            f"({fitted!r} read from the artefact itself), so it cannot be shown to be "
            f"servable. An artefact that does not say what it was fitted on -- "
            f"unfitted, or fitted from a NumPy array, which records no column names -- "
            f"is refused rather than passed: serving emits {servable_columns} and an "
            "unknown fitted list cannot be checked against it. Re-run training "
            "(`python -m nfl_predictor.models.manifest`)."
        )

    # A mapping payload's `feature_cols` is self-reported, so it is only trusted
    # when the estimator inside the same pickle agrees. Where both records exist
    # and differ, the payload was assembled from two different fits and neither
    # list can be taken at face value.
    #
    # The inner read is guarded too, and `None` from it is NOT treated as
    # agreement: an inner estimator with no readable record leaves the payload's
    # own claim uncorroborated, and this is the only place it could be corroborated.
    if isinstance(artefact, Mapping) and "feature_cols" in artefact:
        inner = artefact.get("model")
        inner_fitted = None
        if inner is not None:
            try:
                inner_fitted = _fitted_feature_columns(inner)
            except TypeError as exc:
                raise ValueError(
                    f"{model_name} has a malformed fitted-column record on the model "
                    f"inside it ({exc}), so it cannot be shown to be servable. Its "
                    f"'feature_cols' claims {list(fitted)}. Re-run training "
                    "(`python -m nfl_predictor.models.manifest`)."
                ) from exc
        if inner_fitted and list(inner_fitted) != list(fitted):
            raise ValueError(
                f"{model_name} disagrees with itself: the payload's 'feature_cols' is "
                f"{list(fitted)} but the model inside it was fitted on {list(inner_fitted)}. "
                "Serving reindexes by the payload's list, so the two disagreeing means the "
                "model would be scored on columns it was not fitted on. Re-run training "
                "(`python -m nfl_predictor.models.manifest`)."
            )

    if expected_cols is not None and list(fitted) != list(expected_cols):
        raise ValueError(
            f"{model_name} was fitted on {list(fitted)}, but serving scores it on "
            f"{list(expected_cols)}. Serving reindexes every live row by that list before "
            "predicting, so a pickle whose own fitted columns differ from it is scored on "
            "features it was never fitted on -- a constant zero for anything it is missing, "
            "and shifted for everything after. This is the check the artefact fingerprint "
            "cannot make: it pins the manifest's declared list against the code's and never "
            "opens a pickle. Re-run training "
            "(`python -m nfl_predictor.models.manifest`)."
        )

    _assert_servable_columns(list(fitted), model_name, servable_columns, servable_source)
    return list(fitted)


def _player_serving_columns() -> list[str]:
    """What `player_usage.build_features_for_player` actually emits.

    Read by CALLING the builder on a two-week probe frame, not by reading
    `player_usage.SERVING_FEATURE_COLUMNS`. That constant is a hand-maintained
    declaration, and trusting it is precisely how the old guard became `X ⊆ X`:
    the declared list and the list passed to the assertion were built from the
    same expression. The builder is the thing serving actually calls, so it is the
    thing the audit compares against.

    Fails closed. A builder that cannot be run means the audit cannot be
    performed, and an audit that cannot be performed is not a passing audit.
    """
    probe = pd.DataFrame(
        [
            {
                "player_id": "audit-probe", "season": 2000, "week": week,
                **{stat: 1.0 for stat in player_usage.ROLL_STATS},
                "passing_tds": 1.0,
            }
            for week in (1, 2)
        ]
    )
    try:
        row = player_usage.build_features_for_player(
            "audit-probe", probe, season=2000, week=3,
            window=player_usage.DEFAULT_ROLL_WINDOW,
        )
    except Exception as exc:  # pragma: no cover - fail-closed branch
        raise ValueError(
            "player_usage.build_features_for_player could not be run, so the "
            "fitted-vs-served audit cannot be performed for the player models. That "
            f"is refused rather than skipped: {exc!r}"
        ) from exc
    return list(row.index)


def _game_serving_columns() -> list[str]:
    """What `feature_build.build_features_for_game` actually emits.

    The game half of the same audit, on the same fail-closed terms. This is the
    row `routes._predict_game_from_models` reindexes by `manifest["feature_cols"]`
    and `fillna(0)`s, so a game feature the builder stops emitting -- or that the
    pickles were never fitted on -- reaches the model as a constant zero. Nothing
    audited the game pickles at all before this.
    """
    games = pd.DataFrame(
        [
            {
                "game_id": f"{home}_{away}_{week}", "season": 2000, "week": week,
                "gameday": pd.Timestamp("2000-09-01") + pd.Timedelta(days=7 * (week - 1)),
                "home_team": home, "away_team": away, "home_score": 21, "away_score": 14,
            }
            for week in (1, 2)
            for home, away in (("A", "B"), ("A", "C"), ("B", "C"))
        ]
    )
    try:
        row = feature_build.build_features_for_game("A", "C", games)
    except Exception as exc:  # pragma: no cover - fail-closed branch
        raise ValueError(
            "feature_build.build_features_for_game could not be run, so the "
            "fitted-vs-served audit cannot be performed for the game models. That "
            f"is refused rather than skipped: {exc!r}"
        ) from exc
    return list(row.index)


def _load_passing_td_model(manifest: dict, player_serving: list[str]) -> dict | None:
    """The passing-TD artifact, or None when the manifest says there is none.

    **Gated on the manifest, not on the file being present**, which is the change
    from probing the filesystem. The rule now:

    * manifest records a fitted model -> the artifact is REQUIRED. A manifest
      that says it trained one and a directory without the file is a corrupt or
      half-copied deployment, and serving a payload with the market silently
      missing turns that into "this week the app has no QB passing-TD picks",
      which is indistinguishable from the model declining to project anyone.
    * manifest records `null` (no QB history at the time of training) or has no
      `qb_passing_td` key at all (a manifest written before this model existed)
      -> the market is genuinely absent and `predict_props` omits it, exactly as
      it omits a yardage market with no model.

    The old version asked only "does the file exist", so a manifest that promised
    a model and a directory that had lost one read as "no model this time" --
    which is how the feature stayed dormant after a retrain without anyone being
    told.

    Whichever branch is taken, the artefact that comes back has been through
    `_assert_artefact_columns_are_served`. The previous version called
    `_assert_servable_columns(list(fitted.get("feature_cols") or []), ...)`, which
    is the same vacuous shape the player-side guard had: `or []` turns a payload
    that records nothing into an empty fitted list, and an empty list is a subset
    of anything, so the one artefact that DID carry its own fitted column list --
    and therefore the one artefact that could genuinely be checked -- was passed
    whenever that list was unreadable. It fails closed now.
    """
    declared = manifest.get("qb_passing_td")
    if declared is None:
        return None
    path = _artifact_path(PASSING_TD_MODEL_FILENAME)
    try:
        exists = path.exists()
    except AttributeError:
        # `_artifact_path` is stubbed to a bare object in a few tests; the load
        # below is stubbed with it, so treat the path as present. The audit still
        # runs on whatever comes back, because skipping it here is how this
        # function's own check used to be reached without being performed.
        fitted = _load_pickle(path)
    else:
        if not exists:
            raise FileNotFoundError(
                f"{MANIFEST_PATH} records a fitted {qb_passing_td.PASSING_TD_MARKET} model "
                f"(distribution={declared.get('distribution')!r}, n_train={declared.get('n_train')!r}) "
                f"but {path} is missing. Re-run training to restore the artifact; serving without "
                "it would silently drop the market from every QB prop row."
            )
        fitted = _load_pickle(path)
    _assert_artefact_columns_are_served(
        fitted, PASSING_TD_MODEL_FILENAME, player_serving,
        "player_usage.build_features_for_player")
    return fitted


def load_models() -> dict:
    """Load all saved artifacts and their feature metadata.

    Two independent guards run before a single prediction is scored, and they are
    not substitutes for one another:

    * `_verify_artifact_fingerprint` (above) pins the manifest's DECLARED feature
      lists against the CODE's. It catches the code having moved on. It never
      opens a pickle, so it cannot see an artefact that disagrees with its own
      manifest -- a pickle fitted on 3 of 6 columns sits there agreeing with both
      sides.
    * `_assert_artefact_columns_are_served` (below) opens every artefact the
      payload ships and holds its OWN recorded fitted columns against the columns
      the serving builder actually emits. It is the only check that reads
      `feature_names_in_`, and it is the only one that can see the artefact.

    The old call here was `_assert_servable_columns(player_usage.PLAYER_FEATURE_COLUMNS, ...)`
    against a body that read `player_usage.SERVING_FEATURE_COLUMNS`, a constant
    defined as `[*PLAYER_FEATURE_COLUMNS, PASSING_TDS_ROLL_COLUMN]` -- so it
    compared a list with itself plus one, on the player side only, and inspected
    no model at all. `tests/test_fitted_vs_served_columns.py` is the generic test
    for the guard that replaced it.
    """
    manifest = load_manifest()
    # Before anything is unpickled: a stale artefact is cheaper to refuse here
    # than to serve. See `_verify_artifact_fingerprint`.
    _verify_artifact_fingerprint(manifest)

    # What serving actually emits, read by CALLING each builder. See
    # `_player_serving_columns` / `_game_serving_columns` for why this is not the
    # declared constants.
    player_serving = _player_serving_columns()
    game_serving = _game_serving_columns()

    # The code-side half of the audit, and the one thing `_verify_artifact_fingerprint`
    # does not cover. That check pins the manifest to `PLAYER_FEATURE_COLUMNS`, so
    # re-asserting the manifest's copy would only re-prove it. Asserting the
    # constant against the *builder* states what is still open: every column the
    # player models are trained on has to be one `build_features_for_player` emits.
    # The QB passing-TD model is not covered here -- it is fitted on
    # `passing_tds_roll`, which is deliberately NOT in `PLAYER_FEATURE_COLUMNS` --
    # and is audited in `_load_passing_td_model` against its own recorded columns.
    _assert_servable_columns(
        player_usage.PLAYER_FEATURE_COLUMNS,
        "the player models' training columns (player_usage.PLAYER_FEATURE_COLUMNS)",
        player_serving, "player_usage.build_features_for_player")

    player_models = {"feature_cols": manifest["player_feature_cols"]}

    # Every player artefact is fitted on `player_usage.PLAYER_FEATURE_COLUMNS` and
    # is scored by `predict_props` on `manifest["player_feature_cols"]`, which
    # `_verify_artifact_fingerprint` has already pinned to that constant. So the
    # artefact's own `feature_names_in_` has to equal it -- and this is the check
    # that fails when it does not.
    expected_player = list(player_usage.PLAYER_FEATURE_COLUMNS)

    anytime_td = _load_pickle(_artifact_path(ANYTIME_TD_MODEL_FILENAME))
    _assert_artefact_columns_are_served(
        anytime_td, ANYTIME_TD_MODEL_FILENAME, player_serving,
        "player_usage.build_features_for_player", expected_cols=expected_player)
    player_models["anytime_td"] = anytime_td

    for market in manifest["yardage_metrics"]:
        filename = _yardage_model_path(market).name
        model = _load_pickle(_yardage_model_path(market))
        _assert_artefact_columns_are_served(
            model, filename, player_serving, "player_usage.build_features_for_player",
            expected_cols=expected_player)
        player_models[market] = model

    # Optional exactly as the comment below the model says: absent from the
    # manifest means not served, present in the manifest means required.
    passing_td = _load_passing_td_model(manifest, player_serving)
    if passing_td is not None:
        player_models[qb_passing_td.PASSING_TD_MARKET] = passing_td

    # The game half. `routes._predict_game_from_models` reindexes the live row by
    # `manifest["feature_cols"]` and `fillna(0)`s, so a removed game feature is
    # served as a constant zero -- the identical mechanism `passing_tds_roll` was,
    # on the other side of the payload, and the reason the game-level fingerprint
    # in #28 was needed. The fingerprint pins the declared list; this pins the
    # pickle against the builder.
    #
    # Both game artefacts are fitted on `feature_build.FEATURE_COLUMNS`, which
    # `build_training_frame` hands to `train_all`, and both are scored on
    # `manifest["feature_cols"]`, which the fingerprint has pinned to it. Same
    # third leg as the player side.
    expected_game = list(feature_build.FEATURE_COLUMNS)

    game_outcome_model = _load_pickle(_artifact_path(GAME_MODEL_FILENAME))
    total_model = _load_pickle(_artifact_path(TOTAL_MODEL_FILENAME))

    # The manifest and the game artefact have to agree about WHICH candidate was
    # chosen, in addition to agreeing about its columns. `routes` branches on
    # `models["chosen_candidate"]` and, on the `elo` branch, ignores the artefact
    # entirely -- it builds the conversion constant inline. So a payload whose
    # manifest says `elo` while shipping a Ridge serves a prediction that came
    # from neither: the manifest's claim, not the artefact on disk. `sigma` was
    # fitted against whichever candidate actually ran, so the two disagree about
    # the residual spread too.
    #
    # This is not the same failure as an `elo` artefact under a `ridge` manifest,
    # which the column audit below already catches (a conversion constant records
    # no fitted columns). It is the reverse direction, which the column audit
    # cannot see: a fitted model satisfies every column rule and still is not the
    # candidate being served.
    claimed_elo = manifest.get("chosen_candidate") == "elo"
    artefact_is_elo = _is_elo_candidate(game_outcome_model)
    if claimed_elo != artefact_is_elo:
        raise ValueError(
            f"{MANIFEST_PATH} records chosen_candidate="
            f"{manifest.get('chosen_candidate')!r} but {GAME_MODEL_FILENAME} is a "
            f"{'`fit_elo_candidate` conversion constant' if artefact_is_elo else 'fitted regressor'}. "
            "`routes._predict_game_from_models` picks its scoring path from the manifest's "
            "claim and, on the `elo` path, never reads this artefact -- so the two "
            "disagreeing means every game prediction comes from one of them while the "
            "payload reports the other, and `sigma` was fitted against whichever ran. "
            "Re-run training (`python -m nfl_predictor.models.manifest`)."
        )

    _assert_artefact_columns_are_served(
        game_outcome_model, GAME_MODEL_FILENAME, game_serving,
        "feature_build.build_features_for_game",
        expected_cols=expected_game,
        # The Elo candidate is a conversion constant, not a fitted design matrix,
        # so there is no fitted column list to audit. The agreement check above has
        # already established that the artefact really is the Elo one and that the
        # manifest really claims it, so this exemption cannot be claimed by
        # anything else.
        allow_unfitted=claimed_elo,
    )
    _assert_artefact_columns_are_served(
        total_model, TOTAL_MODEL_FILENAME, game_serving,
        "feature_build.build_features_for_game", expected_cols=expected_game)

    return {
        "game_outcome_model": game_outcome_model,
        "total_model": total_model,
        "chosen_candidate": manifest["chosen_candidate"],
        "model_version": model_version(manifest),
        "sigma": manifest["sigma"],
        "total_sigma": manifest["total_sigma"],
        "player_models": player_models,
        "feature_cols": manifest["feature_cols"],
        "player_feature_cols": manifest["player_feature_cols"],
    }


if __name__ == "__main__":
    train_all()

"""Train, save, and load the game-outcome and player-prop models."""

from __future__ import annotations

import json
import pickle
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
    # should not be written to disk in the first place.
    _assert_servable_columns(cols, "the QB passing-TD model")
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
    # discipline it is trained on here, and `_assert_servable_columns` below
    # refuses to save a model whose fitted columns the serving builder cannot
    # produce. This comment previously said the opposite -- that the column was
    # "deliberately absent" from the model because serving did not emit it -- and
    # the code beside it fitted on it anyway, so every QB was served a zero for a
    # column the model had a fitted coefficient for.
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
      passed. A subset test cannot see an artefact that is missing a feature.
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
    is silently never used, and `_assert_servable_columns` cannot see it either
    because it only knows about the PLAYER side
    (`player_usage.SERVING_FEATURE_COLUMNS`) and its verdict would be about the
    wrong models. So the check below is the exact same comparison against the
    exact same kind of authority -- `feature_build.FEATURE_COLUMNS`, the constant
    `build_training_frame` returns to `train_all` -- by the same exact-list-not-
    subset rule, for the same reason a subset cannot detect a missing feature.
    The player-side checks above are untouched by this.
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


def _assert_servable_columns(fitted_cols: list[str], model_name: str) -> None:
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

    Checked for every player model in the payload here -- `anytime_td`, each
    yardage market (all fitted on `player_feature_cols`) and the passing-TD
    model (fitted on its own `feature_cols`) -- so the whole class fails at load
    time rather than one model at a time.
    """
    missing = [c for c in fitted_cols if c not in player_usage.SERVING_FEATURE_COLUMNS]
    if missing:
        raise ValueError(
            f"{model_name} was fitted on columns the serving feature builder does not emit: "
            f"{missing}. `player_usage.build_features_for_player` emits "
            f"{player_usage.SERVING_FEATURE_COLUMNS}. A model fitted on a column serving "
            "cannot produce is scored on fillna(0) for every prediction."
        )


def _load_passing_td_model(manifest: dict) -> dict | None:
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
    """
    declared = manifest.get("qb_passing_td")
    path = _artifact_path(PASSING_TD_MODEL_FILENAME)
    if declared is None:
        return None
    try:
        exists = path.exists()
    except AttributeError:
        # `_artifact_path` is stubbed to a bare object in a few tests; the load
        # below is stubbed with it, so treat the path as present.
        return _load_pickle(path)
    if not exists:
        raise FileNotFoundError(
            f"{MANIFEST_PATH} records a fitted {qb_passing_td.PASSING_TD_MARKET} model "
            f"(distribution={declared.get('distribution')!r}, n_train={declared.get('n_train')!r}) "
            f"but {path} is missing. Re-run training to restore the artifact; serving without "
            "it would silently drop the market from every QB prop row."
        )
    fitted = _load_pickle(path)
    _assert_servable_columns(list(fitted.get("feature_cols") or []), qb_passing_td.PASSING_TD_MARKET)
    return fitted


def load_models() -> dict:
    """Load all saved artifacts and their feature metadata."""
    manifest = load_manifest()
    # Before anything is unpickled: a stale artefact is cheaper to refuse here
    # than to serve. See `_verify_artifact_fingerprint`.
    _verify_artifact_fingerprint(manifest)
    # Against the CODE's list, not the manifest's. `_verify_artifact_fingerprint`
    # has just pinned `manifest["player_feature_cols"]` to
    # `player_usage.PLAYER_FEATURE_COLUMNS`, so re-asserting the manifest's copy
    # would only re-prove that. Asserting the constant states the thing that is
    # still open: every column the player models are trained on has to be one
    # `build_features_for_player` emits. The QB passing-TD model cannot be
    # covered from here and is asserted in `_load_passing_td_model` against its
    # own `feature_cols` -- it is fitted on `passing_tds_roll`, which is
    # deliberately NOT in `PLAYER_FEATURE_COLUMNS`, and that is precisely how it
    # shipped fitted-but-unserved.
    _assert_servable_columns(player_usage.PLAYER_FEATURE_COLUMNS, "the player models (anytime_td, yardage)")
    player_models = {
        "feature_cols": manifest["player_feature_cols"],
        "anytime_td": _load_pickle(_artifact_path(ANYTIME_TD_MODEL_FILENAME)),
    }
    for market in manifest["yardage_metrics"]:
        player_models[market] = _load_pickle(_yardage_model_path(market))

    # Optional exactly as the comment below the model says: absent from the
    # manifest means not served, present in the manifest means required.
    passing_td = _load_passing_td_model(manifest)
    if passing_td is not None:
        player_models[qb_passing_td.PASSING_TD_MARKET] = passing_td

    return {
        "game_outcome_model": _load_pickle(_artifact_path(GAME_MODEL_FILENAME)),
        "total_model": _load_pickle(_artifact_path(TOTAL_MODEL_FILENAME)),
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

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

    The rolling TD feature is built here rather than in `player_usage` so the
    existing `PLAYER_FEATURE_COLUMNS` -- which the anytime-TD classifier and
    every yardage regressor are fitted on, and which `predict_props` indexes by
    name -- stays exactly as it was. Widening that list would silently change
    the feature count of every model already committed.
    """
    qb = player_train_df[player_train_df["position"] == "QB"].copy()
    if qb.empty:
        return None
    qb = qb.sort_values(["player_id", "season", "week"])
    grouped = qb.groupby("player_id")["passing_tds"]
    # Same shift(1)-then-rolling discipline as `_add_rolling`, for the same
    # reason: a pregame feature cannot know the game being predicted.
    qb["passing_tds_roll"] = grouped.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    cols = ["passing_tds_roll", *qb_passing_td.MU_FEATURE_COLUMNS]
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

    # QB passing-TD projection. QBs only, and only the rolling features a
    # pregame feature row actually carries -- `passing_tds_roll` is deliberately
    # absent because `player_usage.build_features_for_player` does not emit it,
    # so training on it would mean serving on a feature that does not exist.
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
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


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


def load_models() -> dict:
    """Load all saved artifacts and their feature metadata."""
    manifest = load_manifest()
    player_models = {
        "feature_cols": manifest["player_feature_cols"],
        "anytime_td": _load_pickle(_artifact_path(ANYTIME_TD_MODEL_FILENAME)),
    }
    for market in manifest["yardage_metrics"]:
        player_models[market] = _load_pickle(_yardage_model_path(market))

    # The passing-TD model is optional in the same way a yardage market is: an
    # older artifact directory has no such file, and serving must not KeyError
    # on it. `predict_props` omits the market when the model is absent.
    passing_td_path = _artifact_path(PASSING_TD_MODEL_FILENAME)
    try:
        present = passing_td_path.exists()
    except AttributeError:
        # `_artifact_path` is stubbed to a bare object in several tests, and a
        # path that cannot answer is treated as absent rather than fatal. The
        # market is then simply not served, which is the documented behaviour.
        present = False
    if present:
        player_models[qb_passing_td.PASSING_TD_MARKET] = _load_pickle(passing_td_path)

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

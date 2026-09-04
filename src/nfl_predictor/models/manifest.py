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
from . import game_outcome, player_props

MANIFEST_PATH = MODELS_DIR / "manifest.json"
GAME_MODEL_PATH = MODELS_DIR / "game_outcome_model.pkl"
TOTAL_MODEL_PATH = MODELS_DIR / "total_points_model.pkl"
ANYTIME_TD_MODEL_PATH = MODELS_DIR / "anytime_td_model.pkl"
YARDAGE_MODEL_PATHS = {
    market: MODELS_DIR / f"{market}_model.pkl" for market in player_props.YARDAGE_TARGETS
}

DEFAULT_TRAIN_SEASONS = 8


def _save_pickle(obj, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_all(seasons: list[int] | None = None) -> dict:
    """Fit all models, persist their artifacts, and return their manifest."""
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    seasons = seasons or schedules.default_completed_seasons(n=DEFAULT_TRAIN_SEASONS)

    games_df = schedules.load_training_data(seasons)
    train_df, feature_cols = feature_build.build_training_frame(games_df)

    folds = walk_forward.prepare_folds(games_df, min_train_seasons=max(1, len(seasons) - 2))
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

    _save_pickle(game_model, GAME_MODEL_PATH)
    _save_pickle(total_model, TOTAL_MODEL_PATH)

    player_df_raw = player_stats.fetch_weekly_player_stats(seasons)
    player_train_df, player_feature_cols = player_usage.build_player_training_frame(player_df_raw)
    player_train_df = player_train_df.dropna(subset=player_feature_cols, how="all")
    X_player = player_train_df[player_feature_cols].fillna(0)

    anytime_td_model = player_props.fit_anytime_td_classifier(X_player, player_train_df["anytime_td"])
    _save_pickle(anytime_td_model, ANYTIME_TD_MODEL_PATH)

    yardage_metrics = {}
    for market, target_col in player_props.YARDAGE_TARGETS.items():
        subset = player_train_df[player_train_df[target_col] > 0]
        if subset.empty:
            continue
        model = player_props.fit_yardage_regressor(subset[player_feature_cols].fillna(0), subset[target_col])
        _save_pickle(model, YARDAGE_MODEL_PATHS[market])
        yardage_metrics[market] = {"n_train": int(len(subset))}

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
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest() -> dict:
    """Load the metadata produced by :func:`train_all`."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No trained models found. Run `python -m nfl_predictor.models.manifest` first.")
    return json.loads(MANIFEST_PATH.read_text())


def load_models() -> dict:
    """Load all saved artifacts and their feature metadata."""
    manifest = load_manifest()
    player_models = {
        "feature_cols": manifest["player_feature_cols"],
        "anytime_td": _load_pickle(ANYTIME_TD_MODEL_PATH),
    }
    for market, path in YARDAGE_MODEL_PATHS.items():
        if path.exists():
            player_models[market] = _load_pickle(path)

    return {
        "game_outcome_model": _load_pickle(GAME_MODEL_PATH),
        "total_model": _load_pickle(TOTAL_MODEL_PATH),
        "chosen_candidate": manifest["chosen_candidate"],
        "sigma": manifest["sigma"],
        "total_sigma": manifest["total_sigma"],
        "player_models": player_models,
        "feature_cols": manifest["feature_cols"],
        "player_feature_cols": manifest["player_feature_cols"],
    }


if __name__ == "__main__":
    train_all()

"""walk_forward.py — season-by-season walk-forward validation for the three
models/game_outcome.py candidates. Mirrors PL_Predictor's/F1_Predictor's own
evaluate/walk_forward.py: builds the full feature frame ONCE (no lookahead —
every feature is already shift(1)/expanding computed before any slicing),
then slices by season so evaluate_candidate can be called repeatedly without
redoing feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..features.build import build_training_frame
from ..models import game_outcome


def prepare_folds(games_df: pd.DataFrame, min_train_seasons: int = 4) -> list[dict]:
    df, feature_cols = build_training_frame(games_df)
    seasons = sorted(df["season"].unique())

    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        folds.append({"val_season": val_season, "train_df": train_df, "val_df": val_df, "feature_cols": feature_cols})
    return folds


def _predict_margins(candidate: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    X_train, y_train = train_df[feature_cols], train_df["margin"]
    X_val = val_df[feature_cols]

    if candidate == "elo":
        model = game_outcome.fit_elo_candidate(train_df)
        preds = np.array(
            [
                game_outcome.predict_margin_elo(model, r, hr, ar)
                for r, hr, ar in zip(val_df["rating_diff"], val_df["home_rest_days"], val_df["away_rest_days"])
            ]
        )
        sigma = game_outcome.residual_sigma(
            type("_", (), {"predict": lambda self, X: np.array(
                [game_outcome.predict_margin_elo(model, r, hr, ar) for r, hr, ar in
                 zip(train_df["rating_diff"], train_df["home_rest_days"], train_df["away_rest_days"])]
            )})(),
            X_train, y_train,
        )
        return preds, sigma

    fit_fn = game_outcome.fit_margin_regression if candidate == "ridge" else game_outcome.fit_xgb_margin
    model = fit_fn(X_train, y_train)
    preds = model.predict(X_val.fillna(0))
    sigma = game_outcome.residual_sigma(model, X_train, y_train)
    return preds, sigma


def evaluate_candidate(folds: list[dict], candidate: str) -> pd.DataFrame:
    rows = []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        preds, sigma = _predict_margins(candidate, train_df, val_df, feature_cols)

        probs = np.array(
            [game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds]
        )
        actual = (val_df["margin"] > 0).astype(int).to_numpy()
        # Clip away from exact 0/1 so log_loss never receives a probability
        # that would make it -inf on a single miss.
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        rows.append(
            {
                "val_season": fold["val_season"],
                "n_games": len(val_df),
                "log_loss": log_loss(actual, probs, labels=[0, 1]),
                "brier": brier_score_loss(actual, probs),
            }
        )
    return pd.DataFrame(rows)

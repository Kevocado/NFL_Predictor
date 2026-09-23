"""player_props.py — anytime-TD classifier and per-position yardage
regressors. Mirrors PL_Predictor's player goal/assist classifier approach
for anytime_td; yardage props are a genuinely new shape (continuous
regression, not PL_Predictor has an equivalent for) since NFL props are
commonly priced as an over/under yardage line rather than a probability."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

YARDAGE_TARGETS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "carries": "carries",
    "receptions": "receptions",
}

# Which markets apply to which position — a QB doesn't have a meaningful
# rushing-yards prop line in practice, a WR/TE doesn't have a passing one,
# etc. Each position can now have multiple markets (e.g. RB gets both
# rushing_yards and carries).
POSITION_MARKETS: dict[str, list[str]] = {
    "QB": ["passing_yards"],
    "RB": ["rushing_yards", "carries"],
    "WR": ["receiving_yards", "receptions"],
    "TE": ["receiving_yards", "receptions"],
}


def fit_anytime_td_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train.fillna(0), y_train)
    return model


def fit_yardage_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train.fillna(0), y_train)
    return model


def predict_props(models: dict, feature_row: pd.Series, position: str) -> dict:
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0).to_numpy().reshape(1, -1)

    result = {"anytime_td_prob": float(models["anytime_td"].predict_proba(X)[0, 1])}

    for market in POSITION_MARKETS.get(position, []):
        if market in models:
            result[market] = float(models[market].predict(X)[0])

    return result

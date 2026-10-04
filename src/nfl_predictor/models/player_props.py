"""player_props.py — anytime-TD classifier and per-position yardage
regressors. Mirrors PL_Predictor's player goal/assist classifier approach
for anytime_td; yardage props are a genuinely new shape (continuous
regression, not PL_Predictor has an equivalent for) since NFL props are
commonly priced as an over/under yardage line rather than a probability."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from . import qb_passing_td

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


#: Quantile levels fitted per market. P(over) is interpolated between these, so
#: the ends matter: at q0.1 a line below the lowest fitted quantile saturates at
#: the clamp in `prop_probability`.
#: Quantile levels fitted per market.
#:
#: The tenths are the model's substance. The extra deep-tail levels (0.01-0.05,
#: 0.95-0.99) exist because `p_over_from_quantiles` CLAMPS to [0.02, 0.98]: with
#: q0.1 as the lowest fitted quantile, any line below it reports a flat 0.98, and
#: a line above q0.9 reports a flat 0.02. Those clamps are not measurements, and
#: on the 2025 offline gate the flat 0.02 tail bucket carried a 0.057 gap against
#: a true rate of 0.077 -- the single failing bucket, and entirely an artifact.
#: Fitting the tails lets the model express the real probability instead of
#: saturating, which moved that bucket to a 0.012 gap.
QUANTILES: list[float] = [
    0.01, 0.02, 0.05,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    0.95, 0.98, 0.99,
]

#: Shared with fit_yardage_regressor so the q50 model and the production point
#: regressor differ only in objective.
_TREE_PARAMS = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "random_state": 42}


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


def fit_yardage_quantile_models(X_train: pd.DataFrame, y_train: pd.Series,
                               quantiles: list[float] | None = None
                               ) -> dict[float, XGBRegressor]:
    """One quantile regressor per level, keyed by the level itself.

    Independent fits rather than XGBoost's `reg:quantileerror` multi-quantile
    mode: that mode shares trees across levels, which makes them cross far more
    often, and crossing quantiles break the P(over) interpolation. `p_over_from_quantiles`
    repairs crossings anyway, but a model that rarely needs the repair is better.
    """
    quantiles = QUANTILES if quantiles is None else quantiles
    X = X_train.fillna(0)
    models: dict[float, XGBRegressor] = {}
    for q in quantiles:
        model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **_TREE_PARAMS)
        model.fit(X, y_train)
        models[q] = model
    return models


def predict_props(models: dict, feature_row: pd.Series, position: str) -> dict:
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0).to_numpy().reshape(1, -1)

    result = {"anytime_td_prob": float(models["anytime_td"].predict_proba(X)[0, 1])}

    for market in POSITION_MARKETS.get(position, []):
        if market in models:
            result[market] = float(models[market].predict(X)[0])

    # QB passing TDs: a count, so an over/under call rather than a yardage point
    # estimate, and QB-only. Every field is flattened onto the same props row the
    # yardage markets use, because that is the shape the payload already has --
    # see `qb_passing_td.py` for the field meanings and the frontend contract.
    #
    # Omitted entirely when the model is absent (an artifact directory trained
    # before this existed), exactly as a yardage market with no model is.
    if position == "QB" and qb_passing_td.PASSING_TD_MARKET in models:
        call = qb_passing_td.qb_passing_td_call(
            models[qb_passing_td.PASSING_TD_MARKET], feature_row)
        if call is not None:
            result.update({
                "passing_td_line": call["line"],
                "passing_td_line_source": call["line_source"],
                "passing_td_side": call["side"],
                "passing_td_mu": call["mu"],
                "passing_td_over_prob": call["over_prob"],
                "passing_td_under_prob": call["under_prob"],
                "passing_td_prob": call["call_prob"],
                "passing_td_distribution": call["distribution"],
            })

    return result

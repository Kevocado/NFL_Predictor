"""player_props.py — anytime-TD classifier and per-position yardage
regressors. Mirrors PL_Predictor's player goal/assist classifier approach
for anytime_td; yardage props are a genuinely new shape (continuous
regression, not PL_Predictor has an equivalent for) since NFL props are
commonly priced as an over/under yardage line rather than a probability."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from . import qb_passing_td, receptions_zinb

YARDAGE_TARGETS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "carries": "carries",
    "receptions": "receptions",
    # Market -> label column for the QB rushing market. The legacy
    # point-regressor training loop in manifest.py skips this market (see
    # QUANTILE_ONLY_MARKETS): it is fitted and served from the quantile
    # artifact, with q50 as the point prediction.
    "qb_rushing_yards": "rushing_yards",
}

#: Markets with no legacy point-regressor pickle: fitted and served from the
#: quantile artifact only. A pooled point fit for `qb_rushing_yards` would
#: serve RB-scale rushing predictions to QBs (RB median 26 yds vs QB median
#: 7); the serving path uses the quantile q50 as the point prediction instead
#: (see `predict_props`).
QUANTILE_ONLY_MARKETS = frozenset({"qb_rushing_yards"})

# Which markets apply to which position — a WR/TE doesn't have a passing
# market, etc. Each position can now have multiple markets (e.g. RB gets both
# rushing_yards and carries). QB rushing is its own market, not pooled with
# RBs: 29% of QB games are <= 0 rushing yards (kneel-downs and sacks count as
# rushes) vs 11% for RBs, so one model miscalibrates the QB shape
# (see training.MARKET_LABEL_COLUMN).
POSITION_MARKETS: dict[str, list[str]] = {
    "QB": ["passing_yards", "qb_rushing_yards"],
    # `receptions` (WR/TE) is a ZINB count-model market (see `receptions_zinb`), not a point regressor: the loop
    # below skips it and the dedicated block attaches the over/under call. RB is deliberately absent: RB
    # receptions failed the calibration gate and are not priced.
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
QUANTILES: list[float] = (
    [round(q, 2) for q in (0.01, 0.02, 0.03, 0.04, 0.05)]
    + [round(q, 1) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    + [round(q, 2) for q in (0.95, 0.96, 0.97, 0.98, 0.99)]
)

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
        # `receptions` is a ZINB count model, not an XGB regressor: it has no
        # `.predict` and is handled by the dedicated block below.
        if market == receptions_zinb.RECEPTIONS_MARKET:
            continue
        if market in models:
            result[market] = float(models[market].predict(X)[0])

    # Quantile predictions for P(over) directionality. The full 19-quantile
    # dict is carried on the row so the API layer can compute a line and
    # P(over) against it; the raw dicts are stripped before the rows leave
    # `_get_player_props_live` (they are an internal handoff, not an API
    # field). Omitted entirely when the quantile artifacts are absent (an
    # artifact directory trained before the quantile retrain).
    #
    # Markets with no point-regressor pickle (QUANTILE_ONLY_MARKETS) take
    # their point prediction from the q50 median: the median IS the point
    # forecast, and a pooled point model would serve the wrong distribution.
    for market in POSITION_MARKETS.get(position, []):
        qkey = f"{market}_quantiles"
        qcolkey = f"{market}_quantile_cols"
        if qkey not in models:
            continue
        try:
            qcols = models[qcolkey]
            Xq = feature_row.reindex(qcols).fillna(0).to_numpy().reshape(1, -1)
            qdict = {q: float(m.predict(Xq)[0]) for q, m in models[qkey].items()}
            result[qkey] = qdict
            if market not in result and 0.5 in qdict:
                result[market] = qdict[0.5]
        except Exception:
            # A corrupt or mismatched quantile artifact must not break the
            # point prediction that the row already carries.
            continue

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

    # Receptions: a discrete count, so a ZINB over/under call rather than a
    # yardage point estimate, for WR/TE. Every field is flattened onto the
    # same props row the other markets use. Field naming follows the Phase 1
    # `{market}_p_over` / `{market}_line` API contract.
    #
    # Omitted entirely when the model is absent (an artifact directory trained
    # before this existed), exactly as a yardage market with no model is.
    if (position in receptions_zinb.RECEPTIONS_POSITIONS
            and receptions_zinb.RECEPTIONS_MARKET in models):
        call = receptions_zinb.receptions_call(
            models[receptions_zinb.RECEPTIONS_MARKET], feature_row, position)
        if call is not None:
            result.update({
                "receptions_line": call["line"],
                "receptions_line_source": call["line_source"],
                "receptions_side": call["side"],
                "receptions_mu": call["mu"],
                "receptions_p_over": call["over_prob"],
                "receptions_p_under": call["under_prob"],
                "receptions_prob": call["call_prob"],
                "receptions_distribution": call["distribution"],
                # Serving guardrail: "high_mu" means do not auto-price lines
                # above 7.5 on this row (see receptions_zinb module docstring).
                "receptions_mu_flag": call["mu_flag"],
            })

    return result

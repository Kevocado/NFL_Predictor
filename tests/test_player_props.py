# tests/test_player_props.py
import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models import player_props


def _toy_player_frame(n=80, seed=1):
    rng = np.random.default_rng(seed)
    passing_roll = rng.normal(250, 40, n)
    rushing_roll = rng.normal(80, 20, n)
    receiving_roll = rng.normal(50, 20, n)
    return pd.DataFrame(
        {
            "passing_yards_roll": passing_roll,
            "rushing_yards_roll": rushing_roll,
            "receiving_yards_roll": receiving_roll,
            "targets_roll": rng.normal(5, 2, n),
            "carries_roll": rng.normal(15, 5, n),
            "passing_yards": passing_roll + rng.normal(0, 15, n),
            "rushing_yards": rushing_roll + rng.normal(0, 15, n),
            "receiving_yards": receiving_roll + rng.normal(0, 15, n),
            "anytime_td": (rng.random(n) < (0.3 + rushing_roll / 500)).astype(int),
        }
    )


FEATURE_COLS = ["passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll", "targets_roll", "carries_roll"]


def test_fit_anytime_td_classifier_predicts_probabilities():
    df = _toy_player_frame()
    model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])

    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()


def test_fit_yardage_regressor_predicts_reasonable_values():
    df = _toy_player_frame()
    model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])

    preds = model.predict(df[FEATURE_COLS])
    assert np.corrcoef(preds, df["rushing_yards"])[0, 1] > 0.3


def test_predict_props_only_returns_relevant_yardage_market_for_position():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    rushing_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])
    models = {
        "anytime_td": td_model,
        "rushing_yards": rushing_model,
        "feature_cols": FEATURE_COLS,
    }

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="RB")

    assert "anytime_td_prob" in result
    assert "rushing_yards" in result
    assert "passing_yards" not in result
    assert "receiving_yards" not in result


def test_predict_props_returns_all_markets_for_a_position_when_models_exist():
    # RB now maps to two markets (rushing_yards, carries) -- both should
    # come back when both models are present in the loaded models dict.
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    rushing_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])
    carries_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["carries_roll"])
    models = {
        "anytime_td": td_model,
        "rushing_yards": rushing_model,
        "carries": carries_model,
        "feature_cols": FEATURE_COLS,
    }

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="RB")

    assert "rushing_yards" in result
    assert "carries" in result


def test_predict_props_gracefully_skips_a_market_not_yet_in_models():
    # A position can list a market (e.g. WR -> receptions) whose model
    # hasn't been trained yet -- predict_props must not KeyError, just omit it.
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    receiving_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["receiving_yards"])
    models = {
        "anytime_td": td_model,
        "receiving_yards": receiving_model,
        "feature_cols": FEATURE_COLS,
        # no "receptions" model present
    }

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="WR")

    assert "receiving_yards" in result
    assert "receptions" not in result


def test_position_markets_covers_expected_markets_per_position_including_no_cfb_targets():
    # QB rushing is its own market (not pooled with RB rushing); receptions (WR/TE) is a ZINB count market and RB
    # receptions are deliberately unpriced (they failed the calibration gate).
    assert player_props.POSITION_MARKETS["QB"] == ["passing_yards", "qb_rushing_yards"]
    assert player_props.POSITION_MARKETS["RB"] == ["rushing_yards", "carries"]
    assert player_props.POSITION_MARKETS["WR"] == ["receiving_yards", "receptions"]
    assert player_props.POSITION_MARKETS["TE"] == ["receiving_yards", "receptions"]


def test_predict_props_only_returns_relevant_yardage_market_for_qb():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    passing_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["passing_yards"])
    models = {
        "anytime_td": td_model,
        "passing_yards": passing_model,
        "feature_cols": FEATURE_COLS,
    }

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="QB")

    assert "anytime_td_prob" in result
    assert "passing_yards" in result
    assert "rushing_yards" not in result
    assert "receiving_yards" not in result


def test_predict_props_only_returns_relevant_yardage_market_for_wr():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    receiving_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["receiving_yards"])
    models = {
        "anytime_td": td_model,
        "receiving_yards": receiving_model,
        "feature_cols": FEATURE_COLS,
    }

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="WR")

    assert "anytime_td_prob" in result
    assert "receiving_yards" in result
    assert "passing_yards" not in result
    assert "rushing_yards" not in result


def test_predict_props_only_returns_relevant_yardage_market_for_te():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    receiving_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["receiving_yards"])
    models = {
        "anytime_td": td_model,
        "receiving_yards": receiving_model,
        "feature_cols": FEATURE_COLS,
    }

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="TE")

    assert "anytime_td_prob" in result
    assert "receiving_yards" in result
    assert "passing_yards" not in result
    assert "rushing_yards" not in result


def test_compact_quantiles_repairs_crossings_and_rounds():
    from nfl_predictor.models.prop_probability import compact_quantiles
    assert compact_quantiles({0.1: 10.04, 0.5: 50.0, 0.9: 40.0, 0.3: 30.26}) == [10.0, 30.3, 50.0, 50.0]


def test_every_prop_row_carries_a_line_check_even_when_it_has_nothing_to_offer():
    import numpy as np
    import pandas as pd
    from nfl_predictor.models import player_props

    class _Td:
        def predict_proba(self, X):
            return np.array([[0.9, 0.1]])

    class _Q:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.array([self.v])

    models = {"feature_cols": ["a"], "anytime_td": _Td(),
              "passing_yards_quantiles": {q: _Q(100 + 300 * q) for q in player_props.QUANTILES},
              "passing_yards_quantile_cols": ["a"]}
    row = pd.Series({"a": 1.0})
    qb = player_props.predict_props(models, row, position="QB")
    assert len(qb["line_check"]["yardage"]["passing_yards"]) == len(player_props.QUANTILES)
    assert "passing_yards_quantiles" in qb          # the raw dict is still stripped later, by the route
    assert player_props.predict_props(models, row, position="K")["line_check"] == {}

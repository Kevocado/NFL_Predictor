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


def prepare_folds(games_df: pd.DataFrame, min_train_seasons: int = 2) -> list[dict]:
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


def _predict_margin_elo_batch(candidate: dict, df: pd.DataFrame) -> np.ndarray:
    """Vectorized elo margin prediction over any frame that carries
    rating_diff/home_rest_days/away_rest_days columns — used both for the
    val_df predictions and, via _EloModelAdapter below, for the train_df
    residuals residual_sigma needs. feature_cols always includes these three
    columns (see features/build.py's FEATURE_COLUMNS), so the X frame
    residual_sigma hands us has them whether it's the full fold frame or the
    X_train/X_val feature-only slice."""
    return np.array(
        [
            game_outcome.predict_margin_elo(candidate, r, hr, ar)
            for r, hr, ar in zip(df["rating_diff"], df["home_rest_days"], df["away_rest_days"])
        ]
    )


class _EloModelAdapter:
    """Adapts the elo candidate's dict + free function to the model.predict(X)
    interface residual_sigma expects, without ignoring the X it's given."""

    def __init__(self, candidate: dict):
        self.candidate = candidate

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _predict_margin_elo_batch(self.candidate, X)


def _predict_margins(candidate: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    X_train, y_train = train_df[feature_cols], train_df["margin"]
    X_val = val_df[feature_cols]

    if candidate == "elo":
        model = game_outcome.fit_elo_candidate(train_df)
        preds = _predict_margin_elo_batch(model, val_df)
        sigma = game_outcome.residual_sigma(_EloModelAdapter(model), X_train, y_train)
        return preds, sigma

    if candidate == "ridge":
        fit_fn = game_outcome.fit_margin_regression
    elif candidate == "xgb":
        fit_fn = game_outcome.fit_xgb_margin
    else:
        raise ValueError(f"Unknown candidate: {candidate!r}")

    model = fit_fn(X_train, y_train)
    preds = model.predict(X_val.fillna(0))
    sigma = game_outcome.residual_sigma(model, X_train, y_train)
    return preds, sigma


def pooled_predictions(folds: list[dict], candidate: str) -> tuple[np.ndarray, np.ndarray]:
    """All folds' held-out probs/outcomes concatenated -- enough volume for
    a meaningful reliability curve, unlike any single fold alone."""
    all_probs, all_outcomes = [], []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        preds, sigma = _predict_margins(candidate, train_df, val_df, feature_cols)
        probs = np.array([game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds])
        all_probs.append(np.clip(probs, 1e-6, 1 - 1e-6))
        all_outcomes.append((val_df["margin"] > 0).astype(int).to_numpy())
    return np.concatenate(all_probs), np.concatenate(all_outcomes)


"""quantile walk-forward -- season-by-season validation for the yardage prop
models, mirroring the game-outcome harness above.

The feature frame is built ONCE by the caller, with every rolling value already
shift(1), and then sliced by season. That is the same discipline the
game-outcome folds use, and it is what makes a validation season's predictions
independent of everything after it.
"""

import numpy as np
import pandas as pd

from ..models.player_props import QUANTILES, fit_yardage_quantile_models
from ..models.prop_probability import p_over_from_quantiles

#: Markets the quantile models cover, and the positions that actually get a line
#: for them. A WR has no rushing-yards prop, so training one would be fitting a
#: market that cannot be bet.
QUANTILE_MARKETS: dict[str, tuple[str, ...]] = {
    "passing_yards": ("QB",),
    "rushing_yards": ("RB",),
    "receiving_yards": ("WR", "TE"),
}

#: Proxy lines are not real book lines -- no free historical props exist. Each
#: row is scored against an exogenous line, which tests calibration without
#: pretending to test profitability.
DEFAULT_MIN_PROXY_HISTORY = 3

#: Scoring lines. `exogenous` is the BINDING gate; `own_median` is reported
#: alongside it as a non-binding diagnostic.
#:
#: **Amended 2026-10-04** (plan Task 8, matching the spec). The gate used to score
#: against the player's own trailing median. That line is endogenous to the same
#: recent form the model predicts from, so it compresses the empirical P(over)
#: range -- observed 0.128-0.737 against a predicted 0.02-0.98, with the gap
#: scaling 0.005 mid-range to 0.289 in the tail -- and fails a model whose
#: held-out quantile coverage is good at every level. The cross-sectional median
#: of trailing medians for the same market and week is set by the market rather
#: than by the player being priced, which is the closest available analogue to a
#: book line.
SCORING_LINES = ("exogenous", "own_median")

#: Per-line column names. `proxy_line` stays the name of the BINDING line so
#: existing readers of the offline report keep working unchanged.
LINE_COLUMN = {"exogenous": "proxy_line", "own_median": "own_median_line"}


def proxy_line(values: pd.Series, by: pd.Series, min_history: int = DEFAULT_MIN_PROXY_HISTORY) -> pd.Series:
    """The player's own median yardage over games strictly before this week.

    shift(1) inside an expanding median: a week-W row sees weeks < W only, which
    is the same anti-leakage rule as every rolling feature.
    """
    return values.groupby(by).transform(
        lambda s: s.shift(1).expanding(min_periods=min_history).median())


def naive_prediction(values: pd.Series, by: pd.Series, window: int = 5) -> pd.Series:
    """The cheap honest baseline the quantile models have to beat: the player's
    own lagged rolling mean."""
    return values.groupby(by).transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def exogenous_line(own_median: pd.Series, market_season_week) -> pd.Series:
    """The cross-sectional median of players' trailing medians, per market-week.

    A player scored against this line is scored against a number set by the rest
    of the market, not by his own recent form -- the property the binding gate
    needs. Rows whose own median is unknown (too little history) have no
    cross-sectional median either, and fall out as unscoreable.
    """
    return own_median.groupby(market_season_week).transform("median")


def walk_forward_quantile(feature_df: pd.DataFrame, markets: list[str],
                          seasons=range(2018, 2026), feature_cols: list[str] | None = None,
                          quantiles: list[float] | None = None,
                          window: int = 5) -> pd.DataFrame:
    """Train on seasons < Y, score season Y, for each validation season.

    Returns one row per (market, season, week, player) with P(over) at that
    player's proxy line, the outcome, and both the q50 and naive point
    predictions so MAE can be compared without refitting.
    """
    quantiles = QUANTILES if quantiles is None else quantiles
    feature_cols = list(feature_cols or [])
    rows: list[dict] = []

    for market in markets:
        if market not in QUANTILE_MARKETS:
            raise ValueError(f"unknown market: {market!r}")

        sub = feature_df[feature_df["position"].isin(QUANTILE_MARKETS[market])
                         & feature_df[market].notna()]
        if sub.empty:
            continue
        sub = sub.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        by_player = sub["player_id"]
        own_median = proxy_line(sub[market], by_player)
        sub = sub.assign(
            proxy_line=exogenous_line(own_median, [sub["season"], sub["week"]]),
            own_median_line=own_median,
            naive_pred=naive_prediction(sub[market], by_player, window),
        )

        for val_season in seasons:
            train = sub[sub["season"] < val_season]
            val = sub[sub["season"] == val_season]
            if train.empty or val.empty:
                continue

            models = fit_yardage_quantile_models(train[feature_cols], train[market], quantiles)
            predicted = {q: models[q].predict(val[feature_cols].fillna(0)) for q in quantiles}

            for i, (_, record) in enumerate(val.iterrows()):
                quantiles_at_row = {q: float(predicted[q][i]) for q in quantiles}
                row = {
                    "market": market,
                    "season": int(val_season),
                    "week": int(record["week"]),
                    "player_id": record["player_id"],
                    "position": record["position"],
                    "actual": float(record[market]),
                    "q10": quantiles_at_row.get(0.1, np.nan),
                    "q50": quantiles_at_row.get(0.5, np.nan),
                    "q90": quantiles_at_row.get(0.9, np.nan),
                    "naive_pred": record["naive_pred"],
                }
                # Both lines are scored on the same row so the binding curve and
                # the tail-watch diagnostic are directly comparable.
                for label, column in (("exogenous", "proxy_line"),
                                      ("own_median", "own_median_line")):
                    line = record[column]
                    scoreable = pd.notna(line)
                    row[LINE_COLUMN[label]] = line
                    row[f"p_over_{label}"] = (
                        p_over_from_quantiles(quantiles_at_row, float(line)) if scoreable else np.nan)
                    row[f"covered_{label}"] = (
                        int(record[market] > line) if scoreable else np.nan)
                rows.append(row)

    return pd.DataFrame(rows)


def calibration_report(df: pd.DataFrame, n_bins: int = 10, min_n: int = 100,
                       tolerance: float = 0.05, columns: tuple[str, str] = ("p_over", "covered")
                       ) -> dict[str, dict]:
    """Bucket predicted P(over) and compare each bucket to its empirical rate.

    Bucket edges are derived from the bucket index rather than from np.digitize
    over a linspace: at exactly 0.6 the float edges put the value in "0.5-0.6",
    which reads as an error in a report table whose predicted mean is 0.6.

    Buckets below `min_n` are omitted rather than reported with a caveat: the
    spec's gate is defined on n >= 100, and a verdict drawn from 40 predictions
    is noise wearing a number. `within_tolerance` is the per-bucket gate result.
    """
    probability_column, outcome_column = columns
    probabilities = pd.Series(df[probability_column], index=df.index).astype(float)
    outcomes = pd.Series(df[outcome_column], index=df.index).astype(float)
    keep = probabilities.notna() & outcomes.notna()
    probabilities, outcomes = probabilities[keep], outcomes[keep]

    values = probabilities.to_numpy()
    bucket = np.clip((values * n_bins).astype(int), 0, n_bins - 1)

    report: dict[str, dict] = {}
    for b in range(n_bins):
        selected = bucket == b
        n = int(selected.sum())
        if n < min_n:
            continue
        predicted = float(values[selected].mean())
        empirical = float(outcomes.to_numpy()[selected].mean())
        gap = abs(predicted - empirical)
        report[f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}"] = {
            "predicted": predicted,
            "empirical": empirical,
            "n": n,
            "gap": gap,
            "within_tolerance": bool(gap <= tolerance),
        }
    return report


def calibration_passes(report: dict[str, dict]) -> bool:
    """True when every reported bucket is within tolerance. An empty report is
    not a pass: it means there was not enough data to judge, which is a stop."""
    return bool(report) and all(bucket["within_tolerance"] for bucket in report.values())


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

"""receptions_zinb.py -- a per-player expected-receptions projection (mu), a model
line derived from it, and an over/under call on that line, for WR/TE (running backs are NOT priced: see RECEPTIONS_POSITIONS).

Why a ZINB and not the yardage quantile regressor
-------------------------------------------------
Commit `b3331be` added `receptions` to the aggregate quantile gate and
documented the failure: worst gap 0.138, overconfident in 7/8 buckets despite
good MAE. Receptions are discrete counts, not continuous yardage. The Phase 4
experiment (2026-10-13; `~/workspace/dev/predictor-work/PHASE4-REPORT.md`)
measured three count distributions walk-forward on 2017-2024, winner by
held-out mean NLL:

* Poisson: 1.8230 / 1.8044 / 1.7916 (2022 / 2023 / 2024)
* NB2 (alpha ~ 4.7): 1.7778 / 1.7730 / 1.7546 -- genuine overdispersion,
  unlike passing TDs where alpha -> infinity and Poisson wins
* ZINB (alpha ~ 7.2): 1.7520 / 1.7494 / 1.7213 -- wins 3/3 folds

The zero inflation is structural: P(y=0) is 0.68 actual vs 0.45 NB-predicted at
mu 0.5-1.0 (fringe/bench players who dress but are not targeted). The hurdle
component fixes it, and the walk-forward calibration gate PASSES with worst
gap 0.035 across the 5 populated bins (0.6+ bins are structurally empty: the
model line tracks mu, so extreme P(over) does not occur).

Reuses `qb_passing_td._NB_LOG_ALPHA_BOUNDS` and `qb_passing_td.model_line` --
not reimplemented. The ZINB tail probability is new (the hurdle changes the
survival function), derived in `_zinb_tail_probs`.

The "model line", and what it is not
------------------------------------
`model_line(mu)` (from `qb_passing_td`) is the nearest half point to the
model's own expectation. It is derived from the projection, so it is not a
sportsbook line and calling it an edge would be a category error. It is
labelled `"model_line"` everywhere it is produced and stored.

Serving guardrail
-----------------
mu > 6 (~3% of rows, star WRs) carries level bias -0.8 to -2.0, so P(over) at
fixed book lines >= 5.5 is overstated ~0.1-0.15. The call dict flags
`mu_flag="high_mu"` when mu > 7.0; consumers must not auto-price reception
lines above 7.5 on a flagged row. L2 on the ZINB slopes does not fix the bias
(NLL worsens) and isotonic recalibration overfits the train era -- the
guardrail is the honest fix.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .qb_passing_td import _NB_LOG_ALPHA_BOUNDS, model_line

#: The market name this writes. Stored under this key in `player_models` and
#: in the manifest, mirroring `qb_passing_td.PASSING_TD_MARKET`.
RECEPTIONS_MARKET = "receptions"

#: Positions this model prices. WR and TE only: the walk-forward gate passes on receivers (worst gap 0.037) but fails
#: on running backs (rushing-first backs at model line 1.5: P(over) 0.29 vs 0.18 actual, n=141, worsening each season
#: 2022-2024). A back's reception total IS a heavily bet prop, so this is a known gap, not a judgment that it is
#: unimportant: RB receptions stay unpriced until a model passes the gate for them. Do not widen without re-running it.
RECEPTIONS_POSITIONS = ("WR", "TE")

#: The only value `line_source` ever takes. A constant, not a parameter,
#: because "model line" is a fact about provenance rather than a choice.
MODEL_LINE_SOURCE = "model_line"

#: Serving guardrail: flag rows above this mu; do not auto-price lines above
#: `MAX_AUTO_LINE` on a flagged row. See the module docstring for the measured
#: bias that motivates it.
MU_GUARDRAIL = 7.0
MAX_AUTO_LINE = 7.5

#: Winsorization percentiles. The log link extrapolates exponentially past the
#: largest feature values ever seen (unclipped mu hit 14.5 on a 99.9th-pct
#: targets_roll); clipping at train-time 0.5/99.5 pcts bounds it. Bounds come
#: from TRAIN only and are stored in the artifact for serving.
_WINSOR_LO_Q = 0.005
_WINSOR_HI_Q = 0.995

#: Features the ZINB reads. Six are rolling means from `FORWARD_FEATURE_COLUMNS`;
#: `catch_rate` is `receptions_roll / targets_roll` (the prototype's
#: `catch_rate_roll`, same signal, computable from the frame without a new
#: builder); `is_WR` / `is_TE` are position dummies derived from `position`.
#:
#: The prototype also used `target_share_roll` and `roll_n`, which are not in
#: the repo frame. This list must be re-validated through the calibration gate
#: before the artifact ships -- `training.train_all` refuses to write it
#: without a passing receptions gate.
RECEPTIONS_FEATURE_COLUMNS = (
    "receptions_roll", "targets_roll", "receiving_yards_roll",
    "carries_roll", "rushing_yards_roll", "yards_per_target_roll",
    "catch_rate", "is_WR", "is_TE",
)

#: Roll columns the frame must carry for `_derive_features` to run.
_BASE_ROLL_COLUMNS = (
    "receptions_roll", "targets_roll", "receiving_yards_roll",
    "carries_roll", "rushing_yards_roll", "yards_per_target_roll",
)

#: Filename of the fitted artifact, beside the quantile pickles.
RECEPTIONS_ZINB_FILENAME = "receptions_zinb_model.pkl"


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """The 9 ZINB features from a frame carrying the roll columns and `position`.

    Derived in-module (not by a feature builder) so fit and serving cannot
    drift apart: both call this. A missing base column raises rather than
    filling -- scoring a player on a constant-zero feature is the defect
    `qb_passing_td.expected_passing_tds` guards against, per player.
    """
    missing = [c for c in (*_BASE_ROLL_COLUMNS, "position") if c not in df.columns]
    if missing:
        raise KeyError(
            f"the receptions ZINB model needs {missing} on the frame; "
            "scoring without them would fill with 0.0"
        )
    out = pd.DataFrame(index=df.index)
    for col in _BASE_ROLL_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # Ratio of rolls, not a rolling mean of ratios -- computable from the frame
    # and the same signal the prototype validated as `catch_rate_roll`.
    out["catch_rate"] = out["receptions_roll"] / out["targets_roll"].clip(lower=1e-9)
    # A catch rate above 1 is a data artifact (rolls drawn from different
    # windows); cap it rather than feeding the log link an impossible ratio.
    out["catch_rate"] = out["catch_rate"].clip(upper=1.0)
    pos = df["position"].astype(str)
    out["is_WR"] = (pos == "WR").astype(float)
    out["is_TE"] = (pos == "TE").astype(float)
    return out[list(RECEPTIONS_FEATURE_COLUMNS)]


def _nb_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    p = alpha / (alpha + np.clip(mu, 1e-12, None))
    return stats.nbinom.logpmf(y, alpha, p)


def _zinb_nll(theta: np.ndarray, Xsi: np.ndarray, y: np.ndarray, k: int) -> float:
    """Mean negative log-likelihood of the ZINB.

    `logit(pi) = z.g`, `log(mu) = x.b`, `y ~ pi.d0 + (1-pi).NB2(mu, alpha)`.
    `Xsi` carries the intercept column; `g` and `b` each have `k + 1`
    coefficients and `theta[2*k + 2]` is `log(alpha)`.
    """
    g, b, la = theta[:k + 1], theta[k + 1:2 * k + 2], theta[2 * k + 2]
    alpha = math.exp(float(la))
    pi = 1.0 / (1.0 + np.exp(-(Xsi @ g)))
    mu = np.exp(Xsi @ b)
    ll_nb = _nb_logpmf(y, mu, alpha)
    log1m_pi = np.log1p(-np.clip(pi, 0.0, 1.0 - 1e-12))
    ll = np.where(
        y == 0,
        np.logaddexp(np.log(np.clip(pi, 1e-12, 1.0)), log1m_pi + ll_nb),
        log1m_pi + ll_nb,
    )
    nll = float(-np.mean(ll))
    return nll if math.isfinite(nll) else 1e12


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_ = X.mean(axis=0)
    sd_ = X.std(axis=0) + 1e-12
    return (X - mu_) / sd_, mu_, sd_


def fit_receptions_zinb(df: pd.DataFrame, y: pd.Series) -> dict:
    """Fit the ZINB on WR/TE history; also scores Poisson/NB for the manifest.

    `df` carries the roll columns and `position`; `y` is the `receptions`
    label. Features are winsorized at train-time 0.5/99.5 pcts (bounds stored
    in the artifact and applied at serving). Optimizer inits come from a
    hurdle logistic regression and a Poisson mean model, with alpha starting
    at the prototype's measured ~4.7.

    Returns the artifact dict, including the three-way log-loss comparison so
    the manifest records the choice inspectably, like `qb_passing_td` does.
    """
    Xf = _derive_features(df)
    X = Xf.to_numpy(dtype=float)
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yv)
    X, yv = X[mask], yv[mask]
    if len(yv) == 0:
        raise ValueError("no WR/TE reception history to fit on")

    lo = np.quantile(X, _WINSOR_LO_Q, axis=0)
    hi = np.quantile(X, _WINSOR_HI_Q, axis=0)
    Xw = np.clip(X, lo, hi)
    Xs, feat_mu, feat_sd = _standardize(Xw)
    k = Xs.shape[1]
    Xsi = np.column_stack([np.ones(len(Xs)), Xs])

    # Poisson and NB2 log losses, for the manifest's three-way comparison.
    # The NB mean is held at the Poisson fit so the comparison measures
    # overdispersion rather than a better mean (same discipline as
    # `qb_passing_td.fit_qb_passing_td_model`).
    from sklearn.linear_model import LogisticRegression, PoissonRegressor

    pr = PoissonRegressor(alpha=1e-8, max_iter=1000).fit(Xs, yv)
    mu_pois = np.clip(pr.predict(Xs), 1e-12, None)
    poisson_ll = float(-stats.poisson.logpmf(yv, mu_pois).mean())

    from .qb_passing_td import _nb_alpha_nll

    nb_fit = optimize.minimize_scalar(
        lambda la: _nb_alpha_nll(la, yv, mu_pois),
        bounds=_NB_LOG_ALPHA_BOUNDS, method="bounded",
    )
    nb_ll = float(nb_fit.fun) if math.isfinite(nb_fit.fun) else math.inf

    # ZINB joint MLE, inits from the hurdle + Poisson fits.
    lr = LogisticRegression(max_iter=1000).fit(Xs, (yv == 0).astype(int))
    g0 = np.concatenate([[float(lr.intercept_[0])], lr.coef_[0]])
    b0 = np.concatenate([[float(pr.intercept_)], pr.coef_])
    theta0 = np.concatenate([g0, b0, [math.log(4.7)]])
    bounds = [(None, None)] * (2 * k + 2) + [_NB_LOG_ALPHA_BOUNDS]
    res = optimize.minimize(
        _zinb_nll, theta0, args=(Xsi, yv, k),
        method="L-BFGS-B", bounds=bounds, options={"maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"ZINB fit did not converge: {res.message}")
    g, b, la = res.x[:k + 1], res.x[k + 1:2 * k + 2], float(res.x[2 * k + 2])
    alpha = math.exp(la)
    zinb_ll = float(res.fun)

    return {
        "g": np.asarray(g, dtype=float),
        "b": np.asarray(b, dtype=float),
        "alpha": float(alpha),
        "feat_mu": np.asarray(feat_mu, dtype=float),
        "feat_sd": np.asarray(feat_sd, dtype=float),
        "clip_lo": np.asarray(lo, dtype=float),
        "clip_hi": np.asarray(hi, dtype=float),
        "feature_cols": list(RECEPTIONS_FEATURE_COLUMNS),
        "distribution": "zero_inflated_negative_binomial",
        "log_loss": {
            "poisson": poisson_ll,
            "negative_binomial": nb_ll,
            "zero_inflated_negative_binomial": zinb_ll,
        },
        "n_train": int(len(yv)),
        # var/mean on the raw label, so the overdispersion the ZINB is
        # reacting to is visible in the artifact itself.
        "variance_ratio": float(np.var(yv) / max(np.mean(yv), 1e-12)),
        "mu_guardrail": MU_GUARDRAIL,
        "max_auto_line": MAX_AUTO_LINE,
    }


def _predict_parts(fitted: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pi, mu_nb, mu) for each row: hurdle prob, NB mean, unconditional mean."""
    X = _derive_features(df).to_numpy(dtype=float)
    lo, hi = fitted.get("clip_lo"), fitted.get("clip_hi")
    if lo is not None and hi is not None:
        X = np.clip(X, np.asarray(lo), np.asarray(hi))
    Xs = np.column_stack([
        np.ones(len(X)),
        (X - np.asarray(fitted["feat_mu"])) / np.asarray(fitted["feat_sd"]),
    ])
    pi = 1.0 / (1.0 + np.exp(-(Xs @ np.asarray(fitted["g"]))))
    mu_nb = np.exp(Xs @ np.asarray(fitted["b"]))
    mu = (1.0 - pi) * mu_nb
    return pi, mu_nb, mu


def _zinb_tail_probs(pi: float, mu_nb: float, alpha: float,
                     threshold: float) -> tuple[float, float]:
    """(P(count > threshold), P(count < threshold)) under the ZINB.

    `threshold` is the line itself and always ends in .5, so `>` and `<` are
    exhaustive over the integers: no push. Zeros are always "under" -- the
    hurdle mass sits entirely below any bettable line.
    """
    mu_nb = max(float(mu_nb), 1e-12)
    alpha = float(alpha)
    p = alpha / (alpha + mu_nb)
    sf = float(stats.nbinom.sf(threshold, alpha, p))
    cdf = float(stats.nbinom.cdf(threshold, alpha, p))
    return (1.0 - pi) * sf, (1.0 - pi) * cdf + pi


def expected_receptions(fitted: dict, feature_row: pd.Series, position: str) -> float:
    """mu for one player, from that player's own pregame feature row.

    The row carries the roll columns; `position` supplies the dummies. A row
    missing a base column raises rather than filling -- see `_derive_features`.
    """
    if fitted is None:
        raise ValueError("no fitted receptions model")
    frame = feature_row.to_frame().T.assign(position=position)
    _, _, mu = _predict_parts(fitted, frame)
    return max(float(mu[0]), 0.0)


def receptions_call(fitted: dict, feature_row: pd.Series, position: str,
                    line: float | None = None) -> dict | None:
    """Projection -> line -> side for one WR/TE, or None when unproducible.

    The line is `model_line(mu)` unless one is passed in. If a real sportsbook
    line is ever wired in it is passed here instead, and nothing about this
    function or the pick record changes; `line_source` is the only field that
    has to describe where it came from. Until then it is always `"model_line"`,
    and because the line is derived from mu this is NOT an edge claim.

    `mu_flag` is `"high_mu"` when mu exceeds the serving guardrail -- the
    caller must not auto-price lines above 7.5 on a flagged row.
    """
    if fitted is None:
        return None
    if position not in RECEPTIONS_POSITIONS:
        return None
    mu = expected_receptions(fitted, feature_row, position)
    call_line = model_line(mu) if line is None else float(line)
    # The NB mean for the tail, recovered from the unconditional mu.
    frame = feature_row.to_frame().T.assign(position=position)
    pi, mu_nb, _ = _predict_parts(fitted, frame)
    over_prob, under_prob = _zinb_tail_probs(float(pi[0]), float(mu_nb[0]),
                                               float(fitted["alpha"]), call_line)
    side = "over" if over_prob > under_prob else "under"
    return {
        "market": RECEPTIONS_MARKET,
        "line": call_line,
        "line_source": MODEL_LINE_SOURCE,
        "side": side,
        "mu": float(mu),
        "over_prob": over_prob,
        "under_prob": under_prob,
        "call_prob": max(over_prob, under_prob),
        "distribution": "zero_inflated_negative_binomial",
        # A half-point line against an integer count cannot tie.
        "push_prob": 0.0,
        "mu_flag": "high_mu" if mu > MU_GUARDRAIL else None,
    }


def walk_forward_receptions(frame: pd.DataFrame,
                            validation_seasons: list[int],
                            min_train_seasons: int = 5) -> pd.DataFrame:
    """Expanding-window ZINB walk-forward, scored at the model line.

    For each validation season Y, fits on seasons < Y (at least
    `min_train_seasons` of them) and scores season Y: mu, model line,
    P(over) at that line, and whether the actual went over. Returns one row
    per (season, week, player). The caller bins P(over) with the repo's
    `calibration_report` arithmetic to produce the gate verdict that
    `training.train_all` requires as `receptions_calibration`.

    Mirrors the Phase 4 experiment's folds (train 2017-2021 -> score 2022,
    etc.); regular season only (week <= 18).
    """
    from ..evaluate.walk_forward import calibration_report  # local import: evaluate imports training

    rows = frame[
        frame["position"].isin(RECEPTIONS_POSITIONS)
        & frame["receptions"].notna()
        & (frame["week"] <= 18)
    ].sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    scored = []
    for season in validation_seasons:
        train = rows[rows["season"] < season]
        test = rows[rows["season"] == season]
        if test.empty:
            continue
        train_seasons = sorted(train["season"].unique())
        if len(train_seasons) < min_train_seasons:
            continue
        fitted = fit_receptions_zinb(train, train["receptions"])
        pi, mu_nb, mu = _predict_parts(fitted, test)
        alpha = float(fitted["alpha"])
        for idx, (pi_i, mu_nb_i, mu_i, actual) in enumerate(
                zip(pi, mu_nb, mu, test["receptions"].to_numpy())):
            line = model_line(float(mu_i))
            over_prob, _ = _zinb_tail_probs(float(pi_i), float(mu_nb_i), alpha, line)
            scored.append({
                "season": season,
                "player_id": test.iloc[idx]["player_id"],
                "p_over": over_prob,
                "covered": bool(actual > line),
                "line": line,
                "mu": float(mu_i),
            })
    return pd.DataFrame(scored)


def receptions_calibration_gate(scored: pd.DataFrame) -> dict:
    """Bin the walk-forward P(over) with the repo's calibration arithmetic.

    Same 10 bins, tol 0.05, n>=100 as the quantile gate -- the number
    `training.train_all` checks via `_receptions_gate_passes`.
    """
    from ..evaluate.walk_forward import calibration_report  # local import: evaluate imports training

    return calibration_report(scored, columns=("p_over", "covered"))


def save_receptions_artifact(fitted: dict, out_dir: Path | str) -> Path:
    """Write the fitted ZINB pickle beside the quantile artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RECEPTIONS_ZINB_FILENAME
    path.write_bytes(pickle.dumps(fitted))
    return path


def receptions_manifest_entry(fitted: dict) -> dict:
    """The ZINB's provenance for the manifest, mirroring `qb_passing_td`.

    All three log losses are written, not just the winner's: the choice among
    Poisson, NB and ZINB is only meaningful if the losing numbers are
    inspectable.
    """
    return {
        "distribution": fitted["distribution"],
        "log_loss": fitted["log_loss"],
        "alpha": fitted["alpha"],
        "n_train": fitted["n_train"],
        "variance_ratio": fitted["variance_ratio"],
        "feature_cols": fitted["feature_cols"],
        "mu_guardrail": fitted["mu_guardrail"],
        "max_auto_line": fitted["max_auto_line"],
    }

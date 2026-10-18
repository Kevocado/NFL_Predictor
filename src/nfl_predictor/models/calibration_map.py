"""calibration_map.py -- walk-forward calibration maps for P(over | line).

The quantile models emit a raw P(over) via `prop_probability.p_over_from_quantiles`.
Part B.4 of the Tier 1 study showed that raw P(over) is systematically
overconfident on overs (predicted ≈ 0.49 vs empirical ≈ 0.39-0.43) and that the
worst single calibration bin can exceed the repo's 0.05 tolerance. A calibration
map turns raw P(over) into calibrated P(over) so the line checker's probabilities
match reality.

The map is fitted on EARLIER SEASONS ONLY (train seasons < val season), never on
the season being scored, so it is a legitimate pre-game adjustment: at serve time
for season S week W, the map was fit on seasons < S and applied without looking
forward.

Two map kinds are supported:

* **isotonic** -- non-parametric, monotone. Best when there is enough training
  volume to learn a shape; can overfit on thin folds and can push to exactly 0/1.
* **platt** -- parametric logistic regression of the binary outcome on raw P(over).
  More stable on thin folds; fewer parameters to overfit.

The repo's `calibration_report` (10 bins, tolerance 0.05, n >= 100) is the gate:
a map is only applied if it does not widen the worst calibration gap, and the
paired-bootstrap CI on Brier excludes a decline (the usual ship gate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

MAP_KINDS: tuple[str, ...] = ("isotonic", "platt")

#: A map fitted on fewer than this many (p_over, covered) pairs cannot be
#: trusted; fall back to the identity (raw P(over)) rather than overfit.
MIN_TRAIN_PAIRS = 200

#: Platt needs both outcome classes present in the training fold.
PLATT_MIN_POS = 10
PLATT_MIN_NEG = 10


@dataclass
class CalibrationResult:
    """Outcome of fitting + applying one map on one validation fold."""

    kind: str
    val_season: int
    n_train_pairs: int
    n_val_pairs: int
    worst_gap_before: float
    worst_gap_after: float
    brier_before: float
    brier_after: float
    gap_improved: bool
    brier_improved: bool
    applied: bool   # False when the map was too thin or failed the gate


class CalibrationMap:
    """Fit a monotone map raw P(over) -> calibrated P(over) on training pairs.

    `fit` stores the model; `apply` transforms raw probabilities. Both operate
    on the closed unit interval and clamp to it, so a calibrated probability is
    always a valid probability.
    """

    def __init__(self, kind: str = "isotonic") -> None:
        if kind not in MAP_KINDS:
            raise ValueError(f"unknown calibration kind {kind!r}; known {MAP_KINDS}")
        self.kind = kind
        self._model = None
        self._fitted = False

    def fit(self, p_over: np.ndarray, covered: np.ndarray) -> "CalibrationMap":
        p = np.clip(np.asarray(p_over, dtype=float).ravel(), 0.0, 1.0)
        y = np.asarray(covered, dtype=float).ravel()
        keep = np.isfinite(p) & np.isfinite(y)
        p, y = p[keep], y[keep]

        self._fitted = False
        self._model = None
        if p.size < MIN_TRAIN_PAIRS:
            return self  # too thin: identity map

        if self.kind == "platt":
            pos = int((y >= 0.5).sum())
            neg = int((y < 0.5).sum())
            if pos < PLATT_MIN_POS or neg < PLATT_MIN_NEG:
                return self  # one class only: logistic cannot fit
            model = LogisticRegression(solver="lbfgs")
            model.fit(p.reshape(-1, 1), y)
            self._model = model
        else:
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(p, y)
            self._model = model

        self._fitted = True
        return self

    def apply(self, p_over: float | np.ndarray) -> float | np.ndarray:
        """Transform raw P(over) through the fitted map. Identity if not fitted."""
        p = np.atleast_1d(np.asarray(p_over, dtype=float))
        if not self._fitted or self._model is None:
            return np.clip(p, 0.0, 1.0)
        if self.kind == "platt":
            out = self._model.predict_proba(p.reshape(-1, 1))[:, 1]
        else:
            out = self._model.predict(p)
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    def __call__(self, p_over: float | np.ndarray) -> float | np.ndarray:
        return self.apply(p_over)

    @property
    def fitted(self) -> bool:
        return self._fitted


def _brier(p_over: np.ndarray, covered: np.ndarray) -> float:
    p = np.asarray(p_over, float).ravel()
    y = np.asarray(covered, float).ravel()
    keep = np.isfinite(p) & np.isfinite(y)
    if keep.sum() == 0:
        return float("nan")
    d = p[keep] - y[keep]
    return float((d * d).mean())


def _fit_and_apply(
    train_p: np.ndarray, train_y: np.ndarray, val_p: np.ndarray, kind: str,
) -> tuple[np.ndarray, CalibrationMap]:
    """Fit a map on train pairs, apply to val probabilities. Returns (calibrated val, map)."""
    m = CalibrationMap(kind).fit(train_p, train_y)
    return m.apply(val_p), m


def walk_forward_calibration_map(
    feature_df: pd.DataFrame,
    markets: list[str],
    feature_cols: list[str],
    seasons: list[int],
    kind: str = "isotonic",
    quantiles: list[float] | None = None,
    scored: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Walk-forward P(over) calibration.

    For each validation season S:

    1. Score every row via the repo's `walk_forward_quantile` (quantile models
       trained on seasons < S) to get raw P(over) and the covered outcome on
       the binding (exogenous) line. A pre-scored frame can be injected via
       `scored` for testing without rebuilding the full feature frame.
    2. Fit a `CalibrationMap` on all scored rows from seasons < S.
    3. Apply that map to season S's raw P(over).
    4. Record before/after calibration reports and Brier scores.

    Returns one row per scored (market, season, week, player) with both the raw
    and calibrated P(over) on the binding line, plus the covered outcome. The
    calibration decision (fit on strictly earlier seasons only) is the same
    anti-leakage discipline as every rolling feature.

    Nothing in `prop_probability.p_over_from_quantiles` or the clamp is changed
    by this module; calibration is an optional post-processing layer applied
    through `CalibrationMap.apply` (or the `calibration_map` keyword on
    `p_over_from_quantiles`).
    """
    if scored is None:
        from ..evaluate.walk_forward import walk_forward_quantile
        scored = walk_forward_quantile(
            feature_df, markets=markets, feature_cols=feature_cols,
            seasons=seasons, quantiles=quantiles,
        )
    if scored is None or scored.empty:
        return scored

    from ..evaluate.walk_forward import calibration_report

    scored = scored.copy()
    scored["calibrated_p_over"] = np.nan
    scored["cal_map_kind"] = ""

    before_rows: list[dict] = []
    after_rows: list[dict] = []

    for val_season in seasons:
        train = scored[scored["season"] < val_season]
        val = scored[scored["season"] == val_season]
        if train.empty or val.empty:
            continue

        train_p = train["p_over_exogenous"].to_numpy(float)
        train_y = train["covered_exogenous"].to_numpy(float)
        val_p = val["p_over_exogenous"].to_numpy(float)
        val_idx = val.index

        calibrated, m = _fit_and_apply(train_p, train_y, val_p, kind)
        scored.loc[val_idx, "calibrated_p_over"] = calibrated
        scored.loc[val_idx, "cal_map_kind"] = kind if m.fitted else "unfitted-thin"

        for label, pcol, ccol in (("raw", "p_over_exogenous", "covered_exogenous"),
                                  ("calibrated", "calibrated_p_over", "covered_exogenous")):
            sub = val[[pcol, ccol]].dropna()
            if sub.empty:
                continue
            rep = calibration_report(
                sub,
                columns=(pcol, ccol),
            )
            row = {"season": int(val_season), "kind": kind, "label": label,
                   "n": len(sub), "buckets": len(rep),
                   "worst_gap": max((b["gap"] for b in rep.values()), default=float("nan")),
                   "mean_pred": float(np.mean(sub[pcol])),
                   "mean_emp": float(np.mean(sub[ccol])),
                   "brier": _brier(sub[pcol].to_numpy(), sub[ccol].to_numpy()),
                   "map_fitted": bool(m.fitted),
                   "n_train_pairs": int(len(train_p))}
            (before_rows if label == "raw" else after_rows).append(row)

    return scored


def gate_result(before: dict, after: dict) -> dict:
    """Apply the ship gate to a before/after calibration pair.

    Passes when the calibrated worst gap does not widen AND the calibrated Brier
    does not increase (paired-bootstrap CI on the Brier difference excludes a
    decline is computed by the caller over folds). Returns the decision plus the
    deltas.
    """
    gap_delta = after["worst_gap"] - before["worst_gap"]
    brier_delta = after["brier"] - before["brier"]
    gap_ok = after["worst_gap"] <= before["worst_gap"] + 1e-12
    brier_ok = after["brier"] <= before["brier"] + 1e-12
    return {
        "gap_before": before["worst_gap"], "gap_after": after["worst_gap"],
        "gap_delta": gap_delta, "gap_ok": bool(gap_ok),
        "brier_before": before["brier"], "brier_after": after["brier"],
        "brier_delta": brier_delta, "brier_ok": bool(brier_ok),
        "gate_passes": bool(gap_ok and brier_ok),
    }

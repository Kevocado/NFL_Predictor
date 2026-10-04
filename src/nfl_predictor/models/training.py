"""training.py -- fit the quantile yardage models and write the versioned artifacts.

**This is the gate.** `train_all` refuses to write anything unless it is handed a
walk-forward verdict, and refuses again if that verdict's calibration buckets
fail. The offline gate deciding whether an artifact may exist is the whole point
of Task 9 following Task 14; a training script that writes unconditionally would
make the gate advisory.

Writes only `models/<market>_quantile_2025.pkl` and one additive manifest key. The
live site's point-regressor pickles are never a write target.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .player_props import fit_yardage_quantile_models
from .quantile_registry import save_quantile_artifacts, verify_quantile_artifacts

logger = logging.getLogger(__name__)

#: Market -> the yardage column it models.
MARKETS = ("passing_yards", "rushing_yards", "receiving_yards")

#: Positions that actually get a line for each market. Fitting a WR's
#: rushing-yards model would be fitting a market nobody quotes.
MARKET_POSITIONS = {
    "passing_yards": ("QB",),
    "rushing_yards": ("RB",),
    "receiving_yards": ("WR", "TE"),
}

#: The features the offline gate validated, and the only ones an artifact may be
#: fitted on.
#:
#: **Declared, not derived.** Deriving the set from the frame -- "every numeric
#: column that is not the label" -- silently includes the SAME-WEEK raw stats
#: (`passing_yards`, `targets`, `completions`, `attempts`, `fantasy_points`, ...),
#: which are outcomes of the game being predicted. That is textbook target-week
#: leakage: the walk-forward gate is scored against this declared list and passes,
#: while an artifact fitted on the derived list would be fitted on the answer and
#: would collapse at serving time, where those columns do not exist yet.
#:
#: Every entry must therefore be a lagged, shift(1)-computed feature or a
#: pre-game fact (schedule, weather, injury). If a feature is not, it does not go
#: in here.
FORWARD_FEATURE_COLUMNS: tuple[str, ...] = (
    # rolling usage (shift(1)-then-rolling)
    "passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll",
    "targets_roll", "carries_roll", "receptions_roll",
    # opponent defence vs position
    "opp_pass_yds_allowed_roll", "opp_rush_yds_allowed_roll", "opp_rec_yds_allowed_roll",
    # game context, all known pre-kickoff
    "spread_line", "total_line", "is_home", "rest_days", "implied_team_total",
    "game_total", "is_outdoor", "temp_c", "wind_kph", "precip_mm", "high_wind_flag",
    # availability, all known pre-kickoff
    "inj_Q", "inj_D", "inj_O", "ol_injuries_out", "depth_rank_change",
    # opportunity and form, all lagged
    "snap_share", "snap_share_trend", "route_participation", "form_deviation",
    "separation_avg",
)


def _feature_columns(frame: pd.DataFrame, market: str) -> list[str]:
    """`FORWARD_FEATURE_COLUMNS` minus `market`, restricted to what the frame has.

    Missing columns are tolerated (a caller may not have weather, say) rather
    than raising, because the serving path must be able to run with what it has.
    A column that exists but is not in the declared list is NEVER added: that is
    the leak this function exists to prevent.
    """
    return [c for c in FORWARD_FEATURE_COLUMNS
            if c != market and c in frame.columns
            and pd.api.types.is_numeric_dtype(frame[c])]


def _training_rows(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    rows = frame[
        frame["position"].isin(MARKET_POSITIONS[market]) & frame[market].notna()
    ]
    return rows.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def train_all(frame: pd.DataFrame, trained_seasons: list[int], out_dir: Path | str,
              manifest: dict, manifest_path: Path | str,
              walkforward_mae: float | dict[str, float] | None = None,
              walkforward_calibration: dict | None = None) -> dict[str, dict]:
    """Fit and persist every market. Returns the artifacts, keyed by market.

    `walkforward_mae` and `walkforward_calibration` are the offline gate's
    verdict, passed in rather than recomputed here. Both are required.

    `walkforward_mae` may be a single number or a `{market: mae}` mapping.
    Per-market is preferred: the pooled mean stamps the same figure on all three
    artifacts and hides that passing yards and receiving yards differ by 3x.
    """
    if walkforward_mae is None or walkforward_calibration is None:
        raise ValueError(
            "refusing to write artifacts without a walkforward verdict: the offline "
            "gate decides whether these models may exist (walkforward_mae and "
            "walkforward_calibration are both required)")

    failing = sorted(label for label, bucket in walkforward_calibration.items()
                     if not bucket.get("within_tolerance", False))
    if failing:
        raise ValueError(
            f"refusing to write artifacts: calibration failed in {failing}")

    artifacts: dict[str, dict] = {}
    for market in MARKETS:
        rows = _training_rows(frame, market)
        if rows.empty:
            logger.warning("no rows for %s; not writing an artifact", market)
            continue
        feature_cols = _feature_columns(frame, market)
        if not feature_cols:
            logger.warning("no numeric features for %s; not writing an artifact", market)
            continue

        models = fit_yardage_quantile_models(rows[feature_cols], rows[market])
        mae = walkforward_mae.get(market) if isinstance(walkforward_mae, dict) \
            else walkforward_mae
        artifacts[market] = {
            "quantile_models": models,
            "feature_cols": feature_cols,
            "trained_seasons": list(trained_seasons),
            "walkforward_mae": mae,
            "walkforward_calibration": walkforward_calibration,
        }
        logger.info("fitted %s on %d rows, %d features", market, len(rows), len(feature_cols))

    extended = save_quantile_artifacts(artifacts, out_dir=out_dir, manifest=manifest,
                                       manifest_path=manifest_path)

    problems = verify_quantile_artifacts(extended, out_dir=out_dir)
    if problems:
        raise RuntimeError(f"artifacts written but do not verify: {problems}")

    logger.info("wrote %d artifacts and extended the manifest", len(artifacts))
    return artifacts
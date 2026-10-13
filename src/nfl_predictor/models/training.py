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

import json
import logging
from pathlib import Path

import pandas as pd

from .player_props import fit_yardage_quantile_models
from .quantile_registry import save_quantile_artifacts, verify_quantile_artifacts
from .receptions_zinb import (
    RECEPTIONS_MARKET,
    fit_receptions_zinb,
    receptions_manifest_entry,
    save_receptions_artifact,
)

logger = logging.getLogger(__name__)

#: Market -> the yardage column it models.
#:
#: `receptions` is here because the book quotes it: `player_receptions` is a
#: valid The Odds API market (probed 2026-10-05) and `forward_tick.MARKET_MAP`
#: already routes it here, so an artifact makes the tick price it automatically.
#: Without one the tick skipped every reception prop and reported a clean week
#: while a third of the book went unexamined.
#:
#: `receptions` is a discrete count, not continuous yardage, so it is NOT
#: fitted by the quantile regressor -- see `COUNT_MARKETS`. The Phase 4
#: experiment (2026-10-13) proved a zero-inflated negative binomial calibrates
#: it (walk-forward gate PASS, worst gap 0.035) where the yardage quantile
#: regressor failed (worst gap 0.138, overconfident in 7/8 buckets -- see
#: b3331be). It stays in MARKETS so the market is a first-class citizen
#: everywhere a market is enumerated; `train_all` routes it to the ZINB.
MARKETS = ("passing_yards", "rushing_yards", "receiving_yards",
          "qb_rushing_yards", "receptions")

#: Markets fitted by a count model rather than the quantile regressor.
#: `train_all` branches on this; the walk-forward quantile gate does not score
#: these (see `evaluate.walk_forward.QUANTILE_MARKETS`).
COUNT_MARKETS = ("receptions",)

#: The data column each market's label comes from. For every market but one
#: the name IS the column; `qb_rushing_yards` models the `rushing_yards`
#: column restricted to QB rows. QB rushing is a different distribution from
#: RB rushing (29% of QB games are <= 0 yards -- kneel-downs and sacks count
#: as rushes -- vs 11% for RBs; medians 7 vs 26), so the two positions do not
#: share a model. `walk_forward` and the serving path resolve the label
#: through this mapping, never by assuming name == column.
MARKET_LABEL_COLUMN: dict[str, str] = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "qb_rushing_yards": "rushing_yards",
    "receptions": "receptions",
}

#: Positions that actually get a line for each market. Fitting a WR's
#: rushing-yards model would be fitting a market nobody quotes.
#:
#: `receptions` covers WR/TE only. A back's reception total is a heavily bet prop, but the ZINB fails the gate on
#: backs (see the `receptions` entry below), so RB stays unpriced rather than served uncalibrated.
#:
#: **Single source of truth, imported not restated.** `walk_forward.QUANTILE_
#: MARKETS` used to declare the same mapping independently, which is exactly how
#: the gate came to validate one feature set while the artifacts carried another
#: with nothing comparing the two. `walk_forward` now imports this.
MARKET_POSITIONS = {
    "passing_yards": ("QB",),
    "rushing_yards": ("RB",),
    "receiving_yards": ("WR", "TE"),
    # `receptions` is back as of 2026-10-13, fitted by the ZINB count model
    # (`models/receptions_zinb.py`), not the yardage quantile regressor -- the
    # Phase 4 experiment proved the ZINB calibrates it (gate PASS, worst gap
    # 0.035) where quantiles failed (worst gap 0.138, see b3331be).
    # WR and TE only. The ZINB passes its walk-forward gate on receivers (worst gap 0.037) but NOT on running backs:
    # RBs with a rushing-first role (targets_roll ~0.7, carries_roll ~5) at model line 1.5 are over-predicted
    # (P(over) 0.29 vs 0.18 actual, n=141, worsening 2022->2024), so RB receptions stay UNPRICED until a model
    # passes the gate for them. Do not widen this tuple without re-running the gate.
    "receptions": ("WR", "TE"),
    # QB rushing is its own market, not pooled with RB rushing: the
    # distributions differ enough that one model miscalibrates the QB shape
    # (see MARKET_LABEL_COLUMN). The label column is `rushing_yards`.
    "qb_rushing_yards": ("QB",),
}

#: THE feature list — the single authority for both the offline gate and the
#: fitted artifact. Imported by `scripts/train_quantile_props.py` rather than
#: re-declared: two hand-maintained lists for one purpose is exactly how the
#: gate came to validate 28 features while the artifacts were fitted on 30, and
#: no test compared them.
#:
#: **Declared, not derived.** Deriving the set from the frame -- "every numeric
#: column that is not the label" -- silently includes the SAME-WEEK raw stats
#: (`passing_yards`, `targets`, `completions`, `attempts`, `fantasy_points`, ...),
#: which are outcomes of the game being predicted. That is textbook target-week
#: leakage: the walk-forward gate is scored against this declared list and passes,
#: while an artifact fitted on the derived list would be fitted on the answer and
#: would collapse at serving time, where those columns do not exist yet.
#:
#: Every entry must be a lagged, shift(1)-computed feature or a fact knowable
#: pre-kickoff. If a feature is not, it does not go in here.
#:
#: **Excluded: `route_participation`.** nflverse dropped the `route` column from
#: the 2026 play-by-play while 2017-2025 all carry it. The feature is a lagged
#: rolling mean, so a 2026 row's window is drawn entirely from 2026 weeks and is
#: therefore always missing at serving -- a feature present in training and absent
#: in production, which is the same train/serve skew the closing lines were.
#:
#: **Excluded: `spread_line`, `total_line`, `implied_team_total`, `game_total`.**
#: nflverse's weekly `spread_line`/`total_line` are the CLOSING lines, and the
#: latter two are derived from them (see `features/matchup.py`). A pre-kickoff
#: snapshot has the opening line at best, so a model carrying these is fitted on
#: information it will not have when it matters. That is a train/serve skew, and
#: it inflates the offline calibration number it is scored by.
FORWARD_FEATURE_COLUMNS: tuple[str, ...] = (
    # rolling usage (shift(1)-then-rolling)
    "passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll",
    "targets_roll", "carries_roll", "receptions_roll",
    "attempts_roll", "completions_roll",
    # derived efficiency metrics
    "yards_per_attempt_roll", "yards_per_completion_roll", "yards_per_target_roll", "yards_per_carry_roll",
    # QB-specific rushing
    "qb_rushing_yards_roll", "qb_rushing_tds_roll",
    # opponent defence vs position
    "opp_pass_yds_allowed_roll", "opp_rush_yds_allowed_roll", "opp_rec_yds_allowed_roll",
    # game context, knowable pre-kickoff
    "is_home", "rest_days", "is_outdoor",
    "temp_c", "wind_kph", "precip_mm", "high_wind_flag",
    # availability, knowable pre-kickoff
    "inj_Q", "inj_D", "inj_O", "ol_injuries_out", "depth_rank_change",
    # opportunity and form, all lagged
    "snap_share", "snap_share_trend", "opp_share", "form_deviation",
    "separation_avg",
    # interaction features: products of *independent* signals. The 2026-10-12
    # ablation proved deterministic re-expressions of existing features (the
    # efficiency ratios) add nothing for a depth-3 XGBoost; products the trees
    # cannot build themselves do. Batch 1 (expected_air_yards,
    # target_concentration, wind_pass_interaction, expected_plays) measured
    # mean yardage ΔMAE -0.082 over 17/21 folds (re-measured after fixing an
    # opponent-tempo mapping bug). Batch 2 (rz_carry_share,
    # pressure_exposure, protection_time, gunslinger -- see
    # features/interactions.py::EXPERIMENTAL_COLUMNS) measured +0.026 / 4/21
    # and stays OUT until it clears the bar with better data.
    # Built by `features/interactions.py`, wired into
    # `scripts/train_quantile_props.py::build_feature_frame` after
    # `add_matchup_features` / `add_availability_features` (it reads
    # `high_wind_flag` and `ol_injuries_out` off the frame).
    "expected_air_yards", "target_concentration", "wind_pass_interaction",
    "expected_plays",
)



def _feature_columns(frame: pd.DataFrame, market: str) -> list[str]:
    """`FORWARD_FEATURE_COLUMNS` minus the market's label column, restricted
    to what the frame has.

    Missing columns are tolerated (a caller may not have weather, say) rather
    than raising, because the serving path must be able to run with what it has.
    A column that exists but is not in the declared list is NEVER added: that is
    the leak this function exists to prevent.
    """
    label = MARKET_LABEL_COLUMN[market]
    return [c for c in FORWARD_FEATURE_COLUMNS
            if c != label and c in frame.columns
            and pd.api.types.is_numeric_dtype(frame[c])]


def _training_rows(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    label = MARKET_LABEL_COLUMN[market]
    rows = frame[
        frame["position"].isin(MARKET_POSITIONS[market]) & frame[label].notna()
    ]
    return rows.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def _receptions_gate_passes(receptions_calibration: dict | None) -> tuple[bool, str]:
    """The ZINB's own gate, same arithmetic as the quantile gate.

    The receptions count model is scored by its own walk-forward (P(over) at
    the model line, 10 bins, tol 0.05, n>=100), not by the quantile gate --
    the quantile gate is what the count model exists to escape. Returns
    (passed, detail).
    """
    if not receptions_calibration:
        return False, "no receptions calibration report (no bucket met the n>=100 floor)"
    failing = sorted(label for label, bucket in receptions_calibration.items()
                     if not bucket.get("within_tolerance", False))
    if failing:
        return False, f"receptions calibration failed in {failing}"
    return True, ""


def train_all(frame: pd.DataFrame, trained_seasons: list[int], out_dir: Path | str,
              manifest: dict, manifest_path: Path | str,
              walkforward_mae: float | dict[str, float] | None = None,
              walkforward_calibration: dict | None = None,
              receptions_calibration: dict | None = None) -> dict[str, dict]:
    """Fit and persist every market. Returns the artifacts, keyed by market.

    `walkforward_mae` and `walkforward_calibration` are the offline gate's
    verdict, passed in rather than recomputed here. Both are required.

    `walkforward_mae` may be a single number or a `{market: mae}` mapping.
    Per-market is preferred: the pooled mean stamps the same figure on all three
    artifacts and hides that passing yards and receiving yards differ by 3x.

    `receptions_calibration` is the ZINB count model's own walk-forward gate
    verdict, required when `receptions` is in MARKETS. It uses the same bin
    arithmetic as the quantile gate but is scored on the ZINB's P(over) at its
    model line -- the quantile gate cannot score a count model.
    """
    if walkforward_mae is None or walkforward_calibration is None:
        raise ValueError(
            "refusing to write artifacts without a walkforward verdict: the offline "
            "gate decides whether these models may exist (walkforward_mae and "
            "walkforward_calibration are both required)")

    # An empty report means no bucket reached the n>=100 floor, i.e. no evidence
    # at all. `calibration_passes` already treats that as not-a-pass; this must
    # agree, or a run that measured nothing writes artifacts as though it had.
    failing = sorted(label for label, bucket in walkforward_calibration.items()
                     if not bucket.get("within_tolerance", False))
    if not walkforward_calibration or failing:
        detail = "no calibration bucket met the n>=100 floor" if not walkforward_calibration \
            else f"calibration failed in {failing}"
        raise ValueError(f"refusing to write artifacts: {detail}")

    if RECEPTIONS_MARKET in MARKETS:
        passed, detail = _receptions_gate_passes(receptions_calibration)
        if not passed:
            raise ValueError(f"refusing to write receptions artifact: {detail}")

    artifacts: dict[str, dict] = {}
    for market in MARKETS:
        if market in COUNT_MARKETS:
            # Discrete-count market: the ZINB, not the quantile regressor.
            # Its gate was checked above; the artifact is a standalone pickle
            # with its own manifest entry (mirroring `qb_passing_td`), not a
            # quantile artifact.
            rows = _training_rows(frame, market)
            if rows.empty:
                logger.warning("no rows for %s; not writing an artifact", market)
                continue
            fitted = fit_receptions_zinb(rows, rows[MARKET_LABEL_COLUMN[market]])
            path = save_receptions_artifact(fitted, out_dir)
            manifest[RECEPTIONS_MARKET] = receptions_manifest_entry(fitted)
            manifest[RECEPTIONS_MARKET]["path"] = path.name
            manifest[RECEPTIONS_MARKET]["trained_seasons"] = list(trained_seasons)
            manifest[RECEPTIONS_MARKET]["walkforward_calibration"] = receptions_calibration
            Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
            artifacts[market] = fitted
            logger.info("fitted %s ZINB on %d rows", market, fitted["n_train"])
            continue

        rows = _training_rows(frame, market)
        if rows.empty:
            logger.warning("no rows for %s; not writing an artifact", market)
            continue
        feature_cols = _feature_columns(frame, market)
        if not feature_cols:
            logger.warning("no numeric features for %s; not writing an artifact", market)
            continue

        models = fit_yardage_quantile_models(rows[feature_cols], rows[MARKET_LABEL_COLUMN[market]])
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

    # Count-model artifacts (ZINB) are standalone pickles with their own
    # manifest entries, written above -- the quantile registry only handles
    # quantile regressor payloads.
    quantile_artifacts = {m: a for m, a in artifacts.items()
                          if m not in COUNT_MARKETS}
    extended = save_quantile_artifacts(quantile_artifacts, out_dir=out_dir,
                                       manifest=manifest,
                                       manifest_path=manifest_path)

    problems = verify_quantile_artifacts(extended, out_dir=out_dir)
    if problems:
        raise RuntimeError(f"artifacts written but do not verify: {problems}")

    logger.info("wrote %d artifacts and extended the manifest", len(artifacts))
    return artifacts
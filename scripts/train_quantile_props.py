#!/usr/bin/env python
"""train_quantile_props.py -- build the feature frame, walk it forward, report.

    python scripts/train_quantile_props.py --seasons 2017-2026 --out-dir data/cache/nflverse

Writes nothing outside `--out-dir` and the printed report. Task 9 adds the
artifact-writing step; this entrypoint exists so the offline gate can be run and
read before any pickle is committed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.data import season_pull  # noqa: E402
from nfl_predictor.evaluate.walk_forward import (  # noqa: E402
    QUANTILE_MARKETS, calibration_passes, calibration_report, walk_forward_quantile,
)
from nfl_predictor.features.availability import add_availability_features  # noqa: E402
from nfl_predictor.features.matchup import add_matchup_features  # noqa: E402
from nfl_predictor.features.player_usage import PLAYER_FEATURE_COLUMNS, _add_rolling  # noqa: E402

#: Everything the quantile models are fitted on: the six existing rolling usage
#: features plus the Task 4/5 groups.
FEATURE_COLUMNS = [
    *PLAYER_FEATURE_COLUMNS,
    "opp_pass_yds_allowed_roll", "opp_rush_yds_allowed_roll", "opp_rec_yds_allowed_roll",
    "is_home", "rest_days", "implied_team_total", "game_total",
    "wind_kph", "temp_c", "precip_mm", "is_outdoor", "high_wind_flag",
    "inj_Q", "inj_D", "inj_O", "ol_injuries_out", "depth_rank_change",
    "snap_share", "snap_share_trend", "route_participation", "form_deviation",
    "separation_avg",
]


def parse_seasons(text: str) -> list[int]:
    """'2017-2026' or '2019,2020'."""
    if "-" in text:
        start, end = (int(p) for p in text.split("-", 1))
        return list(range(start, end + 1))
    return [int(p) for p in text.split(",") if p.strip()]


def load_weekly(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    """Official weekly where nflverse publishes it, derived from pbp where not.

    nflverse's player_stats release ends at 2024. Deriving the later seasons from
    play-by-play keeps one code path for the training frame instead of branching
    the whole pipeline on which seasons exist.
    """
    frames = []
    for season in seasons:
        official = cache_dir / f"weekly_{season}.parquet"
        if official.exists():
            frames.append(pd.read_parquet(official))
            continue
        pbp_path = cache_dir / f"pbp_{season}.parquet"
        if not pbp_path.exists():
            raise FileNotFoundError(
                f"season {season}: no weekly_{season}.parquet and no pbp_{season}.parquet")
        print(f"  {season}: derived from pbp (nflverse player_stats stops at 2024)")
        frames.append(season_pull.weekly_from_pbp(pd.read_parquet(pbp_path)))
    return pd.concat(frames, ignore_index=True)


def build_feature_frame(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    """Assemble the model frame: weekly labels + usage rolls + matchup + availability."""
    print("loading weekly labels...")
    weekly = load_weekly(seasons, cache_dir)

    print("loading depth charts, injuries, NGS, schedules...")
    rosters = season_pull.pull_rosters(seasons, cache_dir)
    weekly = season_pull.fill_positions(weekly, rosters)

    injuries = season_pull.pull_injuries(seasons, cache_dir)
    schedules = season_pull.pull_schedules(seasons, cache_dir)
    ngs = season_pull.pull_ngs(seasons, cache_dir)

    print("building rolling usage features...")
    frame = _add_rolling(weekly)

    print("building matchup + game-context features...")
    frame = add_matchup_features(frame, schedules)

    print("building availability features (this reads play-by-play)...")
    pbp = pd.concat([pd.read_parquet(cache_dir / f"pbp_{s}.parquet") for s in seasons],
                    ignore_index=True)
    frame = add_availability_features(frame, injuries, rosters, ngs_df=ngs, pbp_df=pbp)

    return frame.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def report(frame: pd.DataFrame, validation_seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    markets = list(QUANTILE_MARKETS)
    scored = walk_forward_quantile(frame, markets=markets, seasons=validation_seasons,
                                   feature_cols=FEATURE_COLUMNS)

    print(f"\nscored {len(scored)} rows across {markets} and seasons {validation_seasons}")
    summary, report_ = [], calibration_report(scored)
    for (market, season), group in scored.dropna(subset=["q50", "actual"]).groupby(["market", "season"]):
        mae = float((group["q50"] - group["actual"]).abs().mean())
        naive_mae = float((group["naive_pred"] - group["actual"]).abs().mean())
        summary.append({"market": market, "season": season, "n": len(group),
                        "mae_q50": mae, "mae_naive": naive_mae, "q50_beats_naive": mae < naive_mae})
    mae_table = pd.DataFrame(summary)
    return mae_table, report_


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2017-2026")
    parser.add_argument("--validate", default="2019-2025")
    parser.add_argument("--out-dir", default="data/cache/nflverse")
    args = parser.parse_args()

    cache_dir = Path(args.out_dir)
    seasons = parse_seasons(args.seasons)
    validation = parse_seasons(args.validate)

    frame = build_feature_frame(seasons, cache_dir)
    print(f"feature frame: {frame.shape[0]} rows, {frame['player_id'].nunique()} players")

    mae_table, buckets = report(frame, validation)
    print("\n=== MAE ===")
    print(mae_table.to_string(index=False))

    print("\n=== calibration (P(over) buckets, n>=100) ===")
    for label, bucket in buckets.items():
        flag = "ok " if bucket["within_tolerance"] else "MISS"
        print(f"{flag} {label}  predicted={bucket['predicted']:.3f} "
              f"empirical={bucket['empirical']:.3f} gap={bucket['gap']:.3f} n={bucket['n']}")

    passed = calibration_passes(buckets)
    print(f"\ncalibration gate: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
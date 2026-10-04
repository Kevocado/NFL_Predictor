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
from nfl_predictor.data.weather_cache import load_cached_weather, weather_for_games  # noqa: E402
from nfl_predictor.evaluate.walk_forward import (  # noqa: E402
    LINE_COLUMN, QUANTILE_MARKETS, calibration_passes, calibration_report,
    walk_forward_quantile,
)

#: The binding gate scores the exogenous cross-sectional line; the own-median
#: curve is reported beside it as the tail-watch diagnostic (amended 2026-10-04).
BINDING_COLUMNS = ("p_over_exogenous", "covered_exogenous")
DIAGNOSTIC_COLUMNS = ("p_over_own_median", "covered_own_median")
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


#: Play-by-play is ~450k rows x 398 columns; the availability features read
#: seven of them. Loading the whole frame to build a snap share is how a
#: walk-forward turns into an out-of-memory kill.
PBP_COLUMNS = ["season", "week", "posteam", "receiver_player_id",
               "rusher_player_id", "passer_player_id", "route"]


def build_weather(schedules: pd.DataFrame, cache_dir: Path, fetch_weather: bool = False) -> dict:
    """Per-game weather, `{game_id: reading}`.

    Off unless asked for: it is one HTTP call per outdoor game (about 2.3k for
    2017-2025, all free), and the matchup builder is NaN-tolerant without it, so
    a training run should not quietly spend ten minutes on the network. The cache
    is on disk either way, so a second run is free.
    """
    weather_dir = cache_dir.parent / "weather"
    games = (schedules.dropna(subset=["gameday"])
             .drop_duplicates(subset=["game_id"])[["game_id", "stadium", "gameday"]]
             .to_dict("records"))
    if not fetch_weather:
        cached = load_cached_weather(weather_dir)
        print(f"weather: {len(cached)} cached readings, not fetching "
              f"(pass --fetch-weather to refresh)")
        return {g: cached[g] for g in (row["game_id"] for row in games) if g in cached}

    print(f"weather: fetching for {len(games)} games (cached ones are free)...")
    fresh, cached = weather_for_games(games, cache_dir=weather_dir)
    combined = {row["game_id"]: cached[row["game_id"]] for row in games
                if row["game_id"] in cached}
    combined.update(fresh)
    print(f"weather: {len(fresh)} fetched, {len(cached)} cached, {len(combined)} usable")
    return combined


def build_feature_frame(seasons: list[int], cache_dir: Path,
                        fetch_weather: bool = False) -> pd.DataFrame:
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
    weather = build_weather(schedules, cache_dir, fetch_weather=fetch_weather)
    frame = add_matchup_features(frame, schedules, weather_by_game=weather)

    print("building availability features (this reads play-by-play)...")
    pbp = pd.concat([pd.read_parquet(cache_dir / f"pbp_{s}.parquet", columns=PBP_COLUMNS)
                     for s in seasons], ignore_index=True)
    frame = add_availability_features(frame, injuries, rosters, ngs_df=ngs, pbp_df=pbp)

    return frame.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def report(frame: pd.DataFrame, validation_seasons: list[int]):
    """MAE table plus both calibration curves."""
    markets = list(QUANTILE_MARKETS)
    scored = walk_forward_quantile(frame, markets=markets, seasons=validation_seasons,
                                   feature_cols=FEATURE_COLUMNS)

    print(f"\nscored {len(scored)} rows across {markets} and seasons {validation_seasons}")
    summary = []
    for (market, season), group in scored.dropna(subset=["q50", "actual"]).groupby(["market", "season"]):
        mae = float((group["q50"] - group["actual"]).abs().mean())
        naive_mae = float((group["naive_pred"] - group["actual"]).abs().mean())
        summary.append({"market": market, "season": season, "n": len(group),
                        "mae_q50": mae, "mae_naive": naive_mae, "q50_beats_naive": mae < naive_mae})
    mae_table = pd.DataFrame(summary)
    return mae_table, scored


def show_curve(label: str, scored: pd.DataFrame, columns) -> dict:
    buckets = calibration_report(scored, columns=columns)
    print(f"\n=== calibration at the {label} line (n>=100) ===")
    for name, bucket in buckets.items():
        flag = "ok " if bucket["within_tolerance"] else "MISS"
        print(f"{flag} {name}  predicted={bucket['predicted']:.3f} "
              f"empirical={bucket['empirical']:.3f} gap={bucket['gap']:.3f} n={bucket['n']}")
    return buckets


def tail_watch(scored: pd.DataFrame, columns=BINDING_COLUMNS) -> pd.DataFrame:
    """Calibration by |line - model median|, the spec's 4-week early tail check.

    The tail is where the model is weakest and where the 5% edge gate fires most,
    so it is watched separately rather than averaged into one verdict.
    """
    frame = scored.dropna(subset=[columns[0], LINE_COLUMN["exogenous"], "q50"]).copy()
    frame["distance"] = (frame[LINE_COLUMN["exogenous"]] - frame["q50"]).abs()
    if frame.empty:
        return frame
    frame["third"] = pd.qcut(frame["distance"], 3, labels=["inner", "middle", "outer"])
    rows = []
    for third, group in frame.groupby("third", observed=True):
        predicted, empirical = group[columns[0]].mean(), group[columns[1]].mean()
        rows.append({"third": third, "n": len(group), "predicted": predicted,
                     "empirical": empirical, "gap": empirical - predicted,
                     "overconfident": (predicted - empirical) > 0.05})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2017-2026")
    parser.add_argument("--validate", default="2019-2025")
    parser.add_argument("--out-dir", default="data/cache/nflverse")
    parser.add_argument("--fetch-weather", action="store_true",
                        help="pull Open-Meteo readings for uncached games (free, ~2.3k calls)")
    args = parser.parse_args()

    cache_dir = Path(args.out_dir)
    seasons = parse_seasons(args.seasons)
    validation = parse_seasons(args.validate)

    frame = build_feature_frame(seasons, cache_dir, fetch_weather=args.fetch_weather)
    print(f"feature frame: {frame.shape[0]} rows, {frame['player_id'].nunique()} players")

    mae_table, scored = report(frame, validation)
    print("\n=== MAE ===")
    print(mae_table.to_string(index=False))

    binding = show_curve("exogenous (BINDING)", scored, BINDING_COLUMNS)
    show_curve("own-median (diagnostic, non-binding)", scored, DIAGNOSTIC_COLUMNS)

    tail = tail_watch(scored)
    if not tail.empty:
        print("\n=== tail watch: calibration by |line - q50| ===")
        print(tail.to_string(index=False))

    passed = calibration_passes(binding)
    print(f"\ncalibration gate (exogenous line): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
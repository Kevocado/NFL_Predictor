#!/usr/bin/env python
"""train_quantile_props.py -- build the feature frame, walk it forward, report.

    python scripts/train_quantile_props.py --seasons 2017-2026 --out-dir data/cache/nflverse

Writes nothing outside `--out-dir` and the printed report. Task 9 adds the
artifact-writing step; this entrypoint exists so the offline gate can be run and
read before any pickle is committed.
"""
from __future__ import annotations

import argparse
import json
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
from nfl_predictor.models import receptions_zinb  # noqa: E402
from nfl_predictor.models.training import (  # noqa: E402
    COUNT_MARKETS, FORWARD_FEATURE_COLUMNS, QUANTILE_FEATURE_COLUMNS, train_all,
)
from nfl_predictor.features.availability import add_availability_features  # noqa: E402
from nfl_predictor.features.interactions import (  # noqa: E402
    add_interaction_features, add_ngs_features, add_pbp_features,
    split_ngs_by_stat_type,
)
from nfl_predictor.features.matchup import add_matchup_features  # noqa: E402
from nfl_predictor.features.player_usage import _add_rolling, _compute_derived_features  # noqa: E402

#: The gate's features come from the SAME constant the artifacts are fitted on.
#: Re-declaring the list here is what let the gate validate 28 features while the
#: artifacts carried 30, with nothing comparing the two.
# Only what the props route serves (see training.QUANTILE_FEATURE_COLUMNS). The frame still BUILDS the full forward set.
FEATURE_COLUMNS = list(QUANTILE_FEATURE_COLUMNS)


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
#: walk-forward turns into an out-of-memory kill. The interaction features
#: (`features/interactions.py`) need the tempo/pressure/target columns below;
#: a season missing one reads it as NaN (same discipline as the absent-column
#: handling in `build_feature_frame`).
PBP_COLUMNS = ["season", "week", "game_id", "posteam", "defteam",
               "home_team", "away_team", "play_id", "pass_attempt",
               "rush_attempt", "down", "goal_to_go", "qb_scramble",
               "air_yards", "sack", "qb_hit", "receiver_player_id",
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


def _games_map_from_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team) -> opponent, from the schedules pull.

    Both sides are pre-kickoff knowable: the interaction features use the
    week-W opponent to fetch the opponent's lagged tempo/pressure, never the
    game's own outcome.
    """
    needed = {"season", "week", "home_team", "away_team"}
    if not needed.issubset(schedules.columns):
        return pd.DataFrame(columns=["season", "week", "team", "opponent"])
    home = schedules.rename(columns={"home_team": "team", "away_team": "opponent"})
    away = schedules.rename(columns={"away_team": "team", "home_team": "opponent"})
    return (pd.concat([home[["season", "week", "team", "opponent"]],
                       away[["season", "week", "team", "opponent"]]],
                      ignore_index=True)
            .drop_duplicates())


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
    # The derived efficiency ratios are part of FORWARD_FEATURE_COLUMNS (the
    # list this script scores and fits against), so they must be computed here
    # too -- `_add_rolling` only produces the base rolling means. Without this
    # call the frame lacks the four ratio columns and `walk_forward_quantile`
    # raises KeyError when it selects `feature_cols`.
    frame = _compute_derived_features(frame)

    print("building matchup + game-context features...")
    weather = build_weather(schedules, cache_dir, fetch_weather=fetch_weather)
    frame = add_matchup_features(frame, schedules, weather_by_game=weather)

    print("building availability features (this reads play-by-play)...")
    # Checked up front: a missing season file used to raise here, AFTER the
    # weekly, roster, injury, schedule, NGS and weather steps had all run.
    missing = [s for s in seasons if not (cache_dir / f"pbp_{s}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"no cached play-by-play for {missing} in {cache_dir}; run "
            f"season_pull.pull_pbp first")
    # Read only the columns a season actually has. nflverse DROPPED `route` from
    # the 2026 file while 2017-2025 all carry it, so a fixed column list crashes
    # the whole run on the current season. A missing column becomes NaN, which is
    # what an absent observation is; the alternative -- skipping the season --
    # would silently train on a stale era.
    import pyarrow.parquet as _pq

    frames = []
    for season in seasons:
        path = cache_dir / f"pbp_{season}.parquet"
        available = set(_pq.ParquetFile(path).schema_arrow.names)
        columns = [c for c in PBP_COLUMNS if c in available]
        absent = [c for c in PBP_COLUMNS if c not in available]
        if absent:
            print(f"  {season}: no {absent} in play-by-play; those columns are NaN")
        frames.append(pd.read_parquet(path, columns=columns).assign(
            **{c: float("nan") for c in absent}))
    pbp = pd.concat(frames, ignore_index=True)
    frame = add_availability_features(frame, injuries, rosters, ngs_df=ngs, pbp_df=pbp)

    print("building interaction features (products of independent signals)...")
    # Order matters: add_interaction_features reads `high_wind_flag` (from
    # add_matchup_features' Open-Meteo forecast path) and `ol_injuries_out`
    # (from add_availability_features) off the frame, so this runs last.
    frame = add_ngs_features(frame, split_ngs_by_stat_type(ngs))
    frame = add_pbp_features(frame, pbp, _games_map_from_schedules(schedules))
    frame = add_interaction_features(frame, injuries)

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


#: The receptions ZINB's walk-forward folds, as in the Phase 4 experiment (train 2017-2021 -> score 2022, ...). The
#: model needs >= `min_train_seasons` (5) prior seasons, so 2022 is the first scoreable fold of a 2017+ frame.
DEFAULT_RECEPTIONS_VALIDATE = "2022-2024"


def receptions_gate(frame: pd.DataFrame, validation_seasons: list[int]) -> tuple[dict, pd.DataFrame]:
    """The ZINB count model's own walk-forward gate (step 2 of docs/PHASE4: run BEFORE --write-artifacts).

    Scored at the model's own line (P(over) at `model_line(mu)`), binned with the repo's `calibration_report`
    arithmetic: 10 bins, tol 0.05, n >= 100. Returns (gate_dict, scored). An empty gate dict means no bucket met the
    n >= 100 floor, which `train_all` treats as a FAIL, not a pass.
    """
    scored = receptions_zinb.walk_forward_receptions(frame, validation_seasons=validation_seasons)
    if scored.empty:
        return {}, scored
    return receptions_zinb.receptions_calibration_gate(scored), scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2017-2026")
    parser.add_argument("--validate", default="2019-2025")
    parser.add_argument("--out-dir", default="data/cache/nflverse")
    parser.add_argument("--fetch-weather", action="store_true",
                        help="pull Open-Meteo readings for uncached games (free, ~2.3k calls)")
    parser.add_argument("--train-seasons", default=None,
                        help="seasons to FIT on for the artifacts (default: --seasons)")
    parser.add_argument("--models-dir", default="models",
                        help="where to write *_quantile_2025.pkl")
    parser.add_argument("--dump-feature-frame", default=None,
                        help="write the assembled feature frame here; "
                             "scripts/forward_tick.py --feature-frame consumes it")
    parser.add_argument("--receptions-validate", default=DEFAULT_RECEPTIONS_VALIDATE,
                        help="validation seasons for the receptions ZINB gate (needs >= 5 prior seasons each)")
    parser.add_argument("--write-artifacts", action="store_true",
                        help="fit on --train-seasons and write the versioned artifacts; "
                             "requires the walk-forward gate to PASS")
    args = parser.parse_args()

    cache_dir = Path(args.out_dir)
    seasons = parse_seasons(args.seasons)
    validation = parse_seasons(args.validate)

    frame = build_feature_frame(seasons, cache_dir, fetch_weather=args.fetch_weather)
    print(f"feature frame: {frame.shape[0]} rows, {frame['player_id'].nunique()} players")

    if args.dump_feature_frame:
        Path(args.dump_feature_frame).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.dump_feature_frame, index=False)
        print(f"feature frame written to {args.dump_feature_frame}")

    mae_table, scored = report(frame, validation)
    print("\n=== MAE ===")
    print(mae_table.to_string(index=False))

    binding = show_curve("exogenous (BINDING)", scored, BINDING_COLUMNS)
    show_curve("own-median (diagnostic, non-binding)", scored, DIAGNOSTIC_COLUMNS)

    tail = tail_watch(scored)
    if not tail.empty:
        print("\n=== tail watch: calibration by |line - q50| ===")
        print(tail.to_string(index=False))

    quantile_passed = calibration_passes(binding)
    print(f"\ncalibration gate (exogenous line): {'PASS' if quantile_passed else 'FAIL'}")

    # Receptions is a COUNT market: the quantile gate above cannot score it (that is why QUANTILE_MARKETS excludes
    # COUNT_MARKETS), so the ZINB is gated by its own walk-forward. Both gates must pass before anything is written.
    receptions_seasons = parse_seasons(args.receptions_validate)
    print(f"\nreceptions ZINB walk-forward on seasons {receptions_seasons}...")
    receptions_calibration, receptions_scored = receptions_gate(frame, receptions_seasons)
    print(f"scored {len(receptions_scored)} receptions rows")
    if receptions_scored.empty:
        print("no receptions rows scored (not enough prior seasons?)")
    else:
        show_curve("receptions ZINB (model line, BINDING)", receptions_scored, ("p_over", "covered"))
    receptions_passed = calibration_passes(receptions_calibration)
    print(f"\nreceptions ZINB gate: {'PASS' if receptions_passed else 'FAIL'}")

    passed = quantile_passed and receptions_passed
    print(f"\nBOTH GATES: {'PASS' if passed else 'FAIL'} "
          f"(quantile={'PASS' if quantile_passed else 'FAIL'}, receptions={'PASS' if receptions_passed else 'FAIL'})")

    if not args.write_artifacts:
        print("\n(dry run: pass --write-artifacts to fit and write)")
        return 0 if passed else 1

    if not passed:
        failed = [name for name, ok in (("quantile", quantile_passed), ("receptions", receptions_passed)) if not ok]
        print(f"refusing to write artifacts: the {' and '.join(failed)} calibration gate failed. "
              "Tolerances are NOT lowered to force a write; diagnose the failing buckets above.")
        return 1

    train_seasons = parse_seasons(args.train_seasons) if args.train_seasons else seasons
    models_dir = Path(args.models_dir)
    manifest_path = models_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    # Per-market MAE: the pooled mean would record the same number against all
    # three artifacts, hiding that passing (69.7) and receiving (21.2) differ.
    per_market_mae = (mae_table.groupby("market")["mae_q50"].mean().to_dict()
                      if not mae_table.empty else {})

    artifacts = train_all(
        frame=frame, trained_seasons=train_seasons, out_dir=models_dir,
        manifest=manifest, manifest_path=manifest_path,
        walkforward_mae=per_market_mae,
        walkforward_calibration=binding,
        receptions_calibration=receptions_calibration,
    )
    print(f"\nwrote {len(artifacts)} artifacts to {models_dir}/ and extended the manifest")
    for market in sorted(artifacts):
        if market in COUNT_MARKETS:
            fitted = artifacts[market]
            print(f"  {receptions_zinb.RECEPTIONS_ZINB_FILENAME}  ZINB, {len(fitted['feature_cols'])} features, "
                  f"n_train={fitted['n_train']}, alpha={fitted['alpha']:.2f}")
            continue
        mae = per_market_mae.get(market)
        mae_text = "n/a" if mae is None else f"MAE {mae:.1f}"
        print(f"  {market}_quantile_2025.pkl  {len(artifacts[market]['feature_cols'])} "
              f"features, {len(artifacts[market]['quantile_models'])} quantiles, {mae_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
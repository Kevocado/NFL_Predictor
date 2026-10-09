"""Decide whether a feature block earns its place. Paired bootstrap on identical held-out games."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_predictor.evaluate import walk_forward as wf
from nfl_predictor.models import game_outcome
from nfl_predictor.data import schedules
from nfl_predictor.data.pbp_agg import load_pbp_agg, team_game_efficiency, qb_games
from nfl_predictor.config import DEPTH_CHARTS_CACHE_DIR
from nfl_predictor.features import build as feature_build


def paired_bootstrap(a, b, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of mean(a - b). Positive means b is better when lower is better."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def calibration_gap(probs, y, buckets: int = 5) -> float:
    probs, y = np.asarray(probs, float), np.asarray(y, float)
    edges = np.quantile(probs, np.linspace(0, 1, buckets + 1))
    idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, buckets - 1)
    gaps = [abs(probs[idx == b].mean() - y[idx == b].mean()) for b in range(buckets) if (idx == b).any()]
    return float(max(gaps)) if gaps else 0.0


def _per_game(folds, candidate):
    errs, briers, probs, ys = [], [], [], []
    for fold in folds:
        preds, sigma = wf._predict_margins(candidate, fold["train_df"], fold["val_df"], fold["feature_cols"])
        margin = fold["val_df"]["margin"].to_numpy()
        p = np.array([wf.game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds])
        errs.append(np.abs(preds - margin))
        y = (margin > 0).astype(float)
        briers.append((p - y) ** 2)
        probs.append(p)
        ys.append(y)
    return tuple(np.concatenate(x) for x in (errs, briers, probs, ys))


def _per_game_totals(folds, candidate):
    """Held-out |predicted total - actual total| per game. The conditions block claims
    to help TOTALS, so its metric has to be total MAE, not spread MAE."""
    errs = []
    for fold in folds:
        preds = wf._fit_predict(candidate, fold["train_df"], fold["val_df"], fold["feature_cols"], "total_points")
        total = fold["val_df"]["total_points"].to_numpy(float)
        errs.append(np.abs(preds - total))
    return np.concatenate(errs) if errs else np.array([])


def paired_bootstrap_gap(per_fold_gaps_base: list[float], per_fold_gaps_block: list[float],
                         n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of mean(gap_base - gap_block) paired across folds.

    Positive means the block narrows the calibration gap. Comparing two scalars
    (0.0287 vs 0.0288) says nothing about whether the narrowing is real, so the
    difference gets its own interval on the paired per-fold gaps.
    """
    d = np.asarray(per_fold_gaps_base, float) - np.asarray(per_fold_gaps_block, float)
    if d.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, d.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _per_fold_gap(folds, candidate) -> list[float]:
    """Calibration gap computed within each fold's held-out games, so gaps can be paired."""
    gaps = []
    for fold in folds:
        preds, sigma = wf._predict_margins(candidate, fold["train_df"], fold["val_df"], fold["feature_cols"])
        margin = fold["val_df"]["margin"].to_numpy()
        p = np.array([wf.game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds])
        gaps.append(calibration_gap(p, (margin > 0).astype(float)))
    return gaps


def evaluate_block(games_df, aux, block: str, candidate: str = "ridge") -> dict:
    base = _per_game(wf.prepare_folds(games_df), candidate)
    withb = _per_game(wf.prepare_folds(games_df, blocks=(block,), aux=aux), candidate)
    assert len(base[0]) == len(withb[0]), f"block {block} drops games: {len(base[0])} vs {len(withb[0])}"
    mae_ci, brier_ci = paired_bootstrap(base[0], withb[0]), paired_bootstrap(base[1], withb[1])
    gap_base, gap_block = calibration_gap(base[2], base[3]), calibration_gap(withb[2], withb[3])
    # The calibration gap is decided on a paired interval of the DIFFERENCE, not by
    # eyeballing two scalars: 0.0287 vs 0.0288 is indistinguishable, 0.0547 vs 0.0186 is not.
    gap_ci = paired_bootstrap_gap(_per_fold_gap(wf.prepare_folds(games_df), candidate),
                                  _per_fold_gap(wf.prepare_folds(games_df, blocks=(block,), aux=aux), candidate))
    # The conditions block is claimed for TOTALS, so its headline metric is total MAE.
    total_mae = None
    if block == "conditions":
        tb = _per_game_totals(wf.prepare_folds(games_df), candidate)
        tw = _per_game_totals(wf.prepare_folds(games_df, blocks=(block,), aux=aux), candidate)
        assert len(tb) == len(tw), f"block {block} drops games on totals: {len(tb)} vs {len(tw)}"
        tci = paired_bootstrap(tb, tw)
        total_mae = {"delta": float(tb.mean() - tw.mean()), "ci": tci, "clears": bool(tci[0] > 0)}
    return {
        "block": block, "n_games": int(len(base[0])),
        "mae_delta": float(base[0].mean() - withb[0].mean()), "mae_ci": mae_ci,
        "brier_delta": float(base[1].mean() - withb[1].mean()), "brier_ci": brier_ci,
        "gap_base": gap_base, "gap_block": gap_block, "gap_ci": gap_ci,
        "total_mae": total_mae,
        "clears": bool(mae_ci[0] > 0 and brier_ci[0] > 0 and gap_ci[0] > 0),
    }

def load_recent_nfl_data(start_year: int = 2000, end_year: int = None) -> tuple[pd.DataFrame, feature_build.Aux]:
    """Games plus the aux the blocks read (efficiency / qb_games from play-by-play), cached per season."""
    if end_year is None:
        end_year = datetime.now().year
    seasons = list(range(start_year, end_year + 1))

    print(f"Loading NFL data for seasons {seasons}...")
    games_df = schedules.load_training_data(seasons)

    efficiency_frames, qb_frames = [], []
    loaded = []
    for season in seasons:
        try:
            pbp = load_pbp_agg(season)
        except Exception as e:
            print(f"Warning: could not load PBP for {season}: {e}")
            continue
        if pbp is None or pbp.empty:
            print(f"Warning: no PBP rows for {season}")
            continue
        efficiency_frames.append(team_game_efficiency(pbp))
        qb_frames.append(qb_games(pbp))
        loaded.append(season)

    efficiency = pd.concat(efficiency_frames, ignore_index=True) if efficiency_frames else None
    qb_games_df = pd.concat(qb_frames, ignore_index=True) if qb_frames else None
    missing = sorted(set(seasons) - set(loaded))
    if missing:
        print(f"Warning: no efficiency/qb_games for seasons {missing} -- "
              f"those games' epa/qb features are NaN, which drags the block down on them alone")
    aux = feature_build.Aux(efficiency=efficiency, qb_games=qb_games_df, upcoming_starters={})
    return games_df, aux


def serving_realistic_expected_starters(season_range, cache_dir=DEPTH_CHARTS_CACHE_DIR) -> dict:
    """(game_id, team) -> depth-chart expected starting QB, for every regular-season week in range.

    This is ALL a serving path may use: the chart is published pre-game, so it never sees who
    actually started. It is fed to the qb block as `upcoming_starters`.
    """
    from nfl_predictor.data import depth_charts as _dc

    out = {}
    for season in season_range:
        for week in range(1, 19):
            try:
                out.update(_expected_starters_for_week(season, week, cache_dir))
            except Exception:
                continue
    return out


def _expected_starters_for_week(season: int, week: int, cache_dir) -> dict:
    """One week's expected starters: the depth_team=='1' QB per club, resolved to that week's game."""
    from nfl_predictor.data import depth_charts as _dc

    chart_full = _dc.load_depth_charts(season, cache_dir)
    if chart_full is None or chart_full.empty:
        return {}
    if "season" not in chart_full.columns:
        chart_full = chart_full.copy()
        chart_full["season"] = season
    chart = _dc.resolve_chart(chart_full, season, week)
    if chart is None or chart.empty:
        return {}

    try:
        week_games = schedules.fetch_week_games(season, week)
    except Exception:
        week_games = schedules.fetch_schedules([season])
        week_games = week_games[week_games["week"] == week]
    if week_games.empty:
        return {}
    club_to_game = {}
    for _, row in week_games.iterrows():
        club_to_game[row["home_team"]] = (row["game_id"], row["home_team"])
        club_to_game[row["away_team"]] = (row["game_id"], row["away_team"])

    qbs = chart[chart["depth_position"] == "QB"].copy()
    if qbs.empty:
        return {}
    qbs["is_starter"] = qbs["depth_team"].astype(str) == _dc.STARTER_DEPTH_TEAM
    starters = qbs[qbs["is_starter"]].drop_duplicates(subset=["club_code"], keep="first")
    expected = {}
    for _, row in starters.iterrows():
        key = club_to_game.get(row["club_code"])
        if key is not None:
            expected[key] = row["gsis_id"]
    return expected


def with_serving_realistic_qb_games(aux, expected_starters: dict):
    """An aux whose `qb_games` names the EXPECTED starter per game, not the actual one.

    Training sees who really started; serving only knows the chart. Slicing qb_games to the
    expected starter's own rows makes the block's rollups read like a serving path would.
    Returns the aux unchanged when there is nothing to filter.
    """
    if aux is None or getattr(aux, "qb_games", None) is None or not expected_starters:
        return aux
    qb_games_df = aux.qb_games
    if not {"game_id", "team", "qb_id"}.issubset(qb_games_df.columns):
        return aux
    want = set(expected_starters.items())
    mask = qb_games_df.apply(
        lambda r: ((r["game_id"], r["team"]), r["qb_id"]) in want, axis=1
    )
    return feature_build.Aux(
        efficiency=getattr(aux, "efficiency", None),
        qb_games=qb_games_df[mask].reset_index(drop=True),
        upcoming_starters=dict(expected_starters),
    )


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NFL feature blocks")
    parser.add_argument("--blocks", nargs="+", required=True, help="e.g. epa qb conditions")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--candidate", type=str, default="ridge")
    parser.add_argument(
        "--serving-realistic-qb", action="store_true",
        help="evaluate the qb block using ONLY depth-chart expected starters, the information a serving path has",
    )
    args = parser.parse_args()

    games_df, aux = load_recent_nfl_data(args.start_year, args.end_year)
    if args.serving_realistic_qb:
        expected = serving_realistic_expected_starters(
            sorted(games_df["season"].astype(int).unique())
        )
        aux = with_serving_realistic_qb_games(aux, expected)
        print(f"SERVING-REALISTIC QB: {len(expected)} games carry an expected starter from the depth chart")
    if games_df.empty:
        print("Error: no games loaded")
        raise SystemExit(1)

    print(f"Loaded {len(games_df)} games. Evaluating blocks: {', '.join(args.blocks)} (candidate={args.candidate})\n")

    results = []
    for block in args.blocks:
        print(f"Evaluating block: {block}...")
        try:
            results.append(evaluate_block(games_df, aux, block, args.candidate))
        except Exception as e:
            print(f"Error evaluating block {block}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 96)
    print("NFL BLOCK EVALUATION RESULTS")
    print("=" * 96)
    hdr = (f"{'Block':<11} {'N':>5} {'MAE d':>9} {'MAE 95% CI':>21} {'Brier d':>9} {'Brier 95% CI':>21} "
           f"{'GapB':>7} {'GapBlk':>7} {'Gap d CI':>21} {'Clears':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r is None:
            print(f"{'ERROR':<11}")
            continue
        print(f"{r['block']:<11} {r['n_games']:>5} {r['mae_delta']:>9.4f} {_fmt_ci(r['mae_ci']):>21} "
              f"{r['brier_delta']:>9.4f} {_fmt_ci(r['brier_ci']):>21} "
              f"{r['gap_base']:>7.4f} {r['gap_block']:>7.4f} {_fmt_ci(r['gap_ci']):>21} "
              f"{'Yes' if r['clears'] else 'No':>6}")
        if r["total_mae"] is not None:
            tm = r["total_mae"]
            print(f"{'  ^ total MAE':<11} {'':>5} {tm['delta']:>9.4f} {_fmt_ci(tm['ci']):>21} "
                  f"{'(positive = block better on TOTALS)':>9}")
    print("=" * 96)
    print("Sign convention: delta = base - block, so POSITIVE means the block is better (MAE/Brier lower).")


if __name__ == "__main__":
    main()

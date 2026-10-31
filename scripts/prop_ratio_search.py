#!/usr/bin/env python
"""prop_ratio_search.py -- pre-registered ratio-feature search for player props.

The frozen procedure is docs/prop-ratio-search-prereg.md. Two phases:

    python scripts/prop_ratio_search.py select --cache-dir <nflverse cache>
    python scripts/prop_ratio_search.py holdout

`select` builds the frame (reusing scripts/train_quantile_props.py), builds the
candidate ratios, scores each one walk-forward on the SELECTION seasons, and writes
docs/prop-ratio-search-selection.json. `holdout` reads that file, scores the kept set
once on the HOLDOUT seasons, and writes docs/prop-ratio-search-holdout.json. It
refuses to run without the selection file and refuses to overwrite its own output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_quantile_props as tqp  # noqa: E402
from nfl_predictor.features.player_usage import with_passing_tds_roll  # noqa: E402
from nfl_predictor.models.player_props import fit_yardage_quantile_models  # noqa: E402
from nfl_predictor.models.training import QUANTILE_FEATURE_COLUMNS  # noqa: E402

SELECTION_SEASONS = list(range(2019, 2025))
HOLDOUT_SEASONS = [2025, 2026]
WINDOW = 5
SEED = 20261030
BOOTSTRAP_B = 2000
ALPHA = 0.05
KEEP_PER_MARKET = 3
STABILITY_MIN = 5

#: market -> (positions, label column). Pre-registered; see the prereg doc.
MARKETS: dict[str, tuple[tuple[str, ...], str]] = {
    "passing_yards": (("QB",), "passing_yards"),
    "rushing_yards": (("RB",), "rushing_yards"),
    "receiving_yards": (("WR", "TE"), "receiving_yards"),
    "receptions": (("WR", "TE"), "receptions"),
    "passing_tds": (("QB",), "passing_tds"),
}

#: Base rolling stats for the pairwise pool (pre-registered, 16 entries).
BASES = ["targets", "receptions", "receiving_yards", "carries", "rushing_yards",
         "passing_yards", "attempts", "completions", "air_yards", "rz_targets",
         "rz_carries", "snap_share", "team_pass_att", "team_plays", "team_air_yards",
         "team_rush_att"]

#: Pairs (num, den) that equal a named ratio; dropped from the pairwise pool so each
#: definition is counted once.
NAMED_PAIR_KEYS = {("targets", "team_pass_att"), ("air_yards", "team_air_yards"),
                   ("receiving_yards", "targets"), ("carries", "team_rush_att"),
                   ("rushing_yards", "carries"), ("receptions", "targets"),
                   ("team_pass_att", "team_plays")}

PBP_COLUMNS = ["season", "week", "posteam", "pass_attempt", "rush_attempt", "air_yards",
               "yardline_100", "receiver_player_id", "rusher_player_id"]


# --------------------------------------------------------------------- pure helpers

def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def lagged_roll(df: pd.DataFrame, key: str, col: str, window: int = WINDOW) -> pd.Series:
    """Prior-`window` mean within `key`, excluding the current row (shift(1) first)."""
    return df.groupby(key)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den with zero or non-finite denominators turned into NaN."""
    out = num / den.where(den != 0)
    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------- data

def load_pbp(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    frames = []
    for season in seasons:
        path = cache_dir / f"pbp_{season}.parquet"
        available = set(pq.ParquetFile(path).schema_arrow.names)
        frames.append(pd.read_parquet(path, columns=[c for c in PBP_COLUMNS if c in available]))
    pbp = pd.concat(frames, ignore_index=True)
    pbp["pass_attempt"] = pbp["pass_attempt"].fillna(0)
    pbp["rush_attempt"] = pbp["rush_attempt"].fillna(0)
    pbp["rz"] = (pbp["yardline_100"] <= 20).fillna(False).astype(int)
    return pbp


def player_game_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    targets = pbp[(pbp.pass_attempt == 1) & pbp.receiver_player_id.notna()]
    targets = (targets.groupby(["season", "week", "receiver_player_id"])
               .agg(pbp_air_yards=("air_yards", "sum"), pbp_rz_targets=("rz", "sum"))
               .reset_index().rename(columns={"receiver_player_id": "player_id"}))
    carries = pbp[(pbp.rush_attempt == 1) & pbp.rusher_player_id.notna()]
    carries = (carries.groupby(["season", "week", "rusher_player_id"])
               .agg(pbp_rz_carries=("rz", "sum"))
               .reset_index().rename(columns={"rusher_player_id": "player_id"}))
    return targets.merge(carries, on=["season", "week", "player_id"], how="outer")


def team_game_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp.posteam.notna()].copy()
    p["pass_play"] = p["pass_attempt"]
    p["rush_play"] = p["rush_attempt"]
    p["plays"] = ((p.pass_play + p.rush_play) > 0).astype(int)
    p["air_pass"] = np.where(p.pass_play == 1, p["air_yards"].fillna(0), 0.0)
    p["rz_pass"] = p.pass_play * p.rz
    p["rz_rush"] = p.rush_play * p.rz
    team = (p.groupby(["season", "week", "posteam"])
            .agg(team_pass_att=("pass_play", "sum"), team_rush_att=("rush_play", "sum"),
                 team_plays=("plays", "sum"), team_air_yards=("air_pass", "sum"),
                 team_rz_pass_att=("rz_pass", "sum"), team_rz_rush_att=("rz_rush", "sum"))
            .reset_index().rename(columns={"posteam": "recent_team"}))
    return team.sort_values(["recent_team", "season", "week"]).reset_index(drop=True)


def build_frame(cache_dir: Path, seasons: list[int]) -> pd.DataFrame:
    """The production feature frame plus the lagged candidate columns (`c__<name>`)."""
    frame = tqp.build_feature_frame(seasons, cache_dir, fetch_weather=False)
    if "passing_tds_roll" not in frame.columns:
        frame = with_passing_tds_roll(frame)
    frame = frame.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    pbp = load_pbp(seasons, cache_dir)
    pg = player_game_pbp(pbp)
    frame = frame.merge(pg, on=["season", "week", "player_id"], how="left")

    # Rolls are taken on a 0-filled COPY. The raw columns stay as they are: they are
    # the regression labels, and filling them would create new training rows.
    player_sources = {"air_yards": "pbp_air_yards", "rz_targets": "pbp_rz_targets",
                      "rz_carries": "pbp_rz_carries", "targets": "targets",
                      "receptions": "receptions", "receiving_yards": "receiving_yards",
                      "carries": "carries", "rushing_yards": "rushing_yards",
                      "passing_yards": "passing_yards", "attempts": "attempts",
                      "completions": "completions"}
    for out, src in player_sources.items():
        filled = frame.assign(_v=frame[src].fillna(0))
        frame[f"r_{out}"] = lagged_roll(filled, "player_id", "_v")

    team = team_game_pbp(pbp)
    team_cols = ["team_pass_att", "team_rush_att", "team_plays", "team_air_yards",
                 "team_rz_pass_att", "team_rz_rush_att"]
    for col in team_cols:
        team[f"r_{col}"] = lagged_roll(team, "recent_team", col)
    frame = frame.merge(team[["season", "week", "recent_team"] + [f"r_{c}" for c in team_cols]],
                        on=["season", "week", "recent_team"], how="left")
    # snap_share is already lagged in the frame (availability.py); the candidate uses it as is.
    frame["r_snap_share"] = frame["snap_share"]
    return frame


def candidate_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Named ratios (16) plus pairwise ratios of the base set. Returns name -> Series."""
    r = {b: frame[f"r_{b}"] for b in BASES}
    r.update({"team_rz_pass_att": frame["r_team_rz_pass_att"],
              "team_rz_rush_att": frame["r_team_rz_rush_att"]})
    opp_pass = frame["opp_pass_yds_allowed_roll"]
    opp_rush = frame["opp_rush_yds_allowed_roll"]
    opp_rec = frame["opp_rec_yds_allowed_roll"]
    named = {
        "target_share": safe_ratio(r["targets"], r["team_pass_att"]),
        "air_yards_share": safe_ratio(r["air_yards"], r["team_air_yards"]),
        "yards_per_target": safe_ratio(r["receiving_yards"], r["targets"]),
        "yards_per_touch": safe_ratio(r["receiving_yards"] + r["rushing_yards"],
                                      r["targets"] + r["carries"]),
        "carries_share": safe_ratio(r["carries"], r["team_rush_att"]),
        "yards_per_carry": safe_ratio(r["rushing_yards"], r["carries"]),
        "rz_target_share": safe_ratio(r["rz_targets"], r["team_rz_pass_att"]),
        "rz_carry_share": safe_ratio(r["rz_carries"], r["team_rz_rush_att"]),
        "catch_rate": safe_ratio(r["receptions"], r["targets"]),
        "team_pass_rate": safe_ratio(r["team_pass_att"], r["team_plays"]),
        "passing_vs_opp": safe_ratio(r["passing_yards"], opp_pass),
        "rushing_vs_opp": safe_ratio(r["rushing_yards"], opp_rush),
        "receiving_vs_opp": safe_ratio(r["receiving_yards"], opp_rec),
        "opp_rec_over_pass": safe_ratio(opp_rec, opp_pass),
        "opp_rush_over_pass": safe_ratio(opp_rush, opp_pass),
        "opp_rec_over_rush": safe_ratio(opp_rec, opp_rush),
    }
    out = {f"named__{k}": v for k, v in named.items()}
    for a in BASES:
        for b in BASES:
            if a == b or (a, b) in NAMED_PAIR_KEYS:
                continue
            out[f"pair__{a}__over__{b}"] = safe_ratio(r[a], r[b])
    return out


# --------------------------------------------------------------------- scoring

def baseline_cols(market: str) -> list[str]:
    label = MARKETS[market][1]
    cols = [c for c in QUANTILE_FEATURE_COLUMNS if c != label]
    if market == "passing_tds":
        cols.append("passing_tds_roll")
    return cols


def market_rows(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    positions, label = MARKETS[market]
    sub = frame[frame["position"].isin(positions) & frame[label].notna()]
    return sub.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame) -> np.ndarray:
    """The production q50 model (reg:quantileerror, alpha 0.5), fitted and predicted."""
    models = fit_yardage_quantile_models(X_train, y_train, quantiles=[0.5])
    return models[0.5].predict(X_val.fillna(0))


def paired_test(d: np.ndarray) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    if len(d) < 2 or np.allclose(d, d[0]):
        return float(d.mean()) if len(d) else float("nan"), 1.0
    t_stat, p = stats.ttest_1samp(d, 0.0, alternative="greater")
    return float(d.mean()), float(p)


def run_select(frame: pd.DataFrame, cand_names: list[str], log) -> dict:
    results: dict = {"markets": {}, "candidates": cand_names}
    per_test = []  # (market, candidate) entries for the family-wide Holm correction
    for market in MARKETS:
        sub = market_rows(frame, market)
        base = baseline_cols(market)
        label = MARKETS[market][1]
        y = sub[label]
        pooled = {c: [] for c in cand_names}
        by_season = {c: {} for c in cand_names}
        t0 = time.time()
        for season in SELECTION_SEASONS:
            tr = (sub["season"] < season).to_numpy()
            va = (sub["season"] == season).to_numpy()
            if tr.sum() == 0 or va.sum() == 0:
                continue
            y_tr, y_va = y[tr], y[va].to_numpy()
            e0 = np.abs(fit_predict(sub.loc[tr, base], y_tr, sub.loc[va, base]) - y_va)
            for i, c in enumerate(cand_names):
                cols = base + [c]
                ec = np.abs(fit_predict(sub.loc[tr, cols], y_tr, sub.loc[va, cols]) - y_va)
                pooled[c].append(e0 - ec)
                by_season[c][season] = float(e0.mean() - ec.mean())
            log(f"  {market} season {season} done ({len(cand_names)} candidates, "
                f"{time.time() - t0:.0f}s)")
        market_out = {}
        for c in cand_names:
            d = np.concatenate(pooled[c]) if pooled[c] else np.array([])
            mean_gain, p = paired_test(d)
            positive = sum(1 for v in by_season[c].values() if v > 0)
            market_out[c] = {"mean_mae_gain": mean_gain, "p_one_sided": p,
                             "season_gains": by_season[c], "positive_seasons": positive,
                             "n_rows": int(len(d))}
            per_test.append((market, c))
        results["markets"][market] = market_out
    p_values = [results["markets"][m][c]["p_one_sided"] for m, c in per_test]
    holm = holm_adjust(p_values)
    for (m, c), h in zip(per_test, holm):
        entry = results["markets"][m][c]
        entry["holm_p"] = float(h)
        entry["stable"] = entry["positive_seasons"] >= STABILITY_MIN
        entry["passes"] = bool(h < ALPHA and entry["stable"] and entry["mean_mae_gain"] > 0)
    selected = []
    for market in MARKETS:
        passing = [(results["markets"][market][c]["holm_p"], c) for c in cand_names
                   if results["markets"][market][c]["passes"]]
        for _, c in sorted(passing)[:KEEP_PER_MARKET]:
            results["markets"][market][c]["selected"] = True
            selected.append({"market": market, "candidate": c})
    results["selected"] = selected
    results["family_size"] = len(per_test)
    return results


def run_holdout(frame: pd.DataFrame, selected: list[dict], log) -> dict:
    unique = sorted({s["candidate"] for s in selected})
    records = {}  # (market, candidate) -> dict of arrays
    for market in MARKETS:
        sub = market_rows(frame, market)
        base = baseline_cols(market)
        y = sub[MARKETS[market][1]]
        for c in unique:
            e0_parts, ec_parts, player_parts, season_parts = [], [], [], []
            for season in HOLDOUT_SEASONS:
                tr = (sub["season"] < season).to_numpy()
                va = (sub["season"] == season).to_numpy()
                if tr.sum() == 0 or va.sum() == 0:
                    continue
                y_tr, y_va = y[tr], y[va].to_numpy()
                e0 = np.abs(fit_predict(sub.loc[tr, base], y_tr, sub.loc[va, base]) - y_va)
                cols = base + [c]
                ec = np.abs(fit_predict(sub.loc[tr, cols], y_tr, sub.loc[va, cols]) - y_va)
                e0_parts.append(e0)
                ec_parts.append(ec)
                player_parts.append(sub.loc[va, "player_id"].astype(str).to_numpy())
                season_parts.append(np.full(va.sum(), season))
            records[(market, c)] = {
                "d": np.concatenate(e0_parts) - np.concatenate(ec_parts),
                "players": np.concatenate(player_parts),
                "seasons": np.concatenate(season_parts),
            }
        log(f"  holdout {market} done")

    k = len(selected)
    alpha_k = ALPHA / k
    rng = np.random.default_rng(SEED)
    out_pairs = []
    for s in selected:
        m, c = s["market"], s["candidate"]
        rec = records[(m, c)]
        d = rec["d"]
        codes, uniques = pd.factorize(rec["players"])
        n_players = len(uniques)
        sum_p = np.bincount(codes, weights=d, minlength=n_players)
        n_p = np.bincount(codes, minlength=n_players).astype(float)
        boots = np.empty(BOOTSTRAP_B)
        for b in range(BOOTSTRAP_B):
            draw = np.bincount(rng.integers(0, n_players, n_players), minlength=n_players)
            boots[b] = (draw @ sum_p) / (draw @ n_p)
        lo, hi = np.percentile(boots, [100 * alpha_k / 2, 100 * (1 - alpha_k / 2)])
        others = {}
        for m2 in MARKETS:
            if m2 == m:
                continue
            d2 = records[(m2, c)]["d"]
            others[m2] = {"mae_gain": float(d2.mean()), "n_rows": int(len(d2))}
        season_gains = {}
        for season in HOLDOUT_SEASONS:
            mask = rec["seasons"] == season
            if mask.any():
                season_gains[str(season)] = float(d[mask].mean())
        out_pairs.append({
            "market": m, "candidate": c, "n_rows": int(len(d)),
            "mae_gain": float(d.mean()), "ci_level": 1 - alpha_k, "ci_low": float(lo),
            "ci_high": float(hi), "season_gains": season_gains, "other_markets": others,
            "other_markets_not_worse": all(v["mae_gain"] >= 0 for v in others.values()),
            "passes": bool(lo > 0 and all(v["mae_gain"] >= 0 for v in others.values())),
        })
    return {"seed": SEED, "bootstrap_B": BOOTSTRAP_B, "k_selected": k,
            "pairs": out_pairs,
            "verdict": "PASS" if any(p["passes"] for p in out_pairs) else "NEGATIVE"}


# --------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["select", "holdout"])
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache" / "nflverse"),
                        help="nflverse cache (pbp, weekly, rosters, schedules, ...)")
    parser.add_argument("--work-dir", default=str(ROOT / "data" / "cache" / "ratio_search"),
                        help="gitignored scratch for the built frame")
    parser.add_argument("--docs-dir", default=str(ROOT / "docs"))
    args = parser.parse_args()

    cache_dir, work_dir, docs = Path(args.cache_dir), Path(args.work_dir), Path(args.docs_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sel_path = docs / "prop-ratio-search-selection.json"
    hold_path = docs / "prop-ratio-search-holdout.json"
    frame_path = work_dir / "frame_with_candidates.parquet"

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.phase == "select":
        all_seasons = list(range(2017, 2027))
        log("building frame...")
        frame = build_frame(cache_dir, all_seasons)
        cands = candidate_columns(frame)
        for name, series in cands.items():
            frame[f"c__{name}"] = series.to_numpy()
        frame.to_parquet(frame_path, index=False)
        cand_names = [f"c__{n}" for n in cands]
        log(f"candidate pool: {len(cand_names)} (named {sum(n.startswith('named') for n in cands)}, "
            f"pairs {sum(n.startswith('pair') for n in cands)})")
        if len(cand_names) > 300:
            raise SystemExit("candidate pool exceeds the pre-registered cap of 300")
        results = run_select(frame, cand_names, log)
        results["pool_size"] = len(cand_names)
        results["pool_definition"] = {"bases": BASES, "named": [n for n in cands if n.startswith("named")],
                                      "window": WINDOW, "selection_seasons": SELECTION_SEASONS}
        sel_path.write_text(json.dumps(results, indent=1, sort_keys=True))
        log(f"wrote {sel_path}; selected {len(results['selected'])}: {results['selected']}")
        return 0

    if not sel_path.exists():
        log("holdout refused: selection JSON does not exist; run `select` first")
        return 2
    if hold_path.exists():
        log("holdout refused: holdout JSON already exists; the holdout is looked at once")
        return 2
    selection = json.loads(sel_path.read_text())
    selected = selection["selected"]
    if not selected:
        log("no candidates selected: negative result, no holdout look")
        return 0
    frame = pd.read_parquet(frame_path)
    holdout = run_holdout(frame, selected, log)
    hold_path.write_text(json.dumps(holdout, indent=1, sort_keys=True))
    log(f"wrote {hold_path}; verdict {holdout['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

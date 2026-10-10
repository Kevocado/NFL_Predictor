"""How often does the pre-game expected starter equal the quarterback who actually started?"""
from __future__ import annotations

import argparse
from pathlib import Path

from nfl_predictor.config import CURRENT_SEASON, DEPTH_CHARTS_CACHE_DIR
from nfl_predictor.data import depth_charts, player_stats, schedules


def agreement(actual: dict, expected: dict) -> dict:
    """Fraction of (game, team) where the pre-game expected starter equals the actual starter.

    Both sides must be KNOWN (non-None) to count as agreement: a missing expectation
    is "we do not know", never a match — otherwise a week with no depth charts anywhere
    reports a perfect rate.
    """
    # Count all actual games; agreement only where both sides known
    n = len(actual)
    agree = sum(1 for k, q in actual.items() if q is not None and expected.get(k) is not None and expected.get(k) == q)
    return {"n": n, "agree": agree, "rate": agree / n if n else 0.0,
            "no_expectation": sum(1 for k in actual if expected.get(k) is None)}

def _is_gating_status(status: str) -> bool:
    """An injury report status that reliably keeps a player out."""
    return status in ("Out",)


def get_expected_starters(season: int, week: int) -> dict:
    """(game_id, team) -> expected starting QB's gsis_id, from the pre-game depth chart.

    This is the number a SERVING system may use: the depth chart is published
    before kickoff, so a week-1 game is answered by last season's chart, and a
    later week by the newest chart at or before it. `resolve_chart` does that
    resolution.

    Two traps the frame sets, both already cost time in this repo:
    - there is no `depth_slot` column; order within a `depth_position` is file order;
    - `depth_team` is the STRING '1'/'2'/'3', and `depth_position` for QBs is "QB"
      (not "QB1"), so the starter is the row with depth_team == '1'.
    """
    chart_full = depth_charts.load_depth_charts(season, DEPTH_CHARTS_CACHE_DIR)
    if chart_full is None or chart_full.empty:
        return {}
    if "season" not in chart_full.columns:
        chart_full = chart_full.copy()
        chart_full["season"] = season

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

    chart = depth_charts.resolve_chart(chart_full, season, week)
    if chart.empty:  # nothing published yet this season: last season's final chart
        prev = depth_charts.load_depth_charts(season - 1, DEPTH_CHARTS_CACHE_DIR)
        chart = depth_charts.resolve_chart(prev, season - 1, 99) if prev is not None and not prev.empty else chart
    if chart is None or chart.empty:
        return {}

    qbs = chart[chart["depth_position"] == "QB"].copy()
    if qbs.empty:
        return {}
    qbs["is_starter"] = qbs["depth_team"].astype(str) == depth_charts.STARTER_DEPTH_TEAM
    # First file-order starter QB per club -- deterministic, and a player listed
    # twice at QB appears twice, so keep one row per club.
    starters = (qbs[qbs["is_starter"]]
                .drop_duplicates(subset=["club_code"], keep="first"))

    expected = {}
    for _, row in starters.iterrows():
        key = club_to_game.get(row["club_code"])
        if key is not None:
            expected[key] = row["gsis_id"]
    return expected


def get_actual_starters(season: int, week: int) -> dict:
    """(game_id, team) -> the QB who actually started, i.e. threw the most passes for that team.

    Used only to MEASURE agreement. A serving path cannot see this at prediction time.
    """
    week_stats = player_stats.fetch_weekly_player_stats([season])
    if week_stats.empty:
        return {}
    qb = week_stats[(week_stats["position"] == "QB") & (week_stats["week"] == week)].copy()
    if qb.empty:
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

    qb["game_key"] = qb["recent_team"].map(lambda t: club_to_game.get(t, None))
    # Tie-break on attempts then rushing yards: a backup who came in for a blowout
    # must not out-throw the starter he relieved.
    qb = qb.sort_values(["passing_yards", "carries", "rushing_yards"], ascending=False)
    actual = {}
    for key, grp in qb.groupby("game_key"):
        if key is None or grp.empty:
            continue
        actual[key] = grp.iloc[0]["player_id"]
    return actual


def expected_starters_serving_view(season: int, weeks: list[int]) -> dict:
    """(game_id, team) -> expected starter QB, for every game in `weeks`, both team slots filled.

    This is the aux a serving path can build from depth charts alone, before kickoff.
    """
    out = {}
    for week in weeks:
        out.update(get_expected_starters(season, week))
    return out


def main() -> None:
    import csv

    parser = argparse.ArgumentParser(description="How often does the expected starter actually start?")
    parser.add_argument("--season", type=int, default=None, help="default: current season")
    parser.add_argument("--weeks", nargs="+", type=int, default=None, help="default: 1 2 3 4")
    parser.add_argument("--output", type=str, default=None, help="write the per-team-week CSV here")
    args = parser.parse_args()

    season = args.season if args.season is not None else CURRENT_SEASON
    weeks = args.weeks or [1, 2, 3, 4]

    print(f"Analyzing QB agreement for season {season}, weeks {weeks}...\n")
    rows, all_disagreements, failed_weeks = [], [], []
    for week in weeks:
        print(f"Processing week {week}...")
        expected = get_expected_starters(season, week)
        actual = get_actual_starters(season, week)
        if not expected:
            print(f"  Unavailable: no expected starters for week {week}")
            rows.append({"season": season, "week": week, "game_id": "", "team": "",
                         "expected_qb_id": "", "actual_qb_id": "",
                         "match": False, "status": "no_expected"})
            continue
        if not actual:
            print(f"  Unavailable: no actual starters for week {week}")
            for key, exp_id in expected.items():
                rows.append({
                    "season": season, "week": week, "game_id": key[0], "team": key[1],
                    "expected_qb_id": exp_id, "actual_qb_id": "",
                    "match": False, "status": "no_actual",
                })
            continue
        agr = agreement(actual, expected)
        disagreements = []
        for key, exp_id in expected.items():
            act_id = actual.get(key)
            rows.append({
                "season": season, "week": week, "game_id": key[0], "team": key[1],
                "expected_qb_id": exp_id, "actual_qb_id": act_id if act_id is not None else "",
                "match": bool(act_id is not None and act_id == exp_id),
            })
            if act_id is None or act_id != exp_id:
                disagreements.append((week, key[0], key[1], exp_id, act_id))
        all_disagreements.extend(disagreements)
        print(f"  expected starters: {len(expected)}  agreement rate: {agr['rate']:.3f}")

    print("\n" + "=" * 70)
    print("QB AGREEMENT (expected-from-depth-chart vs actual)")
    print("=" * 70)
    if not rows:
        print("No rows produced.")
        return
    # Agreement rate only over rows where both sides are known
    known = [r for r in rows if r.get("status", "") == ""]
    n = len(known)
    agree = sum(1 for r in known if r["match"])
    rate = agree / n if n else 0.0
    unavailable = len(rows) - n
    print(f"team-weeks: {n}   agree: {agree}   rate: {rate:.3f}")
    if unavailable:
        print(f"unavailable: {unavailable} (no expected/actual data)")
    if failed_weeks:
        print(f"failed: {len(failed_weeks)} week(s) - " + ", ".join(f"week {w}: {e}" for w, e in failed_weeks))
        return 1
    if all_disagreements:
        print("\nDisagreements (week, game_id, team, expected_qb_id, actual_qb_id):")
        for tup in all_disagreements:
            print(f"  {tup}")
    else:
        print("\nNo disagreements found.")
    return 0

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["season", "week", "game_id", "team", "expected_qb_id", "actual_qb_id", "match", "status"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nPer-team-week details written to {args.output}")


if __name__ == "__main__":
    main()

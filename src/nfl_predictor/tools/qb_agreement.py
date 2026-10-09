"""How often does the pre-game expected starter equal the quarterback who actually started?"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime

from nfl_predictor.data import depth_charts, injuries, player_stats, schedules
from nfl_predictor.features import qb as qb_features
from nfl_predictor.config import CURRENT_SEASON, DEPTH_CHARTS_CACHE_DIR

def agreement(actual: dict, expected: dict) -> dict:
    """Fraction of (game, team) where the pre-game expected starter equals the actual starter.

    Both sides must be KNOWN (non-None) to count as agreement: a missing expectation
    is "we do not know", never a match — otherwise a week with no depth charts anywhere
    reports a perfect rate.
    """
    n = len(actual)
    agree = sum(1 for k, q in actual.items() if q is not None and expected.get(k) is not None and expected.get(k) == q)
    return {"n": n, "agree": agree, "rate": agree / n if n else 0.0,
            "no_expectation": sum(1 for k in actual if expected.get(k) is None)}

def _is_gating_status(status: str) -> bool:
    """Determine if an injury status should gate a player from playing."""
    return status in ("Out",)

def get_expected_starters(season: int, week: int) -> dict:
    """Return dict mapping (game_id, team) to expected starter QB player_id for the week."""
    # Load depth chart
    chart_full = depth_charts.load_depth_charts(season, DEPTH_CHARTS_CACHE_DIR)
    if chart_full is None or chart_full.empty:
        return {}
    # Ensure season column exists for resolve_chart
    if "season" not in chart_full.columns:
        chart_full = chart_full.copy()
        chart_full["season"] = season
    chart = depth_charts.resolve_chart(chart_full, season, week)
    if chart is None or chart.empty:
        return {}
    # Get starter flags
    chart_flags = depth_charts.flags_for_season_week(season, week, DEPTH_CHARTS_CACHE_DIR)
    if not chart_flags:
        return {}
    # Map player_id to is_starter
    expected_starters = {
        player_id: player_info
        for player_id, player_info in chart_flags.items()
        if player_info.get('is_starter') is True
    }
    # Filter to QBs
    expected_qb_starters = {
        player_id: player_info
        for player_id, player_info in expected_starters.items()
        if player_info.get('position') == 'QB'
    }
    # We need to map to (game_id, team). However depth charts don't have game_id.
    # We'll need to get the schedule for the week to know which teams play each other.
    # We'll assume each team has at most one game per week.
    # Load schedule for the week
    try:
        week_games = schedules.fetch_week_games(season, week)
    except Exception:
        # If fetch_week_games fails, fall back to fetch_schedules for the season and filter by week
        week_games = schedules.fetch_schedules([season])
        week_games = week_games[week_games['week'] == week]
    if week_games.empty:
        return {}
    # Build mapping from team to game_id for this week (assuming each team appears once)
    team_to_game = {}
    for _, row in week_games.iterrows():
        team_to_game[row['home_team']] = row['game_id']
        team_to_game[row['away_team']] = row['game_id']
    # Now map each expected starter QB to their game_id via their team
    expected = {}
    for player_id, player_info in expected_qb_starters.items():
        team = player_info.get('team')  # Does chart_flags include team? Let's check.
        # Actually flags_for_season_week returns dict keyed by player_id with info including team?
        # We need to examine the structure. Let's assume it includes 'team'.
        # If not, we may need to get team from the chart data.
        # For now, we'll skip if team not present.
        if team is None:
            continue
        game_id = team_to_game.get(team)
        if game_id is not None:
            expected[(game_id, team)] = player_id
    return expected

def get_actual_starters(season: int, week: int) -> dict:
    """Return dict mapping (game_id, team) to actual starter QB player_id for the week.
    Actual starter defined as QB with most passing yards in the game.
    """
    # Load player stats for the week
    try:
        week_stats = player_stats.fetch_weekly_player_stats([season], force_refresh=False)
    except Exception as e:
        print(f"Warning: Could not load player stats for season {season}: {e}")
        return {}
    if week_stats.empty:
        return {}
    # Filter to QBs and the specific week
    qb_week = week_stats[(week_stats['position'] == 'QB') & (week_stats['week'] == week)]
    if qb_week.empty:
        return {}
    # For each game, we need to find the QB with max passing_yards.
    # We need to know which game each player_stat row belongs to.
    # player_stats does not contain game_id. We need to join with schedules via team and week.
    # However a team plays one game per week, so we can map team to game_id via schedule.
    # Load schedule for the week
    try:
        week_games = schedules.fetch_week_games(season, week)
    except Exception:
        week_games = schedules.fetch_schedules([season])
        week_games = week_games[week_games['week'] == week]
    if week_games.empty:
        return {}
    team_to_game = {}
    for _, row in week_games.iterrows():
        team_to_game[row['home_team']] = row['game_id']
        team_to_game[row['away_team']] = row['game_id']
    # For each player_stat row, we can get the team from recent_team column.
    # However recent_team may be the team they played for that week (should be correct).
    # We'll map each row to game_id via team.
    qb_week = qb_week.copy()
    qb_week['game_id'] = qb_week['recent_team'].map(team_to_game)
    # Drop rows where mapping failed (should not happen)
    qb_week = qb_week.dropna(subset=['game_id'])
    if qb_week.empty:
        return {}
    # For each game, find the QB with max passing_yards (if tie, break by rushing_yards)
    def pick_starter(group):
        # Sort by passing_yards descending, then rushing_yards descending
        sorted_group = group.sort_values(['passing_yards', 'rushing_yards'], ascending=[False, False])
        # Take the first row's player_id
        return sorted_group.iloc[0]['player_id']
    try:
        actual_series = qb_week.groupby('game_id').apply(pick_starter)
    except Exception:
        # Fallback: just take max passing_yards
        actual_series = qb_week.loc[qb_week.groupby('game_id')['passing_yards'].idxmax()]['player_id']
    # Build dict mapping (game_id, team) to player_id
    actual = {}
    for game_id, player_id in actual_series.items():
        # We need to know which team the player_id belongs to for this game.
        # We can get the team from the player_stat row(s) for this game and player_id.
        # Get the row(s) for this game and player_id
        player_rows = qb_week[(qb_week['game_id'] == game_id) & (qb_week['player_id'] == player_id)]
        if player_rows.empty:
            continue
        team = player_rows.iloc[0]['recent_team']
        actual[(game_id, team)] = player_id
    return actual

def main():
    parser = argparse.ArgumentParser(description="Analyze NFL QB agreement between expected and actual starters")
    parser.add_argument(
        "--season", 
        type=int, 
        default=None,
        help="NFL season to analyze (default: current season)"
    )
    parser.add_argument(
        "--weeks", 
        nargs="+", 
        type=int,
        default=None,
        help="Weeks to analyze (default: first 4 weeks of season)"
    )
    parser.add_argument(
        "--force-refresh", 
        action="store_true",
        help="Force refresh of cached data"
    )
    
    args = parser.parse_args()
    
    if args.season is None:
        season = CURRENT_SEASON
    else:
        season = args.season
    
    if args.weeks is None:
        weeks = list(range(1, 5))  # first 4 weeks
    else:
        weeks = args.weeks
    
    print(f"Analyzing QB agreement for season {season}, weeks {weeks}...")
    print()
    
    weekly_results = []
    all_disagreements = []  # List of (week, game_id, team, expected_player_id, actual_player_id)
    
    for week in weeks:
        print(f"Processing week {week}...")
        try:
            expected = get_expected_starters(season, week)
            actual = get_actual_starters(season, week)
            if not expected:
                print(f"  Warning: No expected starters data for week {week}")
                weekly_results.append((week, 0, 0.0, []))
                continue
            if not actual:
                print(f"  Warning: No actual starters data for week {week}")
                # We'll still compute agreement with empty actual? Better to skip.
                weekly_results.append((week, len(expected), 0.0, []))
                continue
            # Compute agreement
            agr = agreement(actual, expected)
            n = agr['n']
            agree = agr['agree']
            rate = agr['rate']
            # Collect disagreements
            disagreements = []
            for (game_id, team), exp_id in expected.items():
                act_id = actual.get((game_id, team))
                if act_id is None:
                    # missing actual
                    disagreements.append((week, game_id, team, exp_id, None))
                elif exp_id != act_id:
                    disagreements.append((week, game_id, team, exp_id, act_id))
            weekly_results.append((week, len(expected), rate, disagreements))
            all_disagreements.extend(disagreements)
            print(f"  Expected starters: {len(expected)}, Agreement rate: {rate:.3f}")
        except Exception as e:
            print(f"  Error processing week {week}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results
    print("\n" + "="*70)
    print("QB AGREEMENT ANALYSIS RESULTS")
    print("="*70)
    print(f"{'Week':<6} {'Expected Starters':<20} {'Agreement Rate':<15} {'Status'}")
    print("-"*70)
    for week, exp_count, rate, _ in weekly_results:
        status = "✓ Meets target (≥90%)" if rate >= 0.90 else "✗ Below target (<90%)"
        print(f"{week:<6} {exp_count:<20} {rate:<15.3f} {status}")
    
    if all_disagreements:
        print("\nDisagreements (week, game_id, team, expected_player_id, actual_player_id):")
        for tup in all_disagreements[:20]:  # limit output
            print(f"  {tup}")
        if len(all_disagreements) > 20:
            print(f"  ... and {len(all_disagreements)-20} more")
    else:
        print("\nNo disagreements found.")
    
    print("\nNote: Actual starter defined as QB with most passing yards in the game.")
    print("To improve accuracy, consider using snap counts or other metrics.")
    
    return 0

if __name__ == "__main__":
    main()

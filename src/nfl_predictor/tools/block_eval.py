
def load_recent_nfl_data(start_year: int = 2000, end_year: int = None) -> tuple[pd.DataFrame, feature_build.Aux]:
    """Load NFL data for evaluation.

    Args:
        start_year: First season to include (default: 2000).
        end_year: Last season to include (default: current year).

    Returns:
        Tuple of (games_df, aux).
    """
    if end_year is None:
        end_year = datetime.now().year
    seasons = list(range(start_year, end_year + 1))

    print(f"Loading NFL data for seasons {seasons}...")

    # Load games data
    games_df = schedules.load_training_data(seasons)

    # Load PBP aggregates for aux
    efficiency_frames = []
    qb_frames = []

    for season in seasons:
        try:
            pbp = load_pbp_agg(season)
            if not pbp.empty:
                efficiency_frames.append(team_game_efficiency(pbp))
                qb_frames.append(qb_games(pbp))
        except Exception as e:
            print(f"Warning: Could not load PBP data for season {season}: {e}")

    efficiency = pd.concat(efficiency_frames, ignore_index=True) if efficiency_frames else pd.DataFrame()
    qb_games_df = pd.concat(qb_frames, ignore_index=True) if qb_frames else pd.DataFrame()

    # Create Aux object
    aux = feature_build.Aux(efficiency=efficiency, qb_games=qb_games_df, upcoming_starters={})

    return games_df, aux


def main():
    parser = argparse.ArgumentParser(description="Evaluate NFL feature blocks")
    parser.add_argument(
        "--blocks",
        nargs="+",
        required=True,
        help="Feature blocks to evaluate (e.g., epa qb conditions)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="First season to include (default: 2000)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last season to include (default: current year)",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="ridge",
        help="Candidate model to use (default: ridge)",
    )

    args = parser.parse_args()

    # Load data
    games_df, aux = load_recent_nfl_data(start_year=args.start_year, end_year=args.end_year)

    if games_df.empty:
        print("Error: No games data loaded.")
        return 1

    print(f"Loaded {len(games_df)} games for evaluation.")
    print(f"Evaluating blocks: {', '.join(args.blocks)}")
    print(f"Using candidate model: {args.candidate}")
    print()

    # Evaluate each block
    results = []
    for block in args.blocks:
        print(f"Evaluating block: {block}...")
        try:
            result = evaluate_block(games_df, aux, block, args.candidate)
            results.append(result)
        except Exception as e:
            print(f"Error evaluating block {block}: {e}")
            import traceback
            traceback.print_exc()
            results.append(None)

    # Print results in a table format suitable for PR description
    print("\n" + "=" * 80)
    print("NFL BLOCK EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Block':<12} {'N Games':<10} {'MAE Δ':<12} {'MAE 95% CI':<20} {'Brier Δ':<12} {'Brier 95% CI':<20} {'Gap Base':<10} {'Gap Block':<10} {'Clears'}")
    print("-" * 80)

    for result in results:
        if result is None:
            print(f"{'ERROR':<12} {'N/A':<10} {'N/A':<12} {'N/A':<20} {'N/A':<12} {'N/A':<20} {'N/A':<10} {'N/A':<10} {'N/A'}")
            continue

        block = result['block']
        n_games = result['n_games']
        mae_delta = result['mae_delta']
        mae_ci = result['mae_ci']
        brier_delta = result['brier_delta']
        brier_ci = result['brier_ci']
        gap_base = result['gap_base']
        gap_block = result['gap_block']
        clears = result['clears']

        # Format confidence intervals as strings
        mae_ci_str = f"[{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]"
        brier_ci_str = f"[{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]"

        print(f"{block:<12} {n_games:<10} {mae_delta:<12.4f} {mae_ci_str:<20} {brier_delta:<12.4f} {brier_ci_str:<20} {gap_base:<10.4f} {gap_block:<10.4f} {'Yes' if clears else 'No'}")

    print("=" * 80)

    return 0

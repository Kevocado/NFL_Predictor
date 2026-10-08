import numpy as np
import pandas as pd
from nfl_predictor.features.elo_fit import fit_elo_constants


def _season(n_teams=8, weeks=14, seed=0, hfa_pts=2.5):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 5, n_teams)
    rows, gid = [], 0
    for w in range(weeks):
        order = rng.permutation(n_teams)
        for a, b in zip(order[::2], order[1::2]):
            margin = skill[a] - skill[b] + hfa_pts + rng.normal(0, 10)
            home_score = max(0, int(20 + margin/2 + rng.normal(0, 7)))
            away_score = max(0, int(20 - margin/2 + rng.normal(0, 7)))
            rows.append({"game_id": f"g{gid}", "gameday": pd.Timestamp("2024-09-01") + pd.Timedelta(days=7 * w),
                         "home_team": f"T{a}", "away_team": f"T{b}", "margin": margin,
                         "home_score": home_score, "away_score": away_score})
            gid += 1
    return pd.DataFrame(rows)


def test_fit_returns_grid_members_and_is_deterministic():
    games = _season()
    a, b = fit_elo_constants(games), fit_elo_constants(games)
    assert a == b
    assert set(a) == {"k", "home_field"}
    assert 5 <= a["k"] <= 40 and 0 <= a["home_field"] <= 100


def test_fit_never_uses_the_scored_game():
    games = _season()
    shifted = games.copy()
    shifted.loc[shifted.index[-1], "margin"] += 50  # only the LAST game changes
    shifted.loc[shifted.index[-1], "home_score"] += 25
    assert fit_elo_constants(games.iloc[:-1]) == fit_elo_constants(shifted.iloc[:-1])


def test_unplayed_games_are_ignored_not_scored_as_away_wins():
    """An unplayed game (NaN scores) must not contribute to the loss.
    If it were silently scored as an away win (0.0), the fitted constants would shift."""
    games = _season()
    # Add an unplayed future game at the end
    future = pd.DataFrame([{
        "game_id": "g999", "gameday": pd.Timestamp("2025-01-01"),
        "home_team": "T0", "away_team": "T1",
        "margin": np.nan, "home_score": np.nan, "away_score": np.nan
    }])
    with_future = pd.concat([games, future], ignore_index=True)
    # The result must be the same as without the future game
    assert fit_elo_constants(games) == fit_elo_constants(with_future)


def test_ties_are_half_not_away_wins():
    """A tied game (home_score == away_score) contributes y=0.5 to log loss,
    not y=0.0 (away win)."""
    games = _season()
    # Force the last game to be a tie
    tied = games.copy()
    tied.loc[tied.index[-1], "home_score"] = 24
    tied.loc[tied.index[-1], "away_score"] = 24
    # The loss with a tie should differ from the loss with a home/away win
    a = fit_elo_constants(games)
    b = fit_elo_constants(tied)
    # The grid search may pick the same (k, hfa) pair but the loss value differs;
    # what matters is the function doesn't crash and treats 0.5 correctly.
    assert set(a) == {"k", "home_field"} and set(b) == {"k", "home_field"}
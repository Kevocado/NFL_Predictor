"""NFL offence-versus-defence duels for one game, from play-by-play efficiency known BEFORE the game.

Each duel is one side's attack (EPA/play, pass or rush) against the other side's matching defence, expressed as league
ranks. Only games of `season` strictly before `as_of`, and each team's last `WINDOW` of them, are used: early in a season
there are fewer than `MIN_GAMES` and the answer is no duels, which is better than ranking last year's roster.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from nfl_predictor.signals.duel import Duel, make_duel, edge_strength


WINDOW = 8
MIN_GAMES = 3
MIN_GAP = 8

DUALS = [
    ("pass_off_vs_pass_def", "passing offence", "pass defence", "epa_off_pass", "epa_def_pass"),
    ("rush_off_vs_rush_def", "rushing offence", "rush defence", "epa_off_rush", "epa_def_rush"),
]


def _recent_means(efficiency: pd.DataFrame, games_df: pd.DataFrame, as_of, season: int, team: str) -> dict[str, float]:
    """Mean EPA for a team's last WINDOW games of this season strictly before as_of."""
    team_games = games_df[(games_df["season"] == season) &
                          (games_df["gameday"] < as_of) &
                          ((games_df["home_team"] == team) | (games_df["away_team"] == team))]
    if len(team_games) < MIN_GAMES:
        return {}
    recent = team_games.sort_values("gameday").tail(WINDOW)
    game_ids = recent["game_id"].tolist()
    eff = efficiency[efficiency["game_id"].isin(game_ids) & (efficiency["team"] == team)]
    if len(eff) < MIN_GAMES:
        return {}
    return {
        "epa_off_pass": float(eff["epa_off_pass"].mean()),
        "epa_off_rush": float(eff["epa_off_rush"].mean()),
        "epa_def_pass": float(eff["epa_def_pass"].mean()),
        "epa_def_rush": float(eff["epa_def_rush"].mean()),
    }


def _ranks_from_means(means: dict[str, float], higher_is_better: bool) -> dict[str, int]:
    from nfl_predictor.signals.duel import ranks
    return ranks(means, higher_is_better=higher_is_better)


def matchups_for_game(home: str, away: str, games_df: pd.DataFrame, efficiency: pd.DataFrame,
                      as_of, season: int, min_gap: int = MIN_GAP) -> list:
    """Return list of duels for a game, strongest first.

    Only games of `season` strictly before `as_of` are used. Each team's last `WINDOW` games are averaged.
    """
    home_means = _recent_means(efficiency, games_df, as_of, season, home)
    away_means = _recent_means(efficiency, games_df, as_of, season, away)
    if not home_means or not away_means:
        return []

    # Build league-wide ranks for this season
    all_teams = pd.unique(games_df[["home_team", "away_team"]].values.ravel())
    off_means = {t: _recent_means(efficiency, games_df, pd.Timestamp.max, season, t) for t in all_teams}
    def_means = {t: _recent_means(efficiency, games_df, pd.Timestamp.max, season, t) for t in all_teams}

    off_means = {t: v for t, v in off_means.items() if v}
    def_means = {t: v for t, v in def_means.items() if v}

    if not off_means or not def_means:
        return []

    # Build league ranks
    off_pass_ranks = _ranks_from_means({t: v["epa_off_pass"] for t, v in off_means.items() if "epa_off_pass" in v}, higher_is_better=True)
    off_rush_ranks = _ranks_from_means({t: v["epa_off_rush"] for t, v in off_means.items() if "epa_off_rush" in v}, higher_is_better=True)
    def_pass_ranks = _ranks_from_means({t: v["epa_def_pass"] for t, v in def_means.items() if "epa_def_pass" in v}, higher_is_better=False)
    def_rush_ranks = _ranks_from_means({t: v["epa_def_rush"] for t, v in def_means.items() if "epa_def_rush" in v}, higher_is_better=False)

    duels = []
    for duel_id, stat, foil, off_col, def_col in DUALS:
        if duel_id == "pass_off_vs_pass_def":
            a_rank = off_pass_ranks.get(home)
            d_rank = def_pass_ranks.get(away)
        else:
            a_rank = off_rush_ranks.get(home)
            d_rank = def_rush_ranks.get(away)

        if a_rank is None or d_rank is None:
            continue

        duel = make_duel(
            duel_id, "passing offence" if "pass" in duel_id else "rushing offence",
            "pass defence" if "pass" in duel_id else "rush defence",
            home=home, away=away, attacker_side="home",
            attack_ranks=off_pass_ranks if "pass" in duel_id else off_rush_ranks,
            defence_ranks=def_pass_ranks if "pass" in duel_id else def_rush_ranks,
            history_gaps=[], min_gap=min_gap)
        if duel:
            # Fix attacker/defender to be home/away teams
            duel = Duel(
                duel.id, duel.attacker, duel.defender, duel.stat, duel.foil,
                a_rank, d_rank, duel.n_teams, duel.toward, duel.strength)
            # Fix attacker/defender to be home/away teams
            duel = Duel(
                duel.id, home, away, duel.stat, duel.foil,
                a_rank, d_rank, duel.n_teams, duel.toward, duel.strength)
            duels.append(duel)

    duels.sort(key=lambda d: d.strength, reverse=True)
    return duels


def to_context(duels: list, pick_side: str | None = None, limit: int = 4) -> list[dict]:
    """Convert duels to context dicts with pick-relative direction."""
    out = []
    for d in duels[:limit]:
        toward_pick = None
        if pick_side is not None:
            toward_pick = (d.toward == pick_side)
        out.append({
            "id": d.id,
            "attacker": d.attacker,
            "defender": d.defender,
            "stat": d.stat,
            "foil": d.foil,
            "attacker_rank": d.attacker_rank,
            "defender_rank": d.defender_rank,
            "n_teams": d.n_teams,
            "toward": d.toward,
            "toward_pick": toward_pick,
            "strength": d.strength,
        })
    return out

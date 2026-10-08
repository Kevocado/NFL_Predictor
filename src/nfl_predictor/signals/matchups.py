"""NFL offence-versus-defence duels for one game, from play-by-play efficiency known BEFORE the game.

Each duel is one side's attack (EPA/play, pass or rush) against the other side's matching defence, expressed as league
ranks. Only games of `season` strictly before `as_of`, and each team's last `WINDOW` of them, are used: early in a season
there are fewer than `MIN_GAMES` and the answer is no duels, which is better than ranking last year's roster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .duel import Duel, make_duel, ranks

WINDOW = 8
MIN_GAMES = 3
#: (duel id, attack column, defence column, attack noun, defence noun). Defence columns are EPA ALLOWED: lower is better.
DUELS = [
    ("pass_off_vs_pass_def", "epa_off_pass", "epa_def_pass", "passing offence", "pass defence"),
    ("rush_off_vs_rush_def", "epa_off_rush", "epa_def_rush", "rushing offence", "rush defence"),
]


def _recent_means(efficiency: pd.DataFrame, games_df: pd.DataFrame, as_of: pd.Timestamp, season: int) -> pd.DataFrame:
    """Last WINDOW games per team, from `season` strictly before `as_of`, with at least MIN_GAMES of data."""
    meta = games_df[["game_id", "gameday", "season"]].copy()
    meta["gameday"] = pd.to_datetime(meta["gameday"])
    eff = efficiency.assign(
        gameday=pd.to_datetime(efficiency["game_id"].map(meta["gameday"])),
        season=efficiency["game_id"].map(meta["season"]),
    )
    eff = eff[(eff["gameday"] < as_of) & (eff["season"] == season)].sort_values("gameday")
    last = eff.groupby("team").tail(WINDOW)
    counts = last.groupby("team").size()
    means = last.groupby("team").mean(numeric_only=True)
    return means[counts.reindex(means.index) >= MIN_GAMES]


def matchups_for_game(
    home: str, away: str, games_df: pd.DataFrame, efficiency: pd.DataFrame, as_of, season: int,
    history_gaps: dict[str, np.ndarray] | None = None, min_gap: int = 8,
) -> list[Duel]:
    """Up to four duels (two kinds x two directions), strongest first. Empty when either team lacks data."""
    means = _recent_means(efficiency, games_df, pd.Timestamp(as_of), season)
    if home not in means.index or away not in means.index:
        return []
    history_gaps = history_gaps or {}
    out: list[Duel] = []
    for duel_id, attack_col, defence_col, attack_noun, defence_noun in DUELS:
        attack_ranks = ranks(means[attack_col].to_dict(), higher_is_better=True)
        defence_ranks = ranks(means[defence_col].to_dict(), higher_is_better=False)
        for attacker, defender in ((home, away), (away, home)):
            d = make_duel(
                f"{duel_id}:{attacker}", attacker, defender,
                home=home, away=away,
                attack_ranks=attack_ranks, defence_ranks=defence_ranks,
                history_gaps=history_gaps.get(duel_id, np.array([])), min_gap=min_gap,
            )
            if d is not None:
                out.append(d)
    out.sort(key=lambda d: d.strength, reverse=True)
    return out


def to_context(duels: list[Duel], pick_side: str | None, limit: int = 4) -> list[dict]:
    """Facts-bundle form. `toward_pick` is the direction relative to the pick, or None when there is no pick."""
    out: list[dict] = []
    for d in duels[:limit]:
        toward = d.toward
        if pick_side == "home":
            toward_pick = "home" if toward == "home" else "away"
        elif pick_side == "away":
            toward_pick = "away" if toward == "away" else "home"
        else:
            toward_pick = None
        out.append({
            "id": d.id,
            "attacker": d.attacker,
            "defender": d.defender,
            "stat": "",
            "foil": "",
            "attacker_rank": d.attacker_rank,
            "defender_rank": d.defender_rank,
            "n_teams": d.n_teams,
            "toward_pick": toward_pick,
        })
    return out
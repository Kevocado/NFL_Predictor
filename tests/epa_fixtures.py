"""Synthetic NFL games and play-by-play shaped like nflverse's, deterministic and offline."""
import numpy as np
import pandas as pd

from nfl_predictor.data.pbp_agg import PBP_AGG_COLUMNS

TEAMS = [f"T{i}" for i in range(8)]


def make_games(seed=3, seasons=(2023, 2024, 2025), weeks=14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            order = list(rng.permutation(TEAMS))
            n_games = 3 if week % 4 == 0 else 4
            for i in range(0, 2 * n_games, 2):
                home, away = order[i], order[i + 1]
                rows.append({
                    "game_id": f"{season}_{week:02d}_{home}_{away}", "season": season, "week": week,
                    "gameday": pd.Timestamp(f"{season}-09-07") + pd.Timedelta(days=7 * (week - 1)),
                    "home_team": home, "away_team": away,
                    "home_score": int(rng.integers(6, 42)), "away_score": int(rng.integers(6, 42)), "div_game": 0,
                })
    return pd.DataFrame(rows)


def qb_ids(team: str) -> tuple[str, str]:
    return f"{team}-QB1", f"{team}-QB2"


def make_pbp(games: pd.DataFrame, seed=5, plays_per_side=50) -> pd.DataFrame:
    """Plays for every game. Each team has a latent offence and defence level; its QB1 starts ~85% of games and
    QB2 starts the rest (QB2 also takes ~15% of dropbacks in games QB1 starts)."""
    rng = np.random.default_rng(seed)
    off = {t: rng.normal(0.0, 0.08) for t in TEAMS}
    dfn = {t: rng.normal(0.0, 0.08) for t in TEAMS}
    rows = []
    for _, g in games.iterrows():
        for posteam, defteam in ((g["home_team"], g["away_team"]), (g["away_team"], g["home_team"])):
            qb1, qb2 = qb_ids(posteam)
            starter = qb1 if rng.random() < 0.85 else qb2
            for _ in range(plays_per_side):
                is_pass = rng.random() < 0.58
                epa = off[posteam] - dfn[defteam] + rng.normal(0.0, 1.2) + (0.05 if is_pass else -0.03)
                qb = starter if rng.random() < 0.85 else (qb2 if starter == qb1 else qb1)
                rows.append({
                    "game_id": g["game_id"], "season": g["season"], "week": g["week"], "posteam": posteam, "defteam": defteam,
                    "play_type": "pass" if is_pass else "run", "epa": epa, "success": float(epa > 0),
                    "qb_dropback": 1 if is_pass else 0, "passer_player_id": qb if is_pass else None,
                    "passer_player_name": f"Name {qb}" if is_pass else None,
                })
    return pd.DataFrame(rows, columns=PBP_AGG_COLUMNS)


def perturb_pbp_from(pbp: pd.DataFrame, games: pd.DataFrame, cut_date: str, seed=9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    late = set(games[pd.to_datetime(games["gameday"]) >= pd.Timestamp(cut_date)]["game_id"])
    out = pbp.copy()
    mask = out["game_id"].isin(late)
    out.loc[mask, "epa"] = out.loc[mask, "epa"] + rng.normal(0, 3.0, mask.sum())
    return out
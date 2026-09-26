"""Per-team season efficiency from nflverse play-by-play (the NFL's
equivalent of PL's xG), plus record, form and recent games from the
schedule. Missing values are None, never 0: the site shows a dash."""
from __future__ import annotations

import pandas as pd

from ..config import CACHE_DIR

PBP_COLUMNS = ["game_id", "posteam", "defteam", "epa", "success", "yards_gained", "pass", "rush",
               "play_type", "interception", "fumble_lost", "week", "season_type"]


def load_pbp(season: int) -> pd.DataFrame:
    path = CACHE_DIR / "pbp" / f"pbp_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import nfl_data_py as nfl

    df = nfl.import_pbp_data([season], columns=PBP_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _results(team: str, played: pd.DataFrame) -> list[dict]:
    out = []
    for _, g in played.sort_values("gameday").iterrows():
        home = g["home_team"] == team
        ts, os_ = (g["home_score"], g["away_score"]) if home else (g["away_score"], g["home_score"])
        out.append({"gameday": str(g["gameday"])[:10], "opponent": g["away_team"] if home else g["home_team"],
                    "is_home": bool(home), "team_score": int(ts), "opponent_score": int(os_),
                    "result": "W" if ts > os_ else "L" if ts < os_ else "T"})
    return out


def _streak(results: list[dict]) -> int:
    if not results:
        return 0
    last = results[-1]["result"]
    n = 0
    for r in reversed(results):
        if r["result"] != last:
            break
        n += 1
    return n if last == "W" else -n if last == "L" else 0


def _trend(results: list[dict]) -> str:
    if len(results) < 3:
        return "new"
    nets = [r["team_score"] - r["opponent_score"] for r in results]
    diff = sum(nets[-3:]) / 3 - sum(nets) / len(nets)
    return "up" if diff > 3 else "down" if diff < -3 else "steady"


def team_efficiency(pbp: pd.DataFrame, games: pd.DataFrame, season: int) -> list[dict]:
    plays = pbp[pbp["play_type"].isin(["pass", "run"])] if not pbp.empty else pbp
    season_games = games[games["season"] == season]
    teams = sorted(set(season_games["home_team"]) | set(season_games["away_team"]))
    played = season_games[season_games["home_score"].notna() & season_games["away_score"].notna()]
    rows = []
    for team in teams:
        mine = played[(played["home_team"] == team) | (played["away_team"] == team)]
        results = _results(team, mine)
        n = len(results)
        off = plays[plays["posteam"] == team] if not plays.empty else plays
        dfn = plays[plays["defteam"] == team] if not plays.empty else plays
        give = int(off["interception"].sum() + off["fumble_lost"].sum()) if len(off) else 0
        take = int(dfn["interception"].sum() + dfn["fumble_lost"].sum()) if len(dfn) else 0
        rows.append({
            "team": team, "games": n,
            "wins": sum(r["result"] == "W" for r in results),
            "losses": sum(r["result"] == "L" for r in results),
            "ties": sum(r["result"] == "T" for r in results),
            "points_for_pg": _r(sum(r["team_score"] for r in results) / n, 1) if n else None,
            "points_against_pg": _r(sum(r["opponent_score"] for r in results) / n, 1) if n else None,
            "off_epa_play": _r(off["epa"].mean()) if len(off) else None,
            "def_epa_play": _r(dfn["epa"].mean()) if len(dfn) else None,
            "off_success_rate": _r(off["success"].mean()) if len(off) else None,
            "def_success_rate": _r(dfn["success"].mean()) if len(dfn) else None,
            "yards_per_play": _r(off["yards_gained"].mean(), 2) if len(off) else None,
            "pass_rate": _r(off["pass"].mean()) if len(off) else None,
            "turnover_margin": (take - give) if (len(off) or len(dfn)) else None,
            "streak": _streak(results),
            "form": [r["result"] for r in results[-5:]],
            "form_trend": _trend(results),
            "recent_games": list(reversed(results[-5:])),
        })
    return rows

"""Team efficiency (EPA per play) as pre-game features, plus the offence-versus-defence matchup edges.

EPA is roughly zero-sum across the league, so a team's rolling value is shrunk toward ZERO by n/(n+k) (n = prior games
with data): two games of data do not make an offence elite. Every value for game g uses only games before g.
"""
from __future__ import annotations

import pandas as pd

EWM_HALFLIFE = 6.0
SHRINK_K = 4.0
STATS = [
    "epa_off", "epa_def", "epa_off_pass", "epa_off_rush", "epa_def_pass", "epa_def_rush", "success_off", "success_def",
]
EDGE_COLUMNS = ["home_pass_edge", "home_rush_edge", "away_pass_edge", "away_rush_edge", "epa_net_diff"]


def epa_columns() -> list[str]:
    return [f"{side}_{s}_ewm" for s in STATS for side in ("home", "away")] + EDGE_COLUMNS


def _shrunk_ewm(series: pd.Series, halflife: float, k: float) -> pd.Series:
    lagged = series.shift(1)
    n = lagged.notna().cumsum()
    return lagged.ewm(halflife=halflife, min_periods=1).mean() * (n / (n + k))


def add_epa_features(games_df: pd.DataFrame, efficiency: pd.DataFrame, halflife: float = EWM_HALFLIFE, k: float = SHRINK_K) -> pd.DataFrame:
    """Add `{home,away}_<stat>_ewm` for each stat in STATS and the matchup edges. `efficiency` is team_game_efficiency()."""
    rows = []
    for side in ("home", "away"):
        rows.append(games_df[["game_id", "gameday", f"{side}_team"]].rename(columns={f"{side}_team": "team"}))
    long = pd.concat(rows, ignore_index=True).drop_duplicates(["game_id", "team"])
    long["gameday"] = pd.to_datetime(long["gameday"])
    long = long.merge(efficiency, on=["game_id", "team"], how="left").sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    for stat in STATS:
        long[f"{stat}_ewm"] = long.groupby("team")[stat].transform(lambda s: _shrunk_ewm(s, halflife, k))
    keyed = long.set_index(["game_id", "team"])[[f"{s}_ewm" for s in STATS]]
    out = games_df.copy()
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([out["game_id"], out[f"{side}_team"]])
        vals = keyed.reindex(idx)
        for s in STATS:
            out[f"{side}_{s}_ewm"] = vals[f"{s}_ewm"].to_numpy()
    out["home_pass_edge"] = out["home_epa_off_pass_ewm"] - out["away_epa_def_pass_ewm"]
    out["home_rush_edge"] = out["home_epa_off_rush_ewm"] - out["away_epa_def_rush_ewm"]
    out["away_pass_edge"] = out["away_epa_off_pass_ewm"] - out["home_epa_def_pass_ewm"]
    out["away_rush_edge"] = out["away_epa_off_rush_ewm"] - out["home_epa_def_rush_ewm"]
    out["epa_net_diff"] = (out["home_epa_off_ewm"] - out["home_epa_def_ewm"]) - (out["away_epa_off_ewm"] - out["away_epa_def_ewm"])
    return out
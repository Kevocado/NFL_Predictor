"""The starting quarterback as a feature: his career EPA per dropback (shrunk), experience, and whether he changed.

Training uses the QB who actually started (the passer with the most dropbacks). Serving has no result, so it uses the
EXPECTED starter: the depth chart's first QB who is not reported Out. That skew is real and is measured by
`tools/qb_starter_agreement.py`, not assumed away. A QB's own past games are the only data his rating reads.
"""
from __future__ import annotations

import pandas as pd

SHRINK_K = 200.0        # dropbacks of prior worth: a QB with 200 dropbacks is half his own record, half the prior
PRIOR_EPA_PER_DROPBACK = 0.0
NEW_QB_GAMES = 3        # fewer career games than this in the data = a new / unproven starter
QB_COLUMNS = [f"{side}_{c}" for c in ("qb_epa_pd", "qb_games", "qb_changed", "qb_new") for side in ("home", "away")]


def _history(qb_games_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.concat([
        games_df[["game_id", "gameday"]],
    ]).drop_duplicates("game_id")
    h = qb_games_df.merge(dates, on="game_id", how="left")
    h["gameday"] = pd.to_datetime(h["gameday"])
    return h.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def add_qb_features(
    games_df: pd.DataFrame, qb_games_df: pd.DataFrame,
    upcoming_starters: dict[tuple[str, str], str | None] | None = None,
    k: float = SHRINK_K, prior: float = PRIOR_EPA_PER_DROPBACK, new_games: int = NEW_QB_GAMES,
) -> pd.DataFrame:
    """Add `{home,away}_qb_epa_pd / qb_games / qb_changed / qb_new` for every game.

    `qb_games_df` holds one row per (game, team, qb) — every passer, not just the starter.
    The starter for a completed game is the passer with the most dropbacks (lowest id on a tie);
    his rating and experience accumulate ALL his prior dropbacks, including relief appearances
    in games he did not start. A game with no QB row (upcoming) uses `upcoming_starters[(game_id, team)]`,
    and a game with neither gets the prior, zero games and "new": the honest "we do not know who starts".
    """
    upcoming_starters = upcoming_starters or {}
    hist = _history(qb_games_df, games_df)
    hist["cum_db"] = hist.groupby("qb_id")["dropbacks"].cumsum() - hist["dropbacks"]
    hist["cum_epa"] = hist.groupby("qb_id")["epa_sum"].cumsum() - hist["epa_sum"]
    hist["cum_games"] = hist.groupby("qb_id").cumcount()
    # Starter per (game, team): most dropbacks, lowest id on a tie. Pick within each
    # game-team group; order the starters chronologically for prev_qb.
    ranked = hist.sort_values(["game_id", "team", "dropbacks", "qb_id"], ascending=[True, True, False, True])
    starter_rows = ranked.drop_duplicates(["game_id", "team"], keep="first").sort_values(["gameday", "game_id"])
    starter_rows = starter_rows.copy()
    starter_rows["prev_qb"] = starter_rows.groupby("team")["qb_id"].shift(1)
    by_starter = starter_rows.set_index(["game_id", "team"])
    # What each QB has accumulated through his LAST game: the starting point for a game that has not been played.
    totals = hist.groupby("qb_id").agg(db=("dropbacks", "sum"), epa=("epa_sum", "sum"), games=("dropbacks", "size"))
    last_qb_of_team = starter_rows.groupby("team")["qb_id"].last()

    out = games_df.copy()
    for side in ("home", "away"):
        epa_pd, n_games, changed, new = [], [], [], []
        for game_id, team in zip(out["game_id"], out[f"{side}_team"]):
            key = (game_id, team)
            if key in by_starter.index:
                r = by_starter.loc[key]
                epa_pd.append((r["cum_epa"] + k * prior) / (r["cum_db"] + k))
                n_games.append(float(r["cum_games"]))
                changed.append(float(isinstance(r["prev_qb"], str) and r["prev_qb"] != r["qb_id"]))
            else:
                qb = upcoming_starters.get(key)
                if qb is not None and qb in totals.index:
                    t = totals.loc[qb]
                    epa_pd.append((t["epa"] + k * prior) / (t["db"] + k))
                    n_games.append(float(t["games"]))
                    changed.append(float(team in last_qb_of_team.index and last_qb_of_team[team] != qb))
                else:
                    epa_pd.append(prior)
                    n_games.append(0.0)
                    changed.append(0.0)
            new.append(float(n_games[-1] < new_games))
        out[f"{side}_qb_epa_pd"], out[f"{side}_qb_games"] = epa_pd, n_games
        out[f"{side}_qb_changed"], out[f"{side}_qb_new"] = changed, new
    return out


def expected_starters(depth_chart: pd.DataFrame, out_ids: set[str] | None = None) -> dict[str, str | None]:
    """{team: gsis_id} of each team's expected starting QB: the first QB on its depth chart who is not reported Out.

    `depth_chart` is the nflverse chart for the week (columns club_code, position, depth_team (a STRING '1','2',...),
    gsis_id). A team with no usable QB maps to None, never to a guess.
    """
    out_ids = set(out_ids or ())
    if depth_chart is None or depth_chart.empty:
        return {}
    qbs = depth_chart[depth_chart["position"] == "QB"].copy()
    qbs["slot"] = pd.to_numeric(qbs["depth_team"], errors="coerce")
    result: dict[str, str | None] = {}
    for team, grp in qbs.groupby("club_code"):
        ordered = grp.sort_values("slot")
        pick = next((g for g in ordered["gsis_id"] if isinstance(g, str) and g not in out_ids), None)
        result[team] = pick
    return result
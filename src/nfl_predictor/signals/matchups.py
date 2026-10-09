"""NFL offence-versus-defence duels for one game, from play-by-play efficiency known BEFORE the game.

Each duel is one side's attack (EPA/play, pass or rush) against the other side's matching defence, expressed as league
ranks. Only games of `season` strictly before `as_of`, and each team's last `WINDOW` of them, are used: early in a season
there are fewer than `MIN_GAMES` and the answer is no duels, which is better than ranking last year's roster.
"""
from __future__ import annotations

import json
from pathlib import Path

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


def load_history_gaps(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Past absolute rank gaps per duel type, from `data/duel_gaps.json` (written by the Task 10 tool
    from the walk-forward games). An absent file or an absent type yields {}, which sends
    `edge_strength` down the gap-scaled fallback (Task 1) instead of a percentile."""
    path = Path(path) if path is not None else Path(__file__).resolve().parents[3] / "data" / "duel_gaps.json"
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        if isinstance(v, list) and v:
            try:
                out[k] = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                continue
    return out


def _recent_means(efficiency: pd.DataFrame, games_df: pd.DataFrame, as_of: pd.Timestamp, season: int) -> pd.DataFrame:
    """Last WINDOW games per team, from `season` strictly before `as_of`, with at least MIN_GAMES of data."""
    if efficiency.empty or games_df.empty:
        return pd.DataFrame(columns=efficiency.columns) if not efficiency.empty else pd.DataFrame()
    meta = games_df[["game_id", "gameday", "season"]].copy()
    meta["gameday"] = pd.to_datetime(meta["gameday"])
    efficiency_idxed = efficiency.merge(meta, on="game_id", how="left")
    eff = efficiency_idxed.assign(
        gameday=pd.to_datetime(efficiency_idxed["gameday"]),
        season=efficiency_idxed["season"],
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
        for attacker_side in ("home", "away"):
            d = make_duel(
                f"{duel_id}:{attacker_side}",
                home=home, away=away, attacker_side=attacker_side,
                attack_ranks=attack_ranks, defence_ranks=defence_ranks,
                history_gaps=history_gaps.get(duel_id, np.array([])), min_gap=min_gap,
                stat=attack_noun, foil=defence_noun,
            )
            if d is not None:
                out.append(d)
    out.sort(key=lambda d: d.strength, reverse=True)
    return out


def _headline_ids(duels: list[Duel]) -> set[str]:
    """The duel(s) the residual-lift headline lens validates per type: the
    largest |rank gap|, ties to the home attack -- the same rule build_rows
    (duel_lift.py) uses to sample one observation per game+type. Everything
    else is a direction the lift never measured."""
    best: dict[str, tuple[float, str]] = {}
    for d in duels:
        t = d.id.split(":")[0]
        gap = abs(d.attacker_rank - d.defender_rank)
        cur = best.get(t)
        if cur is None or gap > cur[0] or (gap == cur[0] and d.id.endswith(":home")):
            best[t] = (gap, d.id)
    return {d_id for _, d_id in best.values()}


def to_context(duels: list[Duel], pick_side: str | None, limit: int = 4,
               lift_gate: dict[str, bool] | None = None) -> list[dict]:
    """Facts-bundle form. `toward_pick` is True when the duel favours the pick
    AND the residual-lift gate (Task 10) has proven the duel's TYPE: `lift_gate`
    maps a duel type (\"pass_off_vs_pass_def\", id without the :side suffix) to
    whether it passes. Unproven (missing/failing) types ship `toward_pick` None
    -- neutral context, never Edge or Risk.

    The gate is FAIL CLOSED: `lift_gate=None` behaves exactly like `{}` (nothing
    is proven), so there is no "ungated" mode -- a caller that forgets to wire
    the gate can never emit Edge/Risk-capable rows before a type is proven.

    Within a proven type only the HEADLINE duel may be directed: build_rows
    (duel_lift.py) samples one duel per game+type -- the largest |rank gap|,
    ties to the home attack -- and the opposite-direction duel of the same type
    favours the other side, which the lift never measured. Directing it would
    certify an Edge/Risk the gate did not validate, so it stays neutral.
    """
    headline = _headline_ids(duels[:limit])
    out: list[dict] = []
    for d in duels[:limit]:
        proven = (lift_gate or {}).get(d.id.split(":")[0], False)
        toward_pick = None if (pick_side is None or not proven or d.id not in headline) else (d.toward == pick_side)
        out.append({
            "id": d.id,
            "attacker": d.attacker,
            "defender": d.defender,
            "stat": d.stat,
            "foil": d.foil,
            "attacker_rank": d.attacker_rank,
            "defender_rank": d.defender_rank,
            "n_teams": d.n_teams,
            "toward_pick": toward_pick,
        })
    return out
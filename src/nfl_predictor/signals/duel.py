"""Matchup duels: copied identically into NFL/CFB/PL/NBA repos."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Duel:
    id: str; attacker: str; defender: str; stat: str; foil: str
    attacker_rank: int; defender_rank: int; n_teams: int; toward: str; strength: float

def ranks(values: dict[str,float], higher_is_better: bool=True) -> dict[str,int]:
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    out, last_v, last_r = {}, None, 0
    for i,(team,v) in enumerate(ordered, start=1):
        last_r = last_r if v==last_v else i
        last_v = v
        out[team] = last_r
    return out

def edge_strength(gap: float, history: np.ndarray, n_teams: int|None=None) -> float:
    history = np.abs(np.asarray(history,float))
    if len(history)==0:
        n = n_teams or 32
        pos = float(abs(gap))
        return min(1.0, max(0.01, pos / (n/2)))
    return float(np.mean(np.abs(history) >= abs(gap))) if len(history) else 0.5

def make_duel(id_, attacker, defender, home, away, attack_ranks, defence_ranks, history_gaps, min_gap=8):
    if attacker not in attack_ranks or defender not in defence_ranks: return None
    gap = attack_ranks[attacker] - defence_ranks[defender]
    if abs(gap) < min_gap: return None
    n_teams = max(len(attack_ranks), len(defence_ranks))
    return Duel(
        id=id_, attacker=attacker, defender=defender, stat="", foil="",
        attacker_rank=attack_ranks[attacker], defender_rank=defence_ranks[defender],
        n_teams=n_teams, toward="home" if attacker==home else "away",
        strength=edge_strength(gap, history_gaps or [], n_teams)
    )

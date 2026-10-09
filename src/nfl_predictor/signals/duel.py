"""Matchup duels: one team's strength against the other's weakness, in ranks a fan can read.

A duel compares an ATTACK stat of one side with the DEFENCE stat of the other. Ranks are 1 = best of `n_teams`. The
duel's strength is how unusual the rank gap is against every past duel of the same kind, so a 25-place gap in a stat
that is usually noisy does not outrank a 25-place gap in a stable one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Duel:
    id: str            # stable key, e.g. "pass_off_vs_pass_def"
    attacker: str      # team whose strength is used
    defender: str      # team whose weakness is tested
    stat: str          # human noun, e.g. "passing offence"
    foil: str          # human noun for the other side, e.g. "pass defence"
    attacker_rank: int
    defender_rank: int
    n_teams: int
    toward: str        # "home" | "away": which side the duel favours
    strength: float    # 0..1, percentile of |gap| among past duels of this kind


def ranks(values: dict[str, float], higher_is_better: bool = True) -> dict[str, int]:
    """1 = best. Ties share the better rank so a tie never invents an ordering."""
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    out, last_v, last_r = {}, None, 0
    for i, (team, v) in enumerate(ordered, start=1):
        last_r = last_r if v == last_v else i
        last_v = v
        out[team] = last_r
    return out


def edge_strength(gap: float, history: np.ndarray, n_teams: int | None = None) -> float:
    """Percentile of |gap| among past |gaps|. With no history the gap is scaled by the league size instead, so duels
    still order by how big the gap is rather than all tying at zero and keeping declaration order."""
    history = np.abs(np.asarray(history, float))
    if history.size == 0:
        return min(1.0, abs(gap) / (n_teams - 1)) if n_teams and n_teams > 1 else 0.0
    return float((history <= abs(gap)).mean())


def make_duel(duel_id: str, stat: str, foil: str, *, home: str, away: str, attacker_side: str,
              attack_ranks: dict[str, int], defence_ranks: dict[str, int], history_gaps, min_gap: int = 8) -> Duel | None:
    """The duel of `attacker_side`'s attack against the other side's defence, or None when it is too close to call.

    A rank gap under `min_gap` places says nothing, so the duel is not produced rather than produced weak.
    """
    attacker = home if attacker_side == "home" else away
    defender = away if attacker_side == "home" else home
    a, d = attack_ranks.get(attacker), defence_ranks.get(defender)
    if a is None or d is None:
        return None
    gap = d - a  # positive: the defence ranks WORSE than the attack ranks, so the attack has the edge
    if abs(gap) < min_gap:
        return None
    toward = attacker_side if gap > 0 else ("away" if attacker_side == "home" else "home")
    return Duel(duel_id, attacker, defender, stat, foil, a, d, len(attack_ranks), toward,
                edge_strength(gap, history_gaps, len(attack_ranks)))

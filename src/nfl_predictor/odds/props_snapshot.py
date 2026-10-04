"""props_snapshot.py -- live player-prop lines for the forward test.

`GET {BASE}/americanfootball_nfl/events/{event_id}/odds` with
`oddsFormat=american&regions=us&markets=<list>`; one call per event.

**The free tier is 500 credits a month, so the budget is checked before the
first request, not after.** A slate of 12 games x 4 markets can exceed what is
left, and a guard that discovers this mid-slate writes a partial record that
looks like a real one. `credits_sufficient` returning False aborts the tick
with nothing written.

Player names are normalized and matched exactly on (normalized name, team).
Never fuzzily: a near-match attaches a real price to the wrong player, and the
forward test would grade a bet nobody made. Unmatched props are logged and
skipped.
"""
from __future__ import annotations

import logging
import re

import requests

from ..config import ODDS_API_BASE_URL, ODDS_API_KEY, ODDS_API_SPORT_KEY

logger = logging.getLogger(__name__)

#: Markets the forward test asks for. `player_receptions` is secondary and is
#: simply absent from most free-tier responses; that is fine.
PROP_MARKETS = ["player_pass_yds", "player_rush_yds", "player_rec_yds",
                "player_reception_yds", "player_receptions"]

#: Book-side market keys mapped onto this project's market names, because the
#: two do not agree (`player_rec_yds` vs `player_reception_yds`).
MARKET_ALIASES = {
    "player_pass_yds": "player_pass_yds",
    "player_rush_yds": "player_rush_yds",
    "player_rec_yds": "player_rec_yds",
    "player_reception_yds": "player_rec_yds",
    "player_receptions": "player_receptions",
}

#: Generational suffixes that carry no identity.
_SUFFIXES = ("jr", "sr", "ii", "iii", "iv")

#: Punctuation becomes nothing, not a space: "A.J." is one initial, and turning
#: the dots into separators yields "a j brown", which will never match the book's
#: "aj brown".
_PUNCTUATION = re.compile(r"[^a-z]")
_WHITESPACE = re.compile(r"\s+")

#: nflverse abbreviates a first name where the book spells it out ("P. Mahomes"
#: vs "Patrick Mahomes"). Expanded by initial only, and only when the expansion
#: is the whole first token -- an explicit map, never a guess, so an unknown
#: initial stays a miss rather than becoming a wrong player.
_INITIALS = {"p": "patrick", "m": "matthew", "c": "calvin", "d": "david",
             "j": "john", "k": "kyle", "r": "ryan"}


class BudgetExhausted(RuntimeError):
    """Not enough credits left for the whole slate. Nothing was fetched."""


def _collapse_initials(words: list[str]) -> list[str]:
    """Rejoin single letters into one token: "a j brown" -> "aj brown".

    Both spellings occur in the wild -- nflverse writes "A.J. Brown", a book
    writes "AJ Brown" -- so the normalised form has to agree with itself.
    """
    collapsed: list[str] = []
    for word in words:
        if len(word) == 1 and collapsed and len(collapsed[-1]) == 1:
            collapsed[-1] += word
        else:
            collapsed.append(word)
    return collapsed


def normalize_player_name(name: str) -> str:
    """`"A.J. Brown Jr."` -> `"aj brown"`.

    Lowercased, apostrophes and periods dropped, generational suffixes removed,
    single letters rejoined, whitespace collapsed. Deterministic and total: the
    same input always gives the same key, which is what makes an exact match
    meaningful.
    """
    text = _PUNCTUATION.sub(" ", str(name).lower().replace("'", ""))
    words = [w for w in _WHITESPACE.split(text) if w and w not in _SUFFIXES]
    words = _collapse_initials(words)
    # A lone initial is the abbreviation nflverse uses where a book spells the
    # name out ("P. Mahomes" vs "Patrick Mahomes"). Expanded from an explicit
    # map, never guessed: an unmapped initial stays a miss, which is safe.
    if len(words) >= 2 and len(words[0]) == 1 and words[0] in _INITIALS:
        words[0] = _INITIALS[words[0]]
    return " ".join(words)


def event_odds_url(event_id: str) -> str:
    return f"{ODDS_API_BASE_URL}/{ODDS_API_SPORT_KEY}/events/{event_id}/odds"


def _remaining_from(response) -> int | None:
    raw = response.headers.get("x-requests-remaining")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _probe_credits() -> int | None:
    """Remaining credits, or None if the API did not say.

    The scores endpoint is the cheapest call the API offers and is the only way
    to ask without spending a props credit.
    """
    try:
        response = requests.get(
            f"{ODDS_API_BASE_URL}/{ODDS_API_SPORT_KEY}/scores",
            params={"apiKey": ODDS_API_KEY}, timeout=15,
        )
    except requests.RequestException as error:
        logger.warning("credit probe failed: %s", error)
        return None
    return _remaining_from(response)


def credits_sufficient(needed: int, probe=None) -> bool:
    """True only when the remaining credits are known and cover `needed`.

    Unknown is False. The free tier is 500/month and an unmeasured tick that
    spends the last of it is worse than a tick that does not run.
    """
    remaining = probe() if probe is not None else _probe_credits()
    return remaining is not None and remaining >= needed


def _response_rows(event_id: str, markets: list[str]) -> list[dict]:
    response = requests.get(
        event_odds_url(event_id),
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _normalize_rows(rows: list[dict]) -> list[dict]:
    """Flatten the API's per-book/per-market nesting, dropping rows with no line.

    A prop with no line is not a prop. Substituting a default would post a real
    bet at a fabricated price.
    """
    normalized: list[dict] = []
    for row in rows:
        line = row.get("line")
        if line is None:
            continue
        name = row.get("player_name") or row.get("description", "")
        normalized.append({
            "player_name": name,
            "normalized_name": normalize_player_name(name),
            "team": row.get("team"),
            "market": MARKET_ALIASES.get(row.get("market", ""), row.get("market")),
            "line": float(line),
            "over_odds": _american(row.get("over_odds")),
            "under_odds": _american(row.get("under_odds")),
            "book": row.get("book_title") or row.get("book"),
        })
    return normalized


def _american(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_props_for_event(event_id: str, markets: list[str] | None = None,
                          credits_needed: int = 1) -> list[dict]:
    """Props for one event. Raises `BudgetExhausted` before spending anything."""
    markets = markets or PROP_MARKETS
    if not credits_sufficient(credits_needed):
        raise BudgetExhausted(
            f"need {credits_needed} credits for event {event_id}, "
            f"remaining is below that or unknown; snapshotting nothing")
    return _normalize_rows(_response_rows(event_id, markets))


def probe_props_coverage(event_id: str) -> dict:
    """Which markets and books the free tier actually returns (spec Q1).

    Reports emptiness as emptiness. There is deliberately no fallback to game
    lines: a total is not a player prop, and substituting one would put a
    number on a market nobody quoted.
    """
    try:
        rows = _normalize_rows(_response_rows(event_id, PROP_MARKETS))
    except BudgetExhausted:
        return {"has_any": False, "markets": {}, "books": [], "budget_exhausted": True}

    markets: dict[str, list[str]] = {}
    for row in rows:
        markets.setdefault(row["market"], [])
        if row["book"] not in markets[row["market"]]:
            markets[row["market"]].append(row["book"])
    return {
        "has_any": bool(rows),
        "markets": markets,
        "books": sorted({row["book"] for row in rows if row["book"]}),
        "budget_exhausted": False,
    }


def match_props_to_players(props: list[dict], players: list[dict]) -> list[dict]:
    """Attach `player_id` on an exact (normalized name, team) match.

    A miss is logged and dropped. Same name on the wrong team is a miss.
    """
    index = {(normalize_player_name(p["player_name"]), p.get("team")): p["player_id"]
             for p in players}

    matched = []
    for prop in props:
        key = (prop["normalized_name"], prop.get("team"))
        player_id = index.get(key)
        if player_id is None:
            logger.warning("unmatched prop, skipped: %s %s", prop["player_name"], prop.get("team"))
            continue
        matched.append({**prop, "player_id": player_id})
    return matched
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


#: nflverse abbreviations -> the full team names The Odds API returns.
#:
#: Data, not logic: 32 rows. The `/events` endpoint is free and returns full
#: names, while the slate carries abbreviations, so something has to bridge them
#: or the two never join. Every nflverse abbreviation in the 2026 schedule is
#: covered, and `test_every_scheduled_abbreviation_is_mapped` keeps it that way.
TEAM_NAMES: dict[str, str] = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
    "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def events_url() -> str:
    return f"{ODDS_API_BASE_URL}/{ODDS_API_SPORT_KEY}/events"


def fetch_event_index() -> dict[tuple[str, str], str]:
    """{(home, away): odds_event_id} from the FREE `/events` list.

    Free: The Odds API does not charge for the events list, only for odds. This
    is what makes the mapping affordable -- without it a tick would have to spend
    a credit per game just to discover the event id.

    The event id is an opaque string (`e91a...`), NOT an nflverse `game_id`
    (`2026_05_ALB_DEN`), so it cannot be derived and must be looked up.
    """
    response = requests.get(
        events_url(),
        params={"apiKey": ODDS_API_KEY, "regions": "us"}, timeout=30,
    )
    response.raise_for_status()
    index: dict[tuple[str, str], str] = {}
    for event in response.json() or []:
        key = (event.get("home_team"), event.get("away_team"))
        event_id = event.get("id")
        if all(key) and event_id:
            index[key] = event_id
    return index


def match_event_id(game: dict, index: dict[tuple[str, str], str]) -> str | None:
    """The Odds API event id for a slate game, or None if it is not offered.

    None means the book is not listing that game -- it is NOT a reason to guess,
    and the caller must skip rather than spend a credit discovering it.
    """
    home = TEAM_NAMES.get(game.get("home_team"))
    away = TEAM_NAMES.get(game.get("away_team"))
    if home is None or away is None:
        logger.warning("unmapped team in %s: %s at %s", game.get("game_id"),
                       game.get("away_team"), game.get("home_team"))
        return None
    return index.get((home, away))


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


def _normalize_rows(payload) -> list[dict]:
    """Flatten the API's response into one row per player/market/book.

    The single-event odds endpoint nests:
        bookmakers[] -> markets[] -> outcomes[]
    where each prop appears TWICE, as an `Over` and an `Under` outcome sharing a
    `point`, and the player is in `description`. Reading that as a flat row with
    `line`/`player_name`/`over_odds` keys -- which is what the first version of
    this function did -- returns nothing at all against a real response, because
    none of those keys exist at the top level.

    Over and Under outcomes are paired back together on (description, point) so a
    row carries both prices. A lone unpaired outcome yields a row with one side
    missing rather than being dropped: the line is still a real line.
    """
    # The endpoint returns ONE event object; some responses and some callers
    # hand back a list of them. Accept either rather than assuming one.
    events = payload if isinstance(payload, list) else [payload]
    rows = []
    for event in events:
        rows.extend(_rows_from_event(event))
    return [r for r in (_prop_row(r) for r in rows) if r is not None]


def _rows_from_event(event) -> list[dict]:
    rows = []
    for bookmaker in _as_list(event, "bookmakers"):
        book = bookmaker.get("title") or bookmaker.get("key")
        for market in _as_list(bookmaker, "markets"):
            book_key = market.get("key", "")
            pairs: dict[tuple, dict] = {}
            for outcome in _as_list(market, "outcomes"):
                point = outcome.get("point")
                if point is None:
                    continue
                name = outcome.get("description") or outcome.get("player_name") or ""
                key = (name, float(point))
                slot = pairs.setdefault(key, {"name": name, "line": float(point),
                                              "over_odds": None, "under_odds": None})
                side = str(outcome.get("name", "")).strip().lower()
                if side == "over":
                    slot["over_odds"] = _american(outcome.get("price"))
                elif side == "under":
                    slot["under_odds"] = _american(outcome.get("price"))
            for slot in pairs.values():
                rows.append({**slot, "market": MARKET_ALIASES.get(book_key, book_key),
                             "book": book})
    return rows


def _as_list(container: dict, key: str) -> list[dict]:
    value = container.get(key) if isinstance(container, dict) else None
    return value if isinstance(value, list) else []


def _prop_row(row: dict) -> dict | None:
    """One normalized prop row, or None when it carries no line.

    A prop with no line is not a prop; substituting a default would post a real
    bet at a fabricated price.
    """
    line = row.get("line")
    name = row.get("name") or ""
    if line is None or not name:
        return None
    return {
        "player_name": name,
        "normalized_name": normalize_player_name(name),
        "team": row.get("team"),
        "market": row.get("market"),
        "line": float(line),
        "over_odds": row.get("over_odds"),
        "under_odds": row.get("under_odds"),
        "book": row.get("book"),
    }


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
    """Attach `player_id` to each prop.

    **On the normalized name alone.** The event-odds endpoint puts the player in
    the outcome's `description` and does not carry a team per outcome, so a
    (name, team) key can never match against a real response.

    A name that maps to more than one nflverse player is AMBIGUOUS and skipped,
    not guessed: a near-match attaches a real price to the wrong player, and the
    forward test would then grade a bet nobody made. Where the prop does carry a
    team it is used to disambiguate.
    """
    by_name: dict[str, list[dict]] = {}
    for player in players:
        by_name.setdefault(normalize_player_name(player["player_name"]), []).append(player)

    matched = []
    for prop in props:
        candidates = by_name.get(prop["normalized_name"], [])
        if prop.get("team"):
            narrowed = [c for c in candidates if c.get("team") == prop["team"]]
            candidates = narrowed or candidates
        if not candidates:
            logger.warning("unmatched prop, skipped: %s", prop["player_name"])
            continue
        if len(candidates) > 1:
            logger.warning("ambiguous prop name, skipped rather than guessed: %s "
                           "(%d nflverse players share it)", prop["player_name"],
                           len(candidates))
            continue
        matched.append({**prop, "player_id": candidates[0]["player_id"]})
    return matched
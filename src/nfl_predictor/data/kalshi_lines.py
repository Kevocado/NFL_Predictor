"""Collect Kalshi's per-game NFL player-stat markets as a free, public record of what the market thought.

Kalshi lists a ladder of "Player: N+ stat" markets for each game (passing/rushing/receiving yards, receptions). Every rung is
a binary contract whose price is the market's probability that the player reaches N. Reading them needs no account, key or
credits. Snapshots taken before kickoff are the "lines" we had no historical source for; the same markets settle afterwards
with a yes/no `result`, so each snapshot is its own labelled outcome and needs no player_stats join to be scored.

`floor_strike` is the real threshold: "325+" has floor_strike 324.5, i.e. the over of a 324.5 line. Prices are dollars in
[0, 1] (0.62 = 62%). A missing or zero bid/ask means no quote, not a 0% market.
"""
from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
#: Kalshi series ticker -> the stat it prices.
SERIES = {
    "KXNFLPASSYDS": "passing_yards",
    "KXNFLRSHYDS": "rushing_yards",
    "KXNFLRECYDS": "receiving_yards",
    "KXNFLREC": "receptions",
}
LINES_DIR = Path("data/lines")
PAGE_LIMIT = 1000
SETTLED_LOOKBACK_DAYS = 4


def _dollars(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v


def normalize(market: dict, series: str, snapshot_ts: str) -> dict:
    """One flat row per market rung. Pure; the unit of everything downstream."""
    title = market.get("title") or ""
    player = title.split(":", 1)[0].strip() if ":" in title else None
    return {
        "snapshot_ts": snapshot_ts,
        "series": series,
        "stat": SERIES.get(series),
        "ticker": market.get("ticker"),
        "event_ticker": market.get("event_ticker"),
        "player": player,
        "line": _dollars(market.get("floor_strike")),
        "yes_bid": _dollars(market.get("yes_bid_dollars")),
        "yes_ask": _dollars(market.get("yes_ask_dollars")),
        "no_bid": _dollars(market.get("no_bid_dollars")),
        "no_ask": _dollars(market.get("no_ask_dollars")),
        "last": _dollars(market.get("last_price_dollars")),
        "volume": _dollars(market.get("volume_fp")),
        "open_interest": _dollars(market.get("open_interest_fp")),
        "status": market.get("status"),
        "result": market.get("result") or None,
        "settlement_value": market.get("expiration_value") or None,
        "occurrence": market.get("occurrence_datetime") or market.get("expected_expiration_time"),
        "close_time": market.get("close_time"),
    }


def _get(session, path: str, params: dict) -> dict:
    for attempt in range(4):
        r = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if r.status_code == 429:  # public API rate limit: back off, never hammer
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi rate-limited {path} after retries")


def fetch_series(session, series: str, status: str, min_close_ts: int | None = None) -> list[dict]:
    markets: list[dict] = []
    cursor = None
    while True:
        params = {"series_ticker": series, "status": status, "limit": PAGE_LIMIT}
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if cursor:
            params["cursor"] = cursor
        body = _get(session, "/markets", params)
        markets.extend(body.get("markets") or [])
        cursor = body.get("cursor")
        if not cursor:
            return markets


def collect(session=None, now: datetime | None = None) -> list[dict]:
    """Open markets (the pre-game lines) plus the last few days of settled ones (their outcomes)."""
    session = session or requests.Session()
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat(timespec="seconds")
    since = int((now - timedelta(days=SETTLED_LOOKBACK_DAYS)).timestamp())
    rows: list[dict] = []
    for series in SERIES:
        for status, min_close in (("open", None), ("settled", since)):
            for m in fetch_series(session, series, status, min_close):
                rows.append(normalize(m, series, ts))
    return rows


def write_snapshot(rows: list[dict], out_dir: Path = LINES_DIR, now: datetime | None = None) -> Path | None:
    """Append to today's gzip JSONL file (one file per UTC day, so the repo grows by one small file a day)."""
    if not rows:
        return None
    now = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"kalshi_nfl_props_{now:%Y-%m-%d}.jsonl.gz"
    with gzip.open(path, "at", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def main() -> int:
    rows = collect()
    path = write_snapshot(rows)
    open_rows = sum(1 for r in rows if r["status"] == "active")
    print(f"kalshi lines: {len(rows)} rows ({open_rows} open) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------------------------------------------------
# Live ladders for the line checker: what Kalshi is pricing RIGHT NOW, matched to our players.
# ---------------------------------------------------------------------------------------------------------------------
import re
import threading

LADDER_TTL_SECONDS = 600
#: Kalshi's team codes that differ from nflverse's (ours: JAX, LA, WAS).
TEAM_ALIAS = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS"}
_TEAM_CODES = sorted(
    ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAC", "JAX", "KC",
     "LAC", "LAR", "LA", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS", "WSH"],
    key=len, reverse=True)
_ladder_lock = threading.Lock()
_ladder_cache: tuple[float, dict] | None = None


def parse_player_token(ticker: str, event_ticker: str) -> dict | None:
    """('KXNFLPASSYDS-26OCT18CARPHI-PHIJHURTS1-325', 'KXNFLPASSYDS-26OCT18CARPHI') -> team PHI, initial J, last HURTS.

    The player segment is TEAM + first initial + LASTNAME + jersey. The two teams in the event segment are glued together
    ('CARPHI'), so the player's team is the longest known code that starts the token AND occurs in that glued string.
    """
    try:
        parts = ticker.split("-")
        player_segment = parts[2]
        glued = event_ticker.split("-")[1][7:]          # '26OCT18' + teams
    except IndexError:
        return None
    team = next((code for code in _TEAM_CODES if player_segment.startswith(code) and code in glued), None)
    if team is None:
        return None
    m = re.match(r"([A-Z])([A-Z]+?)(\d+)$", player_segment[len(team):])
    if not m:
        return None
    return {"team": TEAM_ALIAS.get(team, team), "initial": m.group(1), "last": m.group(2)}


def implied_probability(row: dict) -> float | None:
    """Kalshi's price for 'reaches N+' as a probability: the bid/ask midpoint, else the last trade. None = no quote."""
    bid, ask = row.get("yes_bid"), row.get("yes_ask")
    if bid is not None and ask is not None and ask > 0 and bid >= 0 and ask >= bid:
        return round((bid + ask) / 2, 4)
    last = row.get("last")
    return last if last else None


def build_ladders(rows: list[dict]) -> list[dict]:
    """Group open rungs by (team, initial, last, stat). `line` is the real threshold (324.5 for '325+')."""
    grouped: dict[tuple, dict] = {}
    for r in rows:
        if r.get("status") not in ("active", "open") or r.get("line") is None or r.get("stat") is None:
            continue
        who = parse_player_token(r.get("ticker") or "", r.get("event_ticker") or "")
        p = implied_probability(r)
        if who is None or p is None:
            continue
        key = (who["team"], who["initial"], who["last"], r["stat"])
        entry = grouped.setdefault(key, {**who, "stat": r["stat"], "player": r.get("player"), "rungs": []})
        entry["rungs"].append({"line": r["line"], "p_over": p, "volume": r.get("volume")})
    out = list(grouped.values())
    for e in out:
        e["rungs"].sort(key=lambda x: x["line"])
    return out


def current_ladders(session=None, now: float | None = None) -> dict:
    """Cached for LADDER_TTL_SECONDS so a busy page costs Kalshi one fetch per ten minutes. Any failure serves the last
    good value (or an empty list): a missing market comparison must never break the line checker."""
    import time as _time
    global _ladder_cache
    now = _time.time() if now is None else now
    with _ladder_lock:
        if _ladder_cache and now - _ladder_cache[0] < LADDER_TTL_SECONDS:
            return _ladder_cache[1]
        try:
            session = session or requests.Session()
            ts = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
            rows = []
            for series in SERIES:
                rows += [normalize(m, series, ts) for m in fetch_series(session, series, "open")]
            data = {"fetched_at": ts, "lines": build_ladders(rows)}
        except Exception:  # noqa: BLE001 - enhancement; stale or empty beats an error page
            return _ladder_cache[1] if _ladder_cache else {"fetched_at": None, "lines": []}
        _ladder_cache = (now, data)
        return data

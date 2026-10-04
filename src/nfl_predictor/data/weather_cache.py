"""weather_cache.py -- one JSON file per game, so a rerun costs no HTTP calls.

A failure is deliberately NOT cached. The archive endpoint lags real time by a
few days, so the newest games legitimately cannot be fetched yet; remembering
that failure would drop them from training permanently instead of on the rerun
that would have worked.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .weather import STADIUM_COORDS, fetch_game_weather, is_outdoor

logger = logging.getLogger(__name__)


def load_cached_weather(cache_dir: Path | str) -> dict[str, dict]:
    """Every readable cache file, keyed by game_id. A corrupt file is skipped."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {}
    cached: dict[str, dict] = {}
    for path in sorted(cache_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            cached[path.stem] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("skipping unreadable weather cache file %s", path.name)
    return cached


def write_weather_cache(weather: dict[str, dict], cache_dir: Path | str) -> int:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for game_id, reading in weather.items():
        (cache_dir / f"{game_id}.json").write_text(json.dumps(reading))
    return len(weather)


#: Open-Meteo's free tier is generous but not unbounded. ~2/s stays under it and
#: costs ~4 minutes over a full rebuild, against ~14 minutes of requests.
REQUEST_INTERVAL_SECONDS = 0.2


def _kickoff_utc(game: dict) -> str:
    """The game's kickoff as an ISO-8601 UTC instant.

    nflverse publishes `gametime` as an ET clock string ("20:20"); `gameday` is a
    bare date. Eastern is UTC-4 from the second Sunday in March to the first in
    November and UTC-5 either side, which is enough resolution for weather.
    """
    from datetime import datetime, timedelta, timezone

    raw = str(game.get("gametime") or "").strip()
    day = str(game.get("gameday"))
    try:
        kickoff_et = datetime.fromisoformat(f"{day}T{raw or '13:00'}")
    except ValueError:
        kickoff_et = datetime.fromisoformat(f"{day}T13:00")

    year = kickoff_et.year
    dst_start = datetime(year, 3, 1)  # second Sunday, approximated below
    dst_start += timedelta(days=(6 - dst_start.weekday()) % 7)
    dst_end = datetime(year, 11, 1)
    dst_end += timedelta(days=(6 - dst_end.weekday()) % 7)
    in_dst = dst_start < kickoff_et < dst_end
    # Eastern is BEHIND UTC, so ET -> UTC ADDS the offset. Subtracting would put a
    # 1pm EST kickoff at 08:00Z, eight hours early.
    return (kickoff_et + timedelta(hours=4 if in_dst else 5)).replace(
        tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def weather_for_games(games: list[dict], cache_dir: Path | str,
                      fetch=fetch_game_weather) -> tuple[dict[str, dict], dict[str, dict]]:
    """(newly written, already cached) readings for `games`.

    `games` are dicts with game_id, stadium and gameday. Roofed stadiums and
    stadiums absent from the coords table are skipped rather than fetched --
    there is no reading to take, and a request would waste credits for a NaN.
    """
    cache_dir = Path(cache_dir)
    cached = load_cached_weather(cache_dir)

    fresh: dict[str, dict] = {}
    for game in games:
        game_id = game["game_id"]
        stadium = game.get("stadium")
        coords = STADIUM_COORDS.get(stadium)
        if coords is None or not is_outdoor(stadium):
            continue
        if game_id in cached or game_id in fresh:
            continue
        # The KICKOFF hour, not midnight. `T00:00:00Z` on the game date is the
        # evening before an afternoon US kickoff, so every reading was hours off.
        # nflverse carries `gametime` as HH:MM ET; ET->UTC is +4 (EDT) or +5 (EST).
        try:
            fresh[game_id] = fetch(coords[0], coords[1], _kickoff_utc(game))
        except Exception as error:  # noqa: BLE001 - one bad game must not stop the rest
            # NOT cached, so a re-run retries it -- which is only safe because a
            # failure must never be mistaken for "no weather signal".
            logger.warning("weather unavailable for %s: %s", game_id, error)
        # Paced. A full rebuild is ~1.3k requests; unpaced it trips Open-Meteo's
        # rate limit, and every 429 becomes a silent miss -- the same shape as the
        # wrong-endpoint bug that left all weather columns NaN and read as "no
        # weather signal".
        time.sleep(REQUEST_INTERVAL_SECONDS)

    write_weather_cache(fresh, cache_dir)
    return fresh, {g: cached[g] for g in cached if any(x["game_id"] == g for x in games)}
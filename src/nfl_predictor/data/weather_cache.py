"""weather_cache.py -- one JSON file per game, so a rerun costs no HTTP calls.

A failure is deliberately NOT cached. The archive endpoint lags real time by a
few days, so the newest games legitimately cannot be fetched yet; remembering
that failure would drop them from training permanently instead of on the rerun
that would have worked.
"""
from __future__ import annotations

import json
import logging
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
        try:
            fresh[game_id] = fetch(coords[0], coords[1], f"{game['gameday']}T00:00:00Z")
        except Exception as error:  # noqa: BLE001 - one bad game must not stop the rest
            logger.warning("weather unavailable for %s: %s", game_id, error)

    write_weather_cache(fresh, cache_dir)
    return fresh, {g: cached[g] for g in cached if any(x["game_id"] == g for x in games)}
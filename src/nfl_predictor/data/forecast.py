"""Kickoff-hour forecast from Open-Meteo (free, keyless) as the small `conditions` object the sites draw as an icon.

Only games from now to 16 days out have a forecast. Anything else, and any failure, returns None: no chip beats a
wrong chip. Roofed stadiums return kind "dome" without a request.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from .weather import INDOOR_STADIUMS, STADIUM_COORDS

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HORIZON_DAYS = 16

#: WMO weather interpretation codes -> the handful of pictures the UI can draw.
_KINDS = {
    0: "clear", 1: "clear", 2: "partly", 3: "cloudy",
    45: "fog", 48: "fog",
    51: "rain", 53: "rain", 55: "rain", 56: "rain", 57: "rain", 61: "rain", 63: "rain", 65: "rain",
    66: "rain", 67: "rain", 80: "rain", 81: "rain", 82: "rain",
    71: "snow", 73: "snow", 75: "snow", 77: "snow", 85: "snow", 86: "snow",
    95: "storm", 96: "storm", 99: "storm",
}


def kind_for(code) -> str | None:
    """None for a code this table does not know: an unknown sky is not drawn as a guessed one."""
    try:
        return _KINDS.get(int(code))
    except (TypeError, ValueError):
        return None


def forecast_for(stadium: str, kickoff_utc_iso: str, now: datetime | None = None, get=requests.get) -> dict | None:
    if stadium in INDOOR_STADIUMS:
        return {"kind": "dome", "temp_f": None, "wind_mph": None, "precip_pct": None, "source": "roof"}
    coords = STADIUM_COORDS.get(stadium)
    if coords is None:
        return None
    now = now or datetime.now(timezone.utc)
    # Normalize any offset the caller's ISO carries before slicing: the request
    # and the hour match are UTC, so a "+02:00" instant must not read its local
    # date/hour off the raw string.
    kickoff = datetime.fromisoformat(kickoff_utc_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    if not (now - timedelta(hours=6) <= kickoff <= now + timedelta(days=HORIZON_DAYS)):
        return None
    day = kickoff.strftime("%Y-%m-%d")
    params = {
        "latitude": coords[0], "longitude": coords[1], "start_date": day, "end_date": day, "timezone": "UTC",
        "hourly": "temperature_2m,precipitation_probability,weather_code,wind_speed_10m",
    }
    try:
        response = get(FORECAST_URL, params=params, timeout=15)
        response.raise_for_status()
        hourly = response.json()["hourly"]
        i = next(i for i, t in enumerate(hourly["time"]) if t[:13] == kickoff.strftime("%Y-%m-%dT%H"))
        kind = kind_for(hourly["weather_code"][i])
        if kind is None:
            return None
        return {
            "kind": kind,
            "temp_f": round(hourly["temperature_2m"][i] * 9 / 5 + 32),
            "wind_mph": round(hourly["wind_speed_10m"][i] / 1.609344),
            "precip_pct": hourly["precipitation_probability"][i],
            "source": "open-meteo",
        }
    except Exception:  # noqa: BLE001 - any failure means "no chip", never a guess
        return None

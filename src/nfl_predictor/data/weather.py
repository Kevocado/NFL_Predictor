"""weather.py -- free Open-Meteo game weather. No API key, no paid tier.

`STADIUM_COORDS` keys are the exact `schedules.stadium` strings nflverse
publishes, because a key that does not match joins to nothing and silently
yields NaN weather for a whole stadium.
"""
from __future__ import annotations

import requests

#: **Historical** weather comes from the archive endpoint, not the forecast one.
#: `api.open-meteo.com/v1/forecast` only serves roughly the last three months
#: and answers anything older with HTTP 400 "start_date is out of allowed range".
#: A mocked test cannot see that, because the mock has no date range: a pull
#: built against the forecast URL silently returns nothing for every game in
#: 2017-2025 and the weather columns come out all-NaN, which looks like "no
#: weather signal" rather than "wrong endpoint". The archive endpoint has no
#: such limit.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

#: Recent games (inside the archive's usual lag) still resolve on the archive
#: endpoint, so one URL covers both training history and the forward test.
URL = ARCHIVE_URL

#: 15 mph, the plan's high-wind threshold, in the km/h Open-Meteo reports.
HIGH_WIND_KPH = 15 * 1.609344

#: nflverse stadium name -> (lat, lon).
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "AT&T Stadium": (32.7473, -97.0945),
    "Acrisure Stadium": (42.7738, -78.7870),
    "Allegiant Stadium": (36.0909, -115.1833),
    "Bank of America Stadium": (35.2258, -80.8528),
    "Empower Field at Mile High": (39.7439, -105.0201),
    "FedExField": (38.9076, -76.8645),
    "FirstEnergy Stadium": (41.5061, -81.6995),
    "Ford Field": (42.3400, -83.0456),
    "GEHA Field at Arrowhead Stadium": (39.0489, -94.4839),
    "Gillette Stadium": (42.0909, -71.2643),
    "Hard Rock Stadium": (25.9580, -80.2389),
    "Lambeau Field": (44.5013, -88.0622),
    "Levi's Stadium": (37.4030, -121.9698),
    "Lincoln Financial Field": (39.9008, -75.1675),
    "Lucas Oil Stadium": (39.7601, -86.1639),
    "Lumen Field": (47.5952, -122.3316),
    "M&T Bank Stadium": (39.2780, -76.6228),
    "Mercedes-Benz Stadium": (33.7554, -84.4008),
    "Mercedes-Benz Superdome": (29.9509, -90.0811),
    "MetLife Stadium": (40.8135, -74.0745),
    "NRG Stadium": (29.6847, -95.4107),
    "New Era Field": (42.7738, -78.7870),
    "Nissan Stadium": (36.1665, -86.7713),
    "Paycor Stadium": (39.0954, -84.5160),
    "Raymond James Stadium": (27.9759, -82.5033),
    "SoFi Stadium": (33.9535, -118.3392),
    "Soldier Field": (41.8623, -87.6167),
    "State Farm Stadium": (33.5276, -112.2626),
    "TIAA Bank Stadium": (30.3239, -81.6373),
    "U.S. Bank Stadium": (44.9738, -93.2575),
    # 2026 rename: the Bills' Orchard Park venue.
    "Highmark Stadium": (42.7738, -78.7870),
}

#: Fixed or retractable roof: weather does not reach the field, so these get
#: `is_outdoor == False` and their wind/temp features are dropped.
INDOOR_STADIUMS: frozenset[str] = frozenset({
    "Allegiant Stadium", "AT&T Stadium", "Ford Field", "Hard Rock Stadium",
    "Lucas Oil Stadium", "Mercedes-Benz Stadium", "Mercedes-Benz Superdome",
    "MetLife Stadium", "NRG Stadium", "SoFi Stadium", "State Farm Stadium",
    "U.S. Bank Stadium",
})


def is_outdoor(stadium: str) -> bool:
    return stadium not in INDOOR_STADIUMS


def fetch_game_weather(lat: float, lon: float, kickoff_iso: str) -> dict:
    """Hourly weather at kickoff. `kickoff_iso` is UTC, e.g. 2025-10-05T17:00:00Z.

    Raises LookupError when the response carries no entry for the kickoff hour,
    rather than falling back to the first hour -- a silently wrong wind reading
    becomes a silently wrong feature.
    """
    day = kickoff_iso[:10]
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "start_date": day, "end_date": day,
        "timezone": "UTC",
    }
    response = requests.get(URL, params=params, timeout=30)
    response.raise_for_status()

    hourly = response.json()["hourly"]
    idx = next((i for i, t in enumerate(hourly["time"]) if t[:13] == kickoff_iso[:13]), None)
    if idx is None:
        raise LookupError(f"Open-Meteo returned no {kickoff_iso[:13]} reading for {lat},{lon}")

    return {
        "temp_c": float(hourly["temperature_2m"][idx]),
        "wind_kph": float(hourly["wind_speed_10m"][idx]),
        "precip_mm": float(hourly["precipitation"][idx]),
    }
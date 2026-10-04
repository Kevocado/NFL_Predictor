"""weather tests -- the Open-Meteo fetch is mocked; nothing here hits the network."""
from __future__ import annotations

import pytest

from nfl_predictor.data.weather import (
    STADIUM_COORDS, fetch_game_weather, is_outdoor,
)

URL = "https://api.open-meteo.com/v1/forecast"


def _hourly(hour: int = 17, **overrides):
    payload = {
        "time": [f"2025-10-05T{hour:02d}:00"],
        "temperature_2m": [12.0],
        "wind_speed_10m": [25.0],
        "precipitation": [0.0],
    }
    payload.update(overrides)
    return {"hourly": payload}


def test_fetch_game_weather_shape(requests_mock):
    requests_mock.get(URL, json=_hourly())

    w = fetch_game_weather(41.88, -87.63, "2025-10-05T17:00:00Z")

    assert set(w) == {"temp_c", "wind_kph", "precip_mm"}
    assert w["temp_c"] == 12.0
    assert w["wind_kph"] == 25.0
    assert w["precip_mm"] == 0.0


def test_fetch_game_weather_requests_the_kickoff_date(requests_mock):
    m = requests_mock.get(URL, json=_hourly())

    fetch_game_weather(41.88, -87.63, "2025-10-05T17:00:00Z")

    q = m.last_request.qs
    assert q["start_date"] == ["2025-10-05"]
    assert q["end_date"] == ["2025-10-05"]
    assert q["latitude"] == ["41.88"]


def test_fetch_game_weather_picks_the_kickoff_hour_not_the_first(requests_mock):
    """The plan's version indexed positionally and read the wrong hour."""
    requests_mock.get(URL, json={
        "hourly": {
            "time": ["2025-10-05T00:00", "2025-10-05T17:00"],
            "temperature_2m": [1.0, 12.0],
            "wind_speed_10m": [3.0, 25.0],
            "precipitation": [0.5, 0.0],
        }
    })

    w = fetch_game_weather(41.88, -87.63, "2025-10-05T17:00:00Z")

    assert w["temp_c"] == 12.0
    assert w["wind_kph"] == 25.0
    assert w["precip_mm"] == 0.0


def test_missing_kickoff_hour_fails_loudly(requests_mock):
    """No silent fallback to hour 0 -- a wrong weather reading is a wrong feature."""
    requests_mock.get(URL, json=_hourly(hour=3))

    with pytest.raises(LookupError):
        fetch_game_weather(41.88, -87.63, "2025-10-05T17:00:00Z")


def test_http_error_is_not_swallowed(requests_mock):
    requests_mock.get(URL, status_code=500)

    with pytest.raises(Exception):
        fetch_game_weather(41.88, -87.63, "2025-10-05T17:00:00Z")


def test_is_outdoor_distinguishes_domes():
    assert is_outdoor("Soldier Field") is True
    assert is_outdoor("Lambeau Field") is True
    assert is_outdoor("Mercedes-Benz Superdome") is False
    assert is_outdoor("NRG Stadium") is False


def test_every_nflverse_stadium_has_coords():
    """Keys must match nflverse `schedules.stadium` exactly, or the join is silent."""
    known = {
        "AT&T Stadium", "Acrisure Stadium", "Allegiant Stadium",
        "Bank of America Stadium", "Empower Field at Mile High", "FedExField",
        "FirstEnergy Stadium", "Ford Field", "GEHA Field at Arrowhead Stadium",
        "Gillette Stadium", "Hard Rock Stadium", "Lambeau Field",
        "Levi's Stadium", "Lincoln Financial Field", "Lucas Oil Stadium",
        "Lumen Field", "M&T Bank Stadium", "Mercedes-Benz Stadium",
        "Mercedes-Benz Superdome", "MetLife Stadium", "NRG Stadium",
        "New Era Field", "Nissan Stadium", "Paycor Stadium",
        "Raymond James Stadium", "SoFi Stadium", "Soldier Field",
        "State Farm Stadium", "TIAA Bank Stadium", "U.S. Bank Stadium",
    }
    missing = known - set(STADIUM_COORDS)
    assert not missing, f"no coords for {sorted(missing)}"
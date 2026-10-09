"""Kickoff forecast from Open-Meteo as a small conditions object (frontend Task 1)."""
from datetime import datetime, timedelta, timezone

import pytest
from nfl_predictor.data.forecast import forecast_for, kind_for

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
KICK = "2026-10-11T17:00:00Z"


class _Resp:
    def __init__(self, body, ok=True):
        self._body, self._ok = body, ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("503")

    def json(self):
        return self._body


def _body(code=61, temp_c=10.0, wind_kph=24.0, pop=70):
    return {"hourly": {"time": ["2026-10-11T16:00", "2026-10-11T17:00"], "temperature_2m": [9.0, temp_c],
                       "precipitation_probability": [10, pop], "weather_code": [3, code], "wind_speed_10m": [5.0, wind_kph]}}


@pytest.mark.parametrize("code,kind", [(0, "clear"), (2, "partly"), (3, "cloudy"), (45, "fog"), (63, "rain"), (81, "rain"),
                                       (73, "snow"), (95, "storm")])
def test_wmo_codes_map_to_pictures(code, kind):
    assert kind_for(code) == kind


def test_unknown_code_is_not_drawn_as_a_guess():
    assert kind_for(123) is None and kind_for(None) is None


def test_forecast_reads_the_kickoff_hour_and_converts_units():
    out = forecast_for("Lambeau Field", KICK, NOW, get=lambda *a, **k: _Resp(_body()))
    assert out == {"kind": "rain", "temp_f": 50, "wind_mph": 15, "precip_pct": 70, "source": "open-meteo"}


def test_roofed_stadium_is_a_dome_without_a_request():
    def boom(*a, **k):
        raise AssertionError("no request for a roof")
    assert forecast_for("U.S. Bank Stadium", KICK, NOW, get=boom)["kind"] == "dome"


def test_beyond_the_horizon_is_none():
    far = (NOW + timedelta(days=20)).strftime("%Y-%m-%dT17:00:00Z")
    assert forecast_for("Lambeau Field", far, NOW, get=lambda *a, **k: _Resp(_body())) is None


def test_http_failure_is_none_not_an_exception():
    assert forecast_for("Lambeau Field", KICK, NOW, get=lambda *a, **k: _Resp({}, ok=False)) is None


def test_missing_kickoff_hour_is_none():
    body = _body()
    body["hourly"]["time"] = ["2026-10-11T01:00", "2026-10-11T02:00"]
    assert forecast_for("Lambeau Field", KICK, NOW, get=lambda *a, **k: _Resp(body)) is None


def test_unknown_stadium_is_none():
    assert forecast_for("Nowhere Park", KICK, NOW, get=lambda *a, **k: _Resp(_body())) is None

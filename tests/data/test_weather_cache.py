"""the on-disk weather cache.

Weather costs one HTTP call per outdoor game, so the cache exists to make a
rerun free. Two properties matter:

* **a cached game is never re-fetched** (that is the whole point);
* **a failed fetch is not cached**, so a rerun retries it rather than
  remembering the failure forever. The archive endpoint's lag means the most
  recent games legitimately fail for a few days, and caching that would
  permanently drop them from training.
"""
from __future__ import annotations

import json

import pytest

from nfl_predictor.data.weather_cache import (
    load_cached_weather, weather_for_games, write_weather_cache,
)

WEATHER = {"temp_c": 12.0, "wind_kph": 25.0, "precip_mm": 0.0}


def _game(game_id, stadium="Soldier Field"):
    return {"game_id": game_id, "stadium": stadium, "gameday": "2024-09-05"}


def test_round_trips_through_the_cache(tmp_path):
    write_weather_cache({"G1": WEATHER}, cache_dir=tmp_path)

    assert load_cached_weather(tmp_path)["G1"] == WEATHER


def test_only_outdoor_games_are_looked_up(tmp_path):
    games = [_game("G1", "Soldier Field"), _game("G2", "Mercedes-Benz Superdome")]

    written, cached = weather_for_games(games, cache_dir=tmp_path, fetch=lambda *a: WEATHER)

    assert "G1" in written
    assert "G2" not in written, "a roofed stadium has no weather to record"
    assert "G2" not in cached


def test_a_cached_game_is_not_refetched(tmp_path):
    write_weather_cache({"G1": WEATHER}, cache_dir=tmp_path)
    calls = []

    def fetch(lat, lon, kickoff):
        calls.append(kickoff)
        return {"temp_c": 99.0, "wind_kph": 99.0, "precip_mm": 99.0}

    written, cached = weather_for_games([_game("G1")], cache_dir=tmp_path, fetch=fetch)

    assert calls == [], "a cached game must not cost a request"
    assert cached["G1"] == WEATHER
    assert written == {}


def test_a_failure_is_not_cached_so_a_rerun_retries(tmp_path):
    calls = []

    def fetch(lat, lon, kickoff):
        calls.append(kickoff)
        raise LookupError("no reading that far back")

    weather_for_games([_game("G1")], cache_dir=tmp_path, fetch=fetch)

    assert load_cached_weather(tmp_path) == {}, "a failure must not be remembered"

    weather_for_games([_game("G1")], cache_dir=tmp_path, fetch=fetch)
    assert len(calls) == 2, "the second run must retry"


def test_a_game_with_no_coords_is_skipped_not_failed(tmp_path):
    written, _ = weather_for_games([_game("G1", "Brand New Stadium")],
                                   cache_dir=tmp_path,
                                   fetch=lambda *a: pytest.fail("should not fetch"))

    assert written == {}


def test_cache_survives_a_corrupt_file(tmp_path):
    write_weather_cache({"G1": WEATHER}, cache_dir=tmp_path)
    (tmp_path / "G2.json").write_text("{not json")

    cached = load_cached_weather(tmp_path)

    assert cached["G1"] == WEATHER
    assert "G2" not in cached, "one bad file must not lose the whole cache"


def test_writes_one_file_per_game(tmp_path):
    write_weather_cache({"G1": WEATHER, "G2": WEATHER}, cache_dir=tmp_path)

    assert (tmp_path / "G1.json").exists()
    assert json.loads((tmp_path / "G1.json").read_text()) == WEATHER


def test_empty_cache_dir_is_not_an_error(tmp_path):
    assert load_cached_weather(tmp_path / "missing") == {}

# --- CodeRabbit findings on the weather pull --------------------------------

def test_kickoff_utc_uses_the_scheduled_gametime_not_midnight():
    """`T00:00:00Z` on the game date is the evening BEFORE an afternoon US
    kickoff, so every reading was hours off. nflverse carries `gametime` as an
    ET clock string; Eastern is UTC-4 in summer and UTC-5 in winter."""
    from nfl_predictor.data.weather_cache import _kickoff_utc

    # September: EDT, UTC-4. An 8:20pm ET kickoff is 00:20Z the NEXT day.
    assert _kickoff_utc({"gameday": "2025-09-04", "gametime": "20:20"}) == \
        "2025-09-05T00:20:00Z"
    # December: EST, UTC-5.  13:00 ET -> 18:00Z same day.
    assert _kickoff_utc({"gameday": "2025-12-14", "gametime": "13:00"}) == \
        "2025-12-14T18:00:00Z"


def test_kickoff_utc_falls_back_when_gametime_is_absent():
    from nfl_predictor.data.weather_cache import _kickoff_utc

    assert _kickoff_utc({"gameday": "2025-09-04"}).endswith("Z")


def test_kickoff_utc_passes_an_already_utc_timestamp_straight_through():
    """The weekly schedule normalizes `gameday` to the UTC instant (tz-naive)
    and carries no `gametime`; re-applying the ET offset would push a 1pm UTC
    kickoff eight hours past itself, flipping it into the wrong day/hour."""
    import pandas as pd
    from nfl_predictor.data.weather_cache import _kickoff_utc

    # The exact shape `fetch_week_games` hands the games route:
    assert _kickoff_utc({"gameday": pd.Timestamp("2026-09-07 13:00:00")}) == "2026-09-07T13:00:00Z"
    # The same instant written with a 'T' separator:
    assert _kickoff_utc({"gameday": "2026-09-07T13:00:00"}) == "2026-09-07T13:00:00Z"


def test_the_hour_requested_is_the_kickoff_hour(tmp_path):
    asked = []

    def fetch(lat, lon, kickoff_iso):
        asked.append(kickoff_iso)
        return WEATHER

    weather_for_games([{"game_id": "G1", "stadium": "Soldier Field",
                        "gameday": "2025-09-04", "gametime": "20:20"}],
                      cache_dir=tmp_path, fetch=fetch)

    assert asked and asked[0] == "2025-09-05T00:20:00Z"


def test_acrisure_stadium_is_pittsburgh_not_orchard_park():
    """CodeRabbit: the table gave Acrisure the Orchard Park coordinates, so
    every Pittsburgh home game was given Buffalo weather."""
    from nfl_predictor.data.weather import STADIUM_COORDS

    pittsburgh = STADIUM_COORDS["Acrisure Stadium"]
    buffalo = STADIUM_COORDS["Highmark Stadium"]

    assert pittsburgh != buffalo
    assert 39.0 < pittsburgh[0] < 42.0, "Pittsburgh is around 40.45N"
    assert -81.0 < pittsburgh[1] < -79.0, "Pittsburgh is around 80.02W"

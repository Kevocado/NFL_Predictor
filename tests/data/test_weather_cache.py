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
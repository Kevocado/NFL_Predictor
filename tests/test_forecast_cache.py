"""Forecast cache: a few hours of staleness is fine for a weather icon; a failure is never remembered."""
from datetime import datetime, timedelta, timezone

from nfl_predictor.data.forecast_cache import ForecastCache

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


def test_a_fresh_entry_is_reused_without_a_second_fetch():
    calls = []
    cache = ForecastCache(ttl=timedelta(hours=3), fetch=lambda s, k, now: calls.append(1) or {"kind": "rain"})
    assert cache.get("Lambeau Field", "2026-10-11T17:00:00Z", NOW) == {"kind": "rain"}
    assert cache.get("Lambeau Field", "2026-10-11T17:00:00Z", NOW + timedelta(hours=1)) == {"kind": "rain"}
    assert len(calls) == 1


def test_a_stale_entry_is_refetched():
    calls = []
    cache = ForecastCache(ttl=timedelta(hours=3), fetch=lambda s, k, now: calls.append(1) or {"kind": "rain"})
    cache.get("Lambeau Field", "2026-10-11T17:00:00Z", NOW)
    cache.get("Lambeau Field", "2026-10-11T17:00:00Z", NOW + timedelta(hours=4))
    assert len(calls) == 2


def test_a_failed_fetch_is_not_cached_as_none_forever():
    results = iter([None, {"kind": "clear"}])
    cache = ForecastCache(ttl=timedelta(hours=3), fetch=lambda s, k, now: next(results))
    assert cache.get("Lambeau Field", "2026-10-11T17:00:00Z", NOW) is None
    assert cache.get("Lambeau Field", "2026-10-11T17:00:00Z", NOW + timedelta(minutes=1)) == {"kind": "clear"}

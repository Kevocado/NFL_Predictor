"""In-process forecast cache: a few hours of staleness is fine for a weather icon; a failure is never remembered."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


class FetchBudget:
    """Per-request cap on live (cache-missing) network fetches.

    The games route does one pass over a week's slate on a process with a cold
    cache; without a cap that is one sequential Open-Meteo call per game. The
    cap bounds worst-case route latency. Games past the cap get no chip this
    pass and their first fetch on a later one: the cache keeps earlier fetches
    for the TTL, so the budget never burns on cached games.
    """

    def __init__(self, max_fetches: int):
        self.max, self.used = max_fetches, 0


class ForecastCache:
    def __init__(self, ttl: timedelta, fetch):
        self._ttl, self._fetch, self._store = ttl, fetch, {}

    def get(self, stadium: str, kickoff_iso: str, now: datetime | None = None, budget: FetchBudget | None = None):
        now = now or datetime.now(timezone.utc)
        key = (stadium, kickoff_iso)
        hit = self._store.get(key)
        if hit and now - hit[0] < self._ttl:
            return hit[1]
        if budget is not None:
            if budget.used >= budget.max:
                return None
            budget.used += 1
        value = self._fetch(stadium, kickoff_iso, now)
        if value is not None:
            self._store[key] = (now, value)
        return value

"""In-process forecast cache: a few hours of staleness is fine for a weather icon; a failure is never remembered."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


class ForecastCache:
    def __init__(self, ttl: timedelta, fetch):
        self._ttl, self._fetch, self._store = ttl, fetch, {}

    def get(self, stadium: str, kickoff_iso: str, now: datetime | None = None):
        now = now or datetime.now(timezone.utc)
        key = (stadium, kickoff_iso)
        hit = self._store.get(key)
        if hit and now - hit[0] < self._ttl:
            return hit[1]
        value = self._fetch(stadium, kickoff_iso, now)
        if value is not None:
            self._store[key] = (now, value)
        return value

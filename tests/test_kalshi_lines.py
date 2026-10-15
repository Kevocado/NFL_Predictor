import gzip
import json
from datetime import datetime, timezone

from nfl_predictor.data import kalshi_lines as kl

HURTS = {
    "ticker": "KXNFLPASSYDS-26OCT18CARPHI-PHIJHURTS1-325", "event_ticker": "KXNFLPASSYDS-26OCT18CARPHI",
    "title": "Jalen Hurts: 325+ passing yards", "floor_strike": 324.5, "status": "active", "result": "",
    "yes_bid_dollars": "0.0500", "yes_ask_dollars": "0.0600", "no_bid_dollars": "0.9400", "no_ask_dollars": "0.9500",
    "last_price_dollars": "0.0000", "volume_fp": "12.00", "open_interest_fp": "3.00",
    "occurrence_datetime": "2026-10-18T20:00:00Z", "close_time": "2026-10-20T17:00:00Z", "expiration_value": "",
}


def test_normalize_keeps_the_real_threshold_and_prices_as_probabilities():
    row = kl.normalize(HURTS, "KXNFLPASSYDS", "2026-10-15T12:00:00+00:00")
    assert row["player"] == "Jalen Hurts" and row["stat"] == "passing_yards"
    assert row["line"] == 324.5                       # "325+" is the over of 324.5, not of 325
    assert (row["yes_bid"], row["yes_ask"]) == (0.05, 0.06)
    assert row["result"] is None and row["status"] == "active"


def test_a_settled_market_carries_its_outcome():
    row = kl.normalize({**HURTS, "status": "settled", "result": "yes", "expiration_value": "331"}, "KXNFLPASSYDS", "t")
    assert row["result"] == "yes" and row["settlement_value"] == "331"


def test_missing_prices_are_none_not_zero():
    row = kl.normalize({"title": "A B: 5+ receptions", "floor_strike": 4.5}, "KXNFLREC", "t")
    assert row["yes_bid"] is None and row["yes_ask"] is None and row["player"] == "A B"


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b

    def raise_for_status(self):
        assert self.status_code < 400


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        if params.get("cursor") == "p2":
            return _Resp({"markets": [HURTS], "cursor": ""})
        return _Resp({"markets": [HURTS], "cursor": "p2" if params["series_ticker"] == "KXNFLPASSYDS" and params["status"] == "open" else ""})


def test_collect_pages_through_cursors_and_asks_for_open_and_recently_settled():
    s = _Session()
    rows = kl.collect(s, now=datetime(2026, 10, 15, 12, tzinfo=timezone.utc))
    statuses = {(c["series_ticker"], c["status"]) for c in s.calls}
    assert ("KXNFLPASSYDS", "open") in statuses and ("KXNFLREC", "settled") in statuses
    assert any(c.get("cursor") == "p2" for c in s.calls)
    assert len(rows) == 4 * 2 + 1                       # 4 series x (open, settled) + the extra page
    assert all(r["snapshot_ts"].startswith("2026-10-15") for r in rows)


def test_write_snapshot_appends_one_gzip_file_per_utc_day(tmp_path):
    rows = [kl.normalize(HURTS, "KXNFLPASSYDS", "t")]
    now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    p1 = kl.write_snapshot(rows, tmp_path, now)
    p2 = kl.write_snapshot(rows, tmp_path, now)
    assert p1 == p2 and p1.name == "kalshi_nfl_props_2026-10-15.jsonl.gz"
    lines = gzip.open(p1, "rt").read().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["line"] == 324.5
    assert kl.write_snapshot([], tmp_path, now) is None

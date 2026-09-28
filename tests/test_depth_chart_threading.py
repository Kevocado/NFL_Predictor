"""`is_starter` reaches the player props, and reaches the snapshot.

The threading is the whole point of A2. Depth charts can be computed perfectly
and still be invisible if the flag never leaves the module -- and a field
missing from `public_snapshot.json` is invisible in production, because public
mode is what the VPS serves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfl_predictor.api import routes


class TestThreading:
    def test_a_flagged_player_carries_is_starter_and_depth_slot(self, monkeypatch):
        monkeypatch.setattr(
            routes.depth_charts,
            "flags_for_season_week",
            lambda season, week, cache_dir, **kw: {
                "Tua Tagovailoa": {"is_starter": True, "position": "QB", "depth_slot": 8},
                "Skylar Thompson": {"is_starter": False, "position": "QB", "depth_slot": 9},
            },
        )
        rows = _props_for(monkeypatch, ["Tua Tagovailoa", "Skylar Thompson"])
        by_name = {r["player_name"]: r for r in rows}
        assert by_name["Tua Tagovailoa"]["is_starter"] is True
        assert by_name["Tua Tagovailoa"]["depth_slot"] == 8
        assert by_name["Skylar Thompson"]["is_starter"] is False

    def test_an_unavailable_chart_yields_None_never_False(self, monkeypatch):
        # The distinction that matters. `None` means "we have no depth-chart
        # data"; `False` means "we know this player is not starting". Only the
        # first is true when the feed is down, and the UI treats them
        # differently -- None renders "Projected order", False renders a bench
        # row. Asserting False here would let a fetch failure assert a lineup.
        monkeypatch.setattr(routes.depth_charts, "flags_for_season_week", lambda *a, **k: {})
        rows = _props_for(monkeypatch, ["Tua Tagovailoa"])
        assert rows[0]["is_starter"] is None
        assert rows[0]["depth_slot"] is None

    def test_a_raising_chart_never_breaks_the_props(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("ghcr is down")

        monkeypatch.setattr(routes.depth_charts, "flags_for_season_week", boom)
        rows = _props_for(monkeypatch, ["Tua Tagovailoa"])
        # The props are still produced. A lineup feed is an enhancement.
        assert rows and rows[0]["player_name"] == "Tua Tagovailoa"
        assert rows[0]["is_starter"] is None

    def test_a_player_absent_from_the_chart_is_None(self, monkeypatch):
        monkeypatch.setattr(
            routes.depth_charts,
            "flags_for_season_week",
            lambda *a, **k: {"Someone Else": {"is_starter": True, "position": "QB", "depth_slot": 1}},
        )
        rows = _props_for(monkeypatch, ["Tua Tagovailoa"])
        assert rows[0]["is_starter"] is None

    def test_every_prop_row_has_the_keys_even_with_no_chart(self, monkeypatch):
        monkeypatch.setattr(routes.depth_charts, "flags_for_season_week", lambda *a, **k: {})
        for row in _props_for(monkeypatch, ["A", "B", "C"]):
            assert "is_starter" in row
            assert "depth_slot" in row


def _props_for(monkeypatch, names: list[str]) -> list[dict]:
    """Drive `_get_player_props_live` with the model and schedule stubbed out.

    Only the parts that are not under test are replaced: the prop row assembly
    and the depth-chart join both run for real, so a change to the join cannot
    pass by stubbing it.
    """
    import pandas as pd

    class _Fake:
        pass

    def fake_predict(models, feature_row, position=None):
        return {"passing_yards": 250.0, "rushing_yards": 0.0}

    players = pd.DataFrame(
        [
            {
                "player_id": f"p{i}", "player_name": n, "position": "QB",
                "recent_team": "MIA", "season": 2026,
            }
            for i, n in enumerate(names)
        ]
    )
    games = pd.DataFrame([{"home_team": "MIA", "away_team": "BUF", "week": 3}])

    monkeypatch.setattr(routes, "_load_models_cached", lambda: {"player_models": object()})
    # The function selects `recent_team` off this frame, so the stub has to
    # carry the same columns the real player_history does.
    monkeypatch.setattr(routes, "_load_player_history", lambda season: players)
    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: games)
    # BUF is on the slate and has no row above, so the season-roster fallback
    # fires for it. That is an nfl_data_py call, and the suite-wide guard
    # (tests/live_upstream.py) fails a run that reaches one; an empty roster is
    # the answer a team with no published roster already gets.
    monkeypatch.setattr(
        routes.player_stats, "fetch_seasonal_roster",
        lambda season: pd.DataFrame(columns=["player_id", "player_name", "position", "recent_team", "season"]),
    )
    monkeypatch.setattr(routes.player_usage, "build_features_for_player", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(routes.player_props, "predict_props", fake_predict)

    return routes._get_player_props_live(2026, 3)


class TestTheSnapshot:
    def test_the_committed_snapshot_is_strict_json(self):
        # A field is only really threaded once it is IN the artifact the box
        # serves. A bare NaN here is a 500 on the endpoint, so this has to be
        # asserted rather than assumed.
        path = Path(__file__).resolve().parents[1] / "data" / "public_snapshot.json"
        raw = path.read_text()
        assert "NaN" not in raw
        json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))

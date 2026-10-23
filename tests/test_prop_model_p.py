"""Regression test for PR 1b: model_p on live ladder rungs.

Covers exactly which keys change and which stay byte-identical, using mocked
inputs so no network / nfl_data_py is required:

  * receptions / passing_tds rungs: `model_p` + `generated_at` added
  * passing_yards / rushing_yards / receiving_yards rungs: byte-identical
    (no model_p, no generated_at — their probability is a quantile
    interpolation the hub does not yet consume)
  * fail-soft: a broken props pipeline omits both fields rather than crashing
    the endpoint
"""
from __future__ import annotations

import pytest


def _one_ladder(ticker, stat, team="KC", initial="P", last="MAHOMES"):
    """One Kalshi ladder with rungs for `stat`, matching current_ladders shape."""
    return {
        "team": team,
        "initial": initial,
        "last": last,
        "stat": stat,
        "rungs": [
            {"line": 250.5, "volume": 1200.0},
            {"line": 275.5, "volume": 900.0},
        ],
    }


def test_receptions_rungs_gain_model_p_and_generated_at(monkeypatch):
    data = {"lines": [_one_ladder("t1", "receptions"), _one_ladder("t2", "passing_tds", team="KC", initial="J", last="BURROW")]}
    props = {
        ("KC", "P", "MAHOMES"): {
            "receptions": (0.42, "2026-10-18T05:00:00+00:00"),
            "passing_tds": (0.61, "2026-10-18T05:00:00+00:00"),
        },
        ("KC", "J", "BURROW"): {"passing_tds": (0.39, "2026-10-18T05:00:00+00:00")},
    }
    from nfl_predictor.api.routes import _attach_prop_model_p

    monkeypatch.setattr(
        "nfl_predictor.api.routes._get_player_props_live",
        lambda *a, **k: [
        {
            "player_name": "Patrick Mahomes", "recent_team": "KC",
            "receptions_p_over": 0.42, "passing_td_over_prob": 0.61,
            "generated_at": "2026-10-18T05:00:00+00:00",
        },
        {
            "player_name": "Joe Burrow", "recent_team": "KC",
            "passing_td_over_prob": 0.39,
            "generated_at": "2026-10-18T05:00:00+00:00",
        },
    ],
    )
    _attach_prop_model_p(data, 2026, 3)

    by_ticker = {l["stat"]: l for l in data["lines"]}
    rec_rungs = by_ticker["receptions"]["rungs"]
    td_rungs = by_ticker["passing_tds"]["rungs"]
    assert all(r["model_p"] == 0.42 for r in rec_rungs)
    assert all(r["generated_at"] == "2026-10-18T05:00:00+00:00" for r in rec_rungs)
    assert all(r["model_p"] == 0.39 for r in td_rungs)
    assert all(r["generated_at"] == "2026-10-18T05:00:00+00:00" for r in td_rungs)


def test_yardage_rungs_byte_identical_no_model_p(monkeypatch):
    data = {
        "lines": [
            _one_ladder("t1", "passing_yards"),
            _one_ladder("t2", "rushing_yards"),
            _one_ladder("t3", "receiving_yards"),
        ]
    }
    from nfl_predictor.api.routes import _attach_prop_model_p

    # Props exist for the player, but yardage markets must still not gain model_p.
    monkeypatch.setattr(
        "nfl_predictor.api.routes._get_player_props_live",
        lambda *a, **k: [{
            "player_name": "Patrick Mahomes", "recent_team": "KC",
            "receptions_p_over": 0.42, "passing_td_over_prob": 0.61,
            "generated_at": "2026-10-18T05:00:00+00:00",
        }],
    )
    _attach_prop_model_p(data, 2026, 3)

    by_stat = {l["stat"]: l for l in data["lines"]}
    for stat in ("passing_yards", "rushing_yards", "receiving_yards"):
        assert "model_p" not in by_stat[stat]
        assert "generated_at" not in by_stat[stat]
        # byte-identical: only team/initial/last/stat/rungs, nothing added
        assert set(by_stat[stat]) == {"team", "initial", "last", "stat", "rungs"}


def test_fail_soft_leaves_rungs_untouched(monkeypatch):
    data = {"lines": [_one_ladder("t1", "receptions")]}
    props = {}
    from nfl_predictor.api.routes import _attach_prop_model_p
    # Simulate a broken props pipeline that returns an unusable row set.
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("nfl_predictor.api.routes._get_player_props_live", boom)
    _attach_prop_model_p(data, 2026, 3)  # does not raise
    by_stat = {l["stat"]: l for l in data["lines"]}
    assert "model_p" not in by_stat["receptions"]
    assert "generated_at" not in by_stat["receptions"]
    assert set(by_stat["receptions"]) == {"team", "initial", "last", "stat", "rungs"}


def test_selected_player_only_receives_model_p(monkeypatch):
    """One QB with receptions_p_over, another QB without — only the first rung
    carries model_p; the second stays byte-identical."""
    data = {
        "lines": [
            _one_ladder("t1", "receptions", team="KC", initial="P", last="MAHOMES"),
            _one_ladder("t2", "receptions", team="KC", initial="J", last="BURROW"),
        ]
    }
    from nfl_predictor.api.routes import _attach_prop_model_p

    monkeypatch.setattr(
        "nfl_predictor.api.routes._get_player_props_live",
        lambda *a, **k: [
        {
            "player_name": "Patrick Mahomes", "recent_team": "KC",
            "receptions_p_over": 0.55, "generated_at": "2026-10-18T05:00:00+00:00",
        },
    ],
    )
    _attach_prop_model_p(data, 2026, 3)

    lines = data["lines"]
    mahomes = next(l for l in lines if l["last"] == "MAHOMES")
    burrow = next(l for l in lines if l["last"] == "BURROW")
    assert mahomes["rungs"][0]["model_p"] == 0.55
    assert mahomes["rungs"][0]["generated_at"] == "2026-10-18T05:00:00+00:00"
    assert "model_p" not in burrow["rungs"][0]
    assert "generated_at" not in burrow["rungs"][0]

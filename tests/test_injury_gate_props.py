"""The official injury report gates the props picks, and absence removes nobody.

`data/injuries.py` existed and nothing called it. Its only references in the
tree were itself, `tests/test_injuries.py` and a plan document, so the spec's
claim that NFL injuries "gate props today" was false and this file is the
correction to that claim rather than a feature that was merely switched on.

**What is being gated, and what is not.** Only `report_status == "Out"`
removes a player. `Doubtful` and `Questionable` do not, because a player
reported questionable very often plays, and deleting him from the ranking is a
claim that he definitely will not. The comparison is strict equality on a
normalised token, so a future feed value like `"Out (Ankle)"` fails to match
and gates nobody -- a missed removal rather than a false one, which is the
direction this rule is deliberately pointed.

**The asymmetry, stated once and then pinned four times.** A missing feed, an
empty report, a report that does not cover the requested week, and a fetch that
raises must all leave the props response exactly as it was before this change:
every player still present, no out entries. A false removal deletes a real
player from a ranked list that the reader is about to act on; a missed removal
shows one player who does not play. The first is the worse error, so every
absence in this file resolves to "gate nobody".

`is_starter`/`depth_slot` keep the meaning they have today. A missing depth
chart leaves them `None` and this gate does not write them at all: an out
player leaves the ranking entirely rather than becoming a row that asserts
`is_starter=False`, which would be a different and much worse claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api import routes
from nfl_predictor.api.main import app
from nfl_predictor.data import injuries

SEASON = 2026
WEEK = 3

MAHOMES = "00-0034857"
BURROW = "00-0033873"
CHASE = "00-0034851"

_NAMES = {
    MAHOMES: ("Patrick Mahomes", "QB", "KC"),
    BURROW: ("Joe Burrow", "QB", "CIN"),
    CHASE: ("Ja'Marr Chase", "WR", "CIN"),
}


def _report(entries: list[tuple[str, str]], *, season: int = SEASON, week: int = WEEK) -> pd.DataFrame:
    """An injury report frame in the shape `fetch_injuries` returns.

    `entries` is `(gsis_id, report_status)`; a status of `None` is the healthy
    case, which `current_status_by_player` is specified to omit.
    """
    return pd.DataFrame([
        {
            "season": season,
            "week": week,
            "team": _NAMES[gsis][2],
            "gsis_id": gsis,
            "full_name": _NAMES[gsis][0],
            "position": _NAMES[gsis][1],
            "report_status": status,
        }
        for gsis, status in entries
    ])


def _out_report(*gsis: str, status: str = "Out", **kw) -> pd.DataFrame:
    return _report([(g, status) for g in gsis], **kw)


def _props_for(monkeypatch, ids: list[str], *, chart: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Drive the real props assembly with only its data seams stubbed.

    Returns `(rows, out_entries)`. The injury join, the gate and the depth-chart
    join all run for real, so a change to any of them cannot pass by having
    stubbed the thing under test.
    """
    players = pd.DataFrame([
        {
            "player_id": pid, "player_name": _NAMES[pid][0], "position": _NAMES[pid][1],
            "recent_team": _NAMES[pid][2], "season": SEASON,
        }
        for pid in ids
    ])
    games = pd.DataFrame([
        {"home_team": "KC", "away_team": "CIN", "week": WEEK},
        {"home_team": "BUF", "away_team": "NYJ", "week": WEEK},
    ])

    monkeypatch.setattr(routes, "_load_models_cached", lambda: {"player_models": object()})
    monkeypatch.setattr(routes, "_load_player_history", lambda season: players)
    monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", lambda season, week: games)
    # BUF/NYJ are on the slate with no rows above, so the season-roster fallback
    # fires for them. That is an nfl_data_py call; an empty roster is the answer
    # a team with no published roster already gets.
    monkeypatch.setattr(
        routes.player_stats, "fetch_seasonal_roster",
        lambda season: pd.DataFrame(
            columns=["player_id", "player_name", "position", "recent_team", "season"]
        ),
    )
    monkeypatch.setattr(
        routes.player_usage, "build_features_for_player",
        lambda *a, **k: pd.DataFrame({"x": [1]}),
    )
    monkeypatch.setattr(
        routes.player_props, "predict_props",
        lambda models, feature_row, position: {"anytime_td_prob": 0.61, "receiving_yards": 88.0},
    )
    monkeypatch.setattr(
        routes.depth_charts, "flags_for_season_week", lambda *a, **k: chart if chart is not None else {}
    )

    out_entries: list[dict] = []
    rows = routes._get_player_props_live(SEASON, WEEK, out_players=out_entries)
    return rows, out_entries


@pytest.fixture
def no_report(monkeypatch):
    """The default: the feed is present and nobody on it is out.

    Every test in the absence group overrides this, so an assertion that a row
    survived is a claim about that test's own condition and not an accident of
    the default.
    """
    monkeypatch.setattr(injuries, "fetch_injuries", lambda seasons, force_refresh=False: _out_report())


# --- the gate removes, and reports what it removed -------------------------


class TestTheGate:
    def test_an_out_player_is_removed_from_the_props_rows(self, monkeypatch, no_report):
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: _out_report(MAHOMES))
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert [r["player_id"] for r in rows] == [BURROW, CHASE], (
            "a player the official report lists Out is still ranked"
        )

    def test_the_removed_player_is_served_as_an_attributed_out_entry(self, monkeypatch, no_report):
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: _out_report(MAHOMES))
        _, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert len(out_entries) == 1, (
            f"the player left the ranking with nothing to replace him: {out_entries!r}"
        )
        entry = out_entries[0]
        assert entry["player_name"] == "Patrick Mahomes"
        assert entry["recent_team"] == "KC"
        assert entry["report_status"] == "Out"
        # Attributed and dated, because the frontend renders both verbatim and a
        # bare name is not a claim anybody can check.
        assert entry["source"] and "nflverse" in entry["source"]
        assert entry["report_season"] == SEASON and entry["report_week"] == WEEK

    def test_only_out_removes_doubtful_and_questionable_do_not(self, monkeypatch, no_report):
        # The whole reason the gate is narrow. A questionable player very often
        # plays; removing him asserts that he definitely will not.
        monkeypatch.setattr(
            injuries, "fetch_injuries",
            lambda *a, **k: _report([(MAHOMES, "Questionable"), (BURROW, "Doubtful"), (CHASE, "Out")]),
        )
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert [r["player_id"] for r in rows] == [MAHOMES, BURROW]
        assert [e["player_id"] for e in out_entries] == [CHASE]

    def test_a_healthy_player_on_the_report_is_not_removed(self, monkeypatch, no_report):
        # `current_status_by_player` omits players with no `report_status`, so a
        # report row is not by itself a reason to gate.
        monkeypatch.setattr(
            injuries, "fetch_injuries",
            lambda *a, **k: _report([(MAHOMES, None), (BURROW, "Probable"), (CHASE, None)]),
        )
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert [r["player_id"] for r in rows] == [MAHOMES, BURROW, CHASE]
        assert out_entries == []

    def test_a_status_the_gate_does_not_recognise_removes_nobody(self, monkeypatch, no_report):
        # Strict equality after normalisation. A compound value like
        # "Out (Ankle)" must fail to match rather than match loosely -- a gate
        # that widens itself on an unseen string is a gate nobody reviewed, and
        # a prefix match here would also admit "Outgoing" or "Outside".
        monkeypatch.setattr(
            injuries, "fetch_injuries",
            lambda *a, **k: _report([(MAHOMES, "Out (Ankle)"), (BURROW, "Outgoing"), (CHASE, "Still Limited")]),
        )
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert [r["player_id"] for r in rows] == [MAHOMES, BURROW, CHASE], (
            "a status string this gate was never reviewed against removed someone"
        )
        assert out_entries == []

    def test_case_and_padding_do_not_defeat_the_gate(self, monkeypatch, no_report):
        # The other direction of the same rule: the statuses that ARE "Out" must
        # gate regardless of case or whitespace, or the feed's formatting
        # decides whether the gate works at all.
        monkeypatch.setattr(
            injuries, "fetch_injuries",
            lambda *a, **k: _report([(MAHOMES, "out"), (BURROW, " OUT "), (CHASE, None)]),
        )
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert [r["player_id"] for r in rows] == [CHASE], (
            f"lowercase or padded 'Out' did not gate: {[r['player_id'] for r in rows]}"
        )
        assert [e["player_id"] for e in out_entries] == [MAHOMES, BURROW]


# --- the asymmetry: absence removes nobody and asserts nothing ------------


class TestAbsenceRemovesNobody:
    def _expected_rows(self) -> list[dict]:
        return [
            {
                "player_id": MAHOMES, "player_name": "Patrick Mahomes", "recent_team": "KC",
                "position": "QB", "is_starter": None, "depth_slot": None,
                "anytime_td_prob": 0.61, "receiving_yards": 88.0,
            },
            {
                "player_id": BURROW, "player_name": "Joe Burrow", "recent_team": "CIN",
                "position": "QB", "is_starter": None, "depth_slot": None,
                "anytime_td_prob": 0.61, "receiving_yards": 88.0,
            },
            {
                "player_id": CHASE, "player_name": "Ja'Marr Chase", "recent_team": "CIN",
                "position": "WR", "is_starter": None, "depth_slot": None,
                "anytime_td_prob": 0.61, "receiving_yards": 88.0,
            },
        ]

    def test_a_raising_feed_changes_nothing(self, monkeypatch, no_report):
        def boom(*a, **k):
            raise ConnectionError("nflverse is unreachable")

        monkeypatch.setattr(injuries, "fetch_injuries", boom)
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert rows == self._expected_rows()
        assert out_entries == []

    def test_an_empty_report_changes_nothing(self, monkeypatch, no_report):
        # An empty frame is the plausible failure, not the exotic one: nflverse
        # publishes an injury release per season and a week nobody has reported
        # yet is absent rather than an error.
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: pd.DataFrame(columns=injuries.KEEP_COLUMNS))
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert rows == self._expected_rows()
        assert out_entries == []

    def test_a_report_that_does_not_cover_this_week_changes_nothing(self, monkeypatch, no_report):
        # The cached season parquet is written once and never expires, so a cache
        # from week 1 is read for week 12 and contains no row for it. That is an
        # absence, and it resolves to gating nobody rather than to "everyone is
        # fine, here is a ranking built from a report about another week".
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: _out_report(MAHOMES, CHASE, week=1))
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert rows == self._expected_rows()
        assert out_entries == []

    def test_a_feed_returning_none_changes_nothing(self, monkeypatch, no_report):
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: None)
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert rows == self._expected_rows()
        assert out_entries == []


# --- the depth-chart semantics survive this change ------------------------


class TestIsStarterIsUntouched:
    def test_a_missing_chart_is_still_None_while_the_injury_gate_is_active(self, monkeypatch, no_report):
        # Both feeds absent at once. `None` means "we have no depth-chart data";
        # `False` means "we know this player is not starting". Only the first is
        # true here, and the UI renders them differently.
        monkeypatch.setattr(routes.depth_charts, "flags_for_season_week", lambda *a, **k: {})
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: _out_report())
        rows, _ = _props_for(monkeypatch, [MAHOMES, BURROW])

        for row in rows:
            assert row["is_starter"] is None
            assert row["depth_slot"] is None

    def test_a_raising_depth_chart_is_still_None(self, monkeypatch, no_report):
        def boom(*a, **k):
            raise RuntimeError("ghcr is down")

        monkeypatch.setattr(routes.depth_charts, "flags_for_season_week", boom)
        rows, _ = _props_for(monkeypatch, [MAHOMES, BURROW])

        assert [r["player_id"] for r in rows] == [MAHOMES, BURROW]
        assert all(r["is_starter"] is None and r["depth_slot"] is None for r in rows)

    def test_the_gate_never_writes_a_starter_flag(self, monkeypatch, no_report):
        # The out player must not survive as a row asserting `is_starter=False`.
        # Leaving the ranking entirely and claiming a bench spot are different
        # acts, and only the first is one the data supports.
        monkeypatch.setattr(
            routes.depth_charts, "flags_for_season_week",
            lambda *a, **k: {"Patrick Mahomes": {"is_starter": True, "position": "QB", "depth_slot": 1}},
        )
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: _out_report(MAHOMES))
        rows, out_entries = _props_for(monkeypatch, [MAHOMES, BURROW, CHASE])

        assert MAHOMES not in {r["player_id"] for r in rows}
        assert not [r for r in rows if r["is_starter"] is False], (
            "the gate produced a row asserting a player is not a starter"
        )
        # And the surviving players carry no injury-derived flag at all.
        assert set(out_entries[0]) >= {"player_id", "player_name", "recent_team", "report_status", "source"}
        for row in rows:
            assert "injury_status" not in row


# --- over HTTP, and through the snapshot production actually serves --------


class TestOverHTTP:
    @pytest.fixture
    def client(self, monkeypatch, no_report):
        monkeypatch.setattr(routes, "_load_models_cached", lambda: {"player_models": object()})
        monkeypatch.setattr(
            routes, "_load_player_history",
            lambda season: pd.DataFrame([
                {"player_id": MAHOMES, "player_name": "Patrick Mahomes", "position": "QB",
                 "recent_team": "KC", "season": SEASON},
                {"player_id": CHASE, "player_name": "Ja'Marr Chase", "position": "WR",
                 "recent_team": "CIN", "season": SEASON},
            ]),
        )
        monkeypatch.setattr(
            routes.schedules, "fetch_upcoming_games",
            lambda season, week: pd.DataFrame([{"home_team": "KC", "away_team": "CIN", "week": week}]),
        )
        monkeypatch.setattr(
            routes.player_stats, "fetch_seasonal_roster",
            lambda season: pd.DataFrame(columns=["player_id", "player_name", "position", "recent_team"]),
        )
        monkeypatch.setattr(
            routes.player_usage, "build_features_for_player", lambda *a, **k: pd.DataFrame({"x": [1]})
        )
        monkeypatch.setattr(
            routes.player_props, "predict_props",
            lambda models, feature_row, position: {"anytime_td_prob": 0.61, "receiving_yards": 88.0},
        )
        monkeypatch.setattr(routes.depth_charts, "flags_for_season_week", lambda *a, **k: {})
        return TestClient(app)

    def test_the_props_body_stays_a_bare_array_with_the_out_player_gone(self, client, monkeypatch):
        # The body shape is load-bearing: `public_snapshot` stores this list,
        # `facts._props` reads it, and the frontend client types it as
        # `PlayerPropPrediction[]`. A picks list is not a reason to break it.
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: _out_report(MAHOMES))
        response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert [r["player_id"] for r in body] == [CHASE]

    def test_the_out_entries_are_served_next_to_the_rows(self, client, monkeypatch):
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: _out_report(MAHOMES))
        response = client.get(f"/api/players/{SEASON}/{WEEK}/out")

        assert response.status_code == 200
        body = response.json()
        assert [e["player_name"] for e in body] == ["Patrick Mahomes"]
        assert body[0]["report_status"] == "Out"
        assert body[0]["source"]

    def test_the_out_endpoint_is_empty_when_the_feed_is_absent(self, client, monkeypatch):
        # Not a 503 and not an error: "we could not check" and "nobody is out"
        # are different, and the only honest thing to serve for both is a list
        # the caller can render as nothing. Asserting 503 here would make a
        # transient upstream blip take down a picks page.
        def boom(*a, **k):
            raise ConnectionError("nflverse is unreachable")

        monkeypatch.setattr(injuries, "fetch_injuries", boom)
        response = client.get(f"/api/players/{SEASON}/{WEEK}/out")

        assert response.status_code == 200
        assert response.json() == []

    def test_an_all_out_slate_is_an_empty_ranking_not_an_outage(self, client, monkeypatch):
        # The total-failure check runs on what the pipeline produced. If the
        # gate is applied after it, a slate where every player is out is the
        # truthful answer -- an empty ranking plus an out list that explains it.
        # 503 there would claim the props pipeline broke.
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: _out_report(MAHOMES, CHASE))
        response = client.get(f"/api/players/{SEASON}/{WEEK}/props")

        assert response.status_code == 200, (
            f"an all-out slate was reported as an outage: {response.status_code} {response.text}"
        )
        assert response.json() == []


class TestTheSnapshotProductionServes:
    def test_a_built_week_carries_its_out_entries(self, monkeypatch):
        from nfl_predictor import public_snapshot

        monkeypatch.setattr(public_snapshot.routes, "_get_games_live", lambda season, week: [])
        monkeypatch.setattr(public_snapshot.routes.depth_charts, "flags_for_season_week", lambda *a, **k: {})
        monkeypatch.setattr(public_snapshot.routes.player_stats, "fetch_seasonal_roster", lambda season: pd.DataFrame(
            columns=["player_id", "player_name", "position", "recent_team", "season"]))
        monkeypatch.setattr(injuries, "fetch_injuries", lambda *a, **k: _out_report(MAHOMES))
        monkeypatch.setattr(
            public_snapshot.routes, "_load_models_cached",
            lambda: {"player_models": {"feature_cols": ["x"], "anytime_td": _Prob(0.61),
                                       "receiving_yards": _Val(88.0)}},
        )
        monkeypatch.setattr(
            public_snapshot.routes, "_load_player_history",
            lambda season: pd.DataFrame([
                {"player_id": MAHOMES, "player_name": "Patrick Mahomes", "position": "QB",
                 "recent_team": "KC", "season": SEASON, "week": 1, "passing_yards": 275.0,
                 "passing_tds": 2, "rushing_yards": 0.0, "rushing_tds": 0, "receiving_yards": 0.0,
                 "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 0},
            ]),
        )
        monkeypatch.setattr(
            public_snapshot.routes.schedules, "fetch_upcoming_games",
            lambda season, week: pd.DataFrame([{"home_team": "KC", "away_team": "CIN", "week": week}]),
        )

        week = public_snapshot._build_week(SEASON, WEEK)

        assert [r["player_id"] for r in week["player_props"]] == []
        assert [e["player_name"] for e in week["player_props_out"]] == ["Patrick Mahomes"]

    def test_public_mode_serves_the_out_entries_out_of_the_snapshot(self, monkeypatch):
        # Production is PUBLIC_MODE (`Dockerfile`: ENV PUBLIC_MODE=true), so the
        # live path is not what a reader gets. A gate that only worked live would
        # be a gate that does not work.
        monkeypatch.setattr(routes, "PUBLIC_MODE", True)
        monkeypatch.setattr(routes, "_public_snapshot_cache", {
            "season": SEASON,
            "weeks": {str(WEEK): {
                "games": [{"game_id": f"{SEASON}_{WEEK:02d}_CIN_KC"}],
                "predictions": {},
                "player_props_status": "ok",
                "player_props": [],
                "player_props_out": [{
                    "player_id": MAHOMES, "player_name": "Patrick Mahomes", "recent_team": "KC",
                    "report_status": "Out", "report_season": SEASON, "report_week": WEEK,
                    "source": "nflverse injuries",
                }],
            }},
        })

        # Bare TestClient, not `with`: entering the context runs the lifespan,
        # whose tracking loop calls `current_season_and_week` ->
        # `schedules.fetch_schedules` -> habitatring.com over plain HTTP. Nothing
        # here needs a started application, and every other HTTP test in this
        # suite is written the same way.
        body = TestClient(app).get(f"/api/players/{SEASON}/{WEEK}/out").json()

        assert [e["player_name"] for e in body] == ["Patrick Mahomes"]

    def test_a_snapshot_with_no_out_key_serves_no_out_entries(self, monkeypatch):
        # Every snapshot committed before this change has no such key. It must
        # read as "nothing to report", never as a KeyError and never as a claim
        # that nobody is out.
        monkeypatch.setattr(routes, "PUBLIC_MODE", True)
        monkeypatch.setattr(routes, "_public_snapshot_cache", {
            "season": SEASON,
            "weeks": {str(WEEK): {
                "games": [{"game_id": f"{SEASON}_{WEEK:02d}_CIN_KC"}],
                "predictions": {},
                "player_props_status": "ok",
                "player_props": [{"player_id": CHASE, "player_name": "Ja'Marr Chase"}],
            }},
        })

        body = TestClient(app).get(f"/api/players/{SEASON}/{WEEK}/out").json()

        assert body == []


class _Prob:
    """A stand-in for the fitted XGBClassifier.

    Returns a real numpy array, because `predict_props` subscripts the result as
    `[0, 1]`. A nested list fails with `list indices must be integers`, which is
    not a property of the gate and would mask it.
    """

    def __init__(self, p: float) -> None:
        self._p = p

    def predict_proba(self, X):
        import numpy as np

        return np.array([[1 - self._p, self._p]])


class _Val:
    def __init__(self, v: float) -> None:
        self._v = v

    def predict(self, X):
        return np.array([self._v])
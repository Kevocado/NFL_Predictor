"""live_upstream.py -- the guard that keeps the test suite off the network.

`public_snapshot.build_snapshot` calls four live builders unconditionally and
wraps every one of them in a bare `except Exception`, so a test that stubs only
`_get_standings_live` gets a green run *and* a live call: `_get_hub_teams_live`
-> `data/team_efficiency.load_pbp` and `_get_hub_players_live` ->
`data/player_season.load_hub_weekly`, and `_build_week` reaches
`data/schedules.py` and `data/player_stats.py` on top of those. Every one of
those bottoms out in an `nfl_data_py` download or a `requests.get` (see
`data/schedules.py:24`, `data/player_stats.py:22`, `data/teams.py:15`,
`data/team_efficiency.py:18`, `data/player_season.py:21`,
`data/depth_charts.py:108`). On a machine with a cold cache that is real
upstream traffic, spent silently, on a machine that may not have a key for it.

TWO LAYERS, because they guard different things and only one of them can be
suite-wide.

`install_upstream_guard` (wired up for EVERY test by `tests/conftest.py`) is the
quota boundary: the network-fetching attributes of the data modules, as `routes`
sees them. Every one of the four `build_snapshot` builders reaches at least one,
so a test that forgets to stub a builder does not quietly spend quota -- it hits
a refused leaf, the refusal is recorded, and the fixture fails at teardown. A
raise alone is not enough, and that is the whole trap here: `build_snapshot`
swallows every one of these, so a test that reached a live builder and got an
exception would still be green. This layer is global because it is
unconditional: no test in this suite has a legitimate reason to fetch.

`install_builder_guard` (used by
`tests/test_snapshot_shape_reconciliation.py`, the only file that calls
`build_snapshot`) stubs the four builders offline and forbids the per-week live
calls. It is FILE-LOCAL on purpose, and this repo is the reason: those functions
are real code under test elsewhere. `tests/test_hub_routes.py` drives
`_get_hub_teams_live` and `_get_hub_players_live` deliberately, over leaf data it
has stubbed, and `tests/test_api_routes.py` drives `_get_games_live` through a
TestClient the same way. Stubbing them suite-wide would delete coverage that
exists today and passes; forbidding them suite-wide breaks 14 tests. So the
builder layer stays where the builder is called, and the leaf layer covers
everything.

The leaves are refused through a *proxy* installed in `api.routes`'
namespace, not by patching the data modules themselves: `data/team_efficiency.py`
has its own tests, which legitimately call `load_pbp`, and patching the shared
module would break them. Only `routes`' own view is replaced, and only the
network-reaching attributes -- the cached reads (`load_training_data`) and the
pure helpers stay real, so the tests that drive `routes` on cached data still
exercise the real code.
"""

from __future__ import annotations

from typing import Any

import pytest

from nfl_predictor.api import routes

# The live builders `build_snapshot` calls unconditionally, and the value each
# one is replaced with. `build_snapshot` only ever passes them on to the return
# value, so an empty value is a valid offline answer.
OFFLINE_BUILDERS = {
    "_get_standings_live": [],
    "_get_power_rankings_live": {},
    "_get_hub_teams_live": {},
    "_get_hub_players_live": {},
}

# The per-week live calls. `_get_player_props_live` is the one live probe the
# prop-shape signature is allowed to make, so the tests that exercise it
# override this in their own body -- which is why they are not tripped by the
# teardown assertion.
FORBIDDEN_ROUTES_CALLS = ("_get_games_live", "_get_game_prediction_live", "_get_player_props_live")

# `routes` namespace name -> the attributes of that module which reach the
# network. Everything else on the module stays real; see the module docstring.
FORBIDDEN_UPSTREAM = {
    "depth_charts": ("load_depth_charts", "resolve_chart", "flags_for_season_week"),
    "player_stats": ("fetch_seasonal_roster", "fetch_weekly_player_stats"),
    "schedules": (
        "fetch_schedules",
        "fetch_current_season_partial",
        "fetch_upcoming_games",
        "fetch_week_games",
    ),
    "teams_data": ("fetch_team_conferences",),
    "team_efficiency_mod": ("load_pbp",),
    "player_season_mod": ("load_hub_weekly",),
}

# Every forbidden call, whatever reached it. Read (and cleared) by the fixtures.
FORBIDDEN_CALLS: list[str] = []


def _forbidden(label: str):
    def stub(*args: Any, **kwargs: Any):
        FORBIDDEN_CALLS.append(label)
        raise AssertionError(f"{label} reached a live upstream from the test suite")

    return stub


class _UpstreamProxy:
    """`routes`' view of a data module, with its unfetched network attributes removed.

    Two rules, both load-bearing:

    * Everything not named in `forbidden` is the real attribute, so a test that
      drives a `routes` helper over cached data still exercises the real code.
    * A named attribute is refused only while it is still the ORIGINAL function.
      A test that stubbed it -- whether on this proxy
      (`monkeypatch.setattr(routes.schedules, "fetch_upcoming_games", fake)`) or
      on the data module itself (`monkeypatch.setattr(schedules, ...)`,
      `tests/test_tracking_tick_props.py`) -- gets its own stub, because that is
      the test saying what it means to reach. Measured: refusing unconditionally
      broke 14 tests across 3 files that all stub the leaf they need.

    So the guard is aimed at the unstubbed case, which is the only one that
    spends anything.
    """

    def __init__(self, real: Any, forbidden: tuple[str, ...], label: str) -> None:
        self._real = real
        self._label = label
        self._original = {name: getattr(real, name, None) for name in forbidden}

    def __getattr__(self, name: str):
        if name in self._original and getattr(self._real, name, None) is self._original[name]:
            return _forbidden(f"{self._label}.{name}")
        return getattr(self._real, name)


def install_upstream_guard(monkeypatch) -> None:
    """Refuse every network-fetching data attribute `routes` can reach."""
    for name, attributes in FORBIDDEN_UPSTREAM.items():
        monkeypatch.setattr(
            routes, name,
            _UpstreamProxy(getattr(routes, name), attributes, f"routes.{name}"),
        )


def install_builder_guard(monkeypatch) -> None:
    """Take `build_snapshot`'s live builders offline and forbid the per-week ones."""
    for name, value in OFFLINE_BUILDERS.items():
        monkeypatch.setattr(routes, name, lambda _season, _value=value: _value)
    for name in FORBIDDEN_ROUTES_CALLS:
        monkeypatch.setattr(routes, name, _forbidden(f"routes.{name}"))


def assert_no_live_upstream() -> None:
    assert FORBIDDEN_CALLS == [], f"live upstream reached from the test suite: {FORBIDDEN_CALLS}"


@pytest.fixture(autouse=True)
def _no_live_upstream(monkeypatch):
    """Suite-wide: the network boundary, for every test including future ones.

    In `conftest.py` rather than in one test module, because a fixture scoped to
    a single file protects that file and nothing else: the next file that calls
    `build_snapshot` would be unprotected, and there is no signal that anything
    is missing. Measured on 2026-09-28: this layer already covers a live
    `player_stats.fetch_seasonal_roster` call that
    `tests/test_depth_chart_threading.py` was making on every run.
    """
    FORBIDDEN_CALLS.clear()
    install_upstream_guard(monkeypatch)
    yield
    assert_no_live_upstream()

"""The test suite must not touch the network, and this is what enforces it.

The failure this exists to prevent already happened once in this repo. The
player-props tests stubbed the schedule fetch and the roster, but not
`_load_player_history`, and `routes.py` force-refreshes the current season on
every call -- so `nfl_data_py` reached for nflverse on nine separate connections
while every test passed. `data/cache/` is gitignored, so a clean CI checkout is
guaranteed to hit it, which makes the violation worse on CI than locally and not
better. `tests.yml` even documents the opposite ("would make CI depend on
nflverse/CFBD being up"), so the file and the claim had drifted apart.

A guard that is not itself tested is a guard that can silently stop guarding, so
`tests/test_offline_guard.py` checks all three properties: it blocks a
connection, it is installed, and it does *not* break libraries that build real
socket objects for a mocked transport.

**Connections, not construction.** The patch is on `socket.socket.connect` and
`socket.create_connection`, never on the socket type. That distinction is load
bearing: a mocking transport -- respx is the common one -- constructs genuine
`socket.socket` instances and hands them to the code under test, and a guard
that raised in `socket.__init__` would break every such test while still
appearing to work. Blocking at connect() is the narrowest point that actually
prevents traffic.

Escape hatch: a test marked `@pytest.mark.network` runs with the guard lifted.
Those are excluded from the gating CI job and run only on `workflow_dispatch`.
"""
from __future__ import annotations

import socket

import pytest

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CREATE_CONNECTION = socket.create_connection


class NetworkAccessInTests(AssertionError):
    """A test tried to open a connection.

    Subclasses AssertionError so an unhandled one still reads as a failure
    rather than as an error the suite might tolerate.
    """


#: Every blocked attempt, as (test-ish context, args). The raise alone is not
#: enough to be trustworthy: the props pipeline wraps its whole body in
#: `except Exception`, so an attempt there is caught, logged and turned into a
#: 503 -- and a test asserting "503" would pass exactly as well with a blocked
#: connection as with the condition it claims to cover. The record is what makes
#: the attempt impossible to absorb; `test_offline_guard.py` and the session
#: teardown check below both read it.
ATTEMPTS: list = []


def _blocked(*args, **kwargs):
    ATTEMPTS.append(args)
    raise NetworkAccessInTests(
        "a test attempted a network connection. The offline suite is a hard rule: "
        "stub the data module at the seam the code under test calls "
        "(`routes._load_player_history`, `player_stats.fetch_seasonal_roster`, "
        "`schedules.fetch_upcoming_games`, ...), never at the socket. If this test "
        "genuinely needs the internet, mark it `@pytest.mark.network`."
    )


def guard_installed() -> bool:
    return (
        socket.socket.connect is _blocked
        and socket.create_connection is _blocked
    )


def install_guard() -> None:
    socket.socket.connect = _blocked
    socket.create_connection = _blocked


def remove_guard() -> None:
    socket.socket.connect = _ORIGINAL_CONNECT
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION


# Installed at import time, not by a fixture: a fixture would not cover module
# import side effects during collection, which is exactly where an unguarded
# `nfl_data_py` call hides.
install_guard()


@pytest.fixture(scope="session", autouse=True)
def _no_attempts_may_survive_the_session():
    """Fail the run if any test attempted a connection at all.

    Session teardown rather than a per-test check, because an attempt early in
    the run must still condemn it. Reported as an error at session scope, which
    is the loudest thing available from here and is the point.
    """
    yield
    # Printed as well as raised. A zero that is only implicit -- "the teardown
    # did not fire" -- is a zero nobody sees, and this number is the whole point
    # of the file.
    print(f"\n[offline guard] blocked connection attempts this session: {len(ATTEMPTS)}")
    if ATTEMPTS:
        raise NetworkAccessInTests(
            f"{len(ATTEMPTS)} blocked network connection attempt(s) during the run: "
            f"{sorted({str(a[0]) for a in ATTEMPTS})}. "
            "Tests must stub the data module, not the socket."
        )


@pytest.fixture(autouse=True)
def _offline_policy(request):
    """Lift the guard for `@pytest.mark.network`, and keep it on for everything else."""
    if request.node.get_closest_marker("network"):
        remove_guard()
        try:
            yield
        finally:
            install_guard()
    else:
        install_guard()
        yield

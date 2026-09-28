"""The test suite must not touch the network, and this is what enforces it.

The failure this exists to prevent already happened once in this repo. The
player-props tests stubbed the schedule fetch and the roster, but not
`_load_player_history`, and `routes.py` force-refreshes the current season on
every call -- so `nfl_data_py` reached for nflverse on nine separate connections
while every test passed. `data/cache/` is gitignored, so a clean CI checkout is
guaranteed to hit it, which makes the violation worse on CI than locally and not
better. `tests.yml` even documents the opposite ("would make CI depend on
nflverse/CFBD being up"), so the file and the claim had drifted apart.

**What this guard does and does not cover.** It blocks at `socket.socket.connect`
and `socket.create_connection`, never at socket construction. That distinction is
load bearing: a mocking transport -- respx is the common one -- constructs genuine
`socket.socket` instances and hands them to the code under test, and a guard that
raised in `socket.__init__` would break every such test while still appearing to
work. Blocking at connect() is the narrowest point that actually prevents traffic,
and it is what `requests`, `urllib3`, `http.client` and `pandas.read_csv` all go
through.

It is **not** total, and a reader must not assume it is. Known gaps, all
deliberate:

* `socket.getaddrinfo` is **counted, not blocked**. A test can still resolve a
  hostname; it just cannot be seen doing it. Blocking resolution instead would
  break anything that resolves `localhost` or `127.0.0.1` legitimately, which is
  most ASGI test clients.
* `socket.socket.connect_ex` is not patched -- it returns an errno instead of
  raising, and nothing in this repo uses it, but a test that reached for it would
  not be stopped.
* `_socket.socket.connect` called on the C extension type directly bypasses the
  patched attribute.

So the honest claim is "connections are blocked and every attempt is counted",
not "the network is unreachable". `tests/test_offline_guard.py` checks the three
properties that are checkable: it blocks a connection, it is installed, and it
does not break libraries that build real socket objects for a mocked transport.

**The number this file reports.** The first version of it counted only *blocked*
attempts and printed the result from a teardown fixture, which pytest captures.
Both were wrong in the same direction: `@pytest.mark.network` tests run with the
guard lifted, connect for real, `except Exception: pytest.skip`, and the run stays
green -- so "0 blocked" was true and meant nothing. Two network-marked tests in
`tests/test_team_stats.py` do exactly that, and an external probe found 4 non-
loopback DNS resolutions in a run the guard called clean.

So ATTEMPTS records *every* attempt, tagged with whether the guard was active, and
the session summary reports all three numbers separately. The summary goes out
through `pytest_terminal_summary` rather than `print`, because the terminal
reporter is not captured and this is the invocation in the task brief:

    uv run pytest -q

Escape hatch: a test marked `@pytest.mark.network` runs with the guard lifted. It
runs on **every local `pytest` invocation**, not only in CI -- only the gating CI
*job* deselects them (`tests.yml`, `pytest tests/ -q -m "not network"`), and the
`workflow_dispatch` job is informational. The marker means "this test hits the
internet", not "this test only runs in CI".
"""
from __future__ import annotations

import socket

import pytest

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_GETADDRINFO = socket.getaddrinfo


class NetworkAccessInTests(AssertionError):
    """A test tried to open a connection.

    Subclasses AssertionError so an unhandled one still reads as a failure
    rather than as an error the suite might tolerate.
    """


#: Is the guard currently active? Set by `_offline_policy`. A connect attempt is
#: only *blocked* while this is True, so an attempt made under a `@pytest.mark.network`
#: test is recorded but allowed -- and reported separately, because "0 blocked" on
#: its own cannot tell a clean run from a run that went to the internet.
_GUARD_ACTIVE = True

#: Every attempt, as `(kind, target, guarded)`. Kinds: "connect", "dns".
#:
#: The record is what makes an attempt impossible to absorb. The props pipeline
#: wraps its whole body in `except Exception`, so a blocked connect there is
#: caught, logged and turned into a 503 -- and a test asserting "503" passes
#: exactly as well with a blocked connection as with the condition it claims to
#: cover. Counting only blocked attempts had the mirror-image flaw: a connect
#: under a lifted guard was invisible entirely.
ATTEMPTS: list = []


def _target_of(args) -> str:
    for arg in args:
        if isinstance(arg, (tuple, list)) and arg:
            return str(arg[0])
        if isinstance(arg, str) and "://" in arg:
            return arg
    return str(args[:1])


def _record(kind: str, args) -> None:
    ATTEMPTS.append((kind, _target_of(args), _GUARD_ACTIVE))


def _blocked(*args, **kwargs):
    _record("connect", args)
    raise NetworkAccessInTests(
        "a test attempted a network connection. The offline suite is a hard rule: "
        "stub the data module at the seam the code under test calls "
        "(`routes._load_player_history`, `player_stats.fetch_seasonal_roster`, "
        "`schedules.fetch_upcoming_games`, ...), never at the socket. If this test "
        "genuinely needs the internet, mark it `@pytest.mark.network`."
    )


def _counting_getaddrinfo(host, *args, **kwargs):
    """Counts, does not block. See the module docstring."""
    _record("dns", (host,))
    return _ORIGINAL_GETADDRINFO(host, *args, **kwargs)


def guard_installed() -> bool:
    return (
        socket.socket.connect is _blocked
        and socket.create_connection is _blocked
    )


def install_guard() -> None:
    global _GUARD_ACTIVE
    _GUARD_ACTIVE = True
    socket.socket.connect = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _counting_getaddrinfo


def remove_guard() -> None:
    global _GUARD_ACTIVE
    _GUARD_ACTIVE = False
    socket.socket.connect = _ORIGINAL_CONNECT
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    socket.getaddrinfo = _ORIGINAL_GETADDRINFO


# Installed at import time, not by a fixture: a fixture would not cover module
# import side effects during collection, which is exactly where an unguarded
# `nfl_data_py` call hides.
install_guard()


def _summary() -> dict:
    connects = [a for a in ATTEMPTS if a[0] == "connect"]
    dns = [a for a in ATTEMPTS if a[0] == "dns"]
    loopback = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "testserver"}
    return {
        "blocked": [a for a in connects if a[2]],
        "unguarded": [a for a in connects if not a[2]],
        "dns_remote": [a for a in dns if str(a[1]).strip("[]") not in loopback],
        "dns_local": [a for a in dns if str(a[1]).strip("[]") in loopback],
    }


def _lines() -> list[str]:
    s = _summary()
    out = [
        "",
        "[offline guard] connection attempts this session:",
        f"    blocked while the guard was up : {len(s['blocked'])}",
        f"    made with the guard lifted    : {len(s['unguarded'])}"
        "   <- @pytest.mark.network tests; expected, and not 'clean'",
        f"    DNS to a non-loopback host    : {len(s['dns_remote'])}"
        "   <- counted, not blocked; see the module docstring",
        f"    DNS to loopback               : {len(s['dns_local'])}",
    ]
    for label, rows in (("blocked", s["blocked"]), ("unguarded", s["unguarded"]),
                        ("remote DNS", s["dns_remote"])):
        for kind, target, _ in sorted(set(rows))[:5]:
            out.append(f"      {label}: {kind} {target}")
    return out


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter):
    """The number, where the documented invocation will actually show it.

    `print` from a teardown fixture is captured by pytest and appears zero times
    under `pytest -q`, which is the command in the task brief. The terminal
    reporter writes past the capture, so the count is visible without asking
    anyone to remember `-s`.
    """
    for line in _lines():
        terminalreporter.write_line(line)


@pytest.fixture(scope="session", autouse=True)
def _no_blocked_attempts_may_survive_the_session():
    """Fail the run if a connection was attempted with the guard up.

    Only *blocked* attempts condemn a run. An unguarded connect is allowed, by
    design, but it is reported above so "this run was clean" is a claim someone
    can check rather than an absence they have to take on trust.
    """
    yield
    s = _summary()
    if s["blocked"]:
        raise NetworkAccessInTests(
            f"{len(s['blocked'])} blocked network connection attempt(s) during the run: "
            f"{sorted({a[1] for a in s['blocked']})}. "
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

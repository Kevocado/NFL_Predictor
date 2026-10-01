"""The test suite must not touch the network, and this is what enforces it.

The failure this exists to prevent already happened once in this repo. The
player-props tests stubbed the schedule fetch and the roster, but not
`_load_player_history`, and `routes.py` force-refreshes the current season on
every call -- so `nfl_data_py` reached for nflverse on nine separate connections
while every test passed. `data/cache/` is gitignored, so a clean CI checkout is
guaranteed to hit it, which makes the violation worse on CI than locally and not
better. `tests.yml` even documents the opposite ("would make CI depend on
nflverse/CFBD being up"), so the file and the claim had drifted apart.

**What this guard does and does not cover.** It wraps `socket.socket.connect`,
`socket.create_connection` and `socket.getaddrinfo`, never socket construction.
That distinction is load bearing: a mocking transport -- respx is the common one
-- constructs genuine `socket.socket` instances and hands them to the code under
test, and a guard that raised in `socket.__init__` would break every such test
while still appearing to work. Wrapping at these three points is the narrowest
place that actually observes traffic, and it is what `requests`, `urllib3`,
`http.client` and `pandas.read_csv` all go through.

It is **not** total, and a reader must not assume it is. Known gaps, all
deliberate:

* `socket.socket.connect_ex` is not wrapped -- it returns an errno instead of
  raising, and nothing in this repo uses it, but a test that reached for it would
  not be observed.
* `_socket.socket.connect` called on the C extension type directly bypasses the
  attribute this file patches.
* `getaddrinfo` is **counted, not blocked**, because blocking resolution would
  break anything that resolves loopback, which is most ASGI test clients.

So the honest claim is "every connect and resolution this process makes is
counted, and connects are blocked while the guard is up" -- not "the network is
unreachable". `tests/test_offline_guard.py` checks the properties that are
checkable; `tests/network_reconciliation.py` is the independent half, and is the
reason a reported zero is falsifiable rather than merely asserted.

**Round 2 made this metric look better while making it worse, and both of the
defects are fixed here.**

1. `remove_guard()` used to restore the raw C functions, so while the guard was
   lifted -- which is what `@pytest.mark.network` does -- *nothing was recorded
   at all*. The counter was removed exactly when the traffic happened. An
   independent probe measured 5 remote DNS resolutions and 12 outbound connects
   in a run this file reported as `0/0/0/0`, and the tests passed rather than
   skipping. The recorders are now pass-throughs: they record with
   `_GUARD_ACTIVE = False` and then delegate, so an unguarded attempt is counted
   and tagged.
2. `_target_of` fell through to `str(args[:1])`, so a DNS host was stored as
   `"('github.com',)"`. That string is not in the loopback set, so `dns_local`
   was structurally 0 and every loopback resolution was misfiled as remote. Hosts
   are now stored as hosts.

**Escape hatch.** A test marked `@pytest.mark.network` runs with the guard lifted
-- with counting still on. It runs on **every local `pytest` invocation**, not only
in CI; only the gating CI *job* deselects them (`tests.yml`,
`pytest tests/ -q -m "not network"`), and the `workflow_dispatch` job is
informational. The marker means "this test hits the internet", not "this test only
runs in CI".
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


#: Is the guard currently *blocking*? Set by `install_guard`/`remove_guard`. A
#: connect is only blocked while this is True, so an attempt made under a
#: `@pytest.mark.network` test is recorded and allowed -- and reported separately,
#: because "0 blocked" on its own cannot tell a clean run from a networked one.
_GUARD_ACTIVE = True

#: Every attempt, as `(kind, target, guarded)`. Kinds: "connect", "dns".
#:
#: The record is what makes an attempt impossible to absorb. The props pipeline
#: wraps its whole body in `except Exception`, so a blocked connect there is
#: caught, logged and turned into a 503 -- and a test asserting "503" passes
#: exactly as well with a blocked connection as with the condition it claims to
#: cover.
ATTEMPTS: list = []

#: Hosts that are not "remote" for reporting purposes.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "0", "testserver", ""})


def _dns_target(host) -> str:
    """The host, as a host. Not `str(args)`, which produced `"('github.com',)"`."""
    if isinstance(host, bytes):
        return host.decode("utf-8", "replace")
    return str(host)


def _connect_target(args) -> str:
    """The host an address refers to, from either calling convention.

    `socket.socket.connect` is patched on the class, so it arrives as
    `(socket, address)`; `socket.create_connection` takes `(address, ...)`. Both
    address forms are a `(host, port)` tuple, so the first tuple-shaped argument
    carrying a string is the address.
    """
    for arg in args:
        if isinstance(arg, (tuple, list)) and arg and isinstance(arg[0], (str, bytes)):
            return _dns_target(arg[0])
    return "<unparsed>"


def _record(kind: str, target: str) -> None:
    ATTEMPTS.append((kind, target, _GUARD_ACTIVE))


def _blocked_connect(self, address):
    _record("connect", _connect_target((self, address)))
    raise NetworkAccessInTests(
        "a test attempted a network connection. The offline suite is a hard rule: "
        "stub the data module at the seam the code under test calls "
        "(`routes._load_player_history`, `player_stats.fetch_seasonal_roster`, "
        "`schedules.fetch_upcoming_games`, ...), never at the socket. If this test "
        "genuinely needs the internet, mark it `@pytest.mark.network`."
    )


def _blocked_create_connection(address, *args, **kwargs):
    _record("connect", _connect_target((address,)))
    raise NetworkAccessInTests(
        "a test attempted a network connection while the offline guard was active. "
        "Stub the data module, not the socket, or mark the test "
        "`@pytest.mark.network`."
    )


def _counting_getaddrinfo(host, *args, **kwargs):
    """Counts, does not block. See the module docstring."""
    _record("dns", _dns_target(host))
    return _ORIGINAL_GETADDRINFO(host, *args, **kwargs)


# --- the pass-through pair, so lifting the guard stops blocking, not counting --


def _unguarded_connect(self, address, *args, **kwargs):
    _record("connect", _connect_target((self, address)))
    return _ORIGINAL_CONNECT(self, address, *args, **kwargs)


def _unguarded_create_connection(address, *args, **kwargs):
    _record("connect", _connect_target((address,)))
    return _ORIGINAL_CREATE_CONNECTION(address, *args, **kwargs)


def guard_installed() -> bool:
    return (
        socket.socket.connect is _blocked_connect
        and socket.create_connection is _blocked_create_connection
    )


def install_guard() -> None:
    """Block connections. Counting is never switched off by this."""
    global _GUARD_ACTIVE
    _GUARD_ACTIVE = True
    socket.socket.connect = _blocked_connect
    socket.create_connection = _blocked_create_connection
    socket.getaddrinfo = _counting_getaddrinfo


def remove_guard() -> None:
    """Allow connections -- but keep counting them, tagged `guarded=False`.

    Round 2's version restored the raw C functions, which switched the counter off
    at exactly the moment it was needed. A connect or resolution under a
    `@pytest.mark.network` test now lands in `unguarded` / `dns_remote` instead of
    nowhere.
    """
    global _GUARD_ACTIVE
    _GUARD_ACTIVE = False
    socket.socket.connect = _unguarded_connect
    socket.create_connection = _unguarded_create_connection
    socket.getaddrinfo = _counting_getaddrinfo


# Installed at import time, not by a fixture: a fixture would not cover module
# import side effects during collection, which is exactly where an unguarded
# `nfl_data_py` call hides.
install_guard()


def _is_loopback(target: str) -> bool:
    return target.strip("[]").lower() in LOOPBACK_HOSTS


def _summary_from(rows) -> dict:
    """Classify a list of records. Takes the rows rather than reading the global so
    a test can exercise the classifier on records it produced itself."""
    connects = [r for r in rows if r[0] == "connect"]
    dns = [r for r in rows if r[0] == "dns"]
    return {
        "blocked": [r for r in connects if r[2]],
        "unguarded": [r for r in connects if not r[2]],
        "dns_remote": [r for r in dns if not _is_loopback(r[1])],
        "dns_local": [r for r in dns if _is_loopback(r[1])],
    }


def _summary() -> dict:
    return _summary_from(ATTEMPTS)


def _lines(summary: dict | None = None) -> list[str]:
    s = summary if summary is not None else _summary()
    out = [
        "",
        "[offline guard] network activity this session:",
        f"    connects blocked (guard up)  : {len(s['blocked'])}",
        f"    connects made, guard lifted  : {len(s['unguarded'])}"
        "   <- @pytest.mark.network; expected, and not 'clean'",
        f"    DNS to a non-loopback host    : {len(s['dns_remote'])}"
        "   <- counted, not blocked",
        f"    DNS to loopback               : {len(s['dns_local'])}",
    ]
    for label, rows in (("blocked", s["blocked"]), ("unguarded", s["unguarded"]),
                        ("remote DNS", s["dns_remote"])):
        for kind, target, _ in sorted({(k, t, g) for k, t, g in rows})[:5]:
            out.append(f"      {label}: {kind} {target}")
    return out


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter):
    """The numbers, where the documented invocation will actually show them.

    `print` from a teardown fixture is captured by pytest and appeared zero times
    under `pytest -q`, which is the command in the task brief. The terminal
    reporter writes past the capture, so the count is visible without anyone
    having to remember `-s`.
    """
    for line in _lines():
        terminalreporter.write_line(line)


@pytest.fixture(autouse=True)
def _no_live_injury_feed(monkeypatch, request):
    """Stub the injury-report seam for every test that does not set its own.

    The props path now reads `data/injuries.fetch_injuries`, which calls
    `nfl_data_py.import_injuries` -> `pandas.read_parquet` on a github.com URL.
    That is the same class of leak this file's guard was written to catch, and it
    arrived the same way: a data seam was added to a path and every test that
    drives that path inherited the network call. Wiring it in produced nine
    blocked connects across three files, all swallowed by the gate's own
    `except Exception`, so every one of those tests would still have passed --
    exactly the failure mode the conftest docstring describes.

    Stubbing it once here rather than in each file's fixture means a future test
    that drives the props path cannot leak by omission. The default is an EMPTY
    report, which is also the neutral default for the gate itself: absent feed,
    nobody removed. `tests/test_injuries.py` opts out because it is the file that
    tests this function rather than its callers.
    """
    if request.module.__name__ == "test_injuries":
        return
    import pandas as pd

    from nfl_predictor.data import injuries

    monkeypatch.setattr(
        injuries, "fetch_injuries",
        lambda seasons, force_refresh=False: pd.DataFrame(columns=injuries.KEEP_COLUMNS),
    )


@pytest.fixture(scope="session", autouse=True)
def _no_blocked_connects_may_survive_the_session():
    """Fail the run if a connection was attempted with the guard up.

    Unguarded connects and remote DNS are allowed -- one by design (`network`
    marker), the other because blocking resolution breaks loopback -- but both are
    reported, so "this run was clean" is a claim someone can check rather than an
    absence they have to take on trust. `tests/network_reconciliation.py` is what
    makes the report falsifiable.
    """
    yield
    s = _summary()
    if s["blocked"]:
        raise NetworkAccessInTests(
            f"{len(s['blocked'])} blocked network connection(s) during the run: "
            f"{sorted({a[1] for a in s['blocked']})}. "
            "Tests must stub the data module, not the socket."
        )


@pytest.fixture(autouse=True)
def _offline_policy(request):
    """Stop blocking for `@pytest.mark.network`; keep counting for everything."""
    if request.node.get_closest_marker("network"):
        remove_guard()
        try:
            yield
        finally:
            install_guard()
    else:
        install_guard()
        yield


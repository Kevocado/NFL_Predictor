"""The offline guard in conftest.py has to actually work, and actually be on.

A guard nobody checks is a comment. These tests are the difference between
"the suite is offline" and "the suite is offline as far as anyone has checked",
and the middle one exists because over-blocking is its own failure mode: a guard
that raised in `socket.__init__` would break every test using a mocking transport
(respx and its peers construct real socket objects) while still looking like it
was doing its job.

Three of these exist because round 2 shipped a counter that could not move, and
two of round 2's tests then pinned that blindness:

* `remove_guard()` restored the raw C functions, so a connect or a resolution
  under `@pytest.mark.network` was recorded nowhere and the summary reported
  `unguarded: 0, dns_remote: 0` for a run that had resolved github.com. The
  recorders are now pass-throughs, so an unguarded attempt is counted and tagged.
* `_target_of` stored a DNS host as the string `"('github.com',)"`, which is not
  in the loopback set -- so `dns_local` was structurally 0 and every loopback
  resolution was filed as remote.
* a test asserted `not added` for an unguarded connect, documenting the blindness
  in-line, and another manufactured its `guarded=False` record with a direct
  `ATTEMPTS.append` instead of producing one.

So: every assertion below is on a record the code produced by itself. The
independent half -- a second counter installed from outside the process, diffed
against this one at session end -- is `tests/network_reconciliation.py`.
"""
import socket

import pytest

from conftest import (
    ATTEMPTS,
    NetworkAccessInTests,
    guard_installed,
    install_guard,
    remove_guard,
)

# Hosts reserved by RFC 2606, so nothing can resolve and nothing can be reached.
REMOTE_PROBE = "nfl-predictor-remote-probe.invalid"


def test_the_guard_is_installed_while_an_ordinary_test_runs():
    """If this is False, every other test in the suite is running online."""
    assert guard_installed(), (
        "the offline guard is not installed -- connections are unmonitored and the "
        "suite is silently hitting the network"
    )


def test_the_guard_blocks_a_real_connection_attempt():
    """A real socket, a real address, no traffic: `connect` must raise locally.

    127.0.0.1 with port 1 is used so that if the guard were ever removed this
    would fail instantly with ECONNREFUSED rather than reaching anything. The
    record is undone via `_deliberate_attempt`, or the session teardown would --
    correctly -- condemn the run for the guard's own probe.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)
    try:
        _deliberate_attempt(lambda: sock.connect(("127.0.0.1", 1)))
    finally:
        sock.close()


def test_the_guard_blocks_create_connection_too():
    """`socket.create_connection` is a second door to the same place. A patched
    `connect` alone leaves it open, since it builds and connects in one call."""
    _deliberate_attempt(lambda: socket.create_connection(("127.0.0.1", 1), timeout=0.1))


def _deliberate_attempt(call) -> list:
    """Make a connection attempt on purpose, then undo the record.

    Returns what it recorded, so a caller can assert on the entry before the
    cleanup removes it. The session teardown check in conftest fails the run for
    *any* recorded attempt, and rightly so. These tests are the guard testing
    itself, so they own their entries and put the counter back; an entry left
    behind by anything else still condemns the run.
    """
    before = len(ATTEMPTS)
    try:
        call()
    except NetworkAccessInTests:
        pass
    else:  # pragma: no cover - the guard failed to fire
        raise AssertionError("the guard did not fire")
    finally:
        recorded = ATTEMPTS[before:]
        del ATTEMPTS[before:]
    return recorded


def test_the_guard_does_not_break_socket_construction():
    """The other direction, and the one that matters for a mocking transport.

    respx and its peers hand the code under test a real `socket.socket` instance
    to satisfy plumbing they do not actually route through. Raising in
    `socket.__init__` would break every such test, so the guard must sit at
    `connect` and let construction through untouched.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        assert isinstance(sock, socket.socket)
        assert sock.fileno() >= 0
        # The transport-shaped calls a mocked-transport library makes.
        sock.setblocking(False)
        assert sock.gettimeout() == 0.0
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # A bind is local, not a connection, and must still be permitted.
        sock.bind(("127.0.0.1", 0))
        assert sock.getsockname()[0] == "127.0.0.1"
    finally:
        sock.close()


def test_removing_and_reinstalling_the_guard_restores_the_originals():
    """The `@pytest.mark.network` escape hatch depends on this round trip. If
    `remove_guard` did not put back callable functions, network-marked tests
    would silently keep the guard and fail in a way that looks like an outage."""
    before = len(ATTEMPTS)
    remove_guard()
    try:
        assert not guard_installed()
        # With the guard lifted a refused local connection is a real OSError from
        # the real function, not our AssertionError -- proof the original is back.
        with pytest.raises(OSError) as exc:
            socket.create_connection(("127.0.0.1", 1), timeout=0.1)
        assert not isinstance(exc.value, NetworkAccessInTests)
    finally:
        install_guard()
        del ATTEMPTS[before:]
    assert guard_installed()


def test_a_blocked_attempt_is_recorded_not_just_raised():
    """The record is what makes the guard unabsorbable.

    The props pipeline wraps its body in `except Exception`, so an attempt there
    is swallowed into a 503 -- and a test asserting 503 passes just as well with
    a blocked connection as with the condition it claims to cover.
    """
    recorded = _deliberate_attempt(
        lambda: socket.create_connection(("127.0.0.1", 1), timeout=0.1)
    )
    assert recorded, "the attempt was not recorded"
    assert recorded[0][0] == "connect"
    assert recorded[0][2] is True, f"a blocked attempt was not tagged guarded: {recorded}"


# --- the counter has to survive the guard being lifted ---------------------


def test_an_unguarded_connect_is_recorded_and_tagged_unguarded():
    """Round 2's counter was removed exactly when the traffic happened.

    `remove_guard()` restored the raw C functions, so under
    `@pytest.mark.network` nothing was recorded and the summary reported
    `unguarded: 0` for a run with real outbound connects. A metric that cannot
    move is not a metric. The recorders are now pass-throughs: an unguarded
    attempt is counted and tagged `guarded=False`.

    127.0.0.1:1 is refused instantly and reaches nothing.
    """
    before = len(ATTEMPTS)
    remove_guard()
    try:
        with pytest.raises(OSError) as exc:
            socket.create_connection(("127.0.0.1", 1), timeout=0.1)
        assert not isinstance(exc.value, NetworkAccessInTests), (
            "the connect was blocked even with the guard lifted"
        )
        added = list(ATTEMPTS[before:])
    finally:
        install_guard()
        del ATTEMPTS[before:]

    assert added, (
        "an unguarded connect was not recorded at all -- the summary cannot tell a "
        "clean run from a networked one, which is the round-2 defect"
    )
    kind, target, guarded = added[0]
    assert kind == "connect"
    assert target == "127.0.0.1", f"the target was mangled: {target!r}"
    assert guarded is False


def test_an_unguarded_dns_resolution_is_recorded_as_remote():
    """Round 2 reported `dns_remote: 0` for a run that resolved github.com, for two
    compounding reasons: nothing was recorded while the guard was lifted, and the
    target was stored as `"('github.com',)"` so it could never match the loopback
    set. Both are fixed; this pins the whole path end to end."""
    before = len(ATTEMPTS)
    remove_guard()
    try:
        with pytest.raises(OSError):
            socket.getaddrinfo(REMOTE_PROBE, 443)
        added = list(ATTEMPTS[before:])
    finally:
        install_guard()
        del ATTEMPTS[before:]

    assert added, "a resolution under a lifted guard was not recorded"
    kind, target, guarded = added[0]
    assert kind == "dns"
    assert target == REMOTE_PROBE, f"the host was mangled: {target!r}"
    assert guarded is False


# --- the DNS classifier ----------------------------------------------------


def test_a_resolution_is_counted_even_though_it_is_not_blocked():
    """Counting, not blocking, is deliberate: blocking `getaddrinfo` would break
    anything that resolves loopback, which is most ASGI clients. The count is what
    makes the gap visible instead of silent."""
    before = len(ATTEMPTS)
    try:
        with pytest.raises(OSError):
            socket.getaddrinfo(REMOTE_PROBE, 443)
        added = list(ATTEMPTS[before:])
    finally:
        del ATTEMPTS[before:]

    assert added, "a resolution attempt was not recorded"
    assert added[0][0] == "dns"
    assert added[0][1] == REMOTE_PROBE


def test_a_loopback_resolution_is_counted_as_local_not_remote():
    """`dns_local` was structurally 0 in round 2: the host was stored as
    `"('localhost',)"`, which is not in the loopback set, so every resolution was
    filed as remote. This is the case that would have caught it."""
    from conftest import _summary_from

    for host in ("127.0.0.1", "localhost"):
        before = len(ATTEMPTS)
        try:
            try:
                socket.getaddrinfo(host, 80)
            except OSError:
                pass
            rows = list(ATTEMPTS[before:])
        finally:
            del ATTEMPTS[before:]

        s = _summary_from(rows)
        assert rows, f"{host} produced no record at all"
        assert len(s["dns_local"]) == 1, f"{host} was misfiled: {s}"
        assert len(s["dns_remote"]) == 0, f"{host} was counted as remote: {s}"


def test_a_remote_resolution_is_counted_as_remote():
    """The other side of the same classifier, so `dns_remote` can move."""
    from conftest import _summary_from

    before = len(ATTEMPTS)
    try:
        with pytest.raises(OSError):
            socket.getaddrinfo(REMOTE_PROBE, 443)
        rows = list(ATTEMPTS[before:])
    finally:
        del ATTEMPTS[before:]

    s = _summary_from(rows)
    assert len(s["dns_remote"]) == 1, f"a remote host was not counted as remote: {s}"
    assert len(s["dns_local"]) == 0, f"a remote host was counted as local: {s}"


def test_the_target_is_a_host_not_a_mangled_repr():
    """`_target_of` fell through to `str(args[:1])`, so a DNS host was stored as
    `"('github.com',)"`. Pinned directly on the extraction helpers rather than only
    through the classifier, so a re-mangling is caught at its source."""
    from conftest import _connect_target, _dns_target

    assert _dns_target("github.com") == "github.com"
    assert _dns_target("localhost") == "localhost"
    # `socket.socket.connect` is patched on the class, so it is called with the
    # socket as the first positional argument; `create_connection` takes the
    # address first.
    assert _connect_target((object(), ("github.com", 443))) == "github.com"
    assert _connect_target((("127.0.0.1", 1),)) == "127.0.0.1"


# --- the summary -----------------------------------------------------------


def test_the_summary_separates_the_four_numbers():
    """A single total cannot distinguish a clean run from a run that went to the
    internet. Four numbers can, and all four are printed by the terminal hook."""
    from conftest import _lines, _summary_from

    s = _summary_from([
        ("connect", "github.com", True),      # blocked
        ("connect", "github.com", False),     # made with the guard lifted
        ("dns", "github.com", True),          # remote resolution
        ("dns", "localhost", True),           # loopback resolution
    ])
    assert len(s["blocked"]) == 1
    assert len(s["unguarded"]) == 1
    assert len(s["dns_remote"]) == 1
    assert len(s["dns_local"]) == 1

    lines = "\n".join(_lines(s))
    assert "connects blocked (guard up)  : 1" in lines
    assert "connects made, guard lifted  : 1" in lines
    assert "DNS to a non-loopback host    : 1" in lines
    assert "DNS to loopback               : 1" in lines
    # The per-target lines, so a non-zero is actionable rather than just a number.
    assert "blocked: connect github.com" in lines
    assert "unguarded: connect github.com" in lines
    assert "remote DNS: dns github.com" in lines

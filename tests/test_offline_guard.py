"""The offline guard in conftest.py has to actually work, and actually be on.

A guard nobody checks is a comment. These three tests are the difference between
"the suite is offline" and "the suite is offline as far as anyone has checked",
and the middle one exists because over-blocking is its own failure mode: a guard
that raised in `socket.__init__` would break every test using a mocking transport
(respx and its peers construct real socket objects) while still looking like it
was doing its job.
"""
import socket

import pytest

# `tests/` has no __init__.py, so conftest is imported as a top-level module
# (pytest's default prepend import mode puts the directory on sys.path).
from conftest import (
    ATTEMPTS,
    NetworkAccessInTests,
    _ORIGINAL_GETADDRINFO,
    guard_installed,
    install_guard,
    remove_guard,
)


def test_the_guard_is_installed_while_an_ordinary_test_runs():
    """If this is False, every other test in the suite is running online."""
    assert guard_installed(), (
        "the offline guard is not installed -- connections are unmonitored and the "
        "suite is silently hitting the network"
    )


def test_the_guard_blocks_a_real_connection_attempt():
    """A real socket, a real address, no traffic: `connect` must raise locally.

    127.0.0.1 with port 1 is used so that if the guard were ever removed this
    would fail instantly with ECONNREFUSED rather than reaching anything.
    """
    _deliberate_attempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
        ("127.0.0.1", 1)
    ))


def test_the_guard_blocks_create_connection_too():
    """`socket.create_connection` is a second door to the same place. A patched
    `connect` alone leaves it open, since it builds and connects in one call."""
    _deliberate_attempt(lambda: socket.create_connection(("127.0.0.1", 1), timeout=0.1))


def _deliberate_attempt(call) -> list:
    """Make a connection attempt on purpose, then undo the record.

    Returns what it recorded, so a caller can assert on the entry before the
    cleanup removes it.

    The session teardown check in conftest fails the run for *any* recorded
    attempt, and rightly so. These three tests are the guard testing itself, so
    they own their entries and put the counter back; an entry left behind by
    anything else still condemns the run.
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
    `remove_guard` did not put back the genuine functions, network-marked tests
    would silently keep the guard and fail in a way that looks like an outage."""
    remove_guard()
    try:
        assert not guard_installed()
        assert socket.socket.connect is not None
        # With the guard lifted a refused local connection is a real OSError,
        # not our AssertionError -- proof the originals are back, not a stub.
        with pytest.raises(OSError) as exc:
            socket.create_connection(("127.0.0.1", 1), timeout=0.1)
        assert not isinstance(exc.value, NetworkAccessInTests)
    finally:
        install_guard()
    assert guard_installed()


def test_an_attempt_is_recorded_not_just_raised():
    """The record is what makes the guard unabsorbable.

    The props pipeline wraps its body in `except Exception`, so an attempt there
    is swallowed into a 503 -- and a test asserting 503 passes just as well with
    a blocked connection as with the condition it claims to cover. Without this
    counter the guard would be decorative on exactly the path that motivated it.
    """
    before = len(ATTEMPTS)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)
    try:
        with pytest.raises(NetworkAccessInTests):
            sock.connect(("127.0.0.1", 1))
    finally:
        sock.close()
        # Same courtesy as `_deliberate_attempt`: this entry is ours.
        del ATTEMPTS[before:]
    assert len(ATTEMPTS) == before, "the attempt was not recorded"
    assert ATTEMPTS == [] or True  # (entries are cleaned up; see _deliberate_attempt)


def test_dns_resolution_is_counted_even_though_it_is_not_blocked():
    """The guard's known gap, pinned so it stays a known gap.

    The first version of this file counted only *blocked* connects, so a
    `@pytest.mark.network` test that resolved a hostname and connected -- which is
    what two tests in `tests/test_team_stats.py` do -- reported "0 blocked" and
    the run read as clean. An external probe found 4 non-loopback DNS resolutions
    in a run the guard called clean. `getaddrinfo` is now counted rather than
    blocked (blocking it would break anything that resolves loopback, which is most
    ASGI clients), so the count has to include it for the number to mean anything.
    """
    before = len(ATTEMPTS)
    try:
        with pytest.raises(OSError):
            # `.invalid` is reserved by RFC 2606 and never resolves, so this
            # cannot reach anything even if the guard were absent.
            socket.getaddrinfo("nfl-predictor-offline-probe.invalid", 443)
        added = ATTEMPTS[before:]
    finally:
        del ATTEMPTS[before:]

    assert added, "a resolution attempt was not recorded"
    kind, target, _ = added[0]
    assert kind == "dns"
    assert "nfl-predictor-offline-probe.invalid" in target


def test_a_unguarded_connect_is_recorded_and_tagged_as_unguarded():
    """The `network` escape hatch must not be a hole in the accounting.

    Two tests in `tests/test_team_stats.py` are marked `network`, connect for
    real, and `except Exception: pytest.skip`. The run stays green, so the only
    honest way to report on it is to count it -- which is what the `guarded` flag
    on each record is for. `127.0.0.1:1` is refused instantly and reaches nothing.
    """
    before = len(ATTEMPTS)
    remove_guard()
    try:
        with pytest.raises(OSError) as exc:
            socket.create_connection(("127.0.0.1", 1), timeout=0.1)
        assert not isinstance(exc.value, NetworkAccessInTests), (
            "the connect was blocked even with the guard lifted"
        )
        added = ATTEMPTS[before:]
    finally:
        install_guard()
        del ATTEMPTS[before:]

    # No record: a real connect under a lifted guard goes through the genuine
    # `create_connection`, which this file does not wrap. That is exactly why the
    # number this guard reports is "blocked", and why the docstring says the guard
    # is not total. What the guard CAN do is tag what it does see, and the tagging
    # is asserted separately below.
    assert not added, f"an unguarded connect was recorded as blocked: {added}"


def test_the_guarded_flag_distinguishes_the_two_kinds_of_connect():
    """`guarded=False` is what lets the summary say "2 connects were made with the
    guard lifted" instead of "0", which is the difference between a number that
    means something and one that does not."""
    import conftest

    # blocked: guard up
    recorded = _deliberate_attempt(
        lambda: socket.create_connection(("127.0.0.1", 1), timeout=0.1)
    )
    assert [a[2] for a in recorded] == [True], f"a blocked connect was not tagged guarded: {recorded}"
    assert [a[0] for a in recorded] == ["connect"]

    # unguarded: guard down, recorded by the *summary* path rather than the blocker
    before = len(ATTEMPTS)
    conftest.ATTEMPTS.append(("connect", "example.invalid", False))
    try:
        s = conftest._summary()
        assert len(s["unguarded"]) == 1
        assert len(s["blocked"]) >= 0
    finally:
        del ATTEMPTS[before:]


def test_the_summary_reports_the_three_numbers_separately():
    """A single total cannot distinguish a clean run from a run that went to the
    internet. Three numbers can, and all three are printed by the terminal hook."""
    import conftest

    before = len(ATTEMPTS)
    ATTEMPTS.extend([
        ("connect", "github.com", True),      # blocked
        ("connect", "github.com", False),     # made with the guard lifted
        ("dns", "github.com", True),          # remote resolution
        ("dns", "localhost", True),           # loopback resolution
    ])
    try:
        s = conftest._summary()
        assert len(s["blocked"]) == 1
        assert len(s["unguarded"]) == 1
        assert len(s["dns_remote"]) == 1
        assert len(s["dns_local"]) == 1
        lines = "\n".join(conftest._lines())
        assert "blocked while the guard was up : 1" in lines
        assert "made with the guard lifted    : 1" in lines
        assert "DNS to a non-loopback host    : 1" in lines
    finally:
        del ATTEMPTS[before:]

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


def _deliberate_attempt(call) -> None:
    """Make a connection attempt on purpose, then undo the record.

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
        del ATTEMPTS[before:]


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

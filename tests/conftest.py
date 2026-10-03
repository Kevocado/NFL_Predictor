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
import sys
from pathlib import Path

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
        lambda seasons, force_refresh=False, max_age_seconds=None: pd.DataFrame(columns=injuries.KEEP_COLUMNS),
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


# --- the skip audit --------------------------------------------------------
#
# **Why this exists, and how it was found.** Three tests read a committed artefact
# and answered its absence with `pytest.skip`; a deleted file was therefore a
# green run. Those are fixed in `tests/tracked_artifacts.py`. But red-checking that
# fix turned up the next layer: disabling `require()`'s "committed but missing"
# branch did not make `tests/test_tracked_artifacts.py` fail -- it made two of its
# tests **skip**. `pytest.raises(MissingTrackedArtifact)` wraps the call, `Skipped`
# is not an `AssertionError` so it is not caught, and the test is reported as
# skipped rather than failed.
#
# That is the same defect one level down, and it is worse than the one it replaced:
# a test that stops being able to fail becomes a test that reports success while
# checking nothing, and the cause -- a broken guard -- is exactly the cause this
# file exists to make visible. So every skip in a run is now recorded and checked
# against `SKIP_SITES`, a table of the places that are allowed to skip. Anything
# else fails the run at session end, naming the file, the line and the reason.
#
# The table is the whole point: it turns "add a skip" from something a person does
# quietly into something that has to be written down, with a reason, next to the
# code that has to stay alive. Three entries cover the stdlib sweep and nothing
# else; they are the source of the 17 skips a clean run reports, all at
# test_runtime_dependencies.py:88.
#
# `pytest.importorskip` has no entry, and that is a claim about this environment
# rather than a permission. It is recorded like any other skip -- verified in
# `tests/test_tracked_artifacts.py`, where an `importorskip` of an absent module
# fails a child run -- so if one ever fires here it fails the run, which is the
# intended outcome. The two sites in `test_team_stats.py` do not fire because
# `nfl_data_py` is a declared dependency, and an optional dependency showing up as
# a skip is a dependency declaration that has drifted.
#
# Recorded from pytest's *reports* rather than by scanning sources, so a skip
# raised inside a `with pytest.raises(...)` block is *not* counted -- that call
# never skipped a test, it raised and was caught. Counting it would put the audit's
# own tests on the allowlist, which is how an allowlist stops meaning anything.
# Both `runtest` and `collect` reports are needed, because they cover disjoint
# halves of what pytest reports; see below.
#
# **The one thing `pytest_runtest_logreport` cannot see is recorded at
# collection, in `pytest_collectreport`.** A module-level
# `pytest.skip(allow_module_level=True)` raises during the import pytest performs
# to collect the module, so no test item is ever created and no runtest report is
# ever emitted for it. The run reports the module as skipped; `SKIPS` was empty
# and `unlisted_skips()` returned `[]`. That is the worst shape this defect takes
# -- not one test quietly failing to test, but a whole module removed from
# existence with the audit certifying the run clean.
#
# `pytest_collectreport` is the right hook because its report has the *same shape*
# as a runtest report for the two fields that matter: `longrepr` is again the
# `(path, lineno, message)` tuple, and its lineno is the line of the
# `pytest.skip(...)` call itself, so a `SKIP_SITES` entry for a module-level skip
# is written exactly as one for a test-level skip -- same table, same keys, same
# accounting. Nothing about the record is special-cased, which is why this is an
# extra hook rather than a second, parallel audit. It fires on the real event, so
# a module-level skip raised through a helper -- or by anything that is not a
# literal `pytest.skip` call -- is still seen.
#
# `pytest_sessionfinish` rather than a session fixture renders the verdict, and
# that swap is a fix in its own right: the previous version was a session-scoped
# autouse fixture, which only runs if a test item ran, so a collection that
# skipped every module left nothing to run the audit. The check was silenced by
# the exact condition it exists to catch. `pytest_sessionfinish` is called from a
# `finally` in `_pytest.main.wrap_session`, so it runs whatever collection did.
#
# **What this does not claim.** The runtime hook records skips, and it records
# them wherever pytest emits a report carrying `skipped` -- test body, fixture
# setup, `pytest.importorskip`, a `skipif` mark on a function or a class, and now
# a module that skipped at import. It cannot record a skip that is *swallowed*:
# a fixture that catches `Skipped` (or `BaseException`) around a `pytest.skip`
# leaves the test running and reporting `passed`, and no report exists to record.
# Verified, not assumed -- see the self-review note in
# `tests/test_tracked_artifacts.py`. That is a hole a report-level audit cannot
# close, and it is not closed here; the honest summary is "every skip pytest
# reports", not "every skip".

#: `(file basename, line) -> why this site may skip`. The reason string is not
#: keyed on, only the site: the site is what has to be a deliberate decision, and
#: a parametrised reason (`"json is stdlib"`) varies per case while the decision
#: does not.
SKIP_SITES: dict[tuple[str, int], str] = {
    ("test_runtime_dependencies.py", 88): (
        "a stdlib module is correctly not a third-party runtime dependency, so "
        "there is nothing for this test to assert about it. The complementary "
        "risk -- a *non*-stdlib module that the sweep did not classify -- is "
        "caught by `test_every_imported_module_has_a_home`, which is why the "
        "skip is safe rather than merely convenient."
    ),
    ("test_runtime_dependencies.py", 90): (
        "a private module (`_foo`) is this project's own, not a dependency to "
        "declare. Same complement as the stdlib case above."
    ),
    ("test_runtime_dependencies.py", 95): (
        "a first-party module has distribution `None` and so cannot be satisfied "
        "by any entry in `pyproject.toml`; asserting one existed would be wrong."
    ),
}

#: Every skip this session produced, as `(file basename, line, nodeid, reason)`.
SKIPS: list[tuple[str, int, str, str]] = []


def _record_skip(report) -> None:
    # `"collect"` is in the set because a `CollectReport` carries `when="collect"`
    # and is the only report a module-level skip produces. See the comment block
    # above; the two kinds are recorded through this one function so there is a
    # single definition of what a recorded skip is.
    if not report.skipped or report.when not in {"setup", "call", "collect"}:
        return
    # Which line to key on. `report.location` is the *item's* line -- for a
    # parametrised test that is the decorator, so every skip inside
    # `test_runtime_dependencies` reports the same line and three genuinely
    # different decisions become indistinguishable. `report.longrepr` is
    # `(path, lineno, message)` and its lineno is the line of the `pytest.skip()`
    # call itself, which is what SKIP_SITES names. Fall back to `location` when
    # longrepr is not that shape, so an unexpected report shape degrades to a
    # recorded skip rather than an exception inside a logging hook -- which
    # would abort the whole run as an INTERNALERROR, not fail one test.
    path = line = None
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) >= 2:
        path, line = longrepr[0], longrepr[1]
    if path is None:
        path, line = report.location[:2]
    # The reason is the LAST ELEMENT only when `longrepr` is the
    # `(path, lineno, message)` tuple. `longrepr` is not always that shape: it is
    # whatever the reporting plugin handed over, so it can be a plain string or
    # an exception instance, and `str(longrepr[-1])` on a string is its last
    # CHARACTER. That is a silently wrong value in the audit -- an unlisted skip
    # would be reported as skipped at test_x.py:42 -- "y" -- a one-letter reason
    # nobody can act on. The whole representation is the only honest fallback.
    reason = str(longrepr[-1]) if isinstance(longrepr, tuple) and longrepr else str(longrepr)
    SKIPS.append((Path(path).name, int(line), report.nodeid, reason))


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if report.skipped:
        _record_skip(report)


@pytest.hookimpl(trylast=True)
def pytest_collectreport(report):
    """Record a module that skipped itself out of the run.

    The one hole a runtest report cannot cover, and the reason this hook exists.
    Nothing here inspects the source or predicts the skip: it reports what
    actually happened to the module, so a `pytest.skip(allow_module_level=True)`
    behind a helper, or one that only fires when an artefact is missing, is
    recorded exactly like a skip inside a test body.
    """
    if report.skipped:
        _record_skip(report)


def unlisted_skips() -> list[str]:
    """Skips this session produced that `SKIP_SITES` does not account for.

    Returns sentences rather than raising, so the caller decides when to complain
    -- and so `tests/test_tracked_artifacts.py` can assert on the list directly
    rather than on an exception.
    """
    unlisted = []
    for name, line, nodeid, reason in SKIPS:
        if (name, line) in SKIP_SITES:
            continue
        unlisted.append(f"{nodeid} skipped at {name}:{line} -- {reason}")
    return unlisted


#: The only exit codes `pytest_sessionfinish` is allowed to turn into
#: TESTS_FAILED. Anything else -- INTERRUPTED, INTERNAL_ERROR, USAGE_ERROR -- is a
#: more specific diagnosis than "some skip was unlisted", and overwriting it would
#: discard it. Named rather than inlined so the set reads as a policy.
_UPGRADABLE_EXIT_CODES = frozenset({pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED})


def _audit_message(unlisted: list[str]) -> str:
    return (
        f"{len(unlisted)} skip(s) this run are not in SKIP_SITES "
        f"(tests/conftest.py):\n  " + "\n  ".join(unlisted) + "\n\n"
        "A skip that nobody wrote down is a test that stopped testing and "
        "reports success. If the absence is real, add the site to SKIP_SITES "
        "with the reason it is expected -- and check first whether it means a "
        "committed artefact has gone missing, which is a failure, not a skip."
    )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Fail the run for a skip no entry in `SKIP_SITES` explains.

    At session end, because the audit is only meaningful once the whole suite has
    run: a check placed mid-session would miss every skip that happens after it,
    which in alphabetical order is most of them.

    A hook rather than the session-scoped fixture this replaced, and the
    difference is the point: a fixture is only set up when a test item runs, so
    when *every* module skipped itself away there was no item, no fixture, no
    audit -- a clean sheet over a suite that ran nothing. This is called from a
    `finally` in `_pytest.main.wrap_session` and so runs whatever happened.

    Ordering against `_no_blocked_connects_may_survive_the_session` does not
    matter: they report different things and neither depends on the other.
    Session-scoped fixtures are torn down before this hook, so nothing is
    recorded after the verdict is taken.
    """
    unlisted = unlisted_skips()
    if not unlisted:
        return
    lines = ["", "[skip audit] FAILURE"] + _audit_message(unlisted).splitlines()
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    for line in lines:
        if reporter is not None:
            reporter.write_line(line)
        else:  # `-p no:terminal`. Nothing would otherwise print the verdict.
            sys.stderr.write(line + "\n")
    # `session.exitstatus` is what `wrap_session` returns, so setting it here is
    # what actually fails the run. `pytest.ExitCode` is the public spelling.
    #
    # Only ever a promotion, never a demotion (CodeRabbit, Minor): an unlisted
    # skip can be recorded before a later interrupt or internal error in the same
    # session, and rewriting INTERRUPTED or INTERNAL_ERROR to TESTS_FAILED would
    # throw away the more specific diagnosis for whoever is reading the exit code.
    # OK and NO_TESTS_COLLECTED are the only codes worth upgrading -- the latter
    # is the empty-collection case this hook exists to cover, and upgrading it to
    # a plain failure is the point.
    if session.exitstatus not in _UPGRADABLE_EXIT_CODES:
        return
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


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


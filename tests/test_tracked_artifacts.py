"""`tests/tracked_artifacts.py` decides whether other tests are allowed to skip.

That is the whole of its job, and it is the job most worth testing. A helper that
gates a skip is not the sort of thing that earns trust by having a docstring: the
docstring is an argument, and the argument has to be checked against real git
repositories or it is decoration. PR #33's precedent applies -- it did not take
`tests/mutation_check.py`'s word that `build_tree` now refuses incomplete copies,
it tested that `build_tree` raises when a required path is missing.

So every branch is driven against a genuine repository built in `tmp_path` by
`git init` and `git commit`, rather than by monkeypatching a function to return
whatever the test wants. `git init` and `git commit` are local operations with no
network involved, and both queries read the same repository state in a synthetic
repo as in this one, so a stubbed `subprocess.run` would be testing the stub.

`_repo(..., commit=True)` is the default because the helper's whole design is
about the difference between the index and HEAD, and a repository with only a
staged index cannot exercise that difference.

Two of the outcomes are unreachable in this repository: every artefact the helper
is pointed at in `main` is committed, so "committed but missing" is the only live
failure here and "uncommitted and absent" never fires. Both are exercised below
against repositories built for the purpose, which is why these tests make their
own rather than reusing this checkout.

One thing to be clear about: this file passing does not prove the call sites are
gated. `test_the_real_checkout_reaches_the_live_branch` confirms the spelling used
in the call sites resolves and that the artefact is committed, and
`test_every_committed_artefact_read_is_gated_in_source` reads the test modules and
fails if any of them skips without asking -- so if a call site stops using
`require()`, that test is what notices.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath

import pytest

import tracked_artifacts
from tracked_artifacts import (
    TRACKED,
    UNTRACKED,
    UNVERIFIABLE,
    MissingTrackedArtifact,
    committed_state,
    in_head,
    in_index,
    require,
)

_GIT_IDENTITY = (
    "-c", "user.email=tests@example.invalid", "-c", "user.name=tests",
)


def _repo(tmp_path: Path, *tracked: str, commit: bool = True) -> Path:
    """A real git repository holding exactly `tracked`, committed by default.

    Committed rather than merely `git add`ed because the distinction between the
    index and HEAD is the thing under test: `committed_state` has to be able to
    tell a file that was removed from the index (`git rm --cached`) from one that
    was never in the repository, and a repo with no commit cannot say either.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    for rel in tracked:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
    if tracked:
        subprocess.run(["git", "add", "--", *tracked], cwd=repo, check=True)
    if commit and tracked:
        subprocess.run(
            ["git", *_GIT_IDENTITY, "commit", "--quiet", "-m", "seed"], cwd=repo, check=True
        )
    return repo


def _untrack(repo: Path, rel: str) -> None:
    """`git rm --cached` and delete: absent from disk *and* from the index.

    The operation this whole module is built to catch. Every commit still carries
    the file; the local checkout has stopped insisting on it.
    """
    subprocess.run(["git", "rm", "--cached", "--quiet", "--", rel], cwd=repo, check=True)
    (repo / rel).unlink()


# --- the two queries, on their own -----------------------------------------


def test_a_committed_file_is_in_the_index_and_in_head(tmp_path):
    repo = _repo(tmp_path, "data/snapshot.json")
    assert in_index("data/snapshot.json", repo_root=repo) is True
    assert in_head("data/snapshot.json", repo_root=repo) is True


def test_a_file_nobody_added_is_in_neither(tmp_path):
    repo = _repo(tmp_path, "data/snapshot.json")
    assert in_index("data/other.json", repo_root=repo) is False
    assert in_head("data/other.json", repo_root=repo) is False


def test_a_directory_is_not_a_file_is_tracked(tmp_path):
    """The reason a pathspec needs the `--` terminator and the exact filename.

    Git matches pathspecs by directory prefix, so asking about `data/snapshot`
    returns the entries underneath it. If this returned True for a path that was
    never added, `require()` would turn a genuinely uncommitted artefact into a
    hard failure -- the opposite error, but the same class of wrong answer, and
    it would be a wrong answer given with confidence.
    """
    repo = _repo(tmp_path, "data/snapshot.json")
    assert in_index("data/snapshot", repo_root=repo) is False


def test_a_similar_name_is_not_confused_for_the_file(tmp_path):
    repo = _repo(tmp_path, "data/snapshot.json")
    assert in_index("data/snapshot.json.bak", repo_root=repo) is False
    assert in_index("data/snapshot.js", repo_root=repo) is False
    assert in_head("data/snapshot.json.bak", repo_root=repo) is False


def test_a_path_that_looks_like_an_option_is_not_parsed_as_one(tmp_path):
    """Without `--`, a leading dash is an option and git would not be asked at all."""
    repo = _repo(tmp_path)
    assert in_index("-dangerous", repo_root=repo) is False


def test_outside_a_repository_is_none_not_false(tmp_path):
    """`None`, never `False`.

    `False` would reach the skip branch and report "this checkout has no copy" for
    a checkout whose committed-ness was never established -- which is the bug this
    module exists to fix, one indirection further away.
    """
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert in_index("data/snapshot.json", repo_root=not_a_repo) is None
    assert in_head("data/snapshot.json", repo_root=not_a_repo) is None
    assert committed_state("data/snapshot.json", repo_root=not_a_repo) is UNVERIFIABLE


def test_a_missing_repo_root_is_none_not_false(tmp_path):
    assert in_index("data/snapshot.json", repo_root=tmp_path / "nope") is None


def test_a_repository_with_no_commits_has_no_head_rather_than_an_error(tmp_path):
    """No HEAD is a fact about the repository, not a git malfunction.

    Distinguishing the two needs `git rev-parse --verify --quiet HEAD` rather than
    pattern-matching `ls-tree`'s error text, which would be both locale-dependent
    and git-version-dependent. Answering `None` here would make every artefact
    unverifiable in a fresh repository, which is a confusing way to say "you have
    not committed anything yet".
    """
    repo = _repo(tmp_path, "data/snapshot.json", commit=False)
    assert in_index("data/snapshot.json", repo_root=repo) is True
    assert in_head("data/snapshot.json", repo_root=repo) is False
    assert committed_state("data/snapshot.json", repo_root=repo) is TRACKED


# --- committed_state: the decision a skip rests on --------------------------


def test_a_committed_file_that_is_still_on_disk_is_tracked(tmp_path):
    repo = _repo(tmp_path, "data/snapshot.json")
    assert committed_state("data/snapshot.json", repo_root=repo) is TRACKED


def test_a_deleted_but_still_indexed_file_is_tracked(tmp_path):
    """`git rm --cached` was not run; the file was simply moved aside."""
    repo = _repo(tmp_path, "data/snapshot.json")
    (repo / "data/snapshot.json").unlink()
    assert in_index("data/snapshot.json", repo_root=repo) is True
    assert committed_state("data/snapshot.json", repo_root=repo) is TRACKED


def test_an_untracked_locally_but_committed_file_is_still_tracked(tmp_path):
    """The case a single `git ls-files` query gets wrong.

    `git rm --cached` leaves every commit carrying the file while removing it
    from the index and from disk. Index-only would answer "not tracked" and skip,
    so the one operation that can quietly untrack a load-bearing artefact would be
    the operation that switched off its own guard.
    """
    repo = _repo(tmp_path, "data/snapshot.json")
    _untrack(repo, "data/snapshot.json")
    assert in_index("data/snapshot.json", repo_root=repo) is False
    assert in_head("data/snapshot.json", repo_root=repo) is True
    assert committed_state("data/snapshot.json", repo_root=repo) is TRACKED


def test_a_file_no_commit_ever_carried_is_untracked(tmp_path):
    repo = _repo(tmp_path, "data/other.json")
    assert committed_state("data/snapshot.json", repo_root=repo) is UNTRACKED


def test_a_file_deleted_in_the_commit_is_untracked(tmp_path):
    """Deliberately, and visibly.

    Once the deletion is a commit the artefact is genuinely not in the
    repository, and this cannot tell it from a file that never existed. That is
    the honest limit: the deletion is a line in the diff and the reading test has
    to go in the same commit. What this module refuses is the *uncommitted* half,
    where nobody would see anything.
    """
    repo = _repo(tmp_path, "data/snapshot.json")
    subprocess.run(["git", "rm", "--quiet", "--", "data/snapshot.json"], cwd=repo, check=True)
    subprocess.run(
        ["git", *_GIT_IDENTITY, "commit", "--quiet", "-m", "drop it"], cwd=repo, check=True
    )
    assert committed_state("data/snapshot.json", repo_root=repo) is UNTRACKED


# --- require: the outcomes that must not skip --------------------------------


def test_a_committed_file_that_is_not_on_disk_fails_loudly(tmp_path):
    """The regression. Moving the artefact aside is a *tracked* absence.

    The call sites used to answer this with `pytest.skip`, so deleting a committed
    artefact left the suite green.
    """
    repo = _repo(tmp_path, "data/snapshot.json")
    (repo / "data/snapshot.json").unlink()

    with pytest.raises(MissingTrackedArtifact) as excinfo:
        require(
            repo / "data/snapshot.json",
            rel="data/snapshot.json",
            required_because="two tests assert properties of it.",
            expected_absent_because="not expected in any checkout",
            repo_root=repo,
        )

    message = str(excinfo.value)
    assert "data/snapshot.json" in message, message
    assert "git checkout" in message, message
    assert "two tests assert properties of it." in message, message


def test_git_rm_cached_fails_loudly_too(tmp_path):
    """The other way to make a committed artefact vanish, and the sneakier one."""
    repo = _repo(tmp_path, "data/snapshot.json")
    _untrack(repo, "data/snapshot.json")

    with pytest.raises(MissingTrackedArtifact) as excinfo:
        require(
            repo / "data/snapshot.json",
            rel="data/snapshot.json",
            required_because="two tests assert properties of it.",
            expected_absent_because="not expected in any checkout",
            repo_root=repo,
        )

    message = str(excinfo.value)
    assert "committed to this repository" in message, message
    # The advice has to name the difference, or someone facing this reads "commit
    # the removal" and runs `git rm --cached`.
    assert "not `git rm --cached`" in message, message


def test_the_uncommitted_and_absent_case_skips_and_says_what_is_missing(tmp_path):
    """The one outcome allowed to skip, and the reason must be specific.

    A skip that fires for the wrong cause is as bad as no check at all, so the
    message is checked as content rather than as a type: it has to name the path,
    say that neither the index nor a commit carries it, and carry the caller's own
    statement of why the absence is expected.
    """
    repo = _repo(tmp_path, "data/other.json")

    with pytest.raises(pytest.skip.Exception) as excinfo:
        require(
            repo / "data/snapshot.json",
            rel="data/snapshot.json",
            required_because="two tests assert properties of it.",
            expected_absent_because="the nightly job publishes it, so a checkout "
                                     "made before it ran has nothing to check.",
            repo_root=repo,
        )

    message = str(excinfo.value)
    assert "data/snapshot.json" in message, message
    assert "neither the index nor any commit" in message, message
    assert "made before it ran has nothing to check." in message, message


def test_a_git_that_cannot_answer_fails_rather_than_skipping(tmp_path):
    """Unverifiable is its own outcome and it is a failure.

    Reproduced by pointing `require` at a directory outside any repository,
    because that is the case a developer actually hits and it needs no monkeypatch
    to bring about.
    """
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()

    with pytest.raises(MissingTrackedArtifact) as excinfo:
        require(
            not_a_repo / "data/snapshot.json",
            rel="data/snapshot.json",
            required_because="two tests assert properties of it.",
            expected_absent_because="not expected",
            repo_root=not_a_repo,
        )

    message = str(excinfo.value)
    assert "cannot establish whether data/snapshot.json is committed" in message, message
    assert "Not skipping" in message, message
    # The refusal must name the underlying git error, or "cannot ask git" is a
    # thing the reader has to reproduce to believe.
    assert "rc=128" in message, message


def test_a_git_that_cannot_be_executed_fails_rather_than_skipping(tmp_path, monkeypatch):
    """The other way git cannot answer: not installed, or not executable.

    `subprocess.run` raises `FileNotFoundError` for a missing binary and
    `PermissionError` for one it may not run; both are `OSError`. Letting either
    escape would surface as an error naming a `FileNotFoundError` rather than as
    the deliberate refusal this is, and an error is not a message anybody reads.
    """
    repo = _repo(tmp_path, "data/snapshot.json")
    (repo / "data/snapshot.json").unlink()

    def no_such_binary(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(tracked_artifacts.subprocess, "run", no_such_binary)

    with pytest.raises(MissingTrackedArtifact) as excinfo:
        require(
            repo / "data/snapshot.json",
            rel="data/snapshot.json",
            required_because="two tests assert properties of it.",
            expected_absent_because="not expected",
            repo_root=repo,
        )
    assert "cannot establish whether data/snapshot.json is committed" in str(excinfo.value)


def test_a_git_that_fails_on_head_only_still_refuses_to_skip(tmp_path, monkeypatch):
    """`in_index` saying "not tracked" is not enough on its own.

    If the HEAD query cannot be answered, the artefact has not been shown to be
    absent from every commit, and the skip is not earned. Monkeypatched because
    this needs a git that works for one query and not the other, which no real
    repository produces.
    """
    repo = _repo(tmp_path, "data/other.json")
    real_run = tracked_artifacts.subprocess.run

    def only_index(*args, **kwargs):
        if "ls-tree" in args[0]:
            return subprocess.CompletedProcess(args[0], 128, "", "fatal: broken\n")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(tracked_artifacts.subprocess, "run", only_index)

    with pytest.raises(MissingTrackedArtifact) as excinfo:
        require(
            repo / "data/snapshot.json",
            rel="data/snapshot.json",
            required_because="two tests assert properties of it.",
            expected_absent_because="not expected",
            repo_root=repo,
        )
    assert "cannot establish whether data/snapshot.json is committed" in str(excinfo.value)
    assert "rc=128" in str(excinfo.value)


# --- require: the paths that let a test read --------------------------------


def test_a_present_file_is_returned_unchanged(tmp_path):
    repo = _repo(tmp_path, "data/snapshot.json")
    path = require(
        repo / "data/snapshot.json",
        rel="data/snapshot.json",
        required_because="two tests assert properties of it.",
        expected_absent_because="not expected",
        repo_root=repo,
    )
    assert path == repo / "data/snapshot.json"
    assert path.read_text(encoding="utf-8") == "{}\n"


def test_an_uncommitted_file_that_happens_to_be_present_is_still_returned(tmp_path):
    """Present beats committed.

    A local artefact nobody committed is still a file with contents, and skipping
    would be the wrong call even though skipping is right for the *absent*
    uncommitted case. One question, decided by presence first -- so the answer
    cannot be "it depends which branch we took".
    """
    repo = _repo(tmp_path, "data/other.json")
    (repo / "data/snapshot.json").write_text('{"weeks": {}}\n', encoding="utf-8")

    path = require(
        repo / "data/snapshot.json",
        rel="data/snapshot.json",
        required_because="two tests assert properties of it.",
        expected_absent_because="not expected",
        repo_root=repo,
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {"weeks": {}}


def test_a_present_file_is_never_interrogated(tmp_path):
    """A file that is there needs no git call, so no git call can fail on it.

    The check is the *absence* of the artefact that has to be decidable. Counting
    the subprocesses keeps a future refactor from moving the queries above the
    `path.exists()` early return and turning an ordinary test run into one that
    depends on git being installed.
    """
    repo = _repo(tmp_path, "data/snapshot.json")
    calls = []
    real_run = tracked_artifacts.subprocess.run

    def counting(command, **kwargs):
        calls.append(command[1])
        return real_run(command, **kwargs)

    tracked_artifacts.subprocess.run = counting
    try:
        require(
            repo / "data/snapshot.json",
            rel="data/snapshot.json",
            required_because="r",
            expected_absent_because="e",
            repo_root=repo,
        )
    finally:
        tracked_artifacts.subprocess.run = real_run
    assert calls == [], calls


def test_a_misspelt_rel_fails_instead_of_skipping(tmp_path):
    """The one input that could bring the original bug back.

    A `rel` naming a different file gets "not committed" from git and the absence
    then skips -- exactly the hole, reached by a typo. So the spelling is
    recomputed from the path and the two must agree.
    """
    repo = _repo(tmp_path, "data/snapshot.json")
    (repo / "data/snapshot.json").unlink()

    with pytest.raises(MissingTrackedArtifact) as excinfo:
        require(
            repo / "data/snapshot.json",
            rel="data/snapshot.json.bak",
            required_because="two tests assert properties of it.",
            expected_absent_because="not expected",
            repo_root=repo,
        )
    assert "was told this path is" in str(excinfo.value)


def test_a_path_outside_the_repo_root_fails_instead_of_skipping(tmp_path):
    repo = _repo(tmp_path, "data/snapshot.json")
    with pytest.raises(MissingTrackedArtifact) as excinfo:
        require(
            tmp_path / "elsewhere/snapshot.json",
            rel="data/snapshot.json",
            required_because="two tests assert properties of it.",
            expected_absent_because="not expected",
            repo_root=repo,
        )
    assert "not inside" in str(excinfo.value)


# --- the helper is reachable from the tests it is for ------------------------


def test_the_real_checkout_reaches_the_live_branch():
    """`require()` on this repository's own committed artefact returns the file.

    Not a mock: the point is that the spelling used in the call sites resolves,
    that the artefact is committed, and that the present case returns rather than
    skipping. If `rel` were wrong here, the three call sites would be skipping
    and nothing in this file would notice.
    """
    from nfl_predictor.config import PUBLIC_SNAPSHOT_PATH

    assert committed_state("data/public_snapshot.json") is TRACKED, (
        "data/public_snapshot.json is the artefact two tests guard and it must be "
        "committed; if it is not, it is uncommitted, the call sites' skip branch "
        "is live for the wrong reason, and the staleness gate PR #29 relies on "
        "has lost its input"
    )
    path = require(
        PUBLIC_SNAPSHOT_PATH,
        rel="data/public_snapshot.json",
        required_because="public_snapshot.assert_publishable refuses to publish "
                         "a snapshot older than the manifest it was built from.",
        expected_absent_because="not expected: the file is committed.",
    )
    assert path == PUBLIC_SNAPSHOT_PATH
    assert json.loads(path.read_text(encoding="utf-8"))["weeks"]


def test_every_scan_root_this_suite_relies_on_is_committed():
    """The three `test_passing_td_record_absence.py` roots, checked directly.

    Those roots' absence is a failure rather than a skip *because* they are
    committed. If one were ever uncommitted the distinction would stop being the
    right one, and it would stop being right quietly -- the tests would go on
    skipping. So the premise is asserted rather than assumed.
    """
    from test_passing_td_record_absence import SCANNED

    for rel in SCANNED:
        assert committed_state(rel) is TRACKED, (
            f"{rel} is not committed, so the absence of a scan root there would "
            f"be reported as an expected absence rather than a broken checkout. "
            f"Either commit it, or remove it from SCANNED and say in that "
            f"module's docstring why a root is optional."
        )


def test_every_committed_artefact_read_is_gated_in_source():
    """No test module guards a read of a committed artefact with a bare skip.

    Written as a scan of the test sources rather than as three separate
    assertions, so the next test that reaches for `pytest.skip` on a committed
    file is caught by the same check. It looks for the *idiom* -- a `pytest.skip`
    in a module that reads one of the repo's committed artefacts -- and requires
    that the skip go through `tracked_artifacts.require`, which is what
    distinguishes committed-but-missing from uncommitted-and-absent.

    A candidate that imports `tracked_artifacts` is scanned exactly like one that
    does not. It used to be skipped over, which meant the escape was not "skip
    without `require()`" but "skip without `require()` *and* without so much as
    importing the helper" -- in the one test whose whole job is catching ungated
    skips. A module can read a committed artefact through a helper it imports
    itself, or through `tracked_artifacts` for an unrelated call, and either way
    its own bare `pytest.skip` is still ungated.

    **The candidate filter used to name the two artefacts rather than the
    property, and that was the second hole.** A module was scanned only if its
    source mentioned `PUBLIC_SNAPSHOT_PATH` or `SCANNED`, so a module that read
    some *other* committed artefact and skipped was never examined at all -- the
    audit could not report on it because it never looked. `_names_a_committed_artefact`
    below decides it the other way round: take the repo-relative path literals a
    module mentions and ask `committed_state()`, so the filter is "reads a
    committed artefact" rather than "reads one of the two we remembered". The
    two named triggers are kept *in addition*, because a module can reach the
    snapshot through `config.PUBLIC_SNAPSHOT_PATH` without the path appearing as
    a literal at all.
    """
    assert _ungated_skip_sites(Path(__file__).resolve().parent) == []


#: Repo-relative paths as they are spelled in source and in git. Anchored at a
#: word boundary and required to carry an extension, because that is the shape
#: every committed artefact and every test path has, and the alternative --
#: asking git about each bare word -- is thousands of subprocesses per run.
_PATH_LITERAL = re.compile(
    r"(?<![\w./-])((?:[\w.-]+/)+[\w.-]+\.\w+)(?![\w./-])"
)


def _committed_paths(paths: set[str]) -> set[str]:
    """Of these candidate paths, the ones `committed_state()` says are committed.

    The decision is `committed_state()`'s and nothing else's. It is also the same
    decision `require()` makes, which is the point: a filter with its own idea of
    what "committed" means is a filter that will disagree with the gate it is
    protecting, and it will disagree silently, in the direction that lets a skip
    through.

    Asking git costs two subprocesses per path, so the caller dedupes across the
    whole directory first: 85 distinct paths appear in `tests/` today, of which
    42 are committed. That is ~170 local `git` invocations per run, roughly two
    seconds, and it buys a filter that cannot go stale. There is deliberately no
    cache across runs -- `tracked_artifacts` refuses to cache `committed_state`
    precisely so a file removed mid-session cannot keep reporting the answer from
    before, and a cached candidate list would be that same staleness one level up.
    """
    return {rel for rel in paths if committed_state(rel) is TRACKED}


def _names_a_committed_artefact(source: str, committed: set[str], *, self_name: str) -> bool:
    """Does this module's text mention one of the paths `_committed_paths` cleared?

    A pure membership test against the already-resolved set, so this function
    costs no subprocesses and can be called per module freely.

    Note the scan is over the module's **text**, not its AST: the point is to
    notice that a path is *named*, and a path named only in a docstring is still a
    module that reasons about that artefact.

    `self_name` is excluded because nearly every module's docstring cites its own
    path, and a self-citation is not a read. Without that exclusion the filter
    matches every module in `tests/`, which would leave it nominally general and
    practically vacuous -- the same failure as naming two artefacts by hand, one
    level up. The comparison is on the *basename* rather than the full
    repo-relative path because the directory being scanned is a parameter and its
    prefix is not this function's to know; a module's own citation always ends in
    its filename, and two modules in one flat directory cannot share a basename.

    Two stated limits, neither an oversight: a bare filename with no directory
    (`README.md`) is not matched, and neither is a path assembled at runtime by
    joining pieces. The first cannot reach `require()`'s callers, which always
    pass a repo-relative `rel`; the second is the same gap the AST-level
    `pytest.skip` scan has. Both err toward over-inclusion where they do fire: a
    docstring mention makes a module a candidate it may not be, and a candidate
    with no bare `pytest.skip` costs nothing.
    """
    return any(
        rel in committed
        for rel in dict.fromkeys(_PATH_LITERAL.findall(source))
        if PurePosixPath(rel).name != self_name
    )


def _ungated_skip_sites(tests_dir: Path) -> list[str]:
    """`file:line` for every `pytest.skip` in a module that reads a committed artefact.

    Split out of the test so a caller can point it at a directory that is not
    this one -- which is how the filter itself gets tested, rather than only the
    scan running over the suite.

    This is a *source* scan and is therefore the second line of defence, not the
    first. It cannot see a `pytest.skip` that pytest itself raises
    (`importorskip`, a `skipif` mark, a fixture, a module-level skip reached
    through a helper) and it cannot see a path a module assembles at runtime.
    Those are the runtime hook's job in `tests/conftest.py`, which sees every
    skip pytest reports regardless of how it was spelled. The two are not
    redundant: the hook cannot tell whether a skip was *justified*, and this scan
    cannot see the skip at all.

    `committed_state()` always consults *this* repository, whatever `tests_dir` is
    passed: the question "is this path committed?" is only meaningful against a
    real index and a real HEAD, so pointing this at a synthetic directory cannot
    invent artefacts. That is also what makes the test below possible -- a
    synthetic module that names a path genuinely committed here is a candidate
    even though the directory it sits in is not.
    """
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(tests_dir.glob("test_*.py"))
        if path.name != "test_tracked_artifacts.py"  # this file names the idiom to forbid it
    }
    # One dedupe across every module, so git is asked once per distinct path in
    # the directory rather than once per module that mentions it.
    committed = _committed_paths({
        rel for source in sources.values() for rel in _PATH_LITERAL.findall(source)
    })
    offenders: list[str] = []
    for name, source in sources.items():
        if not (
            _names_a_committed_artefact(source, committed, self_name=name)
            or "PUBLIC_SNAPSHOT_PATH" in source
            or "SCANNED" in source
        ):
            continue
        for node in ast.walk(ast.parse(source, filename=name)):
            if isinstance(node, ast.Call) and _is_pytest_skip(node.func):
                offenders.append(f"{name}:{node.lineno}")
    return offenders


def _is_pytest_skip(func: ast.expr) -> bool:
    if not isinstance(func, ast.Attribute) or func.attr != "skip":
        return False
    return isinstance(func.value, ast.Name) and func.value.id == "pytest"


# --- the two holes the audit had, red-checked by removing the fix ------------
#
# Everything below runs a *real* `pytest` in a subprocess rather than calling the
# hook with a hand-built report object. That is the whole point of both tests: the
# module-level-skip hole is a property of pytest's own control flow -- a
# `Skipped` raised while a module is being imported produces no runtest report at
# all -- so a test that fed the audit a fabricated `CollectReport` would pass
# while the real thing stayed broken. The subprocess is what makes the red-check
# meaningful. It costs about a second each.
#
# `tests/conftest.py` is *copied* rather than imported, because the hooks have to
# be installed by pytest's plugin manager in the child process; an import in this
# process registers nothing there. The copy is the file under test, so a change
# that stops recording is a change that fails here.


def _run_audit_on(
    tmp_path: Path, modules: dict[str, str], *, conftest_extra: str = ""
) -> subprocess.CompletedProcess:
    """Run this suite's real conftest over `modules` in a child pytest.

    The child runs with the network guard the copied conftest installs at import
    time, so a skip that reached for the internet would be blocked rather than
    fetched -- these tests are inside the "never touches the network" rule rather
    than beside it.

    `conftest_extra` is appended to the copy, which is how a test lists a skip site
    as allowed. It is appended rather than substituted so the child always runs
    the real file; there is no way to hand the child a different audit.

    `PYTEST_DISABLE_PLUGIN_AUTOLOAD` keeps the child's plugin set to pytest's own
    plus this conftest, so a plugin installed in this environment cannot change
    what the child reports. It is set rather than unset in the environment copy:
    the whole environment is passed, because the child needs the same interpreter
    and import path as the parent.
    """
    shutil.copyfile(Path(__file__).parent / "conftest.py", tmp_path / "conftest.py")
    if conftest_extra:
        with (tmp_path / "conftest.py").open("a", encoding="utf-8") as fh:
            fh.write(f"\n{conftest_extra}")
    for name, body in modules.items():
        (tmp_path / name).write_text(textwrap.dedent(body), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )


_SELF_SKIPPING = """
    import pytest

    pytest.skip("models/manifest.json is not in this checkout", allow_module_level=True)

    def test_never_runs():
        assert False
"""

_PASSING = """
    def test_runs():
        assert True
"""


def test_a_module_that_skips_itself_out_fails_the_run(tmp_path):
    """A module-level `pytest.skip(allow_module_level=True)` must not be a clean run.

    The hole, stated as a property: with the audit on `origin/main`, this child
    run reported `1 skipped`, exited **0**, and `unlisted_skips()` was empty --
    because no test item is ever created for a module that skips at import, so
    `pytest_runtest_logreport` never fires and nothing is recorded. The suite
    went green having checked nothing, and the guard certified it.

    Asserted on both halves separately because either alone would let the defect
    back in: the exit code is what fails CI, and the message is what tells a
    reader *which* module and line to go and write down.
    """
    proc = _run_audit_on(tmp_path, {"test_selfskip.py": _SELF_SKIPPING, "test_ok.py": _PASSING})

    assert proc.returncode != 0, (
        f"a module skipped itself out of the run and the run still passed:\n{proc.stdout}"
    )
    assert "1 skipped" in proc.stdout, (
        f"expected the module to be reported skipped, so the audit had something "
        f"to disagree with:\n{proc.stdout}"
    )
    # Line 4 of `_SELF_SKIPPING` above is the `pytest.skip(...)` call, and it is
    # the line `CollectReport.longrepr` carries -- verified against a real child
    # run rather than assumed, since "the line of the skip call" is precisely what
    # a SKIP_SITES entry has to be able to name.
    assert (
        "test_selfskip.py skipped at test_selfskip.py:4"
        in proc.stdout + proc.stderr
    ), f"{proc.stdout}\n{proc.stderr}"


def test_a_listed_module_level_skip_still_passes(tmp_path):
    """The other direction: the fix must not make module-level skips unusable.

    Recorded-but-always-failing would satisfy the test above while making the
    `allow_module_level=True` idiom a landmine. So this writes the site down, in
    a `SKIP_SITES` entry keyed exactly as a test-level skip is, and requires the
    run to pass. It also pins the fact that the entry needs no special syntax for
    a module-level skip -- `(basename, line)` is the whole key.

    The exit code alone would not be enough here: a run that recorded *nothing*
    also exits 0, which is precisely the `origin/main` behaviour. So the child is
    also asked to report its own `SKIPS`, and the test requires the module-level
    skip to be *in* it. Without that, this test would pass against the very defect
    it is the counterpart to.

    The entry and the probe are appended to the *copied* conftest rather than
    injected, because the table is a module-level dict the audit reads through the
    same copy; there is no second mechanism for listing a site that could drift
    from the one the real suite uses.
    """
    proc = _run_audit_on(
        tmp_path,
        {
            "test_selfskip.py": _SELF_SKIPPING,
            "test_ok.py": _PASSING,
            # Collected after the self-skipping module, so `SKIPS` already holds
            # its record. Asserting the exact tuple is what makes "recorded" and
            # "listed" two separate claims rather than one.
            "test_probe.py": (
                "from conftest import SKIPS\n"
                "\n"
                "def test_probe():\n"
                "    assert SKIPS == [('test_selfskip.py', 4, 'test_selfskip.py',\n"
                "        'Skipped: models/manifest.json is not in this checkout')], SKIPS\n"
            ),
        },
        conftest_extra='SKIP_SITES[("test_selfskip.py", 4)] = "listed for the test above"\n',
    )

    assert proc.returncode == 0, (
        f"a module-level skip written down in SKIP_SITES failed the run anyway -- "
        f"so the fix made allow_module_level unusable rather than accountable, or "
        f"it was never recorded and `SKIPS` is empty:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "1 skipped" in proc.stdout and "not in SKIP_SITES" not in proc.stdout


def test_the_audit_still_fails_when_collection_leaves_nothing_to_run(tmp_path):
    """The whole suite skipped away is the worst shape, and the fixture could not see it.

    PR #34 rendered its verdict from a session-scoped autouse fixture, which is
    only set up when a test item runs. So the single most thorough version of the
    defect -- every module gone, no tests, `exitstatus` NO_TESTS_COLLECTED --
    produced no audit at all, and NO_TESTS_COLLECTED is reported by pytest as a
    separate outcome that does not read as a verdict on the skips. This asserts
    the hook runs anyway.

    Red-checked on `origin/main` this is not a clean single failure: the audit
    never runs, so the child exits 5 and the assertion on the verdict message is
    what fails. That is the point -- the defect is an *absence*, so proving it
    requires asserting the message is there.
    """
    proc = _run_audit_on(tmp_path, {"test_selfskip.py": _SELF_SKIPPING})

    assert proc.returncode == pytest.ExitCode.TESTS_FAILED, (
        f"expected TESTS_FAILED (1) from the audit, got {proc.returncode}. An "
        f"empty-collection run is where the old fixture-based audit was "
        f"silenced by the very thing it audits:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "1 skipped" in proc.stdout
    assert "not in SKIP_SITES" in proc.stdout + proc.stderr


def test_the_audit_does_not_downgrade_a_more_specific_exit_status(tmp_path):
    """A later interrupt keeps its own exit code, even with an unlisted skip recorded.

    The hook promotes `session.exitstatus`, and "promote" has to mean only that:
    rewriting INTERRUPTED or INTERNAL_ERROR to TESTS_FAILED would discard the
    specific diagnosis for whoever reads the exit code -- a Ctrl-C would arrive
    looking like a skip-audit failure. So a run that both skips unlisted *and* is
    interrupted must report INTERRUPTED.

    Driven with a real interrupt rather than a stubbed `exitstatus`, because the
    question is whether pytest's own teardown leaves the code it wants: the hook
    runs after that, and a test that set the field by hand would not show it.
    """
    interrupted = """
        import pytest

        pytest.skip("models/manifest.json is not in this checkout", allow_module_level=True)

        def test_never_runs():
            assert False
    """
    conftest = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8") + (
        "\n"
        "# Interrupt *after* the module-level skip has been recorded, which is the"
        "\n"
        "# ordering that makes the promotion-vs-demotion question real."
        "\n"
        "def pytest_collection_finish(session):\n"
        "    if session.testscollected == 0:\n"
        "        pytest.exit('interrupted by the test', returncode=2)\n"
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "conftest.py").write_text(conftest, encoding="utf-8")
    (tmp_path / "test_selfskip.py").write_text(textwrap.dedent(interrupted), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )

    assert proc.returncode == pytest.ExitCode.INTERRUPTED, (
        f"the audit overwrote INTERRUPTED (2) with {proc.returncode}; an interrupt "
        f"must keep its own diagnosis:\n{proc.stdout}\n{proc.stderr}"
    )
    # And the verdict is still reported -- a promotion that also silences the
    # message would be no better than a demotion.
    assert "not in SKIP_SITES" in proc.stdout + proc.stderr


def test_a_test_level_skip_still_fails_the_run_the_same_way(tmp_path):
    """Nothing about the module-level path weakened the ordinary one.

    Worth a test of its own because the fix moved the verdict out of a fixture
    and into `pytest_sessionfinish`: a mistake there could easily have left the
    test-level case passing while reporting, which is the same defect inverted.
    """
    skipping_test = """
        import pytest

        def test_skips():
            pytest.skip("json is stdlib")

        def test_passes():
            assert True
    """
    proc = _run_audit_on(tmp_path, {"test_skips.py": skipping_test})

    assert proc.returncode != 0, f"a test-level unlisted skip passed:\n{proc.stdout}"
    assert (
        "test_skips.py::test_skips skipped at test_skips.py:5"
        in proc.stdout + proc.stderr
    ), f"{proc.stdout}\n{proc.stderr}"


def test_a_skip_if_mark_still_fails_the_run(tmp_path):
    """`@pytest.mark.skipif` is recorded too, and attributed to its decorator line.

    A mark-based skip never calls `pytest.skip`, so it is invisible to every
    source scan in this file -- which is exactly why the runtime hook is the
    primary defence rather than a backstop.

    Two shapes, because they report differently and the difference is a fact a
    `SKIP_SITES` author needs. A skip on a *function* carries the mark's own line
    (4 here). A skip on a *class* carries the line of the first test in it (10),
    not the decorator on the class (9) -- `longrepr` falls back to `location` for
    these, and an item's location is its own first line. Both are asserted as
    observed rather than as predicted, since neither is documented by pytest and
    a future version could move either.
    """
    marked = """
        import pytest

        @pytest.mark.skipif(True, reason="not applicable here")
        def test_marked():
            assert True

        @pytest.mark.skipif(True, reason="not applicable here either")
        class TestClass:
            def test_inside(self):
                assert True
    """
    proc = _run_audit_on(tmp_path, {"test_marked.py": marked})

    assert proc.returncode != 0, f"a skipif mark passed the audit:\n{proc.stdout}"
    out = proc.stdout + proc.stderr
    assert "2 skip(s) this run are not in SKIP_SITES" in out, out
    assert "test_marked.py::test_marked skipped at test_marked.py:4" in out, out
    assert "test_marked.py::TestClass::test_inside skipped at test_marked.py:10" in out, out


def test_a_module_reading_an_unnamed_committed_artefact_is_a_candidate(tmp_path):
    """The filter covers committed artefacts, not the two someone remembered.

    The second hole, red-checked by pointing `_ungated_skip_sites` at a directory
    rather than at this suite. The synthetic module mentions
    `models/manifest.json` -- committed, and named by neither `PUBLIC_SNAPSHOT_PATH`
    nor `SCANNED` -- and skips. Under `origin/main`'s filter it was never examined
    and this returned `[]`; the real hole is precisely that a module can read a
    committed artefact, skip, and be invisible, because the audit decided in
    advance which artefacts were worth looking at.
    """
    reader = """
        import pytest

        MODELS = "models/manifest.json"

        def test_reads_the_manifest():
            path = Path(MODELS)
            if not path.exists():
                pytest.skip("models/manifest.json is not in this checkout")
    """
    (tmp_path / "test_reader.py").write_text(
        "from pathlib import Path\n" + textwrap.dedent(reader), encoding="utf-8"
    )

    offenders = _ungated_skip_sites(tmp_path)
    assert offenders == ["test_reader.py:10"], (
        f"a module reading a committed artefact outside the two named ones was "
        f"not reported as a candidate: {offenders}"
    )


def test_the_filter_ignores_a_module_that_reads_no_committed_artefact(tmp_path):
    """The other direction again, so the widened filter cannot be a blanket.

    Two modules that skip and mention a path, neither of them committed here: a
    package on PyPI and an artefact a hypothetical future module might name.
    Both must stay out of the offender list, or every `pytest.skip` in the suite
    would be reported and the audit would be noise nobody reads.
    """
    (tmp_path / "test_reader.py").write_text(
        textwrap.dedent(
            '''
            import pytest

            def test_reads_something_uncommitted():
                path = Path("nflverse-data/releases/stats_player/week.parquet")
                if not path.exists():
                    pytest.skip("the upstream parquet is not downloaded")
            '''
        ),
        encoding="utf-8",
    )
    assert _ungated_skip_sites(tmp_path) == []


def test_no_skip_in_this_suite_goes_without_a_reason():
    """Every `pytest.skip` names why. The general form of the hole this fixes.

    The three that motivated `tracked_artifacts.py` did carry a reason -- a wrong
    one. So this does not check that a reason exists, it checks the weaker
    property that has to hold no matter what: a reader of the summary must be able
    to tell what was not tested from the line pytest prints. A bare `pytest.skip()`
    prints nothing but a file and a line number.
    """
    tests_dir = Path(__file__).resolve().parent
    unreasoned: list[str] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_pytest_skip(node.func)):
                continue
            if node.args or [kw.value for kw in node.keywords]:
                continue
            unreasoned.append(f"{path.name}:{node.lineno}")
    assert not unreasoned, (
        f"these skips pass no reason, so a run that reports them says only where "
        f"to look: {unreasoned}. pytest.importorskip is exempt -- it carries its "
        f"own."
    )


# --- the seventh path, measured rather than assumed -------------------------
#
# Every claim below was driven through the real child-run harness above, so the
# findings are observed rather than reasoned about. They are recorded here because
# a guard that has needed three follow-ups for holes is itself evidence about its
# coverage, and the next person should not have to re-derive the list.


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(
            'import pytest\n\ndef test_it():\n    pytest.importorskip("a_module_absent_xyz")\n',
            "test_shapes.py::test_it skipped at test_shapes.py:4",
            id="importorskip",
        ),
        pytest.param(
            'import pytest\n\n@pytest.fixture\ndef f():\n    pytest.skip("nope")\n\n'
            'def test_it(f):\n    assert False\n',
            "test_shapes.py::test_it skipped at test_shapes.py:7",
            id="skip-inside-a-fixture",
        ),
    ],
)
def test_every_skip_shape_pytest_reports_is_recorded(tmp_path, body, expected):
    """`importorskip` and a `pytest.skip` raised from a fixture are both caught.

    Neither calls `pytest.skip` at a line any source scan would attribute a site
    to in the ordinary way -- `importorskip` raises it internally, and a fixture's
    skip is reported as the *test's* skip at the fixture's own line. They are here
    to pin the claim in `tests/conftest.py` that the runtime hook covers every
    shape pytest reports, rather than only the ones the source scans can see.

    Each is driven as a real child run, because "is this shape recorded" is a
    property of pytest's reporting, not of this file.
    """
    proc = _run_audit_on(tmp_path, {"test_shapes.py": body})

    assert proc.returncode != 0, f"the skip went unrecorded:\n{proc.stdout}"
    assert expected in proc.stdout + proc.stderr, f"{proc.stdout}\n{proc.stderr}"


def test_a_skip_pytest_does_not_report_cannot_be_caught_by_this_audit(tmp_path):
    """The one hole that stays open, and why it is not worth closing in this file.

    A fixture that wraps `pytest.skip` in `try/except` swallows the `Skipped`
    exception: the test body runs and the test reports `passed`. There is no
    report to record and nothing for `pytest_collectreport` or
    `pytest_runtest_logreport` to see, because the skip stopped being a skip
    before pytest heard about it. `tests/conftest.py`'s docstring says
    "recorded wherever pytest emits a report carrying `skipped`" for this reason,
    and this test is what keeps that claim honest.

    So the assertion is the *negative* one: the swallowing fixture's test passes
    and contributes nothing to `SKIPS`. A future change that made this test fail
    would mean the audit had gained the ability to see swallowed skips, which is
    worth knowing; a change that made it pass by asserting a skip *was* caught
    would be a lie.

    The other names in the brief are checked by `test_no_unittest_or_xfail_escapes_here`
    and `test_this_suite_contains_no_early_return_in_a_test` below, so the claim
    that they are absent is asserted rather than left in prose.
    """
    swallowing = """
        import pytest

        @pytest.fixture
        def swallow():
            try:
                pytest.skip("swallowed")
            except BaseException:
                pass
            yield

        def test_reports_passed(swallow):
            assert True
    """
    proc = _run_audit_on(tmp_path, {"test_swallow.py": swallowing})

    assert proc.returncode == 0, (
        f"expected the swallowing fixture to hide the skip entirely:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "1 passed" in proc.stdout and "not in SKIP_SITES" not in proc.stdout


def test_no_unittest_or_xfail_escapes_here():
    """`unittest.skip`, `skipIf` and `xfail` are absent, and stay that way.

    The last three shapes from the brief, checked by source scan because none of
    them can be checked by a report: an `xfail` that fires reports `xfailed` and
    an `xfail(strict=False)` that unexpectedly *passes* reports `passed`, so
    neither produces a `skipped` report for the audit to record, and both are a
    test declining to test. `unittest.skip` on a `TestCase` method does produce a
    skipped report, so it would be caught at runtime -- but it would be caught by
    a key (`class.py`, `line`) nobody has ever written a `SKIP_SITES` entry for,
    which is the same "invisible by construction" shape as the two holes above.

    Asserted here so the answer is a fact about the suite rather than a note in a
    docstring that goes stale the first time someone adds a `TestCase`.
    """
    tests_dir = Path(__file__).resolve().parent
    found: dict[str, list[str]] = {"unittest": [], "xfail": []}
    for path in sorted(tests_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found["unittest"] += [
                    f"{path.name}:{node.lineno}" for a in node.names if a.name == "unittest"
                ]
            elif isinstance(node, ast.ImportFrom) and node.module == "unittest":
                found["unittest"].append(f"{path.name}:{node.lineno}")
            elif isinstance(node, ast.Attribute) and node.attr == "xfail":
                found["xfail"].append(f"{path.name}:{node.lineno}")

    assert not found["unittest"], (
        f"a unittest import appeared in tests/: {found['unittest']}. "
        f"unittest.skip does produce a skipped report, so the audit would catch "
        f"it -- but it attributes the skip to the method's line in a class, a key "
        f"shape no SKIP_SITES entry uses. Use pytest's own marks."
    )
    assert not found["xfail"], (
        f"an xfail appeared in tests/: {found['xfail']}. An expected failure that "
        f"unexpectedly passes reports `passed` and reports nothing to this audit, "
        f"so a permanently-xfailing test is indistinguishable from a working one."
    )


def test_this_suite_contains_no_early_return_in_a_test():
    """The report-level gap above has no instance here, and that is worth asserting.

    A `return` in the middle of a test is the same shape as a swallowed skip: the
    function exits, pytest reports success, and the assertions after it never
    ran. Nothing in the audit could see it -- so the check that this suite has
    none has to be a source scan, and if it ever fires the honest response is to
    look at whether the remaining assertions are unconditional.
    """
    # Driven against a synthetic tree first, because a scan that cannot see an
    # early return would report this suite clean for the wrong reason -- which is
    # the defect class this whole file is about.
    synthetic = ast.parse(
        "def test_returns():\n    return\n\n"
        "def test_only_helper_returns():\n    def helper():\n        return\n    assert True\n"
    )
    assert [n.name for n in synthetic.body if _early_returns(n)] == ["test_returns"], (
        "the early-return scan must see a test's own return and must not be "
        "fooled by a nested helper's; if this fails the scan is not evidence"
    )

    offenders: list[str] = []
    for path in sorted(Path(__file__).resolve().parent.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                offenders += [
                    f"{path.name}:{lineno} in {node.name}" for lineno in _early_returns(node)
                ]
    assert not offenders, (
        f"these tests return before reaching their assertions, so they report "
        f"success without having tested: {offenders}"
    )


def _early_returns(test: ast.FunctionDef) -> list[int]:
    """Linenos of `return` statements belonging to this test function itself.

    A `return` inside a function *nested in* the test is that function's, not the
    test's: a helper returning early says nothing about whether the test reached
    its assertions. `ast.walk` cannot prune, so the nested functions' own returns
    are subtracted by lineno -- sound because a `lineno` identifies one node in
    one file.
    """
    nested = {
        sub.lineno
        for inner in ast.walk(test)
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)) and inner is not test
        for sub in ast.walk(inner)
        if isinstance(sub, ast.Return) and sub.value is None
    }
    return [
        sub.lineno
        for sub in ast.walk(test)
        if isinstance(sub, ast.Return) and sub.value is None and sub.lineno not in nested
    ]


def test_the_helper_never_asks_git_for_anything_remote():
    """`in_index` and `in_head` run local plumbing commands only.

    Stated because the suite's rule is that nothing under `tests/` touches the
    network, and a subprocess is the one thing here that could: `git ls-files` and
    `git ls-tree` read the local repository and contact nothing. If a future
    change needed `git fetch` or a `git show origin/main`, this is where it would
    be caught.
    """
    source = Path(tracked_artifacts.__file__).read_text(encoding="utf-8")
    assert '"git", "ls-files", "-z", "--", rel' in source
    assert '"ls-tree", "-r", "-z", "--name-only", "HEAD", "--", rel' in source
    assert '"HEAD"' in source and "origin/main" not in source
    for remote in ("fetch", "pull", "clone", "remote"):
        assert f'"{remote}"' not in source, f"tracked_artifacts must not run git {remote}"

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
import subprocess
from pathlib import Path

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

    The scanned set is chosen by content rather than by a hand-kept list of
    filenames: a module is scanned if it mentions `PUBLIC_SNAPSHOT_PATH` or
    `SCANNED`, so a fourth caller does not need this test updated.
    """
    tests_dir = Path(__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name == "test_tracked_artifacts.py":
            continue  # this file names the idiom in order to forbid it
        source = path.read_text(encoding="utf-8")
        if not ("PUBLIC_SNAPSHOT_PATH" in source or "SCANNED" in source):
            continue
        tree = ast.parse(source, filename=str(path))
        if any(
            isinstance(node, ast.ImportFrom) and node.module == "tracked_artifacts"
            for node in ast.walk(tree)
        ):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_pytest_skip(node.func):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"a test reads a committed artefact and skips without asking whether it "
        f"is committed: {offenders}. Import `tracked_artifacts.require` instead "
        f"-- see tests/tracked_artifacts.py for why a committed artefact that "
        f"went missing is a failure and not a skip."
    )


def _is_pytest_skip(func: ast.expr) -> bool:
    if not isinstance(func, ast.Attribute) or func.attr != "skip":
        return False
    return isinstance(func.value, ast.Name) and func.value.id == "pytest"


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

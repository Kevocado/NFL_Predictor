"""Distinguish "this checkout has no copy" from "a tracked file went missing".

Several tests in this suite read a file the repository commits, and they used to
answer "the file is not here" with `pytest.skip`. That is one answer standing in
for two very different situations, and the two differ in how much they should
cost:

* **The environment genuinely lacks the artefact.** An uncommitted local
  artefact, or a path that is legitimately not in the repository because it is
  published by some other means. Nothing is wrong, there is nothing to read, and
  a skip whose reason names *which* file and *why* its absence is expected is
  the correct outcome.
* **A tracked artefact has gone missing.** Someone deleted
  `data/public_snapshot.json`, ran `git rm --cached` on it, or landed a checkout
  half-applied. That is a defect in the tree, and a skip reports it as a
  non-event: the suite goes green having checked nothing, and the green is
  believed. This repo has been bitten by the general shape of that already --
  `tests/mutation_check.py`'s copy manifest skipped any directory that was not
  present, so `scripts/` silently stopped being copied and every mutation verdict
  from that arm came back INCONCLUSIVE (PR #33).

`data/public_snapshot.json` is why this matters *now*. Since PR #29 it is
load-bearing: `public_snapshot.assert_publishable` refuses to publish a snapshot
whose `generated_at` is older than the manifest's `trained_at`, and two tests
assert properties of that exact file. A test that skips when it is absent removes
the guard precisely where it matters most -- and two *other* tests in this suite
read the same artefact and do not skip at all
(`test_snapshot_staleness_gate.py::test_the_committed_snapshot_is_not_older_than_the_committed_models`
and `test_snapshot_shape_reconciliation.py::_shipped`), so before this module a
deletion produced a green skip and a loud failure in the same run, depending only
on which file you read.

**How "tracked" is decided: `git ls-files` and `git ls-tree`, both required to
agree the artefact is not committed.** Not `os.path.exists`, and not a
hand-maintained manifest -- for the same reason PR #33 stopped trusting
`COPY_DIRS`: a list somebody has to remember to update is a list that will not
be.

Two queries, because one is not enough and the gap between them is a hole:

* `git ls-files -- <path>` reads the **index**: the set of files this checkout is
  supposed to have on disk. Move the artefact aside and this still lists it, so
  the file is reported missing -- a failure. Good.
* `git ls-tree HEAD -- <path>` reads the **commit**: the set of files the
  repository actually carries. This is what catches `git rm --cached`, which
  removes a path from the index *and* from disk while leaving it in every commit.
  Index-only would answer "not tracked", the artefact would be reported as
  legitimately absent, and the one operation that can quietly untrack a
  load-bearing artefact would be the one that disabled its own guard.

So the artefact must be shown to be in **neither** the index **nor** HEAD before
a skip is allowed. Both queries are run only when the file is missing -- a file
that is present needs no interrogation, which is the common case and the only
one that runs in CI.

Nothing is cached. A cache would let a file be deleted mid-session and keep
reporting the answer from before, and these are three subprocess calls in a run
of 800 tests.

Outcomes, and only one of them is allowed to skip:

| artefact on disk | in the index | in HEAD | outcome |
| --- | --- | --- | --- |
| yes | – | – | return it |
| no | yes | – | **fail** -- this checkout is missing a file it has |
| no | no | yes | **fail** -- committed, and untracked locally; commit the removal |
| no | no | no | skip, naming the path and why its absence is expected |
| no | unknown | no | **fail** -- could not establish, so do not skip |
| no | no | unknown | **fail** -- likewise |

That last pair is a deliberate choice against the obvious one. When the check
cannot be performed, the tempting move is to assume the benign case and skip. But
"the artefact might be missing and I could not find out" and "the artefact is not
in this repository" are different states, and collapsing them is the hole this
module exists to close, one indirection further from the reader. A suite that
cannot confirm its own preconditions should say so. There is deliberately no
environment variable that switches the check off: an escape hatch that disables
the guard is indistinguishable from the guard being fixed.

**`rel` is verified against the path, not trusted.** A typo in the repo-relative
spelling is the one input that could reintroduce the original bug silently:
`require()` recomputes the path's location relative to `repo_root` and refuses if
it does not match what it was handed, so a misspelt `rel` fails instead of asking
git about a file nobody has heard of and getting "not tracked" back.

Used by `tests/test_public_snapshot_json.py`, `tests/test_player_props_failure.py`
and `tests/test_passing_td_record_absence.py`. `tests/test_tracked_artifacts.py`
tests this module against real git repositories built in `tmp_path`, because a
helper that decides whether other tests may skip should not be trusted on the
strength of its own docstring.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

#: The repository root, as every test module here resolves it.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: What `git` last said it could not do, and for which path. Appended to by the
#: queries below and read by `require()`'s refusal message, so the refusal can
#: name the actual error rather than only saying "git could not be asked".
_LAST_GIT_FAILURE: list[str] = []


class MissingTrackedArtifact(AssertionError):
    """A required artefact is absent, or its committed-ness could not be established.

    Subclasses `AssertionError` so an unhandled one reads as a failure rather
    than as an error -- the same reasoning as
    `tests.conftest.NetworkAccessInTests`, and for the same reason: a test that
    cannot establish its own premise must never be something a runner can be
    lenient about.
    """


def in_index(rel: str, *, repo_root: Path = REPO_ROOT) -> bool | None:
    """Does `git ls-files` list `rel`? `None` when git cannot answer.

    The index is this checkout's own inventory: what should be on disk right now.
    `rel` is repo-relative with forward slashes, exactly as git and the index
    spell it. The `--` terminator and the exact filename both matter -- git
    pathspecs match by directory prefix, so `data/snapshot` would also match
    `data/snapshot.json`, and a spelling starting with `-` would be read as an
    option rather than as a path.

    `None` rather than `False` for "cannot tell" is the point of the return type:
    collapsing the two would let "git is not installed here" reach the skip
    branch and skip a check that was never performed.
    """
    return _git_says(
        ["git", "ls-files", "-z", "--", rel], rel, repo_root, "ls-files"
    )


def in_head(rel: str, *, repo_root: Path = REPO_ROOT) -> bool | None:
    """Does `git ls-tree HEAD` carry `rel`? `None` when git cannot answer.

    HEAD rather than a branch name on purpose: the question is what the
    repository *commits*, and this is the tree of the commit that is checked out,
    so the answer does not change when a local branch is created, reset or
    renamed underneath the run.

    A repository with no commits at all answers `False` rather than `None`. "There
    is no HEAD" is a fact about the repository rather than a git malfunction, and
    `git rev-parse --verify --quiet HEAD` is how that is established without
    pattern-matching git's error text.
    """
    listed = _git_says(
        ["git", "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", rel],
        rel, repo_root, "ls-tree",
    )
    if listed is not None:
        return listed
    if _has_no_commits(repo_root):
        return False
    return None


def _has_no_commits(repo_root: Path) -> bool:
    """Is there no commit to ask about? Distinguished from git being unusable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=False,
        )
    except (OSError, ValueError) as exc:  # no git on PATH, or cwd does not exist
        return False
    return proc.returncode == 1  # 1 = unresolvable, i.e. unborn HEAD


def _git_says(command: list[str], rel: str, repo_root: Path, what: str) -> bool | None:
    try:
        proc = subprocess.run(
            command, cwd=repo_root, capture_output=True, text=True, check=False
        )
    except (OSError, ValueError) as exc:  # no git on PATH, or cwd does not exist
        _LAST_GIT_FAILURE.append(f"git {what} {rel!r}: {exc!r}")
        return None
    if proc.returncode != 0:
        _LAST_GIT_FAILURE.append(
            f"git {what} {rel!r}: rc={proc.returncode} {proc.stderr.strip()}"
        )
        return None
    return any(entry for entry in proc.stdout.split("\0") if entry)


#: `require()`'s verdicts. Named so the table in the module docstring is written
#: in the same vocabulary the code uses.
TRACKED = "tracked"
UNTRACKED = "untracked"
UNVERIFIABLE = "unverifiable"


def committed_state(rel: str, *, repo_root: Path = REPO_ROOT) -> str:
    """Is `rel` committed? One of `TRACKED`, `UNTRACKED`, `UNVERIFIABLE`.

    `UNTRACKED` -- and only `UNTRACKED` -- means both queries positively reported
    the path absent: not in the index, and not in any commit. It is the sole
    condition under which `require()` will skip, which is what makes "a skip
    fired for the wrong cause" a state the code cannot represent.
    """
    listed = in_index(rel, repo_root=repo_root)
    if listed is True:
        return TRACKED
    committed = in_head(rel, repo_root=repo_root)
    if committed is True:
        # Committed but not in the index. `git rm --cached` does exactly this, and
        # so does a partially-applied patch. The commit is the claim; the index is
        # a local opinion about it, and the local opinion is what is wrong.
        return TRACKED
    if listed is False and committed is False:
        return UNTRACKED
    return UNVERIFIABLE


def require(
    path: Path,
    *,
    rel: str,
    required_because: str,
    expected_absent_because: str,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Return `path` if a test may read it. Fail or skip if it may not.

    `path` is the absolute location the test is about to read; `rel` is the same
    file spelled the way git spells it, and the two are checked against each other
    (see the module docstring -- a wrong `rel` is the one input that could bring
    the original bug back).

    `required_because` is what the artefact is for, quoted into the failure
    message: a reader who has just been told their checkout is broken should not
    then have to go and find out what the file was. `expected_absent_because` is
    the skip's reason, and is only ever read on the `UNTRACKED` path.

    Raises `MissingTrackedArtifact` when the artefact is committed and absent, and
    when that cannot be established either way. Calls `pytest.skip` -- which
    raises -- when it is absent *and* uncommitted. There is no path through this
    function that returns a location the caller cannot read.
    """
    derived = _repo_relative(path, repo_root)
    if derived is None:
        raise MissingTrackedArtifact(
            f"{rel} was handed to require() as {path}, which is not inside "
            f"{repo_root}, so there is no repo-relative spelling to ask git "
            f"about. A test cannot decide whether a committed artefact is "
            f"missing if it cannot say which artefact it means."
        )
    if derived != rel:
        raise MissingTrackedArtifact(
            f"require() was told this path is {rel!r} but it is at {derived!r} "
            f"({path}). Refusing, because the repo-relative spelling is what is "
            f"checked against git: a rel naming some other file is answered "
            f"'untracked', which turns a committed artefact back into a skip."
        )
    if path.exists():
        return path

    state = committed_state(rel, repo_root=repo_root)
    if state is UNVERIFIABLE:
        detail = _LAST_GIT_FAILURE[-1] if _LAST_GIT_FAILURE else "no detail recorded"
        raise MissingTrackedArtifact(
            f"cannot establish whether {rel} is committed ({detail}), and it is "
            f"not on disk at {path}. Not skipping: 'missing' and 'not in this "
            f"checkout' are different states, and guessing the benign one is how "
            f"this check was skipped in the first place. {required_because} "
            f"Run the suite from inside the repository, with git available."
        )
    if state is TRACKED:
        raise MissingTrackedArtifact(
            f"{rel} is committed to this repository but is not on disk at "
            f"{path}. A test that reads a committed artefact cannot tell a "
            f"deletion from an environment that never had one, and reporting "
            f"the first as the second is how a guard goes missing while the "
            f"suite stays green. {required_because} Restore it "
            f"(`git checkout -- {rel}`). If the artefact is genuinely meant to "
            f"go away, that has to be a commit this suite can see: `git rm` the "
            f"path, commit, and delete the reading test in the same change -- "
            f"not `git rm --cached`, which leaves every commit carrying a file "
            f"the suite no longer insists on."
        )
    pytest.skip(
        f"{rel} is not on disk, and neither the index nor any commit in this "
        f"repository carries it, so this checkout has no copy to read. "
        f"{expected_absent_because}"
    )


def _repo_relative(path: Path, repo_root: Path) -> str | None:
    """`path` relative to `repo_root`, forward-slashed, or None if it is outside."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None

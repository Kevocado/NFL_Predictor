"""The mutation harness is under test too, because it stopped working silently.

`tests/mutation_check.py` is not a test module. Its filename does not match
pytest's default `python_files = test_*.py`, so `pytest tests/` collected zero
tests from it and nothing in the normal suite could tell that the harness had
stopped being able to run. It had: `scripts/` was not in `COPY_DIRS`, so every
temp copy it built was missing `scripts/null_fabricated_market_hits.py`, which
`tests/test_null_fabricated_market_hits.py` loads at MODULE scope with
`spec_from_file_location`. In the copy that raised

    FileNotFoundError: .../scripts/null_fabricated_market_hits.py

while pytest was *collecting*, so the run reported zero passes and zero
failures, `run_python` scored it INCONCLUSIVE, and `main` printed
"BASELINE IS RED -- aborting" and exited 2 before judging a single one of the
32 mutations. From outside the harness that is indistinguishable from "the tests
are broken", which is the worst possible report: true, and useless.

So the harness has a floor now, and this file is that floor. Every test here is
about the harness refusing to run on a tree it cannot vouch for, because the
failure mode being guarded against is a run that looks green (or merely red for
the wrong reason) instead of one that stops.

Nothing here opens a socket. The one subprocess runs `pytest --collect-only` in a
temp copy with the copy's own `src` on `PYTHONPATH`.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest

import mutation_check
from mutation_check import (
    COPY_DIRS,
    COPY_FILES,
    ROOT,
    _check_the_copy_is_whole,
    _cleanup,
    _module_level_repo_paths,
    _suite_modules,
    build_tree,
)

MIGRATION = Path("scripts") / "null_fabricated_market_hits.py"


@pytest.fixture
def copied_tree():
    """A real copy of the tree, torn down whatever happens."""
    tree = build_tree()
    try:
        yield tree
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def test_the_copy_carries_the_script_the_suite_loads_at_import_time(copied_tree):
    """The regression, stated as a fact instead of described in a comment."""
    assert (copied_tree / MIGRATION).is_file(), (
        f"{MIGRATION} is not in the copy; tests/test_null_fabricated_market_hits.py "
        "loads it during collection, so the copied run aborts with FileNotFoundError "
        "and every mutation verdict from it is INCONCLUSIVE"
    )


def test_the_copy_is_a_repository_in_which_the_artefacts_are_committed(copied_tree):
    """The third way this copy stopped being the repo, found by putting it in CI.

    `tests/tracked_artifacts.py` asks git about `data/public_snapshot.json`,
    about the three scan roots, and about `models/manifest.json`. `committed_state`
    answers UNVERIFIABLE -- not UNTRACKED, which is the only answer allowed to
    skip -- for every one of them when there is no repository to ask. So three of
    that module's assertions failed in the copy, `main()` printed "BASELINE IS RED
    -- aborting", and the harness exited 2 before judging a single mutation.

    Identical in shape to the missing `scripts/` above and it hid for the same
    reason this file exists: the harness is not named `test_*.py`, so
    `pytest tests/` collected zero tests from it and every push reported green.
    A harness that has never been executed cannot report that it does not start.

    Asserted against `committed_state` rather than against the three failing
    tests, because that is the *cause* and it is one git call instead of an 8s
    subprocess suite -- and those three assertions all reduce to this call.
    """
    from tracked_artifacts import TRACKED, committed_state

    for rel in ("data/public_snapshot.json", "src", "models/manifest.json"):
        assert committed_state(rel, repo_root=copied_tree) is TRACKED, (
            f"{rel} is not committed in the copy, so committed_state() answers "
            "'unverifiable' and three baseline assertions in "
            "tests/test_tracked_artifacts.py fail against it. The harness then "
            "aborts at the baseline and judges nothing, reporting a broken suite "
            "rather than a missing repository."
        )


@pytest.mark.parametrize("rel", [*COPY_DIRS, *COPY_FILES])
def test_the_copy_carries_every_declared_path(copied_tree, rel):
    assert (copied_tree / rel).exists(), f"{rel} is declared for copying but is not in the copy"


def test_the_copy_list_is_not_the_only_thing_that_decides_what_is_copied(copied_tree):
    """The list was the whole mechanism, and a hand-written list rots.

    The copy is built from an explicit manifest, so the manifest is the contract
    and it was quietly wrong: the fifth entry it ever needed was `scripts/`, and
    nobody added it. The harness still "worked" -- it just built a tree that was
    not the repo, which is strictly worse than crashing, because a baseline that
    aborts for a missing directory gets reported as a broken suite.

    This is the floor that catches the next one.
    """
    modules = _suite_modules(copied_tree)
    assert modules, "the scanner found no test modules; it is checking nothing"
    # Every test module has to be scanned, or the check is theatre. conftest.py
    # is collected too, but it is not matched by the `test_*.py` glob.
    on_disk = {p.name for p in (copied_tree / "tests").glob("test_*.py")}
    scanned = {m.name for m in modules} - {"conftest.py"}
    assert on_disk == scanned, (
        "the scanner and the directory disagree, so some module is unchecked: "
        f"{sorted(on_disk ^ scanned)}"
    )
    _check_the_copy_is_whole(copied_tree)   # raises, naming every offender


def test_the_scanner_really_finds_the_import_time_paths(copied_tree):
    """Guard the guard: a scanner that silently finds nothing passes vacuously."""
    found = {
        Path(*components).as_posix(): f"{m.name}:{name}"
        for m in _suite_modules(copied_tree)
        for name, components in _module_level_repo_paths(m)
    }
    assert MIGRATION.as_posix() in found, (
        "the scanner no longer sees tests/test_null_fabricated_market_hits.py's "
        f"SCRIPT constant; it found {sorted(found)}"
    )
    # A hit from a second module too, so the first cannot be a coincidence.
    assert found.get("src") == "test_runtime_dependencies.py:SRC", f"it found only {sorted(found)}"


def test_the_copy_check_names_every_missing_path_not_just_the_first(copied_tree):
    """One missing file arrives with more. The message has to carry all of them."""
    (copied_tree / MIGRATION).unlink()
    (copied_tree / ".github" / "workflows" / "deploy.yml").unlink()
    with pytest.raises(AssertionError) as excinfo:
        _check_the_copy_is_whole(copied_tree)
    message = str(excinfo.value)
    assert MIGRATION.name in message, message
    assert "deploy.yml" in message, message


@pytest.mark.parametrize("rel", [*COPY_DIRS, *COPY_FILES])
def test_a_required_path_missing_from_the_repo_is_refused_not_skipped(tmp_path, monkeypatch, rel):
    """No silent skip: a path that is gone must stop the run, not shrink the copy.

    `build_tree` used to say `if src.exists():`. Deleting the path produced a
    copy without it, the copied suite died during collection, and the harness
    said "BASELINE IS RED" -- a true statement about the wrong cause, with no way
    for a reader to tell a missing directory from a broken test. Nothing is
    skipped and nothing is degraded now: it refuses, and it says which path.
    """
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    for keep in (*COPY_DIRS, *COPY_FILES):
        if keep == rel:
            continue
        src, dst = ROOT / keep, fake_repo / keep
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    monkeypatch.setattr(mutation_check, "ROOT", fake_repo)
    with pytest.raises(FileNotFoundError) as excinfo:
        build_tree()
    assert rel in str(excinfo.value), (
        f"build_tree refused to copy a tree without {rel} but the message does not "
        f"name it: {excinfo.value}"
    )
    _cleanup()   # build_tree may already have made a temp dir before refusing


def test_the_copied_suite_collects_clean(copied_tree):
    """End to end, and the reason all of the above matters.

    This fails exactly the way the harness failed: pytest collecting `tests/`
    inside an unmutated copy must collect every module with no error. A non-zero
    exit here is precisely what turned all 32 verdicts into INCONCLUSIVE.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--collect-only", "tests"],
        cwd=copied_tree, capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONPATH": str(copied_tree / "src")},
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"collecting the copied suite failed (exit {proc.returncode}), so the copy "
        f"is not the repo:\n{out[-3000:]}"
    )
    # Not a substring search for "error": the suite has tests named
    # `test_an_empty_frame_is_no_flags_not_an_error`, so that would key off prose.
    # These are the shapes pytest prints when collection dies.
    problem = re.search(r"^ERROR collecting|^Interrupted:|^\S*\d+ errors? in", out, re.M)
    assert problem is None, (
        "the copied suite reported a collection problem "
        f"({problem.group(0) if problem else ''}):\n{out[-3000:]}"
    )
    collected = re.search(r"^(\d+) tests collected", out, re.M)
    assert collected and int(collected.group(1)) > 700, (
        f"the copy collected suspiciously few tests: {out[-500:]}"
    )


def test_the_harness_is_reachable_from_the_suite_at_all(request):
    """Why this file exists.

    `tests/mutation_check.py` is not named `test_*.py`, so `pytest tests/` does
    not collect it. That is fine -- it is a harness, not tests -- but it means
    every check above has to live in a module pytest *does* collect, or it is
    documentation.

    This asks pytest what it actually collects rather than assuming, because the
    earlier version compared `"tests/mutation_check.py"` -- a *path*, with a
    separator in it -- against a set of `test_*` *basenames*. Those can never be
    equal, so the assertion was incapable of failing and the reason this file
    exists was guarded by nothing. Two things can make the harness collectible and
    both are checked here: renaming it to a `test_*` name, and widening
    `python_files`. If either ever has to happen, delete this assertion on
    purpose rather than let it rot.
    """
    harness = Path(__file__).parent / "mutation_check.py"
    patterns = request.config.getini("python_files")
    collectible = sorted(
        p.name
        for p in Path(__file__).parent.iterdir()
        if any(fnmatch(p.name, pattern) for pattern in patterns)
    )
    assert harness.name not in collectible, (
        f"the harness is now collectible -- python_files ({patterns}) matches "
        f"{harness.name}. Either fold tests/test_mutation_check.py into it or take "
        "this assertion out on purpose; do not leave it asserting a condition "
        "that cannot be false."
    )
    assert harness.is_file()


#: The shapes the resolver has to see through. A module-level constant names the
#: *directory* and the next one reads the file inside it, at module scope, where
#: its absence aborts collection. The third reads two repo files in one call --
#: keeping only the first is the same hole as keeping none.
_WRAPPED_READ = '''\
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MODELS_DIR = REPO_ROOT / "models"
COMMITTED_MANIFEST = json.loads((COMMITTED_MODELS_DIR / "manifest.json").read_text())
BOTH_AT_ONCE = dict(
    manifest=COMMITTED_MODELS_DIR / "manifest.json",
    workflow=REPO_ROOT / ".github" / "workflows" / "deploy.yml",
)
'''


def test_the_resolver_reaches_the_file_a_module_reads_through_a_call(tmp_path):
    """A directory the copy happens to have is not the file the module reads.

    `tests/test_fitted_vs_served_columns.py` reads `models/manifest.json` at module
    scope, via a module-level constant. `_module_level_repo_paths` stopped at the
    constant above it and reported only `models`, so `_check_the_copy_is_whole`
    asked whether `models/` existed -- which `COPY_DIRS` already guarantees -- and
    never asked about the file whose absence raises `FileNotFoundError`
    mid-collection. Every verdict from such a copy is INCONCLUSIVE, which is
    precisely the failure this harness exists to make loud.

    The second half is the same defect one level along: a call that reads *two*
    repo files used to report the first and discard the second, which is a quiet
    hole exactly where the loud one used to be.
    """
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests" / "test_probe.py").write_text(_WRAPPED_READ, encoding="utf-8")

    rows = _module_level_repo_paths(tree / "tests" / "test_probe.py")
    found = dict(rows)
    assert found["COMMITTED_MODELS_DIR"] == ("models",), rows
    assert found["COMMITTED_MANIFEST"] == ("models", "manifest.json"), (
        f"the resolver stopped at the directory: {rows}. The read is the file, so "
        "the file is what has to be in the copy."
    )
    assert [c for name, c in rows if name == "BOTH_AT_ONCE"] == [
        ("models", "manifest.json"),
        (".github", "workflows", "deploy.yml"),
    ], f"a call reading two repo files reported only one: {rows}"

    # And the check bites on both: the directories present, the files gone.
    (tree / "models").mkdir()
    (tree / ".github" / "workflows").mkdir(parents=True)
    with pytest.raises(AssertionError) as excinfo:
        _check_the_copy_is_whole(tree)
    message = str(excinfo.value)
    assert "manifest.json" in message, message
    assert "deploy.yml" in message, message

    (tree / "models" / "manifest.json").write_text("{}", encoding="utf-8")
    (tree / ".github" / "workflows" / "deploy.yml").write_text("{}", encoding="utf-8")
    _check_the_copy_is_whole(tree)  # still silent once the files are there

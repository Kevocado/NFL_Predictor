"""tests/test_passing_td_record_absence.py -- `tracking/qb_passing_td_record.py`
was deleted, and this is what makes putting it back a decision.

Why it was deleted
------------------
`qb_passing_td_record.py` computed a QB passing-TD track record. Nothing wrote
the rows it read and nothing read what it wrote:

* **No writer.** `routes.background_tracking_tick` is the only caller of
  `store.record_player_prop_predictions` in `src/`. It builds a row for
  `market="anytime_td"` and for every market in `player_props.POSITION_MARKETS`
  (`passing_yards`, `rushing_yards`, `carries`, `receiving_yards`). `"passing_tds"`
  is in neither, so no `passing_tds` row has ever been written and the `line`,
  `line_source`, `side`, `mu` and `call_prob` columns are all NULL on the live
  database. The reader's own SQL filtered on `market = 'passing_tds' AND
  resolved = 1`, so it read an empty frame from a table that has never held one.
* **No consumer.** `passing_td_track_record` had no caller in `src/`, in
  `scripts/`, or in `frontend/`; `read_resolved_passing_td_picks` had no caller
  outside the module. `get_track_record` reports `anytime_td` plus the five
  `_YARDAGE_MARKETS` and has no `passing_tds` section for it to land in, so even
  a written row would have been dropped from the served record. None of the
  eight `passing_td_*` payload keys `models/player_props.predict_props` flattens
  onto a props row is read by `frontend/` either -- `frontend/src/types.ts`'s
  `PlayerPropPrediction` carries `anytime_td_prob` and the yardage keys and stops
  there.
* **And it hid its own failure.** `passing_td_track_record` wrapped the read in a
  bare `except Exception` returning an empty record, justified by a comment
  claiming "the migration is additive and runs on connect". So if it had been
  wired up, a broken query would have published a clean, permanently empty record
  that looks exactly like "no picks yet" -- and nothing could tell the two apart.

That combination is worse than absent code: a record nothing writes, guarded so
that it can never report an error, reads on the page as a feature. The writer
that would fix it belongs in `api/routes.py`, which this change does not touch, so
deleting was the only option available here rather than the preferred one.

**A note on the wording, since it is load-bearing.** "Committed" and "tracked" are
used the same way throughout, and both mean "git has it in the index or in a
commit". `tests/tracked_artifacts.py` asks about both, because `git rm --cached`
removes a path from the index and from disk while leaving it in every commit --
so index-only would answer "not tracked" and skip, making the one command that can
quietly untrack a directory the command that disabled the guard.

What these tests pin
--------------------
That the module is gone, that no source file names it, and that the serving
path still writes no `passing_tds` row. All three are the facts the deletion
rests on, so if a future change wires the write path up, the first two fail and
that is the intended signal: reintroduction has to be a deliberate act, with a
consumer named, not a resurrection of dead code.

**The scan cannot come back empty.** `_source_files()` used to `continue` past a
scan root that was not on disk. That is the same defect as a skip that is not
justified, one syntax further out: with `scripts/` missing the file still parsed,
every `DELETED_SYMBOLS` parametrisation still passed having read
`src/` and `frontend/src/` only, and `test_no_product_code_or_consumer_imports_or_calls_the_deleted_report`
reported "nothing in the product references the deleted report" as a fact about
code that was never opened. PR #33 found the same hole in
`tests/mutation_check.py`'s copy manifest, where a missing `scripts/` made every
mutation verdict INCONCLUSIVE; both are now closed the same way. Each root goes
through `tracked_artifacts.require`, so a committed root that has gone missing is a
failure rather than a smaller scan, and
`test_the_scan_examines_a_non_zero_number_of_files` refuses the other version --
roots present, nothing to read.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tracked_artifacts import require

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every symbol the deleted module exported. All of them are named rather than
#: just the module, so a partial reintroduction -- someone rebuilding
#: `summarize_passing_td_calls` under a new filename -- is still caught, because
#: these are the names a resurrected report would have.
DELETED_SYMBOLS = (
    "qb_passing_td_record",
    "passing_td_track_record",
    "read_resolved_passing_td_picks",
    "summarize_passing_td_calls",
    "counted_passing_td_picks",
)

#: Directories that are product code or consumers. `tests/` is deliberately NOT
#: scanned: this file and the docstring of `tests/test_qb_passing_td.py` both
#: have to be able to say the name out loud in order to assert it is gone.
#:
#: Every entry is REQUIRED and is checked with `tracked_artifacts.require`, so a
#: root that has gone missing fails the tests rather than quietly reducing the
#: scan. `scripts/` is listed because the scan covers it, not because this file
#: needs it -- and that is why the missing-`scripts/` hole went unnoticed here
#: while `tests/mutation_check.py` (which copies it) was the thing that noticed.
SCANNED = ("src", "frontend/src", "scripts")

#: Why each root has to be there. Quoted into the failure message so a reader
#: told their checkout is incomplete learns what the directory was for.
WHY_ROOT_REQUIRED = (
    "This file's whole claim is that no product code or consumer reaches the "
    "deleted report, and that claim is only worth anything if every directory "
    "the report could have been reached from is actually read. `scripts/` was "
    "the one that vanished silently -- see tests/mutation_check.py, where the "
    "same absence made 32 mutation verdicts INCONCLUSIVE."
)

#: What would have to be true for a root's absence to be expected. All three are
#: committed at `main`, so none of this branch is reachable here and it says so
#: rather than inventing a plausible story for it.
WHY_ROOT_ABSENT_IS_EXPECTED = (
    "Nothing in this repository predicts it: this directory is committed at main. "
    "If you are reading this, your checkout is incomplete -- restore the "
    "directory rather than committing a test that tolerates its absence."
)

#: The file extensions the scan reads. Kept as a named constant because the
#: count floor below is expressed against it: a root can be present and still
#: contribute nothing, which is `frontend/src` if it ever held only `index.css`.
SOURCE_SUFFIXES = frozenset({".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".sh"})

#: How many files the scan must read. A floor, not the current count: the count
#: is 46 (src 38, frontend/src 7, scripts 1) and will change with ordinary work,
#: but "somewhere between one and a few hundred" is where a live scan lives and
#: anything far outside it means the scan stopped reading what it claims to.
MIN_SCANNED_FILES = 20


def _scan_root(rel: str) -> Path:
    return require(
        REPO_ROOT / rel,
        rel=rel,
        required_because=WHY_ROOT_REQUIRED,
        expected_absent_because=WHY_ROOT_ABSENT_IS_EXPECTED,
    )


def _source_files() -> list[Path]:
    """Every source file under `SCANNED`, and never nothing."""
    files = _files_by_root()
    return sorted(p for root_files in files.values() for p in root_files)


def _files_by_root() -> dict[str, list[Path]]:
    """`{scan root: the source files under it}`, refusing an empty one.

    A root that is missing raises (via `tracked_artifacts.require`) and a root
    that is present but yields no matching file raises too. Either way the caller
    cannot end up with an empty list, which is the only outcome this function is
    no longer allowed to have: two of the three tests that read it report
    "nothing references the deleted report", and that sentence must be about code
    they opened.
    """
    per_root: dict[str, list[Path]] = {}
    for rel in SCANNED:
        root = _scan_root(rel)
        found = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix in SOURCE_SUFFIXES
        )
        if not found:
            raise AssertionError(
                f"scan root {rel}/ is on disk but holds no file with a source "
                f"suffix in {sorted(SOURCE_SUFFIXES)}. Reading nothing and "
                f"reporting 'no reference found' is the failure this function "
                f"exists to prevent; if the directory has genuinely become "
                f"non-source, delete it from SCANNED and say why."
            )
        per_root[rel] = found
    return per_root


def _assert_the_scan_is_wide_enough(per_root: dict[str, list[Path]]) -> int:
    """Every root contributed, and the total clears the floor. Returns the total.

    Extracted from `test_the_scan_examines_a_non_zero_number_of_files` so its two
    conditions can be handed a per-root map directly. Inline, both are satisfied
    trivially by the real scan -- three roots, 46 files, every floor cleared -- so
    neither could ever go red, and an assertion that cannot fail is not a guard.

    The two conditions are separate on purpose. "Every root contributed" catches a
    root that went empty; the floor catches a scan that quietly narrowed to one of
    the three, which the first cannot see.
    """
    counts = {rel: len(files) for rel, files in per_root.items()}
    assert all(counts.values()), (
        f"these scan roots contributed no files to the scan: "
        f"{ {rel: n for rel, n in counts.items() if not n} }"
    )

    total = sum(counts.values())
    assert total >= MIN_SCANNED_FILES, (
        f"the scan read {total} source files, below the floor of "
        f"{MIN_SCANNED_FILES} (src 38, frontend/src 7, scripts 1 at the time "
        f"this was written). Either the repository has shrunk very fast or the "
        f"scan is no longer reading the roots it names -- and 'no reference "
        f"found' over a handful of files is not the same claim."
    )
    return total


def test_the_scan_examines_a_non_zero_number_of_files():
    """Named, so a scan that stopped reading says so in one line.

    `_files_by_root()` already refuses to return an empty list and already fails
    on a root that has gone missing. This is the third thing: a scan that reads
    *some* files. Five files, or forty thousand, both pass "did not return empty",
    and a suffix set that stopped matching the frontend -- or a `rglob` that
    started resolving somewhere else -- would be indistinguishable from a codebase
    that genuinely mentions nothing.

    The per-root counts and the whole scan are also reconciled against each other,
    because a test that sums a map it built itself would not notice if `_files_by_root`
    were handing back a different set than it claims to have read.
    """
    per_root = _files_by_root()
    total = _assert_the_scan_is_wide_enough(per_root)
    assert total == len(_source_files()), (
        f"the per-root counts sum to {total} but the scan reads "
        f"{len(_source_files())} files; two views of the same scan disagree, and "
        f"one of them is lying about what was read"
    )


def test_a_root_present_but_holding_no_source_file_is_refused(monkeypatch):
    """The root-exists-but-contributes-nothing half, which the floor does not cover.

    `data/` is a committed directory with no `.py`, `.ts`, `.tsx` or `.js` file in
    it at all, so pointing `SCANNED` at it produces a root that passes every
    existence check and yields nothing. The floor would catch that too -- one file
    is below twenty -- but it would catch it for the wrong reason, and a root
    yielding two files would clear the floor while having stopped matching the
    frontend entirely. So the refusal names the suffix set, and this proves the
    refusal is reachable rather than a branch no test enters.
    """
    monkeypatch.setattr(sys.modules[__name__], "SCANNED", ("data",))
    with pytest.raises(AssertionError, match="holds no file with a source suffix"):
        _files_by_root()


def test_the_floor_rejects_a_scan_narrowed_to_one_root(monkeypatch):
    """The floor bites, and the case it catches is a real one.

    `scripts/` is a real committed root holding a real `.py`, so scanning only it
    satisfies every existence check and the empty-root refusal, and is stopped
    solely by `MIN_SCANNED_FILES`. Without the floor, "the scan read one file" and
    "the scan read the repository" would be the same claim -- which is the
    difference between checking three roots and checking whichever one survived.
    """
    monkeypatch.setattr(sys.modules[__name__], "SCANNED", ("scripts",))
    per_root = _files_by_root()  # one real file; must not raise the empty-root way
    assert sum(len(f) for f in per_root.values()) == 1
    with pytest.raises(AssertionError, match="below the floor"):
        _assert_the_scan_is_wide_enough(per_root)


def test_the_empty_root_condition_names_every_offending_root():
    """The message has to be able to say *which* root, or it is not diagnosable.

    A single-entry map makes this cheap: with three roots, a message that only
    counted them would pass while telling the reader nothing about which one.
    """
    with pytest.raises(AssertionError, match="frontend/src"):
        _assert_the_scan_is_wide_enough({"src": [Path("a.py")], "frontend/src": []})
    with pytest.raises(AssertionError, match="scripts"):
        _assert_the_scan_is_wide_enough({"src": [Path("a.py")], "scripts": []})


def test_the_deleted_module_cannot_be_imported():
    """Not a lint, an assertion about behaviour: `import` must fail.

    `from nfl_predictor.tracking import qb_passing_td_record` used to succeed. If
    this ever passes again the module is back, and the rest of this file is the
    list of things that would have to be true for that to be a good idea.
    """
    proc = subprocess.run(
        [sys.executable, "-c",
         "from nfl_predictor.tracking import qb_passing_td_record"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert proc.returncode != 0, (
        "nfl_predictor.tracking.qb_passing_td_record is importable again; "
        "reintroducing it is allowed but must be deliberate -- see this file's "
        "docstring for what has to be true first"
    )


def test_the_deleted_module_file_is_not_on_disk():
    assert not (REPO_ROOT / "src/nfl_predictor/tracking/qb_passing_td_record.py").exists()


@pytest.mark.parametrize("symbol", DELETED_SYMBOLS)
def test_no_product_code_or_consumer_imports_or_calls_the_deleted_report(symbol):
    """The evidence that it had no consumer, kept as an executable fact.

    Python is parsed rather than grepped, so this is about CODE -- imports, attribute
    access, bare names -- and not about prose. A comment naming a deleted module cannot
    break anything, and one of them survives on purpose (see
    `test_the_one_known_stale_comment_reference_is_named` below); what must not exist
    is a line that tries to import, call or reach the report.

    Non-Python files are filtered line by line, which is good enough: `frontend/` and
    `scripts/` contain no reference at all today, so the filter is only there to let a
    future explanatory comment through.

    This is the test the missing-scan-root hole made expensive. Its result is "no
    product code or consumer reaches the deleted report", and with
    `_source_files()` skipping an absent root that sentence was a claim about the
    roots it happened to find rather than about all three. It is now gated on
    `tracked_artifacts.require` and on
    `test_the_scan_examines_a_non_zero_number_of_files`.
    """
    offenders = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".py":
            offenders += [f"{path.relative_to(REPO_ROOT)}:{node.lineno}" for node in
                          _python_references(ast.parse(text), symbol)]
        else:
            code_lines = [line for line in text.splitlines()
                          if not line.lstrip().startswith(("#", "*", "/*", "//"))]
            if any(symbol in line for line in code_lines):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"{symbol} is imported, called or reached by product code or a consumer: "
        f"{offenders}. The passing-TD report was deleted because nothing wrote its rows "
        "and nothing read it; re-adding a reference means that has changed, and this "
        "test is where you say so."
    )


def _python_references(tree, symbol):
    """Every node in `tree` that is code referring to `symbol`.

    `ast` is used rather than a substring search so docstrings and comments -- which
    are `Constant` and discarded entirely -- cannot produce a false positive, and so
    `from ... import qb_passing_td_record` and `record.passing_td_track_record(conn)`
    are both caught.
    """
    short = symbol.rsplit(".", 1)[-1]
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [n for n in node.names if symbol in n.name or n.name.endswith(short)]
        elif isinstance(node, ast.ImportFrom):
            if symbol in (node.module or "") or any(
                symbol in n.name or n.name == short for n in node.names
            ):
                hits.append(node)
        elif isinstance(node, ast.Attribute) and node.attr == short:
            hits.append(node)
        elif isinstance(node, ast.Name) and node.id == short:
            hits.append(node)
    return hits


def test_every_prose_reference_is_a_comment_and_only_one_is_stale():
    """What the code test above deliberately tolerates, stated so none of it is lost.

    The deleted module's name still appears in prose in three places, in two kinds:

    * `tracking/store.py` -- written by this change, on `_PASSING_TD_PROP_COLUMNS` and
      `_MARKET_TO_STAT_COLUMN`, so the surviving columns are not mistaken for a live
      feature: "a writer in `api/routes.py` will need no migration, and the report that
      read them was deleted for having no consumer".
    * `models/qb_passing_td.py` -- **stale, and left stale on purpose.** It says "the
      grading assertion in `qb_passing_td_record`" refuses a whole-number line, and that
      module is gone. This change is scoped to `src/nfl_predictor/tracking/`; `models/`
      is off limits for it, so the sentence could not be corrected here.

    The first half asserts every hit is prose -- a `#` comment or a docstring -- and so
    cannot be code. The second asserts that outside `tracking/` exactly one file still
    names it, so a NEW stale reference cannot slip in unnoticed. Pinned rather than
    ignored, so a green run is not misread as "nothing mentions it any more".

    The arithmetic the stale comment describes is not lost: it is asserted in
    `tests/test_qb_passing_td.py::test_a_whole_number_line_would_be_a_push`, and the
    structural fix it refers to -- `model_line` never returning a whole number -- is
    unchanged.
    """
    prose_hits: dict[str, int] = {}
    for path in _source_files():
        if path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno in _prose_lines(text):
            if "qb_passing_td_record" in text.splitlines()[lineno - 1]:
                rel = str(path.relative_to(REPO_ROOT))
                prose_hits[rel] = prose_hits.get(rel, 0) + 1

    # Nothing may reference it from a docstring AND from code at the same line --
    # that is the AST test's job. This one only cares that the files which still name
    # it are doing so in prose.
    outside_tracking = {f: n for f, n in prose_hits.items()
                        if not f.startswith("src/nfl_predictor/tracking/")}
    stale_file = "src/nfl_predictor/models/qb_passing_td.py"
    # **ZERO references outside tracking/ is the goal, not a failure.** Someone fixing
    # the stale comment is the outcome this test wants, so it must not fail when they do:
    # an assertion keyed on the stale reference existing turns the fix into a red test
    # and teaches the next person to leave the lie in place. What IS rejected is a
    # reference in some *other* file, and more than one in the known stale file (a second
    # would be a new one hiding in the same place).
    assert outside_tracking.get(stale_file, 0) <= 1, (
        f"{stale_file} now names the deleted report {outside_tracking[stale_file]} times; "
        "more than one means a new reference was added alongside the stale one"
    )
    unexpected = {f: n for f, n in outside_tracking.items() if f != stale_file}
    assert not unexpected, (
        f"files outside tracking/ now name the deleted report: {unexpected}. A new one "
        "needs a look before it lands."
    )
    assert "src/nfl_predictor/tracking/store.py" in prose_hits, (
        "store.py no longer explains why the passing-TD columns and the grader route "
        "survive with no writer -- that explanation is the thing a future reader "
        "needs to avoid treating them as a live feature"
    )


def _prose_lines(source: str) -> set[int]:
    """Every 1-indexed line of `source` that is a `#` comment or part of a docstring.

    Derived from the parse rather than from a regex, so a name appearing inside a
    function's own docstring counts as prose and one appearing in real code does not.
    """
    lines = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body:
                first = node.body[0]
                lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    for i, line in enumerate(source.splitlines(), 1):
        if line.lstrip().startswith("#"):
            lines.add(i)
    return lines


def test_serving_still_writes_no_passing_td_row_and_the_record_has_no_slot_for_one():
    """The two halves of "there is no consumer", asserted against the live paths.

    Three checks, and the second one exists because of a hole a reviewer found in an
    earlier draft of this test. Reading `routes.py` for a `"passing_tds"` literal was
    not enough on its own: the snapshot loop iterates
    `player_props.POSITION_MARKETS`, so adding `"passing_tds"` to that dict would
    make serving write the rows with no `"passing_tds"` string anywhere in `routes.py`
    at all, and the text check would have gone on passing while the premise of the
    whole deletion quietly stopped being true. So the configured markets are asserted
    too, by value rather than by reading the source of the dict's owner.

    The third check is the other half: `get_track_record` reports `anytime_td` plus the
    five yardage markets, so even a written row had nowhere to be summarised. Between
    them: no writer, and no slot for a result.
    """
    from nfl_predictor.models import player_props

    configured = {m for markets in player_props.POSITION_MARKETS.values() for m in markets}
    assert "passing_tds" not in configured, (
        f"`POSITION_MARKETS` now offers {sorted(configured)}, so serving writes "
        "`passing_tds` rows through the position-market loop and the text check on "
        "routes.py below cannot see it. If that is deliberate the report has a writer "
        "and belongs in _prop_markets, not in a deleted module."
    )

    routes = (REPO_ROOT / "src/nfl_predictor/api/routes.py").read_text(encoding="utf-8")
    snapshot_block = routes[routes.index("prop_rows.append"):routes.index(
        "store.record_player_prop_predictions(prop_rows)")]
    assert '"passing_tds"' not in snapshot_block, (
        "serving now snapshots a passing_tds prop row -- if that is deliberate the "
        "report has a writer and belongs in _prop_markets, not in a deleted module"
    )

    store_src = (REPO_ROOT / "src/nfl_predictor/tracking/store.py").read_text(encoding="utf-8")
    assert '"passing_tds": "passing_tds",' in store_src, (
        "the grader's passing_tds stat-column route is gone; see the note on "
        "_MARKET_TO_STAT_COLUMN before removing it again"
    )
    reported = store_src[store_src.index("_YARDAGE_MARKETS = ("):store_src.index(")\n", store_src.index("_YARDAGE_MARKETS = ("))]
    assert "passing_tds" not in reported, (
        "the reported markets changed; re-check whether the passing-TD report has a consumer"
    )
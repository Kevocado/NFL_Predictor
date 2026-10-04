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

**And it has now been written, the way this file said it had to be.** The writer
is `routes._passing_td_prop_row`, called from `background_tracking_tick`, and the
report is `store._passing_td_metrics` -- inside `_prop_markets`, beside every
other market's record, with no bare `except` around it. The module itself stays
deleted and the symbols it exported stay unreferenced: the defect was never the
file, it was a report with no writer, and the fix is a writer, not a file. The
two tests at the bottom of this file are the deliberate signal this docstring
asked for, and they are written by CALLING the writer and the reader, because the
substring check they replaced went green the moment the market went live (see
`test_the_writer_and_the_record_slot_both_exist_now`).

**A note on the wording, since it is load-bearing.** "Committed" and "tracked" are
used the same way throughout, and both mean "git has it in the index or in a
commit". `tests/tracked_artifacts.py` asks about both, because `git rm --cached`
removes a path from the index and from disk while leaving it in every commit --
so index-only would answer "not tracked" and skip, making the one command that can
quietly untrack a directory the command that disabled the guard.

What these tests pin
--------------------
That the module is gone, that no source file names it, **and that the writer and
the record slot it lacked now exist.** The first two are still the facts the
deletion rests on; the third is the change that came after it, so the two are
pinned together rather than one having quietly stopped meaning anything.

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
    """Every source file under `SCANNED`, and never fewer than the floor.

    The floor is enforced HERE rather than only inside
    `test_the_scan_examines_a_non_zero_number_of_files`, because this is the
    function the two symbol scans actually read. It used to enforce nothing: a
    caller went straight to `_files_by_root()`, which only checks that each root
    contributed *something*, so `test_no_product_code_or_consumer_imports_or_calls_the_deleted_report`
    run on its own -- `pytest -k no_product_code`, a bisect, an IDE's run button --
    reported "nothing in the product references the deleted report" as a fact
    about seven files. The sibling test that guards the width was still green in a
    different process and said nothing about this one. A guard that only runs
    beside the thing it guards is not a guard.
    """
    files = _files_by_root()
    _assert_the_scan_is_wide_enough(files)
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


def test_the_floor_is_enforced_on_the_path_the_symbol_scan_actually_reads(monkeypatch):
    """The same floor, on the function the symbol scans call.

    `test_the_scan_examines_a_non_zero_number_of_files` checks the width, but
    `test_no_product_code_or_consumer_imports_or_calls_the_deleted_report` calls
    `_source_files()` directly and used to get no width check at all. Run on its
    own -- `pytest -k no_product_code`, a bisect, an IDE's run button -- it read
    whatever roots happened to survive and reported "nothing in the product
    references the deleted report" as a fact about them. The test that guards
    the width was green in a different process and said nothing about this one.

    `frontend/src` alone is the case: seven real files, so every per-root
    existence check and the empty-root refusal are satisfied, and only the floor
    stops it.
    """
    monkeypatch.setattr(sys.modules[__name__], "SCANNED", ("frontend/src",))
    with pytest.raises(AssertionError, match="below the floor"):
        _source_files()


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

    The deleted module's name now appears in prose in exactly one file:

    * `tracking/store.py` -- on `_PASSING_TD_PROP_COLUMNS` and
      `_MARKET_TO_STAT_COLUMN`, and it is the history rather than a caveat: these
      columns survived the era when nothing wrote the market so that wiring the
      writer up would need no migration, and they are filled in production now.

    `models/qb_passing_td.py` used to name it too, and **was stale** -- it pointed at a
    grading assertion in a module that no longer existed. That is corrected: the refusal
    it describes is `tracking/store.py::_line_is_half_point`, and the arithmetic is
    asserted in `tests/test_qb_passing_td.py::test_a_whole_number_line_would_be_a_push`
    and again on a stored row in `tests/test_passing_td_record.py`.

    The first half asserts every hit is prose -- a `#` comment or a docstring -- and so
    cannot be code. The second rejects a reference in some OTHER file, so a NEW stale
    reference cannot slip in unnoticed, and tolerates *fewer* than one in
    `models/qb_passing_td.py` rather than requiring one: the fix to a stale comment is
    the outcome this test wants, and a test keyed on the lie surviving would teach the
    next person to leave it in place.
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
        "store.py no longer records why the passing-TD columns and the grader route "
        "survived a stretch with no writer -- that history is the thing a future "
        "reader needs in order to trust that the market is real now"
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


def test_the_writer_and_the_record_slot_both_exist_now():
    """The reintroduction PR #31 asked for, carried out deliberately.

    **The module stays deleted.** The three tests above still say so, and nothing
    here resurrects it. What changed is the half that was missing: there is now a
    writer (`routes._passing_td_prop_row`, called from
    `background_tracking_tick`) and a slot for the result
    (`store._prop_markets()[PASSING_TD_MARKET]`). PR #31's docstring asked for
    "a consumer named" before the write path was wired up; the consumer is
    `_passing_td_metrics`, in the same module as every other prop market's record.

    **Every check here CALLS the code, and that is the reason this test is
    rewritten rather than deleted.** The version it replaces asserted the absence
    with a substring search for `"passing_tds"` over the snapshot block of
    `routes.py`, and it went GREEN on the change that made the market live --
    because the writer names the market through
    `player_props.qb_passing_td.PASSING_TD_MARKET` and never spells the string
    anywhere in that file. A text check on a name the code has stopped spelling is
    a check on the absence of a substring, not on the absence of a feature, and it
    would have left the market shipped, graded and recorded while this file
    insisted nothing wrote it.
    """
    import pandas as pd

    from nfl_predictor.api import routes
    from nfl_predictor.models import player_props
    from nfl_predictor.tracking import store

    # One market name, agreed on by the writer and the reader. If these two ever
    # diverged, every row would be written under a name the record does not
    # summarise -- a recorded pick in no figure, which is the failure this whole
    # market has already had once.
    assert store.PASSING_TD_MARKET == player_props.qb_passing_td.PASSING_TD_MARKET

    qb_prop = {
        "player_id": "00-1", "player_name": "Pat QB", "position": "QB",
        "passing_td_line": 2.5, "passing_td_line_source": "model_line",
        "passing_td_side": "over", "passing_td_mu": 2.3, "passing_td_prob": 0.64,
    }
    row = routes._passing_td_prop_row(qb_prop, "2026_03_A_B")
    assert row is not None and row["market"] == store.PASSING_TD_MARKET
    assert row["line"] == 2.5 and row["side"] == "over"

    # No call -> no row, and no fabricated zero. The `None` is what the tick turns
    # into a logged gap rather than a silent omission.
    assert routes._passing_td_prop_row(
        {**qb_prop, "passing_td_line": None}, "2026_03_A_B") is None
    assert routes._passing_td_prop_row(
        {**qb_prop, "position": "WR"}, "2026_03_A_B") is None

    # And there is somewhere for a written row to be summarised.
    assert store.PASSING_TD_MARKET in store._prop_markets(pd.DataFrame())


def test_the_writer_is_reached_from_the_production_tick_and_nowhere_else():
    """Reachability, from the parse.

    A writer that exists but is not called by the tick is the deleted-module defect
    wearing a new hat, and `test_the_writer_and_the_record_slot_both_exist_now`
    above would still pass on it: it calls `_passing_td_prop_row` itself. So this
    one walks the AST of `background_tracking_tick` and requires the call to be
    inside it -- and requires that the whole snapshot loop still ends at the one
    writer, so a second `record_player_prop_predictions` call cannot split the
    market off into a transaction that commits on its own.
    """
    import ast

    from nfl_predictor.api import routes

    tree = ast.parse(Path(routes.__file__).read_text(encoding="utf-8"))
    tick = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "background_tracking_tick")
    called = {_called_name(n) for n in ast.walk(tick) if isinstance(n, ast.Call)}
    assert "_passing_td_prop_row" in called, (
        "the passing-TD writer is not called from background_tracking_tick, so the "
        "market is served and reported but never recorded -- which is exactly what "
        "PR #31 deleted a report for"
    )
    assert "record_player_prop_predictions" in called, (
        "the tick no longer writes any prop rows; the whole prop record is gone"
    )
    assert sum(1 for line_number in _call_lines(tree, "record_player_prop_predictions")
               if _within(line_number, tick)) == 1, (
        "the tick writes prop rows from more than one place, so a failure part-way "
        "through could commit a partial record that reads as complete"
    )


def _called_name(call: ast.Call) -> str | None:
    """The name a call targets, whether it is `f()` or `obj.f()`.

    Both spellings occur in `routes.py`, and collecting only one of them makes the
    reachability test below report a missing call for code that plainly makes it.
    """
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _call_lines(tree: ast.AST, name: str) -> list[int]:
    """Every line calling `name(...)`, so a count can be scoped to one function."""
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _called_name(n) == name]


def _within(lineno: int, node: ast.AST) -> bool:
    return node.lineno <= lineno <= (node.end_lineno or node.lineno)
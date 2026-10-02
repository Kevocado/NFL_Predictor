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

What these tests pin
--------------------
That the module is gone, that no source file names it, and that the serving
path still writes no `passing_tds` row. All three are the facts the deletion
rests on, so if a future change wires the write path up, the first two fail and
that is the intended signal: reintroduction has to be a deliberate act, with a
consumer named, not a resurrection of dead code.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

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
SCANNED = ("src", "frontend/src", "scripts")


def _source_files() -> list[Path]:
    files: list[Path] = []
    for rel in SCANNED:
        root = REPO_ROOT / rel
        if not root.exists():
            continue
        files.extend(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".sh"}
        )
    return sorted(files)


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
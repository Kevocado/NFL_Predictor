"""Mutation harness for the player-props empty-state work.

For each mutation: apply it to a COPY of the tree, run the target against the
copy, and record whether the target caught it. A surviving mutation is a hole in
the tests, not a pass.

    .venv/bin/python tests/mutation_check.py            # all
    .venv/bin/python tests/mutation_check.py M4 M9      # named only
    .venv/bin/python tests/mutation_check.py --shard=0/4   # CI: this run's slice

Five things this harness is built around, each of which broke a previous
version of it:

**It never writes to a tracked file.** The first version applied mutations to
`src/nfl_predictor/api/routes.py` in place and restored from a temp backup in a
`finally`. `finally` does not run on SIGKILL or Ctrl-C, so an interrupted run
left `routes.py` mutated -- in a file whose own comment says that cannot happen.
Worse, two proof scripts in this project's history derived their restore paths by
string-munging and left every mutation in place. So the tree is copied to a
tempdir, `node_modules` and `.venv` are symlinked rather than copied, and the
tracked files are never opened for writing at all. `signal`/`atexit` handlers are
kept as a second line of defence for the tempdir itself.

**Exit codes are not verdicts.** The first version reported CAUGHT for any
non-zero exit, which cannot tell "an assertion fired" from "the runner crashed
or could not start". A missing `jsdom` -- the exact failure mode of a brand-new
dependency -- was reported as `all mutations caught`, exit 0. The python arm now
runs each target's BASELINE first and aborts if it is not green. The frontend arm
requires the literal `all frontend state assertions passed` before believing a
pass and a `FAIL` marker before believing a failure; anything else is
INCONCLUSIVE, which is a harness problem and is reported as one.

**The canary.** A mutation that is known to survive must report SURVIVED. If it
reports CAUGHT, the harness itself has stopped detecting and every other verdict
in the run is suspect, so that fails the whole harness. This is what makes a
broken harness loud instead of green. Every shard carries every canary for the
same reason: `all()` over an empty sequence is True, so a shard with none would
print `0 canary (all correct: True)`.

**The copy has to be a git repository.** `tests/tracked_artifacts.py` asks git
whether `data/public_snapshot.json`, the scan roots, and `models/manifest.json`
are committed, and `committed_state` answers UNVERIFIABLE -- never UNTRACKED, the
only answer allowed to skip -- when there is no repository to ask. Three baseline
assertions failed, the run aborted at the baseline, and 35 mutations were
unjudged while nothing said why. Same class of defect as the copy manifest
robbing `scripts/`: the suite in the copy is not the suite in the repo.

**The copy has to be whole, and it says so.** A copy missing one file is not a
smaller copy, it is a different tree: a test module that resolves a repo path at
import time raises during collection, the runner reports no failures, and every
verdict that arm can produce is INCONCLUSIVE. That is the "exit codes are not
verdicts" failure one level up -- the harness was loud about it, but only about
a symptom, and only once somebody actually ran it. `build_tree` now raises on a
missing required path instead of skipping it, and `_check_the_copy_is_whole` runs
on every copy it builds.

Note that this file is not named `test_*.py`, so `pytest tests/` does not collect
it -- it is a harness, not a test module. What checks the harness is
`tests/test_mutation_check.py`, which is collected, and which also says why.

Verdicts: CAUGHT / SURVIVED / INCONCLUSIVE / SKIPPED.

Known rough edge, not fixed: this block-buffers, so a redirected run shows nothing
until it exits. Not a defect in any verdict, only in watching one.

"""
from __future__ import annotations

import ast
import atexit
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"

# --- what gets copied ------------------------------------------------------
# Explicit, because an implicit copy either misses something the target reads or
# drags in .git and a 400MB node_modules. `data/cache` is excluded for a second
# reason: it is gitignored, so a stale local parquet cache would make a network
# leak pass locally and fail on CI.
#
# Every entry here is REQUIRED and `build_tree` refuses to build a tree that is
# missing one. `if src.exists()` was the silent-skip: a path absent from the repo
# produced a copy that simply lacked it, the copied suite then died during
# collection, and the baseline gate reported the harness as red without ever
# saying which file was gone. That is how `scripts/` went missing for months --
# `scripts/null_fabricated_market_hits.py` is loaded by
# `tests/test_null_fabricated_market_hits.py` at MODULE scope via
# `spec_from_file_location`, so its absence raised FileNotFoundError while
# collecting, `run_python` scored that INCONCLUSIVE, and `main` printed
# "BASELINE IS RED" and exited 2. Every one of the 32 mutations was unjudged and
# the run looked like a broken baseline rather than a missing directory.
#
# `.github` is here because tests/test_deploy_workflow.py reads the real workflow
# files, and a copy without them turns the baseline red. That is the baseline
# gate doing its job on its first run.
# `scripts` is the fifth entry this manifest needed. a03b1d9 introduced the list
# as ["src", "tests", "models", ".github"]; 0393f52 (2026-09-28) then added
# tests/test_null_fabricated_market_hits.py, which loads a repo path at import
# time, and the list was never updated. It is also read by
# `test_passing_td_record_absence.py`, whose scan roots it used to *skip* when
# absent -- a scan that quietly got smaller and still reported "no reference
# found". That half is closed over there (`tracked_artifacts.require`); the
# copy-side floor is `_check_the_copy_is_whole` below.
COPY_DIRS = ["src", "tests", "models", ".github", "scripts"]
COPY_FILES = ["pyproject.toml"]
DATA_GLOBS = ["data/*.json"]
FRONTEND_FILES = [
    "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json",
    "tsconfig.app.json", "tsconfig.node.json", "props_state_check.mjs",
]
FRONTEND_DIRS = ["src"]
# Symlinked, not copied: both are large and neither is mutated.
SYMLINKS = [".venv", "frontend/node_modules"]

FRONTEND_SUCCESS = "all frontend state assertions passed"
FRONTEND_FAIL = "FRONTEND RESULT: FAIL"
FRONTEND_PASS = "FRONTEND RESULT: PASS"


@dataclass(frozen=True)
class Mutation:
    ident: str
    name: str
    relpath: str          # path inside the repo, i.e. inside the copy
    find: str
    replace: str
    arm: str              # "python" | "frontend"
    canary: bool = False
    note: str = ""


M = Mutation
MUTATIONS: list[Mutation] = [
    # --- the original defect ------------------------------------------------
    M("M1", "outer except returns [] again (the original bug)",
      "src/nfl_predictor/api/routes.py",
      '''        logger.exception("Failed to load player props for season=%s week=%s", season, week)
        raise PlayerPropsUnavailable(f"unhandled error in the props pipeline: {e!r}") from e''',
      '''        logger.exception("Failed to load player props for season=%s week=%s", season, week)
        return []''',
      "python"),
    M("M2", "route converts the unavailable error back into an empty 200",
      "src/nfl_predictor/api/routes.py",
      '''        logger.error("player props unavailable for season=%s week=%s: %s", season, week, exc)
        raise HTTPException(status_code=503, detail=PROPS_UNAVAILABLE_DETAIL) from exc''',
      '''        return []''',
      "python"),
    M("M3", "total-failure guard deleted",
      "src/nfl_predictor/api/routes.py",
      "        if latest_players.shape[0] and not results:",
      "        if False:",
      "python"),
    M("M4", "upstream-data-gap check deleted",
      "src/nfl_predictor/api/routes.py",
      '''        if latest_players.empty and (player_history["season"] == season).sum() == 0:''',
      '''        if False:''',
      "python"),
    # One line, not two. The anchor used to span `carried = ...` and `if carried:`
    # as adjacent lines; a four-line comment explaining the carry-forward went
    # between them, the anchor stopped matching, and the harness reported SKIPPED
    # and exit 1 -- which is the designed behaviour for a stale mutation, but it
    # means this job was red on arrival and never said which mutation or why in
    # CI, because CI never ran it. Same edit, same test, fewer lines to rot.
    M("M5", "snapshot drops the previous props on failure (reintroduces the frozen empty)",
      "src/nfl_predictor/public_snapshot.py",
      '''        carried = (previous or {}).get("player_props") or []''',
      '''        carried = []''',
      "python"),
    M("M6", "snapshot status hardcoded to 'ok'",
      "src/nfl_predictor/public_snapshot.py",
      '''            player_props, props_status = carried, "stale"''',
      '''            player_props, props_status = carried, "ok"''',
      "python"),
    M("M7", "PUBLIC_MODE games check deleted (serves the committed empty snapshot)",
      "src/nfl_predictor/api/routes.py",
      '''    if snap.get("games"):
        return f"the snapshot has {len(snap['games'])} game(s) and no player props for it"''',
      '''    if False:
        return "unused"''',
      "python"),
    M("M8", "503 downgraded to 500 (loses the retryable signal)",
      "src/nfl_predictor/api/routes.py",
      '''        raise HTTPException(status_code=503, detail=PROPS_UNAVAILABLE_DETAIL) from exc''',
      '''        raise HTTPException(status_code=500, detail=PROPS_UNAVAILABLE_DETAIL) from exc''',
      "python"),
    M("M9", "frontend error branch deleted (a failed fetch renders the empty state)",
      "frontend/src/pages/PlayerPropsPage.tsx",
      '''      {error && (
        <p role="alert">
          Player props could not be loaded: {error}
        </p>
      )}
''',
      "",
      "frontend"),
    M("M10", "frontend table shown beside the error (zero-row grid)",
      "frontend/src/pages/PlayerPropsPage.tsx",
      """      {!error && !loading && (""",
      """      {!loading && (""",
      "frontend"),
    M("M11", "frontend loading branch deleted (a pending fetch claims no props)",
      "frontend/src/pages/PlayerPropsPage.tsx",
      """      {loading && <p>Loading…</p>}\n""",
      "",
      "frontend"),
    M("M12", "frontend .catch deleted (reintroduces the swallowed rejection)",
      "frontend/src/pages/PlayerPropsPage.tsx",
      '''      .catch((err) => {
        setError(err.message);
        setProps([]);
        setStale(false);
      })
''',
      "",
      "frontend"),
    M("M13", "live no-games branch also raises (every empty becomes an error)",
      "src/nfl_predictor/api/routes.py",
      '''            # this is the one empty result that is a true answer, and it is the
            # only one allowed to reach a reader as an empty list.
            return []''',
      '''            raise PlayerPropsUnavailable("no games in this week")''',
      "python"),
    M("M14", "PUBLIC_MODE no-games week also raises",
      "src/nfl_predictor/api/routes.py",
      '''            return snap.get("player_props") or []''',
      '''            raise HTTPException(status_code=503, detail="no props and no games")''',
      "python"),
    M("M15", "total-failure guard also requires a failure (everyone-skipped slips through)",
      "src/nfl_predictor/api/routes.py",
      "        if latest_players.shape[0] and not results:",
      "        if latest_players.shape[0] and not results and failed:",
      "python"),
    M("M16", "stale header never set (a stale week served as a fresh 200)",
      "src/nfl_predictor/api/routes.py",
      '''            if snap.get("player_props_status") == "stale":''',
      '''            if False:''',
      "python"),
    # Reported SURVIVED on purpose, and kept for that reason rather than deleted.
    # `snapshot_props_unavailable` returns early on `if snap.get("player_props"):
    # return None`, and `_build_week` only ever writes "stale" alongside
    # carried-forward rows, so a stale week carrying no rows is unreachable.
    # The mutation therefore documents a dead branch instead of hiding it.
    M("M17", "CANARY: 'stale' added to the status comparison (an unreachable state)",
      "src/nfl_predictor/api/routes.py",
      '''    if snap.get("player_props_status") == "unavailable":''',
      '''    if snap.get("player_props_status") in ("unavailable", "stale"):''',
      "python", canary=True),
    M("M18", "status checked before rows (blanking carried-forward props)",
      "src/nfl_predictor/api/routes.py",
      '''    if snap.get("player_props"):
        return None
    if snap.get("player_props_status") == "unavailable":''',
      '''    if snap.get("player_props_status") == "unavailable":''',
      "python"),
    M("M19", "facts: no HTTPException catch (the bundle 500s on a 503)",
      "src/nfl_predictor/api/facts.py",
      '''    try:
        return routes.get_player_props(season, week), False
    except HTTPException as exc:
        # A 503 from the props route is the expected shape of "unavailable". A
        # 404 would mean the route itself is gone, which is not something to
        # absorb quietly -- re-raise so it is not mistaken for no players.
        if exc.status_code != 503:
            raise
        logger.warning("facts: player props unavailable for season=%s week=%s: %s", season, week, exc.detail)
        return [], True''',
      '''    return routes.get_player_props(season, week), False''',
      "python"),
    M("M20", "facts: PUBLIC_MODE reverts to its own rule (the two panels disagree)",
      "src/nfl_predictor/api/facts.py",
      '''        if routes.snapshot_props_unavailable(snap) is not None:''',
      '''        if False:''',
      "python"),
    M("M21", "facts: the flag is hardcoded False (the panel goes silent again)",
      "src/nfl_predictor/api/facts.py",
      '''        "players_unavailable": players_unavailable,''',
      '''        "players_unavailable": False,''',
      "python"),
    M("M22", "public 503 body leaks the upstream cause again",
      "src/nfl_predictor/api/routes.py",
      '''        logger.error("player props unavailable for season=%s week=%s: %s", season, week, exc)
        raise HTTPException(status_code=503, detail=PROPS_UNAVAILABLE_DETAIL) from exc''',
      '''        logger.error("player props unavailable for season=%s week=%s: %s", season, week, exc)
        raise HTTPException(status_code=503, detail=f"could not be loaded: {exc}") from exc''',
      "python"),
    M("M23", "skipped and failed conflated again ('every player failed (0 of 1)')",
      "src/nfl_predictor/api/routes.py",
      '''            if skipped and not failed:''',
      '''            if False:''',
      "python"),
    M("M24", "live path stops wrapping (a raw exception reaches the caller)",
      "src/nfl_predictor/api/routes.py",
      '''    except Exception as e:
        # The old `return []`. A 404 on the season's stats, a dead upstream host
        # and a crash in feature building all used to land here and reach a reader
        # as an empty props table.
        logger.exception("Failed to load player props for season=%s week=%s", season, week)
        raise PlayerPropsUnavailable(f"unhandled error in the props pipeline: {e!r}") from e''',
      '''    except Exception:
        raise''',
      "python"),
    # --- frontend stale marker ----------------------------------------------
    M("M25", "frontend never renders the stale notice",
      "frontend/src/pages/PlayerPropsPage.tsx",
      '''      {!loading && !error && stale && (''',
      '''      {!loading && !error && false && (''',
      "frontend"),
    M("M26", "stale week also claims the empty state",
      "frontend/src/pages/PlayerPropsPage.tsx",
      """      {!loading && !error && !stale && props.length === 0 && (""",
      """      {!loading && !error && props.length === 0 && (""",
      "frontend"),
    # --- round 2: the holes the review's probes found ------------------------
    M("M27", "roster-fallback failure logged as `pass` (the outage goes quiet)",
      "src/nfl_predictor/api/routes.py",
      '''                logger.warning("Failed to fetch season roster fallback for teams=%s: %s", missing_teams, roster_err)''',
      '''                pass''',
      "python"),
    M("M28", "stale notice reworded into something that no longer says it is stale",
      "frontend/src/pages/PlayerPropsPage.tsx",
      """          These projections come from an earlier build &mdash; the latest rebuild of this
          week failed, so they may not reflect the current model.""",
      """          Data refreshed recently.""",
      "frontend"),
    M("M29", "backend 503 sentence reverts to 'for this game' (wrong unit, rendered verbatim)",
      "src/nfl_predictor/api/routes.py",
      '''    "Player props are temporarily unavailable for this week. The projections could "
    "not be produced. This is a failure, not a week without props."''',
      '''    "Player props are temporarily unavailable for this game. The projections could "
    "not be produced. This is a failure, not a game without props."''',
      "python"),
    M("M30", "CORS expose_headers dropped (the marker is set but a browser cannot read it)",
      "src/nfl_predictor/api/main.py",
      "    expose_headers=[PROPS_STALE_HEADER],\n",
      "",
      "python"),
    M("M31", "facts contract model drops players_unavailable again",
      "tests/test_facts.py",
      "    players_unavailable: bool = False\n",
      "",
      "python"),
    # --- the canaries ------------------------------------------------------
    # Each must report SURVIVED. If one reports CAUGHT, the detector has stopped
    # detecting and every other verdict in the run is void -- so that fails the
    # whole harness rather than being reported as a good result.
    M("C1", "CANARY: a comment edit no assertion can see (must report SURVIVED)",
      "src/nfl_predictor/api/routes.py",
      "# Producing nothing at all is an outage, not a week without props.",
      "# Producing nothing at all is completely unremarkable, in fact.",
      "python", canary=True),
    M("C2", "CANARY: a docstring edit no assertion can see (must report SURVIVED)",
      "src/nfl_predictor/public_snapshot.py",
      '"""One week of precomputed data.',
      '"""One week of precomputed data (a slightly different opening line).',
      "python", canary=True),
    M("C3", "CANARY: a frontend JSX label edit no assertion can see (must report SURVIVED)",
      "frontend/src/pages/PlayerPropsPage.tsx",
      "Sort by:{\" \"}",
      "Order by:{\" \"}",
      "frontend", canary=True),
    M("C4", "CANARY: a frontend comment edit no assertion can see (must report SURVIVED)",
      "frontend/props_state_check.mjs",
      "let currentTest = '';",
      "let currentTest = 'unmutated';",
      "frontend", canary=True),
]

# --- tree copy -------------------------------------------------------------

_TMP: Path | None = None

# An identity, because `git commit` refuses without one and a runner's global
# config is not something this harness may assume. `gpgsign=false` for the same
# reason: a runner with signing configured would otherwise hang or fail on a key
# it does not have.
_GIT = (
    "git",
    "-c", "user.email=mutation-harness@example.invalid",
    "-c", "user.name=mutation harness",
    "-c", "commit.gpgsign=false",
)


def _git(cwd: Path, *args: str) -> None:
    """Run git in `cwd`, or raise. Never `check=False`: a copy that is only
    half-initialised is the anonymous failure this harness is not allowed to
    produce.

    With `GIT_*` stripped from the environment. `cwd` does not win: `GIT_DIR`,
    `GIT_WORK_TREE` and `GIT_INDEX_FILE` each redirect git away from it, so a
    caller who exports any of them gets `init`, `add` and `commit` aimed at the
    repository the harness was launched from -- the one thing in this file that
    must never be written to. Stripping the whole prefix rather than setting them
    to empty is deliberate: an empty `GIT_DIR` names a directory git will try to
    use, which is worse than not naming one.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run([*_GIT, *args], cwd=cwd, check=True, capture_output=True, text=True, env=env)


def _cleanup(*_args) -> None:
    if _TMP and _TMP.exists():
        shutil.rmtree(_TMP, ignore_errors=True)


def _require(path: Path, rel: str) -> Path:
    """Return `path`, or refuse -- never silently return without copying.

    `rel` is the repo-relative spelling, quoted into the message so the reader is
    told which entry of which list to edit rather than being handed a traceback
    from `shutil`.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{rel} is in COPY_DIRS/COPY_FILES but is not in the tree, so a copy of "
            f"the repo cannot be built. The copied suite resolves repo-relative "
            f"paths at import time; a copy that quietly omits one dies during "
            f"collection and every verdict from it is INCONCLUSIVE. Add the path "
            f"back to the repo, or -- if it is genuinely not needed -- remove it "
            f"from the copy list. Not skipping it."
        )
    return path


def build_tree() -> Path:
    """A writable copy of everything the targets read, and nothing else.

    Raises `FileNotFoundError` naming the path if a required entry of
    `COPY_DIRS`/`COPY_FILES` is not in the repo. It used to skip whatever was
    absent, which is how a whole directory went missing from the copy without a
    word -- see the note on `COPY_DIRS`.
    """
    global _TMP
    _TMP = Path(tempfile.mkdtemp(prefix="props-mutation-"))
    atexit.register(_cleanup)
    # `os._exit` rather than `sys.exit`: a SystemExit raised in a signal handler
    # while the main thread sits in `subprocess.run` is delivered late, and the
    # loop then walked on into a temp dir that had already been removed --
    # correct, but as a FileNotFoundError traceback rather than as a clean exit.
    def _on_signal(_sig, _frame):
        _cleanup()
        print("\ninterrupted; the tracked tree was never written to", file=sys.stderr)
        os._exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    for d in COPY_DIRS:
        src, dst = ROOT / d, _TMP / d
        _require(src, f"{d}/")
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in COPY_FILES:
        src = ROOT / f
        _require(src, f)
        shutil.copy2(src, _TMP / f)
    for pattern in DATA_GLOBS:
        for src in ROOT.glob(pattern):
            dst = _TMP / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for f in FRONTEND_FILES:
        src = ROOT / "frontend" / f
        if src.exists():
            dst = _TMP / "frontend" / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for d in FRONTEND_DIRS:
        src = ROOT / "frontend" / d
        if src.exists():
            shutil.copytree(src, _TMP / "frontend" / d,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for link in SYMLINKS:
        src = ROOT / link
        dst = _TMP / link
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src, dst)
    # The copy has to be a *git repository* in which the copied paths are
    # committed. tests/tracked_artifacts.py asks git about
    # data/public_snapshot.json, about the three scan roots, and about
    # models/manifest.json, and `committed_state` answers UNVERIFIABLE for every
    # one of them when there is no repository to ask -- so three baseline
    # assertions fail and `main()` aborts with exit 2 before judging a single
    # mutation. Same class of defect as the missing `scripts/`: the suite in the
    # copy is not the suite in the repo, and it reads as a broken suite rather
    # than as a missing repository.
    #
    # Three commands, not a copy of `.git`, which is the thing COPY_DIRS exists
    # to avoid. `add -A` records the two symlinks as symlinks -- one entry each,
    # git does not follow them -- so this stays cheap.
    _git(_TMP, "init", "--quiet")
    _git(_TMP, "add", "-A", "--", ".")
    _git(_TMP, "commit", "--quiet", "-m", "the copy under test")
    # The copy is finished; now ask whether it is whole. Cheap (an AST walk of
    # tests/test_*.py) and it turns "the baseline went red for a reason nobody
    # could see" into a sentence naming the file, before a single mutation runs.
    _check_the_copy_is_whole(_TMP)
    return _TMP


# --- is the copy whole? ----------------------------------------------------
# The copy manifest above is a hand-written list, and a hand-written list rots:
# the fifth entry the copy manifest ever needed was `scripts/`, and nobody
# updated the list, so the copy quietly stopped being the repo. The evidence was
# an INCONCLUSIVE baseline, which is loud but anonymous -- it does not say
# `scripts/null_fabricated_market_hits.py` is not there, and it reads as a broken
# suite rather than a missing directory.
#
# So the invariant is checked instead of being trusted. Every path a test module
# builds off `Path(__file__).resolve().parents[1]` at MODULE level is resolved at
# import time, which is exactly the set that can abort collection. Function-level
# ones are deliberately not here: a missing file inside a test body is an ordinary
# loud failure, not an anonymous collection error.

def _is_repo_root(node: ast.AST) -> bool:
    """True for the `Path(__file__).resolve().parents[1]` spelling."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == 1
    )


def _resolve_repo_paths(node: ast.expr, env: dict[str, list[tuple[str, ...]]]):
    """Every repo-relative path `node` reads, or `[]` if it reads none.

    A LIST, not one path. A wrapped call can read more than one repo file --
    `dict(manifest=MODELS_DIR / "manifest.json", workflow=REPO / "deploy.yml")` --
    and returning only the first is how half of what a module reads at import
    time goes unchecked, which is the same class of silent hole as returning
    none.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        tail = node.right
        if not (isinstance(tail, ast.Constant) and isinstance(tail.value, str)):
            return []
        return [head + (tail.value,) for head in _resolve_repo_paths(node.left, env)]
    if isinstance(node, ast.Call):
        # An import-time read is usually *wrapped*: the file is named by one
        # module-level constant and read by the next one --
        #
        #     COMMITTED_MODELS_DIR = ROOT / "models"
        #     COMMITTED_MANIFEST = json.loads((COMMITTED_MODELS_DIR / "manifest.json").read_text())
        #
        # and the resolver used to stop at the first line. `models` is a
        # directory, so `(tree / "models").exists()` is satisfied by the copy
        # manifest even when the file inside it was never copied, and
        # `manifest.json` -- the thing whose absence actually raises during
        # collection -- was never checked at all. So unwrap, and report every
        # path found inside.
        found: list[tuple[str, ...]] = []
        for inner in (*node.args, *(kw.value for kw in node.keywords), node.func):
            found += _resolve_repo_paths(inner, env)
        return found
    if isinstance(node, ast.Attribute):
        # `(MODELS_DIR / "manifest.json").read_text()` -- the path is the receiver.
        return _resolve_repo_paths(node.value, env)
    if isinstance(node, ast.Name):
        return env.get(node.id, [])
    if _is_repo_root(node):
        return [()]
    return []


def _module_level_repo_paths(module_path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """`[(name, components)]` for the module-level repo paths in one test module."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    env: dict[str, list[tuple[str, ...]]] = {}
    found: list[tuple[str, tuple[str, ...]]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        resolved = _resolve_repo_paths(node.value, env)
        if not resolved:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            env[target.id] = resolved
        # `dict.fromkeys` so one constant naming the same path twice through a
        # wrapper is not reported as two required paths.
        found += [
            (target.id if isinstance(target, ast.Name) else "?", components)
            for components in dict.fromkeys(resolved)
            if components
        ]
    return found


def _suite_modules(tree: Path) -> list[Path]:
    """The files pytest imports while collecting, so the ones that can abort it."""
    tests = tree / "tests"
    if not tests.is_dir():
        return []
    return sorted(p for p in tests.iterdir() if p.name == "conftest.py" or p.name.startswith("test_"))


def _check_the_copy_is_whole(tree: Path) -> None:
    """Raise unless every import-time repo path in the suite exists in `tree`.

    Raises `AssertionError` listing every offender, not the first: one missing
    file has a habit of arriving with three more.
    """
    offenders = []
    for module in _suite_modules(tree):
        for name, components in _module_level_repo_paths(module):
            rel = Path(*components)
            if not (tree / rel).exists():
                offenders.append(f"{module.name}: {name} -> {rel}")
    if offenders:
        raise AssertionError(
            f"the copy at {tree} is missing {len(offenders)} path(s) that the "
            f"suite resolves at import time, so it would abort collection:\n  "
            + "\n  ".join(offenders)
            + "\n  Add the containing path to COPY_DIRS/COPY_FILES in "
              "tests/mutation_check.py, or stop reading it from a module-level "
              "constant."
        )


def write_mutation(tree: Path, mutation: Mutation) -> Path:
    target = tree / mutation.relpath
    original = target.read_text()
    if mutation.find not in original:
        raise LookupError(
            f"{mutation.ident}: anchor not found in {mutation.relpath}\n"
            f"  looking for: {mutation.find!r}\n"
            f"  the file starts: {original[:200]!r}"
        )
    target.write_text(original.replace(mutation.find, mutation.replace, 1))
    return target


# --- running ---------------------------------------------------------------

def assert_copy_is_under_test(tree: Path) -> None:
    """Refuse to run unless `nfl_predictor` imports from the COPY.

    This is not paranoia, it is the fourth bug this harness has had. The venv is
    an *editable* install, and its `.pth` puts the real `src` on `sys.path`;
    `PYTHONPATH` puts the copy's first, but only because `PYTHONPATH` is
    processed before site-packages. Get that order wrong -- or trust
    `pyproject.toml`'s `pythonpath = ["src"]`, which is inert here, see the note
    in `assert_copy_is_under_test`'s caller -- and every mutation is applied to a
    file nothing imports. The result is a run that reports twenty mutations
    SURVIVED and looks like a catastrophic hole in the tests, when the truth is
    that the tests never saw a single one of them.
    """
    probe = (
        "import pathlib, nfl_predictor, sys;"
        f"print('MODULE=' + str(pathlib.Path(nfl_predictor.__file__).resolve()))"
    )
    proc = subprocess.run(
        [str(PYTHON), "-c", probe],
        cwd=tree, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(tree / "src")},
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("MODULE=")), "")
    module = line.removeprefix("MODULE=")
    if not module.startswith(str(tree.resolve())):
        raise RuntimeError(
            "HARNESS ERROR: `nfl_predictor` does not import from the copy, so no "
            f"mutation is being tested.\n  expected under: {tree.resolve()}\n"
            f"  actually imported: {module or '<nothing>'}\n"
            "  Without this check the run would report every mutation as SURVIVED."
        )


def run_python(tree: Path) -> tuple[str, str]:
    """Returns ("PASS" | "FAIL" | "INCONCLUSIVE", output).

    The exit code is not consulted for the verdict. `returncode != 0` conflates
    three very different things, and the first version of this harness read all
    three as CAUGHT:

    * a real assertion failure -- the only thing that is evidence;
    * a collection error (exit 2), e.g. the mutation produced a SyntaxError or
      broke an import. M19 did exactly this and was scored as a good catch;
    * no tests collected at all (exit 5).

    So a failure is only believed when pytest's own summary says tests failed.
    Anything else is INCONCLUSIVE and fails the harness, because a runner that
    cannot run is not a runner that detected.

    `-m "not network"` for the same reason tests.yml's own pytest step carries
    it, and here it is not optional. The two deselected tests reconcile against
    live nflverse data, so without the marker the harness opened the network 27
    times per pass and its baseline could go red because an upstream schema
    moved -- reported as "BASELINE IS RED -- aborting" for a cause that has
    nothing to do with whether the tests bite. It also nearly halved the cost of
    a pass (76s -> 40s per run), which is most of why the harness fits in CI at
    all. Same suite, same marker, as the CI pytest step.
    """
    proc = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-m", "not network", "tests"],
        cwd=tree, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(tree / "src")},
    )
    out = proc.stdout + proc.stderr
    if re.search(r"\d+ passed", out) and " failed" not in out and "error" not in out.split("short test summary")[-1]:
        return "PASS", out
    if re.search(r"\d+ failed", out):
        return "FAIL", out
    return "INCONCLUSIVE", out


def run_frontend(tree: Path) -> tuple[str, str]:
    """Returns (verdict, output) where verdict is PASS | FAIL | INCONCLUSIVE.

    Never `returncode`. A runner that cannot import jsdom exits non-zero exactly
    like one that caught an assertion, and reading those as the same thing is how
    a broken harness reported `all mutations caught`.
    """
    # The frontend arm's equivalent of `assert_copy_is_under_test`. The python
    # guard exists because that arm silently tested the real tree once already;
    # this one keeps the other arm honest for the same reason. A missing source
    # file means vite's SSR loader would resolve nothing and the runner would
    # report INCONCLUSIVE for every mutation, which is confusing rather than
    # honest, so it is caught here and named.
    for required in ("src/pages/PlayerPropsPage.tsx", "src/api/client.ts", "props_state_check.mjs"):
        if not (tree / "frontend" / required).exists():
            return "INCONCLUSIVE", f"the copy is missing frontend/{required}; node would not load the copy"
    if not (tree / "frontend" / "node_modules").exists():
        return "INCONCLUSIVE", "the copy has no frontend/node_modules; node would not resolve react"

    try:
        proc = subprocess.run(
            ["node", "props_state_check.mjs"],
            cwd=tree / "frontend", capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return "INCONCLUSIVE", "the runner timed out"
    out = proc.stdout + proc.stderr
    if FRONTEND_SUCCESS in out and FRONTEND_PASS in out:
        return "PASS", out
    if FRONTEND_FAIL in out:
        return "FAIL", out
    return "INCONCLUSIVE", out


# --- main ------------------------------------------------------------------

_SHARD = re.compile(r"^--shard=(\d+)/(\d+)$")


def select(argv: list[str]) -> list[Mutation]:
    """The mutations this run judges: the ones named, or shard I of N.

    Sharding exists because a full pass is minutes, not seconds, and this runs
    on every push. It is `MUTATIONS[i::n]` rather than a list of ids per shard in
    the workflow, because a list in a workflow is a list that rots: a mutation
    added to MUTATIONS and left out of every shard is a mutation nobody judges,
    which is the same silent hole the CI job was added to close, moved inside
    the job.

    Every shard also gets every canary, and that is not an optimisation to undo
    later. `main` computes `all(v == "SURVIVED" for _, v in canaries)`, which
    over an empty sequence is True, so a shard holding no canary would print
    `0 canary (all correct: True)` -- a clean bill of health from a shard that
    never checked whether its own detector works. That is why this is not "the
    canaries go in shard 0": one shard's verdicts are evidence about the tests,
    and the rest would be evidence about nothing. The canaries cost one run each
    per shard, which is the price of a shard's verdict meaning something.
    """
    names = [a for a in argv[1:] if not a.startswith("--")]
    shards = [a for a in argv[1:] if a.startswith("--")]
    picked = [m for m in (_SHARD.match(a) for a in shards) if m]
    unknown = [a for a in shards if not _SHARD.match(a)]
    if unknown:
        raise SystemExit(f"unknown option {unknown[0]!r}; the only one is --shard=I/N")
    if not picked:
        return [m for m in MUTATIONS if not names or m.ident in set(names)]
    if names:
        raise SystemExit(
            f"{picked[0].group(0)} and {sorted(names)} are both given; a run "
            "judges either one slice of every mutation or some of them by name, not both"
        )
    index, count = int(picked[0][1]), int(picked[0][2])
    if count < 1 or not 0 <= index < count:
        raise SystemExit(f"--shard={index}/{count} is not 0 <= i < N with N >= 1")
    # Membership by identity, not by dataclass equality: two mutations with the
    # same fields would satisfy `in` at once, and the ids are not unique either.
    mine = {id(m) for m in MUTATIONS[index::count]}
    return [m for m in MUTATIONS if id(m) in mine or m.canary]


#: How to say the canary verdict, including when there was none to give.
CANARY_VERDICT = {
    True: "all correct: True",
    False: "all correct: False",
    None: "none ran, so the detector was NOT checked",
}


def main(argv: list[str]) -> int:
    wanted = set(argv[1:])
    tree = build_tree()
    selected = select(argv)
    if not selected:
        print(f"no mutation matched {sorted(wanted)}", file=sys.stderr)
        return 2

    assert_copy_is_under_test(tree)

    # Baselines first, once, on the unmutated copy. A red baseline makes every
    # verdict meaningless, so stop before claiming anything.
    print("baseline (unmutated copy):")
    base_verdict, base_out = run_python(tree)
    base_ok = base_verdict == "PASS"
    print(f"  pytest            {base_verdict}")
    front_verdict, _ = run_frontend(tree)
    print(f"  props_state_check {front_verdict}")
    if not base_ok:
        print("\nBASELINE IS RED -- aborting; no verdict from this run means anything.")
        print(base_out[-3000:])
        return 2
    if front_verdict != "PASS":
        print(f"\nFRONTEND BASELINE IS {front_verdict} -- aborting.")
        return 2

    results = []
    for mutation in selected:
        # Restore the pristine file first. The copy is shared between mutations,
        # and a stale edit from the previous one would make the next verdict a
        # lie -- which is the failure mode this harness is meant to be immune to.
        # BEFORE the mutation. Its real job is here, and round 2's comment claimed a
        # job it was not doing: if the previous iteration's `finally` restore failed,
        # the tree is already dirty before this mutation is even applied, and this
        # catches that. It is cheap and it is the common failure.
        assert_copy_is_under_test(tree)

        target = tree / mutation.relpath
        pristine = target.read_text()
        try:
            write_mutation(tree, mutation)
        except LookupError as exc:
            # A stale anchor is a broken harness, not a result. Counting it as
            # either verdict would be a lie, and raising here would throw away
            # every verdict already gathered.
            print(f"SKIPPED      {mutation.ident}: {exc}".splitlines()[0])
            results.append((mutation, "SKIPPED", str(exc)))
            continue
        after = target.read_text()
        assert after != pristine, f"{mutation.ident}: the mutation did not change the file"

        # AFTER the mutation, which is the only place the original claim can be
        # true: a mutation that alters import resolution -- an `__init__.py`, a
        # `sys.path` edit, a `pyproject.toml` `[tool.pytest.ini_options]` change --
        # would otherwise point the suite back at the real tree, and every verdict
        # for this mutation would be about code nobody edited. Round 2 ran the check
        # only before `write_mutation`, where it could not see that.
        try:
            assert_copy_is_under_test(tree)
        except RuntimeError as exc:
            print(f"SKIPPED      {mutation.ident}: {exc}".splitlines()[0])
            results.append((mutation, "SKIPPED", str(exc)))
            target.write_text(pristine)
            continue
        try:
            if mutation.arm == "python":
                pv, out = run_python(tree)
                verdict = {"PASS": "SURVIVED", "FAIL": "CAUGHT", "INCONCLUSIVE": "INCONCLUSIVE"}[pv]
            else:
                fv, out = run_frontend(tree)
                verdict = {"PASS": "SURVIVED", "FAIL": "CAUGHT", "INCONCLUSIVE": "INCONCLUSIVE"}[fv]
        finally:
            target.write_text(pristine)
            restored = target.read_text()
            assert restored == pristine, f"{mutation.ident}: restore was not clean"
        results.append((mutation, verdict, out))
        if mutation.note:
            print(f"             {mutation.note}")
        flag = "CANARY" if mutation.canary else "      "
        print(f"{verdict:<12} {flag} {mutation.ident}: {mutation.name}")
        if verdict in ("SURVIVED", "INCONCLUSIVE"):
            if mutation.canary and verdict == "SURVIVED":
                print("            (expected -- this is the canary)")
            for line in out.splitlines():
                if line.startswith(("FAILED", "  FAIL", "AssertionError")):
                    print(f"            {line.strip()}")
                    break

    _cleanup()
    global _TMP
    _TMP = None

    survivors = [(m, v) for m, v, _ in results if v == "SURVIVED" and not m.canary]
    inconclusive = [(m, v) for m, v, _ in results if v == "INCONCLUSIVE"]
    skipped = [(m, v) for m, v, _ in results if v == "SKIPPED"]
    canaries = [(m, v) for m, v, _ in results if m.canary]
    # `all()` over an empty sequence is True, so a run that selected no canary
    # would print `0 canary (all correct: True)` -- the only verdict in this file
    # that would claim a check it did not do. Named selection can reach that
    # (`mutation_check.py M4`); sharding cannot, because every shard is handed all
    # five. Reported rather than enforced: a named run is a debugging aid, and
    # failing it would be a worse trade than saying out loud what it is.
    canary_ok = all(v == "SURVIVED" for _, v in canaries) if canaries else None

    print()
    if canary_ok is False:
        print("HARNESS BROKEN: a canary mutation did not report SURVIVED, so the")
        print("detector itself is not working and every other verdict here is void.")
        for m, v in canaries:
            if v != "SURVIVED":
                print(f"  {m.ident} reported {v}, expected SURVIVED")
    if inconclusive:
        print(f"{len(inconclusive)} INCONCLUSIVE (the runner did not produce a verdict):")
        for m, v in inconclusive:
            print(f"  {m.ident} {m.name} -> {v}")
    if skipped:
        print(f"{len(skipped)} SKIPPED (the anchor no longer matches the source -- a stale mutation):")
        for m, _ in skipped:
            print(f"  - {m.ident} {m.name}")
    if survivors:
        print(f"{len(survivors)} mutation(s) SURVIVED:")
        for m, _ in survivors:
            print(f"  - {m.ident} {m.name}")
    caught = sum(1 for _, v, _ in results if v == "CAUGHT")
    print(f"\n{caught} caught, {len(survivors)} survived, {len(inconclusive)} inconclusive, "
          f"{len(canaries)} canary ({CANARY_VERDICT[canary_ok]})")
    return 1 if (survivors or inconclusive or skipped or canary_ok is False) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

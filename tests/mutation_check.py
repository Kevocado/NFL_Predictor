"""Mutation harness for tests/test_player_props_failure.py.

For each mutation: apply it to a scratch copy of the source, run the suite
against it, record whether the suite caught it. A surviving mutation is a hole in
the test, not a pass.

Usage: .venv/bin/python tests/mutation_check.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "src/nfl_predictor/api/routes.py"
SNAPSHOT = ROOT / "src/nfl_predictor/public_snapshot.py"
PAGE = ROOT / "frontend/src/pages/PlayerPropsPage.tsx"

# (id, file, find, replace, tests_to_run)
MUTATIONS = [
    (
        "M1 outer-except returns [] again (the original bug)",
        ROUTES,
        """        logger.exception("Failed to load player props for season=%s week=%s", season, week)
        raise PlayerPropsUnavailable(
            f"player props for season {season} week {week} could not be loaded: {e}"
        ) from e""",
        """        logger.exception("Failed to load player props for season=%s week=%s", season, week)
        return []""",
        "tests/test_player_props_failure.py",
    ),
    (
        "M2 route converts the unavailable error back into an empty 200",
        ROUTES,
        """    except PlayerPropsUnavailable as exc:
        # 503, not 200 []. A bare 500 would be honest but unactionable, and
        # returning an empty list here is the bug this whole path exists to fix.
        raise HTTPException(status_code=503, detail=str(exc)) from exc""",
        """    except PlayerPropsUnavailable as exc:
        return []""",
        "tests/test_player_props_failure.py",
    ),
    (
        "M3 per-player total-failure check deleted (empty results served as empty week)",
        ROUTES,
        """        if latest_players.shape[0] and not results:
            raise PlayerPropsUnavailable(
                f"every player in season {season} week {week} failed to predict "
                f"({len(failed)} of {latest_players.shape[0]} failed, first={failed[:3]}); "
                "this is a prediction failure, not a week without props"
            )
""",
        "",
        "tests/test_player_props_failure.py",
    ),
    (
        "M4 upstream-data-gap check deleted",
        ROUTES,
        """        if latest_players.empty and (player_history["season"] == season).sum() == 0:
            raise PlayerPropsUnavailable(
                f"no player stats are available for season {season}, so no props can be "
                f"projected for week {week} of it (upstream player_stats_{season}.parquet is "
                "not published); this is an upstream data gap, not a week without props"
            )
""",
        "",
        "tests/test_player_props_failure.py",
    ),
    (
        "M5 snapshot drops the previous props on failure (reintroduces the frozen empty)",
        SNAPSHOT,
        """        carried = (previous or {}).get("player_props") or []
        if carried:""",
        """        carried = []
        if carried:""",
        "tests/test_player_props_failure.py",
    ),
    (
        "M6 snapshot status is hardcoded to 'ok'",
        SNAPSHOT,
        """            player_props, props_status = carried, "stale\"""",
        """            player_props, props_status = carried, "ok\"""",
        "tests/test_player_props_failure.py",
    ),
    (
        "M7 PUBLIC_MODE games check deleted (serves the committed empty snapshot as empty week)",
        ROUTES,
        """            if not props and snap.get("games"):""",
        """            if False:""",
        "tests/test_player_props_failure.py",
    ),
    (
        "M13 the no-games branch also 503s (every empty result becomes an error)",
        ROUTES,
        """        if upcoming_games.empty:
            # No games in this week at all. Nothing to project and nothing broken --
            # this is the one empty result that is a true answer, and it is the
            # only one allowed to reach a reader as an empty list.
            return []""",
        """        if upcoming_games.empty:
            raise PlayerPropsUnavailable(f"no games in week {week}")""",
        "tests/test_player_props_failure.py",
    ),
    (
        "M14 PUBLIC_MODE no-games week also 503s",
        ROUTES,
        """            if not props:
                # No games either -- an honest empty. Weeks 19-22 in the committed
                # snapshot look exactly like this: the schedule hasn't reached them.
                return []""",
        """            if not props:
                raise HTTPException(status_code=503, detail="no props and no games")""",
        "tests/test_player_props_failure.py",
    ),
    (
        "M8 503 downgraded to 500 (still not an empty state, but loses the retryable signal)",
        ROUTES,
        """        raise HTTPException(status_code=503, detail=str(exc)) from exc""",
        """        raise HTTPException(status_code=500, detail=str(exc)) from exc""",
        "tests/test_player_props_failure.py",
    ),
    (
        "M9 frontend error state dropped (a failed fetch renders the empty-state sentence)",
        PAGE,
        """      {error && (
        <p role="alert">
          Player props could not be loaded: {error}
        </p>
      )}
""",
        "",
        "frontend/props_state_check.mjs",
    ),
    (
        "M10 frontend suppresses the table on failure -> restored (zero-row table beside the error)",
        PAGE,
        """      {!error && (
        <>
          <label>
            Sort by:{" "}""",
        """      {(
        <>
          <label>
            Sort by:{" "}""",
        "frontend/props_state_check.mjs",
    ),
    (
        "M11 frontend loading state dropped (a pending fetch renders the empty-state sentence)",
        PAGE,
        """      {loading && <p>Loading…</p>}
""",
        "",
        "frontend/props_state_check.mjs",
    ),
    (
        "M12 frontend .catch dropped (reintroduces the swallowed rejection)",
        PAGE,
        """      .catch((err) => {
        setError(err.message);
        setProps([]);
      })
""",
        "",
        "frontend/props_state_check.mjs",
    ),
]


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    survivors = []
    for name, path, find, replace, target in MUTATIONS:
        original = path.read_text()
        if find not in original:
            print(f"SKIP  {name}\n        anchor not found in {path.relative_to(ROOT)}")
            survivors.append(name)
            continue
        backup = tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix)
        backup.write(original.encode())
        backup.close()
        try:
            path.write_text(original.replace(find, replace, 1))
            if target.startswith("frontend/"):
                # tsc alone only typechecks, so it is not evidence about
                # behaviour -- a deleted error branch typechecks fine. The
                # assertion runner is the real check; tsc runs alongside it to
                # catch a mutation that only breaks the types.
                code, out = run(["node", target.split("/", 1)[1]], cwd=ROOT / "frontend")
                print(f"{'CAUGHT  ' if code != 0 else 'SURVIVED'} {name}  ({target})")
                if code == 0:
                    survivors.append(name)
                    print("        the runner stayed green -- this assertion proves nothing")
                else:
                    for line in [l for l in out.splitlines() if l.startswith("  FAIL")][:4]:
                        print(f"        {line.strip()}")
                continue
            code, out = run([sys.executable, "-m", "pytest", "-q", target, "-p", "no:cacheprovider"])
            caught = code != 0
            print(f"{'CAUGHT  ' if caught else 'SURVIVED'} {name}")
            if not caught:
                survivors.append(name)
                print("        suite stayed green -- this assertion proves nothing")
            else:
                failed = [l for l in out.splitlines() if l.startswith("FAILED")]
                for line in failed[:4]:
                    print(f"        {line}")
        finally:
            shutil.copyfile(backup.name, path)
            Path(backup.name).unlink()

    print()
    if survivors:
        print(f"{len(survivors)} mutation(s) survived:")
        for s in survivors:
            print(f"  - {s}")
        return 1
    print("all mutations caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

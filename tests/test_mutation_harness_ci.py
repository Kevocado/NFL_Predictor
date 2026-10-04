"""The mutation harness must be in CI, and the CI job must not be skippable.

`tests/mutation_check.py` judges 35 mutations: it edits a source file in a copy
of the tree, re-runs the suite, and requires each edit to be *caught*. It is the
only thing in this repository that says the tests bite. It was never run by
`.github/workflows/tests.yml`, so on every push for as long as it existed the
answer to "do these tests bite?" was a file nobody executed.

Putting it in CI is the easy half. The half that actually matters is that the job
cannot be skipped without anything saying so, because the ways to do that all
report green:

* an `if:` on the job -- GitHub reports a condition-skipped job as *skipped*, and
  a skipped required check counts as satisfied. A harness that runs on
  `workflow_dispatch` and a cron is a harness that does not run on a PR, while
  the PR shows no red.
* `continue-on-error: true`, on the job or on any matrix leg -- the job's
  conclusion becomes `success` no matter what the harness printed. This is the
  one to be most careful of, because it is also the thing that stops a broken
  harness from blocking unrelated work, so it is the change someone makes first.
* a `paths:` filter on the workflow -- the harness then only runs when somebody
  remembers to touch a path it lists, which is the same silence with extra steps.
* `|| true`, or an `exit 0` at the end of the run block.

Each check below is also run against a copy of the file with one thing broken, so
a check that cannot fail cannot be mistaken for a check that passes. That is the
pattern `test_deploy_workflow.py` established for the deploy workflow and it is
the reason this file is not just an assertion that the word `mutation_check`
appears somewhere in a YAML file.

Nothing here runs the harness or opens a socket.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mutation_check import MUTATIONS, select
from test_deploy_workflow import jobs

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
HARNESS = REPO / "tests" / "mutation_check.py"

#: The job that runs the harness. Named, not discovered: a check that finds its
#: own subject by pattern cannot tell a renamed job from a deleted one, and a
#: deleted one is the failure this file exists to catch.
JOB = "mutation-harness"


def text() -> str:
    return WORKFLOW.read_text()


def job(whole: str, name: str = JOB) -> str:
    """One job's body, out of a *given* workflow text.

    Every check takes the whole text and slices what it needs from it. The first
    version passed in the job body but let the trigger check read the file from
    disk, so `test_each_check_can_fail` handed it a deliberately broken workflow
    and it graded the real one instead -- a red-check that could not fail, in the
    test whose whole subject is checks that cannot fail.
    """
    found = jobs(whole).get(name)
    assert found is not None, (
        f"{WORKFLOW} has no `{name}` job. If the harness is no longer in CI then the "
        "tests are unproven again, and this file is the thing that would have said so."
    )
    return found


# ---------------------------------------------------------------- the checks
#
# Each takes the whole workflow text, so `test_each_check_can_fail` below can
# hand it a broken copy. A check that reaches back to the file grades the real
# workflow and passes, which is the one thing these must never do.


def check_trigger(whole: str) -> None:
    """Nothing about the trigger may make a push skip the harness."""
    assert re.search(r"^on:[ \t]*$", whole, re.M), "no top-level `on:` block"
    assert re.search(r"^[ \t]+pull_request:", whole, re.M), (
        "the workflow no longer runs on pull_request, so the harness is not run for "
        "the changes most likely to weaken a test"
    )
    assert re.search(r"^[ \t]+push:[ \t]*$", whole, re.M), "the workflow does not run on push"
    assert re.search(r"^[ \t]+branches:[ \t]*\[?[ \t]*main", whole, re.M), (
        "push no longer targets main; a harness that only runs on main does not run "
        "on a branch"
    )
    assert not re.search(r"^[ \t]+paths:", whole, re.M), (
        "the workflow has a `paths:` filter, so the harness runs only when someone "
        "happens to touch a listed path -- a skip that reports nothing"
    )


def check_not_condition_gated(whole: str) -> None:
    """No `if:` on the job. A skipped job is a satisfied check."""
    body = job(whole)
    # Four spaces, because that is where a job-level key sits in this file:
    # `jobs:` at column 0, the job name at 2, its keys at 4. Steps are deeper
    # still, and the pytest job has a legitimate `if:` on its network step -- so
    # anchoring at 2 matches nothing at all, and anchoring anywhere deeper would
    # forbid the wrong thing. Both were written and shipped-in-comment here first.
    # The mutation in `test_each_check_can_fail` is what caught the 2-space
    # version, which is the point of having one: a check that cannot match a real
    # `if:` also cannot fail.
    offenders = [
        line for line in body.splitlines()
        if re.match(r"^ {4}if:", line)
    ]
    assert not offenders, (
        f"the {JOB} job is gated on a condition ({offenders[0].strip()!r}); GitHub "
        "reports a condition-skipped job as `skipped`, which counts as satisfied, so "
        "the harness would stop running without anything going red"
    )


def check_cannot_report_green_on_failure(whole: str) -> None:
    """No `continue-on-error`, no `|| true`, no `exit 0`."""
    body = job(whole)
    # Anchored to a line that *starts* with the key, because this workflow's own
    # comment explains why the key is forbidden and therefore contains the word.
    # A substring search made the workflow's reasoning fail its own check, which
    # is the shape of mistake worth naming: the guard and the thing it guards have
    # to share a vocabulary without the guard reading the prose.
    assert not re.search(r"^\s*continue-on-error:", whole, re.M), (
        "a `continue-on-error` key somewhere in this workflow: the harness's "
        "conclusion becomes `success` whatever it printed, which is a check that "
        "cannot fail -- the same defect as a test that cannot fail, and the reason "
        "the brief for this job rules it out. Note this is asserted against the "
        "whole file, because on a matrix it is set per-leg and a per-leg "
        "`continue-on-error` is the easy way to ship a harness that never blocks."
    )
    assert not re.search(r"\|\|\s*true", body), "the run block ends in `|| true`"
    assert not re.search(r"^\s*exit 0\s*$", body, re.M), (
        "the run block ends in `exit 0`, which reports success whatever the harness "
        "exited with"
    )
    # A pipe into a formatter is fine; a pipe into something that swallows the code
    # is not, and `tee` is the shape that does it by accident.
    assert not re.search(r"\|\s*tee\b", body), (
        "the run block pipes into `tee`, which without `set -o pipefail` reports the "
        "exit code of the last command in the pipeline rather than the harness's"
    )


def check_runs_every_mutation(whole: str) -> None:
    """It runs the harness, sharded, and the shard count is real."""
    body = job(whole)
    assert re.search(r"tests/mutation_check\.py", body), (
        f"the {JOB} job does not run tests/mutation_check.py; the whole point of the "
        "job is that this file is executed by CI"
    )
    shards = re.search(r"^\s*shard:\s*\[([0-9,\s]+)\]\s*$", body, re.M)
    assert shards, "the job has no `shard:` matrix axis, so one leg runs everything serially"
    indices = [int(i) for i in shards.group(1).split(",") if i.strip()]
    counts = re.search(r"^\s*shards:\s*\[(\d+)\]\s*$", body, re.M)
    assert counts, "the job has no `shards:` axis"
    assert indices == list(range(int(counts.group(1)))), (
        f"the matrix enumerates shards {indices} but declares {counts.group(1)} of "
        "them; the legs that are missing are the mutations nobody judges"
    )
    assert len(indices) > 1, "one shard is the serial run this job exists to avoid"
    assert re.search(r"--shard=\$\{\{\s*matrix\.shard\s*\}\}/\$\{\{\s*matrix\.shards\s*\}\}", body), (
        "the run block does not pass --shard=${{ matrix.shard }}/${{ matrix.shards }}, so "
        "every leg judges the same mutations"
    )


def check_not_in_front_of_the_suite(whole: str) -> None:
    """The ordinary per-push path must not wait for the harness."""
    body = job(whole)
    assert not re.search(r"^\s*needs:", body, re.M), (
        f"the {JOB} job declares `needs:`, which would make it a precondition for "
        "whatever depends on it rather than a parallel signal"
    )
    for name, other in jobs(whole).items():
        if name == JOB:
            continue
        assert not re.search(rf"^\s*needs:.*\b{re.escape(JOB)}\b", other, re.M), (
            f"the {name!r} job depends on {JOB}, so the per-push suite waits for a "
            "multi-minute mutation run"
        )


def check_judges_the_same_suite_ci_does(whole: str) -> None:
    """`-m "not network"` in both, or the two drift and CI opens a socket.

    The harness used to run `pytest ... tests` with no marker filter, so a pass
    hit live nflverse data once per mutation. That is a network dependency on an
    upstream schema for every verdict, and it is why a pass used to cost 76s a
    run instead of 40s. The ordinary pytest step here has always carried the
    marker; these two lines are the check that they stay the same line.
    """
    harness = HARNESS.read_text()
    marked = re.findall(r'"-m",\s*"not network"', harness)
    assert marked, (
        f"{HARNESS.name} runs pytest without `-m \"not network\"`, so the harness "
        "reconciles against live upstream data once per mutation: a network "
        "dependency in every verdict, and a baseline that can go red because "
        "somebody else's schema moved"
    )
    assert "not network" in whole, (
        "the workflow's own pytest step has lost its `-m \"not network\"`, so the "
        "ordinary suite and the harness now judge different suites"
    )


CHECKS = {
    "trigger": check_trigger,
    "not_condition_gated": check_not_condition_gated,
    "cannot_report_green": check_cannot_report_green_on_failure,
    "runs_every_mutation": check_runs_every_mutation,
    "not_in_front_of_the_suite": check_not_in_front_of_the_suite,
    "same_suite": check_judges_the_same_suite_ci_does,
}


# -------------------------------------------------------------------- tests


def test_the_workflow_file_exists():
    assert WORKFLOW.is_file(), WORKFLOW
    assert WORKFLOW.read_text(), "the workflow is empty"


def test_each_check_can_fail():
    """A guard that cannot fail is not a guard.

    Each check runs against the real file and against a copy with one thing
    broken. Every mutation below must be a string appearing exactly once, or it
    is not breaking the thing it claims to.
    """
    good = text()
    for check in CHECKS.values():
        check(good)   # the real workflow satisfies it
    broken = {
        "trigger": ("  pull_request:", "  pull_request_DISABLED:"),
        # Anchored on the job key, not `runs-on:`, which is in all three jobs --
        # breaking one of the three would leave the check satisfied by the other
        # two, which is exactly what the count assertion exists to prevent.
        "not_condition_gated": (f"  {JOB}:\n", f"  {JOB}:\n    if: false\n"),
        "cannot_report_green": (f"  {JOB}:", f"  {JOB}:\n    continue-on-error: true"),
        "runs_every_mutation": ("--shard=${{ matrix.shard }}/${{ matrix.shards }}", "--all"),
        "not_in_front_of_the_suite": (f"  {JOB}:", f"  {JOB}:\n    needs: pytest"),
        "same_suite": ('"-m", "not network"', '"-m", "network"'),
    }
    for name, (old, new) in broken.items():
        haystack = HARNESS.read_text() if name == "same_suite" else good
        assert haystack.count(old) == 1, (
            f"the {name!r} mutation {old!r} appears {haystack.count(old)} times in "
            "mutation_check.py" if name == "same_suite" else
            f"the {name!r} mutation {old!r} appears {haystack.count(old)} times, so "
            "breaking one copy leaves the check satisfied by another"
        )
        try:
            if name == "same_suite":
                original = HARNESS.read_text()
                try:
                    HARNESS.write_text(original.replace(old, new, 1))
                    CHECKS[name](good)
                finally:
                    HARNESS.write_text(original)
            else:
                CHECKS[name](good.replace(old, new, 1))
        except AssertionError:
            continue
        raise AssertionError(f"the {name!r} check passed a workflow with {old!r} broken")


def test_the_harness_job_is_there():
    """The subject of the file exists, and it runs the harness."""
    body = job(text())
    assert "tests/mutation_check.py" in body
    assert "runs-on:" in body


def test_the_harness_job_runs_on_every_push():
    check_trigger(text())


def test_the_harness_job_cannot_be_skipped_or_go_green_on_failure():
    check_not_condition_gated(text())
    check_cannot_report_green_on_failure(text())


def test_the_harness_job_runs_every_mutation_and_nothing_waits_for_it():
    check_runs_every_mutation(text())
    check_not_in_front_of_the_suite(text())


def test_the_harness_judges_the_same_suite_ci_does():
    check_judges_the_same_suite_ci_does(text())


# Read at call time, not at import time. A module-level constant made a removed
# `mutation-harness` job a *collection error*, which interrupts the whole
# `pytest tests/` session rather than failing one test -- so the loud version also
# took down every other result with it. A guard should cost you the guard.
def shard_count(whole: str) -> int:
    found = re.search(r"^\s*shards:\s*\[(\d+)\]\s*$", job(whole), re.M)
    assert found, f"the {JOB} job has no `shards:` matrix axis"
    return int(found.group(1))


def test_the_workflow_declares_more_than_one_shard():
    count = shard_count(text())
    assert count > 1, f"one shard is the serial run this job exists to avoid; found {count}"


def test_every_shard_judges_its_slice_and_every_canary():
    shards = shard_count(text())
    for index in range(shards):
        chosen = {m.ident for m in select(["mutation_check", f"--shard={index}/{shards}"])}
        assert chosen, f"shard {index}/{shards} selects nothing; it would report success having judged nothing"
        # The canaries, in every shard. `main` scores them with `all(...)`, which is
        # True over an empty sequence, so a shard holding none prints
        # `0 canary (all correct: True)` -- the verdict that means nothing, from the
        # one piece of machinery here whose whole job is to mean something.
        canaries = {m.ident for m in MUTATIONS if m.canary}
        assert canaries <= chosen, (
            f"shard {index}/{shards} is missing canaries {sorted(canaries - chosen)}; it "
            "would report its detector as verified without ever running it"
        )


def test_the_shards_between_them_judge_every_mutation_exactly_once():
    """Coverage is the property that matters, and a round-robin can rot.

    Not checked in the workflow and not checkable there: the workflow only knows
    how many legs there are. Adding a mutation to MUTATIONS lands it in exactly
    one shard automatically -- which is the reason the shards are `MUTATIONS[i::n]`
    and not a list of ids in YAML, where a new mutation would be left out of all
    of them and nobody would ever judge it.
    """
    shards = shard_count(text())
    canary_ids = {m.ident for m in MUTATIONS if m.canary}
    seen: list[str] = []
    for index in range(shards):
        seen.extend(m.ident for m in select(["mutation_check", f"--shard={index}/{shards}"]))
    # Exactly once, except the canaries, which every shard re-runs on purpose.
    non_canary = [i for i in seen if i not in canary_ids]
    assert sorted(non_canary) == sorted(m.ident for m in MUTATIONS if m.ident not in canary_ids), (
        "the shards do not partition the non-canary mutations; something is judged "
        "twice or not at all"
    )
    assert len(non_canary) == len(set(non_canary)), "a mutation is in two shards"


def test_a_single_shard_is_the_whole_list():
    """`--shard=0/1` is the serial run, so it must not quietly drop anything."""
    assert [m.ident for m in select(["mutation_check", "--shard=0/1"])] == [
        m.ident for m in MUTATIONS
    ]


def test_a_bad_shard_is_refused_rather_than_run_as_everything():
    for bad in ("--shard=2/2", "--shard=0/0", "--shard=-1/3"):
        with pytest.raises(SystemExit):
            select(["mutation_check", bad])
    with pytest.raises(SystemExit):
        # Both at once is a mistake, not a union: a run that judged "one slice of
        # every mutation OR some by name" is neither.
        select(["mutation_check", "--shard=0/2", "M4"])

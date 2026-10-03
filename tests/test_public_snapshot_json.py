"""A snapshot holding NaN is invalid JSON on disk, whoever reads it.

json.dumps defaults to allow_nan=True, so a single non-finite float used to
be written as a bare NaN token. The public deployment serves that file
verbatim, and starlette renders responses with allow_nan=False -- so the
endpoint 500s rather than returning a body the browser can fail to parse.

**Why the file on disk is read at all, and why its absence is not a skip.**
`data/public_snapshot.json` is committed, and since PR #29 it is load-bearing:
`public_snapshot.assert_publishable` refuses to publish a snapshot whose
`generated_at` is older than the manifest's `trained_at`. This test was the
strictest reader of that file, so it was the worst place to answer "the file is
not here" with a skip -- a deleted artefact would have left the suite green. It
now goes through `tracked_artifacts.require`, which fails when a committed file
is absent and skips only when neither the index nor any commit carries it. Two
further tests read the same artefact and already failed loudly on its absence
(`test_snapshot_staleness_gate.py`, `test_snapshot_shape_reconciliation.py`), so
before this change a deletion produced a green skip and a red failure in the
same run.
"""

import json
import math

import pytest

from nfl_predictor.public_snapshot import sanitize_floats
from tracked_artifacts import require

#: The artefact, named once. `require()` re-derives it from the path and refuses
#: if the two disagree, so this spelling cannot drift from `config`'s silently.
SNAPSHOT_REL = "data/public_snapshot.json"

#: Why the file has to be there. Quoted into the failure message so a reader who
#: has just been told their checkout is broken learns what they are missing.
WHY_REQUIRED = (
    "It is the artefact the public deployment serves verbatim, and "
    "public_snapshot.assert_publishable refuses to publish one older than the "
    "models it was built from, so it is the file this check exists to guard."
)

#: What would have to be true for its absence to be expected. Nothing in this
#: repository: the file is committed at `main`, so this branch is unreachable
#: here and says so rather than inventing a reason for it. It exists because
#: `require()` is shared, and a reason that merely named the path would be
#: indistinguishable from the skip this replaced.
WHY_ABSENT_IS_EXPECTED = (
    "Nothing in this repository predicts it: data/public_snapshot.json is "
    "committed at main. If you are reading this, your checkout is older than the "
    "file or was cloned without it -- update the branch rather than committing a "
    "test that tolerates its absence."
)


def test_nan_becomes_null():
    assert sanitize_floats({"home_cover_prob": float("nan")}) == {"home_cover_prob": None}


def test_infinities_become_null():
    assert sanitize_floats({"a": float("inf"), "b": float("-inf")}) == {"a": None, "b": None}


def test_walks_nested_structures():
    payload = {"weeks": {"9": {"predictions": {"g1": {"sigma": float("nan")}}}}}
    assert sanitize_floats(payload) == {"weeks": {"9": {"predictions": {"g1": {"sigma": None}}}}}


def test_leaves_real_numbers_and_types_alone():
    payload = {"p": 0.5, "n": 3, "s": "x", "b": True, "none": None, "empty": [], "d": {}}
    assert sanitize_floats(payload) == payload


def test_sanitised_snapshot_survives_a_strict_dump():
    """The regression guard: allow_nan=False is what makes a future NaN loud."""
    snapshot = sanitize_floats({"predictions": {"g1": {"over_prob": float("nan")}}})
    text = json.dumps(snapshot, indent=2, allow_nan=False)
    assert json.loads(text)["predictions"]["g1"]["over_prob"] is None


def test_the_committed_snapshot_is_strict_json():
    """The file on disk must parse with allow_nan=False, not just json.loads.

    `require()` rather than `if not path.exists(): pytest.skip(...)`. A committed
    artefact that has gone missing is a defect in the tree, not a fact about the
    environment, and reporting it as the second is what let a deleted
    `data/public_snapshot.json` pass as a green skip.
    """
    from nfl_predictor.config import PUBLIC_SNAPSHOT_PATH

    path = require(
        PUBLIC_SNAPSHOT_PATH,
        rel=SNAPSHOT_REL,
        required_because=WHY_REQUIRED,
        expected_absent_because=WHY_ABSENT_IS_EXPECTED,
    )
    text = path.read_text()
    snapshot = json.loads(text, parse_constant=_reject)  # parse_constant fires on NaN/Infinity

    # Not decoration. `json.loads` accepts `{}`, `null` and `[]`, so without this
    # the test would pass on a file that is valid JSON and not a snapshot -- and
    # the claim being made is about a snapshot. Cheap, and it makes "the file
    # parsed" mean "the artefact parsed".
    assert is_a_snapshot(snapshot), (
        f"{SNAPSHOT_REL} parsed to something that is not a snapshot: "
        f"{type(snapshot).__name__} with keys "
        f"{sorted(snapshot) if isinstance(snapshot, dict) else snapshot!r}. A file "
        f"this empty satisfies the NaN check while proving nothing about it."
    )


def test_the_reject_hook_itself_fires(tmp_path):
    """`_reject` is the guard, and nothing else in the suite calls it.

    Without this the hook could be miswired -- `parse_constant` misspelled, or
    the constant named something json never passes -- and
    `test_the_committed_snapshot_is_strict_json` would go on passing forever,
    because a file with no NaN in it cannot tell a working hook from a broken
    one. Each spelling json actually emits a constant for is checked, written to
    disk first so the bytes are the same ones the real test reads.
    """
    for token in ("NaN", "Infinity", "-Infinity"):
        path = tmp_path / "snapshot.json"
        path.write_text('{"weeks": {"3": {"predictions": {"g1": {"p": '
                        f'{token}}}}}}}\n', encoding="utf-8")
        with pytest.raises(AssertionError) as excinfo:
            json.loads(path.read_text(), parse_constant=_reject)
        assert token in str(excinfo.value)


def test_is_a_snapshot_rejects_the_vacuous_payloads():
    """The counterpart: the vacuous payloads the shape check exists to catch.

    `{}`, `null`, `[]` and `{"weeks": {}}` all parse cleanly under
    `parse_constant=_reject`. Left unchecked, any of them would make the
    strict-JSON test pass while it examined nothing -- which is the failure this
    change is about, in a different costume.

    Written against `is_a_snapshot` rather than against a copy of its condition:
    a test that re-implements the check it is checking is a test that keeps
    passing when the check is weakened, which is the one thing it was added to
    prevent. So if the shape condition is relaxed, this goes red.

    The payloads go through `json.loads` with `_reject` rather than being handed
    to `is_a_snapshot` as literals, so the counter-test also covers the fact that
    all of them really do parse -- if one stopped parsing it would be a different
    test, and one worth failing.
    """
    for payload in ("{}", "null", "[]", '{"weeks": {}}', '{"weeks": null}', '"text"'):
        assert not is_a_snapshot(json.loads(payload, parse_constant=_reject)), (
            f"{payload} satisfied the artefact check"
        )


def test_is_a_snapshot_accepts_a_real_one():
    """The other direction, so the check cannot be passed by rejecting everything."""
    assert is_a_snapshot({"weeks": {"3": {"games": [], "player_props": []}}})


def is_a_snapshot(payload) -> bool:
    """Is this parsed payload a snapshot, rather than any JSON that happens to parse?

    A named function so the strict-JSON test and its own counter-test agree by
    construction. Inlined in both places it would be two conditions that could
    drift, and the counter-test would be checking its own copy rather than the
    check.

    `weeks` is the requirement because that is the key the two consumers read:
    `routes._public_snapshot_cache["weeks"]` and every
    `/api/players/{season}/{week}/props` request built on it. A payload with no
    `weeks` is not a snapshot in any sense this suite cares about, whatever else
    it contains.
    """
    return isinstance(payload, dict) and bool(payload.get("weeks"))


def _reject(name):
    raise AssertionError(f"snapshot contains a bare {name} token")

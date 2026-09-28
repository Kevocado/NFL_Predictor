"""A reused snapshot week must not be able to silently miss a newly added field.

`build_snapshot()` copies weeks outside the rebuild window verbatim. That is the
right call for cost and the wrong call for shape: a copied week cannot pick up a
field the code has started emitting, so a new field lands on the few weeks
inside the window and nowhere else.

This is not hypothetical. `is_starter` and `depth_slot` were added to the prop
rows (routes.py:509-510) while every week carrying props sat outside the window.
The committed snapshot is `current_week` 3, so with `REBUILD_WEEKS_BEHIND = 1`
and `REBUILD_WEEKS_AHEAD = 4` the window is weeks 2-7 -- and weeks 2-7 have no
games yet, while week 1 and weeks 8-18 are copies:

    week  1   928 props   weeks  2-7  no games   week  8   804 props
    week  9   862 props   week 10   806 props   ...          week 18  920 props

so the two new fields reached the artifact *nowhere*: 0 of its 10,416 prop rows
carry them. The symptom is a frontend looking for a field the API is supposed to
serve and finding it missing on the whole season -- and the obvious conclusion,
"the serialization is broken", is wrong. It is the copy, not the writer.

The fix: work out the prop-row shape the CURRENT code produces and rebuild any
reused week that does not match it. Narrow on purpose -- only weeks whose row
shape actually changed are rebuilt, so adding a field costs one build rather than
all twenty-two, and a week carrying fields the code no longer emits is left
alone rather than caught in a rebuild loop.

Two things the first version of that fix got wrong, which is what the tests
below exist to hold down. This module is the SECOND repo to carry the defect;
CFB_Predictor found it here and fixed it there first.

* **"Shape" is not one row's key set.** `predict_props` keys off
  `POSITION_MARKETS` (models/player_props.py:53), so prop rows are
  position-heterogeneous on purpose: a QB row carries `passing_yards`, an RB row
  `rushing_yards`, a WR/TE row `receiving_yards`. Every prop week of the
  committed artifact has three distinct row shapes for exactly that reason (four
  positions, three shapes, because WR and TE carry the same markets). The
  required keys are therefore the position-INVARIANT ones: every key
  `POSITION_MARKETS` can introduce is subtracted by name, and what is left is
  intersected across the sample. *Every* row is checked against that, not row 0.
* **Every test week in the earlier version of this file was single-shape**, which
  is exactly why the row-0 predicate passed review. `_row()` below derives its
  market keys from the real `POSITION_MARKETS`, and
  `TestRefreshingTheCommittedArtifact` runs the committed artifact, so both
  failure directions are reachable: a week whose row 0 is current while the rest
  are stale, and a week that is current on every row but whose row 0 is a
  different position from the sample's.

Nothing in the artifact class hardcodes a week number, a row count or a
`current_week`: `.github/workflows/refresh-public-snapshot.yml` pushes the
refreshed artifact on its own several times a day and `tests.yml` runs this file
on every push to `main` with no exclusion, so a literal about today's artifact
would red `main` on a bot commit. See the class docstring.

The `data/public_snapshot.json` tests read the committed artifact; they do not
write it, and `build_snapshot` never does either -- they assert on the week list
it was asked to build.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from live_upstream import (
    FORBIDDEN_CALLS,
    FORBIDDEN_ROUTES_CALLS,
    FORBIDDEN_UPSTREAM,
    OFFLINE_BUILDERS,
    _forbidden,
    assert_no_live_upstream,
    install_builder_guard,
)
from nfl_predictor import config
from nfl_predictor import public_snapshot as ps
from nfl_predictor.models import player_props


# --------------------------------------------------------------------------- #
# No test in this file may reach a live builder.
#
# `build_snapshot` calls four live builders unconditionally and wraps every one
# of them in a bare `except Exception`, so a test that stubs only
# `_get_standings_live` gets a green run *and* a live call. A raise alone is not
# enough here, precisely because `build_snapshot` swallows every one of these:
# the fixture records each forbidden call and asserts at teardown that none was
# hit.
#
# Suite-wide, the network boundary itself -- the data modules' fetch functions,
# as `routes` sees them -- is already refused for every test by
# `tests/conftest.py`. This layer is the one below it, and it is file-local on
# purpose: the four live builders and the per-week calls are real code under
# test elsewhere in this suite (`tests/test_hub_routes.py` drives
# `_get_hub_teams_live` deliberately, `tests/test_api_routes.py` drives
# `_get_games_live` through a TestClient), so stubbing or forbidding them
# suite-wide would delete coverage that exists today. Measured: a blanket
# conftest guard broke 14 tests across 3 files.
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_live_builder(monkeypatch):
    FORBIDDEN_CALLS.clear()
    install_builder_guard(monkeypatch)
    yield
    assert_no_live_upstream()


def _week(*prop_rows: dict, games: list | None = None) -> dict:
    return {"games": games or [], "predictions": {}, "player_props": list(prop_rows)}


def _marks_on(*owners: Any) -> list[Any]:
    """The `pytestmark`s on `owners`, tolerating any of them being `None`.

    pytest collects marks from three places and stores them differently: a
    decorator on a function lands on the function, while a class-level or
    module-level `pytestmark` is read off the class or module. A guard that
    looks only at the function cannot see the other two.
    """
    return [mark for owner in owners if owner is not None for mark in getattr(owner, "pytestmark", [])]


def _tests_in(namespace: dict, module_obj: Any, module_name: str) -> list[tuple[str, tuple[Any, ...]]]:
    """`(name, owners)` for every test in `namespace`, where `owners` is the
    chain whose marks apply to it: the function, the function and its class,
    both plus the module -- in the order `_marks_on` reads them."""
    found: list[tuple[str, tuple[Any, ...]]] = [
        (name, (obj, None, module_obj))
        for name, obj in namespace.items()
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    for cls in list(namespace.values()):
        if inspect.isclass(cls) and cls.__module__ == module_name:
            found += [
                (f"{cls.__name__}.{name}", (member, cls, module_obj))
                for name, member in vars(cls).items()
                if name.startswith("test_") and inspect.isfunction(member)
            ]
    return found


def _tests_in_this_file() -> list[tuple[str, tuple[Any, ...]]]:
    module = sys.modules[__name__]
    return _tests_in(vars(module), module, module.__name__)


def _marked_network(owners: tuple[Any, ...]) -> bool:
    return any(mark.name == "network" for mark in _marks_on(*owners))


# The row shape before the depth-chart fields and after. `is_starter`/
# `depth_slot` are the two the code added to every row.
OLD_ROW = {
    "player_id": "p", "player_name": "Tua", "position": "QB",
    "recent_team": "MIA", "anytime_td_prob": 0.4, "passing_yards": 250.0,
}
NEW_ROW = {**OLD_ROW, "is_starter": True, "depth_slot": 8}

# The keys `routes._get_player_props_live` writes on every prop row
# (routes.py:502-512) plus the one key `predict_props` always sets. Transcribed
# rather than derived, on purpose: the derivation is checked against it.
CURRENT_ROW_KEYS = frozenset({
    "player_id", "player_name", "recent_team", "position", "anytime_td_prob",
    "is_starter", "depth_slot",
})


def _row(position: str, *, current: bool = True) -> dict:
    """One prop row as the current code emits it for `position`.

    The market keys come from the real `POSITION_MARKETS`, so a row is
    position-heterogeneous by construction -- the thing a row-0 signature could
    not see. `current=False` is the pre-depth-chart row: same position, same
    markets, no `is_starter`/`depth_slot`.

    `carries` and `receptions` have no trained model yet, so today's code skips
    them (models/player_props.py:54) and each prop week of the committed
    artifact has one market per position. This fixture includes them anyway,
    which is what the code would emit once they are trained -- and that does NOT
    make the shape count go to four: a row is one shape per position however
    many markets that position has, so QB, RB and WR/TE still account for three
    shapes between the four positions (WR and TE carry the same markets). What
    it widens is the spread of keys *within* each position, which is the case a
    position-blind check cannot survive, and
    `test_a_single_position_sample_demands_nothing_of_another_position` is the
    test that actually pins it.
    """
    row = {
        "player_id": f"p-{position}", "player_name": "A. Back", "recent_team": "MIA",
        "position": position, "anytime_td_prob": 0.4,
    }
    if current:
        row["is_starter"] = None
        row["depth_slot"] = None
    for market in player_props.POSITION_MARKETS[position]:
        row[market] = 1.0
    return row


def _signature() -> frozenset[str]:
    """The required keys for a week that carries every modelled position.

    A real week is multi-position -- `_get_player_props_live` returns every
    player on every active team -- so this is a realistic sample, but the
    signature it returns is the same for every sample. See
    `test_the_position_invariant_subtraction_covers_a_single_position_sample`
    for what makes that true.
    """
    return ps._position_invariant_keys([_row(p) for p in player_props.POSITION_MARKETS])


def _pin_window(monkeypatch, *, current_week: int = 3, max_week: int = 3) -> None:
    """Pin the rebuild window to a single week, so a test states which weeks are
    reused rather than depending on the default -1/+4."""
    monkeypatch.setattr(ps, "MAX_WEEK", max_week)
    monkeypatch.setattr(ps, "REBUILD_WEEKS_BEHIND", 0)
    monkeypatch.setattr(ps, "REBUILD_WEEKS_AHEAD", 0)
    monkeypatch.setattr(ps.routes, "current_season_and_week", lambda: (2026, current_week))


class TestMismatchDetection:
    def test_a_week_missing_the_new_field_is_a_mismatch(self):
        assert ps._prop_shape_mismatch(_week(OLD_ROW), frozenset(NEW_ROW.keys())) is True

    def test_a_week_already_carrying_it_is_not(self):
        assert ps._prop_shape_mismatch(_week(NEW_ROW), frozenset(NEW_ROW.keys())) is False

    def test_a_superset_is_not_a_mismatch(self):
        # Fields the current code no longer emits must not force a rebuild
        # loop; the signature is a subset test on purpose.
        wider = {**NEW_ROW, "retired_field": 1}
        assert ps._prop_shape_mismatch(_week(wider), frozenset(NEW_ROW.keys())) is False

    def test_a_propless_week_is_never_a_mismatch(self):
        # No rows to be stale. Rebuilding it would be pure cost.
        assert ps._prop_shape_mismatch(_week(), frozenset(NEW_ROW.keys())) is False
        assert ps._prop_shape_mismatch({"player_props": []}, frozenset(NEW_ROW.keys())) is False

    def test_the_signature_is_the_position_invariant_keys_not_one_rows_shape(self):
        assert _signature() == CURRENT_ROW_KEYS
        # A single row's shape is NOT the shape, which is the whole point: a QB
        # row carries `passing_yards` and an RB row carries `rushing_yards`.
        assert frozenset(_row("QB").keys()) != _signature()
        assert _signature().isdisjoint(
            {market for markets in player_props.POSITION_MARKETS.values() for market in markets}
        )

    def test_the_position_invariant_subtraction_covers_a_single_position_sample(self):
        """The property a plain intersection does not give.

        "A real week is always multi-position" is true of the sample, and it is
        NOT a property of the code: nothing makes it true, and within one
        position the market set is already narrower (models/player_props.py:54
        emits a market only `if market in models`, so an RB row can be
        `rushing_yards` with no `carries` on it while those models are missing).
        A sample of one position is therefore reachable -- an all-QB rebuild
        window is the obvious case -- and the plain intersection would then
        demand that position's market of every row in the season.
        """
        all_qb = [_row("QB") for _ in range(5)]
        all_wr = [_row("WR") for _ in range(5)]
        # The raw samples are not position-invariant...
        assert frozenset(all_qb[0].keys()) & {"passing_yards"}
        assert frozenset(all_wr[0].keys()) & {"receiving_yards", "receptions"}
        # ...and the subtraction is what makes the signature so. Unconditionally,
        # before any other row is looked at.
        qb_signature = ps._position_invariant_keys(all_qb)
        wr_signature = ps._position_invariant_keys(all_wr)
        assert qb_signature == CURRENT_ROW_KEYS
        assert wr_signature == CURRENT_ROW_KEYS
        assert "passing_yards" not in qb_signature
        assert wr_signature.isdisjoint({"receiving_yards", "receptions"})
        # One row, too -- the degenerate version of the same thing.
        assert ps._position_invariant_keys([_row("QB")]) == CURRENT_ROW_KEYS
        assert ps._position_invariant_keys([_row("WR")]) == CURRENT_ROW_KEYS

        # And through the predicate, both directions: neither a week of WR rows
        # under an all-QB sample nor a week of QB rows under an all-WR sample is
        # stale, and the row-0 and plain-intersection versions failed both.
        assert ps._prop_shape_mismatch(_week(*all_wr, _row("QB")), qb_signature) is False
        assert ps._prop_shape_mismatch(_week(*all_qb, _row("WR")), wr_signature) is False

    def test_a_single_position_sample_demands_nothing_of_another_position(self):
        """A week of WR rows is not rebuilt for lacking `rushing_yards`.

        The signature comes from an all-RB sample, so a plain intersection puts
        `rushing_yards` in it -- and not one WR row carries that key, ever. The
        subtraction is what stops the demand, and the last two lines are the
        residual a simplifier would reintroduce: they are why the intersection is
        kept as defence in depth and is NOT the mechanism.
        """
        rb_signature = ps._position_invariant_keys([_row("RB") for _ in range(5)])
        assert "rushing_yards" not in rb_signature
        assert ps._prop_shape_mismatch(_week(*[_row("WR") for _ in range(5)]), rb_signature) is False

        # The plain intersection -- CFB's first fix, and the version this repo
        # would fall back to if the subtraction were reverted -- does put the
        # market in, and does then demand it of the WR week.
        plain = frozenset(_row("RB").keys())
        assert "rushing_yards" in plain
        assert ps._prop_shape_mismatch(_week(*[_row("WR") for _ in range(5)]), plain) is True

    def test_a_single_position_sample_reaches_build_snapshot_and_still_stays_out(self, monkeypatch):
        """The same property through the real entry point, where the sample is
        whatever the rebuild window happened to produce."""
        previous = {
            "season": 2026,
            "weeks": {
                "1": _week(_row("WR"), _row("WR"), _row("WR")),  # reused, current
                "2": _week(NEW_ROW),                                # reused, current
                "3": _week(_row("QB"), _row("QB")),                # in the window: all QB
            },
        }
        built: list[int] = []

        def fake_build_week(season, week: int) -> dict:
            built.append(week)
            return _week(_row("QB"), _row("QB"))

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        _pin_window(monkeypatch)

        ps.build_snapshot(previous)

        # The signature came from an all-QB window. Under the row-0 and
        # plain-intersection versions, week 1 would have been rebuilt here for
        # lacking `passing_yards` -- a market its WR rows were never going to
        # carry, every scheduled run, forever.
        assert 1 not in built
        assert 2 not in built

    def test_the_signature_still_ignores_a_row_0_only_new_field(self):
        # Backstop for the intersection half, which is defence in depth: an
        # unknown key present only on row 0 must not become required either.
        rows = [_row("RB", current=False)]
        rows[0]["brand_new_field"] = 1  # not in POSITION_MARKETS, so not subtracted
        assert ps._position_invariant_keys(rows) == (
            (CURRENT_ROW_KEYS - {"is_starter", "depth_slot"}) | {"brand_new_field"}
        )
        # ...and the same sample used as a reused week is still judged on the
        # invariant keys, which it is missing.
        assert ps._prop_shape_mismatch(_week(rows[0], _row("RB")), _signature()) is True

    def test_a_row_past_row_zero_being_stale_makes_the_week_stale(self):
        # Row 0 current, rows 1-2 stale. The row-0 version of this predicate
        # called that week current and never rebuilt it, and nothing in the
        # shipped artifact would have shown it: all 12 of its prop weeks are
        # stale on *every* row, so row 0 was stale too and the two versions
        # agreed by coincidence. A week one field behind is what the next change
        # to this row shape actually looks like.
        week = _week(_row("QB"), _row("WR", current=False), _row("RB", current=False))
        stale_rows = [i for i, row in enumerate(week["player_props"]) if not _signature() <= row.keys()]
        assert stale_rows == [1, 2]
        assert ps._prop_shape_mismatch(week, _signature()) is True

    @pytest.mark.parametrize("row_zero", ["QB", "RB", "WR", "TE"])
    def test_a_current_week_is_current_whatever_position_row_zero_is(self, row_zero):
        # The mirror case: a week that is current on every row used to be judged
        # by row 0 alone, so a WR row 0 read against a QB-derived signature said
        # "stale" on a week that needed nothing.
        week = _week(_row(row_zero), _row("QB"), _row("RB"), _row("WR"))
        assert ps._prop_shape_mismatch(week, _signature()) is False

    def test_a_row_missing_one_of_the_two_new_keys_past_row_zero_is_caught(self):
        # A field written as None can go missing on one row alone -- a partial
        # write, a hand-merged snapshot. The check is per row, so one row is
        # enough, and it is not row 0.
        damaged = _row("WR")
        del damaged["depth_slot"]
        assert ps._prop_shape_mismatch(_week(_row("QB"), _row("RB"), damaged), _signature()) is True

    def test_a_row_that_lost_its_position_specific_market_is_not_a_mismatch(self):
        # A market key follows the position, so a WR row without
        # `receiving_yards` is a player whose position moved between builds --
        # not a stale shape. Demanding that key anyway is the over-rebuild bug in
        # its pure form: a permanent rebuild of a week that is already current.
        damaged = _row("WR")
        damaged.pop("receiving_yards")
        assert ps._prop_shape_mismatch(_week(_row("QB"), _row("RB"), damaged), _signature()) is False


class TestSignature:
    # CHANGED, not extended: the two tests below used to assert
    # `_prop_key_signature(...) == frozenset(NEW_ROW.keys())`. `NEW_ROW` is
    # derived from `OLD_ROW` at module level, and `OLD_ROW` is a QB row, so that
    # expectation was the signature of ONE row including its market key
    # `passing_yards` -- i.e. the row-0 semantics this file exists to hold down,
    # asserted as though it were the invariant one. They now assert
    # `CURRENT_ROW_KEYS`, which is what the function returns. The other change
    # is the week: `_week(NEW_ROW)` is a single-row week, so the sample is one
    # row, and the subtraction -- not the intersection -- is what makes that
    # sample come out position-invariant.
    #
    # This is the one pre-existing assertion this change had to touch, and it is
    # not a weakening: both still fail against the row-0 code for the right
    # reason, and `test_the_signature_is_the_position_invariant_keys_not_one_rows_shape`
    # pins the value they now expect.

    def test_prefers_a_week_that_was_just_rebuilt(self):
        weeks = {"1": _week(OLD_ROW), "3": _week(NEW_ROW)}
        # Week 3 was rebuilt, so the current shape is known for free.
        assert ps._prop_key_signature(2026, 3, weeks, ["1"]) == CURRENT_ROW_KEYS
        # Specifically: one rebuilt QB row does not put `passing_yards` in the
        # required set. Asserted separately from the equality above because it
        # is the part the old expectation got wrong.
        assert "passing_yards" not in ps._prop_key_signature(2026, 3, weeks, ["1"])

    def test_falls_back_to_one_live_probe_when_every_rebuilt_week_is_propless(self, monkeypatch):
        calls = []

        def fake(season, week):
            calls.append((season, week))
            return [NEW_ROW]

        monkeypatch.setattr(ps.routes, "_get_player_props_live", fake)
        weeks = {"1": _week(OLD_ROW), "2": _week(), "3": _week()}
        got = ps._prop_key_signature(2026, 3, weeks, ["1", "2"])
        assert got == CURRENT_ROW_KEYS
        assert calls == [(2026, 3)]  # exactly one probe, not one per week

    def test_returns_none_when_it_cannot_tell_rather_than_guessing(self, monkeypatch):
        def boom(season, week):
            raise RuntimeError("no models")

        monkeypatch.setattr(ps.routes, "_get_player_props_live", boom)
        # None is load-bearing. An empty frozenset would compare equal against
        # every prop-less week and report "nothing to do" -- the exact bug.
        assert ps._prop_key_signature(2026, 3, {"1": _week(), "2": _week()}, ["1", "2"]) is None

    def test_returns_none_when_the_probe_finds_no_props(self, monkeypatch):
        monkeypatch.setattr(ps.routes, "_get_player_props_live", lambda s, w: [])
        assert ps._prop_key_signature(2026, 3, {"1": _week()}, ["1"]) is None

    def test_it_intersects_every_rebuilt_week_not_just_the_first(self):
        # One rebuilt week that happens to be all-QBs and another all-WRs: the
        # pooled intersection drops both markets, so neither is then demanded of
        # every row in the season. This is the mitigation for a single-position
        # sample, and it costs nothing -- those weeks are already built.
        weeks = {"3": _week(_row("QB")), "4": _week(_row("WR"))}
        assert ps._prop_key_signature(2026, 3, weeks, []) == CURRENT_ROW_KEYS

    def test_the_probe_sample_is_intersected_too_not_just_its_first_row(self, monkeypatch):
        # The probe's row 0 is one arbitrary player, exactly like a rebuilt
        # week's row 0 -- and reading it whole is what made a current week look
        # stale forever.
        calls = []

        def fake(season, week):
            calls.append((season, week))
            return [_row("WR"), _row("QB")]

        monkeypatch.setattr(ps.routes, "_get_player_props_live", fake)
        got = ps._prop_key_signature(2026, 3, {"1": _week(), "2": _week()}, ["1", "2"])
        assert got == CURRENT_ROW_KEYS
        assert calls == [(2026, 3)]  # still exactly one probe

    def test_a_reused_week_is_never_the_source_even_when_it_is_the_only_one_with_props(self, monkeypatch):
        # The signature has to describe the CURRENT code. A reused week is a copy
        # of the past, so trusting it would ratify the very shape it exists to
        # check -- and the probe has to happen instead.
        calls = []

        def fake(season, week):
            calls.append((season, week))
            return [_row("WR"), _row("QB")]

        monkeypatch.setattr(ps.routes, "_get_player_props_live", fake)
        got = ps._prop_key_signature(2026, 3, {"1": _week(OLD_ROW)}, ["1"])
        assert got == CURRENT_ROW_KEYS
        assert calls == [(2026, 3)]


class TestBuildSnapshotReconciles:
    def test_a_stale_reused_week_is_rebuilt_and_a_current_one_is_not(self, monkeypatch):
        previous = {
            "season": 2026,
            "weeks": {
                "1": _week(OLD_ROW),      # reused, stale -> should rebuild
                "2": _week(NEW_ROW),      # reused, current -> should survive untouched
                "3": _week(NEW_ROW),      # in the rebuild window
            },
        }
        built: list[int] = []

        def fake_build_week(season, week: int) -> dict:
            built.append(week)
            return _week(NEW_ROW)

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        _pin_window(monkeypatch)

        result: dict[str, Any] = ps.build_snapshot(previous)

        # 3 is in the window; 1 is stale and gets rebuilt; 2 is already current
        # and must NOT be rebuilt -- that is the cost control.
        assert built.count(1) == 1
        assert 2 not in built
        assert frozenset(result["weeks"]["1"]["player_props"][0].keys()) == frozenset(NEW_ROW.keys())

    def test_an_undeterminable_shape_skips_reconciliation_rather_than_guessing(self, monkeypatch):
        previous = {
            "season": 2026,
            "weeks": {"1": _week(OLD_ROW), "2": _week(OLD_ROW), "3": _week(OLD_ROW)},
        }

        def fake_build_week(season, week: int) -> dict:
            return _week()  # every rebuilt week is prop-less

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        _pin_window(monkeypatch)

        def boom(season, week):
            raise RuntimeError("offline")

        monkeypatch.setattr(ps.routes, "_get_player_props_live", boom)

        result = ps.build_snapshot(previous)
        # Untouched rather than wrong: the old shape is still there, and the
        # build said so out loud instead of pretending it had reconciled.
        assert "is_starter" not in result["weeks"]["1"]["player_props"][0]

    def test_a_reused_week_stale_past_row_zero_is_rebuilt(self, monkeypatch):
        # The same defect as `test_a_row_past_row_zero_being_stale_makes_the_week_stale`,
        # seen where it costs something: 1 of 2 rows is stale, row 0 is not, and
        # the old predicate left the week alone -- so the other half of the rows
        # stayed stale for the life of the snapshot.
        previous = {
            "season": 2026,
            "weeks": {
                "1": _week(_row("QB"), _row("WR", current=False)),  # reused, row 1 stale
                "2": _week(NEW_ROW),                                  # reused, current
                "3": _week(_row("QB"), _row("WR"), _row("RB")),      # in the window
            },
        }
        built: list[int] = []

        def fake_build_week(season, week: int) -> dict:
            built.append(week)
            return _week(_row("QB"), _row("WR"), _row("RB"))

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        _pin_window(monkeypatch)

        ps.build_snapshot(previous)

        assert built.count(1) == 1
        assert 2 not in built  # the current control week is the cost control

    @pytest.mark.parametrize("row_zero", ["QB", "RB", "WR"])
    def test_a_current_reused_week_is_left_alone_whatever_row_zero_is(self, monkeypatch, row_zero):
        # And the mirror, end to end. The rebuilt week is multi-position on
        # purpose, so the test would still have passed under an intersection-only
        # signature; the single-position case is
        # `test_a_single_position_sample_reaches_build_snapshot_and_still_stays_out`.
        previous = {
            "season": 2026,
            "weeks": {
                "1": _week(_row(row_zero), _row("QB"), _row("RB"), _row("WR")),  # current throughout
                "2": _week(NEW_ROW),
                "3": _week(_row("QB"), _row("WR"), _row("RB")),                 # in the window
            },
        }
        built: list[int] = []

        def fake_build_week(season, week: int) -> dict:
            built.append(week)
            return _week(_row("QB"), _row("WR"), _row("RB"))

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        _pin_window(monkeypatch)

        ps.build_snapshot(previous)

        assert 1 not in built  # row 0's position must not decide this
        assert 2 not in built


# --------------------------------------------------------------- shipped file


def _shipped() -> dict:
    return json.loads(Path(config.PUBLIC_SNAPSHOT_PATH).read_text())


def _carrying_the_current_keys(week: dict) -> dict:
    """A week whose every prop row carries the keys the current code writes.

    The committed artifact predates the depth-chart fields -- 0 of its 10,416
    prop rows carry `is_starter`, measured 2026-09-28 -- because the reconciliation
    could not run until a rebuilt week held props, and the rebuild window holds
    none. This is what a successful reconciliation leaves behind, and it is the
    state the artifact will be in the first time a window week carries props, so
    it is the state in which "a refresh rebuilds nothing" is the claim worth
    making. Existing keys are left alone, so this keeps working once the bot
    refreshes the artifact past the field.
    """
    return {
        **week,
        "player_props": [
            {**row, **{k: None for k in CURRENT_ROW_KEYS if k not in row}}
            for row in week["player_props"]
        ],
    }


def _reconciled(shipped: dict) -> dict:
    return {**shipped, "weeks": {key: _carrying_the_current_keys(week) for key, week in shipped["weeks"].items()}}


def _without(week: dict, *keys: str) -> dict:
    """A week with `keys` removed from every prop row."""
    return {
        **week,
        "player_props": [{k: v for k, v in row.items() if k not in keys} for row in week["player_props"]],
    }


def _damage_last_row(week: dict, key: str) -> dict:
    """A week whose LAST prop row lost `key` -- the case a row-0 check cannot
    see, at artifact scale (hundreds of rows in)."""
    rows = list(week["player_props"])
    rows[-1] = {k: v for k, v in rows[-1].items() if k != key}
    return {**week, "player_props": rows}


def _window(current_week: int) -> list[str]:
    return [
        str(week) for week in range(
            max(1, current_week - ps.REBUILD_WEEKS_BEHIND),
            min(ps.MAX_WEEK, current_week + ps.REBUILD_WEEKS_AHEAD) + 1,
        )
    ]


def _live_sample(first_position: str) -> list[dict]:
    """What one live probe can return: every modelled position, `first` first.

    Which position the sample's FIRST row happens to be is arbitrary, and the
    whole class below is parametrized on it, because the row-0 predicate's
    verdict depended on exactly that.
    """
    rows = [_row(p) for p in player_props.POSITION_MARKETS]
    return [r for r in rows if r["position"] == first_position] + [
        r for r in rows if r["position"] != first_position
    ]


def _refresh(shipped: dict, monkeypatch, sample: list[dict] | None = None) -> list[str]:
    """Feed a snapshot back through `build_snapshot` and record the weeks that
    got rebuilt. `_build_week` returns that snapshot's own week, so anything
    built outside the window was rebuilt for nothing -- which is what
    "idempotent" has to mean here, and what the over-rebuild failure was not."""
    built: list[str] = []

    def fake_build_week(season, week: int) -> dict:
        built.append(str(week))
        return shipped["weeks"][str(week)]

    monkeypatch.setattr(ps, "_build_week", fake_build_week)
    monkeypatch.setattr(
        ps.routes, "current_season_and_week",
        lambda: (shipped["season"], shipped["current_week"]),
    )
    if sample is not None:
        # The one live probe the signature is allowed to make, taken only when
        # every rebuilt week is prop-less. Stubbed here so the refresh is a pure
        # function of the artifact; the probe's own budget is pinned in
        # TestSignature.
        monkeypatch.setattr(ps.routes, "_get_player_props_live", lambda season, week: sample)
    return built


class TestRefreshingTheCommittedArtifact:
    """The strongest statement available: the committed artifact, with its real
    position mix. It holds 10,416 prop rows across twelve prop weeks and three
    distinct row shapes in each, and it is NOT reconciled yet -- no row carries
    the two depth-chart fields -- so the idempotence claim is made on the
    artifact carried forward to the shape the current code emits
    (`_reconciled`), which is the state the first successful reconciliation
    leaves behind and the state it will be in from then on.

    Every hand-built week above is single-shape or a three-row sketch, which is
    why the row-0 predicate looked right. This is the fixture that reproduces
    both directions of the bug, at the scale where it costs a full `_build_week`
    per affected week on every scheduled run.

    No test here hardcodes a week number, a row count or a `current_week`, and
    the class name says what the run asserts rather than what the file once
    found. That is deliberate.
    `.github/workflows/refresh-public-snapshot.yml` commits and pushes the
    refreshed artifact several times a day on its own, and `tests.yml` runs this
    file on every push to `main` with no exclusion -- so a literal like
    `== ["2", "3", "4", "5", "6", "7"]` would red `main` on a bot commit nobody
    authored, under a name that points the reader at a reconciliation regression
    that did not happen. A failure that arrives that often is a failure everyone
    learns to ignore, which is the thing the artifact pin was supposed to
    prevent. Every expectation below is computed from the artifact itself, so a
    red run means the reconciliation changed behaviour, not that the season
    moved on.
    """

    @pytest.mark.parametrize("first_position", ["QB", "RB", "WR"])
    def test_refreshing_a_reconciled_artifact_rebuilds_nothing_outside_the_window(
        self, monkeypatch, capsys, first_position,
    ):
        reconciled = _reconciled(_shipped())
        built = _refresh(reconciled, monkeypatch, _live_sample(first_position))

        result = ps.build_snapshot(reconciled)
        out = capsys.readouterr().out

        # Before the fix this read [..., "1", "8", ..., "18"] whenever the
        # sample's row 0 was not a QB: the reused weeks' rows are current on
        # every one of their hundreds of rows, and the subset test still failed
        # on a MARKET key, because the signature was one arbitrary player's
        # whole key set. Measured 2026-09-28: 12 of 22 weeks re-fetched and
        # re-diffed, forever, on a snapshot that needed nothing.
        #
        # `_window(...)` is the whole expectation. Do not chain a literal onto
        # it: it would restate the adjacent call, buy no coverage, and red on
        # the next scheduled refresh.
        assert built == _window(reconciled["current_week"])
        assert "reconcil" not in out  # neither "prop shape changed" nor the count line
        # The reused weeks came through with every row intact, not just the
        # window's.
        assert all(
            {"is_starter", "depth_slot"} <= row.keys()
            for week in result["weeks"].values()
            for row in week.get("player_props") or []
        )

    def test_refreshing_it_rebuilds_exactly_the_weeks_whose_rows_predate_the_shape(self, monkeypatch, capsys):
        # The shipped artifact as it stands, with the current shape as the
        # sample: the rebuild list is the window plus every reused week with a
        # row that lacks an invariant key. Both halves are derived, so this is
        # still true the day the bot reconciles the artifact (then the second
        # half is empty and the assertion says "window only").
        shipped = _shipped()
        window = _window(shipped["current_week"])
        required = ps._position_invariant_keys(_live_sample("QB"))
        stale = sorted(
            key for key, week in shipped["weeks"].items()
            if key not in window
            and any(not required <= row.keys() for row in week["player_props"] or [])
        )

        built = _refresh(shipped, monkeypatch, _live_sample("QB"))
        ps.build_snapshot(shipped)
        out = capsys.readouterr().out

        # Compared as a multiset, not a list: the window is built first in
        # ascending order and reconciled weeks are appended afterwards, so the
        # two orders coincide today and need not as the season runs.
        assert sorted(built, key=int) == sorted([*window, *stale], key=int)
        # And the log line follows the data: it appears exactly when there was
        # something to reconcile, so neither half of the run is vacuous.
        assert ("prop shape changed" in out) is bool(stale)

    def test_the_propless_weeks_are_left_alone(self, monkeypatch):
        # A week with no prop rows has no rows to be stale, so the
        # reconciliation has nothing to say about it. Derived rather than named,
        # for the reason in the class docstring.
        shipped = _shipped()
        window = _window(shipped["current_week"])
        propless = {key for key, week in shipped["weeks"].items() if not week["player_props"]}
        assert propless, "the artifact has no prop-less week; this test would pass vacuously"
        # Only the ones outside the window are in question -- a window week is
        # built whatever its shape, that is what a window is.
        reusable = sorted(propless - set(window), key=int)
        assert reusable, f"every prop-less week is inside the rebuild window {window}"

        built = _refresh(_reconciled(shipped), monkeypatch, _live_sample("QB"))
        ps.build_snapshot(_reconciled(shipped))

        assert not set(reusable) & set(built)

    @pytest.mark.parametrize("first_position", ["QB", "WR"])
    def test_a_stale_row_deep_inside_a_week_is_still_caught(self, monkeypatch, first_position):
        # The inverse of the test above, and the reason it can be trusted: the
        # predicate does still say "stale". The LAST prop row of the largest
        # reused week loses `is_starter` -- at the introducing commit, row 927 of
        # 927 -- and that week must come back. Row 0 of it carries every key, so
        # a row-0 check would have missed it under a QB sample, and a
        # market-blind one would have rebuilt the whole season under a WR one.
        reconciled = _reconciled(_shipped())
        weeks = reconciled["weeks"]
        window = _window(reconciled["current_week"])
        # The row count is a precondition, not the expectation: "deep inside" is
        # the point, so refuse to run against a week too small to be deep.
        deep = sorted(
            (
                key for key, week in weeks.items()
                if key not in window and len(week["player_props"]) > 100
            ),
            key=lambda key: (-len(weeks[key]["player_props"]), int(key)),
        )
        assert deep, f"no reused week with > 100 prop rows, window {window}"
        target = deep[0]

        damaged = {**reconciled, "weeks": {**weeks, target: _damage_last_row(weeks[target], "is_starter")}}

        built = _refresh(damaged, monkeypatch, _live_sample(first_position))
        ps.build_snapshot(damaged)

        # The window, plus the one week this test damaged, and nothing else.
        assert sorted(built, key=int) == sorted([*window, target], key=int)

    @pytest.mark.parametrize("first_position", ["QB", "WR"])
    def test_a_week_missing_a_market_key_is_not_stale(self, monkeypatch, first_position):
        # A market key follows the position, so losing one is not a shape
        # change: a player whose position moved between builds no longer carries
        # the old position's market. Rebuilding the week over that is the same
        # permanent fetch burn, only quieter -- and under a WR sample the row-0
        # version did exactly that to every reused week in the season.
        reconciled = _reconciled(_shipped())
        weeks = reconciled["weeks"]
        window = _window(reconciled["current_week"])
        carriers = sorted(
            (
                key for key, week in weeks.items()
                if key not in window
                and any("rushing_yards" in row for row in week["player_props"])
            ),
            key=int,
        )
        assert carriers, f"no reused week carries `rushing_yards`, window {window}"
        target = carriers[0]

        reshaped = {**reconciled, "weeks": {**weeks, target: _without(weeks[target], "rushing_yards")}}

        built = _refresh(reshaped, monkeypatch, _live_sample(first_position))
        ps.build_snapshot(reshaped)

        assert built == window


class TestNoTestHereReachesTheNetwork:
    def test_the_guard_records_and_raises(self):
        stub = _forbidden("upstream.thing")
        with pytest.raises(AssertionError):
            stub(1, kw=2)
        assert FORBIDDEN_CALLS == ["upstream.thing"]
        FORBIDDEN_CALLS.clear()  # or the autouse teardown fails on this test

    def test_every_live_builder_public_snapshot_calls_is_accounted_for(self):
        # The guard is only as good as its list. A fifth live builder added to
        # `build_snapshot` fails this rather than quietly reaching nfl_data_py.
        source = Path(ps.__file__).read_text()
        called = set(re.findall(r"routes\.(_get_\w+_live)", source))
        assert called, "the scan matched nothing; it would pass vacuously"
        accounted = set(OFFLINE_BUILDERS) | set(FORBIDDEN_ROUTES_CALLS)
        assert called <= accounted, f"unaccounted live builders: {sorted(called - accounted)}"

    def test_every_forbidden_upstream_name_still_exists(self):
        # The other half of the same blind spot. A leaf renamed or dropped would
        # leave the suite-wide guard referring to nothing, and it would go quiet
        # rather than fail: `getattr(mod, name, None)` is None for a name that
        # is gone, so every call is "already stubbed".
        from nfl_predictor.api import routes

        for module_name, attributes in FORBIDDEN_UPSTREAM.items():
            module = getattr(routes, module_name, None)
            assert module is not None, f"routes.{module_name} is gone; the guard is stale"
            real = getattr(module, "_real", module)
            for name in attributes:
                assert callable(getattr(real, name, None)), (
                    f"routes.{module_name}.{name} is gone or is not callable; "
                    "the suite-wide guard now protects nothing"
                )

    def test_no_test_in_this_file_is_marked_network(self):
        # The repo's convention (`pyproject.toml`: `network: hits a live
        # upstream API; deselect with '-m "not network"'`) is that a test which
        # needs the network says so and is skipped by default. None of these may:
        # the file has to be reproducible on a machine with no nflverse cache and
        # no upstream credentials, which is the machine this bug shipped from.
        found = _tests_in_this_file()
        assert found, "the scan matched nothing; it would pass vacuously"
        assert [name for name, owners in found if _marked_network(owners)] == []

    def test_that_scan_would_see_a_class_level_or_module_level_mark(self):
        """The guard above is vacuous today, so it needs a case of its own.

        A mark can arrive three ways and only one of them sits on the function:
        a decorator on the test, a class-level `pytestmark`, or a module-level
        `pytestmark`. Scanning the function alone -- which is what this test used
        to do -- would have gone blind to the other two, and a guard that quietly
        stops guarding is worse than no guard, because it is still there to be
        read.
        """

        class _MarkedClass:
            pytestmark = [pytest.mark.network]

            def test_inherited_from_the_class(self):
                pass

        class _UnmarkedClass:
            def test_not_marked(self):
                pass

        @pytest.mark.network
        def test_decorated():
            pass

        namespace = {
            "_MarkedClass": _MarkedClass,
            "_UnmarkedClass": _UnmarkedClass,
            "test_decorated": test_decorated,
        }
        found = _tests_in(namespace, SimpleNamespace(), __name__)
        assert {name for name, _ in found} == {
            "test_decorated",
            "_MarkedClass.test_inherited_from_the_class",
            "_UnmarkedClass.test_not_marked",
        }

        marked = {name for name, owners in found if _marked_network(owners)}
        # The class-level mark is seen, through the class...
        assert "_MarkedClass.test_inherited_from_the_class" in marked
        # ...and the function-level one through the function.
        assert "test_decorated" in marked
        # The unmarked class is not reported, so the scan discriminates rather
        # than blanket-reporting...
        assert "_UnmarkedClass.test_not_marked" not in marked
        # ...and the class-level mark is genuinely invisible on the function
        # itself, which is the whole reason the chain exists.
        assert not _marks_on(_MarkedClass.test_inherited_from_the_class)

        # A module-level `pytestmark` covers every test in the module, so the
        # module has to be in the chain too.
        module_marked = {
            name for name, owners in _tests_in(
                namespace, SimpleNamespace(pytestmark=[pytest.mark.network]), __name__
            )
            if _marked_network(owners)
        }
        assert module_marked == {name for name, _ in found}

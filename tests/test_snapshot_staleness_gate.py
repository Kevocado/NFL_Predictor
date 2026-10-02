"""A snapshot older than the models it was built from must never be published.

The defect this file exists for: `data/public_snapshot.json` carried
`generated_at 2026-10-02T00:50:56Z` while `models/manifest.json` carried
`trained_at 2026-10-02T05:05:04Z`, and 4h14m08s of the anytime-TD retrain sat
between them. The snapshot was never regenerated -- `git log 56c3d84..61e34be --
data/public_snapshot.json` is empty -- so it went on publishing 746 QB rows of
`anytime_td_prob` computed against the OLD `rushing + receiving + passing` label
that PR #26 deleted, while the committed v2 model scored the same rows at about
0.097. The artifact and the code disagreed by half the probability scale and
nothing said so.

A timestamp comparison is necessary but NOT sufficient, and the second half of
this file is the part that matters. `build_snapshot` stamps `generated_at`
itself, so a run that copies every prop week forward re-stamps a superseded
artifact's numbers as fresh. Measured while fixing this: a regeneration with the
timestamp gate but without the reuse gate rebuilt weeks 3-8 and moved 258 of 729
QB rows, then republished the 471 rows in weeks 2 and 9-18 -- verbatim copies --
unchanged under a brand new `generated_at`. A gate that only compared timestamps
would have called that artifact fresh.

The guard against live builders below is the same one
`test_snapshot_shape_reconciliation.py` uses, for the same reason: `build_snapshot`
wraps its four season-level builders in bare `except Exception`, so a test that
stubs only some of them gets a green run *and* a live call.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import pytest

from live_upstream import FORBIDDEN_CALLS, assert_no_live_upstream, install_builder_guard
from nfl_predictor import config
from nfl_predictor import public_snapshot as ps


@pytest.fixture(autouse=True)
def _no_live_builder(monkeypatch):
    FORBIDDEN_CALLS.clear()
    install_builder_guard(monkeypatch)
    yield
    assert_no_live_upstream()


#: The two instants the defect was made of, exactly as they are committed.
SNAPSHOT_GENERATED_AT = "2026-10-02T00:50:56.139860+00:00"
MODELS_TRAINED_AT = "2026-10-02T05:05:04.816068+00:00"

#: The gap between them, computed rather than transcribed: 4h14m08.676208s.
GAP = datetime.fromisoformat(MODELS_TRAINED_AT) - datetime.fromisoformat(SNAPSHOT_GENERATED_AT)


def _manifest(trained_at: str | None = MODELS_TRAINED_AT) -> dict:
    return {"trained_at": trained_at} if trained_at is not None else {}


def _snapshot(generated_at: str | None = SNAPSHOT_GENERATED_AT, **extra: Any) -> dict:
    body: dict[str, Any] = {"season": 2026, **extra}
    if generated_at is not None:
        body["generated_at"] = generated_at
    return body


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #


class TestStalenessGap:
    def test_a_snapshot_older_than_the_models_is_a_positive_gap(self):
        assert ps.staleness_gap(_snapshot(), _manifest()) == GAP

    def test_a_snapshot_newer_than_the_models_is_a_negative_gap(self):
        assert ps.staleness_gap(_snapshot("2026-10-02T09:35:41+00:00"), _manifest()) < timedelta(0)

    def test_the_same_instant_is_a_zero_gap_not_a_stale_one(self):
        # Equality is allowed on purpose: a snapshot written in the same instant
        # the models were is not a snapshot of a different model, and refusing it
        # would make the gate a coin flip on clock resolution.
        assert ps.staleness_gap(_snapshot(MODELS_TRAINED_AT), _manifest()) == timedelta(0)

    def test_both_sides_are_compared_in_utc_not_as_naive_datetimes(self):
        # `trained_at` written with a Z and `generated_at` with +00:00 are the same
        # instant, and Python refuses to compare an aware datetime with a naive
        # one at all -- which is the TypeError this must not raise.
        assert ps.staleness_gap(
            _snapshot("2026-10-02T00:00:00Z"), _manifest("2026-10-02T00:00:00+00:00")
        ) == timedelta(0)

    def test_a_naive_timestamp_is_read_as_utc_rather_than_refused(self):
        assert ps.staleness_gap(
            _snapshot("2026-10-02T00:00:00"), _manifest("2026-10-02T00:00:00")
        ) == timedelta(0)

    def test_an_unreadable_timestamp_is_undecidable_rather_than_fresh(self):
        # The third state, and deliberately not a pass. "Nothing to compare" and
        # "compared and agreed" must not look the same to a caller, because the
        # whole cost of this bug class was silence.
        for snapshot, manifest in (
            (_snapshot(None), _manifest()),
            (_snapshot(), _manifest(None)),
            (_snapshot("not-a-timestamp"), _manifest()),
            (None, _manifest()),
        ):
            assert ps.staleness_gap(snapshot, manifest) is None


class TestAssertPublishable:
    def test_a_snapshot_older_than_the_models_is_refused(self):
        with pytest.raises(ps.StaleSnapshotError):
            ps.assert_publishable(_snapshot(), _manifest())

    def test_a_snapshot_newer_than_the_models_passes(self):
        assert ps.assert_publishable(_snapshot("2026-10-02T09:35:41+00:00"), _manifest()) is None

    def test_a_snapshot_at_the_same_instant_as_the_models_passes(self):
        assert ps.assert_publishable(_snapshot(MODELS_TRAINED_AT), _manifest()) is None

    def test_the_error_names_both_timestamps_and_the_gap(self):
        with pytest.raises(ps.StaleSnapshotError) as excinfo:
            ps.assert_publishable(_snapshot(), _manifest())
        message = str(excinfo.value)
        assert SNAPSHOT_GENERATED_AT in message
        assert MODELS_TRAINED_AT in message
        assert ps._format_gap(GAP) in message
        # The raw seconds too, so the gap is greppable as well as readable.
        assert f"{GAP.total_seconds():.1f}s" in message

    def test_an_undecidable_pair_is_refused_and_says_which_side_is_unreadable(self):
        with pytest.raises(ps.StaleSnapshotError) as excinfo:
            ps.assert_publishable(_snapshot(), _manifest(None))
        message = str(excinfo.value)
        assert "trained_at=None" in message
        assert SNAPSHOT_GENERATED_AT in message

    def test_it_names_the_repair_rather_than_only_the_fault(self):
        with pytest.raises(ps.StaleSnapshotError) as excinfo:
            ps.assert_publishable(_snapshot(), _manifest())
        assert "python -m nfl_predictor.public_snapshot" in str(excinfo.value)


class TestDescribeStaleness:
    def test_the_stale_line_names_both_timestamps_the_gap_and_the_verdict(self, capsys):
        print(ps.describe_staleness(_snapshot(), _manifest()))
        line = capsys.readouterr().out
        assert SNAPSHOT_GENERATED_AT in line
        assert MODELS_TRAINED_AT in line
        assert f"{int(GAP.total_seconds() // 3600)}h" in line
        assert "STALE" in line

    def test_the_fresh_line_says_so(self, capsys):
        print(ps.describe_staleness(_snapshot("2026-10-02T09:35:41+00:00"), _manifest()))
        assert "fresh" in capsys.readouterr().out

    def test_the_undecidable_line_does_not_claim_freshness(self, capsys):
        print(ps.describe_staleness(_snapshot(), _manifest(None)))
        line = capsys.readouterr().out
        assert "UNDECIDABLE" in line
        assert "fresh" not in line


# --------------------------------------------------------------------------- #
# The gate is in the code path that WRITES the file, not only in the workflow
# --------------------------------------------------------------------------- #


def _pin_window(monkeypatch, *, current_week: int = 2, max_week: int = 4) -> None:
    """Pin the rebuild window so a test states which weeks are reused."""
    monkeypatch.setattr(ps, "MAX_WEEK", max_week)
    monkeypatch.setattr(ps, "REBUILD_WEEKS_BEHIND", 0)
    monkeypatch.setattr(ps, "REBUILD_WEEKS_AHEAD", 0)
    monkeypatch.setattr(ps.routes, "current_season_and_week", lambda: (2026, current_week))


def _week(*rows: dict) -> dict:
    return {"games": [], "predictions": {}, "player_props": list(rows)}


#: One row as the superseded model produced it, and one as the current model
#: produces it. Same keys, different probability: the shape check cannot tell these
#: apart, which is the point -- the two rows differ in a NUMBER, so the reused-week
#: reconciliation has to be bypassed for staleness and only the timestamp knows.
STALE_ROW = {"player_name": "A. Back", "position": "QB", "anytime_td_prob": 0.9}
CURRENT_ROW = {"player_name": "A. Back", "position": "QB", "anytime_td_prob": 0.1}


def _payload(generated_at: str) -> dict:
    return {"generated_at": generated_at, "season": 2026, "weeks": {}}


class TestMainRefusesToWrite:
    """`main` is the only writer of the published file, so this is the gate that
    counts: `workflow_dispatch` and a hand-run `python -m
    nfl_predictor.public_snapshot` both land here, and neither can skip it."""

    def _isolate(self, monkeypatch, tmp_path, *, on_disk: str | None, built_at: str, trained_at: str | None) -> Any:
        # `on_disk=None` writes a snapshot with no `generated_at` key at all,
        # which is a different defect from an unparseable one and is exercised
        # separately: the gate must refuse both.
        path = tmp_path / "public_snapshot.json"
        body = {"season": 2026, "weeks": {}}
        if on_disk is not None:
            body["generated_at"] = on_disk
        path.write_text(json.dumps(body))
        monkeypatch.setattr(config, "PUBLIC_SNAPSHOT_PATH", path)
        monkeypatch.setattr(ps.model_manifest, "load_manifest", lambda: _manifest(trained_at))
        monkeypatch.setattr(ps, "build_snapshot", lambda *a, **k: _payload(built_at))
        return path

    def test_a_stale_payload_is_not_written(self, monkeypatch, tmp_path):
        path = self._isolate(
            monkeypatch, tmp_path,
            on_disk="2026-10-02T06:00:00+00:00",
            built_at=SNAPSHOT_GENERATED_AT, trained_at=MODELS_TRAINED_AT,
        )
        with pytest.raises(ps.StaleSnapshotError):
            ps.main()
        # The file still holds the pre-run content: the refusal happens before the
        # write, not after it, so a failed run cannot leave a bad artifact behind.
        assert json.loads(path.read_text())["generated_at"] == "2026-10-02T06:00:00+00:00"

    def test_a_fresh_payload_is_written(self, monkeypatch, tmp_path):
        path = self._isolate(
            monkeypatch, tmp_path,
            on_disk="2026-10-02T06:00:00+00:00",
            built_at="2026-10-02T09:35:41.062814+00:00", trained_at=MODELS_TRAINED_AT,
        )
        ps.main()
        assert json.loads(path.read_text())["generated_at"] == "2026-10-02T09:35:41.062814+00:00"

    def test_the_verdict_for_the_file_on_disk_is_logged_and_passed_to_the_build(
        self, monkeypatch, tmp_path, capsys
    ):
        """A stale artifact on disk is the state this run exists to repair, so it
        does not abort the run -- it is logged, and it is handed to the build so
        the superseded weeks are rebuilt rather than reused."""
        self._isolate(
            monkeypatch, tmp_path,
            on_disk=SNAPSHOT_GENERATED_AT,
            built_at="2026-10-02T09:35:41.062814+00:00", trained_at=MODELS_TRAINED_AT,
        )
        seen: dict[str, Any] = {}

        def fake_build(previous=None, *, previous_predates_models=False):
            seen["previous"] = previous
            seen["previous_predates_models"] = previous_predates_models
            return _payload("2026-10-02T09:35:41.062814+00:00")

        monkeypatch.setattr(ps, "build_snapshot", fake_build)
        ps.main()
        out = capsys.readouterr().out
        assert SNAPSHOT_GENERATED_AT in out  # the stale file's timestamp is on the record
        assert MODELS_TRAINED_AT in out
        assert "STALE" in out
        assert seen["previous_predates_models"] is True
        assert seen["previous"]["generated_at"] == SNAPSHOT_GENERATED_AT  # the file that was read

    def test_a_current_artifact_does_not_claim_the_build_is_superseding_anything(
        self, monkeypatch, tmp_path
    ):
        self._isolate(
            monkeypatch, tmp_path,
            on_disk="2026-10-02T06:00:00+00:00",
            built_at="2026-10-02T09:35:41.062814+00:00", trained_at="2026-10-02T00:00:00+00:00",
        )
        seen: dict[str, Any] = {}

        def fake_build(previous=None, *, previous_predates_models=False):
            seen["previous_predates_models"] = previous_predates_models
            return _payload("2026-10-02T09:35:41.062814+00:00")

        monkeypatch.setattr(ps, "build_snapshot", fake_build)
        ps.main()
        assert seen["previous_predates_models"] is False

    def test_an_existing_artifact_of_unknown_age_is_treated_as_superseded(self, monkeypatch, tmp_path):
        """CodeRabbit's Major, and it is right.

        `gap is None` is the case where freshness cannot be established at all.
        Treating that as "not stale" lets the builder reuse weeks from an artifact
        whose provenance is exactly what is unknown, re-stamp it, and pass the
        write-time gate -- the same silence the gate exists to catch, one layer
        down. So an existing-but-undecidable prior snapshot rebuilds.
        """
        for on_disk, trained_at in (
            ("not-a-timestamp", MODELS_TRAINED_AT),   # unreadable generated_at
            (None, MODELS_TRAINED_AT),                 # no generated_at at all
            ("2026-10-02T06:00:00+00:00", None),      # manifest says nothing
        ):
            self._isolate(
                monkeypatch, tmp_path,
                on_disk=on_disk,
                built_at="2026-10-02T09:35:41.062814+00:00", trained_at=trained_at,
            )
            seen: dict[str, Any] = {}

            def fake_build(previous=None, *, previous_predates_models=False):
                seen["previous_predates_models"] = previous_predates_models
                return _payload("2026-10-02T09:35:41.062814+00:00")

            monkeypatch.setattr(ps, "build_snapshot", fake_build)
            # With no readable `trained_at` the WRITE gate refuses too, and the
            # refusal is the point: what matters here is the reuse decision, which
            # is made before the build.
            if trained_at is None:
                with pytest.raises(ps.StaleSnapshotError):
                    ps.main()
            else:
                ps.main()
            assert seen["previous_predates_models"] is True, (
                f"an existing snapshot with on_disk={on_disk!r} and "
                f"trained_at={trained_at!r} cannot be shown to match the models, "
                "so its rows must not be reused"
            )

    def test_an_absent_artifact_is_not_treated_as_superseded(self, monkeypatch, tmp_path):
        """The asymmetry, and the reason it is a separate case: there is nothing to
        reuse, so freshness of nothing is not in question and the rebuild window
        is all there is. Treating this as superseded would rebuild every one of
        the 22 weeks on a first run, which is what `key not in previous_weeks`
        already does on its own."""
        path = tmp_path / "public_snapshot.json"
        monkeypatch.setattr(config, "PUBLIC_SNAPSHOT_PATH", path)
        monkeypatch.setattr(ps.model_manifest, "load_manifest", lambda: _manifest())
        seen: dict[str, Any] = {}

        def fake_build(previous=None, *, previous_predates_models=False):
            seen["previous"] = previous
            seen["previous_predates_models"] = previous_predates_models
            return _payload("2026-10-02T09:35:41.062814+00:00")

        monkeypatch.setattr(ps, "build_snapshot", fake_build)
        ps.main()
        assert seen["previous"] is None
        assert seen["previous_predates_models"] is False


# --------------------------------------------------------------------------- #
# The half that actually stops the republication
# --------------------------------------------------------------------------- #


class TestSupersededWeeksAreRebuilt:
    """A reused week is a verbatim copy, so across a retrain it republishes a
    superseded model under a fresh `generated_at` -- which the timestamp gate
    alone cannot see."""

    def _build(self, monkeypatch, previous: dict, **kwargs: Any) -> tuple[dict, list[int], list[Any]]:
        built: list[int] = []
        offered: list[Any] = []

        def fake_build_week(season, week: int, previous=None) -> dict:
            built.append(week)
            offered.append(previous)
            return _week(dict(CURRENT_ROW))

        monkeypatch.setattr(ps, "_build_week", fake_build_week)
        _pin_window(monkeypatch)
        return ps.build_snapshot(previous, **kwargs), built, offered

    def _previous_with_superseded_props(self) -> dict:
        return {
            "season": 2026,
            "weeks": {
                "1": _week(dict(STALE_ROW)),   # reused, carries props
                "2": _week(dict(STALE_ROW)),   # in the rebuild window anyway
                "3": _week(),                  # reused, no props to supersede
                "4": _week(dict(STALE_ROW)),   # reused, carries props
            },
        }

    def test_a_prop_week_older_than_the_models_is_rebuilt_not_reused(self, monkeypatch):
        result, built, _ = self._build(
            monkeypatch, self._previous_with_superseded_props(), previous_predates_models=True
        )
        # 1 and 4 carry rows a superseded model produced, so they are rebuilt. Week 3
        # has no rows to be wrong about, so it is still reused: this is not a
        # whole-season rebuild, it is a rebuild of what the old model computed.
        assert built == [1, 2, 4]
        assert 3 not in built
        assert result["weeks"]["1"]["player_props"] == [dict(CURRENT_ROW)]
        assert result["weeks"]["4"]["player_props"] == [dict(CURRENT_ROW)]

    def test_a_superseded_props_build_is_offered_no_fallback_to_the_superseded_rows(self, monkeypatch):
        """`_build_week` reads `previous` for one thing only: carrying the last good
        props forward when a fresh build fails. Those props belong to the old
        model, so they are not offered -- the week reports `unavailable` and the
        route answers 503 rather than a number the code no longer stands behind."""
        _, built, offered = self._build(
            monkeypatch, self._previous_with_superseded_props(), previous_predates_models=True
        )
        assert built == [1, 2, 4]
        assert offered == [None, None, None]

    def test_the_old_props_are_still_offered_when_the_models_have_not_moved(self, monkeypatch):
        # The default path is unchanged, and this is the cost control: without a
        # retrain the whole-season rebuild never happens -- weeks 1 and 4 keep
        # their rows, and the prop fallback `_build_week` relies on is still on
        # offer for the week that IS rebuilt. Row shapes match here so the
        # prop-shape reconciliation has no reason to widen the rebuild either.
        result, built, offered = self._build(monkeypatch, self._previous_with_superseded_props())
        assert built == [2]
        assert result["weeks"]["1"]["player_props"] == [dict(STALE_ROW)]
        assert result["weeks"]["4"]["player_props"] == [dict(STALE_ROW)]
        # The in-window week still gets its previous dict as a carry-forward
        # fallback, which is the whole reason `previous` is threaded into
        # `_build_week` at all.
        assert offered[0]["player_props"] == [dict(STALE_ROW)]

    def test_a_fresh_props_build_does_not_overwrite_a_superseded_week_with_nothing(self, monkeypatch):
        """The control on the rebuild: `previous=None` means no CARRY-FORWARD, not
        "publish an empty week". A week that builds cleanly gets fresh rows, which
        is what makes the superseded-row rebuild safe to run unattended."""
        result, built, _ = self._build(
            monkeypatch, self._previous_with_superseded_props(), previous_predates_models=True
        )
        assert built == [1, 2, 4]
        for key in ("1", "2", "4"):
            assert result["weeks"][key]["player_props"] == [dict(CURRENT_ROW)]

    def test_a_failed_standings_build_does_not_carry_a_superseded_projection(self, monkeypatch):
        """The standings projection is the game model's output -- `project_standings`
        is handed a `predict_fn` built from `load_models` -- so the previous
        snapshot's rows are the previous model's wins."""
        previous = {"season": 2026, "weeks": {}, "standings": [{"team": "NE", "projected_wins": 12}]}

        def boom(season):
            raise RuntimeError("no models")

        monkeypatch.setattr(ps, "_build_week", lambda season, week, previous=None: _week(dict(CURRENT_ROW)))
        monkeypatch.setattr(ps.routes, "_get_standings_live", boom)
        _pin_window(monkeypatch, max_week=2)
        assert ps.build_snapshot(previous, previous_predates_models=True)["standings"] == []
        assert ps.build_snapshot(previous)["standings"] == [{"team": "NE", "projected_wins": 12}]


# --------------------------------------------------------------------------- #
# The committed artifact
# --------------------------------------------------------------------------- #


def test_the_committed_snapshot_is_not_older_than_the_committed_models():
    """The check that would have failed CI on the retrain commit.

    `main`'s gate cannot see this state on its own: it runs on the payload a build
    is about to write, and the defect was a snapshot committed *without* being
    rebuilt. Over the repository's own artifact the comparison is the whole
    point, so it is asserted here rather than left to a run somebody has to
    remember to trigger.
    """
    from nfl_predictor.models import manifest as model_manifest

    snapshot = json.loads(config.PUBLIC_SNAPSHOT_PATH.read_text())
    ps.assert_publishable(snapshot, model_manifest.load_manifest())
import json

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models import manifest

from fitted_stand_ins import load_pickle as _servable_load_pickle


def _code_game_features():
    """The code's game-level feature list, from `features.build.FEATURE_COLUMNS`.

    A function, not a module constant, so it reads the constant live: the
    game-level drift tests monkeypatch `FEATURE_COLUMNS` to simulate the code
    moving on, and a value captured at import would not follow.

    Hand-built manifests below have to carry this in BOTH places `load_models`
    reads it -- the top-level `feature_cols`, which
    `routes._predict_game_from_models` reindexes the live row by, and the
    `artifact_fingerprint` copy -- for the same reason they already have to carry
    the real `player_usage.PLAYER_FEATURE_COLUMNS`: the fingerprint check refuses
    a payload that disagrees with the code.
    """
    return list(manifest.feature_build.FEATURE_COLUMNS)


def _fake_games(seasons):
    rng = np.random.default_rng(3)
    rows = []
    teams = [f"T{i}" for i in range(8)]
    for season in seasons:
        for week in range(1, 6):
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                rows.append(
                    {
                        "game_id": f"{season}_{week}_{home}_{away}", "season": season, "week": week,
                        "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)),
                        "home_team": home, "away_team": away,
                        "home_score": int(rng.integers(10, 35)), "away_score": int(rng.integers(10, 35)),
                    }
                )
    return pd.DataFrame(rows)


def _fake_player_stats(seasons):
    rng = np.random.default_rng(4)
    rows = []
    for season in seasons:
        for week in range(1, 6):
            rows.append(
                {
                    "player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "T0",
                    "season": season, "week": week,
                    "passing_yards": 0, "passing_tds": 0,
                    "rushing_yards": int(rng.integers(40, 120)), "rushing_tds": int(rng.integers(0, 2)),
                    "receiving_yards": int(rng.integers(0, 30)), "receiving_tds": 0,
                    "receptions": 2, "targets": 3, "carries": 18,
                }
            )
    return pd.DataFrame(rows)


def test_train_all_writes_a_manifest_with_chosen_candidate(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))

    result = manifest.train_all(seasons=seasons)

    assert result["chosen_candidate"] in ("elo", "ridge", "xgb")
    assert (tmp_path / "manifest.json").exists()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["chosen_candidate"] == result["chosen_candidate"]
    assert (tmp_path / "game_outcome_model.pkl").exists()
    assert (tmp_path / "total_points_model.pkl").exists()
    assert (tmp_path / "anytime_td_model.pkl").exists()
    assert (tmp_path / "rushing_yards_model.pkl").exists()
    assert (tmp_path / "receiving_yards_model.pkl").exists()


def test_load_models_round_trips_after_train_all(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))

    manifest.train_all(seasons=seasons)
    models = manifest.load_models()

    assert "game_outcome_model" in models
    assert "player_models" in models
    assert "anytime_td" in models["player_models"]


def test_train_all_rejects_training_data_without_walk_forward_fold(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games([2024]))

    with pytest.raises(ValueError, match="walk-forward validation fold"):
        manifest.train_all(seasons=[2024])


def test_load_models_refuses_a_player_model_fitted_on_an_unservable_column(monkeypatch, tmp_path):
    """The general guard, at the entry point that matters.

    `predict_props` scores a live row with `reindex(feature_cols).fillna(0)`, so a
    fitted column the serving builder cannot emit is served as a constant zero for
    every player and every week -- which is exactly how `passing_tds_roll` shipped
    fitted-but-unserved. `load_models` now refuses the payload instead.

    The fingerprint pins the manifest's list to `PLAYER_FEATURE_COLUMNS` before
    this check runs, so the surviving way to reach it is the code's own list
    naming a column the serving builder does not emit. That is simulated by making
    the builder stop emitting it, which is precisely the shape of the original
    bug: a column the training frame grew and the serving builder did not.
    (Asserting the manifest's copy of the list instead would only re-prove the
    fingerprint check.)

    **Narrowing the builder rather than `SERVING_FEATURE_COLUMNS` is the whole
    point of the change, and it is why this test had to be rewritten rather than
    left alone.** The old version monkeypatched that constant, because the old
    guard read it -- and it passed, which looked like evidence the guard worked.
    It was not: the guard compared `PLAYER_FEATURE_COLUMNS` against a constant
    DEFINED as that same list plus one column, so the test was exercising a
    hand-written list against a hand-written list and the real builder was never
    consulted. `manifest._player_serving_columns` now CALLS
    `build_features_for_player` instead, so this test narrows the builder and the
    audit responds to the thing serving actually calls.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    good = {"chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
            "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
            "player_feature_cols": cols, "yardage_metrics": [],
            "qb_passing_td": None,
            "artifact_fingerprint": manifest.artifact_fingerprint(cols, _code_game_features())}
    monkeypatch.setattr(manifest, "load_manifest", lambda: good)
    assert "anytime_td" in manifest.load_models()["player_models"]

    # Serving stops emitting a column the code trains on. Named explicitly rather
    # than sliced off the end, because the builder emits `passing_tds_roll` LAST
    # and that column belongs only to the QB passing-TD model -- dropping it would
    # not exercise the player-model path this test is about, and `iloc[:-1]` did
    # exactly that.
    dropped = cols[-1]
    real_builder = manifest.player_usage.build_features_for_player
    monkeypatch.setattr(
        manifest.player_usage, "build_features_for_player",
        lambda *a, **k: real_builder(*a, **k).drop(labels=[dropped]))
    with pytest.raises(ValueError, match="does not emit") as excinfo:
        manifest.load_models()
    assert dropped in str(excinfo.value)


# --- the artefact fingerprint: a stale model must not serve quietly ---------
#
# The defect these close was measured on the artefact this branch started from.
# `models/manifest.json` listed 5 `player_feature_cols` while the code had 6
# (`receptions_roll` was added), and `anytime_td_model.pkl` was fitted on 5.
# `load_models()` returned that payload without complaint.
#
# The pre-existing guard could not see it, for two separate reasons, and this
# block covers both:
#
#   * `_assert_servable_columns` tests `fitted ⊆ SERVING_FEATURE_COLUMNS`. The
#     stale 5 ARE a subset of the 7 servable columns, so it passed. A subset test
#     cannot see an artefact missing a feature.
#   * The label is never recomputed at serving, so an artefact fitted against the
#     OLD `anytime_td` definition loads exactly like one fitted against the new
#     one: right shape, wrong meaning.


def test_load_models_refuses_a_manifest_with_no_fingerprint(monkeypatch, tmp_path):
    """Absent fingerprint == unverifiable == refused.

    Deliberately not a skip. "No fingerprint recorded" is precisely the state
    being detected -- a manifest written before fingerprinting existed, which is
    every artefact in the wild right now -- so treating it as a pass would leave
    the bug open for exactly the models that have it.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    legacy = {"chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
              "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
              "player_feature_cols": ["passing_yards_roll"], "yardage_metrics": [],
              "qb_passing_td": None}
    assert "artifact_fingerprint" not in legacy
    monkeypatch.setattr(manifest, "load_manifest", lambda: legacy)

    with pytest.raises(ValueError, match="no 'artifact_fingerprint' key"):
        manifest.load_models()


def test_load_models_refuses_an_artefact_fitted_before_a_feature_was_added(monkeypatch, tmp_path):
    """The self-referential version, and the one that shipped.

    Every artefact this branch started from had 5 `player_feature_cols` and 6 in
    the code, and the guard passed them. Not a slip in the fixture: the manifest's
    `player_feature_cols` AND its `artifact_fingerprint` both carried the old
    shorter list, so the check that compared them compared the manifest with
    itself and agreed with itself. Add a column to `PLAYER_FEATURE_COLUMNS`,
    retrain nothing, and that is the whole defect -- `load_models` served a model
    fitted on features the code has moved past, silently, which is the case the
    fingerprint was merged to make loud.

    `_assert_servable_columns` cannot see it: the stale list is a *subset* of the
    serving columns, so the subset test is happy. The fingerprint has to compare
    against `player_usage.PLAYER_FEATURE_COLUMNS` -- the code -- for this to fail.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    code_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    stale_cols = code_cols[:-1]
    # The pre-fix state, asserted rather than assumed: the artefact's list is a
    # strict SUBSET of the code's, and still servable, so `_assert_servable_columns`
    # has nothing to say and the fingerprint is the only guard that can.
    assert set(stale_cols) < set(code_cols)
    assert set(stale_cols) < set(manifest.player_usage.SERVING_FEATURE_COLUMNS)

    stale_but_self_consistent = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
        "player_feature_cols": stale_cols, "yardage_metrics": [], "qb_passing_td": None,
        # Fingerprint taken over the same stale list -- the manifest agreeing
        # with itself, which is all it ever did.
        "artifact_fingerprint": manifest.artifact_fingerprint(stale_cols, _code_game_features()),
    }
    monkeypatch.setattr(manifest, "load_manifest", lambda: stale_but_self_consistent)
    # The two halves of this manifest agree, which is the entire reason the old
    # check passed it: nothing in the artefact contradicts anything else in it.
    assert (stale_but_self_consistent["player_feature_cols"]
            == stale_but_self_consistent["artifact_fingerprint"]["player_feature_cols"])

    with pytest.raises(ValueError, match="records models fitted on"):
        manifest.load_models()


def test_the_feature_mismatch_error_names_both_sides_and_the_fix(monkeypatch, tmp_path):
    """The error has to be actionable on its own.

    "your model is stale" is what this whole bug class looked like from the
    outside. All three facts a reader needs are in the message: the list the
    manifest recorded, the list the code now expects, and the command that
    resolves it.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    code_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    stale_cols = code_cols[:-1]
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
        "player_feature_cols": stale_cols, "yardage_metrics": [], "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(stale_cols, _code_game_features()),
    })

    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    message = str(excinfo.value)

    assert str(stale_cols) in message                  # what the manifest recorded
    assert str(code_cols) in message                   # what the code now expects
    assert "PLAYER_FEATURE_COLUMNS" in message         # where that list lives
    assert "python -m nfl_predictor.models.manifest" in message  # the retrain command


def test_load_models_accepts_a_fingerprint_that_matches_the_code(monkeypatch, tmp_path):
    """The positive control for the new comparison, hand-built rather than trained.

    `test_a_fingerprint_written_by_train_all_verifies_against_its_own_load` proves
    a just-fitted payload loads; this one proves a payload whose fingerprint was
    written by hand against the code's list loads too. Without it, a check that
    raised unconditionally would satisfy every test above -- including this fix's
    own new ones.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    code_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
        "player_feature_cols": code_cols, "yardage_metrics": {}, "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(code_cols, _code_game_features()),
    })

    models = manifest.load_models()

    assert models["player_feature_cols"] == code_cols
    assert models["player_models"]["feature_cols"] == code_cols
    assert "anytime_td" in models["player_models"]


def test_load_models_refuses_a_manifest_whose_fingerprint_contradicts_its_own_feature_list(
    monkeypatch, tmp_path
):
    """The other direction: a self-consistent fingerprint on a manifest that is not.

    `player_props.predict_props` reindexes a live row by
    `manifest["player_feature_cols"]`, not by the fingerprint's copy, so a
    manifest that kept a current fingerprint but a narrowed list would score a
    6-column model on 1 feature and pass a fingerprint check that only looked at
    the fingerprint.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    code_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    narrowed = code_cols[:1]
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
        "player_feature_cols": narrowed, "yardage_metrics": {}, "qb_passing_td": None,
        # Fingerprint is CURRENT -- the code's list, untouched.
        "artifact_fingerprint": manifest.artifact_fingerprint(code_cols, _code_game_features()),
    })

    with pytest.raises(ValueError, match="its own 'player_feature_cols'"):
        manifest.load_models()


def test_load_models_refuses_an_artefact_fitted_on_a_different_feature_list(monkeypatch, tmp_path):
    """Fit with one feature list, load against another. The exact stale case.

    The fingerprint recorded 5 columns; the code now has 6. `_assert_servable_columns`
    is happy -- 5 is a subset of 7 -- so this is the assertion that has to exist
    for the mismatch to be caught at all.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    stale_cols = ["passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll",
                  "targets_roll", "carries_roll"]
    current_cols = [*stale_cols, "receptions_roll"]
    assert set(stale_cols) < set(manifest.player_usage.SERVING_FEATURE_COLUMNS)

    manifest_mismatch = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
        "player_feature_cols": current_cols, "yardage_metrics": [], "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(stale_cols, _code_game_features()),
    }
    monkeypatch.setattr(manifest, "load_manifest", lambda: manifest_mismatch)

    with pytest.raises(ValueError, match="different feature set"):
        manifest.load_models()


def test_load_models_refuses_an_artefact_fitted_on_the_old_anytime_td_label(monkeypatch, tmp_path):
    """The case this whole branch is about, and the quietest one.

    Feature lists identical, shapes identical, nothing for `_assert_servable_columns`
    to complain about. The only difference is which `anytime_td` definition the
    classifier was fitted against: v1 summed passing TDs, v2 does not. Without the
    label version in the fingerprint this payload loads clean and every QB is
    scored against a market the model never saw.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    old_label = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
        "player_feature_cols": cols, "yardage_metrics": [], "qb_passing_td": None,
        "artifact_fingerprint": {**manifest.artifact_fingerprint(cols, _code_game_features()),
                                 "anytime_td_label_version": 1},
    }
    monkeypatch.setattr(manifest, "load_manifest", lambda: old_label)

    with pytest.raises(ValueError, match="label definition v1"):
        manifest.load_models()


def test_a_fingerprint_written_by_train_all_verifies_against_its_own_load(monkeypatch, tmp_path):
    """The positive control: the guard must not refuse a good, just-fitted payload.

    Without this, a fingerprint that always raised would pass every test above.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))

    result = manifest.train_all(seasons=seasons)

    assert result["artifact_fingerprint"]["anytime_td_label_version"] == \
        manifest.player_usage.ANYTIME_TD_LABEL_VERSION
    saved = json.loads((tmp_path / "manifest.json").read_text())
    manifest._verify_artifact_fingerprint(saved)  # must not raise
    assert "anytime_td" in manifest.load_models()["player_models"]


def test_bumping_the_label_version_invalidates_every_committed_artefact(monkeypatch, tmp_path):
    """The guard tracks the definition, so the next redefinition is also caught.

    Fingerprinting the version is only worth anything if changing the version
    breaks loading. Simulated with a monkeypatched v3: a payload fitted at v2 must
    stop loading, which is what happens the next time someone redefines the label.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    payload = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
        "player_feature_cols": cols, "yardage_metrics": [], "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(cols, _code_game_features()),
    }
    monkeypatch.setattr(manifest, "load_manifest", lambda: payload)
    manifest.load_models()  # v2 payload against v2 code: fine

    monkeypatch.setattr(manifest.player_usage, "ANYTIME_TD_LABEL_VERSION", 3)
    with pytest.raises(ValueError, match="label definition v2"):
        manifest.load_models()


# --- the game-level list gets the same treatment -----------------------------
#
# The player-side fingerprint (merged in #27) pinned `player_feature_cols` and
# the anytime-TD label version against the CODE's constants. The manifest records
# a SECOND feature list that nothing pinned:
#
#     "feature_cols" -> features/build.py::FEATURE_COLUMNS, ten per-game columns
#
# and it is not a restatement of the player list. It is per-GAME, not per-player:
# home/away pregame ratings, `rating_diff`, the four rolling scoring/conceding
# averages, both rest-day counts and `div_game` -- built by
# `features.build.build_training_frame` and emitted by `build_features_for_game`.
#
# What it fits:
#
#   * `total_points_model.pkl`  -- always, via `fit_xgb_margin(X_train, y_total)`
#   * `game_outcome_model.pkl`   -- whenever the chosen candidate is ridge or xgb
#                                  (`fit_margin_regression` / `fit_xgb_margin`);
#                                  the elo branch fits no feature matrix at all
#   * `sigma` and `total_sigma`  -- `residual_sigma` is scored on `X_train`
#   * the walk-forward candidate scores, which pick `chosen_candidate` itself
#
# The committed manifest chose `ridge`, so both game pickles and both sigmas are
# in scope on the artefact that is actually shipped.
#
# Why it was silent, in the same two-part way as the player list:
#
#   * `_assert_servable_columns` knows only about the player side -- it compares
#     against `player_usage.SERVING_FEATURE_COLUMNS` -- so even if it were
#     pointed at the game models it could only answer "is every fitted column one
#     the serving builder emits", a subset question that cannot see a MISSING
#     feature.
#   * Nothing recomputes the game features at serving. `routes.
#     _predict_game_from_models` reindexes the live row by
#     `manifest["feature_cols"]` and scores the committed pickles. Add a column
#     to `FEATURE_COLUMNS`, retrain nothing, and the manifest and its pickle stay
#     on the old list and agree with each other: right shape, no new feature, no
#     error anywhere. The new feature is simply never used.


def test_load_models_accepts_a_game_feature_list_that_matches_the_code(monkeypatch, tmp_path):
    """The positive control, and the one that makes the refusal tests meaningful.

    A guard that raised unconditionally would satisfy every other test in this
    section -- and, for the same reason, every player-side test above it. This
    proves a payload whose game-level list IS the code's list loads, so the
    refusals below are refusals rather than a blanket failure.

    `_load_pickle` is stubbed to `object()`, so there is no real model anywhere
    in this test that could be satisfying the guard: the verdict comes from the
    fingerprint and nothing else.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    game_cols = _code_game_features()
    player_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    payload = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": game_cols,
        "player_feature_cols": player_cols, "yardage_metrics": {}, "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(player_cols, game_cols),
    }
    monkeypatch.setattr(manifest, "load_manifest", lambda: payload)

    models = manifest.load_models()

    # Serving reads the top-level list, so that is what has to come back out --
    # and it has to be the code's list, not the manifest's own copy of it.
    assert models["feature_cols"] == game_cols
    assert models["feature_cols"] == list(manifest.feature_build.FEATURE_COLUMNS)


def test_load_models_refuses_an_artefact_fitted_before_a_game_feature_was_added(monkeypatch, tmp_path):
    """The silent-drift case for the game list, and the one that shipped open.

    Simulates the defect rather than the fix's absence: the CODE moves on --
    `FEATURE_COLUMNS` grows a column, as it would when someone adds a feature --
    and the committed artefact is not retrained. The manifest is entirely
    self-consistent, which is the point: its top-level `feature_cols` and its
    fingerprint both carry the old shorter list and agree with each other, and
    every pickle on disk was fitted against exactly that shorter list.

    Asserted rather than assumed, because the assertions are what make this the
    silent case:

      * the artefact's list is a strict SUBSET of the code's, so no subset test
        anywhere can object;
      * the manifest agrees with its own fingerprint, so a check built by
        comparing those two -- the mistake #27 made -- passes;
      * the player half is untouched and current, so the #27 guard is green.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    fitted_game_cols = _code_game_features()[:-1]  # the artefact is one short
    added = "playoff_flag"
    monkeypatch.setattr(
        manifest.feature_build, "FEATURE_COLUMNS", [*fitted_game_cols, added])

    # The pre-fix state of the artefact, asserted: a strict subset, and a
    # manifest that agrees with itself.
    assert set(fitted_game_cols) < set(manifest.feature_build.FEATURE_COLUMNS)
    player_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    stale_but_self_consistent = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": fitted_game_cols,
        "player_feature_cols": player_cols, "yardage_metrics": [], "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(player_cols, fitted_game_cols),
    }
    monkeypatch.setattr(manifest, "load_manifest", lambda: stale_but_self_consistent)
    assert (stale_but_self_consistent["feature_cols"]
            == stale_but_self_consistent["artifact_fingerprint"]["feature_cols"])
    # The player half is current, so #27's guard has nothing to say about this.
    assert (stale_but_self_consistent["player_feature_cols"]
            == stale_but_self_consistent["artifact_fingerprint"]["player_feature_cols"])

    with pytest.raises(ValueError, match="records the game models fitted on"):
        manifest.load_models()

    # The error names both sides, the constant they come from, and the command
    # that resolves it -- "your model is stale" is what this bug class looked
    # like from the outside, and a message that does not say which list or which
    # constant is not actionable.
    with pytest.raises(ValueError) as excinfo:
        manifest.load_models()
    message = str(excinfo.value)
    assert str(fitted_game_cols) in message                       # what the manifest recorded
    assert str([*fitted_game_cols, added]) in message            # what the code now expects
    assert "FEATURE_COLUMNS" in message                          # where that list lives
    assert "python -m nfl_predictor.models.manifest" in message  # the retrain command


def test_load_models_refuses_a_fingerprint_with_no_game_feature_list(monkeypatch, tmp_path):
    """Absent game-level fingerprint == unverifiable == refused. Not a skip.

    This is the state of the artefact that was committed when this section was
    written: `models/manifest.json` carried `player_feature_cols` and
    `anytime_td_label_version` and no game-level list, because the game half did
    not exist yet. Treating that as a pass would leave `total_points_model.pkl`
    and the ridge `game_outcome_model.pkl` unverified -- precisely the models
    with the bug -- so it raises, in its own words rather than as a bogus
    "fitted on []" mismatch.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    player_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    fingerprint = manifest.artifact_fingerprint(player_cols, _code_game_features())
    del fingerprint["feature_cols"]  # written before the game half existed
    payload = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
        "player_feature_cols": player_cols, "yardage_metrics": {}, "qb_passing_td": None,
        "artifact_fingerprint": fingerprint,
    }
    monkeypatch.setattr(manifest, "load_manifest", lambda: payload)

    # Both halves the check looks at are current, so the ONLY thing wrong is the
    # missing key -- which is what makes this a fail-closed test and not another
    # restatement of the mismatch test.
    assert "feature_cols" not in fingerprint
    assert payload["feature_cols"] == list(manifest.feature_build.FEATURE_COLUMNS)
    assert fingerprint["player_feature_cols"] == list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)

    with pytest.raises(ValueError, match="no game-level 'feature_cols'"):
        manifest.load_models()


def test_load_models_refuses_a_null_game_feature_list_rather_than_skipping_it(monkeypatch, tmp_path):
    """An explicit `null` is not an absent key and not a pass.

    `manifest.json` is JSON and `train_all` writes lists, so `null` here means a
    hand-edited or half-written artefact. Reading it as "no opinion recorded" is
    the fail-OPEN branch this section exists to remove: `list(None or [])` would
    compare `[]` against the code's ten and refuse for the wrong reason, while
    anything more lenient would skip the check. It must refuse, naming the key.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    player_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": _code_game_features(),
        "player_feature_cols": player_cols, "yardage_metrics": {}, "qb_passing_td": None,
        "artifact_fingerprint": {
            **manifest.artifact_fingerprint(player_cols, _code_game_features()),
            "feature_cols": None,
        },
    })

    with pytest.raises(ValueError, match="no game-level 'feature_cols'"):
        manifest.load_models()


def test_load_models_refuses_a_manifest_whose_game_fingerprint_contradicts_its_own_list(
    monkeypatch, tmp_path
):
    """The other direction: a current fingerprint on a manifest that is not.

    `load_models` returns `manifest["feature_cols"]` and
    `routes._predict_game_from_models` reindexes the live feature row by THAT
    list, not by the fingerprint's. A manifest that kept a current fingerprint
    but a narrowed top-level list would therefore score a full-width
    total-points model on a single column -- and, on the exact shape the player
    half already guards, pass a fingerprint check that only looked at the
    fingerprint.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    game_cols = _code_game_features()
    player_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0,
        "feature_cols": game_cols[:1],
        "player_feature_cols": player_cols, "yardage_metrics": {}, "qb_passing_td": None,
        # Fingerprint is CURRENT -- the code's list, untouched.
        "artifact_fingerprint": manifest.artifact_fingerprint(player_cols, game_cols),
    })

    with pytest.raises(ValueError, match="its own 'feature_cols'"):
        manifest.load_models()


def test_adding_a_game_feature_to_the_code_stops_a_committed_artefact_serving(monkeypatch, tmp_path):
    """The guard tracks the CODE, so the next added column is caught too.

    Fingerprinting a list is only worth something if changing the list breaks
    loading. Simulated with a monkeypatched `FEATURE_COLUMNS`: a payload fitted
    on the current list must stop loading the moment the code carries one more
    column, which is what happens the next time someone adds a game feature.
    Without this, a guard that merely recorded the manifest's list would keep
    agreeing with itself.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", _servable_load_pickle)

    fitted_game_cols = _code_game_features()
    player_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    payload = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": fitted_game_cols,
        "player_feature_cols": player_cols, "yardage_metrics": {}, "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(player_cols, fitted_game_cols),
    }
    monkeypatch.setattr(manifest, "load_manifest", lambda: payload)
    manifest.load_models()  # payload fitted on the current list, code agrees: fine

    monkeypatch.setattr(
        manifest.feature_build, "FEATURE_COLUMNS", [*fitted_game_cols, "weather_roof"])
    with pytest.raises(ValueError, match="records the game models fitted on"):
        manifest.load_models()


def test_train_all_records_the_game_feature_list_it_actually_fitted_on(monkeypatch, tmp_path):
    """The recording side, so the check above cannot rot into checking nothing.

    Asserted against the value `train_all` actually used -- the list
    `build_training_frame` handed back -- and not against the constant, because a
    fingerprint written from the constant would agree with it by construction
    and prove nothing.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))

    result = manifest.train_all(seasons=seasons)
    _, fitted_frame_cols = manifest.feature_build.build_training_frame(_fake_games(seasons))

    fingerprint = result["artifact_fingerprint"]
    assert fingerprint["feature_cols"] == fitted_frame_cols
    assert fingerprint["feature_cols"] == result["feature_cols"]
    assert fingerprint["player_feature_cols"] == result["player_feature_cols"]

    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["artifact_fingerprint"] == fingerprint
    # Round-trips through `load_models` and out to the serving list.
    assert manifest.load_models()["feature_cols"] == fitted_frame_cols


def test_load_models_ignores_stale_yardage_artifacts_after_retrain(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))
    manifest.train_all(seasons=seasons)
    assert (tmp_path / "receiving_yards_model.pkl").exists()

    without_receiving = _fake_player_stats(seasons)
    without_receiving["receiving_yards"] = 0
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: without_receiving)
    manifest.train_all(seasons=seasons)

    assert not (tmp_path / "receiving_yards_model.pkl").exists()
    assert "receiving_yards" not in manifest.load_models()["player_models"]


def _fake_games_with_conditions(seasons):
    df = _fake_games(seasons)
    df = df.copy()
    df["roof"] = "outdoors"
    df["temp"] = 30.0
    df["wind"] = 10.0
    return df


def test_block_fitted_model_loads_and_mismatch_is_refused(monkeypatch, tmp_path):
    """A model fitted with the conditions block loads when the manifest lists it.

    Failing-first for the review: `train_all` recorded no blocks and the load-time
    expectations derived from the base columns only, so a block-fitted payload could
    never load — and a manifest whose blocks disagreed with its artefacts was not
    refused loudly.
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games_with_conditions(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))

    result = manifest.train_all(seasons=seasons, blocks=("conditions",))
    assert result["feature_blocks"] == ["conditions"]
    assert "wx_known" in result["feature_cols"]

    models = manifest.load_models()
    assert models["feature_blocks"] == ["conditions"]
    assert "wx_known" in models["feature_cols"]

    # Mismatch: manifest claims no blocks while the artefacts were fitted with them.
    saved = json.loads((tmp_path / "manifest.json").read_text())
    saved["feature_blocks"] = []
    (tmp_path / "manifest.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="different feature set|different feature"):
        manifest.load_models()

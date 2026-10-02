import json

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models import manifest


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
    naming a column the serving builder does not emit. That is simulated by
    narrowing `SERVING_FEATURE_COLUMNS`, which is precisely the shape of the
    original bug: a column the training frame grew and the serving builder did
    not. (Asserting the manifest's copy of the list instead would only re-prove
    the fingerprint check.)
    """
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    good = {"chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
            "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
            "player_feature_cols": cols, "yardage_metrics": [],
            "qb_passing_td": None,
            "artifact_fingerprint": manifest.artifact_fingerprint(cols)}
    monkeypatch.setattr(manifest, "load_manifest", lambda: good)
    assert "anytime_td" in manifest.load_models()["player_models"]

    # Serving stops emitting the last feature the code trains on.
    monkeypatch.setattr(
        manifest.player_usage, "SERVING_FEATURE_COLUMNS", cols[:-1])
    with pytest.raises(ValueError, match="does not emit"):
        manifest.load_models()


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
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    legacy = {"chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
              "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
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
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    code_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    stale_cols = code_cols[:-1]
    # The pre-fix state, asserted rather than assumed: the artefact's list is a
    # strict SUBSET of the code's, and still servable, so `_assert_servable_columns`
    # has nothing to say and the fingerprint is the only guard that can.
    assert set(stale_cols) < set(code_cols)
    assert set(stale_cols) < set(manifest.player_usage.SERVING_FEATURE_COLUMNS)

    stale_but_self_consistent = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
        "player_feature_cols": stale_cols, "yardage_metrics": [], "qb_passing_td": None,
        # Fingerprint taken over the same stale list -- the manifest agreeing
        # with itself, which is all it ever did.
        "artifact_fingerprint": manifest.artifact_fingerprint(stale_cols),
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
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    code_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    stale_cols = code_cols[:-1]
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
        "player_feature_cols": stale_cols, "yardage_metrics": [], "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(stale_cols),
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
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    code_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
        "player_feature_cols": code_cols, "yardage_metrics": {}, "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(code_cols),
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
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    code_cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    narrowed = code_cols[:1]
    monkeypatch.setattr(manifest, "load_manifest", lambda: {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
        "player_feature_cols": narrowed, "yardage_metrics": {}, "qb_passing_td": None,
        # Fingerprint is CURRENT -- the code's list, untouched.
        "artifact_fingerprint": manifest.artifact_fingerprint(code_cols),
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
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    stale_cols = ["passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll",
                  "targets_roll", "carries_roll"]
    current_cols = [*stale_cols, "receptions_roll"]
    assert set(stale_cols) < set(manifest.player_usage.SERVING_FEATURE_COLUMNS)

    manifest_mismatch = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
        "player_feature_cols": current_cols, "yardage_metrics": [], "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(stale_cols),
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
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    old_label = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
        "player_feature_cols": cols, "yardage_metrics": [], "qb_passing_td": None,
        "artifact_fingerprint": {**manifest.artifact_fingerprint(cols),
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
    monkeypatch.setattr(manifest, "_load_pickle", lambda path: object())

    cols = list(manifest.player_usage.PLAYER_FEATURE_COLUMNS)
    payload = {
        "chosen_candidate": "ridge", "trained_at": "2026-09-04T22:12:49+00:00",
        "sigma": 12.0, "total_sigma": 10.0, "feature_cols": ["rating_diff"],
        "player_feature_cols": cols, "yardage_metrics": [], "qb_passing_td": None,
        "artifact_fingerprint": manifest.artifact_fingerprint(cols),
    }
    monkeypatch.setattr(manifest, "load_manifest", lambda: payload)
    manifest.load_models()  # v2 payload against v2 code: fine

    monkeypatch.setattr(manifest.player_usage, "ANYTIME_TD_LABEL_VERSION", 3)
    with pytest.raises(ValueError, match="label definition v2"):
        manifest.load_models()


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

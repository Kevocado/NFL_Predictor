import numpy as np
import pandas as pd

from nfl_predictor.features import player_usage
from nfl_predictor.models import player_props


def _player_stats():
    rows = []
    for week in range(1, 4):
        rows.append(
            {
                "player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "BAL",
                "season": 2025, "week": week,
                "passing_yards": 0, "passing_tds": 0, "rushing_yards": 80 + week, "rushing_tds": 1,
                "receiving_yards": 10, "receiving_tds": 0, "receptions": 2, "targets": 3, "carries": 18,
            }
        )
    return pd.DataFrame(rows)


def test_build_player_training_frame_adds_rolling_features_and_target():
    df, feature_cols = player_usage.build_player_training_frame(_player_stats())

    assert "anytime_td" in df.columns
    assert set(feature_cols).issubset(df.columns)
    # Week 1 has no prior games, so its rolling features should be NaN.
    week1 = df[df["week"] == 1].iloc[0]
    assert pd.isna(week1["rushing_yards_roll"])
    # Week 3's rolling rushing yards should reflect weeks 1-2 only.
    week3 = df[df["week"] == 3].iloc[0]
    assert week3["rushing_yards_roll"] == (81 + 82) / 2


def test_build_features_for_player_returns_none_with_no_history():
    row = player_usage.build_features_for_player("unknown", _player_stats())
    assert row is None


def test_build_features_for_player_matches_the_training_row_for_a_stated_week():
    """The target week must be stated, or the assertion means nothing.

    This test previously called the function with no week and asserted
    (81 + 82) / 2 -- the value for a **week-3** fixture. But with no week the
    pregame view is every played week, which is the view for a **week-4**
    fixture, and that is (81 + 82 + 83) / 3. The two numbers describe different
    predictions, and the test picked one without saying which, which is how it
    came to pass against code that matched training at *no* week at all.

    Both are now asserted, each against `_add_rolling`'s own row for that week.
    """
    # Five weeks, so `_add_rolling` produces rows for *both* target weeks and the
    # comparison is like-for-like against real training output rather than against
    # an arithmetic re-derivation.
    base = _player_stats()
    extra = base.iloc[-1:].copy()
    extra["week"] = 4
    base = pd.concat([base, extra], ignore_index=True)
    history = base
    training = player_usage._add_rolling(history)

    # Week-3 fixture: pregame view is weeks 1-2 -> 81.5, matching training at week 3.
    week3 = player_usage.build_features_for_player("p1", history, season=2025, week=3)
    assert week3 is not None
    assert week3["rushing_yards_roll"] == pytest.approx((81 + 82) / 2)
    assert week3["rushing_yards_roll"] == pytest.approx(
        training[training["week"] == 3]["rushing_yards_roll"].iloc[0])

    # Week-4 fixture: pregame view is weeks 1-3 -> 82.0, matching training at week 4.
    week4 = player_usage.build_features_for_player("p1", history, season=2025, week=4)
    assert week4 is not None
    assert week4["rushing_yards_roll"] == pytest.approx((81 + 82 + 83) / 3)
    assert week4["rushing_yards_roll"] == pytest.approx(
        training[training["week"] == 4]["rushing_yards_roll"].iloc[0])


def test_receptions_is_a_rolled_stat_and_feature_column():
    assert "receptions" in player_usage.ROLL_STATS
    assert "receptions_roll" in player_usage.PLAYER_FEATURE_COLUMNS

    row = player_usage.build_features_for_player("p1", _player_stats())
    assert row is not None
    assert row["receptions_roll"] == pytest.approx(2.0)


# --- the anytime-TD label excludes passing TDs (2026-10-01) ----------------
#
# `build_player_training_frame` previously summed `passing_tds` into `anytime_td`
# alongside rushing and receiving. Dropping it is the point of this block: with
# passing included, a QB's anytime-TD was driven by his passing score, so
# quarterbacks sorted to the top of a category whose name never mentions passing.
# Passing TDs are a separate market (`models/qb_passing_td.py`, `passing_td_*`
# fields on QB rows), so `anytime_td` is now rushing-or-receiving only.
#
# The label is the classifier's TRAINING TARGET, so these assert on
# `build_player_training_frame`, the one place it is computed. Nothing recomputes
# it at serving: `player_props.predict_props` builds `X` from the feature row
# alone and calls the committed pickle. So these tests pin the definition the next
# `models/manifest.py::train_all` run will fit against -- they are not, and cannot
# be, a serving-path assertion.


def _labels_for(rows: list[dict]) -> list[int]:
    """`anytime_td` per row, for a one-game-per-week fixture of `rows`."""
    frame = pd.DataFrame(
        [
            {
                "player_id": r.get("player_id", "p1"),
                "player_name": "Player",
                "position": r.get("position", "QB"),
                "recent_team": "BAL",
                "season": 2025,
                "week": i + 1,
                "passing_yards": r.get("passing_yards", 0),
                "passing_tds": r.get("passing_tds", 0),
                "rushing_yards": r.get("rushing_yards", 0),
                "rushing_tds": r.get("rushing_tds", 0),
                "receiving_yards": r.get("receiving_yards", 0),
                "receiving_tds": r.get("receiving_tds", 0),
                "receptions": r.get("receptions", 0),
                "targets": r.get("targets", 0),
                "carries": r.get("carries", 0),
            }
            for i, r in enumerate(rows)
        ]
    )
    df, _ = player_usage.build_player_training_frame(frame)
    return df["anytime_td"].tolist()


def test_a_qb_game_with_passing_tds_only_is_not_an_anytime_td():
    """The regression this whole block exists for.

    Three passing TDs and nothing else. Under the old label this row was
    `anytime_td == 1`; under the new definition it is `0`. If this test ever fails
    someone has put `passing_tds` back into the sum.
    """
    assert _labels_for([{"passing_yards": 280, "passing_tds": 3}]) == [0]


def test_a_qb_rushing_td_is_an_anytime_td():
    """The other half of the same rule: dropping passing must not drop rushing.

    One passing TD and one rushing TD -- only the rushing one may fire.
    """
    assert _labels_for([{"passing_yards": 180, "passing_tds": 1, "carries": 6, "rushing_tds": 1}]) == [1]


def test_a_qb_receiving_td_is_still_an_anytime_td():
    """Sneakiest half of the trap: a WR-shaped passer.

    A player who happens to throw AND catch a TD scores the same way a pure
    receiver does. Only passing is excluded, so the receiving TD carries it.
    """
    assert _labels_for([{"passing_tds": 2, "receiving_tds": 1, "receptions": 3}]) == [1]


def test_a_qb_with_neither_kind_of_td_is_not_an_anytime_td():
    """The null case, so the exclusions above are not passing by accident."""
    assert _labels_for([{"passing_yards": 210, "passing_tds": 0, "rushing_yards": 4}]) == [0]


def test_wr_and_rb_label_behaviour_is_unchanged_by_the_definition_change():
    """Pin what must NOT move. These three rows label identically under the old
    `(rushing + receiving + passing) > 0` and the new `(rushing + receiving) > 0`:
    an RB's rushing TD, a WR's receiving TD, and a WR with no TD at all. Only the
    QB rows above change. Asserting them explicitly is what stops a future edit
    from quietly re-scoping the label for the positions the change was not about.
    """
    assert _labels_for([{"position": "RB", "carries": 20, "rushing_tds": 2}]) == [1]
    assert _labels_for([{"position": "WR", "receiving_tds": 1, "receptions": 6}]) == [1]
    assert _labels_for([{"position": "WR", "receiving_yards": 44, "receptions": 4}]) == [0]


def test_the_label_treats_a_missing_td_column_as_zero():
    """`fillna(0)` is load-bearing: a NaN rushing TD must not make the sum NaN.

    `NaN > 0` is False in pandas, so without the fillna a QB game with a missing
    `rushing_tds` would label 0 anyway -- but a missing *receiving* TD alongside a
    real rushing TD would, because NaN + 1 is NaN and NaN > 0 is False. This pins
    the null-handling so it cannot be dropped.
    """
    frame = pd.DataFrame(
        [{
            "player_id": "p1", "player_name": "Player", "position": "RB", "recent_team": "BAL",
            "season": 2025, "week": 1,
            "passing_yards": 0, "passing_tds": None, "rushing_yards": 90, "rushing_tds": 1,
            "receiving_yards": None, "receiving_tds": None, "receptions": None,
            "targets": None, "carries": 14,
        }]
    )
    df, _ = player_usage.build_player_training_frame(frame)
    assert df["anytime_td"].tolist() == [1]


def test_the_label_change_does_not_move_the_feature_columns():
    """The change is the LABEL only, so nothing about the models' shape moved.

    `PLAYER_FEATURE_COLUMNS` is what every fitted player model and
    `predict_props`' reindex is keyed on. If this test fails the definition change
    has leaked into the feature set, and every committed artefact is wrong in shape
    rather than only in target.
    """
    assert player_usage.PLAYER_FEATURE_COLUMNS == [
        "passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll",
        "targets_roll", "carries_roll", "receptions_roll",
    ]
    assert "anytime_td" not in player_usage.PLAYER_FEATURE_COLUMNS


def test_serving_does_not_recompute_the_label_so_a_committed_model_cannot_self_update():
    """What the label change can and cannot reach.

    Serving calls `predict_props`, which builds its design matrix from the feature
    row and the committed pickle. It never asks what `anytime_td` "is", so a
    committed `anytime_td_model.pkl` keeps predicting the OLD definition no matter
    what this module says until it is refitted. The test asserts that shape --
    `predict_props` needs only the feature row -- which is why the PR has to say
    the committed artefact is stale rather than implying the fix is live.
    """
    class _Prob:
        def predict_proba(self, X):
            # Indexed `[0, 1]` by `predict_props`, so return a numpy-ish 2-D
            # array rather than a nested list.
            return np.array([[0.5, 0.5]])

    out = player_props.predict_props(
        {"feature_cols": ["rushing_yards_roll"], "anytime_td": _Prob()},
        pd.Series({"rushing_yards_roll": 80.0}),
        "RB",
    )
    assert out["anytime_td_prob"] == pytest.approx(0.5)


import pytest  # noqa: E402  (kept local to the test that needs it)

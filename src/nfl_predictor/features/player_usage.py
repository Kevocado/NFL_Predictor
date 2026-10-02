"""player_usage.py — player-level rolling usage/production features for
anytime-TD and yardage prop models. Same shift(1)-then-rolling discipline as
features/rolling_form.py, applied per player instead of per team."""

from __future__ import annotations

import pandas as pd

ROLL_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "targets", "carries", "receptions"]
PLAYER_FEATURE_COLUMNS = [f"{stat}_roll" for stat in ROLL_STATS]

#: The window every rolling feature uses, in training and at serving alike.
DEFAULT_ROLL_WINDOW = 5

#: `passing_tds_roll` -- the rolling passing-TD count the QB passing-TD model
#: (`models/qb_passing_td.py`) is fitted on.
#:
#: It is deliberately NOT in `ROLL_STATS`, so it is not in
#: `PLAYER_FEATURE_COLUMNS`: that list is what the anytime-TD classifier and
#: every yardage regressor are fitted on and what `predict_props` indexes by
#: name, so adding a column to it would change the feature count of every
#: already-committed model. It is computed and served as its own column
#: instead, and `models/manifest.py` reads it through
#: :func:`with_passing_tds_roll` rather than rolling it a second time by hand.
PASSING_TDS_ROLL_COLUMN = "passing_tds_roll"

#: A DECLARATION of every column `build_features_for_player` emits -- the fitted
#: player features plus `passing_tds_roll`.
#:
#: **This constant is no longer what the load-time audit compares against**, and
#: that is the fix. It was: `models/manifest._assert_servable_columns` read it from
#: inside its body while its one caller passed `PLAYER_FEATURE_COLUMNS`, and since
#: this constant is DEFINED as `[*PLAYER_FEATURE_COLUMNS, PASSING_TDS_ROLL_COLUMN]`
#: the guard evaluated `X ⊆ X + 1` -- a tautology over the same declared data,
#: reading neither `build_features_for_player` nor any pickle. It could not fail.
#:
#: `models/manifest._player_serving_columns` now CALLS `build_features_for_player`
#: on a probe frame and uses what it returns. `build_features_for_player` is the
#: thing serving actually calls, so it is the thing the audit has to ask.
#:
#: It is kept because it is a public name that documents the shape of a served row,
#: and because a declared list that silently drifts is worth being able to compare
#: against. `tests/test_qb_passing_td.py` asserts the two agree, so drift here is
#: caught -- but it is a documentation constant now, not the guard.
SERVING_FEATURE_COLUMNS = [*PLAYER_FEATURE_COLUMNS, PASSING_TDS_ROLL_COLUMN]

#: Version of the `anytime_td` DEFINITION (not of the code -- of the label).
#: Bump this whenever `anytime_td_actual`'s arithmetic changes. It is recorded in
#: the manifest at fit time and checked at load time by
#: `models/manifest._verify_artifact_fingerprint`, so an artefact fitted against
#: a different definition raises instead of serving quietly.
#:
#: 1 = `rushing_tds + receiving_tds + passing_tds > 0` (superseded).
#: 2 = `rushing_tds + receiving_tds > 0` (2026-10-01; passing TDs excluded).
ANYTIME_TD_LABEL_VERSION = 2


def with_passing_tds_roll(df: pd.DataFrame, window: int = DEFAULT_ROLL_WINDOW) -> pd.DataFrame:
    """`df` plus `passing_tds_roll`, on `_add_rolling`'s exact discipline.

    Same `shift(1).rolling(window, min_periods=1).mean()` per player, for the same
    reason the other rolled stats use it: a pregame feature cannot know the game
    being predicted. It exists as one function so the training column and the
    column `build_features_for_player` emits cannot drift apart -- which is
    precisely how `passing_tds_roll` came to be fitted on and served as a
    constant zero.

    Returns a copy sorted by `["player_id", "season", "week"]` with a fresh index,
    like `_add_rolling`. `df` must have a unique index.
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    df[PASSING_TDS_ROLL_COLUMN] = df.groupby("player_id")["passing_tds"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df


def _add_rolling(df: pd.DataFrame, window: int = DEFAULT_ROLL_WINDOW) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = df.groupby("player_id")
    for stat in ROLL_STATS:
        df[f"{stat}_roll"] = grouped[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df


def anytime_td_actual(rushing_tds, receiving_tds) -> float:
    """Ground truth for the `anytime_td` market, as 0.0 or 1.0.

    THE definition, in one function, because this quantity has to be computed in
    two places that must never disagree:

    * `build_player_training_frame` below, to produce the classifier's target;
    * `tracking/store.reconcile_player_prop_predictions`, to grade a stored
      prediction. The grader used to carry its own inline copy of the sum, so
      changing the label here alone would have left the grader resolving the
      market against the OLD definition -- every QB's pick scored against a truth
      the model was not fitted on. A grader that disagrees with the model is
      worse than either definition on its own.

    `anytime_td` = **rushing TDs + receiving TDs, and nothing else.** Passing TDs
    are deliberately EXCLUDED (decided 2026-10-01; both call sites previously
    summed `passing_tds` in as well).

    The reason is that the two are not the same market. "Anytime TD" reads to a
    user as a rushing-or-receiving score, but with passing included it fired on
    passing alone, so a quarterback's anytime-TD was dominated by his arm and
    quarterbacks sorted to the top of a category whose name never mentions
    passing. Passing TDs are a separate market with their own per-player model
    line: `models/qb_passing_td.py`, served as `passing_td_*` fields on QB rows
    by `player_props.predict_props` and graded through the same store under the
    `"passing_tds"` market. Nothing is lost by dropping it here -- the same
    `passing_tds` column still feeds `with_passing_tds_roll` for that model.

    Scalars or a Series; missing values read as 0, matching the `fillna(0)` the
    training frame applies and the `(x or 0)` the grader used. Returns a float
    for scalars and a float Series for a Series, so the grader's per-row call and
    the frame-level call share one arithmetic expression.
    """
    if isinstance(rushing_tds, pd.Series):
        rushing, receiving = rushing_tds.fillna(0), receiving_tds.fillna(0)
    else:
        rushing = 0 if rushing_tds is None or pd.isna(rushing_tds) else rushing_tds
        receiving = 0 if receiving_tds is None or pd.isna(receiving_tds) else receiving_tds
    scored = (rushing + receiving) > 0
    return scored.astype(float) if isinstance(rushing_tds, pd.Series) else float(scored)


def build_player_training_frame(player_stats_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _add_rolling(player_stats_df)

    # The LABEL only. `PLAYER_FEATURE_COLUMNS` is untouched, so no model changes
    # shape; and `predict_props` scores a live row from features alone, so the
    # only artefact affected is one refitted against this label.
    df["anytime_td"] = anytime_td_actual(df["rushing_tds"], df["receiving_tds"]).astype(int)
    return df, PLAYER_FEATURE_COLUMNS


def build_features_for_player(
    player_id: str,
    player_stats_df: pd.DataFrame,
    season: int | None = None,
    week: int | None = None,
    window: int = DEFAULT_ROLL_WINDOW,
) -> pd.Series | None:
    """Pregame rolling features for one player, matching the training discipline.

    `week` is the gameweek being predicted, and bounds the history to games strictly
    before it. Pass it whenever it is known -- see the comment at the `history["week"]
    < week` line for why relying on the target row's absence is not enough.

    `season` scopes the history to a single season. This is not a refinement: the
    previous version ranged over every season in the frame, so a player on a 2026
    roster was scored on whatever his last five games were in whatever year those
    were. Measured 2026-09-27, 416 of 458 non-zero served rows were predicted off
    2024 data, and because `recent_team` comes from the current roster while the
    features came from an old one, a traded player was credited to his new team on
    his former team's numbers.

    The rolling mean deliberately **excludes the most recent game**, reproducing
    `_add_rolling`'s `shift(1).rolling(window, min_periods=1)`. A pregame feature
    cannot know the game being predicted, so including it makes the served quantity
    differ from the trained one under the same column name. That skew was real and
    measurable: Aaron Rodgers (`00-0023459`) was served `passing_yards_roll` 254.0
    against a training-consistent 249.0.

    Returns None when the player has no rows in scope. Callers must skip rather than
    substitute a default: the model is never trained on the all-zero region, and
    scoring it returns the origin intercept (passing_yards 62.592, anytime_td_prob
    0.111846), which is how 430 of 880 live rows -- 48.9% -- came back bit-identical
    before this was fixed.
    """
    history = player_stats_df[player_stats_df["player_id"] == player_id]
    if season is not None:
        history = history[history["season"] == season]
    if week is not None:
        # The pregame view of `week`: everything strictly before it.
        #
        # This is the fix, and it is the *opposite* of dropping the last row. An
        # earlier version of this function did `history.iloc[:-1]`, reasoning that
        # "a pregame feature cannot know the game being predicted". That is wrong
        # about the serving situation: nflverse publishes only *played* weeks, so
        # the target game's row is normally absent already and `shift(1)` is
        # implicit. Dropping a row therefore subtracted a real game that training
        # includes. Measured against `_add_rolling`'s own training rows, for a
        # player with 16 played games:
        #
        #     target week   training row   iloc[:-1] (was)   tail(5)
        #              6           30.0            25.0         30.0
        #             10           70.0            60.0         70.0
        #             12           90.0            80.0         90.0
        #             16          130.0           120.0        130.0
        #
        # `iloc[:-1]` matched training at no target week; `tail(window)` matches at
        # every one. So the "fix" introduced a skew that was not there.
        #
        # Passing `week` explicitly is still strictly better than relying on that
        # absence: `_load_player_history(season)` returns *every* played week, so
        # when a team's week-N game has not kicked off but other week-N games have
        # finished, the target row IS present and the window would include the
        # game being predicted. The explicit bound closes that too.
        history = history[history["week"] < week]
    history = history.sort_values(["season", "week"])
    if history.empty:
        return None
    prior = history.tail(window)
    return pd.Series(
        {
            f"{stat}_roll": float(prior[stat].mean()) if not prior.empty else float("nan")
            for stat in ROLL_STATS
        }
        | {
            # `passing_tds_roll`, the QB passing-TD model's own rolling feature,
            # on the same discipline as every neighbour above: the mean of the
            # `window` games strictly before the target week, which is exactly
            # what `shift(1).rolling(window, min_periods=1).mean()` produces for
            # the target row in training (`with_passing_tds_roll`). It was fitted
            # on but never emitted here, so every QB was projected from
            # `fillna(0)` on this column -- a constant-zero feature against a
            # fitted coefficient.
            #
            # `models/manifest.py::_assert_artefact_columns_are_served`, reached
            # from `load_models`, now audits this model's OWN fitted columns against
            # what THIS FUNCTION returns -- read by calling it, not by reading
            # `SERVING_FEATURE_COLUMNS`. Note that a subset rule alone would not
            # have caught the original bug's inverse: the model records a column
            # the builder does not emit, which is the subset direction, but a
            # model missing a column is the direction that needs the exact
            # comparison the audit also makes.
            PASSING_TDS_ROLL_COLUMN: float(prior["passing_tds"].mean())
            if not prior.empty else float("nan")
        }
    )


# --- the load-time audit, and why it asks this function and not a constant ----
#
# `models/manifest.py` has two guards, and they check two different things. Both
# are needed, and neither subsumes the other:
#
#   * `_verify_artifact_fingerprint` pins the manifest's DECLARED feature lists
#     against the CODE's, and the `anytime_td` label version against the code's.
#     It never opens a pickle, so it catches the code having moved on.
#   * `_assert_artefact_columns_are_served` opens every artefact the payload ships
#     and reads that artefact's OWN fitted columns -- `feature_names_in_`, or
#     `feature_cols` for the mapping payloads -- and holds them against the columns
#     THIS FUNCTION emits. It is the only check that can see a pickle which
#     disagrees with its own manifest, which is exactly the case proven against
#     `origin/main`: refit `anytime_td_model.pkl` on 3 of its 6 columns, leave the
#     manifest and the fingerprint untouched, and `load_models()` returned
#     normally.
#
# So the serving side of the second guard is `build_features_for_player`'s actual
# output, read by calling it (`manifest._player_serving_columns`). It was the
# `SERVING_FEATURE_COLUMNS` constant above, and comparing that to
# `PLAYER_FEATURE_COLUMNS` -- its own first element -- is what made the old guard
# vacuous.

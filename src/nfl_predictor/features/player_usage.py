"""player_usage.py — player-level rolling usage/production features for
anytime-TD and yardage prop models. Same shift(1)-then-rolling discipline as
features/rolling_form.py, applied per player instead of per team."""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "targets", "carries", "receptions", "attempts", "completions"]
PLAYER_FEATURE_COLUMNS = [f"{stat}_roll" for stat in ROLL_STATS]

# Derived rolling features computed from the base rolling means
# These are efficiency metrics: yards per attempt, yards per completion, yards per target, yards per carry
DERIVED_ROLL_STATS = [
    "yards_per_attempt",      # passing_yards_roll / attempts_roll
    "yards_per_completion",   # passing_yards_roll / completions_roll
    "yards_per_target",       # receiving_yards_roll / targets_roll
    "yards_per_carry",        # rushing_yards_roll / carries_roll
]

# QB-specific: rushing yards and TDs for QBs (scrambles)
QB_ROLL_STATS = ["rushing_yards", "rushing_tds"]

# All feature columns: base rolling + derived + QB-specific
ALL_PLAYER_FEATURE_COLUMNS = PLAYER_FEATURE_COLUMNS + [f"{s}_roll" for s in DERIVED_ROLL_STATS] + [f"qb_{s}_roll" for s in QB_ROLL_STATS]

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

#: Forward-set team features that are NOT part of the 14-column served rolling
#: set but that the repo already computes in `features/build.py` (the forward
#: set). Serving them required loading play-by-play + schedules, which the props
#: route did not do, so they were zero-filled and dropped from the served set on
#: 2026-10-15. They are genuinely computable pre-game (lagged aggregates of
# # strictly earlier games), so they can be served without zero-filling.
FORWARD_SET_COLUMNS: tuple[str, ...] = (
    "opp_pass_yds_allowed_roll", "plays_roll", "pass_share_roll",
    "opp_plays_allowed_roll", "expected_plays",
)


def _past_rolling_mean(s: pd.Series, window: int = DEFAULT_ROLL_WINDOW) -> pd.Series:
    """shift(1)-then-rolling mean over strictly earlier games."""
    return s.shift(1).rolling(window, min_periods=1).mean()


def _games_map_from_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team) -> opponent, from the schedules pull (home + away sides)."""
    needed = {"season", "week", "home_team", "away_team"}
    if not needed.issubset(schedules.columns):
        return pd.DataFrame(columns=["season", "week", "team", "opponent"])
    home = schedules.rename(columns={"home_team": "team", "away_team": "opponent"})[
        ["season", "week", "team", "opponent"]
    ]
    away = schedules.rename(columns={"away_team": "team", "home_team": "opponent"})[
        ["season", "week", "team", "opponent"]
    ]
    return pd.concat([home, away], ignore_index=True).drop_duplicates()


def build_forward_tables(
    pbp: pd.DataFrame,
    schedules: pd.DataFrame,
    position_by_player: dict[str, str] | None = None,
    window: int = DEFAULT_ROLL_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-set team features, lagged (strictly earlier games), for serving.

    Mirrors `features/interactions.py` (`add_pbp_features`) and
    `features/matchup.py` (`_opponent_allowed`) so training and serving compute
    identical values. Every value is computed from games strictly before the
    row's week, so all five features are pre-game knowable.

    Returns `(own_forward, opp_forward)`:

    * `own_forward`: index `(season, week, team)` -> `plays_roll`,
      `pass_share_roll` (the player's own team).
    * `opp_forward`: index `(season, week, defteam, position)` ->
      `opp_plays_allowed_roll` (team-level) and `opp_pass_yds_allowed_roll`
      (position-specific: passing yards the defence allowed to that position).

    `expected_plays` is derived at lookup time as
    `own_team.plays_roll * opponent.opp_plays_allowed_roll / 100`, mirroring
    `features/interactions.py`.
    """
    plays = pbp[
        (pbp["pass_attempt"].fillna(0).astype(bool))
        | (pbp["rush_attempt"].fillna(0).astype(bool))
    ].copy()

    empty_own = pd.DataFrame(
        columns=["season", "week", "team", "plays_roll", "pass_share_roll"]
    )
    empty_opp = pd.DataFrame(
        columns=[
            "season", "week", "defteam", "position",
            "opp_plays_allowed_roll", "opp_pass_yds_allowed_roll",
        ]
    )
    if plays.empty:
        return empty_own.set_index(["season", "week", "team"]), empty_opp.set_index(
            ["season", "week", "defteam", "position"]
        )

    # ---- own-team offense (team level) -------------------------------------
    own = (
        plays.groupby(["season", "week", "posteam"], as_index=False)
        .agg(plays=("play_id", "size"), pass_attempts=("pass_attempt", "sum"))
        .rename(columns={"posteam": "team"})
    )
    allowed = (
        plays.groupby(["season", "week", "defteam"], as_index=False)
        .agg(plays_allowed=("play_id", "size"))
        .rename(columns={"defteam": "team"})
    )
    team = own.merge(allowed, on=["season", "week", "team"], how="outer")
    team = team.sort_values(["team", "season", "week"]).reset_index(drop=True)
    team["plays_roll"] = team.groupby("team")["plays"].transform(_past_rolling_mean)
    team["pass_attempts_roll"] = team.groupby("team")["pass_attempts"].transform(
        _past_rolling_mean
    )
    team["plays_allowed_roll"] = team.groupby("team")["plays_allowed"].transform(
        _past_rolling_mean
    )
    team["pass_share_roll"] = team["pass_attempts_roll"] / team["plays_roll"].replace(
        0, np.nan
    )

    # opponent mapping via the schedules (team T's week-W opponent).
    gm = _games_map_from_schedules(schedules)
    if not gm.empty:
        team = team.merge(gm, on=["season", "week", "team"], how="left")
        team = team.merge(
            team[["season", "week", "team", "plays_allowed_roll"]].rename(
                columns={
                    "team": "opponent",
                    "plays_allowed_roll": "opp_plays_allowed_roll",
                }
            ),
            on=["season", "week", "opponent"],
            how="left",
        )
    team["expected_plays"] = (
        team["plays_roll"] * team["opp_plays_allowed_roll"] / 100.0
    )

    own_forward = team[["season", "week", "team", "plays_roll", "pass_share_roll"]]

    # ---- defence yards allowed to a position group (position-specific) -------
    # Mirrors matchup._opponent_allowed: group the player-level weekly frame by
    # (defteam, position) and sum the relevant yardage, then roll per (defteam,
    # position). `position_by_player` maps pbp role ids to positions; without it
    # we fall back to the passer/rusher role implied by the yardage column.
    if position_by_player is None:
        position_by_player = {}

    def _pos(pid, fallback: str) -> str:
        return position_by_player.get(pid, fallback)

    pass_plays = plays[plays["pass_attempt"].fillna(0).astype(bool)].copy()
    if not pass_plays.empty and "passing_yards" in pass_plays.columns:
        pass_plays = pass_plays.copy()
        pass_plays["position"] = [
            _pos(p, "QB") for p in pass_plays["passer_player_id"]
        ]
        pass_allowed = (
            pass_plays.groupby(["season", "week", "defteam", "position"], as_index=False)
            .agg(allowed_pass_yards=("passing_yards", "sum"))
            .rename(columns={"defteam": "defteam"})
        )
        pass_allowed = pass_allowed.sort_values(
            ["defteam", "position", "season", "week"]
        ).reset_index(drop=True)
        pass_allowed["opp_pass_yds_allowed_roll"] = pass_allowed.groupby(
            ["defteam", "position"]
        )["allowed_pass_yards"].transform(_past_rolling_mean)
    else:
        pass_allowed = pd.DataFrame(
            columns=["season", "week", "defteam", "position", "opp_pass_yds_allowed_roll"]
        )

    opp_forward = pass_allowed.merge(
        team[["season", "week", "team", "plays_allowed_roll"]].rename(
            columns={"team": "defteam", "plays_allowed_roll": "opp_plays_allowed_roll"}
        ),
        on=["season", "week", "defteam"],
        how="left",
    )
    opp_forward = opp_forward[
        [
            "season", "week", "defteam", "position",
            "opp_plays_allowed_roll", "opp_pass_yds_allowed_roll",
        ]
    ]

    return (
        own_forward.set_index(["season", "week", "team"]),
        opp_forward.set_index(["season", "week", "defteam", "position"]),
    )
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
SERVING_FEATURE_COLUMNS = [*ALL_PLAYER_FEATURE_COLUMNS, PASSING_TDS_ROLL_COLUMN]

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
    # QB-specific rolling: rushing yards and TDs for QBs. The columns are
    # created unconditionally -- `build_player_training_frame` promises
    # `ALL_PLAYER_FEATURE_COLUMNS` and `build_features_for_player` always
    # emits them (NaN for non-QBs), so a frame without a `position` column
    # must still carry them rather than KeyError downstream. Non-QB rows
    # (and frames with no position at all) get NaN, matching serving.
    for stat in QB_ROLL_STATS:
        rolled = grouped[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        if "position" in df.columns:
            df[f"qb_{stat}_roll"] = rolled.where(df["position"] == "QB")
        else:
            df[f"qb_{stat}_roll"] = np.nan
    return df


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived efficiency features from base rolling means.
    
    These are efficiency ratios: yards per attempt, yards per completion, 
    yards per target, yards per carry.
    """
    df = df.copy()
    # yards_per_attempt = passing_yards_roll / attempts_roll
    df["yards_per_attempt_roll"] = np.where(
        df["attempts_roll"] > 0,
        df["passing_yards_roll"] / df["attempts_roll"],
        np.nan
    )
    # yards_per_completion = passing_yards_roll / completions_roll
    df["yards_per_completion_roll"] = np.where(
        df["completions_roll"] > 0,
        df["passing_yards_roll"] / df["completions_roll"],
        np.nan
    )
    # yards_per_target = receiving_yards_roll / targets_roll
    df["yards_per_target_roll"] = np.where(
        df["targets_roll"] > 0,
        df["receiving_yards_roll"] / df["targets_roll"],
        np.nan
    )
    # yards_per_carry = rushing_yards_roll / carries_roll
    df["yards_per_carry_roll"] = np.where(
        df["carries_roll"] > 0,
        df["rushing_yards_roll"] / df["carries_roll"],
        np.nan
    )
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
    df = _compute_derived_features(df)

    # The LABEL only. `ALL_PLAYER_FEATURE_COLUMNS` is the complete feature set.
    df["anytime_td"] = anytime_td_actual(df["rushing_tds"], df["receiving_tds"]).astype(int)
    return df, ALL_PLAYER_FEATURE_COLUMNS


def build_features_for_player(
    player_id: str,
    player_stats_df: pd.DataFrame,
    season: int | None = None,
    week: int | None = None,
    window: int = DEFAULT_ROLL_WINDOW,
    forward_tables: tuple[pd.DataFrame, pd.DataFrame] | None = None,
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

    `forward_tables` is the optional precomputed output of
    `build_forward_tables(pbp, schedules)` -- `(own_forward, opp_forward)` -- used
    to serve the forward-set team features (`plays_roll`, `pass_share_roll`,
    `opp_plays_allowed_roll`, `opp_pass_yds_allowed_roll`, `expected_plays`). They
    are read off the player's latest pregame game, so they carry no lookahead. When
    omitted (or a lookup is unavailable) the forward features are NaN: callers must
    not fabricate a zero for them, since retraining on the served set keeps
    training/serve parity.
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
    # Get player position for QB-specific features
    player_pos = prior["position"].iloc[-1] if "position" in prior.columns and not prior.empty else None
    is_qb = (player_pos == "QB")

    base_features = {
        f"{stat}_roll": float(prior[stat].mean()) if not prior.empty else float("nan")
        for stat in ROLL_STATS
    }
    # Derived efficiency features
    derived_features = {}
    if not prior.empty:
        # yards_per_attempt = passing_yards_roll / attempts_roll
        if prior["attempts"].mean() > 0:
            derived_features["yards_per_attempt_roll"] = float(prior["passing_yards"].mean() / prior["attempts"].mean())
        else:
            derived_features["yards_per_attempt_roll"] = float("nan")
        # yards_per_completion
        if prior["completions"].mean() > 0:
            derived_features["yards_per_completion_roll"] = float(prior["passing_yards"].mean() / prior["completions"].mean())
        else:
            derived_features["yards_per_completion_roll"] = float("nan")
        # yards_per_target
        if prior["targets"].mean() > 0:
            derived_features["yards_per_target_roll"] = float(prior["receiving_yards"].mean() / prior["targets"].mean())
        else:
            derived_features["yards_per_target_roll"] = float("nan")
        # yards_per_carry
        if prior["carries"].mean() > 0:
            derived_features["yards_per_carry_roll"] = float(prior["rushing_yards"].mean() / prior["carries"].mean())
        else:
            derived_features["yards_per_carry_roll"] = float("nan")
    else:
        derived_features = {f"{f}_roll": float("nan") for f in DERIVED_ROLL_STATS}

    # QB-specific rushing features
    qb_features = {}
    if is_qb and not prior.empty:
        qb_features["qb_rushing_yards_roll"] = float(prior["rushing_yards"].mean())
        qb_features["qb_rushing_tds_roll"] = float(prior["rushing_tds"].mean())
    else:
        qb_features = {"qb_rushing_yards_roll": float("nan"), "qb_rushing_tds_roll": float("nan")}

    # Forward-set team features (genuinely computed, never zero-filled). They are
    # read off the player's latest pregame game (`prior.iloc[-1]`) via the
    # precomputed `forward_tables`, so they describe strictly earlier games and
    # carry no lookahead. If a lookup is unavailable the feature is NaN -- the
    # caller (props route) must not fabricate a zero for it; retraining on the
    # same served set keeps training/serve parity.
    forward_features = {f: float("nan") for f in FORWARD_SET_COLUMNS}
    if forward_tables is not None and not prior.empty:
        own_forward, opp_forward = forward_tables
        latest = prior.iloc[-1]
        s = int(latest["season"])
        w = int(latest["week"])
        team = latest["recent_team"]
        opponent = latest["opponent_team"]
        position = latest["position"]
        try:
            orow = own_forward.loc[(s, w, team)]
            forward_features["plays_roll"] = float(orow["plays_roll"])
            forward_features["pass_share_roll"] = float(orow["pass_share_roll"])
        except (KeyError, TypeError):
            pass
        try:
            orow = opp_forward.loc[(s, w, opponent, position)]
            forward_features["opp_plays_allowed_roll"] = float(orow["opp_plays_allowed_roll"])
            forward_features["opp_pass_yds_allowed_roll"] = float(orow["opp_pass_yds_allowed_roll"])
        except (KeyError, TypeError):
            pass
        if pd.notna(forward_features["plays_roll"]) and pd.notna(
            forward_features["opp_plays_allowed_roll"]
        ):
            forward_features["expected_plays"] = (
                forward_features["plays_roll"]
                * forward_features["opp_plays_allowed_roll"]
                / 100.0
            )

    return pd.Series(
        base_features
        | derived_features
        | qb_features
        | forward_features
        | {
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


#: The stats a reader compares a projection against. `qb_rushing_yards` is a QB's rushing_yards.
RECENT_AVG_STATS = ("passing_yards", "rushing_yards", "receiving_yards", "carries", "receptions")


def recent_averages(
    player_id: str,
    player_stats_df: pd.DataFrame,
    season: int,
    week: int,
    window: int = DEFAULT_ROLL_WINDOW,
) -> dict:
    """What a reader means by "his recent average": the plain mean of his last `window` games this season
    before `week`, INCLUDING the most recent one.

    Deliberately NOT `*_roll`: that feature excludes the latest game (training discipline, see
    `build_features_for_player`), so showing it as "last 5 games" would silently be games n-6..n-2.
    `games` is how many games the mean actually covers (fewer than `window` early in a season), so the
    UI can say "3-game avg" instead of pretending. Empty dict when he has no game in scope.
    """
    h = player_stats_df
    if not {"player_id", "season", "week"}.issubset(h.columns):
        return {}  # an enhancement: a frame without the keys has no recent form to show, and must not fail the props
    h = h[(h["player_id"] == player_id) & (h["season"] == season) & (h["week"] < week)]
    h = h.sort_values("week").tail(window)
    if h.empty:
        return {}
    out: dict = {"games": int(len(h))}
    for stat in RECENT_AVG_STATS:
        if stat in h.columns:
            v = pd.to_numeric(h[stat], errors="coerce").mean()
            if pd.notna(v):
                out[stat] = round(float(v), 1)
    if "rushing_yards" in out:
        out["qb_rushing_yards"] = out["rushing_yards"]
    return out

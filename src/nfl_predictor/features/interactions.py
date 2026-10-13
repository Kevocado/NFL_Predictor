"""interactions.py -- interaction features: products of *independent* signals.

**Why this module exists.** An ablation study (2026-10-12) proved the
`feat/derived-player-features` efficiency ratios add nothing on held-out data:
they are deterministic transforms of features already in the model, so a
depth-3 XGBoost learns nothing new. What a depth-3 tree genuinely cannot learn
is a *product of independent signals* -- it would have to burn two of its
three split levels rediscovering a multiplication. This module builds those
products explicitly.

Batch 1 (validated 2026-10-12, re-measured after fixing an opponent-tempo
mapping bug: mean yardage ΔMAE -0.082, 17/21 folds):
expected_air_yards, target_concentration, wind_pass_interaction, expected_plays.
Batch 2 (rz_carry_share, pressure_exposure, protection_time, gunslinger):
built and correctness-tested, but measured +0.026 / 4/21 -- kept as
EXPERIMENTAL_COLUMNS, not promoted, pending better data (true depth-chart
OL-out, TD-prop metrics).

**Leakage discipline** (same as `features/availability.py`): every post-game
observation -- rolls, shares, NGS, pbp-derived quantities -- is shift(1)-lagged
so the value on a week-W row is built only from weeks < W. Pre-kickoff facts
(injury designations, wind, the opponent) land on the week-W row unlagged.
Betting lines stay out entirely, per standing repo policy.
"""

from __future__ import annotations

import pandas as pd

from nfl_predictor.features.availability import _lagged_rolling

#: Rolling window, matching `player_usage.DEFAULT_ROLL_WINDOW`.
ROLL_WINDOW = 5

_KEYS = ["player_id", "season", "week"]

#: NGS columns this module featurizes, per stat type. The repo pulls all three
#: NGS stat types (`data/season_pull.pull_ngs`) but `features/availability.py`
#: only ever consumed `avg_separation`. Verified against the NGS parquet
#: schema. Only the columns feeding an interaction are listed -- the rest of
#: the pulled-but-unused NGS columns measured as noise in the 2026-10-12
#: experiment and are deliberately not featurized here.
#:
#: NOTE: both the passing and receiving tables carry `avg_intended_air_yards`
#: (a QB's vs a receiver's). They are different quantities, so the merge below
#: renames them apart -- see `_NGS_RENAMES`.
NGS_COLUMNS = {
    "passing": ["avg_time_to_throw", "avg_intended_air_yards", "aggressiveness"],
    "receiving": ["avg_intended_air_yards"],
}

#: Disambiguation for NGS columns present in more than one stat type.
_NGS_RENAMES = {
    ("passing", "avg_intended_air_yards"): "intended_air_yards_qb",
    ("receiving", "avg_intended_air_yards"): "intended_air_yards_recv",
}

#: The four VALIDATED interaction features (batch 1). Measured 2026-10-12
#: (corrected 2026-10-12 after fixing an opponent-tempo mapping bug):
#: mean yardage ΔMAE -0.082, wins 17/21 folds. These are the ones promoted
#: to FORWARD_FEATURE_COLUMNS.
INTERACTION_COLUMNS = [
    "expected_air_yards",
    "target_concentration",
    "wind_pass_interaction",
    "expected_plays",
]

#: Batch 2 (added 2026-10-12). Built and correctness-tested, but the
#: 2026-10-12 experiment measured them as a group at mean yardage ΔMAE
#: +0.026 (4/21 folds) -- they do not clear the bar, so they stay OUT of
#: FORWARD_FEATURE_COLUMNS. Likely causes: `pressure_exposure` /
#: `protection_time` are sparse (OL-out is rare, and the experiment's
#: OL-out proxy overcounts vs the depth-chart version); `gunslinger` /
#: `protection_time` are QB-only constants elsewhere; `rz_carry_share` is a
#: TD-prop feature scored here on yardage. Revisit with the true
#: depth-chart `ol_injuries_out` and TD-prop metrics before promoting.
EXPERIMENTAL_COLUMNS = [
    "rz_carry_share",
    "pressure_exposure",
    "protection_time",
    "gunslinger",
]

#: Injury designations that remove a player from the target pool.
_OUT_DESIGNATIONS = frozenset({"Out", "Doubtful"})

#: Skill positions whose absence funnels targets to teammates.
_SKILL_POSITIONS = frozenset({"WR", "TE", "RB"})

#: OL positions, for the experiment-side `ol_injuries_out` proxy. Production
#: computes this from depth charts (`availability._ol_injuries_out`); the
#: experiment harness approximates it -- see `experiment_ol_injuries_out`.
_OL_POSITIONS = frozenset({"C", "G", "T"})


def add_ngs_features(df: pd.DataFrame, ngs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Lagged rolling means of the NGS columns the interactions need.

    `ngs` maps stat type -> raw NGS frame (as `pull_ngs` returns per type).
    NaN where the stat type does not apply to the player (a WR has no
    `avg_time_to_throw`); XGBoost treats that as missing natively.
    """
    out = df.sort_values(_KEYS).reset_index(drop=True)
    for stat_type, columns in NGS_COLUMNS.items():
        frame = ngs.get(stat_type)
        renames = {c: _NGS_RENAMES.get((stat_type, c), c) for c in columns}
        if frame is None or frame.empty or "player_gsis_id" not in frame.columns:
            for column in renames.values():
                if column not in out.columns:
                    out[column] = float("nan")
            continue
        use = [c for c in columns if c in frame.columns]
        per_week = (frame.groupby(["player_gsis_id", "season", "week"], as_index=False)[use]
                    .mean().rename(columns={"player_gsis_id": "player_id",
                                             **{c: renames[c] for c in use}}))
        # Both sides sorted by the same unique keys, so positional assignment
        # below is safe. (A pandas Series assignment here would align on the
        # shuffled index labels and silently scramble the rows -- that bug
        # cost an experiment run on 2026-10-12. `.equals` treats NaN keys
        # -- the weekly frame has some NaN player_id rows -- as equal.)
        merged = (out.merge(per_week, on=_KEYS, how="left")
                  .sort_values(_KEYS).reset_index(drop=True))
        assert merged[_KEYS].equals(out[_KEYS]), "key order drift"
        for column in use:
            out[renames[column]] = _lagged_rolling(
                merged[renames[column]], merged["player_id"], ROLL_WINDOW).values
        for column in columns:
            if column not in use and renames[column] not in out.columns:
                out[renames[column]] = float("nan")
    return out


def split_ngs_by_stat_type(ngs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split `season_pull.pull_ngs`'s concatenated frame back into stat types.

    The pull concatenates the three `import_ngs_data` tables, which carry
    disjoint column sets and no stat-type marker, so the split keys on
    non-null marker columns that exist in exactly one table each (verified
    against the NGS parquet schema 2026-10-12). A row missing its table's
    marker is dropped -- it carries no usable column for the interactions
    anyway.
    """
    if ngs is None or ngs.empty:
        return {}
    markers = {
        "passing": ["avg_time_to_throw", "aggressiveness"],
        "receiving": ["avg_cushion"],
        "rushing": ["efficiency"],
    }
    out = {}
    for stat, cols in markers.items():
        use = [c for c in cols if c in ngs.columns]
        if use:
            mask = ngs[use].notna().any(axis=1)
            out[stat] = ngs[mask].copy()
    return out


def _pbp_team_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team): plays, pass attempts, plays allowed, pressure.

    A play counts when it is a pass attempt or rush attempt. `plays_allowed`
    is the opponent's play count -- read off `defteam`, which is how many
    plays the defense was on the field for. Pressure generated is sacks plus
    QB hits per dropback; in nflverse pbp every sack carries pass_attempt=1
    (verified 2026-10-12: 1392/1392), so dropbacks == pass attempts and no
    double-counting correction is needed.
    """
    plays = pbp[(pbp["pass_attempt"] == 1) | (pbp["rush_attempt"] == 1)]
    own = (plays.groupby(["season", "week", "posteam"], as_index=False)
           .agg(plays=("play_id", "size"),
                pass_attempts=("pass_attempt", "sum"))
           .rename(columns={"posteam": "team"}))
    allowed = (plays.groupby(["season", "week", "defteam"], as_index=False)
               .agg(plays_allowed=("play_id", "size"))
               .rename(columns={"defteam": "team"}))
    team = own.merge(allowed, on=["season", "week", "team"], how="outer")

    if "sack" in pbp.columns and "qb_hit" in pbp.columns:
        drop = pbp[pbp["pass_attempt"] == 1].copy()
        drop["pressure_event"] = drop["sack"].fillna(0) + drop["qb_hit"].fillna(0)
        press = (drop.groupby(["season", "week", "defteam"], as_index=False)
                 .agg(pressure_events=("pressure_event", "sum"),
                      dropbacks=("pass_attempt", "sum"))
                 .rename(columns={"defteam": "team"}))
        press["pressure_rate"] = (
            press["pressure_events"] / press["dropbacks"].replace(0, float("nan")))
        team = team.merge(press[["season", "week", "team", "pressure_rate"]],
                          on=["season", "week", "team"], how="left")
    return team


def _pbp_player_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season, week) raw pbp-derived quantities. NOT lagged here;
    the caller lags, so the lag discipline lives in one place.

    Returns one row per (player, season, week) -- a mid-week trade leaves a
    player under two `posteam`s, so the team with the most touches wins the
    row rather than duplicating it (a duplicate would double the weekly row
    on merge, and then `shift(1)` could hand a player their own same-week
    value as their lagged feature).
    """
    keys = ["player_id", "season", "week"]
    out = pd.DataFrame(columns=keys)

    # Targets and team targets -> target share (feeds expected_air_yards,
    # target_concentration).
    if "receiver_player_id" in pbp.columns:
        recv = pbp[pbp["receiver_player_id"].notna()].copy()
        recv["player_id"] = recv["receiver_player_id"]
        tgt = (recv.groupby(keys + ["posteam"], as_index=False)
               .agg(targets_pbp=("player_id", "size")))
        team_targets = (recv.groupby(["season", "week", "posteam"], as_index=False)
                        .agg(team_targets=("player_id", "size")))
        tgt = tgt.merge(team_targets, on=["season", "week", "posteam"], how="left")
        tgt["target_share_raw"] = tgt["targets_pbp"] / tgt["team_targets"].replace(0, float("nan"))
        tgt = tgt.sort_values("targets_pbp", ascending=False).drop_duplicates(keys)
        out = out.merge(tgt[keys + ["target_share_raw"]], on=keys, how="outer")

    # Goal-to-go carries and team goal-to-go carries -> share (feeds
    # rz_carry_share). TDs concentrate at the goal line; a share *within*
    # that subset is information season totals cannot see.
    if "rusher_player_id" in pbp.columns and "goal_to_go" in pbp.columns:
        gtg = pbp[(pbp["rusher_player_id"].notna()) & (pbp["goal_to_go"] == 1)].copy()
        gtg["player_id"] = gtg["rusher_player_id"]
        own_gtg = (gtg.groupby(keys + ["posteam"], as_index=False)
                   .agg(gtg_carries=("player_id", "size")))
        team_gtg = (gtg.groupby(["season", "week", "posteam"], as_index=False)
                    .agg(team_gtg_carries=("player_id", "size")))
        g = own_gtg.merge(team_gtg, on=["season", "week", "posteam"], how="left")
        g["gtg_carry_share_raw"] = g["gtg_carries"] / g["team_gtg_carries"].replace(0, float("nan"))
        g = g.sort_values("gtg_carries", ascending=False).drop_duplicates(keys)
        out = out.merge(g[keys + ["gtg_carry_share_raw"]], on=keys, how="outer")

    return out


def add_pbp_features(df: pd.DataFrame, pbp: pd.DataFrame,
                     games: pd.DataFrame) -> pd.DataFrame:
    """Lagged pbp-derived quantities + team tempo/pressure table.

    `games` maps (season, week, team) -> opponent; the experiment harness
    builds it from pbp game_ids, production uses the schedules pull (both are
    pre-kickoff knowable). Team-level rolls are kept on the frame because the
    interactions need them at scoring time too.
    """
    out = df.copy().sort_values(_KEYS).reset_index(drop=True)

    team = _pbp_team_frame(pbp).sort_values(["team", "season", "week"])
    for column in ("plays", "pass_attempts", "plays_allowed", "pressure_rate"):
        if column not in team.columns:
            continue
        team[f"{column}_roll"] = _lagged_rolling(
            team[column], team["team"], ROLL_WINDOW)
    team["pass_share_roll"] = team["pass_attempts_roll"] / team["plays_roll"].replace(0, float("nan"))

    # Opponent's plays-allowed and pressure-generated, via the week-W opponent
    # (pre-kickoff knowable). NOTE: this must read the opponent's
    # `plays_allowed_roll` (how many offensive plays their defense faced),
    # NOT their `plays_roll` (their own offensive tempo) -- mapping the wrong
    # one silently redefines the feature (caught by tests 2026-10-12).
    team_plays = team[["season", "week", "team", "plays_roll", "pass_share_roll"]].rename(
        columns={"team": "recent_team"})
    opp_cols = {"team": "opponent", "plays_allowed_roll": "opp_plays_allowed_roll"}
    if "pressure_rate_roll" in team.columns:
        opp_cols["pressure_rate_roll"] = "opp_pressure_rate_roll"
    opp = team[["season", "week"] + list(opp_cols)].rename(columns=opp_cols)
    out = out.merge(team_plays, on=["season", "week", "recent_team"], how="left")
    out = out.merge(games[["season", "week", "team", "opponent"]].rename(
        columns={"team": "recent_team"}), on=["season", "week", "recent_team"], how="left")
    out = out.merge(opp, on=["season", "week", "opponent"], how="left")

    player_raw = _pbp_player_frame(pbp)
    if not player_raw.empty:
        base = out[_KEYS].drop_duplicates().copy()
        merged = (base.merge(player_raw, on=_KEYS, how="left")
                  .sort_values(_KEYS).reset_index(drop=True))
        assert merged[_KEYS].equals(out[_KEYS]), "key order drift"
        for raw_col in [c for c in player_raw.columns if c.endswith("_raw")]:
            feat_col = raw_col[:-4]  # e.g. target_share_raw -> target_share
            out[feat_col] = _lagged_rolling(
                merged[raw_col], merged["player_id"], ROLL_WINDOW).values

    return out


def experiment_ol_injuries_out(injuries: pd.DataFrame) -> pd.DataFrame:
    """Experiment-side proxy for `availability._ol_injuries_out`.

    Production intersects the injury report with depth-chart starters;
    the experiment harness has no depth charts, so this counts OL-position
    players (C/G/T) listed Out on the latest regular-season report. Same
    (season, week, team) grain, same pre-kickoff timing, same "Out"-only
    rule -- it overcounts relative to production (backups included), so the
    experiment measures the signal direction, not the exact magnitude.
    """
    cols = ["season", "week", "team", "ol_injuries_out"]
    if injuries.empty or "gsis_id" not in injuries.columns:
        return pd.DataFrame(columns=cols)
    frame = injuries.copy()
    if "game_type" in frame.columns:
        frame = frame[frame["game_type"].astype(str).str.upper() == "REG"]
    if "date_modified" in frame.columns:
        frame = frame.sort_values("date_modified")
    latest = frame.groupby(["gsis_id", "season", "week"], as_index=False).last()
    out = (latest[(latest["report_status"] == "Out")
                  & (latest["position"].isin(_OL_POSITIONS))]
           .groupby(["season", "week", "team"], as_index=False).size()
           .rename(columns={"size": "ol_injuries_out"}))
    return out


def _teammates_out_doubtful(injuries: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team): skill-position teammates listed Out/Doubtful.

    The injury report is published pre-kickoff, so this lands on the target
    week unlagged -- the same discipline as `inj_Q/D/O` in availability.py.
    """
    cols = ["season", "week", "team", "teammates_out_doubtful"]
    if injuries.empty or "gsis_id" not in injuries.columns:
        return pd.DataFrame(columns=cols)
    frame = injuries.copy()
    if "game_type" in frame.columns:
        frame = frame[frame["game_type"].astype(str).str.upper() == "REG"]
    if "date_modified" in frame.columns:
        frame = frame.sort_values("date_modified")
    latest = frame.groupby(["gsis_id", "season", "week"], as_index=False).last()
    out = (latest[latest["report_status"].isin(_OUT_DESIGNATIONS)
                  & latest["position"].isin(_SKILL_POSITIONS)]
           .groupby(["season", "week", "team"], as_index=False).size()
           .rename(columns={"size": "teammates_out_doubtful"}))
    return out


def add_interaction_features(df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """The eight interaction features.

    `injuries`: raw nflverse injuries frame. `ol_injuries_out` and
    `high_wind_flag` are read off the frame itself -- in production they come
    from `add_availability_features` and `add_matchup_features` (the Open-Meteo
    forecast path), so this function must run AFTER those builders. Requires
    the lagged components from `add_ngs_features` / `add_pbp_features` to
    already be on the frame.
    """
    out = df.copy()

    # Batch 1 (validated 2026-10-12).
    # Lagged target share x lagged NGS intended air yards: volume meets depth.
    out["expected_air_yards"] = (
        out["target_share"].fillna(0) * out["intended_air_yards_recv"].fillna(0))

    # Lagged target share x pre-kickoff teammate absences: the injury-news funnel.
    team_out = _teammates_out_doubtful(injuries)
    out = out.merge(team_out.rename(columns={"team": "recent_team"}),
                    on=["season", "week", "recent_team"], how="left")
    out["teammates_out_doubtful"] = out["teammates_out_doubtful"].fillna(0)
    out["target_concentration"] = (
        out["target_share"].fillna(0) * out["teammates_out_doubtful"])

    # Pre-kickoff wind flag x lagged team pass tendency: weather only matters
    # for teams that actually throw. `high_wind_flag` is the target game's own
    # flag (a TARGET_GAME_COLUMN at serving), NOT a lagged value.
    out["wind_pass_interaction"] = (
        out["high_wind_flag"].fillna(0).astype(int) * out["pass_share_roll"].fillna(0))

    # Lagged own tempo x lagged opponent tempo-allowed: the volume denominator.
    # Scaled by 1/100: the raw product is in the thousands while every other
    # prop feature is O(1)-O(100), and the QB passing-TD model is a GLM
    # (PoissonRegressor), not a tree -- unscaled, the exp link overflows and
    # LBFGS diverges (measured 2026-10-12: +0.09 held-out NLL from numerics
    # alone). Trees are scale-invariant, so this costs them nothing.
    out["expected_plays"] = (
        out["plays_roll"].fillna(0) * out["opp_plays_allowed_roll"].fillna(0) / 100.0)

    # Batch 2 (added 2026-10-12).
    # Goal-to-go carry share: the goal-line role. TDs concentrate at the goal
    # line; a share *within* that subset is information season carry totals
    # cannot see. Already lagged by add_pbp_features.
    out["rz_carry_share"] = out["gtg_carry_share"].fillna(0)

    # Backup linemen x lagged opponent pressure rate: catastrophic only in the
    # joint region (slow release alone is fine, backups alone are survivable).
    out["pressure_exposure"] = (
        out["ol_injuries_out"].fillna(0) * out["opp_pressure_rate_roll"].fillna(0))

    # Lagged NGS time-to-throw x OL absences: slow release meets thin protection.
    # Non-QBs have no passing NGS row, so this is 0 outside QB -- a constant
    # region the trees handle natively.
    out["protection_time"] = (
        out["avg_time_to_throw"].fillna(0) * out["ol_injuries_out"].fillna(0))

    # Lagged NGS aggressiveness x lagged intended air yards: the volatile
    # high-ceiling QB archetype. Both components measured as noise standalone
    # in the 2026-10-12 experiment; their *interaction* is what is tested here.
    out["gunslinger"] = (
        out["aggressiveness"].fillna(0) * out["intended_air_yards_qb"].fillna(0))

    return out

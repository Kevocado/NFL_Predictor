"""availability.py -- injury, depth-chart, opportunity and form features.

**Two kinds of input, two different lags.** Injury designations and depth-chart
ranks are published *before* the game, so they land on the row for the week they
describe. Snap share, routes run, NGS separation and form are only observable
*after* the game, so every one of them is built on shift(1) history. Mixing
these up is the leak Review Focus#1 warns about, and
`test_post_game_features_never_see_the_target_week` pins it.

Join keys are nflverse ids throughout (`injuries.gsis_id`,
`depth_charts.gsis_id`, NGS `player_gsis_id`, all equal to `weekly.player_id`),
so nothing depends on player-name matching.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

#: nflverse `injuries.report_status` -> (inj_Q, inj_D, inj_O). Anything else
#: (IR, PUP, Reserwe, Active, ...) means no flag.
_STATUS_FLAGS = {
    "Questionable": (1, 0, 0),
    "Doubtful": (0, 1, 0),
    "Out": (0, 0, 1),
}

#: The market's yardage column, per position, for the form-deviation feature.
MARKET_TARGET = {"QB": "passing_yards", "RB": "rushing_yards",
                 "WR": "receiving_yards", "TE": "receiving_yards"}

#: Positions that constitute a starting offensive line.
OL_POSITIONS = frozenset({"C", "G", "T"})

_KEYS = ["player_id", "season", "week"]

#: How many past games each rolling window looks at.
_SHORT_WINDOW, _LONG_WINDOW = 3, 5


def _regular_season(df: pd.DataFrame) -> pd.DataFrame:
    """nflverse ships REG and POST rows for injuries and depth charts; weekly
    labels are regular season only, so POST rows must not join."""
    if "game_type" in df.columns:
        return df[df["game_type"].astype(str).str.upper() == "REG"]
    return df


def _lagged_rolling(series: pd.Series, by: pd.Series, window: int) -> pd.Series:
    """shift(1)-then-rolling: the value at row *W* is built only from rows < W."""
    return series.groupby(by).transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())


def _latest_status(designations: pd.DataFrame) -> pd.Series:
    """One status per (player, season, week).

    The injury report is revised several times a week, so the bettable status is
    the most recently modified row -- taking an arbitrary one would make the
    feature depend on nflverse's row order.
    """
    frame = designations
    if "date_modified" in frame.columns:
        frame = frame.sort_values("date_modified")
    return frame.groupby(["gsis_id", "season", "week"], as_index=False)["report_status"].last()


def _injury_features(weekly: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Designation flags for the player's own week. Not lagged: the report for
    week W is published before week W is played, so this is bettable information."""
    base = weekly[["player_id", "season", "week"]].drop_duplicates(subset=_KEYS).copy()
    for column in ("inj_Q", "inj_D", "inj_O"):
        base[column] = 0
    if injuries.empty or "gsis_id" not in injuries.columns:
        return base

    status = _latest_status(_regular_season(injuries))
    flags = status.rename(columns={"gsis_id": "player_id"})
    for index, column in enumerate(("inj_Q", "inj_D", "inj_O")):
        flags[column] = flags["report_status"].map(
            lambda s, i=index: _STATUS_FLAGS.get(s, (0, 0, 0))[i]).astype(int)

    merged = base.drop(columns=["inj_Q", "inj_D", "inj_O"]).merge(flags[_KEYS + ["inj_Q", "inj_D", "inj_O"]],
                                                                  on=_KEYS, how="left")
    return merged.fillna({"inj_Q": 0, "inj_D": 0, "inj_O": 0}).astype({"inj_Q": int, "inj_D": int, "inj_O": int})


def _ol_injuries_out(weekly: pd.DataFrame, injuries: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    """Starting linemen on the player's own team listed Out for that week."""
    base = weekly[["player_id", "season", "week", "recent_team"]].drop_duplicates(subset=_KEYS).copy()
    base["ol_injuries_out"] = 0
    if (injuries.empty or depth.empty or "gsis_id" not in injuries.columns
            or "club_code" not in depth.columns):
        return base.drop(columns=["recent_team"])

    depth_reg = _regular_season(depth)
    rank = pd.to_numeric(depth_reg["depth_position"], errors="coerce")
    starters = depth_reg[depth_reg["position"].isin(OL_POSITIONS) & (rank == 1)]

    out_rows = injuries[injuries["report_status"] == "Out"]
    out_keys = set(zip(out_rows["gsis_id"], out_rows["season"], out_rows["week"]))
    counts: Counter = Counter()
    for gsis, season, week, club in zip(starters["gsis_id"], starters["season"],
                                        starters["week"], starters["club_code"]):
        if (gsis, season, week) in out_keys:
            counts[(season, week, club)] += 1

    team_of_player = base.set_index(_KEYS)["recent_team"]
    base["ol_injuries_out"] = [
        counts.get((season, week, team), 0)
        for season, week, team in zip(base["season"], base["week"], base["recent_team"])
    ]
    return base.drop(columns=["recent_team"])


def _depth_rank_change(weekly: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    """Signed week-over-week change in depth-chart rank. Positive = demoted,
    because nflverse `depth_position` counts down from 1."""
    base = weekly[_KEYS].drop_duplicates(subset=_KEYS).copy()
    if depth.empty or "gsis_id" not in depth.columns:
        base["depth_rank_change"] = float("nan")
        return base

    ranks = _regular_season(depth)
    ranks["depth_position"] = pd.to_numeric(ranks["depth_position"], errors="coerce")
    per_week = ranks.groupby(["gsis_id", "season", "week"], as_index=False)["depth_position"].min()
    per_week = per_week.sort_values(["gsis_id", "season", "week"])
    per_week["depth_rank_change"] = per_week.groupby("gsis_id")["depth_position"].diff()

    # `base` deliberately does not pre-declare the column: defining it on both
    # sides of the merge makes pandas emit depth_rank_change_x / _y.
    merged = base.merge(per_week.rename(columns={"gsis_id": "player_id"})[_KEYS + ["depth_rank_change"]],
                        on=_KEYS, how="left")
    if "depth_rank_change" not in merged.columns:
        merged["depth_rank_change"] = float("nan")
    return merged


def _participation(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (player, team, week) snaps and routes, plus the team's own totals.

    Snaps are counted, not summed: `route` is a string label, and `offense_snaps`
    is not populated consistently across nflverse seasons. Team totals are taken
    from the raw frame rather than from the melted one, because a play with no
    identified participant still counts toward its team's snaps -- dropping those
    rows would make every snap share 1.0.
    """
    player_cols = [c for c in ("receiver_player_id", "rusher_player_id", "passer_player_id")
                   if c in pbp.columns]
    ids = ["season", "week", "posteam"]
    if not player_cols:
        return pd.DataFrame(columns=["player_id", "season", "week", "posteam",
                                     "snaps", "routes", "team_snaps", "team_routes"])

    long = pbp.melt(id_vars=ids, value_vars=player_cols, value_name="gsis_id").dropna(subset=["gsis_id"])
    snaps = (long.groupby(["gsis_id", "season", "week", "posteam"], as_index=False).size()
             .rename(columns={"size": "snaps"}))

    routed = pbp[pbp["route"].notna()] if "route" in pbp.columns else pbp.iloc[0:0]
    routed_long = (routed.melt(id_vars=ids, value_vars=player_cols, value_name="gsis_id")
                   .dropna(subset=["gsis_id"])) if not routed.empty else long.iloc[0:0]
    routes = (routed_long.groupby(["gsis_id", "season", "week", "posteam"], as_index=False).size()
              .rename(columns={"size": "routes"}))

    team_snaps = pbp.groupby(ids, as_index=False).size().rename(columns={"size": "team_snaps"})
    team_routes = routed.groupby(ids, as_index=False).size().rename(columns={"size": "team_routes"})

    merged = (snaps
              .merge(routes, on=["gsis_id", "season", "week", "posteam"], how="outer")
              .merge(team_snaps, on=["season", "week", "posteam"], how="left")
              .merge(team_routes, on=["season", "week", "posteam"], how="left"))
    return merged.rename(columns={"gsis_id": "player_id"}).fillna({"snaps": 0, "routes": 0})


#: Which opportunity share a position is measured on.
_POSITION_ROLE = {"QB": "QB", "RB": "RB"}


def _role_opportunity_share(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season, week): shares on EVERY role the player touched.

    `snap_share` cannot answer "does this player start". A starting QB reads
    about 0.49 there, because `team_snaps` counts every offensive play while the
    player's own count only credits plays with an identified participant -- and
    worse, a share pooled across roles is diluted by the receivers and rushers
    who touch the very same snaps.

    So the share is taken within a role, against a team total drawn from the SAME
    population:

    * QB    -- passing attempts / team passing attempts
    * RB    -- carries / team carries
    * WR/TE -- targets / team targets

    That is sharp where `snap_share` is vague: on 2026, J. Dart reads 1.00 in the
    game he started and 0.15 in the one he came in relief, and S. Darnold 0.98
    against 0.12.

    **Every role is returned, not one.** Which share applies depends on the prop
    being priced, and only the caller knows that: a WR who throws a trick pass
    still needs his TARGET share for a receiving-yards prop. Reducing to a single
    row here is what made a trick-pass thrower look like a quarterback.

    Raw post-game observation; the caller lags it.
    """
    keys = ["player_id", "season", "week", "posteam"]
    out = keys + ["role", "own", "team_total", "share"]
    if pbp.empty:
        return pd.DataFrame(columns=out)

    frames = []

    def _role_counts(frame: pd.DataFrame, id_column: str, label: str, role: str,
                     *, exclude_passer: bool = False) -> None:
        subset = frame[frame[id_column].notna()]
        if exclude_passer and "passer_player_id" in subset.columns:
            subset = subset[subset[id_column].ne(subset.passer_player_id)]
        if subset.empty:
            return
        own = (subset.assign(player_id=subset[id_column])
               .groupby(keys, as_index=False)
               .agg(own=("player_id", "size")))
        total = (own.groupby(["season", "week", "posteam"], as_index=False)["own"]
                 .sum().rename(columns={"own": "team_total"}))
        merged = own.merge(total, on=["season", "week", "posteam"], how="left")
        frames.append(merged.assign(role=role, label=label,
                                    share=lambda d: d.own / d.team_total.replace(0, float("nan"))))

    if "passer_player_id" in pbp:
        _role_counts(pbp, "passer_player_id", "attempts", "QB")
    if "rusher_player_id" in pbp:
        _role_counts(pbp, "rusher_player_id", "carries", "RB")
    if "receiver_player_id" in pbp:
        _role_counts(pbp, "receiver_player_id", "targets", "WR", exclude_passer=True)

    if not frames:
        return pd.DataFrame(columns=out)

    long = pd.concat(frames, ignore_index=True)
    # Collapse a multi-team week to ONE player-week value per role. A player traded
    # mid-week appears under two `posteam`s, and leaving both would duplicate the
    # weekly row on merge -- after which `shift(1)` could hand a player their own
    # same-week value as their lagged feature. Counts are summed rather than
    # shares averaged, so the numerator and denominator stay consistent.
    per_role = (long.groupby(["player_id", "season", "week", "role"], as_index=False)
                [["own", "team_total"]].sum())
    per_role["share"] = per_role["own"] / per_role["team_total"].replace(0, float("nan"))
    return per_role[["player_id", "season", "week", "role", "share"]]


def add_opportunity_share(df: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Attach the role share for the player's OWN position, plus its raw form.

    Three columns, and the distinction matters:

    * `opp_share_raw` -- this week's observed share, NOT lagged. Excluded from
      `FORWARD_FEATURE_COLUMNS` and must stay so; it exists only so forward
      serving can build the target week's lagged value itself.
    * `opp_share` -- the LAGGED feature the models are fitted on. Week W sees only
      weeks before W.

    The raw column is what makes serving correct. `tracking.forward_tick.
    history_row_for` used to copy the previous row's already-lagged `opp_share`,
    which means a week-3 prop saw `mean(raw wk1)` while training's week-3 row saw
    `mean(raw wk1, raw wk2)` -- one game stale on exactly the feature meant to
    catch a QB change. Computing the target week's lag from the raw column closes
    that skew.
    """
    keys = ["player_id", "season", "week"]
    out = df.copy()
    for column in ("opp_share", "opp_share_raw"):
        if column not in out.columns:
            out[column] = float("nan")

    shares = _role_opportunity_share(pbp)
    if shares.empty:
        # Unknown, not zero and not a stale leftover: an empty pbp means the share
        # could not be observed, and retaining a previous calculation would carry
        # a value forward that no longer has an observation behind it.
        out["opp_share"] = float("nan")
        out["opp_share_raw"] = float("nan")
        return out

    # Role comes from the player's own `position`, joined on keys. Assigning a
    # Series instead would align on INDEX -- two different frames -- and quietly
    # produce NaN for every row but the first.
    position = out[keys + ["position"]] if "position" in out.columns else None
    if position is None:
        chosen = shares[shares.role == "WR"].drop(columns=["role"])
    else:
        position = position.assign(_role=position.position.map(
            lambda p: _POSITION_ROLE.get(p, "WR")))
        chosen = shares.merge(position[keys + ["_role"]], on=keys, how="inner")
        chosen = chosen[chosen.role == chosen._role].drop(columns=["role", "_role"])
    chosen = chosen.rename(columns={"share": "opp_share_raw"})

    merged = out.drop(columns=["opp_share", "opp_share_raw"]).merge(chosen, on=keys, how="left")
    merged = merged.sort_values(["player_id", "season", "week"])
    merged["opp_share"] = _lagged_rolling(merged["opp_share_raw"], merged["player_id"], _LONG_WINDOW)
    return merged


def _opportunity_features(weekly: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Snap share and route participation, both lagged off their own history."""
    base = weekly[_KEYS].drop_duplicates(subset=_KEYS).copy()
    for column in ("snap_share", "snap_share_trend", "route_participation"):
        base[column] = float("nan")
    if pbp.empty:
        return base

    part = _participation(pbp)
    if part.empty:
        return base

    part["snap_share"] = part["snaps"] / part["team_snaps"].replace(0, float("nan"))
    part["route_participation"] = part["routes"] / part["team_routes"].replace(0, float("nan"))

    per_week = (part.groupby(["player_id", "season", "week"], as_index=False)
                [["snap_share", "route_participation"]].mean())
    merged = base.drop(columns=["snap_share", "snap_share_trend", "route_participation"]).merge(
        per_week, on=_KEYS, how="left").sort_values(["player_id", "season", "week"])

    # Both windows are built from the raw weekly value and shift internally, so
    # neither can see the row it is attached to.
    raw_share, raw_routes = merged["snap_share"], merged["route_participation"]
    by_player = merged["player_id"]
    base_share = _lagged_rolling(raw_share, by_player, _LONG_WINDOW)
    recent_share = _lagged_rolling(raw_share, by_player, _SHORT_WINDOW)
    merged["snap_share"] = base_share
    merged["snap_share_trend"] = recent_share - base_share
    merged["route_participation"] = _lagged_rolling(raw_routes, by_player, _LONG_WINDOW)
    return merged


def add_form_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Last-3-games deviation from the rolling-5 mean of each player's own market,
    both computed from games strictly before this week. `df` must already be
    sorted by (player, season, week).

    Per-row rather than per-key, because the market column depends on position:
    a QB's form is passing yards, an RB's is rushing.
    """
    df["form_deviation"] = float("nan")
    target = df["position"].map(MARKET_TARGET)
    for column in target.dropna().unique():
        rows = df.index[target == column]
        values, by_player = df.loc[rows, column], df.loc[rows, "player_id"]
        df.loc[rows, "form_deviation"] = (
            _lagged_rolling(values, by_player, _SHORT_WINDOW)
            - _lagged_rolling(values, by_player, _LONG_WINDOW)
        )
    return df


def add_availability_features(weekly_df: pd.DataFrame, injuries_df: pd.DataFrame,
                              rosters_df: pd.DataFrame, ngs_df: pd.DataFrame | None = None,
                              pbp_df: pd.DataFrame | None = None) -> pd.DataFrame:
    weekly = weekly_df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    injuries = injuries_df if injuries_df is not None else pd.DataFrame()
    rosters = rosters_df if rosters_df is not None else pd.DataFrame()
    ngs = ngs_df if ngs_df is not None else pd.DataFrame()
    pbp = pbp_df if pbp_df is not None else pd.DataFrame()

    df = weekly
    for part in (_injury_features(weekly, injuries),
                 _ol_injuries_out(weekly, injuries, rosters),
                 _depth_rank_change(weekly, rosters),
                 _opportunity_features(weekly, pbp)):
        df = df.merge(part, on=_KEYS, how="left")

    # Applied AFTER the merges so it lags `opp_share` off the assembled frame,
    # exactly as `_opportunity_features` does for snap_share -- the same shift(1)
    # discipline, applied to the sharper starter proxy.
    df = add_opportunity_share(df, pbp)
    df = add_form_deviation(df)

    # Merged before the column exists locally, for the same reason as
    # depth_rank_change above.
    if not ngs.empty and "player_gsis_id" in ngs.columns and "avg_separation" in ngs.columns:
        sep = (ngs.groupby(["player_gsis_id", "season", "week"], as_index=False)["avg_separation"].mean()
               .rename(columns={"player_gsis_id": "player_id", "avg_separation": "separation_avg"}))
        df = df.merge(sep, on=_KEYS, how="left").sort_values(["player_id", "season", "week"])
    if "separation_avg" not in df.columns:
        df["separation_avg"] = float("nan")
    df["separation_avg"] = _lagged_rolling(df["separation_avg"], df["player_id"], _LONG_WINDOW)

    return df
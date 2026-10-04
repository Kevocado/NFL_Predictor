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


def _role_opportunity_share(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (player, team, week): the player's share of HIS OWN role's work.

    `snap_share` cannot answer "does this player start". A starting QB reads
    about 0.49 there, because `team_snaps` counts every offensive play while the
    player's own count only credits plays with an identified participant -- and
    worse, a share pooled across roles is diluted by the receivers and rushers
    who touch the very same snaps.

    So the share is taken within the player's role, against a team total drawn
    from the SAME population:

    * QB    -- passing attempts / team passing attempts
    * RB    -- carries / team carries
    * WR/TE -- targets / team targets

    That is sharp where `snap_share` is vague: on 2026, J. Dart reads 1.00 in the
    game he started and 0.21 in the one he came in relief, and S. Darnold 0.97
    against 0.06.

    This is a raw post-game observation; the caller lags it.
    """
    keys = ["player_id", "season", "week", "posteam"]
    if pbp.empty:
        return pd.DataFrame(columns=keys + ["opp_share"])

    frames = []
    passer = pbp[pbp.get("passer_player_id").notna()] if "passer_player_id" in pbp else None
    if passer is not None:
        attempts = (passer.assign(player_id=passer.passer_player_id)
                    .groupby(keys, as_index=False)
                    .agg(attempts=("player_id", "size")))
        team_att = (attempts.groupby(["season", "week", "posteam"], as_index=False)["attempts"]
                    .sum().rename(columns={"attempts": "team_attempts"}))
        frames.append(attempts.merge(team_att, on=["season", "week", "posteam"], how="left")
                      .assign(share=lambda d: d.attempts / d.team_attempts.replace(0, float("nan")),
                              role="QB"))

    if "rusher_player_id" in pbp:
        rushes = pbp[pbp.rusher_player_id.notna()]
        carries = (rushes.assign(player_id=rushes.rusher_player_id)
                   .groupby(keys, as_index=False)
                   .agg(carries=("player_id", "size")))
        team_car = (carries.groupby(["season", "week", "posteam"], as_index=False)["carries"]
                    .sum().rename(columns={"carries": "team_carries"}))
        frames.append(carries.merge(team_car, on=["season", "week", "posteam"], how="left")
                      .assign(share=lambda d: d.carries / d.team_carries.replace(0, float("nan")),
                              role="RB"))

    if "receiver_player_id" in pbp and "passer_player_id" in pbp:
        targets = pbp[pbp.receiver_player_id.notna() & pbp.receiver_player_id.ne(pbp.passer_player_id)]
        tgt = (targets.assign(player_id=targets.receiver_player_id)
               .groupby(keys, as_index=False)
               .agg(targets=("player_id", "size")))
        team_tgt = (tgt.groupby(["season", "week", "posteam"], as_index=False)["targets"]
                    .sum().rename(columns={"targets": "team_targets"}))
        frames.append(tgt.merge(team_tgt, on=["season", "week", "posteam"], how="left")
                      .assign(share=lambda d: d.targets / d.team_targets.replace(0, float("nan")),
                              role="WR"))

    if not frames:
        return pd.DataFrame(columns=keys + ["opp_share"])

    long = pd.concat(frames, ignore_index=True)
    # One row per player-week: a player's PRIMARY role decides which share counts.
    # Passers win ties (a backup QB still throws), and a player appearing in two
    # roles -- a QB who takes a snap at RB, a WR who throws -- is scored as the
    # passer, since that is the role his prop line depends on.
    long = (long.assign(_role_order=long.role.map({"QB": 0, "RB": 1, "WR": 2}).fillna(9))
            .sort_values(keys + ["_role_order"])
            .drop_duplicates(subset=keys, keep="first")
            .drop(columns=["_role_order"]))
    return long[keys + ["share"]].rename(columns={"share": "opp_share"})


def add_opportunity_share(df: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Attach a LAGGED role-opportunity share.

    Lagged through `_lagged_rolling`, so week W sees only weeks before W. This is
    the sharpest starter signal available without a lineups feed, and it is free:
    it comes from the play-by-play already cached.
    """
    keys = ["player_id", "season", "week"]
    if "opp_share" not in df.columns:
        df = df.copy()
        df["opp_share"] = float("nan")

    shares = _role_opportunity_share(pbp)
    if shares.empty:
        return df

    merged = df.drop(columns=["opp_share"]).merge(shares, on=keys, how="left")
    merged = merged.sort_values(["player_id", "season", "week"])
    merged["opp_share"] = _lagged_rolling(merged["opp_share"], merged["player_id"], _LONG_WINDOW)
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
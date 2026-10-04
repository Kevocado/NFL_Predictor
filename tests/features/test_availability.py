"""availability feature tests.

Two classes of input are joined here and they must be lagged differently:

* Pre-game facts -- injury designation, depth-chart rank. These are published
  before the game, so they belong on the row for the week they describe.
* Post-game observables -- snap share, routes run, NGS separation, form. These
  only exist once the game is over, so a week-W feature may not contain week W.
  `test_post_game_features_never_see_the_target_week` is the pin.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.features.availability import add_availability_features

GSIS = "00-0000001"


def _weekly(weeks, position="WR", team="A"):
    return pd.DataFrame([{
        "player_id": GSIS, "player_name": "Test Player", "position": position,
        "season": 2024, "week": w, "recent_team": team, "opponent_team": "ZZ",
        "passing_yards": 0, "rushing_yards": 0, "receiving_yards": 10 * w,
        "targets": 5, "carries": 0, "receptions": 3,
    } for w in weeks])


def _injuries(rows, game_type="REG"):
    """rows: (week, gsis, status)."""
    return pd.DataFrame([{
        "season": 2024, "week": w, "gsis_id": g, "report_status": s,
        "game_type": game_type, "position": "WR", "team": "A",
    } for w, g, s in rows])


def _depth(rows):
    """rows: (week, gsis, depth_position)."""
    return pd.DataFrame([{
        "season": 2024, "week": w, "gsis_id": g, "depth_position": d,
        "game_type": "REG", "club_code": "A", "position": "WR", "full_name": "Test Player",
    } for w, g, d in rows])


def _game(team, week, total, player=None, player_snaps=0, team_routes=0):
    """`total` pbp rows for the team that week, of which the first `player_snaps`
    involve `player`, and the first `team_routes` are pass plays with a route.

    The player's plays are part of the team total, not extra rows on top of it --
    otherwise the snap-share denominator is wrong and every share reads low.
    """
    rows = []
    for i in range(total):
        is_route = i < team_routes
        involved = player is not None and i < player_snaps
        rows.append({
            "season": 2024, "week": week, "posteam": team,
            "receiver_player_id": player if (involved and is_route) else None,
            "rusher_player_id": player if (involved and not is_route) else None,
            "passer_player_id": None, "route": "route" if is_route else None,
        })
    return rows


def _pbp(*games):
    return pd.DataFrame([row for game in games for row in game])


def _ngs(rows):
    """rows: (week, gsis, separation)."""
    return pd.DataFrame([{
        "season": 2024, "week": w, "player_gsis_id": g, "avg_separation": sep,
        "stat_type": "receiving",
    } for w, g, sep in rows])


def test_injury_designation_one_hot_on_its_own_week():
    weekly = _weekly([1, 2, 3])
    injuries = _injuries([
        (1, GSIS, "Questionable"), (2, GSIS, "Out"), (3, GSIS, "Doubtful"),
    ])

    out = add_availability_features(weekly, injuries, _depth([(1, GSIS, 1)]))
    by_week = out.set_index("week")

    assert (by_week.loc[1, ["inj_Q", "inj_D", "inj_O"]] == [1, 0, 0]).all()
    assert (by_week.loc[2, ["inj_Q", "inj_D", "inj_O"]] == [0, 0, 1]).all()
    assert (by_week.loc[3, ["inj_Q", "inj_D", "inj_O"]] == [0, 1, 0]).all()


def test_a_later_designation_never_appears_on_an_earlier_row():
    """Week 5's report says Out. Week 4's row must not know that."""
    weekly = _weekly([4, 5, 6])
    injuries = _injuries([(5, GSIS, "Out")])

    out = add_availability_features(weekly, injuries, _depth([(1, GSIS, 1)]))
    by_week = out.set_index("week")

    assert by_week.loc[4, "inj_O"] == 0
    assert by_week.loc[5, "inj_O"] == 1


def test_postseason_injury_rows_do_not_leak_into_the_regular_season():
    weekly = _weekly([1, 2])
    injuries = pd.concat([
        _injuries([(1, GSIS, "Questionable")]),
        _injuries([(1, GSIS, "Out")], game_type="POST"),
    ], ignore_index=True)

    out = add_availability_features(weekly, injuries, _depth([(1, GSIS, 1)]))

    assert out.set_index("week").loc[1, "inj_O"] == 0
    assert out.set_index("week").loc[1, "inj_Q"] == 1


def test_ol_injuries_out_counts_only_out_starting_linemen():
    ol_out, ol_q, rb = "00-0000010", "00-0000011", "00-0000012"
    weekly = _weekly([1], position="WR", team="A")
    depth = pd.concat([
        _depth([(1, ol_out, 1)]), _depth([(1, ol_q, 1)]), _depth([(1, rb, 1)]),
        # a backup lineman who is Out must not count
        _depth([(1, "00-0000013", 2)]),
    ], ignore_index=True)
    depth.loc[depth["gsis_id"].isin([ol_out, ol_q]), "position"] = "T"
    depth.loc[depth["gsis_id"].isin(["00-0000013"]), "position"] = "G"
    depth.loc[depth["gsis_id"] == rb, "position"] = "RB"

    injuries = _injuries([
        (1, ol_out, "Out"), (1, ol_q, "Questionable"), (1, rb, "Out"),
        (1, "00-0000013", "Out"),
    ])

    out = add_availability_features(weekly, injuries, depth)

    assert out.iloc[0]["ol_injuries_out"] == 1


def test_depth_rank_change_is_signed_and_lagged():
    weekly = _weekly([1, 2, 3])
    depth = _depth([(1, GSIS, 1), (2, GSIS, 1), (3, GSIS, 3)])

    out = add_availability_features(weekly, _injuries([]), depth)
    by_week = out.set_index("week")

    assert pd.isna(by_week.loc[1, "depth_rank_change"])
    assert by_week.loc[2, "depth_rank_change"] == 0
    assert by_week.loc[3, "depth_rank_change"] == 2, "moved from 1st to 3rd: demoted"


def test_post_game_features_never_see_the_target_week():
    """The central pin. Week 3's snap share is 90%; week 3's row must not
    contain it anywhere in its usage block."""
    weekly = _weekly([1, 2, 3])
    pbp = _pbp(
        _game("A", 1, 40, GSIS, player_snaps=10),
        _game("A", 2, 40, GSIS, player_snaps=10),
        _game("A", 3, 40, GSIS, player_snaps=36),
    )

    out = add_availability_features(weekly, _injuries([]), _depth([(1, GSIS, 1)]), pbp_df=pbp)
    wk3 = out[out["week"] == 3].iloc[0]

    # History through week 2 is 10 snaps of 40 = 0.25. Week 3's own 0.9 must
    # not leak in.
    assert wk3["snap_share"] == pytest.approx(0.25)
    assert wk3["snap_share_trend"] == pytest.approx(0.0)


def test_snap_share_trend_detects_a_promotion():
    # Weeks 1-5 off the field, 6-7 on it. The trend needs enough history for the
    # short and long windows to disagree: with only four weeks they coincide by
    # construction, because every lagged window sees the same three values.
    weekly = _weekly(range(1, 9))
    pbp = _pbp(*[
        _game("A", week, 40, GSIS, player_snaps=10 if week <= 5 else 30)
        for week in range(1, 9)
    ])

    out = add_availability_features(weekly, _injuries([]), _depth([(1, GSIS, 1)]), pbp_df=pbp)
    by_week = out.set_index("week")

    assert by_week.loc[2, "snap_share_trend"] == pytest.approx(0.0)
    assert by_week.loc[8, "snap_share"] > by_week.loc[2, "snap_share"]
    # Lagged 3-window at week 8 covers weeks 5-7; lagged 5-window covers 3-7.
    low, high = 0.25, 0.75
    assert by_week.loc[8, "snap_share_trend"] == pytest.approx(
        (low + high + high) / 3 - (low * 3 + high * 2) / 5)


def test_route_participation_is_lagged():
    weekly = _weekly([1, 2], position="WR")
    pbp = _pbp(
        _game("A", 1, 40, GSIS, player_snaps=8, team_routes=20),
        _game("A", 2, 40, GSIS, player_snaps=16, team_routes=20),
    )

    out = add_availability_features(weekly, _injuries([]), _depth([(1, GSIS, 1)]), pbp_df=pbp)
    by_week = out.set_index("week")

    assert pd.isna(by_week.loc[1, "route_participation"]), "no history in week 1"
    # Week 2 may only see week 1: 8 routes of 20 dropbacks, not its own 16/20.
    assert by_week.loc[2, "route_participation"] == pytest.approx(0.4)


def test_form_deviation_is_last3_minus_rolling5():
    weekly = _weekly([1, 2, 3, 4, 5, 6, 7])
    out = add_availability_features(weekly, _injuries([]), _depth([(1, GSIS, 1)]))
    by_week = out.set_index("week")

    # receiving_yards is 10*week, so the trend is a constant +10/week.
    # At week 7: last3 = (50+60+70)/3 = 60 ; rolling5 = (30+40+50+60+70)/5 = 50.
    assert by_week.loc[7, "form_deviation"] == pytest.approx(10.0)


def test_separation_is_lagged_and_nan_tolerant():
    weekly = _weekly([1, 2])
    ngs = _ngs([(1, GSIS, 3.0), (2, GSIS, 9.0)])

    out = add_availability_features(weekly, _injuries([]), _depth([(1, GSIS, 1)]), ngs_df=ngs)
    by_week = out.set_index("week")

    assert pd.isna(by_week.loc[1, "separation_avg"]), "week 1 has no prior separation"
    assert by_week.loc[2, "separation_avg"] == pytest.approx(3.0)


def test_missing_optional_frames_keep_every_row():
    weekly = _weekly([1, 2, 3])

    out = add_availability_features(weekly, pd.DataFrame(), pd.DataFrame())

    assert len(out) == 3
    assert out["snap_share"].isna().all()
    assert out["separation_avg"].isna().all()
    assert (out["ol_injuries_out"] == 0).all()
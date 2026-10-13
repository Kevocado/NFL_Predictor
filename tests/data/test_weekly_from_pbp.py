"""weekly-from-pbp tests.

nflverse's `player_stats` release stops at the 2024 season -- both the per-year
assets and the full-history file. Its play-by-play release, by contrast, runs
through the current season. So 2025+ weekly labels have to be derived from pbp.

The derivation's fidelity against the official series is measured once on real
data (2024: passing yards and receptions exact, rushing/receiving >=99.6% with
correlation >0.999; the residuals are laterals) and recorded in
docs/superpowers/forward-test/OFFLINE_GATE.md. What is pinned here is the
arithmetic, on a synthetic frame, so CI needs neither network nor cache.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.data.season_pull import weekly_from_pbp

PAS = "00-0000001"
RUS = "00-0000002"
REC = "00-0000003"


def _play(week, passer=None, rusher=None, receiver=None, posteam="A", defteam="ZZ",
          passing_yards=None, rushing_yards=None, receiving_yards=None,
          complete_pass=False, rush_attempt=False, pass_attempt=False,
          pass_touchdown=False, rush_touchdown=False, season=2025):
    return {
        "season": season, "week": week, "season_type": "REG",
        "posteam": posteam, "defteam": defteam,
        "passer_player_id": passer, "rusher_player_id": rusher,
        "receiver_player_id": receiver,
        "passer_player_name": passer, "rusher_player_name": rusher,
        "receiver_player_name": receiver,
        "passing_yards": passing_yards or 0.0, "rushing_yards": rushing_yards or 0.0,
        "receiving_yards": receiving_yards or 0.0,
        "complete_pass": complete_pass, "rush_attempt": rush_attempt,
        "pass_attempt": pass_attempt,
        "pass_touchdown": pass_touchdown, "rush_touchdown": rush_touchdown,
    }


@pytest.fixture
def pbp():
    rows = [
        # week 1
        _play(1, passer=PAS, passing_yards=250, pass_attempt=True),
        _play(1, passer=PAS, passing_yards=0, pass_attempt=True),   # incomplete
        _play(1, passer=PAS, receiver=REC, receiving_yards=40, complete_pass=True, pass_attempt=True),
        _play(1, rusher=RUS, rushing_yards=22, rush_attempt=True),
        _play(1),  # a play none of our three players participated in
        # week 2
        _play(2, passer=PAS, passing_yards=300, pass_attempt=True),
        _play(2, passer=PAS, receiver=REC, receiving_yards=15, complete_pass=True, pass_attempt=True),
        _play(2, rusher=RUS, rushing_yards=5, rush_attempt=True),
    ]
    return pd.DataFrame(rows)


def test_weekly_from_pbp_aggregates_each_role(pbp):
    weekly = weekly_from_pbp(pbp)
    row = weekly[(weekly.player_id == REC) & (weekly.week == 2)].iloc[0]

    assert weekly[(weekly.player_id == PAS) & (weekly.week == 1)]["passing_yards"].iloc[0] == 250
    assert weekly[(weekly.player_id == REC) & (weekly.week == 1)]["receiving_yards"].iloc[0] == 40
    assert weekly[(weekly.player_id == REC) & (weekly.week == 2)]["receiving_yards"].iloc[0] == 15
    assert weekly[weekly.player_id == REC]["receiving_yards"].sum() == 55
    assert row["receptions"] == 1, "one per week, two across both"
    total_rush = weekly[weekly.player_id == RUS]["rushing_yards"].sum()
    assert total_rush == 27
    assert weekly[weekly.player_id == RUS]["carries"].sum() == 2


def test_passer_gets_attempts_and_completions_not_targets(pbp):
    """`attempts`/`completions` are derived for the passer role from pbp.

    The same `pass_attempt` flag feeds both `targets` (receiver) and
    `attempts` (passer); role ownership keeps them apart, so the QB is not
    credited with targets and the WR is not credited with attempts.
    """
    weekly = weekly_from_pbp(pbp)
    pas = weekly[weekly.player_id == PAS]
    rec = weekly[weekly.player_id == REC]
    # PAS: 3 attempts week 1 (two solo, one completion to REC) + 2 week 2
    assert pas["attempts"].sum() == 5
    assert pas["completions"].sum() == 2
    assert rec["attempts"].sum() == 0
    assert rec["completions"].sum() == 0
    assert pas["targets"].sum() == 0


def test_touchdowns_credited_by_role():
    """TDs come from pbp: pass_touchdown credits passer AND receiver,
    rush_touchdown credits the rusher. Role ownership keeps them apart."""
    pbp = pd.DataFrame([
        # Passing TD: PAS throws, REC catches
        _play(1, passer=PAS, receiver=REC, receiving_yards=25, complete_pass=True,
              pass_attempt=True, pass_touchdown=True),
        # Rushing TD: RUS scores on the ground
        _play(1, rusher=RUS, rushing_yards=3, rush_attempt=True, rush_touchdown=True),
        # Non-TD plays
        _play(1, passer=PAS, passing_yards=10, pass_attempt=True),
        _play(1, rusher=RUS, rushing_yards=5, rush_attempt=True),
    ])
    weekly = weekly_from_pbp(pbp)
    pas = weekly[weekly.player_id == PAS].iloc[0]
    rus = weekly[weekly.player_id == RUS].iloc[0]
    rec = weekly[weekly.player_id == REC].iloc[0]
    assert pas["passing_tds"] == 1
    assert pas["rushing_tds"] == 0
    assert pas["receiving_tds"] == 0
    assert rus["rushing_tds"] == 1
    assert rus["passing_tds"] == 0
    assert rec["receiving_tds"] == 1
    assert rec["rushing_tds"] == 0


def test_missing_td_columns_read_as_zero():
    """Synthetic pbp without pass_touchdown/rush_touchdown (e.g. old tests)
    yields 0 TDs rather than raising -- same absent-column discipline as the
    quantile script."""
    rows = [
        {"season": 2025, "week": 1, "season_type": "REG",
         "posteam": "A", "defteam": "ZZ",
         "passer_player_id": PAS, "passer_player_name": PAS,
         "rusher_player_id": None, "rusher_player_name": None,
         "receiver_player_id": None, "receiver_player_name": None,
         "passing_yards": 100.0, "rushing_yards": 0.0, "receiving_yards": 0.0,
         "complete_pass": False, "rush_attempt": False, "pass_attempt": True},
    ]
    weekly = weekly_from_pbp(pd.DataFrame(rows))
    assert weekly["passing_tds"].iloc[0] == 0
    assert weekly["rushing_tds"].iloc[0] == 0
    assert weekly["receiving_tds"].iloc[0] == 0


def test_one_row_per_player_per_week(pbp):
    weekly = weekly_from_pbp(pbp)

    assert not weekly.duplicated(subset=["player_id", "season", "week"]).any()
    assert len(weekly) == 6, "3 players x 2 weeks"


def test_sacks_are_not_counted_as_carries(pbp):
    """A play can be a pass attempt and still not be a target or a carry."""
    sacks = pd.DataFrame([_play(1, passer=PAS, passing_yards=-8, pass_attempt=True)])
    weekly = weekly_from_pbp(sacks).set_index("player_id")

    assert weekly.loc[PAS, "carries"] == 0
    assert weekly.loc[PAS, "passing_yards"] == -8


def test_targets_exclude_non_targets(pbp):
    weekly = weekly_from_pbp(pbp)
    # REC was targeted twice, both times a completion. The passer is credited
    # with his own attempts only via passing_yards, never as targets.
    assert weekly[weekly.player_id == REC]["targets"].sum() == 2
    assert weekly[weekly.player_id == PAS]["targets"].sum() == 0


def test_teams_come_from_the_offensive_side(pbp):
    weekly = weekly_from_pbp(pbp)

    assert (weekly["recent_team"] == "A").all()
    assert (weekly["opponent_team"] == "ZZ").all()


def test_postseason_is_excluded(pbp):
    post = pbp.copy()
    post["season_type"] = "POST"
    weekly = weekly_from_pbp(post)

    assert weekly.empty


def test_player_name_is_carried_through(pbp):
    weekly = weekly_from_pbp(pbp)
    assert (weekly["player_name"] == weekly["player_id"]).all()


def test_empty_input_returns_empty_frame_with_columns():
    weekly = weekly_from_pbp(pd.DataFrame())

    assert weekly.empty
    assert {"player_id", "season", "week", "passing_yards",
            "rushing_yards", "receiving_yards"} <= set(weekly.columns)


def test_output_has_the_columns_the_feature_builders_join_on(pbp):
    """Task 4/5 merge on these; a rename here breaks every downstream join."""
    weekly = weekly_from_pbp(pbp)

    assert {"player_id", "player_name", "season", "week", "recent_team",
            "opponent_team", "passing_yards", "rushing_yards", "receiving_yards",
            "targets", "carries", "receptions"} <= set(weekly.columns)
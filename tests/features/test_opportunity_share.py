"""`opp_share` -- the role-opportunity share, and the starter signal it replaces.

Pinned by behaviour, not by non-nullness: a starter must read near 1.0 and a
relief appearance near 0, and the value attached to week W must be built only from
weeks before W.
"""
import numpy as np
import pandas as pd
import pytest

from nfl_predictor.features.availability import (
    _role_opportunity_share,
    add_opportunity_share,
)


def _same(a, b):
    """NaN-aware equality -- week 1's own value is legitimately NaN."""
    return (pd.isna(a) and pd.isna(b)) or a == b


def _pbp(rows):
    """Minimal pbp frame: (player_id, week, posteam, passer, rusher, receiver)."""
    frame = pd.DataFrame(rows, columns=["player_id", "week", "posteam",
                                        "passer_player_id", "rusher_player_id",
                                        "receiver_player_id"])
    frame["season"] = 2026
    return frame


def test_a_starter_reads_near_one_and_a_relief_appearance_near_zero():
    """The whole point of the feature. Dart starts (20 attempts to the team's 20),
    then comes in relief for 1 of 20."""
    pbp = _pbp(
        [["starter", 1, "NYG", "starter", None, "rec"]] * 20
        + [["backup", 1, "NYG", "backup", None, "rec"]]  # the other 0 attempts
    )
    pbp.loc[len(pbp) - 1, "passer_player_id"] = "backup"

    week1 = _role_opportunity_share(pbp)
    starts = float(week1.loc[week1.player_id == "starter", "opp_share"].iloc[0])

    assert starts == pytest.approx(20 / 21)

    # Same player, a week where he only relieves: 1 of 20 attempts.
    relief = _pbp([["starter", 2, "NYG", "other", None, "rec"]] * 20
                  + [["starter", 2, "NYG", "starter", None, "rec"]])
    week2 = _role_opportunity_share(relief)
    cameo = float(week2.loc[week2.player_id == "starter", "opp_share"].iloc[0])

    assert cameo == pytest.approx(1 / 21)
    assert starts > 0.9 and cameo < 0.1, "the feature must separate the two cases"


def test_a_share_never_exceeds_one():
    """Both sides of the ratio come from the same population, so it is bounded."""
    pbp = _pbp([["a", 1, "SF", "a", None, "a"]] * 30)

    shares = _role_opportunity_share(pbp)

    assert (shares.opp_share <= 1.0).all()
    assert (shares.opp_share >= 0.0).all()


def test_the_passer_role_wins_when_a_player_does_both():
    """A WR who throws is scored as the passer -- that is the role his prop line
    depends on. Pins the documented precedence.

    Built so the two candidate answers differ: he takes 6 of the team's 10 pass
    attempts (0.6) but is targeted on all 10 of them (1.0). Only 0.6 proves the
    passer role won.
    """
    pbp = _pbp(
        [["dual", 1, "SEA", "dual", None, "dual"]] * 6      # 6 attempts, all targeted
        + [["other", 1, "SEA", "other", None, "dual"]] * 4  # 4 more attempts, same target
    )

    shares = _role_opportunity_share(pbp)
    value = float(shares.loc[shares.player_id == "dual", "opp_share"].iloc[0])

    assert value == pytest.approx(0.6)


def test_the_value_on_week_w_sees_only_weeks_before_w():
    """Leakage probe. Week 2's row must reflect week 1's share alone, even when
    week 2 itself is a full start -- if it could see week 2 it would jump."""
    pbp = _pbp(
        [["p", 1, "NYG", "p", None, "r"]] * 10               # wk1: 10 of 10
        + [["p", 2, "NYG", "other", None, "r"]] * 5         # wk2: 1 of 10 would
        + [["p", 2, "NYG", "p", None, "r"]] * 4
    )
    weekly = pd.DataFrame({"player_id": ["p", "p"], "season": [2026, 2026],
                           "week": [1, 2]})

    out = add_opportunity_share(weekly, pbp).sort_values("week")

    # Week 1's own value is a shift(1) of nothing -> NaN.
    assert np.isnan(out.iloc[0]["opp_share"])
    # Week 2 sees week 1's share of 1.0, NOT week 2's share of ~0.44.
    assert out.iloc[1]["opp_share"] == pytest.approx(1.0)


def test_rewriting_a_later_week_cannot_change_an_earlier_row():
    """The strongest form of the probe: mutate the future and demand the past is
    byte-identical."""
    early = _pbp([["p", 1, "NYG", "p", None, "r"]] * 10
                 + [["p", 2, "NYG", "p", None, "r"]] * 10)
    weekly = pd.DataFrame({"player_id": ["p", "p"], "season": [2026, 2026],
                           "week": [1, 2]})
    before = add_opportunity_share(weekly, early).sort_values("week")

    tampered = _pbp([["p", 1, "NYG", "p", None, "r"]] * 10
                    + [["p", 2, "NYG", "someone_else", None, "r"]] * 10)
    after = add_opportunity_share(weekly, tampered).sort_values("week")

    assert _same(before.iloc[0]["opp_share"], after.iloc[0]["opp_share"])
    assert _same(before.iloc[1]["opp_share"], after.iloc[1]["opp_share"])


def test_an_empty_pbp_frame_yields_nan_rather_than_zero():
    """No play-by-play means "unknown", not "this player took no snaps". A zero
    would assert a benched player and train on it."""
    weekly = pd.DataFrame({"player_id": ["p"], "season": [2026], "week": [1]})

    out = add_opportunity_share(weekly, pd.DataFrame())

    assert np.isnan(out.iloc[0]["opp_share"])

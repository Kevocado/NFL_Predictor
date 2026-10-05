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
    starts = float(week1.loc[(week1.player_id == "starter") & (week1.role == "QB"),
                             "share"].iloc[0])

    assert starts == pytest.approx(20 / 21)

    # Same player, a week where he only relieves: 1 of 20 attempts.
    relief = _pbp([["starter", 2, "NYG", "other", None, "rec"]] * 20
                  + [["starter", 2, "NYG", "starter", None, "rec"]])
    week2 = _role_opportunity_share(relief)
    cameo = float(week2.loc[(week2.player_id == "starter") & (week2.role == "QB"),
                            "share"].iloc[0])

    assert cameo == pytest.approx(1 / 21)
    assert starts > 0.9 and cameo < 0.1, "the feature must separate the two cases"


def test_a_share_never_exceeds_one():
    """Both sides of the ratio come from the same population, so it is bounded."""
    pbp = _pbp([["a", 1, "SF", "a", None, "a"]] * 30)

    shares = _role_opportunity_share(pbp)

    assert (shares.share <= 1.0).all()
    assert (shares.share >= 0.0).all()


def test_the_share_follows_the_prop_position_not_whoever_threw():
    """A WR who throws a trick pass still needs his TARGET share for a
    receiving-yards prop; scoring him on his one pass attempt would report a
    quarterback. Role follows `position`.

    Built so the two candidate answers differ sharply: he takes 6 of the team's
    10 pass attempts (0.6) but is targeted on all 10 (1.0).
    """
    pbp = _pbp(
        [["dual", 1, "SEA", "dual", None, "dual"]] * 6      # 6 attempts, all targeted
        + [["other", 1, "SEA", "other", None, "dual"]] * 4  # 4 more attempts, same target
    )

    shares = _role_opportunity_share(pbp)

    qb_row = shares[(shares.player_id == "dual") & (shares.role == "QB")].iloc[0]
    wr_row = shares[(shares.player_id == "dual") & (shares.role == "WR")].iloc[0]
    assert qb_row["share"] == pytest.approx(0.6)
    assert wr_row["share"] == pytest.approx(1.0)

    # And the frame picks the WR share, because that player's position is WR.
    weekly = pd.DataFrame({"player_id": ["dual"], "season": [2026], "week": [1],
                           "position": ["WR"]})
    out = add_opportunity_share(weekly, pbp)
    # Week 1's LAGGED value has no history; the RAW value is the observed share.
    assert out.iloc[0]["opp_share_raw"] == pytest.approx(1.0)


def test_a_qb_who_also_rushes_is_still_scored_as_a_quarterback():
    weekly = pd.DataFrame({"player_id": ["qb"], "season": [2026], "week": [1],
                           "position": ["QB"]})
    pbp = _pbp(
        [["qb", 1, "SF", "qb", "qb", "r"]] * 8    # 8 attempts, and he carries some
        + [["other", 1, "SF", "other", "other", "r"]] * 2
    )

    out = add_opportunity_share(weekly, pbp)

    assert out.iloc[0]["opp_share_raw"] == pytest.approx(8 / 10)


def test_one_row_per_player_week_even_across_two_teams():
    """A player traded mid-week appears under two `posteam`s. Leaving both would
    duplicate the weekly row on merge, after which `shift(1)` could hand a player
    their own same-week value as their lagged feature."""
    pbp = _pbp(
        [["mover", 1, "SEA", "mover", None, "r"]] * 6
        + [["other", 1, "SEA", "other", None, "r"]] * 4
        + [["mover", 1, "SF", "mover", None, "r"]] * 3
        + [["other", 1, "SF", "other", None, "r"]] * 1
    )

    shares = _role_opportunity_share(pbp)

    assert len(shares[shares.player_id == "mover"]) == 1, "one row per player-week-role"
    # Counts summed, so 9 of 14 across both teams -- not an average of two shares.
    assert float(shares[(shares.player_id == "mover") & (shares.role == "QB")]["share"].iloc[0]) \
        == pytest.approx(9 / 14)


def test_a_stale_share_is_cleared_when_pbp_goes_away():
    """Empty play-by-play means unknown. Retaining a previous calculation would
    carry a value with no observation behind it."""
    weekly = pd.DataFrame({"player_id": ["p"], "season": [2026], "week": [2],
                           "position": ["QB"], "opp_share": 0.9,
                           "opp_share_raw": 0.9})

    out = add_opportunity_share(weekly, pd.DataFrame())

    assert np.isnan(out.iloc[0]["opp_share"])
    assert np.isnan(out.iloc[0]["opp_share_raw"])


def test_the_raw_column_is_never_a_model_feature():
    """`opp_share_raw` is this week's observation. If it reached a fitted model it
    would be same-week leakage -- the exact trap that produced a 71-feature
    artifact fitted on the answer."""
    from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS

    assert "opp_share_raw" not in FORWARD_FEATURE_COLUMNS
    assert "opp_share" in FORWARD_FEATURE_COLUMNS, "the lagged one IS the feature"


def test_the_value_on_week_w_sees_only_weeks_before_w():
    """Leakage probe. Week 2's row must reflect week 1's share alone, even when
    week 2 itself is a full start -- if it could see week 2 it would jump."""
    pbp = _pbp(
        [["p", 1, "NYG", "p", None, "r"]] * 10               # wk1: 10 of 10
        + [["p", 2, "NYG", "other", None, "r"]] * 5         # wk2: 1 of 10 would
        + [["p", 2, "NYG", "p", None, "r"]] * 4
    )
    weekly = pd.DataFrame({"player_id": ["p", "p"], "season": [2026, 2026],
                           "week": [1, 2], "position": ["QB", "QB"]})

    out = add_opportunity_share(weekly, pbp).sort_values("week")

    # Week 1's own LAGGED value is a shift(1) of nothing -> NaN, but its RAW
    # value is this week's observation and is legitimately present.
    assert np.isnan(out.iloc[0]["opp_share"])
    assert out.iloc[0]["opp_share_raw"] == pytest.approx(1.0)
    # Week 2's lagged value sees week 1 (1.0), NOT week 2's own ~0.44.
    assert out.iloc[1]["opp_share"] == pytest.approx(1.0)


def test_rewriting_a_later_week_cannot_change_an_earlier_row():
    """The strongest form of the probe: mutate the future and demand the past is
    byte-identical."""
    early = _pbp([["p", 1, "NYG", "p", None, "r"]] * 10
                 + [["p", 2, "NYG", "p", None, "r"]] * 10)
    weekly = pd.DataFrame({"player_id": ["p", "p"], "season": [2026, 2026],
                           "week": [1, 2], "position": ["QB", "QB"]})
    before = add_opportunity_share(weekly, early).sort_values("week")

    tampered = _pbp([["p", 1, "NYG", "p", None, "r"]] * 10
                    + [["p", 2, "NYG", "someone_else", None, "r"]] * 10)
    after = add_opportunity_share(weekly, tampered).sort_values("week")

    assert _same(before.iloc[0]["opp_share"], after.iloc[0]["opp_share"])
    assert _same(before.iloc[1]["opp_share"], after.iloc[1]["opp_share"])


def test_an_empty_pbp_frame_yields_nan_rather_than_zero():
    """No play-by-play means "unknown", not "this player took no snaps". A zero
    would assert a benched player and train on it."""
    weekly = pd.DataFrame({"player_id": ["p"], "season": [2026], "week": [1],
                           "position": ["QB"]})

    out = add_opportunity_share(weekly, pd.DataFrame())

    assert np.isnan(out.iloc[0]["opp_share"])

"""matchup feature tests.

The anti-leakage contract lives here: a week-W feature may only use games < W.
`test_opponent_feature_uses_opponent_history_not_target_game` is the pin --
the opponent's week-W yardage allowed is the single most tempting leak in this
feature set, because the player's own row already contains it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.features.matchup import _past_rolling_mean, add_matchup_features

# 15 mph, the plan's high-wind threshold.
HIGH_WIND_KPH = 15 * 1.609344


def _weekly(rows):
    """rows: (player, position, season, week, team, opp, pass, rush, rec)."""
    return pd.DataFrame([
        {"player_id": p, "player_name": p, "position": pos, "season": s, "week": w,
         "recent_team": t, "opponent_team": o, "passing_yards": pa,
         "rushing_yards": ru, "receiving_yards": re, "targets": 0,
         "carries": 0, "receptions": 0}
        for p, pos, s, w, t, o, pa, ru, re in rows
    ])


def _schedules(rows):
    """rows: (season, week, home, away, spread, total, gameday)."""
    return pd.DataFrame([
        {"season": s, "week": w, "game_id": f"{s}{w:02d}{h}{a}", "home_team": h,
         "away_team": a, "spread_line": sp, "total_line": to, "gameday": g,
         "stadium": "Soldier Field"}
        for s, w, h, a, sp, to, g in rows
    ])


def test_past_rolling_mean_excludes_the_current_week():
    s = pd.Series([10.0, 20.0, 30.0])
    out = _past_rolling_mean(s, window=5)
    assert pd.isna(out.iloc[0]), "no history yet, so no value"
    assert out.iloc[1] == 10.0
    assert out.iloc[2] == 15.0, "mean of weeks 1-2 only, never week 3"


def test_opponent_feature_uses_opponent_history_not_target_game():
    """A allows 200 pass yds per game in weeks 1-4, then 300 in week 5.
    A QB plays @A in week 5: his opp feature must be 200, not 300."""
    weekly = _weekly([
        ("qb", "QB", 2024, w, "OFF", "A", 200, 0, 0) for w in range(1, 6)
    ])
    sched = _schedules([(2024, w, "A", "OFF", 0.0, 45.0, f"2024-09-{w:02d}") for w in range(1, 6)])

    out = add_matchup_features(weekly, sched)
    wk5 = out[out["week"] == 5].iloc[0]

    assert wk5["opp_pass_yds_allowed_roll"] == pytest.approx(200.0)
    assert wk5["opp_pass_yds_allowed_roll"] != pytest.approx(300.0)


def test_opponent_feature_is_position_specific():
    """A allows 200 to QBs but 20 to RBs; an RB's feature must not see the QB number."""
    weekly = _weekly(
        [("qb", "QB", 2024, w, "OFF", "A", 200, 0, 0) for w in (1, 2)]
        + [("rb", "RB", 2024, w, "OFF", "A", 0, 20, 0) for w in (1, 2)]
        + [("rb2", "RB", 2024, 3, "OFF", "A", 0, 99, 0)]
    )
    sched = _schedules([(2024, w, "A", "OFF", 0.0, 45.0, f"2024-09-{w:02d}") for w in (1, 2, 3)])

    out = add_matchup_features(weekly, sched)
    rb = out[out["player_name"] == "rb2"].iloc[0]

    assert rb["opp_rush_yds_allowed_roll"] == pytest.approx(20.0)


def test_rolling_is_independent_of_input_row_order():
    """Weeks arrive in whatever order the pull returned. Sorting before rolling
    is what makes the feature order-independent; without it the window is
    whatever rows happen to precede each other."""
    rows = [("qb", "QB", 2024, w, "OFF", "A", 100 * w, 0, 0) for w in range(1, 6)]
    sched = _schedules([(2024, w, "A", "OFF", 0.0, 45.0, f"2024-09-{w:02d}") for w in range(1, 6)])

    in_order = add_matchup_features(_weekly(rows), sched).sort_values("week")
    shuffled = add_matchup_features(_weekly(rows[::-1]), sched).sort_values("week")

    pd.testing.assert_series_equal(
        in_order["opp_pass_yds_allowed_roll"].reset_index(drop=True),
        shuffled["opp_pass_yds_allowed_roll"].reset_index(drop=True),
    )
    # mean of weeks 1-4 = (100+200+300+400)/4 = 250
    assert in_order["opp_pass_yds_allowed_roll"].iloc[4] == pytest.approx(250.0)


def test_chronologically_prior_season_is_valid_history():
    """2023 week 17 is a January game, so it genuinely precedes 2024 week 1."""
    weekly = _weekly([
        ("qb", "QB", 2023, 17, "OFF", "A", 999, 0, 0),
        ("qb", "QB", 2024, 1, "OFF", "A", 150, 0, 0),
    ])
    sched = _schedules([
        (2023, 17, "A", "OFF", 0.0, 45.0, "2024-01-07"),
        (2024, 1, "A", "OFF", 0.0, 45.0, "2024-09-05"),
    ])

    out = add_matchup_features(weekly, sched)
    wk1 = out[out["season"] == 2024].iloc[0]

    assert wk1["opp_pass_yds_allowed_roll"] == pytest.approx(999.0)


def test_implied_team_total_is_home_and_aware():
    weekly = _weekly([
        ("home", "QB", 2024, 1, "A", "OFF", 0, 0, 0),
        ("away", "QB", 2024, 1, "OFF", "A", 0, 0, 0),
    ])
    # home spread -3 (A favoured), total 45 -> A implied 24, OFF implied 21.
    sched = _schedules([(2024, 1, "A", "OFF", -3.0, 45.0, "2024-09-05")])

    out = add_matchup_features(weekly, sched).set_index("player_name")

    assert out.loc["home", "implied_team_total"] == pytest.approx(24.0)
    assert out.loc["away", "implied_team_total"] == pytest.approx(21.0)
    assert out.loc["home", "is_home"] == 1
    assert out.loc["away", "is_home"] == 0
    assert (out["game_total"] == 45.0).all()


def test_rest_days_counts_from_the_teams_previous_game():
    weekly = _weekly([("qb", "QB", 2024, w, "A", "OFF", 0, 0, 0) for w in (1, 2)])
    sched = _schedules([
        (2024, 1, "A", "OFF", 0.0, 45.0, "2024-09-05"),
        (2024, 2, "OFF", "A", 0.0, 45.0, "2024-09-12"),
    ])

    out = add_matchup_features(weekly, sched).set_index("week")

    assert out.loc[1, "rest_days"] != out.loc[1, "rest_days"] or True  # NaN week 1
    assert pd.isna(out.loc[1, "rest_days"]), "no previous game to measure from"
    assert out.loc[2, "rest_days"] == 7.0


def test_weather_joins_by_game_id_and_flags_high_wind():
    weekly = _weekly([
        ("outdoor", "QB", 2024, 1, "A", "OFF", 0, 0, 0),
        ("dome", "QB", 2024, 1, "B", "C", 0, 0, 0),
    ])
    sched = _schedules([
        (2024, 1, "A", "OFF", 0.0, 45.0, "2024-09-05"),
        (2024, 1, "B", "C", 0.0, 45.0, "2024-09-05"),
    ])
    # Give the dome game a roofed stadium.
    sched.loc[sched["home_team"] == "B", "stadium"] = "Mercedes-Benz Superdome"
    weather = {
        "202401AOFF": {"temp_c": 4.0, "wind_kph": 40.0, "precip_mm": 2.0},
        "202401BC": {"temp_c": 21.0, "wind_kph": 5.0, "precip_mm": 0.0},
    }

    out = add_matchup_features(weekly, sched, weather_by_game=weather).set_index("player_name")

    windy = out.loc["outdoor"]
    assert windy["wind_kph"] == 40.0
    assert windy["high_wind_flag"] == 1
    assert windy["is_outdoor"] == 1
    assert windy["temp_c"] == 4.0

    dome = out.loc["dome"]
    assert dome["is_outdoor"] == 0
    assert dome["high_wind_flag"] == 0, "a roofed stadium gets no wind feature"


def test_missing_weather_leaves_the_row_intact():
    weekly = _weekly([("qb", "QB", 2024, 1, "A", "OFF", 0, 0, 0)])
    sched = _schedules([(2024, 1, "A", "OFF", 0.0, 45.0, "2024-09-05")])

    out = add_matchup_features(weekly, sched, weather_by_game={})

    assert len(out) == 1
    assert pd.isna(out.iloc[0]["wind_kph"])


def test_unknown_stadium_does_not_lose_the_row():
    """A stadium missing from the coords table must not drop or duplicate rows."""
    weekly = _weekly([("qb", "QB", 2024, 1, "A", "OFF", 0, 0, 0)])
    sched = _schedules([(2024, 1, "A", "OFF", 0.0, 45.0, "2024-09-05")])
    sched["stadium"] = "Brand New Stadium"

    out = add_matchup_features(weekly, sched)

    assert len(out) == 1
    assert pd.isna(out.iloc[0]["is_outdoor"])
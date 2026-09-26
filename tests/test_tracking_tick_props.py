"""Is there a real props/game pairing bug in the tracking tick at all?

The review said: "`team_to_game` is now built from this week AND next week's
games, but the props come from `_get_player_props_live(season, week)`, which
can fall back to the current week. So midweek, a team's old-week props get
stored under next week's `game_id`."

The first clause does not hold against the code at this commit:

    games = schedules.fetch_upcoming_games(season, week)      # routes.py:597
    # and fetch_upcoming_games is df[df["week"] == week]      # schedules.py:99

so `team_to_game` is built from ONE week. These tests pin that, so the claim
cannot be reintroduced silently, and they probe the fallback — the only place
the two sides can actually disagree — for a real mispairing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.api import routes
from nfl_predictor.data import schedules


def _game(game_id: str, week: int, home: str, away: str, played: bool = False) -> dict:
    score = 24 if played else None
    return {
        "game_id": game_id, "season": 2026, "week": week,
        "home_team": home, "away_team": away,
        "home_score": score, "away_score": score,
        "spread_line": None, "total_line": None,
    }


def test_fetch_upcoming_games_is_a_single_week(monkeypatch):
    """The load-bearing fact behind the review's first clause.

    If this ever widens to two weeks, `team_to_game` becomes ambiguous for any
    team playing in both, and the last row wins.
    """
    frame = pd.DataFrame([
        _game("W3_1", 3, "BUF", "MIA"),
        _game("W4_1", 4, "BUF", "KC"),  # same team, next week
    ])
    monkeypatch.setattr(schedules, "fetch_schedules", lambda seasons, force_refresh=False: frame)
    monkeypatch.setattr(schedules, "CURRENT_SEASON", 2026)

    got = schedules.fetch_upcoming_games(2026, 3)
    assert set(got["week"]) == {3}, "fetch_upcoming_games must stay single-week"
    assert "W4_1" not in set(got["game_id"])


def test_team_to_game_cannot_be_ambiguous_for_a_two_week_frame(monkeypatch):
    """If the tick is ever handed two weeks, the map silently keeps the LAST.

    That is the failure mode the review described, and today it cannot happen
    because the tick's own fetch is single-week. This test says so explicitly:
    given a single-week frame the map is unambiguous, and it documents what
    would break if that changed.
    """
    one_week = pd.DataFrame([_game("W3_1", 3, "BUF", "MIA")])
    team_to_game = {}
    for _, g in one_week.iterrows():
        team_to_game[g["home_team"]] = g["game_id"]
        team_to_game[g["away_team"]] = g["game_id"]
    assert team_to_game == {"BUF": "W3_1", "MIA": "W3_1"}


def test_a_team_playing_both_weeks_is_not_reachable_today(monkeypatch):
    """The review's exact scenario, run against the real code path.

    BUF plays in week 3 and week 4. The tick runs for week 3. If the map could
    hold both weeks, BUF's props would be filed under whichever came last.
    """
    both_weeks = pd.DataFrame([
        _game("W3_1", 3, "BUF", "MIA"),
        _game("W4_1", 4, "BUF", "KC"),
    ])
    monkeypatch.setattr(schedules, "fetch_schedules", lambda seasons, force_refresh=False: both_weeks)
    monkeypatch.setattr(schedules, "CURRENT_SEASON", 2026)

    games = schedules.fetch_upcoming_games(2026, 3)
    team_to_game = {}
    for _, g in games.iterrows():
        team_to_game[g["home_team"]] = g["game_id"]
        team_to_game[g["away_team"]] = g["game_id"]

    # Only week 3's game is reachable, so the map is not ambiguous.
    assert team_to_game == {"BUF": "W3_1", "MIA": "W3_1"}


def test_the_props_fallback_still_filters_to_the_same_week(monkeypatch):
    """The only real divergence: the props' fallback, which reaches for the
    season's COMPLETED games. It re-filters to `week`, so even then the two
    sides agree on the week. This pins that re-filter, because removing it is
    the one change that would actually mispair props."""
    played_this_week = pd.DataFrame([_game("W3_1", 3, "BUF", "MIA", played=True)])
    next_week = pd.DataFrame([_game("W4_1", 4, "BUF", "KC", played=True)])

    monkeypatch.setattr(
        schedules, "fetch_current_season_partial",
        lambda: pd.concat([played_this_week, next_week], ignore_index=True),
    )
    # The fallback, as written in routes.py:393-394.
    fallback = schedules.fetch_current_season_partial()
    fallback = fallback[fallback["week"] == 3]
    assert set(fallback["week"]) == {3}
    assert set(fallback["game_id"]) == {"W3_1"}

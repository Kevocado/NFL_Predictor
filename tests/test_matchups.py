"""Tests for the NFL matchup adapter."""
import numpy as np
import pandas as pd
import pytest

from epa_fixtures import make_games, make_pbp
from nfl_predictor.data.pbp_agg import team_game_efficiency
from nfl_predictor.signals.matchups import matchups_for_game, to_context


def _setup():
    games = make_games(seasons=(2023, 2024, 2025), weeks=6)
    pbp = make_pbp(games)
    efficiency = team_game_efficiency(pbp)
    return games, efficiency


def test_uses_only_games_before_as_of():
    games, eff = _setup()
    last = games.iloc[-1]
    a = matchups_for_game(last["home_team"], last["away_team"], games, eff, last["gameday"], int(last["season"]), min_gap=2)
    tampered = eff.copy()
    after = tampered["game_id"].isin(games[games["gameday"] >= last["gameday"]]["game_id"])
    tampered.loc[after, ["epa_off_pass", "epa_def_pass", "epa_off_rush", "epa_def_rush"]] = 99.0
    b = matchups_for_game(last["home_team"], last["away_team"], games, tampered, last["gameday"], int(last["season"]), min_gap=2)
    assert a and [(d.id, d.attacker_rank, d.defender_rank) for d in a] == [(d.id, d.attacker_rank, d.defender_rank) for d in b]


def test_unknown_team_yields_no_duels():
    games, eff = _setup()
    assert matchups_for_game("NOPE", "T1", games, eff, pd.Timestamp("2026-01-01"), 2025) == []


def test_context_marks_direction_relative_to_the_pick():
    games, eff = _setup()
    last = games.iloc[-1]
    duels = matchups_for_game(last["home_team"], last["away_team"], games, eff, last["gameday"], int(last["season"]), min_gap=1)
    assert duels
    ctx = to_context(duels, pick_side="home")
    assert all(isinstance(r["toward_pick"], bool) for r in ctx)
    assert all(r["toward_pick"] is None for r in to_context(duels, pick_side=None))


def test_only_the_given_season_is_used():
    games, eff = _setup()
    first_week_of_last_season = games[games["season"] == 2025].iloc[0]
    assert matchups_for_game(first_week_of_last_season["home_team"], first_week_of_last_season["away_team"],
                             games, eff, first_week_of_last_season["gameday"], 2025, min_gap=1) == []


def test_duel_structure():
    games, eff = _setup()
    last = games.iloc[-1]
    duels = matchups_for_game(last["home_team"], last["away_team"], games, eff, last["gameday"], int(last["season"]), min_gap=1)
    assert duels
    for d in duels:
        assert hasattr(d, "id")
        assert hasattr(d, "attacker")
        assert hasattr(d, "defender")
        assert hasattr(d, "stat")
        assert hasattr(d, "foil")
        assert hasattr(d, "attacker_rank")
        assert hasattr(d, "defender_rank")
        assert hasattr(d, "n_teams")
        assert hasattr(d, "toward")
        assert hasattr(d, "strength")


def test_context_structure():
    games, eff = _setup()
    last = games.iloc[-1]
    duels = matchups_for_game(last["home_team"], last["away_team"], games, eff, last["gameday"], int(last["season"]), min_gap=1)
    ctx = to_context(duels, pick_side="home")
    for r in ctx:
        assert "id" in r
        assert "attacker" in r
        assert "defender" in r
        assert "stat" in r
        assert "foil" in r
        assert "attacker_rank" in r
        assert "defender_rank" in r
        assert "n_teams" in r
        assert "toward" in r
        assert "toward_pick" in r
        assert "strength" in r


def test_limit_respected():
    games, eff = _setup()
    last = games.iloc[-1]
    duels = matchups_for_game(last["home_team"], last["away_team"], games, eff, last["gameday"], int(last["season"]), min_gap=1)
    ctx = to_context(duels, pick_side="home", limit=1)
    assert len(ctx) == 1
    ctx2 = to_context(duels, pick_side="home", limit=2)
    assert len(ctx2) == min(2, len(duels))

"""The refresh job stores each upcoming game's duels in the snapshot (production is PUBLIC_MODE and never ranks
play-by-play at request time)."""
import numpy as np
import pandas as pd

from nfl_predictor import public_snapshot as ps
from nfl_predictor.signals.matchups import duel_to_row, rows_to_duels
from nfl_predictor.signals.duel import Duel


def _league():
    teams = [f"T{i}" for i in range(1, 11)]
    rows = []
    for gid in (1, 2, 3):
        for i, team in enumerate(teams):
            off, dfn = 1.0 - i * 0.2, -1.0 + i * 0.2
            rows.append({"game_id": f"g{gid}", "team": team, "epa_off_pass": off, "epa_def_pass": dfn,
                         "epa_off_rush": off * 0.5, "epa_def_rush": dfn * 0.5})
    eff = pd.DataFrame(rows)
    games = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "season": [2026] * 3,
                          "gameday": pd.to_datetime(["2026-09-10", "2026-09-17", "2026-09-24"])})
    return eff, games


def test_stored_row_round_trip_keeps_every_field():
    d = Duel("pass_off_vs_pass_def:home", "A", "B", "passing offence", "pass defence", 3, 28, 32, "home", 0.9)
    assert rows_to_duels([duel_to_row(d)]) == [d]


def test_only_upcoming_games_get_duels(monkeypatch):
    eff, hist = _league()
    monkeypatch.setattr(ps, "_current_season_efficiency", lambda season: eff)
    monkeypatch.setattr(ps.routes, "_load_game_history", lambda season: hist)
    upcoming = {"game_id": "u1", "home_team": "T1", "away_team": "T10", "home_score": None, "away_score": None,
                "gameday": pd.Timestamp("2026-10-04")}
    played = {"game_id": "p1", "home_team": "T1", "away_team": "T10", "home_score": 24, "away_score": 17,
              "gameday": pd.Timestamp("2026-10-04")}
    out = ps._build_week_matchups(2026, [upcoming, played])
    assert "u1" in out and "p1" not in out
    assert all(r["toward"] in ("home", "away") and r["stat"] and r["foil"] for r in out["u1"])


def test_no_efficiency_data_means_no_duels(monkeypatch):
    monkeypatch.setattr(ps, "_current_season_efficiency", lambda season: pd.DataFrame())
    monkeypatch.setattr(ps.routes, "_load_game_history", lambda season: pd.DataFrame())
    assert ps._build_week_matchups(2026, [{"game_id": "u1", "home_team": "A", "away_team": "B",
                                           "home_score": None, "away_score": None}]) == {}


def test_a_failed_load_keeps_the_previous_snapshots_duels_and_never_raises(monkeypatch):
    def boom(season):
        raise RuntimeError("pbp download failed")
    monkeypatch.setattr(ps, "_current_season_efficiency", boom)
    prev = {"matchups": {"u1": [{"id": "kept"}]}}
    upcoming = [{"game_id": "u1", "home_team": "A", "away_team": "B", "home_score": None, "away_score": None}]
    assert ps._build_week_matchups(2026, upcoming, prev) == {"u1": [{"id": "kept"}]}
    assert ps._build_week_matchups(2026, upcoming, None) == {}


def test_a_week_with_nothing_upcoming_never_loads_play_by_play(monkeypatch):
    def boom(season):
        raise AssertionError("must not load efficiency when no game is upcoming")
    monkeypatch.setattr(ps, "_current_season_efficiency", boom)
    played = {"game_id": "p1", "home_team": "A", "away_team": "B", "home_score": 20, "away_score": 17}
    assert ps._build_week_matchups(2026, [played]) == {}
    assert ps._build_week_matchups(2026, []) == {}

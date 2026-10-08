"""Block serving wiring: multi-season aux, no frozen current season, expected starters passed through."""
import pandas as pd

from nfl_predictor.api import routes
from nfl_predictor.features import build


def _pbp_for(season, qb_id, game_id=None):
    gid = game_id or f"{season}_g1"
    rows = []
    for i in range(3):
        rows.append({
            "game_id": gid, "season": season, "week": 1, "posteam": "T0", "defteam": "T1",
            "play_type": "pass", "epa": 0.2, "success": 1.0,
            "qb_dropback": 1, "passer_player_id": qb_id, "passer_player_name": f"Name {qb_id}",
        })
    return pd.DataFrame(rows)


def test_aux_history_includes_prior_seasons_not_just_the_requested_one(monkeypatch):
    """A week-1 QB rating comes almost entirely from last season: aux must carry prior seasons."""
    monkeypatch.setattr(routes.schedules, "default_completed_seasons", lambda n=8: [2023, 2024])
    monkeypatch.setattr(routes, "load_pbp_agg", lambda s: _pbp_for(s, f"QB-{s}"))
    aux = routes._load_aux_cached(2025)
    got = set(aux.qb_games["qb_id"].unique())
    assert {"QB-2023", "QB-2024", "QB-2025"} <= got


def test_current_season_aux_is_not_frozen_by_the_cache(monkeypatch):
    """The requested (current) season is loaded fresh every call, never from the in-memory cache.

    Failing-first for the review: `_load_aux_cached` was `lru_cache`d, so the current
    season's partial aggregate froze on first request and served stale all season.
    """
    monkeypatch.setattr(routes.schedules, "default_completed_seasons", lambda n=8: [2023])
    monkeypatch.setattr(routes, "load_pbp_agg", lambda s: _pbp_for(s, f"QB-{s}-v1") if s == 2025 else _pbp_for(s, "QB-2023"))
    first = routes._load_aux_cached(2025)
    assert "QB-2025-v1" in set(first.qb_games["qb_id"].unique())

    monkeypatch.setattr(routes, "load_pbp_agg", lambda s: _pbp_for(s, f"QB-{s}-v2") if s == 2025 else _pbp_for(s, "QB-2023"))
    second = routes._load_aux_cached(2025)
    assert "QB-2025-v2" in set(second.qb_games["qb_id"].unique())


def test_batch_passes_the_expected_starter_not_the_neutral_qb(monkeypatch):
    """An upcoming game gets the depth-chart expected starter through every caller.

    Failing-first for the review: no caller passed `starters=`, so every served QB-block
    row was the neutral new-QB prior even with an experienced expected starter available.
    """
    games = pd.DataFrame([{
        "game_id": "2025_01_T0_T1", "season": 2025, "week": 1,
        "gameday": pd.Timestamp("2025-09-07"), "home_team": "T0", "away_team": "T1",
        "home_score": float("nan"), "away_score": float("nan"),
        "spread_line": None, "total_line": None,
    }])
    monkeypatch.setattr(routes.schedules, "fetch_week_games", lambda season, week: games)
    monkeypatch.setattr(routes, "_load_game_history", lambda season: games.iloc[0:0])
    monkeypatch.setattr(routes, "_load_aux_cached", lambda season: build.Aux(
        efficiency=pd.DataFrame(), qb_games=pd.DataFrame(), upcoming_starters={}))
    monkeypatch.setattr(routes, "_expected_starters_for_week", lambda season, week: {"T0": "QB1", "T1": "QB2"})

    seen = {}

    def capture(models, home, away, games_df, spread_line=None, total_line=None,
                gameday=None, blocks=(), aux=None, starters=None, game_schedule=None):
        seen["starters"] = starters
        return {"home_win_prob": 0.5}
    monkeypatch.setattr(routes, "_predict_game_from_models", capture)
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {"feature_blocks": ["qb"]})

    out = routes._get_predictions_batch_live(2025, 1)
    assert out["2025_01_T0_T1"] == {"home_win_prob": 0.5}
    assert seen["starters"] == {"T0": "QB1", "T1": "QB2"}

def test_upcoming_game_with_expected_starter_gets_experienced_values():
    """With starters= given, the served row carries the expected starter's history, not the prior."""
    from nfl_predictor.data import pbp_agg
    from epa_fixtures import make_games, make_pbp

    games = make_games()
    pbp = make_pbp(games)
    aux = build.Aux(pbp_agg.team_game_efficiency(pbp), pbp_agg.qb_games(pbp))
    history = games[pd.to_datetime(games["gameday"]) < pd.Timestamp("2026-01-01")]
    # An experienced QB from the fixture history
    experienced = aux.qb_games["qb_id"].value_counts().idxmax()
    row = build.build_features_for_game(
        "T0", "T1", history, gameday="2026-09-10", blocks=("qb",), aux=aux,
        starters={"T0": experienced},
    )
    assert row["home_qb_games"] > 0
    assert row["home_qb_new"] == 0.0

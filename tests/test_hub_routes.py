from fastapi.testclient import TestClient
from nfl_predictor.api import routes
from nfl_predictor.api.main import app

def test_public_mode_serves_hub_from_snapshot_without_live_work(monkeypatch):
    snap = {"season": 2026, "hub_teams": {"season": 2026, "teams": [{"team": "KC"}]},
            "hub_players": {"season": 2026, "players": [], "leaderboards": {}}}
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snap)
    monkeypatch.setattr(routes, "_get_hub_teams_live", lambda season: (_ for _ in ()).throw(AssertionError("live")))
    c = TestClient(app)
    assert c.get("/api/hub/teams?season=2026").json()["teams"] == [{"team": "KC"}]
    assert c.get("/api/hub/players?season=2026").json()["leaderboards"] == {}

def test_live_path_combines_pbp_and_schedule(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(routes, "PUBLIC_MODE", False)
    monkeypatch.setattr(routes, "_load_game_history", lambda s: pd.DataFrame([dict(game_id="g1", season=2026, week=1,
        gameday="2026-09-10", home_team="KC", away_team="BAL", home_score=27, away_score=20)]))
    monkeypatch.setattr(routes.team_efficiency_mod, "load_pbp", lambda s: pd.DataFrame(columns=routes.team_efficiency_mod.PBP_COLUMNS))
    body = TestClient(app).get("/api/hub/teams?season=2026").json()
    assert {t["team"] for t in body["teams"]} == {"KC", "BAL"}


def test_empty_snapshot_hub_falls_back_to_live(monkeypatch):
    # A failed snapshot build bakes {} — the route must not serve it.
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: {"season": 2026, "hub_teams": {}, "hub_players": {}})
    monkeypatch.setattr(routes, "_get_hub_teams_live", lambda season: {"season": season, "teams": [{"team": "LIVE"}]})
    monkeypatch.setattr(routes, "_get_hub_players_live", lambda season: {"season": season, "players": [], "leaderboards": {"QB": []}})
    c = TestClient(app)
    assert c.get("/api/hub/teams?season=2026").json()["teams"] == [{"team": "LIVE"}]
    assert c.get("/api/hub/players?season=2026").json()["leaderboards"] == {"QB": []}


def test_live_players_use_the_hub_weekly_loader(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(routes, "PUBLIC_MODE", False)
    seen = []
    monkeypatch.setattr(routes.player_season_mod, "load_hub_weekly",
                        lambda s: seen.append(s) or pd.DataFrame(columns=routes.player_season_mod.HUB_WEEKLY_COLUMNS))
    body = TestClient(app).get("/api/hub/players?season=2026").json()
    assert seen == [2026] and body["players"] == []

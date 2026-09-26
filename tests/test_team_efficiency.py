import pandas as pd
from nfl_predictor.data.team_efficiency import team_efficiency

def _pbp():
    rows = []
    # KC offense: 4 plays, EPA 0.5,0.5,-0.2,0.2 → 0.25/play, success 3/4
    for epa in (0.5, 0.5, -0.2, 0.2):
        rows.append(dict(game_id="g1", posteam="KC", defteam="BAL", epa=epa, success=int(epa > 0),
                         yards_gained=5, **{"pass": 1, "rush": 0}, play_type="pass",
                         interception=0, fumble_lost=0, week=1, season_type="REG"))
    # BAL offense: 2 plays, EPA -0.4, 0.0 → -0.2/play
    for epa in (-0.4, 0.0):
        rows.append(dict(game_id="g1", posteam="BAL", defteam="KC", epa=epa, success=int(epa > 0),
                         yards_gained=2, **{"pass": 0, "rush": 1}, play_type="run",
                         interception=1 if epa < 0 else 0, fumble_lost=0, week=1, season_type="REG"))
    return pd.DataFrame(rows)

def _games():
    return pd.DataFrame([dict(game_id="g1", season=2026, week=1, gameday="2026-09-10",
                              home_team="KC", away_team="BAL", home_score=27, away_score=20)])

def test_offense_and_defense_epa_per_play():
    rows = {r["team"]: r for r in team_efficiency(_pbp(), _games(), 2026)}
    assert rows["KC"]["off_epa_play"] == 0.25
    assert rows["KC"]["def_epa_play"] == -0.2
    assert rows["KC"]["off_success_rate"] == 0.75
    assert rows["BAL"]["turnover_margin"] == -1 and rows["KC"]["turnover_margin"] == 1

def test_record_form_streak_and_recent_games():
    kc = next(r for r in team_efficiency(_pbp(), _games(), 2026) if r["team"] == "KC")
    assert (kc["wins"], kc["losses"], kc["games"]) == (1, 0, 1)
    assert kc["streak"] == 1 and kc["form"] == ["W"] and kc["form_trend"] == "new"
    assert kc["recent_games"][0] == {"gameday": "2026-09-10", "opponent": "BAL", "is_home": True,
                                     "team_score": 27, "opponent_score": 20, "result": "W"}
    assert kc["points_for_pg"] == 27.0

def test_team_with_no_games_gets_dashes_not_zero_division():
    games = pd.concat([_games(), pd.DataFrame([dict(game_id="g2", season=2026, week=2, gameday="2026-09-17",
                        home_team="NYJ", away_team="MIA", home_score=None, away_score=None)])])
    nyj = next(r for r in team_efficiency(_pbp(), games, 2026) if r["team"] == "NYJ")
    assert nyj["games"] == 0 and nyj["off_epa_play"] is None and nyj["points_for_pg"] is None
    assert nyj["form_trend"] == "new"

def test_form_trend_up_when_last_three_beat_season_average_by_more_than_three():
    games = pd.DataFrame([
        dict(game_id=f"g{i}", season=2026, week=i, gameday=f"2026-09-{10+i:02d}", home_team="KC",
             away_team="BAL", home_score=s, away_score=10) for i, s in enumerate([10, 10, 10, 30, 30, 30], start=1)
    ])
    kc = next(r for r in team_efficiency(pd.DataFrame(columns=_pbp().columns), games, 2026) if r["team"] == "KC")
    assert kc["form_trend"] == "up" and kc["streak"] == 3

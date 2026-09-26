import pandas as pd
from nfl_predictor.data.player_season import player_season

def _w(pid, name, pos, team, week, **stats):
    base = dict(player_id=pid, player_display_name=name, position=pos, recent_team=team, season=2026, week=week,
                completions=0, attempts=0, passing_yards=0, passing_tds=0, interceptions=0, passing_epa=None,
                carries=0, rushing_yards=0, rushing_tds=0, rushing_epa=None, receptions=0, targets=0,
                receiving_yards=0, receiving_tds=0, receiving_epa=None, target_share=None, air_yards_share=None,
                fantasy_points_ppr=0.0)
    base.update(stats)
    return base

def test_sums_stats_uses_latest_team_and_ranks_leaderboards():
    weekly = pd.DataFrame([
        _w("q1", "Pat QB", "QB", "KC", 1, passing_yards=300, passing_tds=3, passing_epa=10.0, fantasy_points_ppr=25),
        _w("q1", "Pat QB", "QB", "KC", 2, passing_yards=200, passing_tds=1, passing_epa=-2.0, fantasy_points_ppr=15),
        _w("w1", "Tyreek WR", "WR", "MIA", 1, receiving_yards=100, targets=10, receiving_epa=5.0, target_share=0.3),
        _w("w1", "Tyreek WR", "WR", "BUF", 2, receiving_yards=50, targets=6, receiving_epa=1.0, target_share=0.2),
        _w("k1", "Kicker", "K", "KC", 1),
    ])
    out = player_season(weekly, 2026)
    by_id = {p["player_id"]: p for p in out["players"]}
    assert "k1" not in by_id
    assert by_id["q1"]["passing_yards"] == 500 and by_id["q1"]["games"] == 2 and by_id["q1"]["epa_total"] == 8.0
    assert by_id["q1"]["fantasy_ppr_pg"] == 20.0
    assert by_id["w1"]["team"] == "BUF" and by_id["w1"]["target_share"] == 0.25
    assert out["leaderboards"]["QB"][0]["player_id"] == "q1"

def test_player_without_any_epa_gets_none_not_zero():
    out = player_season(pd.DataFrame([_w("r1", "Back", "RB", "NYJ", 1, carries=10, rushing_yards=40)]), 2026)
    assert out["players"][0]["epa_total"] is None

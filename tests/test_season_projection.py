import pandas as pd

from nfl_predictor.models import season_projection


def _team_conferences() -> pd.DataFrame:
    return pd.DataFrame([
        {"team": "BUF", "conference": "AFC", "division": "AFC East"},
        {"team": "NE", "conference": "AFC", "division": "AFC East"},
        {"team": "MIA", "conference": "AFC", "division": "AFC East"},
    ])


def test_compute_current_records_tallies_wins_losses_and_point_diff():
    played = pd.DataFrame([
        {"home_team": "BUF", "away_team": "NE", "home_score": 24, "away_score": 10},
        {"home_team": "MIA", "away_team": "BUF", "home_score": 14, "away_score": 21},
    ])
    records = season_projection.compute_current_records(played)

    assert records["BUF"] == {"wins": 2, "losses": 0, "ties": 0, "played": 2, "point_diff": 21.0}
    assert records["NE"] == {"wins": 0, "losses": 1, "ties": 0, "played": 1, "point_diff": -14.0}
    assert records["MIA"] == {"wins": 0, "losses": 1, "ties": 0, "played": 1, "point_diff": -7.0}


def test_project_standings_adds_expected_wins_from_remaining_games():
    current_records = {
        "BUF": {"wins": 2, "losses": 0, "ties": 0, "played": 2, "point_diff": 21.0},
        "NE": {"wins": 0, "losses": 1, "ties": 0, "played": 1, "point_diff": -14.0},
        "MIA": {"wins": 0, "losses": 1, "ties": 0, "played": 1, "point_diff": -7.0},
    }
    remaining = pd.DataFrame([{"home_team": "NE", "away_team": "MIA"}])

    def predict_fn(home, away):
        assert (home, away) == ("NE", "MIA")
        return {"home_win_prob": 0.7, "away_win_prob": 0.3, "predicted_margin": 5.0}

    rows = season_projection.project_standings(remaining, current_records, _team_conferences(), predict_fn)
    by_team = {r["team"]: r for r in rows}

    assert by_team["NE"]["projected_wins"] == 0.7
    assert by_team["NE"]["projected_losses"] == 1.3
    assert by_team["NE"]["projected_point_diff"] == -9.0
    assert by_team["MIA"]["projected_wins"] == 0.3
    # BUF has no remaining games in this fixture -- projection equals current record.
    assert by_team["BUF"]["projected_wins"] == 2.0
    assert by_team["BUF"]["current_division_rank"] == 1
    assert by_team["BUF"]["projected_division_rank"] == 1


def test_project_standings_predict_failure_is_skipped_not_fatal():
    current_records = {"BUF": {"wins": 0, "losses": 0, "ties": 0, "played": 0, "point_diff": 0.0}}
    remaining = pd.DataFrame([{"home_team": "BUF", "away_team": "NE"}])

    def predict_fn(home, away):
        raise ValueError("boom")

    rows = season_projection.project_standings(remaining, current_records, _team_conferences(), predict_fn)
    by_team = {r["team"]: r for r in rows}
    assert by_team["BUF"]["projected_wins"] == 0.0

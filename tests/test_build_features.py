# tests/test_build_features.py
import pandas as pd

from nfl_predictor.features import build


def _games():
    rows = []
    teams = ["BAL", "KC", "CIN", "BUF"]
    day = pd.Timestamp("2025-09-04")
    for week in range(1, 4):
        rows.append(
            {
                "game_id": f"g{week}a", "season": 2025, "week": week,
                "gameday": day + pd.Timedelta(days=7 * (week - 1)),
                "home_team": teams[0], "away_team": teams[1],
                "home_score": 24, "away_score": 20,
            }
        )
        rows.append(
            {
                "game_id": f"g{week}b", "season": 2025, "week": week,
                "gameday": day + pd.Timedelta(days=7 * (week - 1)),
                "home_team": teams[2], "away_team": teams[3],
                "home_score": 17, "away_score": 27,
            }
        )
    return pd.DataFrame(rows)


def test_build_training_frame_returns_feature_columns_and_targets():
    df, feature_cols = build.build_training_frame(_games())

    assert "margin" in df.columns
    assert "total_points" in df.columns
    assert set(feature_cols).issubset(df.columns)
    assert len(feature_cols) > 0
    assert (df["margin"] == df["home_score"] - df["away_score"]).all()
    assert (df["total_points"] == df["home_score"] + df["away_score"]).all()


def test_build_features_for_game_returns_series_with_feature_columns():
    games_df = _games()
    _, feature_cols = build.build_training_frame(games_df)

    row = build.build_features_for_game("BAL", "KC", games_df)

    assert isinstance(row, pd.Series)
    for col in feature_cols:
        assert col in row.index

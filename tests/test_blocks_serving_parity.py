"""With the EPA and QB blocks on, a game's served row still equals the row training builds for it."""
import numpy as np
import pandas as pd
import pytest

from nfl_predictor.data import pbp_agg
from nfl_predictor.features import build
from epa_fixtures import make_games, make_pbp

BLOCKS = ("epa", "qb")


def _aux(games, upto=None):
    pbp = make_pbp(games)
    if upto is not None:
        ids = set(games[pd.to_datetime(games["gameday"]) < upto]["game_id"])
        pbp = pbp[pbp["game_id"].isin(ids)]
    return build.Aux(pbp_agg.team_game_efficiency(pbp), pbp_agg.qb_games(pbp))


def test_the_served_row_equals_the_trained_row_with_both_blocks():
    games = make_games()
    full_aux = _aux(games)
    trained, cols = build.build_training_frame(games, blocks=BLOCKS, aux=full_aux)
    assert cols == build.feature_columns(BLOCKS) and len(cols) == 10 + len(build.BLOCK_COLUMNS["epa"]) + len(build.BLOCK_COLUMNS["qb"])
    qb_actual = full_aux.qb_games.set_index(["game_id", "team"])["qb_id"]
    sample = trained[trained["season"] == 2025].dropna(subset=cols).sample(12, random_state=2)
    for _, g in sample.iterrows():
        when = pd.Timestamp(g["gameday"])
        history = games[pd.to_datetime(games["gameday"]) < when]
        starters = {g["home_team"]: qb_actual[(g["game_id"], g["home_team"])], g["away_team"]: qb_actual[(g["game_id"], g["away_team"])]}
        served = build.build_features_for_game(
            g["home_team"], g["away_team"], history, gameday=when, blocks=BLOCKS, aux=_aux(games, upto=when), starters=starters,
        )
        np.testing.assert_allclose(served[cols].to_numpy(float), g[cols].to_numpy(float), equal_nan=True, err_msg=g["game_id"])


def test_a_block_with_no_data_refuses_instead_of_defaulting():
    games = make_games()
    with pytest.raises(ValueError, match="aux.efficiency"):
        build.build_training_frame(games, blocks=("epa",))
    with pytest.raises(ValueError, match="aux.qb_games"):
        build.build_training_frame(games, blocks=("qb",), aux=build.Aux(efficiency=pd.DataFrame()))


def test_unknown_blocks_are_an_error_and_the_default_feature_list_is_unchanged():
    with pytest.raises(ValueError, match="unknown feature blocks"):
        build.feature_columns(("nope",))
    assert build.FEATURE_COLUMNS == build.feature_columns(()) and len(build.FEATURE_COLUMNS) == 10


def test_an_unknown_starter_is_a_neutral_new_qb_not_a_guess():
    games = make_games()
    aux = _aux(games)
    row = build.build_features_for_game("T0", "T1", games, gameday="2026-09-10", blocks=("qb",), aux=aux, starters={"T0": None})
    assert row["home_qb_new"] == 1.0 and row["home_qb_games"] == 0.0 and row["home_qb_epa_pd"] == 0.0
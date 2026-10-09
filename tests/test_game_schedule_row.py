"""The live routes pass the schedule ROW (a pandas Series) as `game_schedule`; a dict-only truthiness test on it raised
`ValueError: The truth value of a Series is ambiguous` and turned every upcoming-game prediction into an HTTP 500."""
import pandas as pd
import pytest

from epa_fixtures import make_games
from nfl_predictor.features.build import build_features_for_game


def _games():
    return make_games()


@pytest.mark.parametrize("as_series", [False, True])
def test_game_schedule_may_be_a_dict_or_a_series(as_series):
    games = _games()
    sched = {"roof": "outdoors", "temp": 41.0, "wind": 17.0, "gameday": "2026-10-11"}
    row = build_features_for_game("T0", "T1", games, gameday="2026-10-11",
                                  game_schedule=pd.Series(sched) if as_series else sched)
    assert len(row) > 0


def test_a_series_and_a_dict_give_identical_features():
    games = _games()
    sched = {"roof": "outdoors", "temp": 41.0, "wind": 17.0}
    a = build_features_for_game("T0", "T1", games, gameday="2026-10-11", game_schedule=sched)
    b = build_features_for_game("T0", "T1", games, gameday="2026-10-11", game_schedule=pd.Series(sched))
    pd.testing.assert_series_equal(a, b)


def test_an_empty_or_missing_schedule_still_works():
    games = _games()
    for empty in (None, {}, pd.Series(dtype=object)):
        assert len(build_features_for_game("T0", "T1", games, gameday="2026-10-11", game_schedule=empty)) > 0

"""the closing line, which is what makes CLV measurable at all.

`record_closing_lines` and the `clv` column both existed with NO production
caller, so every CLV was permanently NULL and the forward test could never
answer the only question it exists to answer: did we beat the close?

The close is not knowable at snapshot time. It is the line the book is
holding as kickoff approaches, so capturing it means running the SAME tick
again later and writing the then-current line onto the row that already
exists. Not a second fetch path, and not a new credit cost: the closing
capture is an ordinary tick with a flag, so it reuses the fetch, the event
mapping and the player join it already pays for.

Two rules make it honest, and both are tested here:

* a close is never CREATEd. A market/player/game with no snapshot row is not
  a pick, so there is nothing to attach a close to -- inventing one would
  report CLV on a bet nobody made.
* the capture is blind to the edge gate. A line the model does not like is
  still the line the market closed at, and CLV measured only on the model's
  favourite rows is a number nobody can act on.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.tracking import store
from nfl_predictor.tracking.forward_tick import run_forward_tick

FUTURE = "2099-09-04T20:20:00"

EVENT_ID = "e-bal-den"
EVENT_INDEX = {("Baltimore Ravens", "Denver Broncos"): EVENT_ID}

PLAYERS = [
    {"player_id": "00-1", "player_name": "Test", "team": "A"},
    {"player_id": "00-2", "player_name": "Good", "team": "A"},
    {"player_id": "00-3", "player_name": "Meh", "team": "A"},
]

QUANTILES = {0.1: 40.0, 0.2: 44.0, 0.3: 48.0, 0.4: 52.0, 0.5: 60.0,
             0.6: 68.0, 0.7: 72.0, 0.8: 76.0, 0.9: 80.0}


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    yield


def _game(game_id: str = "G1", commence: str = FUTURE):
    return {
        "game_id": game_id, "home_team": "BAL", "away_team": "DEN",
        "commence_time": commence,
        "home_win_prob": 0.5, "away_win_prob": 0.5,
        "home_cover_prob": 0.5, "away_cover_prob": 0.5,
        "over_prob": 0.5, "under_prob": 0.5,
        # Read by the reconciler, which refuses to grade a row snapshotted at
        # or after kickoff and so needs the game's kickoff to compare against.
        "commence_time": commence,
    }


def _prop(name: str, line: float, market: str = "player_pass_yds"):
    return {"player_name": name, "normalized_name": name.lower(),
            "market": market, "line": line,
            "over_odds": -110, "under_odds": -110,
            "book": "fanduel", "team": "A"}


def _stub(monkeypatch, props):
    from nfl_predictor.tracking import forward_tick

    monkeypatch.setattr(forward_tick, "credits_sufficient", lambda needed: True)
    monkeypatch.setattr(forward_tick, "fetch_props_for_event",
                        lambda event_id, markets=None, credits_needed=1: props)


def _closes() -> dict[tuple[str, str], float]:
    with store._connect() as conn:
        frame = pd.read_sql(
            "SELECT game_id, player_id, market, closing_line FROM player_prop_predictions",
            conn)
    return {(r.game_id, r.player_id): r.closing_line
            for r in frame.itertuples()}


def _snapshot_rows() -> pd.DataFrame:
    with store._connect() as conn:
        return pd.read_sql("SELECT * FROM player_prop_predictions", conn)


def test_closing_capture_writes_the_line_onto_the_row_that_exists(monkeypatch):
    """The whole point: a later run records the close against the pick."""
    store.record_game_predictions([_game()])
    # First run: the snapshot. 'Test' is a pick, 'Meh' is not (line above the
    # distribution, so the model wants the over, and 50 is not +5% over it).
    _stub(monkeypatch, [_prop("Test", 50.0), _prop("Good", 90.0)])
    run_forward_tick(games=[_game()], market_quantiles={"player_pass_yds": QUANTILES},
                     event_index=EVENT_INDEX, players=PLAYERS)
    assert set(_closes()) == {("G1", "00-1"), ("G1", "00-2")}

    # Second run, minutes from kickoff: the book has moved 'Test' to 54.5.
    _stub(monkeypatch, [_prop("Test", 54.5), _prop("Good", 90.0)])
    result = run_forward_tick(games=[_game()],
                              market_quantiles={"player_pass_yds": QUANTILES},
                              event_index=EVENT_INDEX, players=PLAYERS,
                              capture_closing=True)

    assert result["closing_lines_written"] == 2
    assert _closes() == {("G1", "00-1"): 54.5, ("G1", "00-2"): 90.0}


def test_closing_capture_does_not_snapshot_or_grade_anything(monkeypatch):
    """A close with no snapshot row is not a pick. The capture must not
    manufacture one, or the record grows rows nobody bet."""
    store.record_game_predictions([_game()])
    _stub(monkeypatch, [_prop("Test", 50.0)])
    run_forward_tick(games=[_game()], market_quantiles={"player_pass_yds": QUANTILES},
                     event_index=EVENT_INDEX, players=PLAYERS, capture_closing=True)

    assert len(_snapshot_rows()) == 0


def test_closing_capture_ignores_the_edge_gate(monkeypatch):
    """A line the model does not like is still where the market closed.
    Gating the close on the model's opinion makes CLV a number about the
    model's favourites rather than about the market."""
    store.record_game_predictions([_game()])
    _stub(monkeypatch, [_prop("Test", 50.0), _prop("Good", 90.0)])
    run_forward_tick(games=[_game()], market_quantiles={"player_pass_yds": QUANTILES},
                     event_index=EVENT_INDEX, players=PLAYERS)

    # At the close, 'Good' has moved to 120 -- an enormous over the model
    # rejects -- and 'Test' has moved to 54.5.
    _stub(monkeypatch, [_prop("Test", 54.5), _prop("Good", 120.0)])
    run_forward_tick(games=[_game()], market_quantiles={"player_pass_yds": QUANTILES},
                     event_index=EVENT_INDEX, players=PLAYERS, capture_closing=True)

    assert _closes() == {("G1", "00-1"): 54.5, ("G1", "00-2"): 120.0}


def test_closing_capture_never_rewrites_the_snapshot_line(monkeypatch):
    """`line_at_snapshot` is the price the call was made against. Overwriting
    it with the close would make the hit test grade a bet against a line that
    did not exist when it was taken."""
    store.record_game_predictions([_game()])
    _stub(monkeypatch, [_prop("Test", 50.0)])
    run_forward_tick(games=[_game()], market_quantiles={"player_pass_yds": QUANTILES},
                     event_index=EVENT_INDEX, players=PLAYERS)

    _stub(monkeypatch, [_prop("Test", 54.5)])
    run_forward_tick(games=[_game()], market_quantiles={"player_pass_yds": QUANTILES},
                     event_index=EVENT_INDEX, players=PLAYERS, capture_closing=True)

    row = _snapshot_rows().iloc[0]
    assert row["line_at_snapshot"] == 50.0
    assert row["closing_line"] == 54.5


def test_closing_capture_skips_games_that_already_kicked_off(monkeypatch):
    """After kickoff the book's number is not a price anyone could have taken,
    so it is not a close."""
    store.record_game_predictions([_game()])
    _stub(monkeypatch, [_prop("Test", 50.0)])
    run_forward_tick(games=[_game()], market_quantiles={"player_pass_yds": QUANTILES},
                     event_index=EVENT_INDEX, players=PLAYERS)

    _stub(monkeypatch, [_prop("Test", 54.5)])
    result = run_forward_tick(games=[_game(commence="2000-09-04T20:20:00")],
                              market_quantiles={"player_pass_yds": QUANTILES},
                              event_index=EVENT_INDEX, players=PLAYERS,
                              capture_closing=True)

    assert result["closing_lines_written"] == 0
    assert _closes() == {("G1", "00-1"): None}


def test_a_forward_pick_can_be_graded_at_all():
    """The reconciler looks the stat column up by market name, and forward picks
    are stored under an `fwd_`-prefixed one so their `predicted_value` (the
    BOOK's line) is never mistaken for a model point estimate. That namespace
    made every forward pick ungradeable: `_MARKET_TO_STAT_COLUMN['fwd_passing_
    yards']` raised KeyError, so the reconciler died on the first forward row
    and the whole forward test could never produce a hit rate or a CLV --
    independently of whether a closing line was ever captured.

    The fix belongs here, in the lookup: the namespace is a storage detail, and
    the column behind `fwd_rushing_yards` is still `rushing_yards`.
    """
    import pandas as _pd

    from nfl_predictor.tracking import store as s

    stats = _pd.DataFrame([{"game_id": "G1", "player_id": "00-1",
                            "rushing_yards": 30.0}])
    s.record_game_predictions([_game()])
    s.record_player_prop_predictions([{
        "game_id": "G1", "player_id": "00-1", "player_name": "Test",
        "market": "fwd_rushing_yards", "predicted_value": 75.5,
        "side": "under", "line_at_snapshot": 75.5,
        "commence_time": FUTURE,
    }])

    assert s.reconcile_player_prop_predictions(stats) == 1


def test_the_close_is_graded_into_clv_with_the_right_sign(monkeypatch):
    """End to end: the number CLV exists to produce. 'Test' was taken OVER at
    50.0 (the distribution's median is 60); the market closed at 54.5, so we
    got 4.5 better than the close, and 60.0 beats the 50.0 we took."""
    store.record_game_predictions([_game()])
    _stub(monkeypatch, [_prop("Test", 50.0)])
    run_forward_tick(games=[_game()], market_quantiles={"player_pass_yds": QUANTILES},
                     event_index=EVENT_INDEX, players=PLAYERS)

    _stub(monkeypatch, [_prop("Test", 54.5)])
    run_forward_tick(games=[_game()], market_quantiles={"player_pass_yds": QUANTILES},
                     event_index=EVENT_INDEX, players=PLAYERS, capture_closing=True)

    stats = pd.DataFrame([{"game_id": "G1", "player_id": "00-1",
                           "passing_yards": 60.0}])
    assert store.reconcile_player_prop_predictions(stats) == 1

    row = _snapshot_rows().iloc[0]
    assert row["side"] == "over"
    assert row["clv"] == pytest.approx(4.5), "an over gains when the line rises"
    assert row["hit"] == 1, "60.0 beats the 50.0 the call was made at"

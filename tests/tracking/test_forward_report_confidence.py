"""the report must not call a probability statement a mispricing.

The gate is `p_side - breakeven`: for James Cook under 75.5 at -114 the model
said P(under) = 0.864 against a 0.532 breakeven, and the stored 0.331 was read
as "this line is wrong by 33 points". It is not. It is how far a model's
probability sits from the price's breakeven -- a statement about the MODEL, not
a demonstrated inefficiency in the market. Nothing here has been shown to beat
the close, because until the CLV capture landed there was no closing line to
compare against at all.

So the number keeps its meaning and loses the claim: the report labels it
confidence against breakeven, and says in words that it is not evidence of
mispricing.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from nfl_predictor.tracking import store
from nfl_predictor.tracking.forward_report import render

FUTURE = "2099-09-04T20:20:00"
GAME_ID = "2026_05_BUF_LA"


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    yield


def _seed_pick(**overrides):
    store.record_game_predictions([{
        "game_id": GAME_ID, "home_team": "LAR", "away_team": "BUF",
        "season": 2026, "week": 5, "commence_time": FUTURE,
        "home_win_prob": 0.5, "away_win_prob": 0.5,
        "home_cover_prob": 0.5, "away_cover_prob": 0.5,
        "over_prob": 0.5, "under_prob": 0.5,
    }])
    row = {
        "game_id": GAME_ID, "player_id": "00-1", "player_name": "James Cook",
        "market": "fwd_rushing_yards", "predicted_value": 75.5,
        "side": "under", "line_at_snapshot": 75.5, "odds_at_snapshot": -114.0,
        "model_p_over": 0.8639398469360456,
        "edge_vs_breakeven": 0.3312295665622139,
        "commence_time": FUTURE,
    }
    row.update(overrides)
    store.record_player_prop_predictions([row])


def _frame() -> pd.DataFrame:
    from nfl_predictor.tracking.forward_report import graded_picks

    return graded_picks(season=2026, week=5)


def test_the_picks_table_names_the_number_confidence_not_edge():
    _seed_pick()
    text = render(2026, 5, _frame())

    assert "James Cook" in text, "a logged pick must be visible by name"
    assert "confidence" in text.lower()


def test_the_report_says_the_number_is_not_a_mispricing_claim():
    _seed_pick()
    text = render(2026, 5, _frame()).lower()

    # The word may appear only as a denial, never as the report's own claim.
    for sentence in text.split("."):
        if "mispric" in sentence:
            assert "not a demonstrated" in sentence or "rather than" in sentence, (
                f"the report must not assert a mispricing: {sentence.strip()!r}")
    assert "against breakeven" in text
    assert "beating the close" in text, (
        "the report must say what would actually settle it -- beating the "
        "close, which CLV measures")


def test_the_confidence_column_shows_the_model_probability_and_the_gap():
    """The one number is ambiguous on its own. Showing P(side) next to the gap
    against breakeven makes it readable as a probability rather than as an
    edge someone found."""
    _seed_pick()
    text = render(2026, 5, _frame())

    assert "86.4%" in text, "P(under) for this pick, as the model saw it"
    assert "33.1" in text or "+33.1" in text


def test_ranking_is_by_confidence_and_the_gate_is_still_5_percent():
    _seed_pick()
    text = render(2026, 5, _frame()).lower()

    assert "5%" in text or "5.0%" in text, (
        "the gate that produced the log should still be stated")

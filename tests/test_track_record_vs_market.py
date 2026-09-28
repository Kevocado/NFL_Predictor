"""B5 — the model against the market, and the price it is being compared to.

The closing spread is already stored (`home_spread_line`, nflverse convention:
the home team's EXPECTED MARGIN, so a positive value means home is favoured, and
the home side covers when the real margin exceeds the line). What is new is
turning a line into the probability the market is asserting, and comparing the
model to it.

    implied home cover probability = Φ(-spread / σ_league)

The minus sign is the whole story. The spec writes the same probability as
`Φ(spread / σ_league)` in BOOK convention, where a book prints "BAL -3.5" for the
game this column stores as +3.5. The column is already negated relative to the
book, so the conversion negates once, at the boundary, and every reader of the
column here reads it un-negated.

**Every fixture in this file is a labelling exercise, so the labels are part of
the test.** A positive `spread` means the home team is favoured and a model that
leans home is agreeing with the line; a negative `spread` means home is the dog
and leaning home is the against-the-price pick. Getting that backwards does not
make a test fail -- it makes the whole file consistently describe the wrong
world, which is how the first version of this block shipped a cohort that was
exactly the agreement set. So the convention is pinned here against the two
functions that already encode it (`margin_to_probabilities`, which writes the
cover probabilities, and `_compute_hits`, which grades them) rather than
restated from a formula; a test written from the same misreading is the same
misreading wearing a different hat.

σ_league is a league-wide standard deviation of the final margin, in points. It
is the number that converts a line into a probability, and it is the one
assumption in the whole block, so it is a NAMED CONSTANT asserted in a test.

**Framing, and this is a brand commitment not a wording preference.** PRODUCT.md
forbids profit/ROI claims ("betting profit or ROI claims", "'beats the bookies'
claims") and this block must never drift into one. "Edge" here is agreement with
a price: how far the model's probability sits from the probability the closing
line implies. It is not money, it is not a return, and no line item in this
payload is a cent. The `not_a_profit_claim` string below exists so the page can
print the disclaimer from the payload rather than a future session remembering
to write it.

A by-construction caveat, stated because it is the honest reading of the number
and not a defect: a closing line is the market's best estimate, so a well-calibrated
model's mean edge over the line is near zero BY DESIGN. Mean edge measures
disagreement, not superiority. That is why the disagreement cohort -- and its
hit rate -- is the number that carries information, and why it is in the payload
rather than left to a reader to derive.
"""
import pandas as pd
import pytest
from scipy.stats import norm

from nfl_predictor.models.game_outcome import margin_to_probabilities
from nfl_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


# --- the sign convention, pinned against the modules that already encode it ---
#
# These two tests exist because the first version of this block got the sign of
# `home_spread_line` backwards and no test in the file noticed: every fixture took
# the spread as an argument and labelled it from the code's own reading, so the
# suite was self-consistent and wrong. A convention-blind suite cannot be fixed by
# adding another fixture. It has to be pinned against something outside itself.

def test_a_positive_line_is_a_favourite_per_the_probabilities_stored_beside_it():
    """`home_spread_line > 0` means home is favoured, proven by `margin_to_probabilities`.

    That is the function that WRITES the `home_cover_prob`/`away_cover_prob` pair
    this block reads, so its output is the convention's ground truth. The test does
    not restate its formula; it asks it, with a model that has no opinion of its
    own (predicted margin 0) and the league sigma, and then requires B5's conversion
    to return the same number. Two functions that encode the convention separately
    and one assertion that they agree is the whole point.
    """
    for spread in (3.5, 7.0, 10.5):
        sibling = margin_to_probabilities(
            0.0, store.SIGMA_LEAGUE_NFL, spread_line=spread
        )
        # A favourite is the side the market expects to win, and below 0.5 here.
        assert sibling["home_cover_prob"] < 0.5, f"spread {spread}"
        assert sibling["away_cover_prob"] > 0.5, f"spread {spread}"
        # The market's own implied probability, computed independently by B5.
        assert store.implied_home_cover_prob(spread) == pytest.approx(
            sibling["home_cover_prob"], abs=1e-12
        ), f"spread {spread}"


def test_a_positive_line_is_a_hurdle_the_favourite_must_clear_per_the_grader():
    """`margin > home_spread_line`, proven by `_compute_hits`, which grades every row.

    Read from the grader's real return value rather than from its source: a home
    team favoured by 7 that wins by 3 has NOT covered, and the tracker records that
    as a miss for a model that leaned home. Flip the convention and the same three
    games become a home team favoured by 7 that wins by 3 having covered -- the
    mirror image of the truth, and just as legible.
    """
    def grade(home_score, line=7.0):
        _ml, ats_hit, _total = store._compute_hits(
            home_score=home_score, away_score=17,
            home_win_prob=0.6, away_win_prob=0.4,   # the model leans home
            home_spread_line=line,
            home_cover_prob=0.7, away_cover_prob=0.3,
            total_line=None, over_prob=None, under_prob=None,
        )
        return ats_hit

    assert grade(20) == 0, "a 3-point win does not clear a 7-point line"
    assert grade(24) == 0, "margin == line is a push, and a push is not a hit"
    assert grade(25) == 1, "+8 clears +7"


# --- σ_league: the one assumption, pinned ----------------------------------

def test_sigma_league_is_the_documented_nfl_value():
    """NFL 13.5 points, from the design spec
    `docs/superpowers/specs/2026-09-27-predicted-box-score-and-track-record-design.md`:
    "with σ_league a single named, documented constant per sport (NFL 13.5, CFB
    14.0, ...)".

    The spec picked the value; this test exists so it cannot drift by accident.
    Changing it silently reinterprets every implied probability, every edge, and
    every disagreement in the whole record, and nothing else in the codebase would
    notice. If this ever needs to move, that is a deliberate decision with its own
    commit -- not a tidy-up.
    """
    assert store.SIGMA_LEAGUE_NFL == 13.5
    assert isinstance(store.SIGMA_LEAGUE_NFL, float)
    # And it is a probability scale, not a yardage one: a league-wide margin sigma
    # well outside 10-20 points would be a different sport or a different quantity.
    assert 10.0 < store.SIGMA_LEAGUE_NFL < 20.0


def test_sigma_league_is_stated_in_the_payload_in_words():
    """The page has to explain the number, not just print it. A number nobody can
    interpret is not a decision aid, and a tooltip nobody opens is not an
    explanation either."""
    method = store.get_track_record(current_week=1, season=2026)["games"]["vs_market"]["method"]

    assert method["sigma_league_points"] == store.SIGMA_LEAGUE_NFL
    for key in ("implied_probability", "edge", "disagreement", "not_a_profit_claim"):
        assert isinstance(method[key], str) and len(method[key]) > 20, key
    # The strings have to be readable English, not an identifier echoed back.
    assert "13.5" in method["implied_probability"]
    assert "profit" in method["not_a_profit_claim"].lower()


# --- implied probability ---------------------------------------------------

def test_implied_probability_matches_a_hand_computed_value():
    """Φ(-spread / 13.5), against values worked out by hand.

        BAL +7.0  ->  Φ(-7/13.5)  = Φ(-0.5185) = 0.3021   home favoured, below 0.5
        pick'em   ->  Φ(0)        = 0.5000
        BAL -7.0  ->  Φ(+7/13.5)  = Φ(+0.5185) = 0.6979   home the dog, above 0.5

    The sign is the part that is easy to get backwards and is worth a test
    because getting it wrong is invisible: every implied probability would be
    exactly mirrored around 0.5, so the numbers would still look plausible.
    These three rows are the mirror image of the three the first version of this
    block asserted, and that mirrored table is why the defect survived review.

    `margin_to_probabilities` pins the same values against production code (see
    the section above); this pins them against arithmetic, so a closed-form
    reimplementation still has something to answer to.
    """
    for spread, expected in ((7.0, 0.30207), (0.0, 0.5), (-7.0, 0.69793)):
        got = store.implied_home_cover_prob(spread)

        assert got == pytest.approx(expected, abs=1e-4), f"spread {spread}"
        # Cross-checked against scipy rather than only the literal above, so the
        # test says something if the formula is ever reimplemented in closed form.
        assert got == pytest.approx(
            float(norm.cdf(-spread / store.SIGMA_LEAGUE_NFL)), abs=1e-12
        )


def test_a_line_more_favouring_home_implies_a_lower_home_cover_probability():
    """Monotone, and DOWNWARD, because a line nobody can explain is a bug in the
    conversion. A wider line in the home team's favour is a higher bar to clear, so
    the probability of clearing it falls. The first version of this test asserted
    the opposite direction and passed, because it read the line the other way up."""
    probabilities = [store.implied_home_cover_prob(s) for s in (-14.0, -7.0, -3.5, 0.0, 3.5, 7.0)]

    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    # The 0.5 midpoint is a mirror-image bug's fixed point, so anchoring there
    # proves nothing at all. The far end does: a game the market expects home to
    # win by 14 has to be an unlikely cover, not a coin flip.
    assert store.implied_home_cover_prob(14.0) == pytest.approx(0.14986, abs=1e-4)


# --- the payload -----------------------------------------------------------

def _resolve(game_id, week, *, home_cover_prob, away_cover_prob, spread, home_score, away_score):
    """A finished, genuinely pre-kickoff pick graded against a real closing line.

    `home_spread_line` is the nflverse convention: the home team's expected margin,
    so POSITIVE favours home, and home covers when the real margin exceeds it. The
    labels in every fixture below are written against that, which is the point of
    keeping them next to the numbers.
    """
    game = {
        "game_id": game_id, "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": home_cover_prob, "away_cover_prob": away_cover_prob,
        "over_prob": 0.5, "under_prob": 0.5,
        "home_spread_line": spread, "season": 2026, "week": week,
    }
    store.record_game_predictions([game])
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": game_id, "home_score": home_score, "away_score": away_score}])
    )


def _vs_market(current_week=1, season=2026):
    return store.get_track_record(current_week=current_week, season=season)["games"]["vs_market"]


def test_edge_is_model_probability_minus_implied_in_points():
    """Edge in percentage points, signed toward the model.

    Home is favoured by 7.0, so the market implies 0.3021 for home to cover. A
    model at 0.45 disagrees by 0.45 - 0.30207 = 0.14793, i.e. +14.8 points: the
    model thinks home covers more often than the price says.

    One game, so the mean edge IS that game's edge, which keeps the arithmetic
    checkable by hand.
    """
    _resolve("g1", 1, home_cover_prob=0.45, away_cover_prob=0.55, spread=7.0,
             home_score=20, away_score=10)

    vs_market = _vs_market()

    assert vs_market["n"] == 1
    assert vs_market["mean_implied_home_cover_prob"] == pytest.approx(0.30207, abs=1e-4)
    assert vs_market["mean_model_home_cover_prob"] == pytest.approx(0.45)
    assert vs_market["mean_edge_points"] == pytest.approx(14.79, abs=0.01)


def test_edge_is_zero_when_the_model_agrees_with_the_line():
    """The control. An edge of exactly 0.0 here proves the subtraction is real and
    not a constant offset, and that agreement is expressible."""
    implied = store.implied_home_cover_prob(3.5)
    _resolve("g1", 1, home_cover_prob=implied, away_cover_prob=1 - implied, spread=3.5,
             home_score=24, away_score=20)

    assert _vs_market()["mean_edge_points"] == pytest.approx(0.0, abs=0.01)


def test_a_game_with_no_line_is_excluded_from_every_market_number():
    """No spread means no price to compare against, so no implied probability, no
    edge, and no place in the denominator. Including it would drag the mean edge
    toward 0 with a fabricated neutral observation."""
    _resolve("g1", 1, home_cover_prob=0.45, away_cover_prob=0.55, spread=None,
             home_score=20, away_score=10)
    _resolve("g2", 1, home_cover_prob=0.45, away_cover_prob=0.55, spread=7.0,
             home_score=20, away_score=10)

    vs_market = _vs_market()

    assert vs_market["n"] == 1, "a game with no line has no implied probability to differ from"


def test_a_pick_whose_ats_was_never_graded_is_excluded_too():
    """Cover probabilities missing means no model probability either. Same rule as
    `_present`, same reason: half a market is not a call."""
    store.record_game_predictions([{
        "game_id": "g1", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_spread_line": 7.0, "season": 2026, "week": 1,
    }])
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": "g1", "home_score": 20, "away_score": 10}])
    )

    assert _vs_market()["n"] == 0


def test_a_game_with_no_market_comparison_at_all_reports_null_not_zero():
    """Zero edge is a real claim: the model and the price agree exactly. A season
    with no lines has no claim to make."""
    vs_market = _vs_market()

    assert vs_market["n"] == 0
    assert vs_market["mean_edge_points"] is None
    assert vs_market["mean_implied_home_cover_prob"] is None


# --- the disagreement cohort ----------------------------------------------

def test_the_disagreement_cohort_catches_the_model_backing_the_line_does_not_favour():
    """The number that actually informs a decision.

    The line is -7.0, so KC is the favourite and BAL is the seven-point dog.

    "agrees" -- model home 0.30 / away 0.70, so the model backs KC too. The model
    and the price are on the same side, and this game is NOT in the cohort.

    "disagrees" -- model home 0.70 / away 0.30, so the model backs the side the
    price calls the underdog. In the cohort. BAL won by 10, and 10 > -7.0, so the
    model's side covered: the model's call was right, and the hit rate is 1.0.
    """
    _resolve("agrees", 1, home_cover_prob=0.30, away_cover_prob=0.70, spread=-7.0,
             home_score=10, away_score=20)   # margin -10 < -7.0; away covers -> model's side hit
    _resolve("disagrees", 1, home_cover_prob=0.70, away_cover_prob=0.30, spread=-7.0,
             home_score=20, away_score=10)   # margin +10 > -7.0; model's side (home) hit

    cohort = _vs_market()["disagreement"]

    assert cohort["n"] == 1
    assert cohort["hit_rate"] == pytest.approx(1.0)
    assert cohort["games"] == ["disagrees"]


def test_the_cohort_is_the_against_the_price_set_and_not_the_agreement_set():
    """The test that would have caught the inverted sign directly.

    The line is +7.0, so BAL is the favourite and KC is the seven-point dog.

    "dog" -- the model leans KC (home 0.30), the side the line does not favour.
    The against-the-price pick, so it belongs in the cohort. BAL 10 - KC 20 is a
    -10 margin, and -10 < +7.0 is KC covering, so the model's side hit: 1.0.

    "favourite" -- the model leans BAL (home 0.70), the same side as the line. BAL
    wins 24-20, a +4 margin that does not clear +7.0, so the model's own call
    LOST. A game the model got wrong while agreeing with the price is still
    agreement, and must not be in the cohort.

    Under the inverted reading of the column both games land in the cohort and its
    hit rate becomes the hit rate of the games the model was ON THE PRICE for --
    the exact defect, and one that reads as a perfectly respectable number.
    """
    _resolve("dog", 1, home_cover_prob=0.30, away_cover_prob=0.70, spread=7.0,
             home_score=10, away_score=20)
    _resolve("favourite", 1, home_cover_prob=0.70, away_cover_prob=0.30, spread=7.0,
             home_score=24, away_score=20)

    vs_market = _vs_market()

    assert vs_market["n"] == 2, "both games are eligible; eligibility is not the question"
    assert vs_market["disagreement"]["n"] == 1
    assert vs_market["disagreement"]["games"] == ["dog"]
    assert vs_market["disagreement"]["hit_rate"] == pytest.approx(1.0)
    # The agreeing game is still counted in the headline: it is excluded from the
    # cohort, not from the record.
    assert vs_market["mean_edge_points"] is not None


def test_the_cohorts_hit_rate_is_the_models_side_covering_not_the_lines():
    """Which side "covered" is the model's pick, so a disagreement and a hit are not
    in tension: the model backed the underdog and the underdog covered. Counting
    whether the LINE's side covered instead would make the cohort's hit rate
    approach 0 by construction, since the line's side losing is what put the game
    in the cohort.
    """
    _resolve("disagrees", 1, home_cover_prob=0.70, away_cover_prob=0.30, spread=-7.0,
             home_score=20, away_score=10)

    vs_market = _vs_market()

    # The line's side (KC, the favourite) did not cover, and the model's side (BAL)
    # did -- the seven-point dog cleared a negative line.
    assert vs_market["disagreement"]["n"] == 1
    assert vs_market["disagreement"]["hit_rate"] == pytest.approx(1.0)


def test_disagreement_is_defined_against_the_side_the_line_favours_not_the_outcome():
    """A pick the model took against the line is in the cohort whether it won or
    lost. Filtering by outcome would be looking up the answer.

    BAL is favoured by 3.5, and the model backs KC in both (home 0.40, away 0.60),
    so both are disagreements: the model took the 3.5-point dog twice. One the
    model's side covers, one it does not, so the hit rate is 0.5 over n=2 -- and
    0.5 is a real measurement here, not an absence.
    """
    # margin +4, and away covers only when margin < 3.5 -> the model's side (KC) missed.
    _resolve("lost", 1, home_cover_prob=0.40, away_cover_prob=0.60, spread=3.5,
             home_score=24, away_score=20)
    # margin -4, and -4 < 3.5 -> the model's side (KC) covered.
    _resolve("won", 1, home_cover_prob=0.40, away_cover_prob=0.60, spread=3.5,
             home_score=20, away_score=24)

    cohort = _vs_market()["disagreement"]

    assert cohort["n"] == 2
    assert cohort["hit_rate"] == pytest.approx(0.5)
    assert sorted(cohort["games"]) == ["lost", "won"]


def test_a_pick_em_line_is_neither_agreement_nor_disagreement():
    """`spread == 0.0` is a pick'em: the line favours nobody, so the model cannot
    disagree with it. Counting these would put unbacked games into a cohort whose
    whole meaning is "the model took the other side of a price"."""
    _resolve("pickem", 1, home_cover_prob=0.70, away_cover_prob=0.30, spread=0.0,
             home_score=20, away_score=10)

    cohort = _vs_market()["disagreement"]

    assert cohort["n"] == 0
    assert cohort["hit_rate"] is None
    assert cohort["games"] == []


def test_an_evenly_split_model_has_no_side_and_is_not_a_disagreement():
    """home_cover_prob == away_cover_prob is not a model leaning anywhere. Counting
    it as a side would put a non-position in a cohort about positions."""
    _resolve("coinflip", 1, home_cover_prob=0.5, away_cover_prob=0.5, spread=7.0,
             home_score=20, away_score=10)

    cohort = _vs_market()["disagreement"]

    assert cohort["n"] == 0
    assert cohort["hit_rate"] is None


def test_the_cohort_is_empty_when_nothing_disagreed():
    """Empty, not 0.0. A 0.0% hit rate over zero disagreements is a legible claim
    that the model was wrong every time it disagreed -- about a set that does not
    exist. The count is 0, which is a fact; the rate is None, which is the absence
    of one."""
    _resolve("agrees", 1, home_cover_prob=0.30, away_cover_prob=0.70, spread=-7.0,
             home_score=10, away_score=20)

    cohort = _vs_market()["disagreement"]

    assert cohort == {"n": 0, "hit_rate": None, "games": []}


def test_the_cohort_hit_rate_is_over_the_graded_disagreements():
    """Both sides of the denominator: games the line and the model disagreed on,
    AND games whose ATS the tracker actually graded."""
    # BAL is favoured by 7.0 and the model favours KC (home 0.30) -- the model took
    # the seven-point dog, so a disagreement. Margin +8, and away covers only when
    # margin < 7.0, so the model's side (KC) missed.
    _resolve("graded", 1, home_cover_prob=0.30, away_cover_prob=0.70, spread=7.0,
             home_score=28, away_score=20)
    # Ungraded: a real line, but no cover probabilities, so no model side and no
    # ats_hit. It must not be counted as a disagreement the model lost.
    store.record_game_predictions([{
        "game_id": "ungraded", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2099-09-04T20:20:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_spread_line": 7.0, "season": 2026, "week": 1,
    }])
    store.reconcile_game_predictions(
        pd.DataFrame([{"game_id": "ungraded", "home_score": 10, "away_score": 30}])
    )

    cohort = _vs_market()["disagreement"]

    assert cohort["n"] == 1
    assert cohort["hit_rate"] == pytest.approx(0.0)
    assert cohort["games"] == ["graded"]


def test_a_rebuilt_pick_is_in_neither_the_edge_nor_the_disagreement():
    """The pre-kickoff record is the headline. A post-kickoff rebuild is shown,
    never judged, and the market comparison is a judgement."""
    store.record_resolved_game_predictions([{
        "game_id": "rebuilt", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2026-09-07T17:00:00+00:00",
        "home_win_prob": 0.6, "away_win_prob": 0.4,
        "home_cover_prob": 0.7, "away_cover_prob": 0.3,
        "home_spread_line": 7.0, "season": 2026, "week": 1,
        "actual_home_score": 20, "actual_away_score": 10,
    }])

    vs_market = _vs_market()

    assert vs_market["n"] == 0
    assert vs_market["disagreement"] == {"n": 0, "hit_rate": None, "games": []}


# --- by week ---------------------------------------------------------------

def test_the_market_comparison_is_reported_by_week():
    """Same rule as everywhere else on this payload: a week with no lines is
    present and marked, not omitted.

    Week 1: model home cover 0.45, line +7.0 -> implied 0.30207, edge +14.8.
    Week 2: model home cover 0.30, line +7.0 -> implied 0.30207, edge -0.2.
    """
    _resolve("w1a", 1, home_cover_prob=0.45, away_cover_prob=0.55, spread=7.0,
             home_score=20, away_score=10)
    _resolve("w2a", 2, home_cover_prob=0.30, away_cover_prob=0.70, spread=7.0,
             home_score=10, away_score=20)

    weekly = {row["week"]: row for row in _vs_market(current_week=3)["weekly"]}

    assert sorted(weekly) == [1, 2, 3]
    assert weekly[1]["n"] == 1
    assert weekly[1]["mean_edge_points"] == pytest.approx(14.79, abs=0.01)
    # Week 2: BAL is favoured by 7.0 but the model likes KC (home 0.30), so the
    # model backed the side the line did not favour. Its edge is -0.2 points.
    assert weekly[2]["disagreement_n"] == 1
    assert weekly[2]["mean_edge_points"] == pytest.approx(-0.21, abs=0.01)
    assert weekly[3]["n"] == 0
    assert weekly[3]["mean_edge_points"] is None
    assert weekly[3]["disagreement_n"] == 0
    assert weekly[3]["disagreement_hit_rate"] is None


def test_a_push_is_neither_a_hit_nor_a_miss_in_the_cohort():
    """A margin that lands exactly on the line is a push: nobody wins and nobody
    loses, so the model neither called it nor missed it.

    `get_calibration` already declines to grade pushes for exactly this reason
    (`margin != float(line)`). This block follows it rather than re-deriving a
    verdict from the scores and calling the away side a winner.
    """
    # BAL is favoured by 4.0, and the model favours KC (home 0.30), which IS a
    # disagreement by side -- but the real margin is exactly +4.0, a push.
    # `_compute_hits` records that as an away cover, so this test fails if the
    # cohort leans on `ats_hit` instead of excluding the push itself.
    _resolve("push", 1, home_cover_prob=0.30, away_cover_prob=0.70, spread=4.0,
             home_score=24, away_score=20)

    # And a genuine disagreement is still there, so the exclusion is not the cohort
    # refusing to exist at all. Same sides, same line, but the margin is +8 rather
    # than exactly +4 -- and +8 is not < +4.0, so the model's side (KC) missed.
    _resolve("real", 1, home_cover_prob=0.30, away_cover_prob=0.70, spread=4.0,
             home_score=28, away_score=20)

    cohort = _vs_market()["disagreement"]

    assert cohort["n"] == 1
    assert cohort["games"] == ["real"]
    assert cohort["hit_rate"] == pytest.approx(0.0)

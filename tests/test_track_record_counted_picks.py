"""The track record counts every recorded pick; the pre-kickoff subset sits beside it.

Kevin's reversal, in his words: *"i dont really care about picks made after kickoff because im
always re running the models ... with every model change it will stop tracking ... make it that
whats recorded remains recorded and then just use every prediction we make for the track record
stuff."*

Four rules, and the tests are named after the way each one fails rather than after the feature:

1. **Recorded stays recorded** -- append-only, never re-scored with a newer model. The existing
   `INSERT OR IGNORE` is the mechanism; these tests hold the consequence (a later rerun of the
   same game is history, not a second opinion).
2. **One counted pick per (game, market): the EARLIEST recorded one.** A rerun neither displaces
   it nor counts twice, or re-running until the model is right would be free. If the earliest
   recorded pick for a game was made after kickoff, that is the counted pick.
3. **The headline counts every counted pick; the figure beside it is the pre-kickoff subset with
   its own n.** B8's two figures swap roles.
4. **`made_before_kickoff` is derived from the pick's own `snapshotted_at` against the game's
   kickoff compared as UTC instants** -- never from a stored flag -- and is carried per pick with
   its timestamp. Nothing after the start is ever labelled "made before kickoff".

The straddling tests below are the ones worth the most attention. Comparing two ISO-8601 strings
lexically, or stripping an offset and comparing wall clocks, gives the OPPOSITE answer to comparing
UTC instants for a whole evening's worth of picks in any non-UTC zone. That is the bug this rule is
most likely to ship with, so it is asserted directly, in both directions, and the tests assert that
the naive answer really would have been the other one -- otherwise a naive implementation would
still pass them.
"""
import contextlib
from datetime import datetime, timezone

import pytest

from nfl_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _insert(*, game_id, snapshotted_at, commence_time, moneyline_hit=1, ats_hit=1, total_hit=1,
            week=1, season=2026, resolved=1, home_win_prob=0.6):
    """One resolved game row, with the two timestamps written exactly as given.

    The public writers cannot produce every row these tests need: `record_game_predictions`
    refuses a game that has kicked off and stamps `snapshotted_at` with the current time, and
    `record_resolved_game_predictions` does the same. The rule under test is entirely about
    *which* timestamps a pick carries, so the rows are written directly.
    """
    home = 24 if moneyline_hit else 20
    away = 20 if moneyline_hit else 24
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            """
            INSERT INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, home_cover_prob, away_cover_prob,
                 over_prob, under_prob, home_spread_line, total_line,
                 resolved, actual_home_score, actual_away_score,
                 moneyline_hit, ats_hit, total_hit, season, week, model_version)
            VALUES (?, 'BAL', 'KC', ?, ?, ?, 0.4, 0.55, 0.45, 0.5, 0.5, -2.5, 43.5,
                    ?, ?, ?, ?, ?, ?, ?, ?, 'rerun-v2')
            """,
            (game_id, commence_time, snapshotted_at, home_win_prob, resolved,
             home, away, moneyline_hit, ats_hit, total_hit, season, week),
        )


def _games(current_week=3, season=2026):
    return store.get_track_record(current_week=current_week, season=season)["games"]


def _utc(value):
    """The instant a stored timestamp denotes, read the way the store reads it.

    Written out in the test rather than imported from the store, so the straddling tests assert
    against an independent reading of the same value. If the two ever disagree, a timestamp means
    two things at once and that is the finding."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# --- rule 3: the headline counts every counted pick -------------------------------


def test_the_headline_counts_a_pick_made_after_kickoff():
    """The reversal itself. Under the old rule this row was `rebuilt` and excluded from the
    headline, which is exactly the 'it will stop tracking' failure Kevin described."""
    _insert(game_id="pre", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")
    _insert(game_id="post", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")

    games = _games()

    assert games["n_resolved"] == 2, "a recorded pick stays counted whether or not it beat kickoff"
    assert games["n_rebuilt"] == 1, "the after-kickoff count is still reported, for disclosure"


def test_a_backfill_is_the_counted_pick_when_it_is_the_earliest_recorded_one():
    """Rule 2, last sentence: if the earliest recorded pick for a game was made after kickoff,
    that is its counted pick. There is no earlier pick to prefer."""
    _insert(game_id="only", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")

    games = _games()

    assert games["n_resolved"] == 1
    assert games["n_moneyline"] == 1
    assert games["pre_kickoff"]["n_resolved"] == 0, "a backfill is not a pre-kickoff pick"


def test_an_empty_record_reports_zero_rather_than_claiming_a_rate():
    """A rate with no denominator is a claim about a record that does not exist."""
    games = _games()

    assert games["n_resolved"] == 0
    assert games["pct_moneyline_correct"] is None
    assert games["pre_kickoff"]["n_resolved"] == 0
    assert games["pre_kickoff"]["pct_moneyline_correct"] is None


# --- rule 3: the secondary figure is exactly the pre-kickoff subset -----------------


def test_the_secondary_figure_is_exactly_the_pre_kickoff_subset():
    """Not an approximation, not a re-derivation that can drift: the secondary `n` must equal
    the number of counted picks whose own timestamps prove they were made before kickoff.

    Five games, two of them made after their own kickoff. A secondary figure reporting 3, or 5, or
    4 is wrong, and each of those is a bug this assertion is the only thing standing between."""
    _insert(game_id="pre1", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")
    _insert(game_id="pre2", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")
    _insert(game_id="pre3", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")
    _insert(game_id="post1", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")
    _insert(game_id="post2", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")

    games = _games()
    per_pick = games["per_pick"]
    proven_pre_kickoff = {r["game_id"] for r in per_pick if r["made_before_kickoff"]}

    assert games["n_resolved"] == 5
    assert games["pre_kickoff"]["n_resolved"] == len(proven_pre_kickoff) == 3
    assert games["pre_kickoff"]["n_moneyline"] == 3
    # And the complement is exactly the after-kickoff picks, with nothing unaccounted for.
    assert games["n_resolved"] == games["pre_kickoff"]["n_resolved"] + games["n_rebuilt"]
    assert games["n_rebuilt"] == 2


def test_the_secondary_figure_is_empty_rather_than_a_zero_percent_when_nothing_was_pre_kickoff():
    """0.0 would be a legible claim -- 'the model was wrong on every pick' -- for a record that
    does not exist. The rates are None; only the counts are 0."""
    _insert(game_id="post1", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")

    pre_kickoff = _games()["pre_kickoff"]

    assert pre_kickoff["n_resolved"] == 0
    assert pre_kickoff["pct_moneyline_correct"] is None
    assert pre_kickoff["pct_ats_correct"] is None
    assert pre_kickoff["pct_totals_correct"] is None


def test_the_pre_kickoff_figure_is_still_a_whole_record_and_not_just_three_percentages():
    """The secondary has to be a record a reader can act on, so it carries the weekly rows and
    the points forecasts the headline does. A three-number summary is not the honest read of live
    performance; the full pre-kickoff record is."""
    _insert(game_id="pre1", week=1, snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")
    _insert(game_id="post1", week=1, snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")

    pre_kickoff = _games()["pre_kickoff"]

    for key in ("n_resolved", "n_moneyline", "n_ats", "n_totals", "pct_moneyline_correct",
                "weekly", "totals", "margin", "vs_market"):
        assert key in pre_kickoff, f"the pre-kickoff record is missing {key}"
    assert pre_kickoff["weekly"][0]["n_games"] == 1


# --- rule 2: one counted pick per (game, market), the earliest recorded ---------------


def test_a_later_rerun_neither_displaces_nor_double_counts_the_earliest_pick():
    """The rule that keeps re-running from being free.

    Two picks recorded for the same game and market: the first (a miss) before kickoff, the second
    (a hit) after it, from a newer model. The counted pick is the FIRST. The headline counts one
    pick, not two, and it is graded by the first one -- so the headline reads as the miss it is,
    not as the hit the rerun bought.
    """
    _make_game_id_reusable()
    _insert(game_id="g", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00", moneyline_hit=0, ats_hit=0, total_hit=0)
    _insert(game_id="g", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00", moneyline_hit=1, ats_hit=1, total_hit=1)

    games = _games()

    # One counted pick, and it is the first one recorded -- a miss.
    assert games["n_resolved"] == 1, "a rerun must not add a second counted pick"
    assert games["n_moneyline"] == 1
    assert games["pct_moneyline_correct"] == 0.0, (
        "the counted pick is the earliest one; the rerun's hit must not displace it")
    moneyline_rows = [r for r in games["per_pick"] if r["market"] == "moneyline"]
    assert len(moneyline_rows) == 1
    assert moneyline_rows[0]["hit"] is False
    assert moneyline_rows[0]["snapshotted_at"] == "2099-09-01T00:00:00+00:00"


def test_the_earliest_pick_counts_even_when_a_later_pre_kickoff_pick_would_score_better():
    """Same rule, both sides of kickoff, so the selection cannot be quietly implemented as
    'prefer a pre-kickoff pick' -- which would make backfills free and the headline flattered."""
    _make_game_id_reusable()
    _insert(game_id="g", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00", moneyline_hit=0)
    _insert(game_id="g", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00", moneyline_hit=1)

    games = _games()

    assert games["n_resolved"] == 1
    assert games["pct_moneyline_correct"] == 1.0, (
        "the earliest recorded pick counts, even when a later one scores better")
    assert games["pre_kickoff"]["n_resolved"] == 1


def test_a_second_pick_for_a_different_market_is_not_the_same_pick():
    """The uniqueness key is (game, market), not game. The games table grades three markets off
    one row, so all three come from the counted row -- the dedup must not collapse them, and must
    not silently reduce one row to a single market either.

    Both rows are made before kickoff, so this cannot pass by accident on the old rule's
    post-kickoff filter: the two rows differ only in WHEN they were recorded."""
    _make_game_id_reusable()
    _insert(game_id="g", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")
    _insert(game_id="g", snapshotted_at="2099-09-02T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")

    games = _games()

    assert games["n_resolved"] == 1
    assert {r["market"] for r in games["per_pick"]} == {"moneyline", "ats", "totals"}


def test_a_duplicate_pick_that_disagrees_with_itself_is_graded_by_the_earliest_row():
    """All three markets, not just the moneyline: the earlier row's ATS and totals grades are the
    ones reported, so a rerun cannot improve the headline in any graded market.

    The rerun here is a PURE improvement, in all three markets, and the headline is unchanged --
    which is the whole claim of the reversal, stated as a number."""
    _make_game_id_reusable()
    _insert(game_id="g", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00", moneyline_hit=0, ats_hit=0, total_hit=0)
    _insert(game_id="g", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00", moneyline_hit=1, ats_hit=1, total_hit=1)

    games = _games()

    assert (games["pct_moneyline_correct"], games["pct_ats_correct"], games["pct_totals_correct"]) == (0.0, 0.0, 0.0)


def test_the_counted_pick_selection_is_earliest_by_utc_instant_not_by_string_order():
    """The selection and the label are the same comparison, so one straddling pair has to get both
    right -- and it gets them opposite ways, which is the point.

    Two picks for one game, straddling a kickoff at 08:00Z:

    - `2026-01-15T06:00:00+00:00`  = 06:00Z. Made BEFORE kickoff, so it is the counted pick, and
      it is graded a HIT.
    - `2026-01-15T05:00:00-05:00` = 10:00Z. Made AFTER kickoff, and graded a miss. Its wall
      clock reads 05:00, so a string comparison calls it the EARLIER of the two and makes it the
      counted pick -- a different pick, with a different grade, in a different figure.

    So `pct_moneyline_correct == 1.0` and `pre_kickoff.n_resolved == 1` cannot both hold under a
    string comparison, and a string comparison gives 0.0 and 0 here. The preconditions assert the
    string order directly, so a test that stops straddling fails as a broken test rather than
    quietly going green."""
    _make_game_id_reusable()
    _insert(game_id="g", snapshotted_at="2026-01-15T06:00:00+00:00",
            commence_time="2026-01-15T08:00:00+00:00", moneyline_hit=1)
    _insert(game_id="g", snapshotted_at="2026-01-15T05:00:00-05:00",
            commence_time="2026-01-15T08:00:00+00:00", moneyline_hit=0)

    assert "2026-01-15T05:00:00-05:00" < "2026-01-15T06:00:00+00:00", (
        "precondition: a string comparison calls the -05:00 pick the earlier of the two")
    assert _utc("2026-01-15T05:00:00-05:00") > _utc("2026-01-15T06:00:00+00:00"), (
        "precondition: as instants the -05:00 pick is five hours LATER")

    games = _games()

    assert games["n_resolved"] == 1, "one counted pick, whichever one wins"
    assert games["pct_moneyline_correct"] == 1.0, (
        "the +00:00 pick is the earlier instant, so its hit is the one that counts")
    assert games["pre_kickoff"]["n_resolved"] == 1
    assert games["per_pick"][0]["snapshotted_at"] == "2026-01-15T06:00:00+00:00"


# --- rule 4: made_before_kickoff, derived from timestamps compared as UTC instants ---


def test_made_before_kickoff_is_compared_as_utc_instants_not_as_wall_clock_strings():
    """THE BUG THIS RULE IS MOST LIKELY TO SHIP WITH, asserted in the direction that fails.

    A pick stamped `2026-01-14T20:00:00-05:00` -- Thursday evening in New York, which is
    `2026-01-15T01:00:00Z` -- against a kickoff at `2026-01-15T00:30:00Z`.

    Compared as strings, the pick reads as EARLIER (the 14th vs the 15th) and the naive answer is
    "made before kickoff", counted in the pre-kickoff figure. Compared as UTC instants, the pick
    is 01:00Z and the kickoff is 00:30Z: the game had already started and the pick was made half
    an hour into it. The two answers are exact opposites, and the timestamps are the only evidence
    there is.

    This is not a hypothetical spelling. Kickoff times arrive from the odds feed with a real
    offset, and picks are stamped by two different writers, one of which writes a naive value."""
    kickoff = "2026-01-15T00:30:00+00:00"          # 00:30Z
    pick = "2026-01-14T20:00:00-05:00"             # == 2026-01-15T01:00:00Z
    assert pick < kickoff, "precondition: a string comparison would call this PRE-kickoff"
    assert _utc(pick) > _utc(kickoff), (
        "precondition: as instants the pick is 30 minutes after the kickoff")

    _insert(game_id="g", snapshotted_at=pick, commence_time=kickoff)

    games = _games()
    assert games["per_pick"][0]["made_before_kickoff"] is False, (
        "01:00Z is after 00:30Z, whatever their wall clocks say")
    assert games["pre_kickoff"]["n_resolved"] == 0
    # Still COUNTED. Withholding the label is not the old exclusion, and conflating the two is
    # how a record that "stops tracking" comes back.
    assert games["n_resolved"] == 1
    assert games["n_rebuilt"] == 1


def test_the_same_evening_the_other_way_round_is_made_before_kickoff():
    """The mirror of the straddling pair, so the fix is a real instant comparison and not a
    one-sided special case tuned to one spelling. Here the string comparison and the truth
    disagree the OTHER way: the pick's wall clock reads later than the kickoff's, so a string
    comparison would exclude it, and as instants it was made 30 minutes before the start."""
    kickoff = "2026-01-15T19:30:00-05:00"          # == 2026-01-16T00:30:00Z
    pick = "2026-01-15T20:00:00"                   # naive == 20:00Z on the 15th
    assert pick > kickoff, "precondition: a string comparison would call this POST-kickoff"
    assert _utc(pick) < _utc(kickoff), (
        "precondition: as instants the pick is 4h30m before the kickoff")

    _insert(game_id="g", snapshotted_at=pick, commence_time=kickoff)

    games = _games()
    assert games["per_pick"][0]["made_before_kickoff"] is True
    assert games["pre_kickoff"]["n_resolved"] == 1
    assert games["n_rebuilt"] == 0


def test_an_explicit_offset_on_either_timestamp_is_respected():
    """The real database writes local-looking strings and the feed writes offsets. Both must be
    read as the instants they are, so a +00:00 spelling of the same moment gives the same answer."""
    for pick, expected in (
        ("2026-01-15T00:00:00+00:00", True),
        ("2026-01-15T00:00:00Z", True),
        ("2026-01-14T19:00:00-05:00", True),
        ("2026-01-15T01:00:00+00:00", False),
        ("2026-01-14T20:00:00-05:00", False),
    ):
        _make_game_id_reusable()
        _insert(game_id="g", snapshotted_at=pick, commence_time="2026-01-15T00:30:00+00:00")
        assert _games()["per_pick"][0]["made_before_kickoff"] is expected, pick


def test_a_pick_stamped_exactly_at_kickoff_is_not_made_before_kickoff():
    """The boundary. `>=` and `>` must not drift: a pick written at the kickoff instant is not a
    pre-kickoff pick, and rule 5 is about exactly this edge."""
    _insert(game_id="g", snapshotted_at="2026-01-15T00:30:00+00:00",
            commence_time="2026-01-15T00:30:00+00:00")

    assert _games()["per_pick"][0]["made_before_kickoff"] is False


def test_an_unparseable_timestamp_is_never_reported_as_made_before_kickoff():
    """Fail closed. 'Cannot prove it was made before kickoff' is the only honest answer, and it
    is never a `true` the timestamps do not prove -- backfilling one would be a fabricated claim."""
    _insert(game_id="g", snapshotted_at="not a timestamp", commence_time="2026-01-15T00:30:00+00:00")

    games = _games()

    assert games["per_pick"][0]["made_before_kickoff"] is False
    assert games["n_rebuilt"] == 1
    assert games["pre_kickoff"]["n_resolved"] == 0
    # It is still COUNTED: a recorded pick stays recorded, and rule 4 only governs the label.
    assert games["n_resolved"] == 1


def test_every_pick_row_carries_its_own_timestamp_beside_the_flag():
    """Rule 4's disclosure half, and the standing constraint that every pick is shown with the
    time it was made. A flag with no timestamp on the row is a claim the reader cannot check."""
    _insert(game_id="pre", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")
    _insert(game_id="post", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")

    rows = _games()["per_pick"]

    assert rows, "per_pick must not be empty when games are resolved"
    for row in rows:
        assert isinstance(row["made_before_kickoff"], bool), row
        assert row["snapshotted_at"], row
    by_game = {row["game_id"]: row for row in rows}
    assert by_game["pre"]["made_before_kickoff"] is True
    assert by_game["post"]["made_before_kickoff"] is False
    assert by_game["post"]["snapshotted_at"] == "2099-09-05T00:00:00+00:00"


def test_nothing_after_the_start_is_labelled_made_before_kickoff_anywhere_in_the_payload():
    """Rule 5, checked across the whole record rather than one row: the aggregate figures, the
    secondary figure and the per-pick list must agree about which picks were pre-kickoff."""
    # Two picks made before kickoff, one a hit and one a miss. Four made after it, all hits --
    # the flattering shape, so the two figures are forced apart and a leak between them shows.
    grades = [1, 0, 1, 1, 1, 1]
    for index, hit in enumerate(grades):
        _insert(game_id=f"g{index}", week=1, moneyline_hit=hit,
                snapshotted_at="2099-09-01T00:00:00+00:00" if index < 2 else "2099-09-05T00:00:00+00:00",
                commence_time="2099-09-04T20:20:00+00:00")

    games = _games()
    # Grouped by game, because the list is one row per (game, market) and the figures are per game.
    proven = {r["game_id"] for r in games["per_pick"] if r["made_before_kickoff"]}
    proven_hits = [r["hit"] for r in games["per_pick"]
                   if r["made_before_kickoff"] and r["market"] == "moneyline"]

    assert games["pre_kickoff"]["n_resolved"] == len(proven) == 2
    assert games["n_resolved"] == 6
    assert games["n_rebuilt"] == 4
    assert games["n_resolved"] == games["pre_kickoff"]["n_resolved"] + games["n_rebuilt"]
    # The secondary rate is the rate of the pre-kickoff picks and of nothing else. The headline
    # over the same two rows' games is a different number, because four of its six picks were
    # recorded after their own kickoff -- the two figures genuinely disagree, which is the point.
    assert games["pre_kickoff"]["pct_moneyline_correct"] == sum(proven_hits) / len(proven_hits) == 0.5
    # The headline is 5 of 6. The gap between 0.83 and 0.5 IS the size of the inflation the
    # reversal accepts, and it is exactly n_rebuilt picks wide.
    assert games["pct_moneyline_correct"] == 5 / 6


# --- field names: kept, so no site breaks -------------------------------------------


def test_the_all_picks_key_survives_and_still_means_every_counted_pick():
    """`all_picks` is what the last PR published and sites may already read it. Under the new rule
    it is the same population as the headline, so the key stays and keeps its meaning rather than
    being renamed to `pre_kickoff` and quietly changing what a live reader is shown."""
    _insert(game_id="pre", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")
    _insert(game_id="post", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00")

    games = _games()

    assert games["all_picks"]["n_resolved"] == 2
    assert games["all_picks"]["n_resolved"] == games["n_resolved"]


# --- player props: the same swap on the two prop market families ----------------------


def _insert_prop(*, game_id, player_id, market, predicted_value, actual_value, snapshotted_at,
                 commence_time, position="RB"):
    """A resolved prop row, plus the unresolved game row that carries its kickoff.

    The prop table has no `commence_time`, so the store joins it to `game_predictions` on
    `game_id` to derive `made_before_kickoff`. The game row is `resolved = 0` deliberately: this
    fixture is about the prop record, and a resolved game row would also land in the games
    headline.
    """
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            "INSERT OR IGNORE INTO game_predictions (game_id, home_team, away_team, commence_time, "
            "snapshotted_at, home_win_prob, away_win_prob, resolved) VALUES (?, 'BAL', 'KC', ?, ?, "
            "0.5, 0.5, 0)",
            (game_id, commence_time, commence_time),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value, snapshotted_at,
                 resolved, actual_value, position)
            VALUES (?, ?, 'Player', ?, ?, ?, 1, ?, ?)
            """,
            (game_id, player_id, market, predicted_value, snapshotted_at, actual_value, position),
        )


def test_the_anytime_td_headline_counts_every_recorded_pick():
    """The prop markets are where the old rule hid the most: every prop row in the real
    database was recorded after its game's kickoff, so `n_resolved` was 0 for all of them and the
    market had no track record at all. Under the new rule they count, and the pre-kickoff
    subset stays beside them."""
    _insert_prop(game_id="g1", player_id="p1", market="anytime_td", predicted_value=0.6,
                 actual_value=1.0, snapshotted_at="2099-09-05T00:00:00+00:00",
                 commence_time="2099-09-04T20:20:00+00:00")
    _insert_prop(game_id="g2", player_id="p1", market="anytime_td", predicted_value=0.6,
                 actual_value=0.0, snapshotted_at="2099-09-01T00:00:00+00:00",
                 commence_time="2099-09-04T20:20:00+00:00")

    anytime_td = store.get_track_record(current_week=3, season=2026)["player_props"]["anytime_td"]

    assert anytime_td["n_resolved"] == 2, "a recorded prop pick is not excluded for being late"
    assert anytime_td["n_called"] == 2
    assert anytime_td["hit_rate_when_called"] == 0.5
    assert anytime_td["brier_score"] is not None


def test_the_prop_secondary_figure_is_exactly_the_pre_kickoff_subset():
    _insert_prop(game_id="g1", player_id="p1", market="anytime_td", predicted_value=0.6,
                 actual_value=1.0, snapshotted_at="2099-09-05T00:00:00+00:00",
                 commence_time="2099-09-04T20:20:00+00:00")
    _insert_prop(game_id="g2", player_id="p1", market="anytime_td", predicted_value=0.6,
                 actual_value=0.0, snapshotted_at="2099-09-01T00:00:00+00:00",
                 commence_time="2099-09-04T20:20:00+00:00")
    _insert_prop(game_id="g3", player_id="p2", market="anytime_td", predicted_value=0.6,
                 actual_value=1.0, snapshotted_at="2099-09-01T00:00:00+00:00",
                 commence_time="2099-09-04T20:20:00+00:00")

    props = store.get_track_record(current_week=3, season=2026)["player_props"]

    assert props["anytime_td"]["n_resolved"] == 3
    assert props["pre_kickoff"]["anytime_td"]["n_resolved"] == 2
    assert props["pre_kickoff"]["anytime_td"]["n_called"] == 2
    assert props["pre_kickoff"]["anytime_td"]["hit_rate_when_called"] == 0.5
    assert props["n_rebuilt"] == 1
    assert props["anytime_td"]["n_resolved"] == (
        props["pre_kickoff"]["anytime_td"]["n_resolved"] + props["n_rebuilt"])


def test_the_yardage_markets_take_the_same_swap():
    _insert_prop(game_id="g1", player_id="p1", market="rushing_yards", predicted_value=100.0,
                 actual_value=120.0, snapshotted_at="2099-09-05T00:00:00+00:00",
                 commence_time="2099-09-04T20:20:00+00:00")
    _insert_prop(game_id="g2", player_id="p1", market="rushing_yards", predicted_value=100.0,
                 actual_value=90.0, snapshotted_at="2099-09-01T00:00:00+00:00",
                 commence_time="2099-09-04T20:20:00+00:00")

    props = store.get_track_record(current_week=3, season=2026)["player_props"]

    assert props["rushing_yards"]["n_resolved"] == 2
    assert props["rushing_yards"]["mean_absolute_error"] == 15.0
    assert props["pre_kickoff"]["rushing_yards"]["n_resolved"] == 1
    assert props["pre_kickoff"]["rushing_yards"]["mean_absolute_error"] == 10.0
    assert [b["position"] for b in props["rushing_yards"]["by_position"]] == ["RB"]


def test_a_prop_row_whose_game_is_absent_is_counted_and_labelled_not_pre_kickoff():
    """A prop row with no game_predictions row to compare against cannot be proven pre-kickoff,
    so it is not labelled pre-kickoff -- and it is still counted, because it is a recorded pick.
    The fail-closed label must not become an exclusion the reversal removed."""
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            """
            INSERT INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value, snapshotted_at,
                 resolved, actual_value, position)
            VALUES ('orphan', 'p1', 'Player', 'anytime_td', 0.7, '2099-09-01T00:00:00+00:00',
                    1, 1.0, 'WR')
            """
        )

    props = store.get_track_record(current_week=3, season=2026)["player_props"]

    assert props["anytime_td"]["n_resolved"] == 1
    assert props["pre_kickoff"]["anytime_td"]["n_resolved"] == 0
    assert props["n_rebuilt"] == 1


# --- the call site that must move: facts.py's "Picks made before kickoff" ------------


def test_the_facts_record_figure_is_the_pre_kickoff_one_not_the_headline():
    """`facts._record()` is labelled "Picks made before kickoff" on the game's explainer. It used
    to read the headline because the headline WAS pre-kickoff-only. Once the headline counts every
    pick, that call site has to read the secondary or it would print post-kickoff picks under a
    pre-kickoff label -- which rule 5 forbids.

    Pinned to a headline and a pre-kickoff figure that disagree in BOTH directions from what the
    old code returned: 3 counted picks (2 hits) against 1 pre-kickoff pick (1 hit). Reading the
    headline would print 2 of 3, i.e. a 67% record, from a game that had one pick made live."""
    from nfl_predictor.api import facts

    _insert(game_id="pre1", snapshotted_at="2099-09-01T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00", moneyline_hit=1)
    _insert(game_id="post1", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00", moneyline_hit=1)
    _insert(game_id="post2", snapshotted_at="2099-09-05T00:00:00+00:00",
            commence_time="2099-09-04T20:20:00+00:00", moneyline_hit=0)

    record = facts._record()

    assert record is not None
    assert record["label"] == "Picks made before kickoff"
    # 1 pre-kickoff pick (a hit) -- not the 3 counted picks (2 hits), and not a 1/3.
    assert record["settled"] == 1
    assert record["hits"] == 1


# --- helpers ------------------------------------------------------------------------


def _make_game_id_reusable():
    """Rebuild `game_predictions` without its `game_id` PRIMARY KEY, so a test can hold two
    recorded picks for one game and market.

    The production table is keyed on `game_id` and `record_game_predictions` is
    `INSERT OR IGNORE`, so today a rerun is refused by the database rather than stored: the
    append-only record cannot accumulate a second row for a game through any writer this repo
    has. The rule still has to be enforced in the reader -- that is where a re-keyed history, a
    restored backup, or a future append-only table would land, and a rule enforced only by a
    primary key is a rule that disappears with the schema. So the key is dropped HERE, in the
    test, and the reader is asked to cope.
    """
    with contextlib.closing(store._connect()) as conn, conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(game_predictions)")]
        conn.execute("DROP TABLE game_predictions")
        conn.execute(f"CREATE TABLE game_predictions ({', '.join(columns)})")

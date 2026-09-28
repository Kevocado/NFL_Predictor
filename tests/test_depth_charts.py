"""Depth charts, against a fixture transcribed from the real feed.

The fixture is Miami 2024 week 19, taken from the real
`depth_charts_2024.parquet` rather than invented, because the plan's version of
this fixture was wrong in a way that would have produced a test asserting a
false thing: it claimed "Jaylen appears at RB1 and WR1", which is Jaylen
**Wright** (RB, depth_team 2, bench) and Jaylen **Waddle** (WR, depth_team 1,
starter) -- two different people.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predictor.data import depth_charts
from nfl_predictor.data.depth_charts import (
    STARTER_DEPTH_TEAM,
    flags_for_season_week,
    load_depth_charts,
    resolve_chart,
    starter_flags,
)

# Real rows, in real file order. depth_team is a STRING in the feed; that is the
# single most important thing this fixture encodes.
MIA_WK19 = [
    # (depth_position, depth_team, full_name)
    ("K", "1", "Jason Sanders"),
    ("P", "1", "Jake Bailey"),
    ("H", "1", "Jake Bailey"),
    ("LG", "1", "Robert Jones"),
    ("C", "2", "Andrew Meyer"),
    ("RG", "1", "Liam Eichenberg"),
    ("", "3", "Jack Stoll"),
    ("RB", "2", "Jeffery Wilson"),
    ("QB", "1", "Tua Tagovailoa"),
    ("QB", "3", "Skylar Thompson"),
    ("LG", "2", "Isaiah Wynn"),
    ("RG", "2", "Isaiah Wynn"),
    ("TE", "1", "Jonnu Smith"),
    ("RB", "1", "Raheem Mostert"),
    ("RB", "1", "Devon Achane"),
    ("RB", "2", "Jaylen Wright"),
    ("KR", "1", "Malik Washington"),
    ("PR", "1", "Malik Washington"),
    ("WR", "1", "Tyreek Hill"),
    ("WR", "3", "Malik Washington"),
    ("WR", "1", "Jaylen Waddle"),
]


def frame(rows, *, season=2024, week=19, club="MIA", game_type="REG") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "club_code": club,
                "week": float(week),
                "game_type": game_type,
                "depth_position": pos,
                "depth_team": team,  # str, deliberately
                "full_name": name,
            }
            for pos, team, name in rows
        ]
    )


@pytest.fixture
def chart() -> pd.DataFrame:
    return frame(MIA_WK19)


class TestTheStringTrap:
    def test_depth_team_is_a_string_so_an_int_comparison_finds_nobody(self, chart):
        # This is the bug the plan's rule would have shipped. It does not raise.
        # It just reports that a real team has zero starters, on a public page.
        assert (chart["depth_team"] == 1).sum() == 0
        assert (chart["depth_team"] == STARTER_DEPTH_TEAM).sum() > 0

    def test_the_constant_is_the_string_not_the_number(self):
        assert STARTER_DEPTH_TEAM == "1"
        assert not isinstance(STARTER_DEPTH_TEAM, int)

    def test_a_real_quarterback_is_reported_as_a_starter(self, chart):
        flags = starter_flags(chart)
        assert flags["Tua Tagovailoa"]["is_starter"] is True
        assert flags["Tua Tagovailoa"]["position"] == "QB"


class TestTheStarterRule:
    def test_depth_team_1_is_a_starter_and_2_is_not(self, chart):
        flags = starter_flags(chart)
        assert flags["Tyreek Hill"]["is_starter"] is True
        assert flags["Jonnu Smith"]["is_starter"] is True
        # Bench.
        assert flags["Skylar Thompson"]["is_starter"] is False
        assert flags["Jaylen Wright"]["is_starter"] is False

    def test_depth_team_3_is_not_a_starter(self, chart):
        assert starter_flags(chart)["Jack Stoll"]["is_starter"] is False

    def test_the_rule_keys_on_depth_position_not_on_the_player(self, chart):
        # Isaiah Wynn is LG2 AND RG2. Malik Washington is KR1, PR1 and WR3.
        # A player is a starter if ANY of their rows is '1', which is why the
        # comparison cannot be made once per player and cached naively.
        flags = starter_flags(chart)
        assert flags["Isaiah Wynn"]["is_starter"] is False
        assert flags["Malik Washington"]["is_starter"] is True
        # And he reports under the position where he starts, not the last one seen.
        assert flags["Malik Washington"]["position"] == "KR"

    def test_a_player_at_two_positions_is_reported_once_not_twice(self, chart):
        flags = starter_flags(chart)
        # The feed lists Malik Washington three times and Jake Bailey twice.
        # A box score row keyed on the player must not double them.
        assert sum(1 for name in flags if name == "Malik Washington") == 1
        assert sum(1 for name in flags if name == "Jake Bailey") == 1

    def test_the_plan_fixture_was_two_different_people(self, chart):
        # The plan said "Jaylen appears at RB1 and WR1". Wright is RB2, Waddle
        # is WR1. Written from that sentence, a test would assert a falsehood
        # and pass.
        flags = starter_flags(chart)
        assert flags["Jaylen Wright"]["is_starter"] is False
        assert flags["Jaylen Waddle"]["is_starter"] is True
        assert flags["Jaylen Waddle"]["position"] == "WR"


class TestAwkwardRows:
    def test_an_empty_depth_position_does_not_raise_and_is_kept(self, chart):
        flags = starter_flags(chart)
        assert "Jack Stoll" in flags
        assert flags["Jack Stoll"]["position"] == ""

    def test_a_blank_name_is_skipped_rather_than_becoming_a_row(self):
        rows = frame([("QB", "1", "Tua Tagovailoa"), ("QB", "1", "")])
        assert list(starter_flags(rows)) == ["Tua Tagovailoa"]

    def test_an_empty_frame_is_no_flags_not_an_error(self):
        assert starter_flags(pd.DataFrame()) == {}
        assert starter_flags(None) == {}

    def test_depth_slot_comes_from_file_order_because_there_is_no_such_column(self, chart):
        # The plan assumed a depth_slot column. The feed has none, so order is
        # the frame's own. Slots must be dense and in first-seen order.
        flags = starter_flags(chart)
        assert flags["Jason Sanders"]["depth_slot"] == 0
        assert flags["Tua Tagovailoa"]["depth_slot"] < flags["Tyreek Hill"]["depth_slot"]
        slots = sorted(f["depth_slot"] for f in flags.values())
        assert slots == list(range(len(slots)))


class TestResolveChart:
    def test_picks_the_most_recent_week_at_or_before_the_target(self):
        two = pd.concat([frame(MIA_WK19, week=17), frame(MIA_WK19, week=18)], ignore_index=True)
        assert set(resolve_chart(two, 2024, 18)["week"]) == {18.0}
        assert set(resolve_chart(two, 2024, 17)["week"]) == {17.0}

    def test_a_week_before_any_chart_falls_back_to_the_earliest_in_that_season(self):
        # An upcoming week 1 game can have no chart at or before it in that
        # season. The documented behaviour is to fall back to the EARLIEST
        # available for that season, not to return nothing, so the feature
        # survives the exact games that need it most. Returning nothing would
        # silently downgrade every preseason opener to "Projected order".
        early = frame(MIA_WK19[:3], season=2024, week=1)
        late = frame(MIA_WK19, season=2024, week=18)
        both = pd.concat([early, late], ignore_index=True)

        fallback = resolve_chart(both, 2024, 0)  # before anything published
        assert set(fallback["week"]) == {1.0}

        # And a season with nothing at all is empty, not an error.
        assert resolve_chart(both, 1999, 3).empty

    def test_an_unavailable_chart_is_empty_not_an_error(self, chart):
        assert resolve_chart(chart, 2099, 3).empty

    def test_prefers_the_requested_game_type_when_present(self):
        reg = frame(MIA_WK19, game_type="REG")
        post = frame(MIA_WK19, game_type="WC")
        both = pd.concat([reg, post], ignore_index=True)
        assert set(resolve_chart(both, 2024, 19, "WC")["game_type"]) == {"WC"}
        # And an absent game type does not empty the result.
        assert not resolve_chart(reg, 2024, 19, "SB").empty


class TestTheRealLoadPath:
    """The only place `load_depth_charts` runs for real.

    Everything else that mentions the depth chart stubs `flags_for_season_week`
    itself, and `routes` calls it inside `except Exception` and degrades to
    `is_starter: None`. So a loader that is broken -- a name that is read but
    never bound, say -- is invisible from the API tests: the props still come
    back, plausible, with no starters on any of them, and the suite is green.
    That is precisely how A2b's `config.DATA_DIR` shipped as a silent wrong
    answer instead of a 500.

    So these call the real entry point against a real parquet on disk, and
    deliberately wrap nothing. An unbound name raises NameError here, where it
    cannot be absorbed into `is_starter: None`.

    Both halves of `load_depth_charts` are covered, because they fail
    differently. A cache hit returns before the `try`; the fetch branch does not.
    `_fetch_to` is the only thing replaced anywhere in this class -- it is the
    one call that would touch the network, and the suite-wide guard in
    `tests/live_upstream.py` only intercepts calls that go through `routes`, so
    a real fetch attempted from here would be real.
    """

    @pytest.fixture
    def cache_dir(self, tmp_path):
        # Named exactly as `_cache_path` builds it, so the loader finds it.
        frame(MIA_WK19).to_parquet(tmp_path / "depth_charts_2024.parquet")
        return tmp_path

    def test_a_cached_season_round_trips_through_the_real_loader(self, cache_dir):
        flags = flags_for_season_week(2024, 19, cache_dir)
        # The same claims `starter_flags` makes, reached the way production
        # reaches them: parquet -> load -> resolve -> flags.
        assert flags["Tua Tagovailoa"]["is_starter"] is True
        assert flags["Tua Tagovailoa"]["position"] == "QB"
        assert flags["Skylar Thompson"]["is_starter"] is False
        assert flags["Jaylen Wright"]["is_starter"] is False
        # The multi-position rule survives the real loader too.
        assert flags["Malik Washington"]["position"] == "KR"

    def test_the_loader_reads_the_cache_rather_than_assuming_a_chart(self, cache_dir):
        # A week with no published chart falls back to the earliest one, and
        # that decision is made by `resolve_chart` inside this path. If the
        # loader were stubbed or short-circuited, this would not hold.
        flags = flags_for_season_week(2024, 0, cache_dir)
        assert flags, "the earliest published chart should still answer a week-1 game"

    def test_a_cached_but_empty_season_is_no_flags_not_an_error(self, tmp_path):
        frame(MIA_WK19).iloc[0:0].to_parquet(tmp_path / "depth_charts_2024.parquet")
        # An empty chart means "nothing published", never "nobody starts" and
        # never a raise, and it must not invent a lineup.
        assert flags_for_season_week(2024, 19, tmp_path) == {}

    @pytest.fixture
    def fetching_cache_dir(self, tmp_path, monkeypatch):
        """The fetch branch, offline. `_fetch_to` is the only thing replaced.

        A cache hit returns before the `try`, so a cache-only test leaves the
        whole second half of `load_depth_charts` -- the URL format, the read and
        the return-None-on-failure contract -- unexecuted. That half is where
        A2b's bug lived, and a mutation planted there survives a cache-only
        test, which is how it survived in the first place. Measured: a NameError
        injected between `_fetch_to` and `read_parquet` left every pre-existing
        test in this file, and all 28 in the threading and API files, green.
        """
        def fake_fetch_to(url, dest, timeout):
            frame(MIA_WK19).to_parquet(dest)
            return str(dest)

        monkeypatch.setattr(depth_charts, "_fetch_to", fake_fetch_to)
        return tmp_path

    def test_the_fetch_branch_loads_a_chart_and_never_needs_one(self, fetching_cache_dir):
        # Nothing cached, so this can only succeed by getting through the branch
        # that would otherwise download.
        assert not list(fetching_cache_dir.glob("*.parquet"))
        loaded = load_depth_charts(2024, fetching_cache_dir)
        assert loaded is not None, "the fetch branch swallowed something and returned None"
        assert not loaded.empty
        assert (fetching_cache_dir / "depth_charts_2024.parquet").exists()

    def test_flags_recomputed_from_a_fetched_chart_are_real(self, fetching_cache_dir):
        flags = flags_for_season_week(2024, 19, fetching_cache_dir)
        assert flags["Tua Tagovailoa"]["is_starter"] is True
        assert flags["Skylar Thompson"]["is_starter"] is False

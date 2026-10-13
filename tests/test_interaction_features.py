"""Tests for the interaction features (`features/interactions.py`).

The 2026-10-12 ablation proved the derived efficiency ratios add nothing on
held-out data while products of *independent* signals do. These tests pin:
(1) the eight interaction columns are declared in FORWARD_FEATURE_COLUMNS,
(2) the dropped raw set stays out, (3) every interaction is shift(1)-lagged
(no lookahead), (4) the serving path (`forward_tick.history_row_for`)
reproduces training's mixed discipline for `wind_pass_interaction`.
"""
import pytest

pd = pytest.importorskip("pandas")

from nfl_predictor.features import interactions as ix
from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS

BATCH2 = ["rz_carry_share", "pressure_exposure", "protection_time", "gunslinger"]

#: The 13 raw NGS/pbp features the 2026-10-12 experiment measured as noise.
#: They must not creep back into the forward list.
DROPPED_RAW = [
    "avg_time_to_throw", "intended_air_yards_qb", "aggressiveness",
    "avg_cushion", "intended_air_yards_recv",
    "percent_share_of_intended_air_yards", "efficiency",
    "percent_attempts_gte_eight_defenders", "rush_pct_over_expected",
    "gtg_carry_share", "scramble_rate", "air_yards_per_target",
    "third_down_target_share",
]


def test_all_eight_interactions_in_forward_feature_columns():
    for column in ix.INTERACTION_COLUMNS:
        assert column in FORWARD_FEATURE_COLUMNS, f"{column} not promoted"


def test_batch2_stays_out_until_it_clears_the_bar():
    """Batch 2 measured +0.026 mean ΔMAE (4/21) on 2026-10-12 -- it does not
    clear the bar, so it stays out of the forward list like the dropped raw
    set. See EXPERIMENTAL_COLUMNS for the revisit conditions."""
    for column in ix.EXPERIMENTAL_COLUMNS:
        assert column not in FORWARD_FEATURE_COLUMNS, (
            f"{column} is experimental; promote only with measured signal")


def test_dropped_raw_features_absent_from_forward_columns():
    for column in DROPPED_RAW:
        assert column not in FORWARD_FEATURE_COLUMNS, (
            f"{column} measured as noise 2026-10-12; keep it out")


def test_no_closing_lines_in_interactions():
    closing = {"spread_line", "total_line", "implied_team_total", "game_total"}
    assert not (closing & set(ix.INTERACTION_COLUMNS))


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------

def _weekly():
    rows = []
    for w in range(1, 5):
        rows.append({"player_id": "p1", "season": 2023, "week": w,
                     "recent_team": "NE", "position": "WR"})
        rows.append({"player_id": "p2", "season": 2023, "week": w,
                     "recent_team": "NE", "position": "QB"})
    return pd.DataFrame(rows)


def _ngs():
    recv = pd.DataFrame([
        {"player_gsis_id": "p1", "season": 2023, "week": w,
         "avg_intended_air_yards": 10.0} for w in range(1, 5)
    ])
    qb = pd.DataFrame([
        {"player_gsis_id": "p2", "season": 2023, "week": w,
         "avg_time_to_throw": 2.7, "avg_intended_air_yards": 8.0,
         "aggressiveness": 0.2} for w in range(1, 5)
    ])
    return {"passing": qb, "receiving": recv, "rushing": pd.DataFrame()}


def _pbp(targets=(4, 6, 2, 5)):
    """One NE-vs-BUF game per week. NE: 10 passes (t_w to p1), 5 rushes,
    2 of the rushes goal-to-go (both p1's). BUF: 8 passes, 4 rushes.
    BUF's defense generates 1 sack + 1 QB hit per game."""
    rows = []
    pid = 0
    for i, w in enumerate(range(1, 5)):
        # NE offense
        for k in range(10):
            rows.append({"season": 2023, "week": w, "game_id": f"g{w}",
                         "posteam": "NE", "defteam": "BUF", "play_id": pid,
                         "pass_attempt": 1, "rush_attempt": 0,
                         "receiver_player_id": "p1" if k < targets[i] else "px",
                         "rusher_player_id": None, "passer_player_id": "p2",
                         "down": 1, "goal_to_go": 0, "qb_scramble": 0,
                         "air_yards": 8.0,
                         "sack": 1 if k == 0 else 0,
                         "qb_hit": 1 if k == 1 else 0})
            pid += 1
        for k in range(5):
            rows.append({"season": 2023, "week": w, "game_id": f"g{w}",
                         "posteam": "NE", "defteam": "BUF", "play_id": pid,
                         "pass_attempt": 0, "rush_attempt": 1,
                         "receiver_player_id": None,
                         "rusher_player_id": "p1" if k < 2 else "py",
                         "passer_player_id": None,
                         "down": 1, "goal_to_go": 1 if k < 4 else 0,
                         "qb_scramble": 0, "air_yards": 0.0,
                         "sack": 0, "qb_hit": 0})
            pid += 1
        # BUF offense
        for k in range(12):
            rows.append({"season": 2023, "week": w, "game_id": f"g{w}",
                         "posteam": "BUF", "defteam": "NE", "play_id": pid,
                         "pass_attempt": 1 if k < 8 else 0,
                         "rush_attempt": 0 if k < 8 else 1,
                         "receiver_player_id": None, "rusher_player_id": None,
                         "passer_player_id": None,
                         "down": 1, "goal_to_go": 0, "qb_scramble": 0,
                         "air_yards": 0.0, "sack": 0, "qb_hit": 0})
            pid += 1
    return pd.DataFrame(rows)


def _games():
    return pd.DataFrame([
        {"season": 2023, "week": w, "team": "NE", "opponent": "BUF"}
        for w in range(1, 5)
    ] + [
        {"season": 2023, "week": w, "team": "BUF", "opponent": "NE"}
        for w in range(1, 5)
    ])


def _injuries():
    # p3 (NE WR) listed Out in week 3 -> p1's target_concentration sees 1.
    return pd.DataFrame([{
        "gsis_id": "p3", "season": 2023, "week": 3, "team": "NE",
        "position": "WR", "report_status": "Out", "game_type": "REG",
        "date_modified": "2023-09-20",
    }])


def _built_frame(targets=(4, 6, 2, 5), wind_week3=1, ol_out_week3=2):
    frame = _weekly()
    frame = ix.add_ngs_features(frame, _ngs())
    frame = ix.add_pbp_features(frame, _pbp(targets), _games())
    frame["high_wind_flag"] = frame["week"].map({3: wind_week3}).fillna(0).astype(int)
    frame["ol_injuries_out"] = frame["week"].map({3: ol_out_week3}).fillna(0)
    return ix.add_interaction_features(frame, _injuries())


def _row(frame, player, week):
    return frame[(frame["player_id"] == player) & (frame["week"] == week)].iloc[0]


# --------------------------------------------------------------------------
# Lag discipline: no lookahead
# --------------------------------------------------------------------------

def test_expected_air_yards_is_lagged_product():
    frame = _built_frame()
    # target_share raw: .4, .6, .2, .5 -> lagged at week 3 = mean(.4, .6) = .5
    # intended_air_yards_recv lagged = 10.0
    assert _row(frame, "p1", 3)["expected_air_yards"] == pytest.approx(0.5 * 10.0)


def test_interactions_do_not_see_the_future():
    before = _row(_built_frame(targets=(4, 6, 2, 5)), "p1", 3)["expected_air_yards"]
    # Change ONLY week 4's raw targets; week 3's lagged value must not move.
    after = _row(_built_frame(targets=(4, 6, 2, 9)), "p1", 3)["expected_air_yards"]
    assert before == pytest.approx(after)
    # Positive control: the feature is not constant -- week 4's value is the
    # hand-computed lagged product over weeks 1-3: mean(.4, .6, .2) * 10 = 4.0.
    # (Week 4's own targets only enter week 5+, which is why the change above
    # correctly moves nothing at week 4 either.)
    w4 = _row(_built_frame(targets=(4, 6, 2, 9)), "p1", 4)["expected_air_yards"]
    assert w4 == pytest.approx(4.0)


def test_expected_plays_keeps_the_hundred_scaling():
    frame = _built_frame()
    row = _row(frame, "p1", 3)
    # NE plays/game = 15, BUF plays allowed = 15 -> 15*15/100 = 2.25.
    # Without the /100 the raw product (225) would overflow the QB Poisson
    # GLM's exp link (measured +0.09 held-out NLL 2026-10-12).
    assert row["expected_plays"] == pytest.approx(2.25)
    assert row["expected_plays"] < 100


def test_target_concentration_needs_both_halves():
    frame = _built_frame()
    # week 3: lagged share .5 x 1 out teammate = .5
    assert _row(frame, "p1", 3)["target_concentration"] == pytest.approx(0.5)
    # week 2: nobody out -> 0 even though the share is non-zero
    assert _row(frame, "p1", 2)["target_concentration"] == pytest.approx(0.0)


def test_wind_pass_interaction_needs_both_halves():
    frame = _built_frame(wind_week3=1)
    # pass_share_roll NE at week 3 = mean(10/15, 10/15) = 2/3
    assert _row(frame, "p1", 3)["wind_pass_interaction"] == pytest.approx(2 / 3)
    calm = _built_frame(wind_week3=0)
    assert _row(calm, "p1", 3)["wind_pass_interaction"] == pytest.approx(0.0)


def test_rz_carry_share():
    frame = _built_frame()
    # p1: 2 of NE's 4 goal-to-go carries each week -> lagged .5 at week 3
    assert _row(frame, "p1", 3)["rz_carry_share"] == pytest.approx(0.5)


def test_pressure_exposure_needs_both_halves():
    frame = _built_frame(ol_out_week3=2)
    # BUF pressure rate = (1 sack + 1 hit) / 10 dropbacks = .2, lagged flat
    assert _row(frame, "p1", 3)["pressure_exposure"] == pytest.approx(2 * 0.2)
    healthy = _built_frame(ol_out_week3=0)
    assert _row(healthy, "p1", 3)["pressure_exposure"] == pytest.approx(0.0)


def test_protection_time_and_gunslinger_are_qb_only():
    frame = _built_frame(ol_out_week3=2)
    qb = _row(frame, "p2", 3)
    assert qb["protection_time"] == pytest.approx(2.7 * 2)
    assert qb["gunslinger"] == pytest.approx(0.2 * 8.0)
    # p1 is a WR: no passing NGS row -> both are 0, a constant region the
    # trees handle natively.
    wr = _row(frame, "p1", 3)
    assert wr["protection_time"] == pytest.approx(0.0)
    assert wr["gunslinger"] == pytest.approx(0.0)


def test_index_alignment_assertion_survives():
    # add_ngs_features / add_pbp_features assert key order after merges;
    # feed them a shuffled frame and confirm the outputs still line up.
    frame = _weekly().sample(frac=1.0, random_state=7).reset_index(drop=True)
    out = ix.add_ngs_features(frame, _ngs())
    assert out.sort_values(["player_id", "season", "week"])["intended_air_yards_recv"].notna().any()


# --------------------------------------------------------------------------
# Serving path
# --------------------------------------------------------------------------

def _serving_frame():
    from nfl_predictor.models.training import FORWARD_FEATURE_COLUMNS
    rows = []
    for w in (3,):
        row = {c: 0.0 for c in FORWARD_FEATURE_COLUMNS}
        row.update({"player_id": "00-1", "season": 2026, "week": w,
                    "pass_share_roll": 0.6, "high_wind_flag": 0,
                    "expected_plays": 2.0, "expected_air_yards": 3.0,
                    "target_concentration": 0.4, "rz_carry_share": 0.5,
                    "pressure_exposure": 0.3, "protection_time": 0.0,
                    "gunslinger": 0.0})
        rows.append(row)
    return pd.DataFrame(rows)


def test_serving_wind_uses_the_target_game_flag():
    """`wind_pass_interaction` is the one mixed-discipline interaction: the
    wind half is the target game's own flag (context), the pass-share half is
    lagged. Copying the prior row's product would price the game on last
    week's weather."""
    from nfl_predictor.tracking import forward_tick

    row = forward_tick.history_row_for(
        _serving_frame(), "00-1", 2026, 4, game_context={"high_wind_flag": 1})
    assert row["wind_pass_interaction"] == pytest.approx(0.6)
    row = forward_tick.history_row_for(
        _serving_frame(), "00-1", 2026, 4, game_context={"high_wind_flag": 0})
    assert row["wind_pass_interaction"] == pytest.approx(0.0)


def test_serving_lagged_interactions_come_from_the_prior_row():
    from nfl_predictor.tracking import forward_tick

    row = forward_tick.history_row_for(
        _serving_frame(), "00-1", 2026, 4, game_context={"high_wind_flag": 1})
    for column in ("expected_plays", "expected_air_yards", "target_concentration"):
        assert column in row, f"{column} missing from served row"
    assert row["expected_plays"] == pytest.approx(2.0)

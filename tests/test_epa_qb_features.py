"""EPA and starting-QB features: hand-checked numbers, shrinkage, causality, the matchup edges."""
import numpy as np
import pandas as pd
import pytest

from nfl_predictor.data import pbp_agg
from nfl_predictor.features import epa, qb
from epa_fixtures import make_games, make_pbp, perturb_pbp_from, qb_ids


def _play(game, posteam, defteam, kind, e, success=0.0, dropback=0, passer=None):
    return {"game_id": game, "season": 2025, "week": 1, "posteam": posteam, "defteam": defteam, "play_type": kind, "epa": e,
            "success": success, "qb_dropback": dropback, "passer_player_id": passer, "passer_player_name": passer}


def _mini():
    return pd.DataFrame([
        _play("g1", "A", "B", "pass", 0.4, 1.0, 1, "qa"),
        _play("g1", "A", "B", "run", -0.2, 0.0),
        _play("g1", "B", "A", "pass", 0.1, 1.0, 1, "qb"),
        _play("g1", "A", "B", "punt", 5.0),               # not a scrimmage play: ignored
        _play("g1", "A", "B", "pass", np.nan, 0.0, 1, "qa"),  # no EPA value: ignored
    ])


def test_offence_and_defence_efficiency_with_hand_numbers():
    eff = pbp_agg.team_game_efficiency(_mini()).set_index("team")
    assert eff.loc["A", "epa_off"] == pytest.approx(0.1)
    assert eff.loc["A", "epa_off_pass"] == pytest.approx(0.4) and eff.loc["A", "epa_off_rush"] == pytest.approx(-0.2)
    assert eff.loc["A", "success_off"] == pytest.approx(0.5)
    # what A's defence allowed is what B's offence did against it
    assert eff.loc["A", "epa_def"] == pytest.approx(0.1) and eff.loc["A", "epa_def_pass"] == pytest.approx(0.1)
    assert eff.loc["B", "epa_def"] == pytest.approx(0.1)             # A's offence averaged 0.1 against B


def test_the_starting_qb_is_the_passer_with_the_most_dropbacks():
    pbp = pd.DataFrame([
        _play("g1", "A", "B", "pass", 0.5, 1, 1, "starter"), _play("g1", "A", "B", "pass", 0.1, 0, 1, "starter"),
        _play("g1", "A", "B", "pass", -0.5, 0, 1, "backup"),
    ])
    row = pbp_agg.qb_games(pbp).iloc[0]
    assert row["qb_id"] == "starter" and row["dropbacks"] == 2 and row["epa_sum"] == pytest.approx(0.6)


def test_a_dropback_tie_goes_to_the_lowest_id_so_the_choice_is_deterministic():
    pbp = pd.DataFrame([_play("g1", "A", "B", "pass", 0.2, 1, 1, "z"), _play("g1", "A", "B", "pass", 0.2, 1, 1, "a")])
    assert pbp_agg.qb_games(pbp).iloc[0]["qb_id"] == "a"


def test_an_empty_play_by_play_gives_empty_tables_not_an_error():
    assert pbp_agg.team_game_efficiency(pd.DataFrame()).empty and pbp_agg.qb_games(pd.DataFrame()).empty


def test_a_teams_rolling_epa_is_shrunk_toward_zero_by_how_little_we_have_seen():
    games = make_games(seasons=(2025,), weeks=6)
    eff = pbp_agg.team_game_efficiency(make_pbp(games))
    out = epa.add_epa_features(games, eff)
    team = games.sort_values("gameday").iloc[0]["home_team"]
    own = eff[eff["team"] == team].merge(games[["game_id", "gameday"]], on="game_id").sort_values("gameday")
    second_game = own.iloc[1]["game_id"]
    row = out[out["game_id"] == second_game].iloc[0]
    side = "home" if row["home_team"] == team else "away"
    assert row[f"{side}_epa_off_ewm"] == pytest.approx(own.iloc[0]["epa_off"] * (1 / (1 + epa.SHRINK_K)))


def test_epa_features_never_read_the_game_itself_or_anything_later():
    games = make_games()
    pbp = make_pbp(games)
    cut = sorted(games["gameday"].unique())[30]
    a = epa.add_epa_features(games, pbp_agg.team_game_efficiency(pbp))
    b = epa.add_epa_features(games, pbp_agg.team_game_efficiency(perturb_pbp_from(pbp, games, cut)))
    earlier = pd.to_datetime(games["gameday"]) <= pd.Timestamp(cut)
    cols = epa.epa_columns()
    pd.testing.assert_frame_equal(a[earlier][cols].reset_index(drop=True), b[earlier][cols].reset_index(drop=True))


def test_the_matchup_edges_are_offence_minus_the_opposing_defence():
    games = make_games()
    out = epa.add_epa_features(games, pbp_agg.team_game_efficiency(make_pbp(games))).dropna(subset=epa.epa_columns())
    np.testing.assert_allclose(out["home_pass_edge"], out["home_epa_off_pass_ewm"] - out["away_epa_def_pass_ewm"])
    np.testing.assert_allclose(out["away_rush_edge"], out["away_epa_off_rush_ewm"] - out["home_epa_def_rush_ewm"])
    np.testing.assert_allclose(
        out["epa_net_diff"], (out["home_epa_off_ewm"] - out["home_epa_def_ewm"]) - (out["away_epa_off_ewm"] - out["away_epa_def_ewm"]),
    )


def test_no_epa_column_is_constant():
    games = make_games()
    out = epa.add_epa_features(games, pbp_agg.team_game_efficiency(make_pbp(games))).dropna(subset=epa.epa_columns())
    assert [c for c in epa.epa_columns() if out[c].nunique() <= 1] == []


# ---- the QB ---------------------------------------------------------------------------------------------------


def _qb_table():
    return pd.DataFrame([
        {"game_id": "g1", "team": "A", "qb_id": "q1", "qb_name": "Q1", "dropbacks": 100, "epa_sum": 20.0},
        {"game_id": "g2", "team": "A", "qb_id": "q1", "qb_name": "Q1", "dropbacks": 50, "epa_sum": -5.0},
        {"game_id": "g3", "team": "A", "qb_id": "q2", "qb_name": "Q2", "dropbacks": 40, "epa_sum": 8.0},
        {"game_id": "g1", "team": "B", "qb_id": "b1", "qb_name": "B1", "dropbacks": 30, "epa_sum": 3.0},
        {"game_id": "g2", "team": "B", "qb_id": "b1", "qb_name": "B1", "dropbacks": 30, "epa_sum": 3.0},
        {"game_id": "g3", "team": "B", "qb_id": "b1", "qb_name": "B1", "dropbacks": 30, "epa_sum": 3.0},
    ])


def _qb_games_frame():
    return pd.DataFrame([
        {"game_id": f"g{i}", "gameday": f"2025-09-{7 * i:02d}", "home_team": "A", "away_team": "B"} for i in (1, 2, 3)
    ] + [{"game_id": "g4", "gameday": "2025-10-05", "home_team": "A", "away_team": "B"}])


def test_a_qbs_rating_is_his_prior_epa_per_dropback_shrunk_toward_the_prior():
    out = qb.add_qb_features(_qb_games_frame(), _qb_table()).set_index("game_id")
    # g2: q1 has 100 prior dropbacks and 20 EPA: 20 / (100 + 200)
    assert out.loc["g2", "home_qb_epa_pd"] == pytest.approx(20.0 / 300.0)
    # g1: nothing prior: exactly the prior
    assert out.loc["g1", "home_qb_epa_pd"] == pytest.approx(qb.PRIOR_EPA_PER_DROPBACK)


def test_experience_new_and_changed_flags():
    out = qb.add_qb_features(_qb_games_frame(), _qb_table()).set_index("game_id")
    assert out.loc["g2", "home_qb_games"] == 1 and out.loc["g2", "home_qb_new"] == 1.0 and out.loc["g2", "home_qb_changed"] == 0.0
    assert out.loc["g3", "home_qb_changed"] == 1.0           # q2 replaced q1 in A's lineup
    assert out.loc["g3", "home_qb_games"] == 0               # q2 has no earlier games


def test_a_rating_never_includes_the_game_it_is_for():
    t = _qb_table()
    a = qb.add_qb_features(_qb_games_frame(), t).set_index("game_id").loc["g2", "home_qb_epa_pd"]
    t2 = t.copy()
    t2.loc[(t2["game_id"] == "g2") & (t2["team"] == "A"), "epa_sum"] = 999.0
    b = qb.add_qb_features(_qb_games_frame(), t2).set_index("game_id").loc["g2", "home_qb_epa_pd"]
    assert a == b


def test_an_upcoming_game_uses_the_expected_starter_and_unknown_means_the_prior():
    out = qb.add_qb_features(_qb_games_frame(), _qb_table(), upcoming_starters={("g4", "A"): "q1", ("g4", "B"): None}).set_index("game_id")
    assert out.loc["g4", "home_qb_epa_pd"] == pytest.approx(15.0 / (150.0 + 200.0))
    assert out.loc["g4", "home_qb_games"] == 2 and out.loc["g4", "home_qb_changed"] == 1.0   # q2 started A's last game
    assert out.loc["g4", "away_qb_games"] == 0 and out.loc["g4", "away_qb_new"] == 1.0 and out.loc["g4", "away_qb_epa_pd"] == qb.PRIOR_EPA_PER_DROPBACK


def test_expected_starters_skip_reported_out_and_sort_the_depth_slot_numerically():
    chart = pd.DataFrame([
        {"club_code": "A", "position": "QB", "depth_team": "2", "gsis_id": "second"},
        {"club_code": "A", "position": "QB", "depth_team": "1", "gsis_id": "first"},
        {"club_code": "A", "position": "QB", "depth_team": "10", "gsis_id": "tenth"},
        {"club_code": "A", "position": "WR", "depth_team": "1", "gsis_id": "wr"},
        {"club_code": "B", "position": "QB", "depth_team": "1", "gsis_id": "only"},
    ])
    assert qb.expected_starters(chart) == {"A": "first", "B": "only"}
    assert qb.expected_starters(chart, out_ids={"first"}) == {"A": "second", "B": "only"}
    assert qb.expected_starters(chart, out_ids={"first", "second", "tenth"})["A"] is None
    assert qb.expected_starters(pd.DataFrame()) == {}


def test_real_shaped_data_gives_varied_qb_features():
    games = make_games()
    qbg = pbp_agg.qb_games(make_pbp(games))
    out = qb.add_qb_features(games, qbg)
    assert out["home_qb_epa_pd"].nunique() > 20 and out["home_qb_changed"].sum() > 0 and out["home_qb_new"].sum() > 0
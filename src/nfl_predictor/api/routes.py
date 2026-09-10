"""routes.py — game/player prop/track-record endpoints. Thin HTTP layer
over data/features/models/tracking, same shape as PL_Predictor's/
F1_Predictor's own api/routes.py."""

from __future__ import annotations

import json
import logging
from datetime import date
from functools import lru_cache

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException

from ..config import (
    CURRENT_SEASON,
    PUBLIC_MODE,
    PUBLIC_SNAPSHOT_PATH,
    PUBLIC_SNAPSHOT_REFRESH_URL,
)
from ..data import odds_api, player_stats, schedules
from ..features import build as feature_build
from ..features import player_usage
from ..models import game_outcome, manifest, player_props
from ..odds import value_bets
from ..tracking import store

router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)

_public_snapshot_cache: dict | None = None
_public_snapshot_etag: str | None = None


def _public_snapshot() -> dict:
    """The precomputed data public_snapshot.py generates. Cold-start value
    is whatever was baked into this image at build time; a running process
    then keeps it current via refresh_public_snapshot_from_remote below,
    polled on a timer (see api/main.py's lifespan). Empty dict if none has
    been generated yet, so every PUBLIC_MODE branch below just falls
    through to a live compute instead of a 500."""
    global _public_snapshot_cache
    if _public_snapshot_cache is None:
        _public_snapshot_cache = (
            json.loads(PUBLIC_SNAPSHOT_PATH.read_text()) if PUBLIC_SNAPSHOT_PATH.exists() else {}
        )
    return _public_snapshot_cache


def refresh_public_snapshot_from_remote() -> bool:
    """Re-fetch the precomputed public snapshot from its GitHub raw URL and
    swap it in -- lets the scheduled snapshot-refresh GitHub Action reach
    this running process without a Docker rebuild+redeploy. ETag-conditional
    so an unchanged snapshot costs one small request. Any fetch/parse
    failure is swallowed and leaves the previous snapshot serving -- a
    transient network hiccup here must never blank an already-working
    public site."""
    global _public_snapshot_cache, _public_snapshot_etag
    try:
        headers = {"If-None-Match": _public_snapshot_etag} if _public_snapshot_etag else {}
        resp = requests.get(PUBLIC_SNAPSHOT_REFRESH_URL, headers=headers, timeout=30)
        if resp.status_code == 304:
            return False
        resp.raise_for_status()
        snapshot = resp.json()
    except Exception as exc:
        logger.info("public snapshot remote refresh skipped: %s", exc)
        return False
    _public_snapshot_cache = snapshot
    _public_snapshot_etag = resp.headers.get("ETag")
    return True


def _snapshot_week(season: int, week: int) -> dict | None:
    snap = _public_snapshot()
    if snap.get("season") != season:
        return None
    return snap.get("weeks", {}).get(str(week))


def current_season_and_week() -> tuple[int, int]:
    """Current NFL season/week, anchored to the real schedule's own week-1
    kickoff date rather than a hardcoded month/day guess. A fixed
    `date(season, 9, 1)` anchor drifts every year the season's actual
    opening week doesn't start exactly then (confirmed live: it was a full
    week ahead of the real week 1, which kicks off whenever the schedule
    says it does, not on a fixed calendar date)."""
    today = date.today()
    season = today.year if today.month >= 3 else today.year - 1
    try:
        schedule = schedules.fetch_schedules([season])
        week1_start = schedule.loc[schedule["week"] == 1, "gameday"].min()
        anchor = week1_start.date() if pd.notna(week1_start) else date(season, 9, 1)
    except Exception:
        anchor = date(season, 9, 1)
    week = max(1, min(22, ((today - anchor).days // 7) + 1))
    return season, week


@router.get("/current-week")
def get_current_week():
    season, week = current_season_and_week()
    return {"season": season, "week": week}


@lru_cache(maxsize=1)
def _load_models_cached() -> dict:
    return manifest.load_models()


def _predict_game_from_models(
    models: dict, home: str, away: str, games_df: pd.DataFrame,
    spread_line: float | None = None, total_line: float | None = None,
) -> dict:
    feature_row = feature_build.build_features_for_game(home, away, games_df)
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0)

    if models["chosen_candidate"] == "elo":
        predicted_margin = game_outcome.predict_margin_elo(
            {"points_per_rating_point": game_outcome.ELO_POINTS_PER_RATING_POINT},
            feature_row["rating_diff"], feature_row["home_rest_days"], feature_row["away_rest_days"],
        )
    else:
        predicted_margin = float(models["game_outcome_model"].predict(X.to_numpy().reshape(1, -1))[0])

    predicted_total = float(models["total_model"].predict(X.to_numpy().reshape(1, -1))[0])

    return game_outcome.margin_to_probabilities(
        predicted_margin, models["sigma"],
        spread_line=spread_line, total_line=total_line,
        predicted_total=predicted_total, total_sigma=models["total_sigma"],
    )


def _load_game_history(season: int) -> pd.DataFrame:
    """Historical + requested-season game data for feature building.

    Completed seasons are safe to read through load_training_data()'s
    per-season parquet cache — a finished season's results never change.
    But when `season` is CURRENT_SEASON (a season still being played), that
    same call would cache the season's games-so-far on first request and
    then return that SAME STALE snapshot on every later request that
    season, even after more games are played. schedules.py already has a
    dedicated fetch_current_season_partial() for exactly this case
    (refetched every call, no per-season cache) — use it instead of
    folding the current season into load_training_data().
    """
    history_seasons = schedules.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_df = schedules.load_training_data(history_seasons)
        current_df = schedules.fetch_current_season_partial()
        return pd.concat([historical_df, current_df], ignore_index=True)
    return schedules.load_training_data(history_seasons + [season])


def _load_player_history(season: int) -> pd.DataFrame:
    """Historical + requested-season player stats for feature building.

    Same staleness concern as _load_game_history, but player_stats.py has
    no dedicated "current season, always refetch" helper (only a single
    fetch_weekly_player_stats(seasons, force_refresh) applying to the
    whole list). Rather than extend that module (out of scope for this
    task), fetch the current season on its own with force_refresh=True
    (cheap — one season/week of data per nfl_data_py call) and merge it
    with normally-cached historical seasons.
    """
    history_seasons = schedules.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_df = player_stats.fetch_weekly_player_stats(history_seasons)
        try:
            current_df = player_stats.fetch_weekly_player_stats([season], force_refresh=True)
        except Exception:
            # nflverse doesn't publish a weekly-stats file for the current
            # season until its first games are played (404 before then) —
            # historical-only is the correct fallback, not a hard failure.
            logger.info("No weekly player stats available yet for season=%s", season)
            current_df = historical_df.iloc[0:0]
        return pd.concat([historical_df, current_df], ignore_index=True)
    return player_stats.fetch_weekly_player_stats(history_seasons + [season])


@router.get("/games")
def get_games(season: int, week: int):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap["games"]
    return _get_games_live(season, week)


def _get_games_live(season: int, week: int):
    games = schedules.fetch_week_games(season, week)
    # Unplayed games (the entire point of this endpoint) have home_score/
    # away_score as pandas NaN, not None. Starlette's default JSONResponse
    # uses json.dumps(..., allow_nan=False), so an unconverted NaN raises
    # ValueError -> unhandled 500. Swap NaN for None before returning.
    # NOTE: .where(cond, None) alone is NOT enough here -- on a float64
    # column pandas re-casts the replacement None right back into NaN to
    # preserve the column's dtype. astype(object) first forces the frame
    # into per-cell Python objects so None actually sticks (and other
    # values, e.g. gameday's pd.Timestamp, pass through unchanged).
    games = games.astype(object).where(pd.notna(games), None)
    return games.to_dict("records")


@router.get("/games/{season}/{week}/{game_id}/prediction")
def get_game_prediction(season: int, week: int, game_id: str):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            pred = snap["predictions"].get(game_id)
            if pred is not None:
                return pred
            # Snapshot covers this week but not this specific game's
            # prediction (a build-time failure, or a genuinely unknown
            # game_id) -- fall through to a live compute/404 rather than
            # treating "missing from a partial snapshot" as fatal.
    return _get_game_prediction_live(season, week, game_id)


def _get_game_prediction_live(season: int, week: int, game_id: str):
    games = schedules.fetch_week_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0]

    models = _load_models_cached()
    # Exclude the game's own row from its feature history -- fetch_week_games
    # (unlike the old fetch_upcoming_games) makes already-finished games
    # reachable here too, and without this exclusion a finished game's
    # prediction would leak its own result into its own features. Mirrors
    # the same exclusion background_tracking_tick's backfill path already
    # does for exactly this reason.
    history = _load_game_history(season)
    history = history[history["game_id"] != game_id]
    prediction = _predict_game_from_models(
        models, game["home_team"], game["away_team"], history,
        spread_line=game.get("spread_line"), total_line=game.get("total_line"),
    )
    return prediction


@router.get("/players/{season}/{week}/props")
def get_player_props(season: int, week: int):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap["player_props"]
    return _get_player_props_live(season, week)


def _get_player_props_live(season: int, week: int):
    try:
        models = _load_models_cached()
        player_history = _load_player_history(season)

        # 1. Fetch upcoming games for this specific week to find active teams
        upcoming_games = schedules.fetch_upcoming_games(season, week)
        if upcoming_games.empty:
            # Fallback to completed partial games if none are upcoming
            upcoming_games = schedules.fetch_current_season_partial()
            upcoming_games = upcoming_games[upcoming_games["week"] == week]

        if upcoming_games.empty:
            return []

        active_teams = set(upcoming_games["home_team"]).union(set(upcoming_games["away_team"]))

        # 2. Filter players only belonging to the active teams playing this week
        latest_players = (
            player_history[
                (player_history["season"] == season) &
                (player_history["recent_team"].isin(active_teams))
            ]
            [["player_id", "player_name", "position", "recent_team"]]
            .drop_duplicates("player_id")
        )

        # 3. Fallback: a team with no current-season stats yet (week 1, or a
        # bye-to-opener gap) has no rows above even though its roster exists —
        # pull the season roster for just those teams so props aren't empty.
        found_teams = set(latest_players["recent_team"].unique())
        missing_teams = active_teams - found_teams
        if missing_teams:
            try:
                roster = player_stats.fetch_seasonal_roster(season)
                fallback = roster[
                    roster["recent_team"].isin(missing_teams) & roster["position"].isin(player_props.POSITION_YARDAGE_MARKET)
                ]
                latest_players = pd.concat([latest_players, fallback], ignore_index=True).drop_duplicates("player_id")
            except Exception as roster_err:
                logger.warning("Failed to fetch season roster fallback for teams=%s: %s", missing_teams, roster_err)

        results = []
        for _, player in latest_players.iterrows():
            try:
                feature_row = player_usage.build_features_for_player(player["player_id"], player_history)
                if feature_row is None:
                    # No usage history anywhere in player_history (true rookie,
                    # or a player the roster fallback pulled in) — predict off
                    # a neutral zero-usage baseline rather than skipping them.
                    feature_row = pd.Series({col: 0.0 for col in player_usage.PLAYER_FEATURE_COLUMNS})
                props = player_props.predict_props(models["player_models"], feature_row, position=player["position"])
                results.append({
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "recent_team": player["recent_team"],
                    "position": player["position"],
                    **props,
                })
            except Exception as player_err:
                logger.warning("Failed to predict props for player_id=%s: %s", player.get("player_id"), player_err)
                continue
        return results
    except Exception as e:
        logger.exception("Failed to load player props for season=%s week=%s", season, week)
        return []


@router.get("/track-record")
def get_track_record():
    return store.get_track_record()


@router.get("/games/{game_id}/verdict")
def get_game_verdict(game_id: str):
    verdict = store.get_game_verdict(game_id)
    if verdict is None:
        raise HTTPException(status_code=404, detail="Game not tracked or not yet resolved")
    return verdict


@router.get("/predictions/{season}/{week}")
def get_predictions_for_week(season: int, week: int):
    # A union, not an if/else fallback: a week in progress has both
    # already-final games and still-upcoming ones, and the old if/else
    # dropped every finished game (and its verdict) whenever any game in
    # the week was still unplayed (see this plan's final review, finding B3).
    upcoming = schedules.fetch_upcoming_games(season, week)
    finished = schedules.load_training_data(seasons=[season])
    finished = finished[finished["week"] == week]
    games = pd.concat([finished, upcoming], ignore_index=True).drop_duplicates(subset="game_id")
    return store.get_predictions_for_week(season, week, games)


@router.post("/retrain")
def retrain():
    if PUBLIC_MODE:
        raise HTTPException(status_code=403, detail="Retraining is disabled in public mode")
    result = manifest.train_all()
    _load_models_cached.cache_clear()
    return {"trained_at": result["trained_at"], "chosen_candidate": result["chosen_candidate"]}


def _attach_game_id(stats_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """Join weekly player stats to the game_id of the game each row's
    player actually played in, matched by season/week/team -- weekly
    stats carry a team and week but no game_id of their own."""
    if stats_df.empty or games_df.empty:
        return stats_df.iloc[0:0]
    home = games_df[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "recent_team"})
    away = games_df[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "recent_team"})
    team_game = pd.concat([home, away], ignore_index=True)
    return stats_df.merge(team_game, on=["season", "week", "recent_team"], how="inner")


def background_tracking_tick(season: int, week: int) -> None:
    """Snapshot this week's upcoming-game (and player-prop) predictions,
    then reconcile anything now resolved. Called on a timer from
    api/main.py's lifespan the same way PL_Predictor's own
    background_tracking_tick is."""
    games = schedules.fetch_upcoming_games(season, week)
    if not games.empty:
        models = _load_models_cached()
        history = _load_game_history(season)
        predictions = []
        for _, game in games.iterrows():
            try:
                pred = _predict_game_from_models(
                    models, game["home_team"], game["away_team"], history,
                    spread_line=game.get("spread_line"), total_line=game.get("total_line"),
                )
                predictions.append(
                    {
                        "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                        "commence_time": str(game["gameday"]), "season": season, "week": week,
                        "home_spread_line": game.get("spread_line"), "total_line": game.get("total_line"),
                        **pred,
                    }
                )
            except Exception:
                logger.exception("prediction failed for game_id=%s", game.get("game_id"))
                continue
        store.record_game_predictions(predictions)

        try:
            team_to_game = {}
            for _, g in games.iterrows():
                team_to_game[g["home_team"]] = g["game_id"]
                team_to_game[g["away_team"]] = g["game_id"]
            prop_rows = []
            for prop in _get_player_props_live(season, week):
                game_id = team_to_game.get(prop["recent_team"])
                if game_id is None:
                    continue
                prop_rows.append({
                    "game_id": game_id, "player_id": prop["player_id"], "player_name": prop["player_name"],
                    "market": "anytime_td", "predicted_value": prop["anytime_td_prob"],
                })
                for market in ("passing_yards", "rushing_yards", "receiving_yards"):
                    value = prop.get(market)
                    if value is not None:
                        prop_rows.append({
                            "game_id": game_id, "player_id": prop["player_id"], "player_name": prop["player_name"],
                            "market": market, "predicted_value": value,
                        })
            store.record_player_prop_predictions(prop_rows)
        except Exception:
            logger.exception("player prop snapshot failed for season=%s week=%s", season, week)

    completed = schedules.fetch_current_season_partial()
    store.reconcile_game_predictions(completed[["game_id", "home_score", "away_score"]])

    # Backfill games that finished before this tracker ever ran once
    # (e.g. background_tracking_tick wasn't wired up / wasn't running
    # yet) -- reconcile only ever updates an EXISTING snapshot, so a game
    # with no snapshot at all would otherwise never get a verdict.
    # Excludes each game from its own history so the prediction still
    # reflects strictly pre-game information, not this game's own result.
    try:
        if not completed.empty:
            untracked_ids = store.get_untracked_game_ids(list(completed["game_id"]))
            if untracked_ids:
                models = _load_models_cached()
                history_all = _load_game_history(season)
                backfill_games = []
                for _, game in completed[completed["game_id"].isin(untracked_ids)].iterrows():
                    try:
                        history_excl = history_all[history_all["game_id"] != game["game_id"]]
                        pred = _predict_game_from_models(
                            models, game["home_team"], game["away_team"], history_excl,
                            spread_line=game.get("spread_line"), total_line=game.get("total_line"),
                        )
                        backfill_games.append({
                            "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                            "commence_time": str(game["gameday"]), "season": season,
                            "week": int(game["week"]) if pd.notna(game.get("week")) else None,
                            "home_spread_line": game.get("spread_line"), "total_line": game.get("total_line"),
                            "actual_home_score": game["home_score"], "actual_away_score": game["away_score"],
                            **pred,
                        })
                    except Exception:
                        logger.exception("backfill prediction failed for game_id=%s", game.get("game_id"))
                        continue
                store.record_resolved_game_predictions(backfill_games)
    except Exception:
        logger.exception("backfill of untracked finished games failed")

    try:
        if not completed.empty:
            actual_stats = player_stats.fetch_weekly_player_stats([season])
            store.reconcile_player_prop_predictions(_attach_game_id(actual_stats, completed))
    except Exception:
        logger.exception("player prop reconciliation failed")

    try:
        store.backfill_unresolved_games(schedules)
    except Exception:
        logger.exception("backfill_unresolved_games failed")


def warm_caches() -> None:
    """Pre-fetch schedules/player stats/odds so the first real request
    after startup isn't slow — best-effort, never raises."""
    try:
        schedules.fetch_schedules(schedules.default_completed_seasons(n=8))
        player_stats.fetch_weekly_player_stats(schedules.default_completed_seasons(n=8))
        odds_api.fetch_game_odds()
    except Exception:
        pass

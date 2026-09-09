"""routes.py — game/player prop/track-record endpoints. Thin HTTP layer
over data/features/models/tracking, same shape as PL_Predictor's/
F1_Predictor's own api/routes.py."""

from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..config import CURRENT_SEASON, PUBLIC_MODE
from ..data import odds_api, player_stats, schedules
from ..features import build as feature_build
from ..features import player_usage
from ..models import game_outcome, manifest, player_props
from ..odds import value_bets
from ..tracking import store

router = APIRouter(prefix="/api")

logger = logging.getLogger(__name__)


def current_season_and_week() -> tuple[int, int]:
    """Calendar-based estimate for current NFL season and week."""
    today = date.today()
    season = today.year if today.month >= 3 else today.year - 1
    week = max(1, min(22, ((today - date(season, 9, 1)).days // 7) + 1))
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
    games = schedules.fetch_upcoming_games(season, week)
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
    games = schedules.fetch_upcoming_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0]

    models = _load_models_cached()
    history = _load_game_history(season)
    prediction = _predict_game_from_models(
        models, game["home_team"], game["away_team"], history,
        spread_line=game.get("spread_line"), total_line=game.get("total_line"),
    )
    return prediction


@router.get("/players/{season}/{week}/props")
def get_player_props(season: int, week: int):
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


def background_tracking_tick(season: int, week: int) -> None:
    """Snapshot this week's upcoming-game predictions, then reconcile
    anything now resolved. Called on a timer from api/main.py's lifespan
    the same way PL_Predictor's own background_tracking_tick is."""
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
                        "commence_time": str(game["gameday"]), "season": season,
                        "home_spread_line": game.get("spread_line"), "total_line": game.get("total_line"),
                        **pred,
                    }
                )
            except Exception:
                logger.exception("prediction failed for game_id=%s", game.get("game_id"))
                continue
        store.record_game_predictions(predictions)

    completed = schedules.fetch_current_season_partial()
    store.reconcile_game_predictions(completed[["game_id", "home_score", "away_score"]])

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

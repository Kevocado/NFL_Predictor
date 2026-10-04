"""forward_tick.py -- snapshot a slate of book lines and log the ones worth taking.

For each pre-kickoff game: fetch the book's prop lines, turn them into
P(over) with the quantile models, and record every side whose edge clears 5%.
Either side qualifies -- a 5% under is as much a pick as a 5% over -- and the
side taken is written explicitly, because CLV's sign and the hit test both
depend on it.

**A post-kickoff game is skipped, not corrected.** A line recorded after
kickoff is not a bettable price, so it is not a pick. The rest of the slate
still runs.

**The budget guard runs before the first fetch.** An exhausted month writes
nothing at all rather than a partial slate that looks like a record.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..models.prop_probability import american_to_breakeven, edge_vs_line, p_over_from_quantiles
from ..models.quantile_registry import (
    ARTIFACT_SUFFIX, load_quantile_artifact, verify_quantile_artifacts,
)
from ..models.training import FORWARD_FEATURE_COLUMNS
from ..odds.props_snapshot import (
    BudgetExhausted, credits_sufficient, fetch_event_index, fetch_props_for_event,
    match_event_id, match_props_to_players,
)
from . import store

logger = logging.getLogger(__name__)

#: Kevin's gate: singles only, 5% edge against the line.
EDGE_GATE = 0.05

class MissingPlayerIndex(RuntimeError):
    """No player index: prop names cannot be joined to nflverse ids."""


class MissingEventIndex(RuntimeError):
    """No Odds event index: a slate game_id is not an Odds event id.

    Raised rather than defaulting to an empty mapping, which would skip every
    game and report a clean, empty, wrong week.
    """


#: Book market key -> the market name the quantile models were trained on.
MARKET_MAP = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_rec_yds": "receiving_yards",
    "player_receptions": "receptions",
}

#: Reverse of `MARKET_MAP`, for loading artifacts named after the model market.
BOOK_MARKET_BY_MODEL = {model: book for book, model in MARKET_MAP.items()}

#: Forward picks are stored under a `fwd_`-prefixed market name.
#:
#: `player_prop_predictions` is keyed (game_id, player_id, market) and written
#: with INSERT OR IGNORE, so storing a forward pick as `receiving_yards` either
#: loses it silently -- the background tracking tick may already hold that
#: player/game -- or, if it lands, has its `predicted_value` read as a model's
#: point estimate. That value is the BOOK'S LINE, not a q50, so it would corrupt
#: the published MAE and signed error for that market.
FORWARD_MARKET_PREFIX = "fwd_"


def forward_market(market: str) -> str:
    """The storage name for a forward pick on `market`."""
    return f"{FORWARD_MARKET_PREFIX}{market}"


def quantiles_for(market: str) -> dict[float, float]:
    """Predicted quantiles for one prop row. Replaced at the call site by the
    trained model; named here so the tick's seam is one function."""
    raise NotImplementedError("wire a trained quantile artifact in via quantiles_for")


def predictor_for(models_dir: Path | str) -> dict[str, callable]:
    """Load the versioned artifacts into `{book market: predict(quantiles)}`.

    Raises FileNotFoundError when the directory holds no quantile artifact.
    Silently ticking with no model would price every prop as "no edge" and
    report a clean week having looked at nothing -- the failure would look like
    a good result, which is the worst shape a failure can take.
    """
    models_dir = Path(models_dir)
    paths = sorted(models_dir.glob(f"*{ARTIFACT_SUFFIX}.pkl")) if models_dir.exists() else []
    if not paths:
        raise FileNotFoundError(
            f"no quantile artifacts in {models_dir}. Expected "
            f"models/<market>{ARTIFACT_SUFFIX}.pkl from "
            f"scripts/train_quantile_props.py --write-artifacts")

    # Verify before serving. `load_quantile_artifact` is a bare pickle.loads, so
    # without this a stale artifact from an earlier fit would price picks
    # silently -- the exact incident models/manifest.py documents at length.
    manifest_path = models_dir / "manifest.json"
    if manifest_path.exists():
        import json

        problems = verify_quantile_artifacts(json.loads(manifest_path.read_text()),
                                             out_dir=models_dir)
        if problems:
            raise RuntimeError(f"quantile artifacts do not verify: {problems}")

    predictors: dict[str, callable] = {}
    for path in paths:
        # The stem already carries the market name; `load_quantile_artifact`
        # re-joins the suffix itself, so do not add it a second time.
        market = path.name.removesuffix(".pkl").removesuffix(ARTIFACT_SUFFIX)
        payload = load_quantile_artifact(market, out_dir=models_dir)
        models, feature_cols = payload["quantile_models"], payload["feature_cols"]
        predictors[BOOK_MARKET_BY_MODEL[market]] = _row_predictor(models, feature_cols)
    return predictors


def _row_predictor(models: dict, feature_cols: list[str]) -> callable:
    """`predict(feature_row, line) -> {level: value}` for one market.

    **The feature row is a required positional argument, not an optional
    keyword.** With it optional, `predict(line)` builds an all-zero frame, every
    player gets an identical distribution, and the 5% gate then fires off the
    book's line alone -- the model is bypassed and nothing looks wrong. Making it
    required turns that silent money-loser into a `TypeError`.

    Fitted models predict a batch, so a single row is wrapped in a one-row frame
    and unwrapped.

    A column still missing here is filled with 0, which is NOT what training did:
    training fills `rest_days` with a league-typical 7.0. That is why the tick
    supplies the game's context from the slate rather than relying on this fill
    -- `game_context_for` is the only place `is_home` and `rest_days` can get
    values that are right rather than merely present.
    """
    levels = sorted(models)

    def predict(feature_row: dict, line: float):
        import pandas as pd

        if not feature_row:
            raise ValueError(
                f"no feature row for this prop; the quantile model cannot price a "
                f"line without one (columns expected: {feature_cols})")
        frame = pd.DataFrame([feature_row]).reindex(columns=feature_cols).fillna(0)
        return {q: float(models[q].predict(frame)[0]) for q in levels}

    return predict


def game_context_for(game: dict, prop: dict, feature_frame=None) -> dict:
    """Context for the game being played, from the game's own record.

    `is_home` and `rest_days` are exact: the schedule carries `home_rest` /
    `away_rest`, and home/away follows from the prop's team. Taking these from
    the player's last observed row instead supplies the PREVIOUS game's home
    flag and rest days -- wrong values, not stale ones, and `is_home` is a
    material yardage driver.

    Weather is genuinely unknown at snapshot time (the archive endpoint lags
    real time), so it is imputed from the feature frame's own distribution for
    that column. Filling it with 0 would assert a freezing game every week and a
    perfectly still one; the frame's median is a neutral stand-in and is what
    the value would be on average.
    """
    team = prop.get("team")
    context: dict = {}
    if team and game.get("home_team") and game.get("away_team"):
        context["is_home"] = 1 if team == game["home_team"] else 0
    if team == game.get("away_team") and game.get("away_rest") is not None:
        context["rest_days"] = game["away_rest"]
    elif team == game.get("home_team") and game.get("home_rest") is not None:
        context["rest_days"] = game["home_rest"]

    for column in ("is_outdoor", "temp_c", "wind_kph", "precip_mm", "high_wind_flag"):
        if feature_frame is not None and column in getattr(feature_frame, "columns", []):
            median = feature_frame[column].median()
            context[column] = None if pd.isna(median) else float(median)
    return context


def _target_season_week(game: dict) -> tuple[int, int]:
    """The season/week a game belongs to, from whichever field carries it."""
    try:
        return int(game["season"]), int(game["week"])
    except (KeyError, TypeError, ValueError):
        return 0, 0


#: Features describing the game being played, not the player's history. Taken
#: from the target game itself: a week-3 row carries week 3's `is_home`, and
#: week 3's rest days, which are wrong values for a week-4 game rather than
#: merely stale ones. `is_home` is a material yardage driver, so this is not a
#: rounding error.
TARGET_GAME_COLUMNS: tuple[str, ...] = (
    "is_home", "rest_days", "is_outdoor", "temp_c", "wind_kph", "precip_mm",
    "high_wind_flag",
)


def history_row_for(feature_frame, player_id: str, season: int, week: int,
                    game_context: dict | None = None) -> dict:
    """Feature row for the target game: lagged history plus the game's own context.

    The lagged half comes from the player's most recent row STRICTLY before the
    target week, so it carries only games before that row's week -- less
    information than training used (it omits the most recent game), and never a
    leak. The context half describes the game being played and MUST come from the
    target game; taking it from the history row would silently supply the wrong
    home/away, rest days and weather.

    Returns {} when the player has no prior row, which `_row_predictor` rejects.
    A skipped prop beats a zero-filled one.
    """
    if feature_frame is None or feature_frame.empty:
        return {}
    prior = feature_frame[
        (feature_frame["player_id"] == player_id)
        & ((feature_frame["season"] < season)
           | ((feature_frame["season"] == season) & (feature_frame["week"] < week)))
    ]
    if prior.empty:
        return {}

    row = prior.sort_values(["season", "week"]).iloc[-1]
    context = game_context or {}
    result = {}
    for column in FORWARD_FEATURE_COLUMNS:
        if column in TARGET_GAME_COLUMNS:
            result[column] = context.get(column, float("nan"))
        elif column in prior.columns:
            result[column] = row[column]
    return result


def _has_spread(quantiles: dict) -> bool:
    """False when every predicted quantile is the same value.

    A degenerate distribution is a model that failed to fit this row, not a
    confident one, and pricing it produces a confident-looking number.
    """
    values = [float(v) for v in quantiles.values() if v is not None and not pd.isna(v)]
    return len(values) >= 2 and (max(values) - min(values)) > 0


def _is_post_kickoff(commence_time: str) -> bool:
    try:
        kickoff = store._utc_instant(commence_time)
    except Exception:
        # An unparseable kickoff cannot be proven pre-kickoff, so it is skipped.
        return True
    if kickoff is None:
        return True
    return kickoff <= datetime.now(timezone.utc)


def run_forward_tick(games: list[dict], market_quantiles: dict[str, dict] | None = None,
                     edge_gate: float = EDGE_GATE,
                     players: list[dict] | None = None,
                     feature_frame=None,
                     event_index: dict | None = None) -> dict:
    """Snapshot one slate.

    `market_quantiles` maps a book market key to that row's predicted quantiles
    -- either a plain `{level: value}` dict, or a callable taking the line and
    returning one, which is how a trained model is injected. Injecting rather
    than importing is what lets the tick be tested without an artifact on disk.

    Returns the counters the forward report reads.
    """
    games = list(games)
    result = {
        "games": 0, "games_skipped_post_kickoff": 0,
        "props_snapshotted": 0, "picks_logged": 0, "credits_remaining": 0,
        "no_props_coverage": False, "degenerate_rows": 0,
    }

    live = []
    for game in games:
        if _is_post_kickoff(game["commence_time"]):
            result["games_skipped_post_kickoff"] += 1
            logger.info("skipping %s: kickoff has passed", game["game_id"])
            continue
        live.append(game)
    result["games"] = len(live)
    if not live:
        return result

    # One props call per game per tick. The whole slate is costed up front so a
    # month that cannot afford it writes nothing rather than half a slate.
    # One props call per game. The scores probe that checks affordability does
    # not consume a props credit, so it is not budgeted; the old +1 here could
    # refuse an affordable slate at the monthly boundary.
    needed = len(live)
    if not credits_sufficient(needed):
        logger.warning("budget: %d credits needed, not fetching anything", needed)
        result["credits_remaining"] = 0
        result["no_props_coverage"] = True
        return result

    if event_index is None:
        raise MissingEventIndex(
            "no Odds event index supplied; an nflverse game_id is not an Odds "
            "event id. Build one with props_snapshot.fetch_event_index() "
            "(a free call) and pass it in.")

    rows: list[dict] = []
    for game in live:
        # The Odds API's event id is an opaque string (`e91a...`), NOT the
        # nflverse game_id. Passing the game_id straight into the URL 404s on
        # every game -- and no test could see it, because the tests stub the
        # fetch. Looked up from the FREE events list instead.
        event_id = match_event_id(game, event_index)
        if event_id is None:
            # The book is not listing this game. Not a reason to spend a credit
            # finding out, and never a reason to fall back to another market.
            logger.warning("no Odds event for %s (%s at %s); skipping",
                           game.get("game_id"), game.get("away_team"),
                           game.get("home_team"))
            result["games_not_listed"] = result.get("games_not_listed", 0) + 1
            continue
        try:
            props = fetch_props_for_event(event_id, credits_needed=1)
        except BudgetExhausted:
            logger.warning("budget exhausted mid-slate at %s; stopping", event_id)
            break
        # The Odds API returns names, not ids. Without this join every real prop
        # row is missing `player_id` and the write raises.
        if players is None:
            # The Odds API returns names, not ids, and every snapshot row is keyed
            # by player_id. Without the join nothing is writable. This is a
            # MISCONFIGURATION, not an absence of coverage, so it is reported
            # distinctly -- collapsing the two would file a missing flag as "the
            # book had no props this week".
            raise MissingPlayerIndex(
                "no player index supplied: book prop names cannot be joined to "
                "nflverse ids, so nothing can be recorded. Pass --players-path.")
        result["props_joined"] = result.get("props_joined", 0) + len(props)
        props = match_props_to_players(props, players)
        if not props:
            # The fetch's own result IS the coverage answer. A separate probe was
            # a second full props pull per game, charged whether or not it was
            # needed, and it spent credit even on an exhausted month.
            logger.warning("no player-prop coverage for %s; waiting, not falling back",
                           event_id)
            result["no_props_coverage"] = True
            continue
        result["props_snapshotted"] += len(props)

        for prop in props:
            market = MARKET_MAP.get(prop["market"])
            predictor = (market_quantiles or {}).get(prop["market"])
            if market is None or predictor is None:
                continue
            if callable(predictor):
                # The feature row is mandatory. Predicting from the line alone
                # bypasses the model entirely: every player gets the same
                # distribution and the edge gate fires off the book's line.
                season, week = _target_season_week(game)
                context = game_context_for(game, prop, feature_frame)
                row = (history_row_for(feature_frame, prop["player_id"], season, week,
                                       game_context=context)
                       if feature_frame is not None else {})
                quantiles = predictor(row, prop["line"])
            else:
                quantiles = predictor
            if not _has_spread(quantiles):
                # A zero-width distribution cannot price a line: `p_over_from_
                # quantiles` returns the ceiling for a line sitting on it, which
                # would clear the 5% gate on a fabricated 98% edge.
                logger.warning("degenerate quantiles for %s %s; skipped",
                               prop["player_name"], prop["market"])
                result["degenerate_rows"] += 1
                continue
            p_over = p_over_from_quantiles(quantiles, prop["line"])

            over_edge = edge_vs_line(p_over, prop["over_odds"])
            under_edge = edge_vs_line(1.0 - p_over, prop["under_odds"])
            if over_edge >= edge_gate:
                rows.append(_row(game, prop, market, "over", p_over, over_edge, prop["over_odds"]))
            if under_edge >= edge_gate:
                rows.append(_row(game, prop, market, "under", 1.0 - p_over, under_edge, prop["under_odds"]))

    if rows:
        result["picks_logged"] = store.record_player_prop_predictions(rows)
    return result


def _row(game, prop, market, side, p_side, edge, odds) -> dict:
    return {
        "game_id": game["game_id"],
        "player_id": prop["player_id"],
        "player_name": prop["player_name"],
        "position": prop.get("position"),
        "market": forward_market(market),
        "side": side,
        # The book's line, not a model point estimate. Safe only because the
        # market name is namespaced -- see FORWARD_MARKET_PREFIX.
        "predicted_value": float(prop["line"]),
        "line_at_snapshot": float(prop["line"]),
        "odds_at_snapshot": float(odds) if odds is not None else None,
        "model_p_over": float(p_side),
        "edge_vs_breakeven": float(edge),
    }

# --- the CLI ---------------------------------------------------------------

def _load_games(path: str | None, slate: str | None = None,
                season: int | None = None, week: int | None = None) -> list[dict]:
    """The games to tick, from a JSON file or the schedules feed.

    An empty slate is an error rather than a quiet zero. "No games today" and
    "looked at nothing" produce the same empty result, and the second one would
    be reported as a clean week.
    """
    if path:
        import json

        with open(path) as handle:
            games = json.load(handle)
        if not isinstance(games, list):
            raise ValueError(f"{path}: expected a list of games")
    elif slate:
        target_season, target_week = (int(part) for part in slate.split(":", 1))
        games = _fetch_slate(target_season, target_week)
    elif season is not None and week is not None:
        games = _fetch_slate(season, week)
    else:
        raise ValueError(
            "no games to tick: pass --games-json, or --slate SEASON:WEEK, "
            "or both --season and --week")

    if not games:
        raise ValueError("the slate is empty; refusing to report a clean week "
                         "having looked at nothing")
    return games


def _fetch_slate(season: int, week: int) -> list[dict]:
    from ..data import schedules as schedules_module

    frame = schedules_module.fetch_upcoming_games(season=season, week=week)
    games = frame.to_dict("records")
    # nflverse's schedule calls the kickoff `gameday`; the tick and the tracking
    # store both expect `commence_time`. Without this rename every game reads as
    # post-kickoff (unparseable -> skipped) and the tick silently does nothing.
    for game in games:
        game.setdefault("commence_time", game.get("gameday"))
    return games


def _player_index(path: str | None) -> list[dict] | None:
    """The nflverse player list, or None when the caller has not supplied one.

    None means "no join", and the Odds API returns names rather than ids, so a
    tick without this cannot write a row. It is reported rather than guessed.
    """
    if not path:
        return None
    import json

    with open(path) as handle:
        players = json.load(handle)
    if not isinstance(players, list) or not players:
        raise ValueError(f"{path}: expected a non-empty list of players")
    return players


def _read_parquet(path: str):
    import pandas as pd

    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> int:
    """One forward-test tick.

    Refuses to run without an API key: no key means no request, no snapshot and
    no credit spent, which is the correct outcome rather than an error to route
    around.
    """
    import argparse

    from requests import RequestException

    from ..config import ODDS_API_KEY
    from .forward_report import write_weekly_report

    parser = argparse.ArgumentParser(
        description="Snapshot a slate and log edge-qualifying picks.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--games-json", default=None,
                        help="games to tick; omit to use --season/--week from the schedules")
    parser.add_argument("--slate", default=None,
                        help="fetch the slate for SEASON:WEEK, e.g. 2026:5")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--edge-gate", type=float, default=EDGE_GATE)
    parser.add_argument("--report-dir", default=None,
                        help="write the weekly markdown report here")
    parser.add_argument("--players-path", default=None,
                        help="JSON list of {player_id, player_name, team} used to "
                             "join book prop names to nflverse ids")
    parser.add_argument("--feature-frame", default=None,
                        help="parquet of the assembled feature frame (required)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what a tick would cost and record; spend nothing")
    args = parser.parse_args(argv)

    if not ODDS_API_KEY:
        print("error: no Odds API key. Set ODDS_API_KEY (or SPORTSBOOK_API_KEY, "
              "which config accepts as an alias) in the environment or a .env "
              "file. No request was made and no credit was spent.")
        return 2

    games = _load_games(args.games_json, slate=args.slate, season=args.season,
                        week=args.week)
    live = [g for g in games if not _is_post_kickoff(g.get("commence_time", ""))]
    cost = len(live)  # one props call per game; the credit probe is free
    print(f"{len(games)} games, {len(live)} pre-kickoff, ~{cost} credits estimated")

    if args.dry_run:
        print("--dry-run: no request made, nothing snapshotted.")
        return 0

    try:
        predictors = predictor_for(args.models_dir)
    except FileNotFoundError as error:
        print(f"error: {error}")
        return 2
    print(f"loaded markets: {sorted(predictors)}")

    # (A missing key is guarded above, before any request: without that check the
    # credit probe 401s, `credits_sufficient` reports False, and the tick would
    # log "budget exhausted" -- a credential problem reported as a money problem,
    # which is the most misleading way round.)

    if not args.feature_frame:
        print("error: --feature-frame is required. The tick consumes the assembled "
              "feature frame and does not build it: producing that frame is "
              "scripts/train_quantile_props.py's job, and src/ must not import "
              "from scripts/ (scripts/ is not installed in the Docker image).\n"
              "  python scripts/train_quantile_props.py --seasons 2017-2026 --out-dir ...\n"
              "  # then point --feature-frame at the frame it wrote")
        return 2
    try:
        players = _player_index(args.players_path)
    except (OSError, ValueError) as error:
        print(f"error: cannot read --players-path {args.players_path}: {error}")
        return 2
    try:
        feature_frame = _read_parquet(args.feature_frame)
    except (OSError, ValueError) as error:
        print(f"error: cannot read --feature-frame {args.feature_frame}: {error}")
        return 2

    # Fetched outside the tick so a transport failure is reported like every
    # other misconfiguration (exit 2, one line) instead of a traceback from
    # `raise_for_status()` deep inside the request.
    try:
        event_index = fetch_event_index()
    except RequestException as error:
        print(f"error: cannot fetch the Odds event list: {error}")
        return 2

    try:
        result = run_forward_tick(games=games, market_quantiles=predictors,
                                  edge_gate=args.edge_gate,
                                  players=players, feature_frame=feature_frame,
                                  event_index=event_index)
    except (MissingPlayerIndex, MissingEventIndex) as error:
        print(f"error: {error}")
        return 2
    for key in ("games", "games_skipped_post_kickoff", "props_snapshotted",
                "picks_logged", "degenerate_rows", "no_props_coverage"):
        print(f"  {key}: {result[key]}")

    if args.report_dir and args.season and args.week:
        path = write_weekly_report(season=args.season, week=args.week, out_dir=args.report_dir)
        print(f"report: {path}")
    return 0

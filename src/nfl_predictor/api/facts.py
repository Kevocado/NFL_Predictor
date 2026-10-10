"""facts.py — the read-only /facts bundle the match explainer consumes.

Read-only and suggest-only: this router never writes, never trains and
never places anything. It assembles the spec's facts contract from data
this API already has, and in PUBLIC_MODE it reads the precomputed public
snapshot only — no live model is ever computed there.

Honesty rules this module exists to enforce:

* a pick for a game that has started comes only from the pre-start stored
  snapshot, never from today's model;
* no stored pick means ``pick`` is null and ``pick_timing`` is ``none``;
* a snapshot taken at/after kickoff is ``rebuilt`` and is never counted;
* ``result.pick_won`` appears only for a ``pre_kickoff`` pick;
* spread/total markets are skipped for a started game unless they come
  from the pre-start snapshot.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..config import PUBLIC_MODE
from ..tools.duel_lift import load_lift_results
from ..tracking import store
from . import routes

router = APIRouter()
logger = logging.getLogger(__name__)

#: The position's headline yardage line, used to rank props for the panel.
_YARDS_BY_POSITION = {
    "QB": "passing_yards",
    "RB": "rushing_yards",
}
_YARD_KEYS = ("passing_yards", "rushing_yards", "receiving_yards")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _public_snapshot() -> dict:
    return routes._public_snapshot()


# --- small helpers ------------------------------------------------------

def _parse_game_id(game_id: str) -> tuple[int, int] | None:
    """'2026_05_KC_BAL' -> (2026, 5). Season and week live in the id, so a
    facts lookup never has to guess which week to search."""
    parts = game_id.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _as_utc(value: Any) -> pd.Timestamp | None:
    """The schedule stores gameday as a zoneless UTC timestamp."""
    if value in (None, ""):
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _iso_utc(value: Any) -> str:
    stamp = _as_utc(value)
    return "" if stamp is None else stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(value: Any) -> float | None:
    """NaN/None/blank -> None, so 'skip this market' is a real state."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _spread_line(home_team: str, nflverse_spread_line: Any) -> str | None:
    """nflverse's spread_line is the home team's expected margin, so a
    positive value means home is favoured. The site (and every book) words
    a favourite as giving points, hence the negation."""
    line = _num(nflverse_spread_line)
    if line is None:
        return None
    return f"{home_team} {-line:+.1f}"


def _pick(home_team: str, away_team: str, home_prob: Any, away_prob: Any) -> dict | None:
    home = _num(home_prob)
    away = _num(away_prob)
    if home is None or away is None:
        return None
    # Ties go to the home team, matching margin_to_probabilities' >=.
    if home >= away:
        return {"label": home_team, "prob": home}
    return {"label": away_team, "prob": away}


def _projected_yards(prop: dict) -> float:
    key = _YARDS_BY_POSITION.get(str(prop.get("position") or "").upper(), "receiving_yards")
    value = _num(prop.get(key))
    if value is None:
        present = [v for v in (_num(prop.get(k)) for k in _YARD_KEYS) if v is not None]
        return max(present) if present else 0.0
    return value


def _players(props: list[dict], teams: set[str]) -> list[dict]:
    """The top three props for these two teams by projected yards."""
    ranked = [p for p in props if p.get("recent_team") in teams]
    ranked.sort(key=_projected_yards, reverse=True)
    out = []
    for prop in ranked[:3]:
        yards = _projected_yards(prop)
        key = _YARDS_BY_POSITION.get(str(prop.get("position") or "").upper(), "receiving_yards")
        out.append({
            "name": prop.get("player_name"),
            "team": prop.get("recent_team"),
            "projection": f"{yards:.0f} {str(key).replace('_yards', '')} yds",
        })
    return out


#: role -> (stat key on the prop row, stat label). Top passer is the best passing_yards row,
#: top rusher the best rushing_yards row (any position), top receiver the best receiving_yards.
_PLAYER_ROLES = (
    ("top_rusher", "rushing_yards", "rushing yards"),
    ("top_passer", "passing_yards", "passing yards"),
    ("top_receiver", "receiving_yards", "receiving yards"),
)


def _player_context(props: list[dict], teams: list[str]) -> list[dict]:
    """Each team's predicted top rusher/passer/receiver, for the explainer's context only.

    Reads the same prop rows as `_players`. Listed-out players are already removed upstream
    (`_get_player_props_live` drops them; the snapshot carries them separately in
    `player_props_out`), so every row here is an available player. Pure: no I/O.
    """
    out: list[dict] = []
    for team in teams:
        for role, key, stat in _PLAYER_ROLES:
            rows = [(v, p) for p in props if p.get("recent_team") == team and (v := _num(p.get(key))) is not None and v > 0]
            if rows:
                value, prop = max(rows, key=lambda r: r[0])
                out.append({"team": team, "name": prop.get("player_name"), "role": role, "stat": stat, "value": round(value)})
    return out


def _record() -> dict | None:
    """Pre-kickoff-only accuracy, and the block is LABELLED that way, so it has to read the
    pre-kickoff figure.

    This call site moved with the 2026-10-01 reversal. `get_track_record()["games"]` used to be
    pre-kickoff by construction, so reading the headline was correct; the headline now counts
    every recorded pick, and reading it here would print post-kickoff picks under a
    "Picks made before kickoff" label. `pre_kickoff` is the subset whose own timestamps prove the
    same thing, so the label and the numbers are the same claim.
    """
    games = (store.get_track_record() or {}).get("games") or {}
    pre_kickoff = games.get("pre_kickoff") or {}
    settled = int(pre_kickoff.get("n_resolved") or 0)
    if settled <= 0:
        return None
    pct = _num(pre_kickoff.get("pct_moneyline_correct"))
    return {
        "label": "Picks made before kickoff",
        "hits": None if pct is None else int(round(pct * settled)),
        "settled": settled,
    }


# --- data access --------------------------------------------------------

def _snapshot_game(game_id: str) -> tuple[int, int, dict] | None:
    snap = _public_snapshot()
    season = snap.get("season")
    for week_key, week in (snap.get("weeks") or {}).items():
        for game in week.get("games") or []:
            if game.get("game_id") == game_id:
                return int(game.get("season") or season), int(week_key), game
    return None


def _snapshot_week(season: int, week: int, game_id: str) -> dict | None:
    snap = _public_snapshot()
    if snap.get("season") != season:
        return None
    return (snap.get("weeks") or {}).get(str(week))


def _load_game(game_id: str) -> tuple[int, int, dict]:
    """The game row, from the snapshot in public mode and the schedule live."""
    parsed = _parse_game_id(game_id)
    if PUBLIC_MODE:
        found = _snapshot_game(game_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
        return found
    if parsed is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    season, week = parsed
    games = routes.schedules.fetch_week_games(season, week)
    matches = games[games["game_id"] == game_id] if not games.empty else games
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    return season, week, matches.iloc[0].to_dict()


def _snapshot_prediction(season: int, week: int, game_id: str) -> dict | None:
    """The prediction baked into the public snapshot. Only for games that
    haven't started: the snapshot rebuilds recent weeks after kickoff."""
    snap = _snapshot_week(season, week, game_id) or {}
    return (snap.get("predictions") or {}).get(game_id)


def _current_prediction(season: int, week: int, game_id: str) -> dict | None:
    """Today's model read, for a game that has not started yet. Public mode
    reads the snapshot, which is the precomputed equivalent; neither path
    may train or recompute models."""
    if PUBLIC_MODE:
        return _snapshot_prediction(season, week, game_id)
    return routes.get_game_prediction(season, week, game_id)


def _week_row(season: int, week: int, game_id: str) -> dict | None:
    """The stored pick row for this game: the single authority on whether a
    pick was snapshotted before kickoff, and whether it was rebuilt."""
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week, game_id) or {}
        games = snap.get("games") or []
    else:
        games = routes.get_games(season, week)
    if not games:
        return None
    frame = pd.DataFrame(games)
    frame = frame[frame["game_id"] == game_id]
    if frame.empty:
        return None
    rows = routes.store.get_predictions_for_week(season, week, frame)
    for row in rows or []:
        if row.get("game_id") == game_id:
            return row
    return None


def _props(season: int, week: int) -> tuple[list[dict], bool]:
    """(rows, unavailable) for this week.

    Non-raising on purpose. This is a second reader of the same data as
    `routes.get_player_props`, and when that route started answering 503 for an
    unavailable week this function propagated it -- with no handler at the call
    site, which took down the whole `/facts/{game_id}` bundle: pick, markets,
    drivers, context and record all died because a props panel could not be
    filled. Before that it degraded to `"players": []`. A panel is allowed to be
    empty; it is not allowed to take the page with it.

    The PUBLIC_MODE branch calls `routes.snapshot_props_unavailable` rather than
    reimplementing the rule. While the two were separate, the props page said
    "could not be loaded" for week 3 while this panel said nothing, for the same
    week -- the same reader looking at two panels, one loud and one silent. The
    `unavailable` flag is what lets the panel say something true instead of
    implying there are simply no players.
    """
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week, "") or {}
        if routes.snapshot_props_unavailable(snap) is not None:
            logger.warning("facts: player props unavailable for season=%s week=%s", season, week)
            return [], True
        return snap.get("player_props") or [], False
    try:
        return routes.get_player_props(season, week), False
    except HTTPException as exc:
        # A 503 from the props route is the expected shape of "unavailable". A
        # 404 would mean the route itself is gone, which is not something to
        # absorb quietly -- re-raise so it is not mistaken for no players.
        if exc.status_code != 503:
            raise
        logger.warning("facts: player props unavailable for season=%s week=%s: %s", season, week, exc.detail)
        return [], True


# --- bundle assembly ----------------------------------------------------

def _status(game: dict, now: datetime) -> str:
    home_score = _num(game.get("home_score"))
    away_score = _num(game.get("away_score"))
    if home_score is not None and away_score is not None:
        return "final"
    gameday = _as_utc(game.get("gameday"))
    if gameday is not None and gameday <= pd.Timestamp(now):
        return "live"
    return "upcoming"


def _markets(game: dict, prediction: dict | None, moneyline_from: dict | None = None) -> list[dict]:
    """The markets, from `prediction` — except the moneyline, which may come from
    `moneyline_from`.

    Why the exception: the bundle's `pick` is deliberately the number that was
    snapshotted before kickoff, because that is the record which will be judged.
    For an UPCOMING game the snapshot also holds today's model, recomputed, and
    the two genuinely differ. Emitting the moneyline from the recomputation meant
    the bundle stated the same pick twice at two probabilities:

        pick    : {'label': 'BAL', 'prob': 0.62}
        markets : [{'market': 'moneyline', 'model': {'BAL': 0.55, 'KC': 0.45}}]

    which is not cosmetic. The v2 panel computes its confidence band from
    `pick.prob` and draws its moneyline tile and split bar from the market, so it
    would have shown 55% beside a band derived from 62%.

    The spread and the total are left on `prediction`, and that is a real
    trade-off rather than a proof: they are different claims with their own
    numbers, the row carries none, and nothing in the bundle dates them. A
    future decision could take the whole market set from the row and lose the
    spread and total for every snapshotted pick; that is a product call, and it
    is not this bug.
    """
    if not prediction:
        return []
    home_team = game["home_team"]
    away_team = game["away_team"]
    out: list[dict] = []

    source = moneyline_from if moneyline_from is not None else prediction
    home_prob = _num(source.get("home_win_prob"))
    away_prob = _num(source.get("away_win_prob"))
    if home_prob is not None and away_prob is not None:
        out.append({
            "market": "moneyline",
            "model": {home_team: home_prob, away_team: away_prob},
        })

    margin = _num(prediction.get("predicted_margin"))
    line = _spread_line(home_team, game.get("spread_line"))
    if margin is not None and line is not None:
        cover = _num(prediction.get("home_cover_prob"))
        market = {"market": "spread", "model_margin": margin, "line": line}
        if cover is not None:
            market["model_cover_prob"] = cover
        out.append(market)

    total = _num(prediction.get("predicted_total"))
    total_line = _num(game.get("total_line"))
    if total is not None and total_line is not None:
        over = _num(prediction.get("over_prob"))
        market = {"market": "total", "model_total": total, "line": total_line}
        if over is not None:
            market["model_over_prob"] = over
        out.append(market)

    return out


def _drivers(game: dict, season: int, home_team: str, away_team: str, live_ok: bool = True) -> list[dict]:
    """Why the model leans the way it does. In public mode only what the
    snapshot actually carries — a rating gap would mean computing live."""
    drivers: list[dict] = []
    rating_diff = None
    home_rest = _num(game.get("home_rest"))
    away_rest = _num(game.get("away_rest"))
    div_game = game.get("div_game")

    if not PUBLIC_MODE and live_ok:
        try:
            history = routes._load_game_history(season)
            history = history[history["game_id"] != game["game_id"]]
            row = routes.feature_build.build_features_for_game(home_team, away_team, history, gameday=game.get("gameday"))
            rating_diff = _num(row.get("rating_diff"))
            home_rest = _num(row.get("home_rest_days"))
            away_rest = _num(row.get("away_rest_days"))
            div_game = row.get("div_game")
        except Exception:
            logger.info("driver features unavailable for game_id=%s", game.get("game_id"))

    if rating_diff is not None:
        drivers.append({
            "name": "Rating gap",
            "value": f"{rating_diff:+.1f} pts",
            "direction": home_team if rating_diff >= 0 else away_team,
        })
    if home_rest is not None and away_rest is not None:
        drivers.append({
            "name": "Rest",
            "value": f"{home_rest:.0f} v {away_rest:.0f} days",
            "direction": home_team if home_rest >= away_rest else away_team,
        })
    drivers.append({"name": "Home field", "value": "advantage", "direction": home_team})
    if div_game:
        drivers.append({
            "name": "Divisional",
            "value": "rivalry game",
            "direction": "both",
        })
    return drivers


def _context(game: dict, home_rest: Any, away_rest: Any, matchups: list[dict] | None = None,
             player_context: list[dict] | None = None) -> dict:
    context: dict[str, Any] = {}
    temp = _num(game.get("temp"))
    wind = _num(game.get("wind"))
    if temp is not None or wind is not None:
        parts = []
        if temp is not None:
            parts.append(f"{temp:.0f}F")
        if wind is not None:
            parts.append(f"{wind:.0f} mph wind")
        context["weather"] = ", ".join(parts)
    roof = game.get("roof")
    if roof:
        context["roof"] = str(roof)
    if home_rest is not None and away_rest is not None:
        context["rest"] = f"Rest {home_rest:.0f} v {away_rest:.0f} days"
    # AI plan Task 3: the offence-versus-defence duels, computed in `get_facts`
    # (which has the season, the pick and the live/non-public gate) and handed
    # in here so the context builder never reaches for a data source itself.
    # Neutral until Task 10: every row travels with whatever toward_pick the
    # caller computed (None when there is no pick), never a label.
    if matchups:
        context["matchups"] = matchups
    if player_context:
        context["player_context"] = player_context
    # Injuries deliberately absent: the cached nflverse report goes stale,
    # and the explainer gets injuries from ESPN news instead.
    return context


def _stored_matchup_rows(season: int, week: int, game_id: str, pick_side: str | None) -> list[dict]:
    """The snapshot's stored duels for this game as facts rows (fail closed on direction)."""
    from ..signals.matchups import rows_to_duels, to_context
    snap = _snapshot_week(season, week, game_id) or {}
    rows = (snap.get("matchups") or {}).get(game_id) or []
    return to_context(rows_to_duels(rows), pick_side, lift_gate=load_lift_results())


def _matchup_rows(home: str, away: str, games_df: pd.DataFrame, efficiency: pd.DataFrame,
                  as_of, season: int, pick_side: str | None) -> list[dict]:
    """The real offence-versus-defence duel rows for one game, or [] when the
    efficiency data cannot back them. `games_df` carries per-game gameday/season
    (the same schedule frame the aux efficiency was built from); efficiency
    ranks come from play-by-play known before `as_of`, never after it.
    """
    from ..signals.matchups import load_history_gaps, matchups_for_game, to_context
    if efficiency is None or len(efficiency) == 0:
        return []
    duels = matchups_for_game(home, away, games_df, efficiency, as_of, season,
                              history_gaps=load_history_gaps())
    # Fail-closed Task 10 gate: only duel types the residual-lift runs have
    # proven (data/duel_lift.json) may claim a direction. An absent file --
    # the gate has never run -- proves nothing, so every row is neutral.
    return to_context(duels, pick_side, lift_gate=load_lift_results())


def _result(game: dict, status: str, pick_timing: str, prediction: dict | None) -> dict | None:
    if status != "final":
        return None
    home_score = _num(game.get("home_score"))
    away_score = _num(game.get("away_score"))
    if home_score is None or away_score is None:
        return None
    result: dict[str, Any] = {
        "score": f"{game['home_team']} {home_score:.0f}-{away_score:.0f}"
    }
    if pick_timing != "pre_kickoff" or not prediction:
        return result
    home_prob = _num(prediction.get("home_win_prob"))
    away_prob = _num(prediction.get("away_win_prob"))
    if home_prob is None or away_prob is None:
        return result
    predicted_home = home_prob >= away_prob
    actual_home = home_score > away_score
    result["pick_won"] = bool(predicted_home == actual_home)
    return result


@router.get("/facts/upcoming")
def get_facts_upcoming(hours: int = 72) -> dict:
    """Ids of games starting within the window, for the pre-generation loop."""
    if hours < 0:
        raise HTTPException(status_code=422, detail="hours must be >= 0")
    now = _now()
    cutoff = pd.Timestamp(now) + timedelta(hours=hours)

    games: list[dict] = []
    if PUBLIC_MODE:
        snap = _public_snapshot()
        for week in (snap.get("weeks") or {}).values():
            games.extend(week.get("games") or [])
    else:
        season, week = routes.current_season_and_week()
        games = routes.get_games(season, week) or []

    ids = []
    for game in games:
        gameday = _as_utc(game.get("gameday"))
        if gameday is None or gameday <= pd.Timestamp(now) or gameday > cutoff:
            continue
        game_id = game.get("game_id")
        if game_id and game_id not in ids:
            ids.append(game_id)
    return {"ids": ids}


@router.get("/facts/{game_id}")
def get_facts(game_id: str) -> dict:
    now = _now()
    season, week, game = _load_game(game_id)
    home_team, away_team = game["home_team"], game["away_team"]
    status = _status(game, now)
    started = status in ("live", "final")

    row = _week_row(season, week, game_id)
    row_probs = (
        {"home_win_prob": row.get("home_win_prob"), "away_win_prob": row.get("away_win_prob")}
        if row is not None and row.get("home_win_prob") is not None
        else None
    )

    # THE RULE: the pick number and its timing label always come from the SAME
    # source, so a number is never described by another source's honesty.
    #
    # * STARTED -> the tracking row stored before kickoff, in both modes. The
    #   public snapshot is NOT that: it rebuilds the current and previous week
    #   every few hours, so its prediction for a started game is today's model,
    #   recomputed after kickoff. The row carries no margin/total, so those are
    #   simply absent. No row at all -> no pick, and "none".
    # * UPCOMING -> the row when one exists, so the number shown is the very
    #   record that will be judged and a late-written row still reports itself
    #   as 'rebuilt'. With no row yet (weeks out) the live forecast is used and
    #   labelled 'pre_kickoff': a game that has not been played cannot have
    #   been predicted after it, so that label is always true of the number.
    if started:
        stored = row_probs
    else:
        stored = _current_prediction(season, week, game_id)

    pick_source = row_probs if row_probs is not None else (None if started else stored)
    pick = (
        _pick(home_team, away_team, pick_source.get("home_win_prob"), pick_source.get("away_win_prob"))
        if pick_source is not None
        else None
    )
    if pick is None:
        pick_timing = "none"
    elif pick_source is row_probs:
        pick_timing = "rebuilt" if row.get("rebuilt") else "pre_kickoff"
    else:
        pick_timing = "pre_kickoff"

    # A started game with no stored pick has no honest spread/total to quote.
    # The moneyline comes from `pick_source` so the bundle states the pick once,
    # at one probability -- see _markets' docstring for why that is not the
    # same thing as taking every market from the row.
    markets = [] if (started and stored is None) else _markets(game, stored, moneyline_from=pick_source)

    # Resolved before the return dict so the panel and the flag can never come
    # from different reads. `started` short-circuits: a started game quotes no
    # props at all, and "unavailable" would be a false claim about a thing the
    # bundle deliberately does not show.
    if started:
        props_rows, players_unavailable = [], False
    else:
        props_rows, players_unavailable = _props(season, week)

    # Matchup duels, for an UPCOMING game in live (non-public) mode only:
    # ranking current-season efficiency is a live computation from play-by-play,
    # the same rule the rating-gap driver lives by, so a public bundle (which is
    # snapshot-only by law) never invents a duel it cannot back with data it
    # actually computed. Fail-open: no efficiency, no duels, no crash.
    pick_side = None
    if pick is not None:
        pick_side = "home" if pick["label"] == home_team else "away"
    matchups_rows: list[dict] = []
    if not started and PUBLIC_MODE:
        # Public mode never ranks play-by-play itself: it serves the duels the refresh job STORED in the snapshot,
        # with the pick-relative direction and the residual-lift gate applied now (fail closed: no lift data, no
        # direction).
        try:
            matchups_rows = _stored_matchup_rows(season, week, game_id, pick_side)
        except Exception:
            logger.info("stored matchups unavailable for game_id=%s", game_id)
    elif not started:
        try:
            aux = routes._load_aux_cached(season)
            matchups_rows = _matchup_rows(
                home_team, away_team, routes._load_game_history(season),
                aux.efficiency, game.get("gameday"), season, pick_side,
            )
        except Exception:
            logger.info("matchups unavailable for game_id=%s", game_id)

    return {
        "sport": "nfl",
        "id": game_id,
        "title": f"{away_team} at {home_team}",
        "starts_at": _iso_utc(game.get("gameday")),
        "status": status,
        "pick_timing": pick_timing,
        "pick": pick,
        "markets": markets,
        # Player props and the live rating gap are rebuilt/computed now, so a
        # started game quotes neither; rest and divisional status are fixed.
        "drivers": _drivers(game, season, home_team, away_team, live_ok=not started),
        "context": _context(game, game.get("home_rest"), game.get("away_rest"), matchups=matchups_rows,
                         player_context=_player_context(props_rows, [home_team, away_team])),
        "players": [] if started else _players(props_rows, {home_team, away_team}),
        # True when props exist upstream but could not be produced, so a consumer
        # can say "unavailable" rather than rendering an empty list, which is
        # indistinguishable from a game whose players have no prop markets.
        #
        # **This repo has no reader for it.** The consumer is the match explainer
        # in predictor-hub, which is outside this tree; the local `Facts` contract
        # model in tests/test_facts.py ignores extras, so the contract test does
        # not pin the field either. So the honest statement is: this exists for a
        # consumer this repo does not own, and until that consumer renders it, the
        # only reader-visible change from this work is that a 500 became a 200 --
        # in PUBLIC_MODE a reader still sees `players: []` exactly as before.
        #
        # Flagged rather than invented around: making the panel lie-safe without a
        # consumer would mean either dropping the field (losing the distinction the
        # next consumer needs) or changing `players` to carry a sentinel this repo
        # would then be obliged to keep.
        "players_unavailable": players_unavailable,
        "record": _record(),
        "result": _result(game, status, pick_timing, stored),
    }

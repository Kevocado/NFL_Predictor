"""main.py — FastAPI app entry point.

Run with:
    PYTHONPATH=$(pwd)/src uvicorn nfl_predictor.api.main:app --reload --host 0.0.0.0 --port 8001
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import PUBLIC_MODE, PUBLIC_SNAPSHOT_POLL_SECONDS
from .routes import (
    current_season_and_week,
    refresh_public_snapshot_from_remote,
    router,
    warm_caches,
    background_tracking_tick,
)

logger = logging.getLogger(__name__)

_TRACKING_INTERVAL_SECONDS = 300


async def _run_tracking_tick():
    try:
        season, week = current_season_and_week()
        await asyncio.to_thread(background_tracking_tick, season, week)
    except Exception:
        logger.exception("background_tracking_tick failed")


async def _tracking_loop():
    # Tick once immediately, THEN sleep -- this app scales to zero between
    # requests, so a sleep-first loop risks never running at all if the
    # container doesn't stay warm continuously for a full interval
    # (confirmed live on CFB_Predictor's identical loop: games had long
    # since finished, but nothing had ever reconciled them). Ticking on
    # every cold start instead means every burst of traffic gets at least
    # one real reconciliation pass.
    await _run_tracking_tick()
    while True:
        await asyncio.sleep(_TRACKING_INTERVAL_SECONDS)
        await _run_tracking_tick()


async def _public_snapshot_poll_loop():
    while True:
        await asyncio.to_thread(refresh_public_snapshot_from_remote)
        await asyncio.sleep(PUBLIC_SNAPSHOT_POLL_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The public deployment serves games/predictions/player-props from
    # public_snapshot.py's precomputed file (see routes.py's PUBLIC_MODE
    # branches) instead of computing them per-request. The tracking loop
    # still runs even in PUBLIC_MODE, unlike PL_Predictor's equivalent --
    # unlike PL's full live-serving pipeline, this project's
    # background_tracking_tick is cheap (current week only) and runs on a
    # timer, not per-request, and it's what feeds the track-record/verdict
    # history the frontend shows; skipping it would leave that feature
    # permanently empty on the public deployment. warm_caches is skipped
    # in PUBLIC_MODE since the snapshot already covers what it would warm.
    tasks = [asyncio.create_task(_tracking_loop())]
    if PUBLIC_MODE:
        tasks.append(asyncio.create_task(_public_snapshot_poll_loop()))
    else:
        asyncio.create_task(asyncio.to_thread(warm_caches))
    yield
    for task in tasks:
        task.cancel()


app = FastAPI(title="NFL Predictor API", lifespan=lifespan)

# Allow cross-origin requests from the Sports_Predictor frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "NFL Predictor API",
        "docs": "/docs",
    }
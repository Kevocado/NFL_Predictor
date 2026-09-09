"""main.py — FastAPI app entry point.

Run with:
    PYTHONPATH=$(pwd)/src uvicorn nfl_predictor.api.main:app --reload --host 0.0.0.0 --port 8001
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import current_season_and_week, router, warm_caches, background_tracking_tick

logger = logging.getLogger(__name__)

_TRACKING_INTERVAL_SECONDS = 300


async def _tracking_loop():
    while True:
        await asyncio.sleep(_TRACKING_INTERVAL_SECONDS)
        try:
            season, week = current_season_and_week()
            await asyncio.to_thread(background_tracking_tick, season, week)
        except Exception:
            logger.exception("background_tracking_tick failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    asyncio.create_task(asyncio.to_thread(warm_caches))
    tracking_task = asyncio.create_task(_tracking_loop())
    yield
    tracking_task.cancel()


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
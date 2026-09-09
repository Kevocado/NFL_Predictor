"""main.py — FastAPI app entry point.

Run with:
    PYTHONPATH=$(pwd)/src uvicorn nfl_predictor.api.main:app --reload --host 0.0.0.0 --port 8001
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router, warm_caches, background_tracking_tick

logger = logging.getLogger(__name__)

_TRACKING_INTERVAL_SECONDS = 300


def _current_season_and_week() -> tuple[int, int]:
    """Calendar-based estimate for current NFL season and week."""
    today = date.today()
    season = today.year if today.month >= 3 else today.year - 1
    week = max(1, min(22, ((today - date(season, 9, 1)).days // 7) + 1))
    return season, week


async def _tracking_loop():
    while True:
        await asyncio.sleep(_TRACKING_INTERVAL_SECONDS)
        try:
            season, week = _current_season_and_week()
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
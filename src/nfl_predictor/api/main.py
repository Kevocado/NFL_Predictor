"""main.py — FastAPI app entrypoint.

MINIMAL STUB for Task 16: this task's own tests import `app` from this
module (`from nfl_predictor.api.main import app`), so a bare app that
mounts `routes.router` is required just to run them standalone. Task 17
is the task that actually owns this file — CORS, static frontend
mounting, startup/shutdown hooks, etc. belong there, not here.
"""

from __future__ import annotations

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="NFL Predictor API")
app.include_router(router)

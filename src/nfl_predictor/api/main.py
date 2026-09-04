"""main.py — FastAPI app entry point.

Run with:
    PYTHONPATH=$(pwd)/src uvicorn nfl_predictor.api.main:app --reload --host 0.0.0.0 --port 8001
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import FRONTEND_DIST_DIR
from .routes import router

app = FastAPI(title="NFL Predictor API")

# Same wide-open CORS as PL_Predictor/F1_Predictor: this server is only ever
# reached over a private network or this project's own public read-only
# deployment, never with a login to protect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
else:

    @app.get("/")
    def root():
        return {"status": "ok", "docs": "/docs"}

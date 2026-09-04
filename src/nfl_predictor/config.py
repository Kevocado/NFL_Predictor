"""config.py — paths, env loading, and shared constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"

PUBLIC_MODE = os.getenv("PUBLIC_MODE", "false").lower() == "true"

SCHEDULES_CACHE_DIR = CACHE_DIR / "schedules"
PLAYER_STATS_CACHE_DIR = CACHE_DIR / "player_stats"
INJURIES_CACHE_DIR = CACHE_DIR / "injuries"
ODDS_CACHE_DIR = CACHE_DIR / "odds"

CURRENT_SEASON = 2026  # bump each new NFL league year (typically March)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_SPORT_KEY = "americanfootball_nfl"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# ESPN's unofficial site API — same free/keyless role data/espn.py plays in
# PL_Predictor, here supplying closer-to-kickoff inactive/injury status.
ESPN_NFL_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

TRACKING_DB_PATH = DATA_DIR / "tracking.db"

for _d in (
    DATA_DIR,
    CACHE_DIR,
    MODELS_DIR,
    SCHEDULES_CACHE_DIR,
    PLAYER_STATS_CACHE_DIR,
    INJURIES_CACHE_DIR,
    ODDS_CACHE_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

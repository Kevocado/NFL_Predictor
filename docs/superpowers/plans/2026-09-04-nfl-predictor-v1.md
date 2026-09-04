# NFL Predictor v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working NFL prediction dashboard — pre-kickoff moneyline/spread/total probabilities for every game and anytime-TD/passing/rushing/receiving-yardage player props — served through a FastAPI backend and React frontend, with every prediction snapshotted before kickoff and honestly reconciled after.

**Architecture:** Third sibling project to `PL_Predictor` and `F1_Predictor`, same skeleton (`src/nfl_predictor/{data,features,models,evaluate,tracking,odds,api}`, React+TS+Vite frontend, SQLite snapshot-then-reconcile tracking). Game-outcome models are built as a candidate race (Elo baseline, ridge margin regression, XGBoost margin regression) walk-forward validated on held-out log-loss, mirroring how `PL_Predictor` races Dixon-Coles/Bivariate-Poisson/XGBoost and `F1_Predictor` races Elo/XGBoost-ranker — whichever wins is served. Player props are a separate classifier (anytime-TD) plus regressors (passing/rushing/receiving yards).

**Tech Stack:** Python 3.10+, `nfl_data_py` (nflverse), FastAPI, uvicorn, pandas, numpy, scipy, scikit-learn, xgboost, requests, python-dotenv, SQLite; React 19 + TypeScript + Vite for the frontend.

**Spec:** `/Users/sigey/Documents/Projects/NFL_Predictor/docs/superpowers/specs/2026-09-04-nfl-predictor-design.md`

## Global Constraints

- No new paid API keys — reuse the existing Odds API key already configured for `PL_Predictor` (same account/env var name `ODDS_API_KEY`), sport key `americanfootball_nfl`.
- All historical/current-season data comes from `nfl_data_py` (nflverse) plus ESPN's unofficial scoreboard API for closer-to-kickoff status — both free and keyless.
- Every feature-construction consumer (training, backtest, live serving) goes through one `features/build.py` entry point — no duplicated feature logic.
- Every game/player prediction is snapshotted to SQLite before kickoff and never overwritten — reconciliation only fills in outcome columns on already-existing rows.
- Value-bet detection compares model probability to Shin-de-vigged market probability and surfaces at most one recommended market per game — never a parlay.
- Candidate models are selected by walk-forward held-out log-loss, never a single random train/test split (temporal leakage).
- `pip install -e ".[dev]"` on this machine silently skips `.pth`-based editable installs — use `export PYTHONPATH=$(pwd)/src` in every shell that runs `python`/`uvicorn`/`pytest` for this project (see `[[feedback_editable_install_pth]]`).
- Editing files under `/Users/sigey/Documents/Projects/predictor-hub` requires switching the shell's working directory there — it's a separate git repo (`github.com/Kevocado/predictor-hub`) from `NFL_Predictor`. Commit there, but do not push without the user's explicit go-ahead (visible/shared-state action).

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/nfl_predictor/__init__.py`
- Create: `src/nfl_predictor/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.PROJECT_ROOT`, `config.DATA_DIR`, `config.CACHE_DIR`, `config.MODELS_DIR`, `config.FRONTEND_DIST_DIR`, `config.TRACKING_DB_PATH`, `config.SCHEDULES_CACHE_DIR`, `config.PLAYER_STATS_CACHE_DIR`, `config.INJURIES_CACHE_DIR`, `config.ODDS_CACHE_DIR`, `config.CURRENT_SEASON` (int), `config.ODDS_API_KEY`, `config.ODDS_API_SPORT_KEY`, `config.ODDS_API_BASE_URL`, `config.ESPN_NFL_SCOREBOARD_URL`, `config.PUBLIC_MODE` (bool).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "nfl_predictor"
version = "0.1.0"
description = "NFL game outcome and player prop predictor"
requires-python = ">=3.10"
dependencies = [
    "nfl_data_py>=0.3.2",
    "pandas",
    "numpy",
    "scipy",
    "xgboost",
    "scikit-learn",
    "requests",
    "python-dotenv",
    "fastapi",
    "uvicorn[standard]",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-mock"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write `.env.example` and `.gitignore`**

```bash
# .env.example
ODDS_API_KEY=
PUBLIC_MODE=false
```

```gitignore
.venv/
__pycache__/
*.egg-info/
data/cache/
data/tracking.db
data/tracking.db-shm
data/tracking.db-wal
models/*.json
models/*.pkl
.env
frontend/node_modules/
frontend/dist/
.pytest_cache/
```

- [ ] **Step 3: Write the failing test for `config.py`**

```python
# tests/test_config.py
from pathlib import Path

from nfl_predictor import config


def test_paths_are_absolute_and_created():
    assert config.PROJECT_ROOT.is_absolute()
    assert config.CACHE_DIR.is_dir()
    assert config.MODELS_DIR.is_dir()
    assert config.SCHEDULES_CACHE_DIR.is_dir()
    assert config.PLAYER_STATS_CACHE_DIR.is_dir()
    assert config.INJURIES_CACHE_DIR.is_dir()
    assert config.ODDS_CACHE_DIR.is_dir()


def test_current_season_is_reasonable():
    assert 2020 <= config.CURRENT_SEASON <= 2100


def test_odds_api_sport_key():
    assert config.ODDS_API_SPORT_KEY == "americanfootball_nfl"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor'`

- [ ] **Step 5: Write `src/nfl_predictor/config.py`**

```python
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
```

- [ ] **Step 6: Create the package `__init__.py` files**

```python
# src/nfl_predictor/__init__.py
```

- [ ] **Step 7: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/nfl_predictor/__init__.py src/nfl_predictor/config.py tests/test_config.py
git commit -m "feat: project scaffolding and config"
```

---

## Task 2: Data — schedules and game results

**Files:**
- Create: `src/nfl_predictor/data/__init__.py`
- Create: `src/nfl_predictor/data/schedules.py`
- Test: `tests/test_schedules.py`

**Interfaces:**
- Consumes: `config.SCHEDULES_CACHE_DIR`, `config.CURRENT_SEASON`.
- Produces: `schedules.fetch_schedules(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame`, `schedules.default_completed_seasons(n: int = 8) -> list[int]`, `schedules.load_training_data(seasons: list[int]) -> pd.DataFrame`, `schedules.fetch_current_season_partial() -> pd.DataFrame`, `schedules.fetch_upcoming_games(season: int, week: int) -> pd.DataFrame`. Returned DataFrame columns: `game_id, season, week, gameday (datetime), home_team, away_team, home_score, away_score, home_rest, away_rest, div_game, roof, surface, temp, wind, spread_line, total_line`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schedules.py
import pandas as pd
import pytest

from nfl_predictor.data import schedules


def _raw_schedule_frame():
    return pd.DataFrame(
        [
            {
                "game_id": "2025_01_KC_BAL", "season": 2025, "week": 1,
                "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
                "home_score": 27, "away_score": 20, "home_rest": 7, "away_rest": 7,
                "div_game": 0, "roof": "outdoors", "surface": "grass",
                "temp": 72.0, "wind": 5.0, "spread_line": -2.5, "total_line": 46.5,
            },
            {
                "game_id": "2025_01_PHI_GB", "season": 2025, "week": 1,
                "gameday": "2025-09-05", "home_team": "GB", "away_team": "PHI",
                "home_score": None, "away_score": None, "home_rest": 7, "away_rest": 7,
                "div_game": 0, "roof": "outdoors", "surface": "grass",
                "temp": None, "wind": None, "spread_line": 1.5, "total_line": 45.0,
            },
        ]
    )


def test_fetch_schedules_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    calls = []

    def fake_import(years):
        calls.append(list(years))
        return _raw_schedule_frame()

    monkeypatch.setattr(schedules, "_import_schedules", fake_import)

    first = schedules.fetch_schedules([2025])
    second = schedules.fetch_schedules([2025])

    assert len(calls) == 1  # second call hit the cache, not the network
    assert len(first) == 2
    assert list(first.columns).__contains__("home_team")
    assert second.equals(first)


def test_load_training_data_drops_unplayed_games(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(schedules, "_import_schedules", lambda years: _raw_schedule_frame())

    df = schedules.load_training_data([2025])

    assert len(df) == 1
    assert df.iloc[0]["game_id"] == "2025_01_KC_BAL"


def test_default_completed_seasons_excludes_current_season(monkeypatch):
    monkeypatch.setattr(schedules, "CURRENT_SEASON", 2026)
    seasons = schedules.default_completed_seasons(n=3)
    assert seasons == [2023, 2024, 2025]


def test_fetch_upcoming_games_filters_season_week(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(schedules, "_import_schedules", lambda years: _raw_schedule_frame())

    upcoming = schedules.fetch_upcoming_games(2025, 1)

    assert list(upcoming["game_id"]) == ["2025_01_PHI_GB"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_schedules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.data'`

- [ ] **Step 3: Write `src/nfl_predictor/data/__init__.py` and `schedules.py`**

```python
# src/nfl_predictor/data/__init__.py
```

```python
"""schedules.py — game schedule and results, cache-or-fetch from nfl_data_py.

One parquet file per season under SCHEDULES_CACHE_DIR, mirroring the
per-season cache-or-fetch pattern PL_Predictor's data/football_data.py and
F1_Predictor's data/jolpica.py both use — a season's results never change
once played, so caching per-season is safe and avoids re-hitting nfl_data_py
on every call.
"""

from __future__ import annotations

import pandas as pd

from ..config import CURRENT_SEASON, SCHEDULES_CACHE_DIR

KEEP_COLUMNS = [
    "game_id", "season", "week", "gameday", "home_team", "away_team",
    "home_score", "away_score", "home_rest", "away_rest", "div_game",
    "roof", "surface", "temp", "wind", "spread_line", "total_line",
]


def _import_schedules(years: list[int]) -> pd.DataFrame:
    """Thin wrapper around nfl_data_py so tests can monkeypatch just this
    one function rather than the whole nfl_data_py import machinery."""
    import nfl_data_py as nfl

    return nfl.import_schedules(years)


def _season_cache_path(season: int) -> "Path":
    return SCHEDULES_CACHE_DIR / f"{season}.parquet"


def fetch_schedules(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """One row per game across every requested season. Seasons already
    cached on disk are read from cache; anything missing (or force_refresh)
    is fetched from nfl_data_py in one batched call, then split back out to
    per-season cache files."""
    frames = []
    missing = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(season)

    if missing:
        fetched = _import_schedules(missing)[KEEP_COLUMNS].copy()
        fetched["gameday"] = pd.to_datetime(fetched["gameday"])
        for season in missing:
            season_df = fetched[fetched["season"] == season].reset_index(drop=True)
            season_df.to_parquet(_season_cache_path(season))
            frames.append(season_df)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def default_completed_seasons(n: int = 8) -> list[int]:
    return list(range(CURRENT_SEASON - n, CURRENT_SEASON))


def load_training_data(seasons: list[int]) -> pd.DataFrame:
    """Only games with a final score — excludes future/postponed games from
    the same fetch_schedules() call."""
    df = fetch_schedules(seasons)
    return df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)


def fetch_current_season_partial() -> pd.DataFrame:
    """Completed games so far in CURRENT_SEASON, refetched every call (no
    per-season cache for the still-in-progress season, since its cache file
    would go stale after every week's games)."""
    df = fetch_schedules([CURRENT_SEASON], force_refresh=True)
    return df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)


def fetch_upcoming_games(season: int, week: int) -> pd.DataFrame:
    """Games in a given season/week that haven't been played yet."""
    df = fetch_schedules([season], force_refresh=(season == CURRENT_SEASON))
    week_df = df[df["week"] == week]
    return week_df[week_df["home_score"].isna()].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_schedules.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/data/__init__.py src/nfl_predictor/data/schedules.py tests/test_schedules.py
git commit -m "feat: schedules data module with cache-or-fetch"
```

---

## Task 3: Data — weekly player stats

**Files:**
- Create: `src/nfl_predictor/data/player_stats.py`
- Test: `tests/test_player_stats.py`

**Interfaces:**
- Consumes: `config.PLAYER_STATS_CACHE_DIR`, `config.CURRENT_SEASON`.
- Produces: `player_stats.fetch_weekly_player_stats(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame` with columns `player_id, player_name, position, recent_team, season, week, passing_yards, passing_tds, rushing_yards, rushing_tds, receiving_yards, receiving_tds, receptions, targets, carries`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_stats.py
import pandas as pd

from nfl_predictor.data import player_stats


def _raw_weekly_frame():
    return pd.DataFrame(
        [
            {
                "player_id": "00-0033873", "player_name": "P. Mahomes", "position": "QB",
                "recent_team": "KC", "season": 2025, "week": 1,
                "passing_yards": 291, "passing_tds": 2, "rushing_yards": 12, "rushing_tds": 0,
                "receiving_yards": 0, "receiving_tds": 0, "receptions": 0, "targets": 0, "carries": 3,
            }
        ]
    )


def test_fetch_weekly_player_stats_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    calls = []

    def fake_import(years, columns=None):
        calls.append(list(years))
        return _raw_weekly_frame()

    monkeypatch.setattr(player_stats, "_import_weekly_data", fake_import)

    first = player_stats.fetch_weekly_player_stats([2025])
    second = player_stats.fetch_weekly_player_stats([2025])

    assert len(calls) == 1
    assert len(first) == 1
    assert second.equals(first)


def test_fetch_weekly_player_stats_force_refresh_refetches(monkeypatch, tmp_path):
    monkeypatch.setattr(player_stats, "PLAYER_STATS_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        player_stats, "_import_weekly_data",
        lambda years, columns=None: calls.append(years) or _raw_weekly_frame(),
    )

    player_stats.fetch_weekly_player_stats([2025])
    player_stats.fetch_weekly_player_stats([2025], force_refresh=True)

    assert len(calls) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.data.player_stats'`

- [ ] **Step 3: Write `src/nfl_predictor/data/player_stats.py`**

```python
"""player_stats.py — weekly player-level stats, cache-or-fetch from
nfl_data_py. Same per-season parquet caching pattern as data/schedules.py."""

from __future__ import annotations

import pandas as pd

from ..config import PLAYER_STATS_CACHE_DIR

KEEP_COLUMNS = [
    "player_id", "player_name", "position", "recent_team", "season", "week",
    "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions", "targets", "carries",
]


def _import_weekly_data(years: list[int], columns: list[str] | None = None) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_weekly_data(years, columns=columns)


def _season_cache_path(season: int) -> "Path":
    return PLAYER_STATS_CACHE_DIR / f"{season}.parquet"


def fetch_weekly_player_stats(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    frames = []
    missing = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(season)

    if missing:
        fetched = _import_weekly_data(missing, columns=KEEP_COLUMNS)[KEEP_COLUMNS].copy()
        for season in missing:
            season_df = fetched[fetched["season"] == season].reset_index(drop=True)
            season_df.to_parquet(_season_cache_path(season))
            frames.append(season_df)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week"]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_stats.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/data/player_stats.py tests/test_player_stats.py
git commit -m "feat: weekly player stats data module"
```

---

## Task 4: Data — Odds API (NFL h2h/spreads/totals)

**Files:**
- Create: `src/nfl_predictor/data/odds_api.py`
- Test: `tests/test_odds_api.py`

**Interfaces:**
- Consumes: `config.ODDS_API_KEY`, `config.ODDS_API_SPORT_KEY`, `config.ODDS_API_BASE_URL`.
- Produces: `odds_api.fetch_game_odds() -> pd.DataFrame` with columns `event_id, commence_time, home_team, away_team, bookmaker, market, outcome_name, price, point, odds_fetched_at`. Empty DataFrame (not an exception) when `ODDS_API_KEY` is unset or the request fails.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_odds_api.py
import pandas as pd
import pytest

from nfl_predictor.data import odds_api


def _raw_odds_response():
    return [
        {
            "id": "abc123",
            "commence_time": "2025-09-04T20:20:00Z",
            "home_team": "Baltimore Ravens",
            "away_team": "Kansas City Chiefs",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Baltimore Ravens", "price": 1.87},
                                {"name": "Kansas City Chiefs", "price": 1.95},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 1.91, "point": 46.5},
                                {"name": "Under", "price": 1.91, "point": 46.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]


def test_fetch_game_odds_flattens_bookmaker_markets(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "fake-key")
    monkeypatch.setattr(odds_api, "_fetch_raw_odds", lambda: _raw_odds_response())

    df = odds_api.fetch_game_odds()

    assert len(df) == 4  # 2 h2h outcomes + 2 totals outcomes
    assert set(df["market"]) == {"h2h", "totals"}
    assert df.iloc[0]["event_id"] == "abc123"


def test_fetch_game_odds_returns_empty_frame_without_api_key(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", None)

    df = odds_api.fetch_game_odds()

    assert df.empty


def test_fetch_game_odds_returns_empty_frame_on_request_error(monkeypatch):
    monkeypatch.setattr(odds_api, "ODDS_API_KEY", "fake-key")

    def _raise():
        raise RuntimeError("network error")

    monkeypatch.setattr(odds_api, "_fetch_raw_odds", _raise)

    df = odds_api.fetch_game_odds()

    assert df.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_odds_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.data.odds_api'`

- [ ] **Step 3: Write `src/nfl_predictor/data/odds_api.py`**

```python
"""odds_api.py — The Odds API bulk h2h/totals/spreads feed for NFL.

Same account/key already configured for PL_Predictor (data/odds_api.py
there) — only ODDS_API_SPORT_KEY differs. Best-effort: any failure (missing
key, network error, bad response) returns an empty DataFrame rather than
raising, matching every other best-effort data source across these projects.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from ..config import ODDS_API_BASE_URL, ODDS_API_KEY, ODDS_API_SPORT_KEY


def _fetch_raw_odds() -> list[dict]:
    url = f"{ODDS_API_BASE_URL}/{ODDS_API_SPORT_KEY}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_game_odds() -> pd.DataFrame:
    if not ODDS_API_KEY:
        return pd.DataFrame()

    try:
        events = _fetch_raw_odds()
    except Exception:
        return pd.DataFrame()

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for event in events:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": event["id"],
                            "commence_time": event["commence_time"],
                            "home_team": event["home_team"],
                            "away_team": event["away_team"],
                            "bookmaker": bookmaker["key"],
                            "market": market["key"],
                            "outcome_name": outcome["name"],
                            "price": outcome["price"],
                            "point": outcome.get("point"),
                            "odds_fetched_at": fetched_at,
                        }
                    )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_odds_api.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/data/odds_api.py tests/test_odds_api.py
git commit -m "feat: Odds API data module for NFL h2h/spreads/totals"
```

---

## Task 5: Data — injuries

**Files:**
- Create: `src/nfl_predictor/data/injuries.py`
- Test: `tests/test_injuries.py`

**Interfaces:**
- Consumes: `config.INJURIES_CACHE_DIR`.
- Produces: `injuries.fetch_injuries(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame` with columns `season, week, team, player_id, player_name, position, report_status` (`report_status` in `{"Out", "Doubtful", "Questionable", None}`). `injuries.current_status_by_player(injuries_df: pd.DataFrame, season: int, week: int) -> dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_injuries.py
import pandas as pd

from nfl_predictor.data import injuries


def _raw_injury_frame():
    return pd.DataFrame(
        [
            {"season": 2025, "week": 1, "team": "KC", "gsis_id": "00-0033873",
             "full_name": "Patrick Mahomes", "position": "QB", "report_status": "Questionable"},
            {"season": 2025, "week": 1, "team": "KC", "gsis_id": "00-0031234",
             "full_name": "Backup Guy", "position": "RB", "report_status": None},
        ]
    )


def test_fetch_injuries_caches_per_season(monkeypatch, tmp_path):
    monkeypatch.setattr(injuries, "INJURIES_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        injuries, "_import_injuries",
        lambda years: calls.append(years) or _raw_injury_frame(),
    )

    first = injuries.fetch_injuries([2025])
    second = injuries.fetch_injuries([2025])

    assert len(calls) == 1
    assert len(first) == 2
    assert second.equals(first)


def test_current_status_by_player_only_includes_flagged_players(monkeypatch, tmp_path):
    monkeypatch.setattr(injuries, "INJURIES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(injuries, "_import_injuries", lambda years: _raw_injury_frame())

    df = injuries.fetch_injuries([2025])
    status = injuries.current_status_by_player(df, season=2025, week=1)

    assert status == {"00-0033873": "Questionable"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_injuries.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.data.injuries'`

- [ ] **Step 3: Write `src/nfl_predictor/data/injuries.py`**

```python
"""injuries.py — official weekly injury reports, cache-or-fetch from
nfl_data_py. Supplements ESPN's closer-to-kickoff scoreboard for gating
player-prop predictions on real availability."""

from __future__ import annotations

import pandas as pd

from ..config import INJURIES_CACHE_DIR

KEEP_COLUMNS = ["season", "week", "team", "gsis_id", "full_name", "position", "report_status"]


def _import_injuries(years: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_injuries(years)


def _season_cache_path(season: int) -> "Path":
    return INJURIES_CACHE_DIR / f"{season}.parquet"


def fetch_injuries(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    frames = []
    missing = []
    for season in seasons:
        path = _season_cache_path(season)
        if not force_refresh and path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(season)

    if missing:
        fetched = _import_injuries(missing)[KEEP_COLUMNS].copy()
        for season in missing:
            season_df = fetched[fetched["season"] == season].reset_index(drop=True)
            season_df.to_parquet(_season_cache_path(season))
            frames.append(season_df)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def current_status_by_player(injuries_df: pd.DataFrame, season: int, week: int) -> dict[str, str]:
    """gsis_id -> report_status for players actually flagged (Out/Doubtful/
    Questionable) in a given season/week — players with no report_status
    (the common case: healthy, no injury report entry) are omitted rather
    than included with a None value, so callers can treat "in this dict" as
    "has a real status to gate on"."""
    week_df = injuries_df[(injuries_df["season"] == season) & (injuries_df["week"] == week)]
    flagged = week_df[week_df["report_status"].notna()]
    return dict(zip(flagged["gsis_id"], flagged["report_status"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_injuries.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/data/injuries.py tests/test_injuries.py
git commit -m "feat: injuries data module"
```

---

## Task 6: Features — team power ratings (Elo)

**Files:**
- Create: `src/nfl_predictor/features/__init__.py`
- Create: `src/nfl_predictor/features/power_ratings.py`
- Test: `tests/test_power_ratings.py`

**Interfaces:**
- Produces: `power_ratings.compute_pregame_ratings(games_df: pd.DataFrame, k: float = 20.0, home_field: float = 65.0, start_rating: float = 1500.0) -> pd.DataFrame` — returns `games_df` with two new columns `home_pregame_rating`, `away_pregame_rating` (each game's rating *before* that game, chronological, no lookahead). `power_ratings.final_ratings(games_df: pd.DataFrame, **kwargs) -> dict[str, float]` — every team's rating after the last game in `games_df`, for live serving.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_power_ratings.py
import pandas as pd

from nfl_predictor.features import power_ratings


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "season": 2025, "week": 1, "gameday": "2025-09-04",
             "home_team": "BAL", "away_team": "KC", "home_score": 27, "away_score": 20},
            {"game_id": "g2", "season": 2025, "week": 2, "gameday": "2025-09-11",
             "home_team": "KC", "away_team": "BAL", "home_score": 17, "away_score": 24},
        ]
    )


def test_first_game_uses_start_rating_for_both_teams():
    result = power_ratings.compute_pregame_ratings(_games(), start_rating=1500.0)

    first = result.iloc[0]
    assert first["home_pregame_rating"] == 1500.0
    assert first["away_pregame_rating"] == 1500.0


def test_rating_moves_after_a_result():
    result = power_ratings.compute_pregame_ratings(_games(), start_rating=1500.0)

    second = result.iloc[1]
    # KC lost game 1 as the away team, so KC's pregame rating for game 2
    # (now at home) should have dropped below 1500.
    assert second["home_pregame_rating"] < 1500.0
    # BAL won game 1, so BAL's pregame rating for game 2 (now away) should
    # have risen above 1500.
    assert second["away_pregame_rating"] > 1500.0


def test_final_ratings_reflects_every_game():
    ratings = power_ratings.final_ratings(_games(), start_rating=1500.0)

    assert set(ratings) == {"BAL", "KC"}
    # BAL won both meetings on net score margin, should end above start.
    assert ratings["BAL"] > 1500.0
    assert ratings["KC"] < 1500.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_power_ratings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.features'`

- [ ] **Step 3: Write `src/nfl_predictor/features/__init__.py` and `power_ratings.py`**

```python
# src/nfl_predictor/features/__init__.py
```

```python
"""power_ratings.py — Elo-style team power ratings, margin-of-victory
weighted.

Standard NFL Elo shape (à la 538's NFL model): expected score from a
logistic function of the rating gap plus a home-field bonus, update scaled
by both the surprise (actual - expected) and a margin-of-victory
multiplier so a 40-point win moves ratings more than a 3-point win.
Chronological — every game's *pregame* rating only reflects games played
strictly before it, so this is safe to use as a training feature with no
lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START_RATING = 1500.0
DEFAULT_K = 20.0
DEFAULT_HOME_FIELD = 65.0
ELO_SCALE = 400.0


def _expected_home_win_prob(home_rating: float, away_rating: float, home_field: float) -> float:
    diff = (home_rating + home_field) - away_rating
    return 1.0 / (1.0 + 10 ** (-diff / ELO_SCALE))


def _margin_multiplier(margin: float, rating_diff: float) -> float:
    """538's NFL Elo margin-of-victory multiplier: log of the margin, damped
    when the favorite already led the ratings by a lot (an autocorrelation
    correction — a huge win over a much weaker team shouldn't move ratings
    as much as the same margin over an evenly matched one)."""
    return np.log(max(abs(margin), 1) + 1) * (2.2 / ((rating_diff * 0.001) + 2.2))


def compute_pregame_ratings(
    games_df: pd.DataFrame,
    k: float = DEFAULT_K,
    home_field: float = DEFAULT_HOME_FIELD,
    start_rating: float = DEFAULT_START_RATING,
) -> pd.DataFrame:
    games_df = games_df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    home_pregame = []
    away_pregame = []

    for _, game in games_df.iterrows():
        home, away = game["home_team"], game["away_team"]
        home_rating = ratings.get(home, start_rating)
        away_rating = ratings.get(away, start_rating)
        home_pregame.append(home_rating)
        away_pregame.append(away_rating)

        if pd.isna(game["home_score"]) or pd.isna(game["away_score"]):
            continue  # unplayed game: record pregame rating, no update

        margin = game["home_score"] - game["away_score"]
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        expected = _expected_home_win_prob(home_rating, away_rating, home_field)
        multiplier = _margin_multiplier(margin, home_rating - away_rating)
        delta = k * multiplier * (actual - expected)

        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta

    result = games_df.copy()
    result["home_pregame_rating"] = home_pregame
    result["away_pregame_rating"] = away_pregame
    return result


def final_ratings(
    games_df: pd.DataFrame,
    k: float = DEFAULT_K,
    home_field: float = DEFAULT_HOME_FIELD,
    start_rating: float = DEFAULT_START_RATING,
) -> dict[str, float]:
    """Every team's rating after the last played game in games_df — used to
    seed live predictions for upcoming games."""
    rated = compute_pregame_ratings(games_df, k=k, home_field=home_field, start_rating=start_rating)
    played = rated[rated["home_score"].notna() & rated["away_score"].notna()]
    if played.empty:
        return {}

    ratings: dict[str, float] = {}
    for _, game in played.iterrows():
        home, away = game["home_team"], game["away_team"]
        home_rating = ratings.get(home, game["home_pregame_rating"])
        away_rating = ratings.get(away, game["away_pregame_rating"])
        margin = game["home_score"] - game["away_score"]
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        expected = _expected_home_win_prob(home_rating, away_rating, home_field)
        multiplier = _margin_multiplier(margin, home_rating - away_rating)
        delta = k * multiplier * (actual - expected)
        ratings[home] = home_rating + delta
        ratings[away] = away_rating - delta
    return ratings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_power_ratings.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/features/__init__.py src/nfl_predictor/features/power_ratings.py tests/test_power_ratings.py
git commit -m "feat: Elo-style team power ratings feature"
```

---

## Task 7: Features — rolling form and rest days

**Files:**
- Create: `src/nfl_predictor/features/rolling_form.py`
- Create: `src/nfl_predictor/features/rest_days.py`
- Test: `tests/test_rolling_form.py`
- Test: `tests/test_rest_days.py`

**Interfaces:**
- Produces: `rolling_form.add_rolling_form(games_df: pd.DataFrame, window: int = 5) -> pd.DataFrame` — adds `home_points_scored_roll`, `home_points_allowed_roll`, `away_points_scored_roll`, `away_points_allowed_roll` (mean of each team's own last `window` played games, shifted so the current game is excluded — NaN until a team has at least one prior game). `rest_days.add_rest_days(games_df: pd.DataFrame) -> pd.DataFrame` — adds `home_rest_days`, `away_rest_days` (days since that team's previous game; NaN for a team's first game in the frame — a wide bye/season-opener signal, not zero-filled here so `features/build.py` can decide the fill value deliberately).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rolling_form.py
import pandas as pd

from nfl_predictor.features import rolling_form


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
             "home_score": 27, "away_score": 20},
            {"game_id": "g2", "gameday": "2025-09-11", "home_team": "KC", "away_team": "CIN",
             "home_score": 10, "away_score": 14},
            {"game_id": "g3", "gameday": "2025-09-18", "home_team": "BAL", "away_team": "CIN",
             "home_score": 30, "away_score": 17},
        ]
    )


def test_first_appearance_has_no_rolling_form():
    result = rolling_form.add_rolling_form(_games(), window=5)

    g1 = result.iloc[0]
    assert pd.isna(g1["home_points_scored_roll"])
    assert pd.isna(g1["away_points_scored_roll"])


def test_second_game_reflects_only_the_prior_game():
    result = rolling_form.add_rolling_form(_games(), window=5)

    # KC's second appearance (game g2, as home) should reflect only
    # its away-team performance in g1 (scored 20, allowed 27).
    g2 = result.iloc[1]
    assert g2["home_points_scored_roll"] == 20.0
    assert g2["home_points_allowed_roll"] == 27.0
```

```python
# tests/test_rest_days.py
import pandas as pd

from nfl_predictor.features import rest_days


def _games():
    return pd.DataFrame(
        [
            {"game_id": "g1", "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC"},
            {"game_id": "g2", "gameday": "2025-09-11", "home_team": "KC", "away_team": "CIN"},
            {"game_id": "g3", "gameday": "2025-09-25", "home_team": "BAL", "away_team": "CIN"},
        ]
    )


def test_first_appearance_has_no_rest_days():
    result = rest_days.add_rest_days(_games())

    assert pd.isna(result.iloc[0]["home_rest_days"])
    assert pd.isna(result.iloc[0]["away_rest_days"])


def test_rest_days_counts_days_since_last_game():
    result = rest_days.add_rest_days(_games())

    g2 = result.iloc[1]
    assert g2["home_rest_days"] == 7  # KC away in g1 (09-04) -> home in g2 (09-11)

    g3 = result.iloc[2]
    assert g3["home_rest_days"] == 21  # BAL home in g1 (09-04) -> home in g3 (09-25), bye week
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_rolling_form.py tests/test_rest_days.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/nfl_predictor/features/rolling_form.py`**

```python
"""rolling_form.py — each team's own rolling scoring form, no lookahead.

Reshapes games into one row per team-appearance (home and away rows each
carry "points_scored"/"points_allowed" from that team's own perspective),
computes a shift(1) rolling mean per team so the current game is always
excluded, then reshapes back — same shift(1)-then-reshape discipline
PL_Predictor's features/rolling_form.py uses for goals scored/conceded.
"""

from __future__ import annotations

import pandas as pd


def _team_appearances(games_df: pd.DataFrame) -> pd.DataFrame:
    home = games_df[["game_id", "gameday", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_scored", "away_score": "points_allowed"}
    )
    away = games_df[["game_id", "gameday", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "points_scored", "home_score": "points_allowed"}
    )
    appearances = pd.concat([home, away], ignore_index=True)
    return appearances.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def add_rolling_form(games_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    appearances = _team_appearances(games_df)
    grouped = appearances.groupby("team")
    appearances["points_scored_roll"] = grouped["points_scored"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    appearances["points_allowed_roll"] = grouped["points_allowed"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    home_form = appearances.rename(
        columns={
            "team": "home_team",
            "points_scored_roll": "home_points_scored_roll",
            "points_allowed_roll": "home_points_allowed_roll",
        }
    )[["game_id", "home_team", "home_points_scored_roll", "home_points_allowed_roll"]]
    away_form = appearances.rename(
        columns={
            "team": "away_team",
            "points_scored_roll": "away_points_scored_roll",
            "points_allowed_roll": "away_points_allowed_roll",
        }
    )[["game_id", "away_team", "away_points_scored_roll", "away_points_allowed_roll"]]

    result = games_df.merge(home_form, on=["game_id", "home_team"], how="left")
    result = result.merge(away_form, on=["game_id", "away_team"], how="left")
    return result
```

- [ ] **Step 4: Write `src/nfl_predictor/features/rest_days.py`**

```python
"""rest_days.py — days since each team's previous game, no lookahead.

Bye weeks (roughly one per team per season) make this a bigger swing for
NFL than PL_Predictor's midweek-fixture-congestion equivalent — a team off
a bye typically shows 14 rest days instead of the usual 7.
"""

from __future__ import annotations

import pandas as pd


def _team_appearances(games_df: pd.DataFrame) -> pd.DataFrame:
    home = games_df[["game_id", "gameday", "home_team"]].rename(columns={"home_team": "team"})
    away = games_df[["game_id", "gameday", "away_team"]].rename(columns={"away_team": "team"})
    appearances = pd.concat([home, away], ignore_index=True)
    return appearances.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def add_rest_days(games_df: pd.DataFrame) -> pd.DataFrame:
    games_df = games_df.copy()
    games_df["gameday"] = pd.to_datetime(games_df["gameday"])
    appearances = _team_appearances(games_df)
    appearances["gameday"] = pd.to_datetime(appearances["gameday"])
    appearances["prior_gameday"] = appearances.groupby("team")["gameday"].shift(1)
    appearances["rest_days"] = (appearances["gameday"] - appearances["prior_gameday"]).dt.days

    home_rest = appearances.rename(columns={"team": "home_team", "rest_days": "home_rest_days"})[
        ["game_id", "home_team", "home_rest_days"]
    ]
    away_rest = appearances.rename(columns={"team": "away_team", "rest_days": "away_rest_days"})[
        ["game_id", "away_team", "away_rest_days"]
    ]

    result = games_df.merge(home_rest, on=["game_id", "home_team"], how="left")
    result = result.merge(away_rest, on=["game_id", "away_team"], how="left")
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_rolling_form.py tests/test_rest_days.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/nfl_predictor/features/rolling_form.py src/nfl_predictor/features/rest_days.py tests/test_rolling_form.py tests/test_rest_days.py
git commit -m "feat: rolling form and rest days features"
```

---

## Task 8: Features — single build entry point

**Files:**
- Create: `src/nfl_predictor/features/build.py`
- Test: `tests/test_build_features.py`

**Interfaces:**
- Consumes: `power_ratings.compute_pregame_ratings`, `power_ratings.final_ratings`, `rolling_form.add_rolling_form`, `rest_days.add_rest_days`.
- Produces: `build.FEATURE_COLUMNS: list[str]`, `build.build_training_frame(games_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]` (adds a `margin` and `total_points` target column too), `build.build_features_for_game(home_team: str, away_team: str, games_df: pd.DataFrame) -> pd.Series` (one live feature row for an upcoming game, using ratings/form computed from every played game in `games_df`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_features.py
import pandas as pd

from nfl_predictor.features import build


def _games():
    rows = []
    teams = ["BAL", "KC", "CIN", "BUF"]
    day = pd.Timestamp("2025-09-04")
    for week in range(1, 4):
        rows.append(
            {
                "game_id": f"g{week}a", "season": 2025, "week": week,
                "gameday": day + pd.Timedelta(days=7 * (week - 1)),
                "home_team": teams[0], "away_team": teams[1],
                "home_score": 24, "away_score": 20,
            }
        )
        rows.append(
            {
                "game_id": f"g{week}b", "season": 2025, "week": week,
                "gameday": day + pd.Timedelta(days=7 * (week - 1)),
                "home_team": teams[2], "away_team": teams[3],
                "home_score": 17, "away_score": 27,
            }
        )
    return pd.DataFrame(rows)


def test_build_training_frame_returns_feature_columns_and_targets():
    df, feature_cols = build.build_training_frame(_games())

    assert "margin" in df.columns
    assert "total_points" in df.columns
    assert set(feature_cols).issubset(df.columns)
    assert len(feature_cols) > 0
    assert (df["margin"] == df["home_score"] - df["away_score"]).all()
    assert (df["total_points"] == df["home_score"] + df["away_score"]).all()


def test_build_features_for_game_returns_series_with_feature_columns():
    games_df = _games()
    _, feature_cols = build.build_training_frame(games_df)

    row = build.build_features_for_game("BAL", "KC", games_df)

    assert isinstance(row, pd.Series)
    for col in feature_cols:
        assert col in row.index
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_build_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.features.build'`

- [ ] **Step 3: Write `src/nfl_predictor/features/build.py`**

```python
"""build.py — the single feature-construction entry point for game-outcome
models. Every consumer (training, walk-forward evaluation, live serving)
must call build_training_frame / build_features_for_game rather than
reimplementing feature logic inline — same discipline PL_Predictor's
features/build.py documents.
"""

from __future__ import annotations

import pandas as pd

from . import power_ratings, rest_days, rolling_form

FEATURE_COLUMNS = [
    "home_pregame_rating", "away_pregame_rating", "rating_diff",
    "home_points_scored_roll", "home_points_allowed_roll",
    "away_points_scored_roll", "away_points_allowed_roll",
    "home_rest_days", "away_rest_days",
    "div_game",
]


def _assemble(games_df: pd.DataFrame) -> pd.DataFrame:
    df = power_ratings.compute_pregame_ratings(games_df)
    df = rolling_form.add_rolling_form(df)
    df = rest_days.add_rest_days(df)
    df["rating_diff"] = df["home_pregame_rating"] - df["away_pregame_rating"]
    df["home_rest_days"] = df["home_rest_days"].fillna(7)
    df["away_rest_days"] = df["away_rest_days"].fillna(7)
    if "div_game" not in df.columns:
        df["div_game"] = 0
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    return df


def build_training_frame(games_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _assemble(games_df)
    played = df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)
    played["margin"] = played["home_score"] - played["away_score"]
    played["total_points"] = played["home_score"] + played["away_score"]
    return played, FEATURE_COLUMNS


def build_features_for_game(home_team: str, away_team: str, games_df: pd.DataFrame) -> pd.Series:
    """One live feature row for an upcoming home_team vs away_team game,
    computed from every played game in games_df (ratings/rolling form as of
    right now)."""
    ratings = power_ratings.final_ratings(games_df)
    played = games_df[games_df["home_score"].notna() & games_df["away_score"].notna()]

    def _recent_form(team: str) -> tuple[float, float]:
        appearances = pd.concat(
            [
                played[played["home_team"] == team][["gameday", "home_score", "away_score"]].rename(
                    columns={"home_score": "scored", "away_score": "allowed"}
                ),
                played[played["away_team"] == team][["gameday", "away_score", "home_score"]].rename(
                    columns={"away_score": "scored", "home_score": "allowed"}
                ),
            ]
        ).sort_values("gameday")
        recent = appearances.tail(5)
        if recent.empty:
            return float("nan"), float("nan")
        return float(recent["scored"].mean()), float(recent["allowed"].mean())

    def _rest_days(team: str) -> float:
        appearances = pd.concat(
            [
                played[played["home_team"] == team][["gameday"]],
                played[played["away_team"] == team][["gameday"]],
            ]
        ).sort_values("gameday")
        if appearances.empty:
            return 7.0
        last_game = pd.to_datetime(appearances.iloc[-1]["gameday"])
        return float((pd.Timestamp.now().normalize() - last_game).days)

    home_scored, home_allowed = _recent_form(home_team)
    away_scored, away_allowed = _recent_form(away_team)
    home_rating = ratings.get(home_team, power_ratings.DEFAULT_START_RATING)
    away_rating = ratings.get(away_team, power_ratings.DEFAULT_START_RATING)

    return pd.Series(
        {
            "home_pregame_rating": home_rating,
            "away_pregame_rating": away_rating,
            "rating_diff": home_rating - away_rating,
            "home_points_scored_roll": home_scored,
            "home_points_allowed_roll": home_allowed,
            "away_points_scored_roll": away_scored,
            "away_points_allowed_roll": away_allowed,
            "home_rest_days": _rest_days(home_team) or 7.0,
            "away_rest_days": _rest_days(away_team) or 7.0,
            "div_game": 0,
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_build_features.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/features/build.py tests/test_build_features.py
git commit -m "feat: single feature-construction entry point for game outcomes"
```

---

## Task 9: Features — player usage

**Files:**
- Create: `src/nfl_predictor/features/player_usage.py`
- Test: `tests/test_player_usage.py`

**Interfaces:**
- Produces: `player_usage.PLAYER_FEATURE_COLUMNS: list[str]`, `player_usage.build_player_training_frame(player_stats_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]` (adds rolling-form feature columns plus `anytime_td` target derived from `rushing_tds + receiving_tds + passing_tds > 0`), `player_usage.build_features_for_player(player_id: str, player_stats_df: pd.DataFrame) -> pd.Series | None` (`None` if the player has no prior games in `player_stats_df`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_usage.py
import pandas as pd

from nfl_predictor.features import player_usage


def _player_stats():
    rows = []
    for week in range(1, 4):
        rows.append(
            {
                "player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "BAL",
                "season": 2025, "week": week,
                "passing_yards": 0, "passing_tds": 0, "rushing_yards": 80 + week, "rushing_tds": 1,
                "receiving_yards": 10, "receiving_tds": 0, "receptions": 2, "targets": 3, "carries": 18,
            }
        )
    return pd.DataFrame(rows)


def test_build_player_training_frame_adds_rolling_features_and_target():
    df, feature_cols = player_usage.build_player_training_frame(_player_stats())

    assert "anytime_td" in df.columns
    assert set(feature_cols).issubset(df.columns)
    # Week 1 has no prior games, so its rolling features should be NaN.
    week1 = df[df["week"] == 1].iloc[0]
    assert pd.isna(week1["rushing_yards_roll"])
    # Week 3's rolling rushing yards should reflect weeks 1-2 only.
    week3 = df[df["week"] == 3].iloc[0]
    assert week3["rushing_yards_roll"] == (81 + 82) / 2


def test_build_features_for_player_returns_none_with_no_history():
    row = player_usage.build_features_for_player("unknown", _player_stats())
    assert row is None


def test_build_features_for_player_returns_series_with_history():
    row = player_usage.build_features_for_player("p1", _player_stats())
    assert row is not None
    assert row["rushing_yards_roll"] == pytest.approx((81 + 82 + 83) / 3)


import pytest  # noqa: E402  (kept local to the test that needs it)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_usage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.features.player_usage'`

- [ ] **Step 3: Write `src/nfl_predictor/features/player_usage.py`**

```python
"""player_usage.py — player-level rolling usage/production features for
anytime-TD and yardage prop models. Same shift(1)-then-rolling discipline as
features/rolling_form.py, applied per player instead of per team."""

from __future__ import annotations

import pandas as pd

ROLL_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "targets", "carries"]
PLAYER_FEATURE_COLUMNS = [f"{stat}_roll" for stat in ROLL_STATS]


def _add_rolling(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = df.groupby("player_id")
    for stat in ROLL_STATS:
        df[f"{stat}_roll"] = grouped[stat].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df


def build_player_training_frame(player_stats_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _add_rolling(player_stats_df)
    df["anytime_td"] = (
        (df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0) + df["passing_tds"].fillna(0)) > 0
    ).astype(int)
    return df, PLAYER_FEATURE_COLUMNS


def build_features_for_player(player_id: str, player_stats_df: pd.DataFrame) -> pd.Series | None:
    history = player_stats_df[player_stats_df["player_id"] == player_id].sort_values(["season", "week"])
    if history.empty:
        return None
    recent = history.tail(5)
    return pd.Series({f"{stat}_roll": float(recent[stat].mean()) for stat in ROLL_STATS})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_usage.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/features/player_usage.py tests/test_player_usage.py
git commit -m "feat: player usage rolling features"
```

---

## Task 10: Models — game outcome candidates (Elo, ridge, XGBoost)

**Files:**
- Create: `src/nfl_predictor/models/__init__.py`
- Create: `src/nfl_predictor/models/game_outcome.py`
- Test: `tests/test_game_outcome.py`

**Interfaces:**
- Consumes: `features.power_ratings`, `features.build.FEATURE_COLUMNS`.
- Produces: `game_outcome.fit_elo_candidate(train_df: pd.DataFrame) -> dict` (just carries the Elo home-field/scale constants — Elo needs no fitting beyond what `power_ratings` already computed into `train_df`), `game_outcome.fit_margin_regression(X_train: pd.DataFrame, y_margin: pd.Series) -> sklearn.linear_model.Ridge`, `game_outcome.fit_xgb_margin(X_train: pd.DataFrame, y_margin: pd.Series) -> xgboost.XGBRegressor`, `game_outcome.residual_sigma(model, X_val, y_val) -> float`, `game_outcome.margin_to_probabilities(predicted_margin: float, sigma: float, spread_line: float | None = None, total_line: float | None = None, predicted_total: float | None = None, total_sigma: float | None = None) -> dict` (returns `home_win_prob`, `away_win_prob`, and, when the corresponding line is given, `home_cover_prob`/`away_cover_prob` and `over_prob`/`under_prob`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_game_outcome.py
import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models import game_outcome


def _toy_frame():
    rng = np.random.default_rng(42)
    n = 60
    rating_diff = rng.normal(0, 100, n)
    margin = rating_diff * 0.05 + rng.normal(0, 10, n)
    return pd.DataFrame(
        {
            "home_pregame_rating": 1500 + rating_diff / 2,
            "away_pregame_rating": 1500 - rating_diff / 2,
            "rating_diff": rating_diff,
            "home_points_scored_roll": rng.normal(24, 5, n),
            "home_points_allowed_roll": rng.normal(21, 5, n),
            "away_points_scored_roll": rng.normal(23, 5, n),
            "away_points_allowed_roll": rng.normal(22, 5, n),
            "home_rest_days": 7,
            "away_rest_days": 7,
            "div_game": 0,
            "margin": margin,
        }
    )


FEATURE_COLS = [
    "home_pregame_rating", "away_pregame_rating", "rating_diff",
    "home_points_scored_roll", "home_points_allowed_roll",
    "away_points_scored_roll", "away_points_allowed_roll",
    "home_rest_days", "away_rest_days", "div_game",
]


def test_fit_margin_regression_predicts_signed_margin():
    df = _toy_frame()
    model = game_outcome.fit_margin_regression(df[FEATURE_COLS], df["margin"])

    preds = model.predict(df[FEATURE_COLS])
    # A model fit on data where margin correlates with rating_diff should
    # recover a positive relationship.
    assert np.corrcoef(preds, df["margin"])[0, 1] > 0.3


def test_fit_xgb_margin_predicts_signed_margin():
    df = _toy_frame()
    model = game_outcome.fit_xgb_margin(df[FEATURE_COLS], df["margin"])

    preds = model.predict(df[FEATURE_COLS])
    assert np.corrcoef(preds, df["margin"])[0, 1] > 0.3


def test_residual_sigma_is_positive():
    df = _toy_frame()
    model = game_outcome.fit_margin_regression(df[FEATURE_COLS], df["margin"])

    sigma = game_outcome.residual_sigma(model, df[FEATURE_COLS], df["margin"])

    assert sigma > 0


def test_margin_to_probabilities_favors_positive_margin():
    result = game_outcome.margin_to_probabilities(predicted_margin=7.0, sigma=13.0)

    assert result["home_win_prob"] > 0.5
    assert result["home_win_prob"] + result["away_win_prob"] == pytest.approx(1.0)


def test_margin_to_probabilities_includes_cover_and_total_when_lines_given():
    result = game_outcome.margin_to_probabilities(
        predicted_margin=7.0, sigma=13.0, spread_line=-3.0,
        total_line=45.0, predicted_total=48.0, total_sigma=10.0,
    )

    assert "home_cover_prob" in result
    assert "away_cover_prob" in result
    assert result["home_cover_prob"] + result["away_cover_prob"] == pytest.approx(1.0)
    assert "over_prob" in result
    assert "under_prob" in result
    assert result["over_prob"] + result["under_prob"] == pytest.approx(1.0)
    assert result["over_prob"] > 0.5  # predicted_total (48) is above total_line (45)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_game_outcome.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.models'`

- [ ] **Step 3: Write `src/nfl_predictor/models/__init__.py` and `game_outcome.py`**

```python
# src/nfl_predictor/models/__init__.py
```

```python
"""game_outcome.py — margin-of-victory candidates for the game-outcome
race, plus the shared margin -> win/cover/total-probability conversion.

NFL scoring isn't a low-count Poisson process like PL_Predictor's goals, so
this predicts a continuous point margin (home_score - away_score) and
total_points, then converts each to probabilities via a fitted-Normal
residual distribution — the standard shape used across public NFL
win-probability models. Three candidates are raced in evaluate/walk_forward.py:
Elo (implicit in features.power_ratings' rating_diff, converted directly via
margin_to_probabilities with a fixed points-per-Elo-point scale), ridge
regression, and XGBoost — whichever wins held-out log-loss is served.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Points-per-Elo-point conversion for the Elo candidate: derived once from a
# league-average relationship (roughly 25 Elo points per point of expected
# margin, close to 538's NFL Elo scale) — evaluate/walk_forward.py measures
# whether this beats the fitted regressors on real held-out data.
ELO_POINTS_PER_RATING_POINT = 1.0 / 25.0


def fit_elo_candidate(train_df: pd.DataFrame) -> dict:
    """The Elo candidate needs no additional fitting beyond the ratings
    features.power_ratings already computed into train_df — this just
    returns the fixed conversion constant so predict_margin_elo has
    everything it needs, kept in dict form for interface parity with the
    other two candidates' fitted model objects."""
    return {"points_per_rating_point": ELO_POINTS_PER_RATING_POINT}


def predict_margin_elo(candidate: dict, rating_diff: float, home_rest_days: float, away_rest_days: float) -> float:
    return rating_diff * candidate["points_per_rating_point"] + 0.05 * (home_rest_days - away_rest_days)


def fit_margin_regression(X_train: pd.DataFrame, y_margin: pd.Series) -> Ridge:
    model = Ridge(alpha=1.0)
    model.fit(X_train.fillna(0), y_margin)
    return model


def fit_xgb_margin(X_train: pd.DataFrame, y_margin: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        reg_lambda=1.0, reg_alpha=0.0, random_state=42,
    )
    model.fit(X_train.fillna(0), y_margin)
    return model


def residual_sigma(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    preds = model.predict(X_val.fillna(0))
    residuals = np.asarray(y_val) - preds
    return float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(residuals) or 1.0)


def margin_to_probabilities(
    predicted_margin: float,
    sigma: float,
    spread_line: float | None = None,
    total_line: float | None = None,
    predicted_total: float | None = None,
    total_sigma: float | None = None,
) -> dict:
    """margin ~ Normal(predicted_margin, sigma). home_win_prob = P(margin > 0).
    spread_line follows the schedules.py convention: the home team's
    closing spread (negative means home favored by that many points) — the
    home team covers when margin > -spread_line. total_points ~
    Normal(predicted_total, total_sigma); over_prob = P(total > total_line)."""
    home_win_prob = float(1.0 - norm.cdf(0.0, loc=predicted_margin, scale=sigma))
    result = {"home_win_prob": home_win_prob, "away_win_prob": 1.0 - home_win_prob}

    if spread_line is not None:
        home_cover_prob = float(1.0 - norm.cdf(-spread_line, loc=predicted_margin, scale=sigma))
        result["home_cover_prob"] = home_cover_prob
        result["away_cover_prob"] = 1.0 - home_cover_prob

    if total_line is not None and predicted_total is not None and total_sigma is not None:
        over_prob = float(1.0 - norm.cdf(total_line, loc=predicted_total, scale=total_sigma))
        result["over_prob"] = over_prob
        result["under_prob"] = 1.0 - over_prob

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_game_outcome.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/models/__init__.py src/nfl_predictor/models/game_outcome.py tests/test_game_outcome.py
git commit -m "feat: game outcome model candidates and margin-to-probability conversion"
```

---

## Task 11: Evaluate — walk-forward validation and candidate race

**Files:**
- Create: `src/nfl_predictor/evaluate/__init__.py`
- Create: `src/nfl_predictor/evaluate/walk_forward.py`
- Test: `tests/test_walk_forward.py`

**Interfaces:**
- Consumes: `features.build.build_training_frame`, `models.game_outcome.*`.
- Produces: `walk_forward.prepare_folds(games_df: pd.DataFrame, min_train_seasons: int = 2) -> list[dict]` (each fold: `val_season, train_df, val_df, feature_cols`), `walk_forward.evaluate_candidate(folds: list[dict], candidate: str) -> pd.DataFrame` (`candidate` in `{"elo", "ridge", "xgb"}`; one row per fold with `val_season, n_games, log_loss, brier`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_walk_forward.py
import numpy as np
import pandas as pd
import pytest

from nfl_predictor.evaluate import walk_forward


def _multi_season_games():
    rng = np.random.default_rng(7)
    rows = []
    teams = [f"T{i}" for i in range(8)]
    for season in (2022, 2023, 2024):
        for week in range(1, 6):
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                home_score = int(rng.integers(10, 35))
                away_score = int(rng.integers(10, 35))
                rows.append(
                    {
                        "game_id": f"{season}_{week}_{home}_{away}", "season": season, "week": week,
                        "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)),
                        "home_team": home, "away_team": away,
                        "home_score": home_score, "away_score": away_score,
                    }
                )
    return pd.DataFrame(rows)


def test_prepare_folds_holds_out_each_season_after_minimum():
    folds = walk_forward.prepare_folds(_multi_season_games(), min_train_seasons=2)

    val_seasons = [fold["val_season"] for fold in folds]
    assert val_seasons == [2024]  # only season 2024 has >= 2 prior seasons


def test_evaluate_candidate_returns_a_row_per_fold_for_each_candidate():
    folds = walk_forward.prepare_folds(_multi_season_games(), min_train_seasons=2)

    for candidate in ("elo", "ridge", "xgb"):
        result = walk_forward.evaluate_candidate(folds, candidate)
        assert len(result) == len(folds)
        assert (result["log_loss"] > 0).all()
        assert (result["brier"] >= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_walk_forward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.evaluate'`

- [ ] **Step 3: Write `src/nfl_predictor/evaluate/__init__.py` and `walk_forward.py`**

```python
# src/nfl_predictor/evaluate/__init__.py
```

```python
"""walk_forward.py — season-by-season walk-forward validation for the three
models/game_outcome.py candidates. Mirrors PL_Predictor's/F1_Predictor's own
evaluate/walk_forward.py: builds the full feature frame ONCE (no lookahead —
every feature is already shift(1)/expanding computed before any slicing),
then slices by season so evaluate_candidate can be called repeatedly without
redoing feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..features.build import build_training_frame
from ..models import game_outcome


def prepare_folds(games_df: pd.DataFrame, min_train_seasons: int = 4) -> list[dict]:
    df, feature_cols = build_training_frame(games_df)
    seasons = sorted(df["season"].unique())

    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        folds.append({"val_season": val_season, "train_df": train_df, "val_df": val_df, "feature_cols": feature_cols})
    return folds


def _predict_margins(candidate: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]):
    X_train, y_train = train_df[feature_cols], train_df["margin"]
    X_val = val_df[feature_cols]

    if candidate == "elo":
        model = game_outcome.fit_elo_candidate(train_df)
        preds = np.array(
            [
                game_outcome.predict_margin_elo(model, r, hr, ar)
                for r, hr, ar in zip(val_df["rating_diff"], val_df["home_rest_days"], val_df["away_rest_days"])
            ]
        )
        sigma = game_outcome.residual_sigma(
            type("_", (), {"predict": lambda self, X: np.array(
                [game_outcome.predict_margin_elo(model, r, hr, ar) for r, hr, ar in
                 zip(train_df["rating_diff"], train_df["home_rest_days"], train_df["away_rest_days"])]
            )})(),
            X_train, y_train,
        )
        return preds, sigma

    fit_fn = game_outcome.fit_margin_regression if candidate == "ridge" else game_outcome.fit_xgb_margin
    model = fit_fn(X_train, y_train)
    preds = model.predict(X_val.fillna(0))
    sigma = game_outcome.residual_sigma(model, X_train, y_train)
    return preds, sigma


def evaluate_candidate(folds: list[dict], candidate: str) -> pd.DataFrame:
    rows = []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        preds, sigma = _predict_margins(candidate, train_df, val_df, feature_cols)

        probs = np.array(
            [game_outcome.margin_to_probabilities(m, sigma)["home_win_prob"] for m in preds]
        )
        actual = (val_df["margin"] > 0).astype(int).to_numpy()
        # Clip away from exact 0/1 so log_loss never receives a probability
        # that would make it -inf on a single miss.
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        rows.append(
            {
                "val_season": fold["val_season"],
                "n_games": len(val_df),
                "log_loss": log_loss(actual, probs, labels=[0, 1]),
                "brier": brier_score_loss(actual, probs),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_walk_forward.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/evaluate/__init__.py src/nfl_predictor/evaluate/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: walk-forward validation harness for game outcome candidates"
```

---

## Task 12: Models — player props (anytime-TD classifier + yardage regressors)

**Files:**
- Create: `src/nfl_predictor/models/player_props.py`
- Test: `tests/test_player_props.py`

**Interfaces:**
- Produces: `player_props.fit_anytime_td_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> xgboost.XGBClassifier`, `player_props.fit_yardage_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> xgboost.XGBRegressor`, `player_props.YARDAGE_TARGETS: dict[str, str]` (maps prop name -> target column, e.g. `{"passing_yards": "passing_yards", "rushing_yards": "rushing_yards", "receiving_yards": "receiving_yards"}`), `player_props.predict_props(models: dict, feature_row: pd.Series, position: str) -> dict` (returns `anytime_td_prob` always; yardage predictions only for the stat categories relevant to `position` — QB gets `passing_yards`, RB gets `rushing_yards`, WR/TE get `receiving_yards`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_props.py
import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models import player_props


def _toy_player_frame(n=80, seed=1):
    rng = np.random.default_rng(seed)
    rushing_roll = rng.normal(80, 20, n)
    return pd.DataFrame(
        {
            "passing_yards_roll": rng.normal(250, 40, n),
            "rushing_yards_roll": rushing_roll,
            "receiving_yards_roll": rng.normal(50, 20, n),
            "targets_roll": rng.normal(5, 2, n),
            "carries_roll": rng.normal(15, 5, n),
            "rushing_yards": rushing_roll + rng.normal(0, 15, n),
            "anytime_td": (rng.random(n) < (0.3 + rushing_roll / 500)).astype(int),
        }
    )


FEATURE_COLS = ["passing_yards_roll", "rushing_yards_roll", "receiving_yards_roll", "targets_roll", "carries_roll"]


def test_fit_anytime_td_classifier_predicts_probabilities():
    df = _toy_player_frame()
    model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])

    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()


def test_fit_yardage_regressor_predicts_reasonable_values():
    df = _toy_player_frame()
    model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])

    preds = model.predict(df[FEATURE_COLS])
    assert np.corrcoef(preds, df["rushing_yards"])[0, 1] > 0.3


def test_predict_props_only_returns_relevant_yardage_market_for_position():
    df = _toy_player_frame()
    td_model = player_props.fit_anytime_td_classifier(df[FEATURE_COLS], df["anytime_td"])
    rushing_model = player_props.fit_yardage_regressor(df[FEATURE_COLS], df["rushing_yards"])
    models = {
        "anytime_td": td_model,
        "rushing_yards": rushing_model,
        "feature_cols": FEATURE_COLS,
    }

    result = player_props.predict_props(models, df[FEATURE_COLS].iloc[0], position="RB")

    assert "anytime_td_prob" in result
    assert "rushing_yards" in result
    assert "passing_yards" not in result
    assert "receiving_yards" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_props.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.models.player_props'`

- [ ] **Step 3: Write `src/nfl_predictor/models/player_props.py`**

```python
"""player_props.py — anytime-TD classifier and per-position yardage
regressors. Mirrors PL_Predictor's player goal/assist classifier approach
for anytime_td; yardage props are a genuinely new shape (continuous
regression, not PL_Predictor has an equivalent for) since NFL props are
commonly priced as an over/under yardage line rather than a probability."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

YARDAGE_TARGETS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}

# Which yardage market applies to which position — a QB doesn't have a
# meaningful rushing-yards prop line in practice, a WR/TE doesn't have a
# passing one, etc. RB can occasionally have a receiving line too, but v1
# keeps one yardage market per position for simplicity (see spec's v1 scope).
POSITION_YARDAGE_MARKET = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards"}


def fit_anytime_td_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train.fillna(0), y_train)
    return model


def fit_yardage_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train.fillna(0), y_train)
    return model


def predict_props(models: dict, feature_row: pd.Series, position: str) -> dict:
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0).to_numpy().reshape(1, -1)

    result = {"anytime_td_prob": float(models["anytime_td"].predict_proba(X)[0, 1])}

    market = POSITION_YARDAGE_MARKET.get(position)
    if market and market in models:
        result[market] = float(models[market].predict(X)[0])

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_player_props.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/models/player_props.py tests/test_player_props.py
git commit -m "feat: anytime-TD classifier and yardage regressor player prop models"
```

---

## Task 13: Models — manifest (train/save/load orchestration)

**Files:**
- Create: `src/nfl_predictor/models/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `data.schedules`, `data.player_stats`, `features.build`, `features.player_usage`, `evaluate.walk_forward`, `models.game_outcome`, `models.player_props`.
- Produces: `manifest.train_all(seasons: list[int] | None = None) -> dict`, `manifest.load_manifest() -> dict`, `manifest.load_models() -> dict` (returns `{"game_outcome_model": ..., "chosen_candidate": str, "sigma": float, "total_model": ..., "total_sigma": float, "player_models": {...}, "feature_cols": [...], "player_feature_cols": [...]}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_manifest.py
import json

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.models import manifest


def _fake_games(seasons):
    rng = np.random.default_rng(3)
    rows = []
    teams = [f"T{i}" for i in range(8)]
    for season in seasons:
        for week in range(1, 6):
            for i in range(0, len(teams), 2):
                home, away = teams[i], teams[i + 1]
                rows.append(
                    {
                        "game_id": f"{season}_{week}_{home}_{away}", "season": season, "week": week,
                        "gameday": pd.Timestamp(f"{season}-09-01") + pd.Timedelta(days=7 * (week - 1)),
                        "home_team": home, "away_team": away,
                        "home_score": int(rng.integers(10, 35)), "away_score": int(rng.integers(10, 35)),
                    }
                )
    return pd.DataFrame(rows)


def _fake_player_stats(seasons):
    rng = np.random.default_rng(4)
    rows = []
    for season in seasons:
        for week in range(1, 6):
            rows.append(
                {
                    "player_id": "p1", "player_name": "Runner", "position": "RB", "recent_team": "T0",
                    "season": season, "week": week,
                    "passing_yards": 0, "passing_tds": 0,
                    "rushing_yards": int(rng.integers(40, 120)), "rushing_tds": int(rng.integers(0, 2)),
                    "receiving_yards": int(rng.integers(0, 30)), "receiving_tds": 0,
                    "receptions": 2, "targets": 3, "carries": 18,
                }
            )
    return pd.DataFrame(rows)


def test_train_all_writes_a_manifest_with_chosen_candidate(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))

    result = manifest.train_all(seasons=seasons)

    assert result["chosen_candidate"] in ("elo", "ridge", "xgb")
    assert (tmp_path / "manifest.json").exists()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["chosen_candidate"] == result["chosen_candidate"]


def test_load_models_round_trips_after_train_all(monkeypatch, tmp_path):
    from nfl_predictor import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")

    seasons = [2021, 2022, 2023, 2024]
    monkeypatch.setattr(manifest.schedules, "load_training_data", lambda s: _fake_games(seasons))
    monkeypatch.setattr(manifest.player_stats, "fetch_weekly_player_stats", lambda s: _fake_player_stats(seasons))

    manifest.train_all(seasons=seasons)
    models = manifest.load_models()

    assert "game_outcome_model" in models
    assert "player_models" in models
    assert "anytime_td" in models["player_models"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.models.manifest'`

- [ ] **Step 3: Write `src/nfl_predictor/models/manifest.py`**

```python
"""manifest.py — train/save/load orchestration for both the game-outcome
candidate race and the player prop models. Mirrors PL_Predictor's/
F1_Predictor's own models/manifest.py: one manifest.json aggregating
metadata/metrics, symmetric load_manifest()/load_models() helpers.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import pandas as pd

from ..config import MODELS_DIR
from ..data import player_stats, schedules
from ..evaluate import walk_forward
from ..features import build as feature_build
from ..features import player_usage
from . import game_outcome, player_props

MANIFEST_PATH = MODELS_DIR / "manifest.json"
GAME_MODEL_PATH = MODELS_DIR / "game_outcome_model.pkl"
TOTAL_MODEL_PATH = MODELS_DIR / "total_points_model.pkl"
ANYTIME_TD_MODEL_PATH = MODELS_DIR / "anytime_td_model.pkl"
YARDAGE_MODEL_PATHS = {
    market: MODELS_DIR / f"{market}_model.pkl" for market in player_props.YARDAGE_TARGETS
}

DEFAULT_TRAIN_SEASONS = 8


def _save_pickle(obj, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_all(seasons: list[int] | None = None) -> dict:
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    seasons = seasons or schedules.default_completed_seasons(n=DEFAULT_TRAIN_SEASONS)

    games_df = schedules.load_training_data(seasons)
    train_df, feature_cols = feature_build.build_training_frame(games_df)

    folds = walk_forward.prepare_folds(games_df, min_train_seasons=max(1, len(seasons) - 2))
    candidate_scores = {}
    for candidate in ("elo", "ridge", "xgb"):
        scored = walk_forward.evaluate_candidate(folds, candidate) if folds else pd.DataFrame()
        candidate_scores[candidate] = float(scored["log_loss"].mean()) if not scored.empty else float("inf")
    chosen = min(candidate_scores, key=candidate_scores.get)

    X_train = train_df[feature_cols]
    y_margin = train_df["margin"]
    y_total = train_df["total_points"]

    if chosen == "elo":
        game_model = game_outcome.fit_elo_candidate(train_df)
    elif chosen == "ridge":
        game_model = game_outcome.fit_margin_regression(X_train, y_margin)
    else:
        game_model = game_outcome.fit_xgb_margin(X_train, y_margin)

    if chosen == "elo":
        margin_preds = train_df.apply(
            lambda r: game_outcome.predict_margin_elo(game_model, r["rating_diff"], r["home_rest_days"], r["away_rest_days"]),
            axis=1,
        )
        sigma_model = type("_", (), {"predict": lambda self, X: margin_preds.to_numpy()})()
        sigma = game_outcome.residual_sigma(sigma_model, X_train, y_margin)
    else:
        sigma = game_outcome.residual_sigma(game_model, X_train, y_margin)

    total_model = game_outcome.fit_xgb_margin(X_train, y_total)
    total_sigma = game_outcome.residual_sigma(total_model, X_train, y_total)

    _save_pickle(game_model, GAME_MODEL_PATH)
    _save_pickle(total_model, TOTAL_MODEL_PATH)

    player_df_raw = player_stats.fetch_weekly_player_stats(seasons)
    player_train_df, player_feature_cols = player_usage.build_player_training_frame(player_df_raw)
    player_train_df = player_train_df.dropna(subset=player_feature_cols, how="all")
    X_player = player_train_df[player_feature_cols].fillna(0)

    anytime_td_model = player_props.fit_anytime_td_classifier(X_player, player_train_df["anytime_td"])
    _save_pickle(anytime_td_model, ANYTIME_TD_MODEL_PATH)

    yardage_metrics = {}
    for market, target_col in player_props.YARDAGE_TARGETS.items():
        subset = player_train_df[player_train_df[target_col] > 0]
        if subset.empty:
            continue
        model = player_props.fit_yardage_regressor(subset[player_feature_cols].fillna(0), subset[target_col])
        _save_pickle(model, YARDAGE_MODEL_PATHS[market])
        yardage_metrics[market] = {"n_train": int(len(subset))}

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(int(s) for s in seasons),
        "n_train": int(len(train_df)),
        "feature_cols": feature_cols,
        "player_feature_cols": player_feature_cols,
        "chosen_candidate": chosen,
        "candidate_scores": candidate_scores,
        "sigma": sigma,
        "total_sigma": total_sigma,
        "yardage_metrics": yardage_metrics,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No trained models found. Run `python -m nfl_predictor.models.manifest` first.")
    return json.loads(MANIFEST_PATH.read_text())


def load_models() -> dict:
    manifest = load_manifest()
    player_models = {"feature_cols": manifest["player_feature_cols"], "anytime_td": _load_pickle(ANYTIME_TD_MODEL_PATH)}
    for market, path in YARDAGE_MODEL_PATHS.items():
        if path.exists():
            player_models[market] = _load_pickle(path)

    return {
        "game_outcome_model": _load_pickle(GAME_MODEL_PATH),
        "total_model": _load_pickle(TOTAL_MODEL_PATH),
        "chosen_candidate": manifest["chosen_candidate"],
        "sigma": manifest["sigma"],
        "total_sigma": manifest["total_sigma"],
        "player_models": player_models,
        "feature_cols": manifest["feature_cols"],
        "player_feature_cols": manifest["player_feature_cols"],
    }


if __name__ == "__main__":
    train_all()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_manifest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/models/manifest.py tests/test_manifest.py
git commit -m "feat: manifest train/save/load orchestration"
```

---

## Task 14: Tracking — SQLite snapshot-then-reconcile store

**Files:**
- Create: `src/nfl_predictor/tracking/__init__.py`
- Create: `src/nfl_predictor/tracking/store.py`
- Test: `tests/test_tracking_store.py`

**Interfaces:**
- Produces: `store.record_game_predictions(games: list[dict]) -> int`, `store.reconcile_game_predictions(results_df: pd.DataFrame) -> int`, `store.get_track_record() -> dict`, `store.record_player_prop_predictions(props: list[dict]) -> int`, `store.reconcile_player_prop_predictions(player_stats_df: pd.DataFrame) -> int`. Each `games` dict: `game_id, home_team, away_team, commence_time, home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob`. Each `props` dict: `game_id, player_id, player_name, market, predicted_value` (`market` in `{"anytime_td", "passing_yards", "rushing_yards", "receiving_yards"}`; `predicted_value` is a probability for `anytime_td`, a yardage point estimate otherwise).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracking_store.py
import pandas as pd
import pytest

from nfl_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _game():
    return {
        "game_id": "2025_01_BAL_KC", "home_team": "BAL", "away_team": "KC",
        "commence_time": "2025-09-04T20:20:00", "home_win_prob": 0.58, "away_win_prob": 0.42,
        "home_cover_prob": 0.52, "away_cover_prob": 0.48, "over_prob": 0.55, "under_prob": 0.45,
    }


def test_record_game_predictions_is_idempotent():
    n1 = store.record_game_predictions([_game()])
    n2 = store.record_game_predictions([_game()])

    assert n1 == 1
    assert n2 == 0  # already logged, INSERT OR IGNORE


def test_reconcile_game_predictions_fills_actual_outcome():
    store.record_game_predictions([_game()])
    results = pd.DataFrame(
        [{"game_id": "2025_01_BAL_KC", "home_score": 27, "away_score": 20}]
    )

    n = store.reconcile_game_predictions(results)

    assert n == 1
    record = store.get_track_record()
    assert record["n_resolved_games"] == 1
    assert record["pct_moneyline_correct"] == 1.0  # predicted home win, home won


def test_record_and_reconcile_player_prop_predictions():
    prop = {"game_id": "2025_01_BAL_KC", "player_id": "p1", "player_name": "Runner",
            "market": "rushing_yards", "predicted_value": 85.0}
    store.record_player_prop_predictions([prop])

    player_stats_df = pd.DataFrame(
        [{"game_id": "2025_01_BAL_KC", "player_id": "p1", "rushing_yards": 92, "receiving_yards": 5,
          "passing_yards": 0, "rushing_tds": 1, "receiving_tds": 0, "passing_tds": 0}]
    )
    n = store.reconcile_player_prop_predictions(player_stats_df)

    assert n == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_tracking_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.tracking'`

- [ ] **Step 3: Write `src/nfl_predictor/tracking/__init__.py` and `store.py`**

```python
# src/nfl_predictor/tracking/__init__.py
```

```python
"""store.py — SQLite persistence for the live prediction track record.

Snapshots each game's core-market predictions and each tracked player prop
*before* kickoff, then reconciles them against actual results once games are
played — the only honest way to measure "how good are the predictions
really." Same discipline as PL_Predictor's/F1_Predictor's tracking/store.py:
snapshot rows are immutable (INSERT OR IGNORE), reconciliation only fills in
outcome columns on already-existing rows.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ..config import TRACKING_DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRACKING_DB_PATH), timeout=15)
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS game_predictions (
            game_id TEXT PRIMARY KEY,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            home_cover_prob REAL,
            away_cover_prob REAL,
            over_prob REAL,
            under_prob REAL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            moneyline_hit INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_prop_predictions (
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            market TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            snapshotted_at TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_value REAL,
            PRIMARY KEY (game_id, player_id, market)
        )
        """
    )
    return conn


def record_game_predictions(games: list[dict]) -> int:
    if not games:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            g["game_id"], g["home_team"], g["away_team"], g["commence_time"], now,
            float(g["home_win_prob"]), float(g["away_win_prob"]),
            g.get("home_cover_prob"), g.get("away_cover_prob"),
            g.get("over_prob"), g.get("under_prob"),
        )
        for g in games
    ]
    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO game_predictions
                (game_id, home_team, away_team, commence_time, snapshotted_at,
                 home_win_prob, away_win_prob, home_cover_prob, away_cover_prob, over_prob, under_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


def reconcile_game_predictions(results_df: pd.DataFrame) -> int:
    if results_df.empty:
        return 0
    with _connect() as conn:
        unresolved = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(results_df, on="game_id", how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            home_win = row["home_score"] > row["away_score"]
            predicted_home_win = row["home_win_prob"] >= row["away_win_prob"]
            moneyline_hit = int(predicted_home_win == home_win)
            conn.execute(
                """
                UPDATE game_predictions
                SET resolved = 1, actual_home_score = ?, actual_away_score = ?, moneyline_hit = ?
                WHERE game_id = ?
                """,
                (int(row["home_score"]), int(row["away_score"]), moneyline_hit, row["game_id"]),
            )
            resolved_count += 1
        return resolved_count


def get_track_record() -> dict:
    with _connect() as conn:
        resolved = pd.read_sql("SELECT * FROM game_predictions WHERE resolved = 1", conn)
    if resolved.empty:
        return {"n_resolved_games": 0, "pct_moneyline_correct": None}
    return {
        "n_resolved_games": int(len(resolved)),
        "pct_moneyline_correct": float(resolved["moneyline_hit"].mean()),
    }


def record_player_prop_predictions(props: list[dict]) -> int:
    if not props:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (p["game_id"], p["player_id"], p["player_name"], p["market"], float(p["predicted_value"]), now)
        for p in props
    ]
    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value, snapshotted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


_MARKET_TO_STAT_COLUMN = {
    "anytime_td": None,  # handled specially: derived from the *_tds columns
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}


def reconcile_player_prop_predictions(player_stats_df: pd.DataFrame) -> int:
    if player_stats_df.empty:
        return 0
    with _connect() as conn:
        unresolved = pd.read_sql("SELECT * FROM player_prop_predictions WHERE resolved = 0", conn)
        if unresolved.empty:
            return 0

        merged = unresolved.merge(player_stats_df, on=["game_id", "player_id"], how="inner")
        resolved_count = 0
        for _, row in merged.iterrows():
            if row["market"] == "anytime_td":
                actual = float(
                    (row.get("rushing_tds", 0) or 0) + (row.get("receiving_tds", 0) or 0) + (row.get("passing_tds", 0) or 0) > 0
                )
            else:
                stat_col = _MARKET_TO_STAT_COLUMN[row["market"]]
                if stat_col not in row or pd.isna(row[stat_col]):
                    continue
                actual = float(row[stat_col])
            conn.execute(
                """
                UPDATE player_prop_predictions
                SET resolved = 1, actual_value = ?
                WHERE game_id = ? AND player_id = ? AND market = ?
                """,
                (actual, row["game_id"], row["player_id"], row["market"]),
            )
            resolved_count += 1
        return resolved_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_tracking_store.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/tracking/__init__.py src/nfl_predictor/tracking/store.py tests/test_tracking_store.py
git commit -m "feat: SQLite snapshot-then-reconcile tracking store"
```

---

## Task 15: Odds — value bet detection (Shin de-vig)

**Files:**
- Create: `src/nfl_predictor/odds/__init__.py`
- Create: `src/nfl_predictor/odds/value_bets.py`
- Test: `tests/test_value_bets.py`

**Interfaces:**
- Consumes: `data.odds_api` output shape, model prediction dict shape from `models.game_outcome.margin_to_probabilities`.
- Produces: `value_bets.devig_h2h(home_price: float, away_price: float) -> dict | None` (Shin method via `scipy`-based implied-probability solve — no `penaltyblog` dependency for NFL since it's a soccer-oriented package; implemented directly here with the two-way Shin closed form), `value_bets.devig_totals(over_price: float, under_price: float) -> dict | None`, `value_bets.build_value_bet_table(games_df: pd.DataFrame, odds_df: pd.DataFrame, predictions: dict[str, dict], edge_threshold: float = 0.05) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_value_bets.py
import pandas as pd
import pytest

from nfl_predictor.odds import value_bets


def test_devig_h2h_removes_the_bookmaker_margin():
    result = value_bets.devig_h2h(home_price=1.91, away_price=1.91)

    assert result is not None
    assert result["home_win"] == pytest.approx(0.5, abs=0.01)
    assert result["home_win"] + result["away_win"] == pytest.approx(1.0, abs=1e-6)


def test_devig_totals_removes_the_bookmaker_margin():
    result = value_bets.devig_totals(over_price=1.91, under_price=1.91)

    assert result is not None
    assert result["over"] == pytest.approx(0.5, abs=0.01)


def test_build_value_bet_table_flags_a_positive_edge():
    games_df = pd.DataFrame(
        [{"game_id": "g1", "home_team": "BAL", "away_team": "KC", "commence_time": "2025-09-04T20:20:00"}]
    )
    odds_df = pd.DataFrame(
        [
            {"event_id": "g1", "market": "h2h", "outcome_name": "BAL", "price": 2.20, "point": None, "bookmaker": "dk", "odds_fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
            {"event_id": "g1", "market": "h2h", "outcome_name": "KC", "price": 1.75, "point": None, "bookmaker": "dk", "odds_fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
        ]
    )
    predictions = {"g1": {"home_win_prob": 0.60, "away_win_prob": 0.40}}

    table = value_bets.build_value_bet_table(games_df, odds_df, predictions)

    row = table.iloc[0]
    assert row["home_win_edge"] > 0
    assert "home_win" in row["value_bet_flags"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_value_bets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.odds'`

- [ ] **Step 3: Write `src/nfl_predictor/odds/__init__.py` and `value_bets.py`**

```python
# src/nfl_predictor/odds/__init__.py
```

```python
"""value_bets.py — join model predictions to live Odds API lines and
surface edges (model probability - de-vigged implied probability) above a
threshold. Same Shin de-vig + single-recommendation-never-a-parlay
discipline as PL_Predictor's odds/value_bets.py, implemented directly here
(two-way closed-form Shin solve) rather than via penaltyblog, which is a
soccer-oriented package not used elsewhere in this project.
"""

from __future__ import annotations

import pandas as pd
from scipy.optimize import brentq

MAX_ODDS_AGE_SECONDS = 60 * 60


def _shin_two_way(price_a: float, price_b: float) -> tuple[float, float] | None:
    """Two-outcome Shin de-vig: solves for the insider-trading parameter z
    such that the resulting probabilities sum to 1, per Shin (1992/1993).
    For two outcomes this has a direct closed-form-friendly root; brentq is
    used for robustness against the occasional pathological price pair."""
    pi_a, pi_b = 1.0 / price_a, 1.0 / price_b
    overround = pi_a + pi_b
    if overround <= 1.0:
        return None  # no bookmaker margin at all — implausible, treat as bad input

    def _z_equation(z: float) -> float:
        # Shin's formula for a single outcome's true probability given
        # overround-implied probability pi and insider parameter z.
        def true_prob(pi):
            inside = z * z + 4 * (1 - z) * (pi * pi) / overround
            return (max(inside, 0.0) ** 0.5 - z) / (2 * (1 - z)) if z < 1 else pi / overround

        return true_prob(pi_a) + true_prob(pi_b) - 1.0

    try:
        z = brentq(_z_equation, 0.0, 0.2)
    except ValueError:
        z = 0.0  # fall back to simple proportional de-vig if Shin doesn't converge

    if z <= 0.0:
        return pi_a / overround, pi_b / overround

    def true_prob(pi):
        inside = z * z + 4 * (1 - z) * (pi * pi) / overround
        return (max(inside, 0.0) ** 0.5 - z) / (2 * (1 - z))

    p_a, p_b = true_prob(pi_a), true_prob(pi_b)
    total = p_a + p_b
    return p_a / total, p_b / total


def devig_h2h(home_price: float, away_price: float) -> dict | None:
    result = _shin_two_way(home_price, away_price)
    if result is None:
        return None
    home, away = result
    return {"home_win": home, "away_win": away}


def devig_totals(over_price: float, under_price: float) -> dict | None:
    result = _shin_two_way(over_price, under_price)
    if result is None:
        return None
    over, under = result
    return {"over": over, "under": under}


def _best_price(odds_df: pd.DataFrame, event_id, market: str, outcome_name: str) -> float | None:
    rows = odds_df[(odds_df["event_id"] == event_id) & (odds_df["market"] == market) & (odds_df["outcome_name"] == outcome_name)]
    if rows.empty:
        return None
    return float(rows.loc[rows["price"].idxmax()]["price"])


def build_value_bet_table(
    games_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    predictions: dict[str, dict],
    edge_threshold: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for _, game in games_df.iterrows():
        game_id, home, away = game["game_id"], game["home_team"], game["away_team"]
        pred = predictions.get(game_id, {})
        row = {"game_id": game_id, "home_team": home, "away_team": away, "commence_time": game["commence_time"], **pred}

        home_price = _best_price(odds_df, game_id, "h2h", home)
        away_price = _best_price(odds_df, game_id, "h2h", away)
        implied = devig_h2h(home_price, away_price) if home_price and away_price else None

        over_price = _best_price(odds_df, game_id, "totals", "Over")
        under_price = _best_price(odds_df, game_id, "totals", "Under")
        implied_totals = devig_totals(over_price, under_price) if over_price and under_price else None

        row["home_win_edge"] = (pred.get("home_win_prob", 0) - implied["home_win"]) if implied else None
        row["away_win_edge"] = (pred.get("away_win_prob", 0) - implied["away_win"]) if implied else None
        row["over_edge"] = (pred.get("over_prob", 0) - implied_totals["over"]) if implied_totals and "over_prob" in pred else None
        row["under_edge"] = (pred.get("under_prob", 0) - implied_totals["under"]) if implied_totals and "under_prob" in pred else None

        row["value_bet_flags"] = [
            side for side in ("home_win", "away_win", "over", "under")
            if row.get(f"{side}_edge") is not None and row[f"{side}_edge"] > edge_threshold
        ]
        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_value_bets.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/odds/__init__.py src/nfl_predictor/odds/value_bets.py tests/test_value_bets.py
git commit -m "feat: Shin de-vig value bet detection for NFL game markets"
```

---

## Task 16: API — schemas and routes

**Files:**
- Create: `src/nfl_predictor/api/__init__.py`
- Create: `src/nfl_predictor/api/schemas.py`
- Create: `src/nfl_predictor/api/routes.py`
- Test: `tests/test_api_routes.py`

**Interfaces:**
- Consumes: `data.schedules`, `data.odds_api`, `data.player_stats`, `models.manifest`, `models.game_outcome`, `models.player_props`, `tracking.store`, `odds.value_bets`, `features.build`, `features.player_usage`.
- Produces: `routes.router` (FastAPI `APIRouter`) with `GET /games`, `GET /games/{season}/{week}/{game_id}/prediction`, `GET /players/{season}/{week}/props`, `GET /track-record`, `POST /retrain`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_routes.py
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api.main import app
from nfl_predictor.api import routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "2025_01_BAL_KC", "season": season, "week": week,
              "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
              "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5}]
        ),
    )
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "BAL", "away_team": "KC", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {
            "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
            "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
            "feature_cols": [], "player_feature_cols": [],
        },
    )
    monkeypatch.setattr(
        routes, "_predict_game_from_models",
        lambda models, home, away, games_df, spread_line=None, total_line=None: {
            "home_win_prob": 0.6, "away_win_prob": 0.4, "home_cover_prob": 0.55, "away_cover_prob": 0.45,
            "over_prob": 0.52, "under_prob": 0.48,
        },
    )
    monkeypatch.setattr(routes.store, "get_track_record", lambda: {"n_resolved_games": 0, "pct_moneyline_correct": None})
    return TestClient(app)


def test_get_games_returns_week_slate(client):
    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["game_id"] == "2025_01_BAL_KC"


def test_get_game_prediction(client):
    response = client.get("/api/games/2025/1/2025_01_BAL_KC/prediction")

    assert response.status_code == 200
    body = response.json()
    assert body["home_win_prob"] == 0.6


def test_get_game_prediction_404s_for_unknown_game(client):
    response = client.get("/api/games/2025/1/nonexistent/prediction")

    assert response.status_code == 404


def test_get_track_record(client):
    response = client.get("/api/track-record")

    assert response.status_code == 200
    assert response.json()["n_resolved_games"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_api_routes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nfl_predictor.api'`

- [ ] **Step 3: Write `src/nfl_predictor/api/__init__.py`, `schemas.py`, and `routes.py`**

```python
# src/nfl_predictor/api/__init__.py
```

```python
"""schemas.py — Pydantic response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class GameSummary(BaseModel):
    game_id: str
    season: int
    week: int
    gameday: str
    home_team: str
    away_team: str
    home_score: int | None = None
    away_score: int | None = None


class GamePrediction(BaseModel):
    home_win_prob: float
    away_win_prob: float
    home_cover_prob: float | None = None
    away_cover_prob: float | None = None
    over_prob: float | None = None
    under_prob: float | None = None


class TrackRecord(BaseModel):
    n_resolved_games: int
    pct_moneyline_correct: float | None = None


class RetrainResponse(BaseModel):
    trained_at: str
    chosen_candidate: str
```

```python
"""routes.py — game/player prop/track-record endpoints. Thin HTTP layer
over data/features/models/tracking, same shape as PL_Predictor's/
F1_Predictor's own api/routes.py."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..data import odds_api, player_stats, schedules
from ..features import build as feature_build
from ..features import player_usage
from ..models import game_outcome, manifest, player_props
from ..odds import value_bets
from ..tracking import store

router = APIRouter(prefix="/api")


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


@router.get("/games")
def get_games(season: int, week: int):
    games = schedules.fetch_upcoming_games(season, week)
    return games.to_dict("records")


@router.get("/games/{season}/{week}/{game_id}/prediction")
def get_game_prediction(season: int, week: int, game_id: str):
    games = schedules.fetch_upcoming_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0]

    models = _load_models_cached()
    history = schedules.load_training_data(schedules.default_completed_seasons(n=8) + [season])
    prediction = _predict_game_from_models(
        models, game["home_team"], game["away_team"], history,
        spread_line=game.get("spread_line"), total_line=game.get("total_line"),
    )
    return prediction


@router.get("/players/{season}/{week}/props")
def get_player_props(season: int, week: int):
    models = _load_models_cached()
    player_history = player_stats.fetch_weekly_player_stats(schedules.default_completed_seasons(n=8) + [season])

    latest_players = (
        player_history[player_history["season"] == season]
        [["player_id", "player_name", "position", "recent_team"]]
        .drop_duplicates("player_id")
    )
    results = []
    for _, player in latest_players.iterrows():
        feature_row = player_usage.build_features_for_player(player["player_id"], player_history)
        if feature_row is None:
            continue
        props = player_props.predict_props(models["player_models"], feature_row, position=player["position"])
        results.append({"player_id": player["player_id"], "player_name": player["player_name"], **props})
    return results


@router.get("/track-record")
def get_track_record():
    return store.get_track_record()


@router.post("/retrain")
def retrain():
    result = manifest.train_all()
    _load_models_cached.cache_clear()
    return {"trained_at": result["trained_at"], "chosen_candidate": result["chosen_candidate"]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_api_routes.py -v`
Expected: PASS (4 tests) — note this task depends on Task 17's `api/main.py` existing; write that file first if running this task's tests standalone, or complete Tasks 16-17 together.

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/api/__init__.py src/nfl_predictor/api/schemas.py src/nfl_predictor/api/routes.py tests/test_api_routes.py
git commit -m "feat: API routes for games, player props, and track record"
```

---

## Task 17: API — FastAPI app entry point

**Files:**
- Create: `src/nfl_predictor/api/main.py`

**Interfaces:**
- Consumes: `api.routes.router`, `config.FRONTEND_DIST_DIR`.
- Produces: `main.app` (FastAPI instance) — importable as `nfl_predictor.api.main:app` for uvicorn.

- [ ] **Step 1: Write `src/nfl_predictor/api/main.py`**

```python
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
```

- [ ] **Step 2: Run the full backend test suite to verify everything wires together**

Run: `PYTHONPATH=$(pwd)/src pytest tests/ -v`
Expected: PASS (all tests from Tasks 1-16)

- [ ] **Step 3: Manually smoke-test the running server**

```bash
PYTHONPATH=$(pwd)/src uvicorn nfl_predictor.api.main:app --host 0.0.0.0 --port 8001
```

In another shell: `curl http://localhost:8001/docs` should return the FastAPI Swagger UI HTML.

- [ ] **Step 4: Commit**

```bash
git add src/nfl_predictor/api/main.py
git commit -m "feat: FastAPI app entry point"
```

---

## Task 18: Train the first real manifest and verify end-to-end

**Files:**
- Modify: none (verification task — runs the real pipeline against real nflverse data for the first time)

- [ ] **Step 1: Install the package and dependencies**

```bash
cd /Users/sigey/Documents/Projects/NFL_Predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONPATH=$(pwd)/src
```

- [ ] **Step 2: Run the full test suite against the real environment**

Run: `pytest tests/ -v`
Expected: PASS (all tests — these use monkeypatched data, so this doesn't hit the network yet)

- [ ] **Step 3: Train the real manifest against live nflverse data**

Run: `python -m nfl_predictor.models.manifest`

Expected: prints training progress, writes `models/manifest.json`, `models/game_outcome_model.pkl`, `models/total_points_model.pkl`, `models/anytime_td_model.pkl`, and the yardage model files. This is the first real network call to `nfl_data_py` — if it fails, check `nfl_data_py`'s current API against `data/schedules.py::_import_schedules`/`data/player_stats.py::_import_weekly_data`/`data/injuries.py::_import_injuries`'s column expectations (`KEEP_COLUMNS` in each file) and adjust column names if nflverse has changed them since this plan was written.

- [ ] **Step 4: Sanity-check the manifest**

```bash
python -c "import json; m = json.load(open('models/manifest.json')); print(m['chosen_candidate'], m['candidate_scores'])"
```

Expected: a real `chosen_candidate` and three finite `candidate_scores` (elo/ridge/xgb), confirming the walk-forward race actually ran against real historical data.

- [ ] **Step 5: Start the API and confirm a real prediction**

```bash
uvicorn nfl_predictor.api.main:app --host 0.0.0.0 --port 8001 &
sleep 2
curl "http://localhost:8001/api/games?season=2026&week=1"
```

Expected: a JSON list of real week-1 2026 games (or the nearest upcoming week if week 1 has already started/finished by the time this runs).

- [ ] **Step 6: No commit needed** — this task verifies the pipeline works against real data; `models/*.pkl`/`manifest.json` are gitignored (Task 1's `.gitignore`).

---

## Task 19: Frontend — scaffold, API client, and types

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tsconfig.app.json`, `frontend/tsconfig.node.json`, `frontend/index.html`
- Create: `frontend/src/main.tsx`, `frontend/src/types.ts`, `frontend/src/api/client.ts`, `frontend/src/index.css`

- [ ] **Step 1: Scaffold with Vite**

```bash
cd /Users/sigey/Documents/Projects/NFL_Predictor
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

- [ ] **Step 2: Write `frontend/src/types.ts`**

```typescript
export interface GameSummary {
  game_id: string;
  season: number;
  week: number;
  gameday: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  spread_line: number | null;
  total_line: number | null;
}

export interface GamePrediction {
  home_win_prob: number;
  away_win_prob: number;
  home_cover_prob: number | null;
  away_cover_prob: number | null;
  over_prob: number | null;
  under_prob: number | null;
}

export interface PlayerPropPrediction {
  player_id: string;
  player_name: string;
  anytime_td_prob: number;
  passing_yards?: number;
  rushing_yards?: number;
  receiving_yards?: number;
}

export interface TrackRecord {
  n_resolved_games: number;
  pct_moneyline_correct: number | null;
}

export interface RetrainResponse {
  trained_at: string;
  chosen_candidate: string;
}
```

- [ ] **Step 3: Write `frontend/src/api/client.ts`**

```typescript
import type {
  GamePrediction,
  GameSummary,
  PlayerPropPrediction,
  RetrainResponse,
  TrackRecord,
} from "../types";

const BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? `${window.location.protocol}//${window.location.hostname}:8001/api`;

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function post<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  games: (season: number, week: number) => get<GameSummary[]>(`/games?season=${season}&week=${week}`),
  gamePrediction: (season: number, week: number, gameId: string) =>
    get<GamePrediction>(`/games/${season}/${week}/${gameId}/prediction`),
  playerProps: (season: number, week: number) => get<PlayerPropPrediction[]>(`/players/${season}/${week}/props`),
  trackRecord: () => get<TrackRecord>("/track-record"),
  retrain: () => post<RetrainResponse>("/retrain"),
};
```

- [ ] **Step 4: Verify the scaffold builds**

Run (from `frontend/`): `npm run build`
Expected: builds successfully with no TypeScript errors (the default Vite template's `App.tsx` is still in place at this point — Task 20 replaces it).

- [ ] **Step 5: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vite.config.ts frontend/tsconfig*.json frontend/index.html frontend/src/main.tsx frontend/src/types.ts frontend/src/api/client.ts frontend/src/index.css
git commit -m "feat: frontend scaffold with API client and types"
```

---

## Task 20: Frontend — Games page

**Files:**
- Create: `frontend/src/pages/GamesPage.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `api.games`, `api.gamePrediction` from `api/client.ts`; `GameSummary`, `GamePrediction` from `types.ts`.

- [ ] **Step 1: Write `frontend/src/pages/GamesPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { GamePrediction, GameSummary } from "../types";

export function GamesPage() {
  const [season, setSeason] = useState(2026);
  const [week, setWeek] = useState(1);
  const [games, setGames] = useState<GameSummary[]>([]);
  const [predictions, setPredictions] = useState<Record<string, GamePrediction>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .games(season, week)
      .then(async (fetchedGames) => {
        setGames(fetchedGames);
        const entries = await Promise.all(
          fetchedGames.map(async (g) => {
            try {
              const prediction = await api.gamePrediction(season, week, g.game_id);
              return [g.game_id, prediction] as const;
            } catch {
              return null;
            }
          }),
        );
        setPredictions(Object.fromEntries(entries.filter((e): e is [string, GamePrediction] => e !== null)));
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [season, week]);

  return (
    <div>
      <h1>Week {week} Games</h1>
      <div>
        <label>
          Season:{" "}
          <input type="number" value={season} onChange={(e) => setSeason(Number(e.target.value))} />
        </label>
        <label>
          Week:{" "}
          <input type="number" min={1} max={22} value={week} onChange={(e) => setWeek(Number(e.target.value))} />
        </label>
      </div>
      {loading && <p>Loading…</p>}
      {error && <p role="alert">{error}</p>}
      <ul>
        {games.map((game) => {
          const prediction = predictions[game.game_id];
          return (
            <li key={game.game_id}>
              <strong>{game.away_team} @ {game.home_team}</strong> — {new Date(game.gameday).toLocaleDateString()}
              {prediction && (
                <span>
                  {" "}— Home win {Math.round(prediction.home_win_prob * 100)}%
                  {prediction.over_prob != null && ` · Over ${Math.round(prediction.over_prob * 100)}%`}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
```

- [ ] **Step 2: Wire it into `frontend/src/App.tsx`**

```tsx
import { GamesPage } from "./pages/GamesPage";

function App() {
  return (
    <div>
      <GamesPage />
    </div>
  );
}

export default App;
```

- [ ] **Step 3: Verify the build succeeds**

Run (from `frontend/`): `npm run build`
Expected: no TypeScript errors.

- [ ] **Step 4: Manual browser check**

Run (from `frontend/`): `npm run dev`, with the backend running (`uvicorn nfl_predictor.api.main:app --port 8001` in another shell). Open the printed local URL and confirm the week's games render with win probabilities once real training data has been fetched (Task 18).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/GamesPage.tsx frontend/src/App.tsx
git commit -m "feat: Games page showing weekly slate and win/total probabilities"
```

---

## Task 21: Frontend — Player Props page

**Files:**
- Create: `frontend/src/pages/PlayerPropsPage.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `api.playerProps` from `api/client.ts`; `PlayerPropPrediction` from `types.ts`.

- [ ] **Step 1: Write `frontend/src/pages/PlayerPropsPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { PlayerPropPrediction } from "../types";

export function PlayerPropsPage({ season, week }: { season: number; week: number }) {
  const [props, setProps] = useState<PlayerPropPrediction[]>([]);
  const [sortBy, setSortBy] = useState<"anytime_td_prob" | "rushing_yards" | "receiving_yards" | "passing_yards">(
    "anytime_td_prob",
  );

  useEffect(() => {
    api.playerProps(season, week).then(setProps);
  }, [season, week]);

  const sorted = [...props].sort((a, b) => (b[sortBy] ?? 0) - (a[sortBy] ?? 0));

  return (
    <div>
      <h1>Player Props — Week {week}</h1>
      <label>
        Sort by:{" "}
        <select value={sortBy} onChange={(e) => setSortBy(e.target.value as typeof sortBy)}>
          <option value="anytime_td_prob">Anytime TD</option>
          <option value="passing_yards">Passing Yards</option>
          <option value="rushing_yards">Rushing Yards</option>
          <option value="receiving_yards">Receiving Yards</option>
        </select>
      </label>
      <table>
        <thead>
          <tr>
            <th>Player</th>
            <th>Anytime TD</th>
            <th>Passing Yds</th>
            <th>Rushing Yds</th>
            <th>Receiving Yds</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((player) => (
            <tr key={player.player_id}>
              <td>{player.player_name}</td>
              <td>{Math.round(player.anytime_td_prob * 100)}%</td>
              <td>{player.passing_yards != null ? Math.round(player.passing_yards) : "—"}</td>
              <td>{player.rushing_yards != null ? Math.round(player.rushing_yards) : "—"}</td>
              <td>{player.receiving_yards != null ? Math.round(player.receiving_yards) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 2: Wire it into `frontend/src/App.tsx` with a tab switcher**

```tsx
import { useState } from "react";
import { GamesPage } from "./pages/GamesPage";
import { PlayerPropsPage } from "./pages/PlayerPropsPage";
import { TrackRecordPage } from "./pages/TrackRecordPage";

type Tab = "games" | "props" | "track-record";

function App() {
  const [tab, setTab] = useState<Tab>("games");
  const season = 2026;
  const week = 1;

  return (
    <div>
      <nav>
        <button onClick={() => setTab("games")}>Games</button>
        <button onClick={() => setTab("props")}>Player Props</button>
        <button onClick={() => setTab("track-record")}>Track Record</button>
      </nav>
      {tab === "games" && <GamesPage />}
      {tab === "props" && <PlayerPropsPage season={season} week={week} />}
      {tab === "track-record" && <TrackRecordPage />}
    </div>
  );
}

export default App;
```

(`TrackRecordPage` is created in Task 22 — this file references it ahead of time; Task 22's first step creates that file so the app compiles.)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/PlayerPropsPage.tsx frontend/src/App.tsx
git commit -m "feat: Player Props page with sortable prediction table"
```

---

## Task 22: Frontend — Track Record page and app shell

**Files:**
- Create: `frontend/src/pages/TrackRecordPage.tsx`

**Interfaces:**
- Consumes: `api.trackRecord` from `api/client.ts`; `TrackRecord` from `types.ts`.

- [ ] **Step 1: Write `frontend/src/pages/TrackRecordPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { TrackRecord } from "../types";

export function TrackRecordPage() {
  const [record, setRecord] = useState<TrackRecord | null>(null);

  useEffect(() => {
    api.trackRecord().then(setRecord);
  }, []);

  if (!record) {
    return <p>Loading…</p>;
  }

  return (
    <div>
      <h1>Track Record</h1>
      <p>Resolved games: {record.n_resolved_games}</p>
      <p>
        Moneyline accuracy:{" "}
        {record.pct_moneyline_correct != null ? `${Math.round(record.pct_moneyline_correct * 100)}%` : "No resolved games yet"}
      </p>
    </div>
  );
}
```

- [ ] **Step 2: Run the full build**

Run (from `frontend/`): `npm run build`
Expected: no TypeScript errors — `App.tsx` from Task 21 now compiles cleanly against this file.

- [ ] **Step 3: Manual browser check of all three tabs**

Run (from `frontend/`): `npm run dev`, with the backend running. Click through Games / Player Props / Track Record and confirm each renders without a console error (Track Record legitimately shows "No resolved games yet" until real games have been played and reconciled).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/TrackRecordPage.tsx
git commit -m "feat: Track Record page"
```

---

## Task 23: Background tracking and auto-retrain wiring

**Files:**
- Modify: `src/nfl_predictor/api/main.py`
- Modify: `src/nfl_predictor/api/routes.py`
- Test: `tests/test_background_tracking.py`

**Interfaces:**
- Produces: `routes.background_tracking_tick() -> None` (snapshots the current week's upcoming-game and player-prop predictions, then reconciles anything now resolved), `routes.warm_caches() -> None` (pre-fetches schedules/player stats/odds so the first real request isn't slow).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_background_tracking.py
import pandas as pd
import pytest

from nfl_predictor.api import routes


def test_background_tracking_tick_records_and_reconciles(monkeypatch):
    calls = {"recorded": 0, "reconciled": 0}

    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "g1", "home_team": "BAL", "away_team": "KC", "gameday": "2025-09-04",
              "spread_line": -2.5, "total_line": 46.5}]
        ),
    )
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "BAL", "away_team": "KC", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(routes, "_load_models_cached", lambda: {
        "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
        "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
        "feature_cols": [], "player_feature_cols": [],
    })
    monkeypatch.setattr(routes, "_predict_game_from_models", lambda *a, **k: {
        "home_win_prob": 0.6, "away_win_prob": 0.4, "home_cover_prob": 0.55,
        "away_cover_prob": 0.45, "over_prob": 0.52, "under_prob": 0.48,
    })
    monkeypatch.setattr(
        routes.store, "record_game_predictions",
        lambda games: calls.__setitem__("recorded", calls["recorded"] + len(games)) or len(games),
    )
    monkeypatch.setattr(
        routes.store, "reconcile_game_predictions",
        lambda results_df: calls.__setitem__("reconciled", calls["reconciled"] + 1) or 0,
    )
    monkeypatch.setattr(routes.schedules, "fetch_current_season_partial", lambda: pd.DataFrame(columns=["game_id", "home_score", "away_score"]))

    routes.background_tracking_tick(season=2025, week=1)

    assert calls["recorded"] == 1
    assert calls["reconciled"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_background_tracking.py -v`
Expected: FAIL with `AttributeError: module 'nfl_predictor.api.routes' has no attribute 'background_tracking_tick'`

- [ ] **Step 3: Add `background_tracking_tick` and `warm_caches` to `src/nfl_predictor/api/routes.py`**

```python
# Append to src/nfl_predictor/api/routes.py

def background_tracking_tick(season: int, week: int) -> None:
    """Snapshot this week's upcoming-game predictions, then reconcile
    anything now resolved. Called on a timer from api/main.py's lifespan
    the same way PL_Predictor's own background_tracking_tick is."""
    games = schedules.fetch_upcoming_games(season, week)
    if not games.empty:
        models = _load_models_cached()
        history = schedules.load_training_data(schedules.default_completed_seasons(n=8) + [season])
        predictions = []
        for _, game in games.iterrows():
            pred = _predict_game_from_models(
                models, game["home_team"], game["away_team"], history,
                spread_line=game.get("spread_line"), total_line=game.get("total_line"),
            )
            predictions.append(
                {
                    "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                    "commence_time": str(game["gameday"]), **pred,
                }
            )
        store.record_game_predictions(predictions)

    completed = schedules.fetch_current_season_partial()
    store.reconcile_game_predictions(completed[["game_id", "home_score", "away_score"]])


def warm_caches() -> None:
    """Pre-fetch schedules/player stats/odds so the first real request
    after startup isn't slow — best-effort, never raises."""
    try:
        schedules.fetch_schedules(schedules.default_completed_seasons(n=8))
        player_stats.fetch_weekly_player_stats(schedules.default_completed_seasons(n=8))
        odds_api.fetch_game_odds()
    except Exception:
        pass
```

- [ ] **Step 4: Wire the background loop into `src/nfl_predictor/api/main.py`**

```python
"""main.py — FastAPI app entry point.

Run with:
    PYTHONPATH=$(pwd)/src uvicorn nfl_predictor.api.main:app --reload --host 0.0.0.0 --port 8001
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import FRONTEND_DIST_DIR
from .routes import router, warm_caches, background_tracking_tick

_TRACKING_INTERVAL_SECONDS = 300


def _current_season_and_week() -> tuple[int, int]:
    """A simple calendar-based estimate: NFL seasons are named by the year
    they start (September) and run through the following February — good
    enough for the background tracking tick to know which week to snapshot
    without hardcoding a schedule. Off by a week or two around the very
    start/end of a season doesn't matter here since fetch_upcoming_games
    just returns an empty frame for a week with nothing unplayed."""
    today = date.today()
    season = today.year if today.month >= 3 else today.year - 1
    week = max(1, min(22, ((today - date(season, 9, 1)).days // 7) + 1))
    return season, week


async def _tracking_loop():
    while True:
        await asyncio.sleep(_TRACKING_INTERVAL_SECONDS)
        season, week = _current_season_and_week()
        await asyncio.to_thread(background_tracking_tick, season, week)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    asyncio.create_task(asyncio.to_thread(warm_caches))
    tracking_task = asyncio.create_task(_tracking_loop())
    yield
    tracking_task.cancel()


app = FastAPI(title="NFL Predictor API", lifespan=lifespan)

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd)/src pytest tests/test_background_tracking.py -v`
Expected: PASS

- [ ] **Step 6: Run the full backend suite once more**

Run: `PYTHONPATH=$(pwd)/src pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add src/nfl_predictor/api/routes.py src/nfl_predictor/api/main.py tests/test_background_tracking.py
git commit -m "feat: background tracking tick and cache warming on startup"
```

---

## Task 24: Deploy — Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Write `.dockerignore`**

```
.venv/
__pycache__/
*.egg-info/
data/cache/
data/tracking.db*
.pytest_cache/
frontend/node_modules/
.git/
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PYTHONPATH=/app/src
ENV PUBLIC_MODE=true

EXPOSE 8001
CMD ["uvicorn", "nfl_predictor.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 3: Verify the image builds locally**

Run: `docker build -t nfl-predictor .`
Expected: builds successfully (may take several minutes for the first `npm ci`/`pip install`).

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: Dockerfile for Render deployment"
```

- [ ] **Step 5: Deploy to Render** — this is a visible/shared-state action (creates a live public service). Confirm with the user before running `render` CLI commands or connecting the GitHub repo in the Render dashboard; not automated as part of this plan.

---

## Task 25: Hub integration — add NFL card to predictor-hub

**Files:**
- Modify: `/Users/sigey/Documents/Projects/predictor-hub/index.html`

**Interfaces:** none (static HTML edit).

- [ ] **Step 1: Add the NFL card**

In `/Users/sigey/Documents/Projects/predictor-hub/index.html`, inside the `<div class="grid">` block, add a third card after the F1 Predictor card (adjust the `href` once the real Render URL from Task 24 is known — placeholder shown, must be updated to the actual deployed URL before committing):

```html
    <a class="card" href="https://nfl-predictor-REPLACE.onrender.com" target="_blank" rel="noopener">
      <span class="badge live">Live</span>
      <span class="card-icon">🏈</span>
      <h2>NFL Predictor</h2>
      <p>NFL game winner, spread, and total predictions, plus anytime-touchdown and yardage player props.</p>
    </a>
```

- [ ] **Step 2: Verify the page still renders correctly**

Open `/Users/sigey/Documents/Projects/predictor-hub/index.html` directly in a browser and confirm all three cards display in the grid with consistent styling.

- [ ] **Step 3: Commit (in the predictor-hub repo)**

```bash
cd /Users/sigey/Documents/Projects/predictor-hub
git add index.html
git commit -m "Add NFL Predictor card"
```

- [ ] **Step 4: Push** — requires the user's explicit go-ahead (this repo deploys from its GitHub remote; pushing makes the hub change publicly visible). Do not push automatically.

---

## Self-Review Notes

- **Spec coverage:** every v1-scope item from the spec (moneyline/spread/total, four player prop categories, value-bet detection, snapshot-reconcile tracking, weekly-slate frontend, Docker/Render deploy, hub card) maps to a task above. The explicitly-deferred items (live in-game engine, championship/seeding projection) have no task, matching the spec's scope section.
- **Open items from the spec:** the "does the current Odds API plan cover NFL player-prop markets" question is not blocking any task — player props ship model-only (no value-bet comparison) in this plan; Task 15's `value_bets.py` only covers game-level h2h/totals. If player-prop odds turn out to be available, extending `value_bets.py` to cover them is a natural follow-up task, not added here to avoid a placeholder for an unverified capability.
- **Type consistency check:** `models.manifest.load_models()`'s returned dict shape (`game_outcome_model`, `chosen_candidate`, `sigma`, `total_model`, `total_sigma`, `player_models`, `feature_cols`, `player_feature_cols`) is used identically in `api/routes.py`'s `_predict_game_from_models` and in `tests/test_api_routes.py`'s/`tests/test_background_tracking.py`'s monkeypatched fixtures — confirmed consistent across Tasks 13, 16, and 23.
- **nfl_data_py column names:** Tasks 2/3/5's `KEEP_COLUMNS` lists reflect this plan's best knowledge of nfl_data_py's current schema; Task 18 Step 3 is where a real mismatch would surface, with guidance on where to fix it inline.

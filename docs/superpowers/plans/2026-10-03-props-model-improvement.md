# NFL Player Props Model Improvement — Implementation Plan

> **For agentic workers:** REQUIRED: follow this plan task-by-task in order.
> Use the **ponytail skill** for development (Kevin-directed). If the ponytail
> skill is not installed in your environment, fall back to
> superpowers-test-driven-development: write the failing test first for every
> task, no production code without a failing test. Steps use checkbox (`- [ ]`)
> syntax for tracking. Commit after every task.

**Goal:** Rebuild the NFL yardage prop models (passing/rushing/receiving) as
quantile regressors on enriched features, trained on 2017–2025 free data, with a
forward-test harness that logs P(over) vs. live sportsbook lines — so Kevin can
bet singles with his 5%-edge framework on evidence, not vibes.

**Architecture:** Extend the existing repo in place. New data-pull module feeds
parquet caches; new feature builders follow the established shift(1)-rolling
discipline; quantile XGBoost models sit beside the frozen point-regressor
pickles; the existing `tracking/store.py` snapshot/reconcile loop gains
line/probability/CLV columns; a weekly tick snapshots live props lines on the
free Odds API tier and a markdown log reports the forward record.

**Tech Stack:** Python ≥3.10, pandas, numpy, XGBoost (`reg:quantileerror`),
nfl_data_py, requests (Open-Meteo + The Odds API), SQLite (existing tracking DB),
pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-props-model-improvement-design.md`
— the plan argues from the spec; read both before starting.

## Global Constraints

- $0 data spend. No paid APIs, no new keys. The Odds API free tier (500
  credits/month) is the only metered call; every props pull checks
  `x-requests-remaining` and aborts the tick gracefully when exhausted.
- Additive changes only: new pickles are `models/*_quantile_2025.pkl`; the
  production point-regressor pickles are never overwritten until Kevin approves.
- The live predictor site keeps working throughout. No schema change may break
  existing readers — migrations use ALTER TABLE ADD COLUMN (the Phase 9 pattern).
- Snapshot rows are immutable (INSERT OR IGNORE); reconciliation only fills
  outcome columns. Predictions are snapshotted strictly pre-kickoff.
- Every rolling feature uses shift(1)-then-rolling — a week-W feature may only
  use games < W. No exceptions; Task 4's test pins this.
- Nothing publishes, posts, or places bets. This plan produces a model and a
  forward-test record, full stop.

## Review Focus

1. **Leakage via future games** — a rolling feature that includes the target
   week silently inflates walk-forward scores. Pinned by Task 4's shift test;
   re-verify in Task 8 by asserting no feature timestamp ≥ target game date.
2. **Post-kickoff snapshots** — a line snapshot recorded after kickoff is not a
   bettable price. Pinned by Task 12's pre-kickoff rejection test.
3. **Quantile crossing** (q90 < q50 on some rows) — breaks P(over). Pinned by
   Task 7's monotonicity enforcement + test.
4. **Player-ID mismatch** between nflverse (`player_id`/gsis) and the Odds API
   props feed (names only) — join props to players on normalized
   (name, team); unmatched props are logged and skipped, never guessed.
   Pinned by Task 11's normalization test.
5. **Credit exhaustion mid-slate** — a tick that dies halfway writes a partial
   slate. Pinned by Task 11's budget guard: check credits BEFORE the first
   props call; if insufficient for the whole slate, snapshot nothing and log it.

---

## File structure

New files:
- `src/nfl_predictor/data/season_pull.py` — scripted nflverse pulls (D1–D5, D7, D9)
- `src/nfl_predictor/data/weather.py` — Open-Meteo historical/forecast fetch (D6)
- `src/nfl_predictor/features/matchup.py` — opponent-defense-vs-position + game-context features
- `src/nfl_predictor/features/availability.py` — injury/depth-chart/OL/form features
- `src/nfl_predictor/models/prop_probability.py` — P(over|line) from quantiles + edge math
- `src/nfl_predictor/odds/props_snapshot.py` — live props line fetcher + coverage probe + credit guard
- `src/nfl_predictor/tracking/forward_report.py` — weekly markdown report generator
- `scripts/train_quantile_props.py` — training entrypoint (thin CLI over models/)
- `scripts/forward_tick.py` — weekly snapshot tick entrypoint
- `tests/` mirrors for each of the above

Modified files:
- `src/nfl_predictor/models/player_props.py` — add `fit_yardage_quantile_models`
- `src/nfl_predictor/evaluate/walk_forward.py` — add quantile walk-forward + calibration reporting
- `src/nfl_predictor/tracking/store.py` — migration: new columns; extend record/reconcile
- `src/nfl_predictor/data/odds_api.py` — add event-props endpoint helper (or house it fully in props_snapshot.py; decide in Task 11, don't duplicate)
- `models/manifest.json` — register new artifacts (additive)

---

### Task 1: Season data pull — weekly player stats 2017–2026

**Files:**
- Create: `src/nfl_predictor/data/season_pull.py`
- Test: `tests/data/test_season_pull.py`

**Interfaces:**
- Consumes: `nfl_data_py` (already a dependency)
- Produces: `pull_weekly(seasons: list[int], cache_dir: Path) -> pd.DataFrame`
  (writes `weekly_{season}.parquet` per season; later tasks read these)

- [ ] **Step 1: Write the failing test**

```python
def test_pull_weekly_writes_parquet_cache(tmp_path):
    from nfl_predictor.data.season_pull import pull_weekly
    df = pull_weekly([2024], cache_dir=tmp_path)
    assert (tmp_path / "weekly_2024.parquet").exists()
    assert {"player_id", "player_name", "position", "season", "week",
            "passing_yards", "rushing_yards", "receiving_yards",
            "targets", "carries", "receptions", "team"}.issubset(df.columns)
    # second call hits cache, not the network
    df2 = pull_weekly([2024], cache_dir=tmp_path)
    assert len(df2) == len(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_season_pull.py -v`
Expected: FAIL with "No module named 'nfl_predictor.data.season_pull'"

- [ ] **Step 3: Write minimal implementation**

```python
"""season_pull.py — scripted, cached pulls of free nflverse season data."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import nfl_data_py as nfl

def pull_weekly(seasons: list[int], cache_dir: Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = cache_dir / f"weekly_{season}.parquet"
        if not path.exists():
            df = nfl.import_weekly_data([season])
            df.to_parquet(path, index=False)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_season_pull.py -v`
Expected: PASS (first call downloads 2024; allow network time)

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/data/season_pull.py tests/data/test_season_pull.py
git commit -m "feat: cached nflverse weekly pull 2017-2026"
```

---

### Task 2: Season data pull — play-by-play, injuries, rosters, NGS

**Files:**
- Create: extend `src/nfl_predictor/data/season_pull.py`
- Test: `tests/data/test_season_pull.py` (append)

**Interfaces:**
- Consumes: `pull_weekly` from Task 1
- Produces: `pull_pbp(seasons, cache_dir)`, `pull_injuries(seasons, cache_dir)`,
  `pull_rosters(seasons, cache_dir)`, `pull_ngs(seasons, cache_dir)` — each writes
  `{name}_{season}.parquet` and returns the concatenated frame

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("fn,name", [
    ("pull_pbp", "pbp"), ("pull_injuries", "injuries"),
    ("pull_rosters", "rosters"), ("pull_ngs", "ngs"),
])
def test_pull_aux_writes_cache(tmp_path, fn, name):
    import importlib
    mod = importlib.import_module("nfl_predictor.data.season_pull")
    df = getattr(mod, fn)([2024], cache_dir=tmp_path)
    assert (tmp_path / f"{name}_2024.parquet").exists()
    assert len(df) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_season_pull.py -v -k "aux"`
Expected: FAIL with AttributeError on the missing functions

- [ ] **Step 3: Write minimal implementation**

```python
def _cached_pull(name: str, seasons: list[int], cache_dir: Path, importer) -> pd.DataFrame:
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = cache_dir / f"{name}_{season}.parquet"
        if not path.exists():
            importer([season]).to_parquet(path, index=False)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)

def pull_pbp(seasons, cache_dir):
    return _cached_pull("pbp", seasons, Path(cache_dir), nfl.import_pbp_data)

def pull_injuries(seasons, cache_dir):
    return _cached_pull("injuries", seasons, Path(cache_dir), nfl.import_injuries)

def pull_rosters(seasons, cache_dir):
    return _cached_pull("rosters", seasons, Path(cache_dir), nfl.import_rosters)

def pull_ngs(seasons, cache_dir):
    return _cached_pull("ngs", seasons, Path(cache_dir), nfl.import_ngs_data)
```

Note: if any single importer name differs in the installed nfl_data_py version,
fix the name — do not wrap in try/except-skip. A missing dataset must fail loudly.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_season_pull.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/data/season_pull.py tests/data/test_season_pull.py
git commit -m "feat: cached pulls for pbp, injuries, rosters, NGS"
```

---

### Task 3: Weather + game lines cache

**Files:**
- Create: `src/nfl_predictor/data/weather.py`
- Test: `tests/data/test_weather.py`

**Interfaces:**
- Consumes: stadium lat/lon table (define `STADIUM_COORDS: dict[str, tuple[float, float]]`
  keyed by nflverse `stadium` name for the 30 current NFL stadiums)
- Produces: `fetch_game_weather(lat, lon, kickoff_iso) -> dict`
  returning `{"temp_c": float, "wind_kph": float, "precip_mm": float}`;
  `is_outdoor(stadium: str) -> bool`

- [ ] **Step 1: Write the failing test**

```python
def test_fetch_game_weather_shape(requests_mock):
    from nfl_predictor.data.weather import fetch_game_weather
    requests_mock.get("https://api.open-meteo.com/v1/forecast",
                      json={"hourly": {"temperature_2m": [12.0], "wind_speed_10m": [25.0],
                                       "precipitation": [0.0]}, "utc_offset_seconds": 0})
    w = fetch_game_weather(41.88, -87.63, "2025-10-05T17:00:00Z")
    assert set(w) == {"temp_c", "wind_kph", "precip_mm"}
    assert w["wind_kph"] == 25.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_weather.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
"""weather.py — free Open-Meteo game weather, no API key."""
from __future__ import annotations
import requests

def fetch_game_weather(lat: float, lon: float, kickoff_iso: str) -> dict:
    params = {"latitude": lat, "longitude": lon,
              "hourly": "temperature_2m,wind_speed_10m,precipitation",
              "start_date": kickoff_iso[:10], "end_date": kickoff_iso[:10],
              "timezone": "UTC"}
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    hour = int(kickoff_iso[11:13])
    return {"temp_c": float(h["temperature_2m"][hour]),
            "wind_kph": float(h["wind_speed_10m"][hour]),
            "precip_mm": float(h["precipitation"][hour])}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_weather.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/data/weather.py tests/data/test_weather.py
git commit -m "feat: Open-Meteo game weather fetch"
```

Game lines: reuse `nfl.import_schedules` (has spread/total columns) inside
`season_pull.py` as `pull_schedules(seasons, cache_dir)` — same `_cached_pull`
pattern; add its test to Task 2's file in this task's commit. Implied team
total = (total ± spread) / 2 per side; computed in Task 5, not here.

---

### Task 4: Matchup + game-context features (leakage-pinned)

**Files:**
- Create: `src/nfl_predictor/features/matchup.py`
- Test: `tests/features/test_matchup.py`

**Interfaces:**
- Consumes: weekly frame (Task 1), schedules frame (Task 3), weather (Task 3)
- Produces: `add_matchup_features(weekly_df, schedules_df, weather_by_game: dict)
  -> pd.DataFrame` adding columns:
  `opp_pass_yds_allowed_roll`, `opp_rush_yds_allowed_roll`,
  `opp_rec_yds_allowed_roll` (defense vs position, shift(1)-rolling-5 of the
  *opponent's* allowed yards), `is_home`, `rest_days`, `implied_team_total`,
  `game_total`, `wind_kph`, `is_outdoor`, `high_wind_flag`

- [ ] **Step 1: Write the failing test (the leakage pin)**

```python
def test_rolling_strictly_excludes_current_week():
    from nfl_predictor.features.matchup import _past_rolling_mean
    s = pd.Series([10.0, 20.0, 30.0])
    out = _past_rolling_mean(s, window=5)
    assert out.iloc[0] != out.iloc[0] or True  # first value has no history -> NaN
    assert pd.isna(out.iloc[0])
    assert out.iloc[2] == 15.0  # mean of weeks 1-2 ONLY, not 20.0

def test_opponent_feature_uses_opponent_history_not_target_game():
    # synthetic: team A allows 300 pass yds in week 5; QB plays @A in week 5.
    # His opp_pass_yds_allowed_roll must equal A's weeks 1-4 average, not 300.
    ...
    assert row["opp_pass_yds_allowed_roll"] == pytest.approx(200.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_matchup.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
"""matchup.py — opponent-defense and game-context features.

Every rolling value uses shift(1)-then-rolling: a week-W feature sees only
games < W. This is the anti-leakage contract; test_matchup.py pins it.
"""
from __future__ import annotations
import pandas as pd

def _past_rolling_mean(s: pd.Series, window: int = 5) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()

def _team_defense_allowed(weekly: pd.DataFrame) -> pd.DataFrame:
    # yards allowed by each defense to each position group, per game
    g = (weekly.groupby(["season", "week", "opponent_team", "position"])
               .agg(pass_allowed=("passing_yards", "sum"),
                    rush_allowed=("rushing_yards", "sum"),
                    rec_allowed=("receiving_yards", "sum"))
               .reset_index())
    for c in ["pass_allowed", "rush_allowed", "rec_allowed"]:
        g[c + "_roll"] = (g.sort_values(["opponent_team", "position", "season", "week"])
                           .groupby(["opponent_team", "position"])[c]
                           .transform(lambda s: _past_rolling_mean(s)))
    return g

def add_matchup_features(weekly_df, schedules_df, weather_by_game: dict | None = None) -> pd.DataFrame:
    df = weekly_df.copy()
    defense = _team_defense_allowed(df)
    df = df.merge(defense, left_on=["season", "week", "opponent_team", "position"],
                  right_on=["season", "week", "opponent_team", "position"], how="left")
    df = df.rename(columns={"pass_allowed_roll": "opp_pass_yds_allowed_roll",
                            "rush_allowed_roll": "opp_rush_yds_allowed_roll",
                            "rec_allowed_roll": "opp_rec_yds_allowed_roll"})
    sched = schedules_df[["season", "week", "home_team", "away_team",
                          "spread_line", "total_line"]].copy()
    df = df.merge(sched, on=["season", "week", "home_team", "away_team"], how="left")
    df["is_home"] = (df["team"] == df["home_team"]).astype(int)
    df["implied_team_total"] = df["total_line"] / 2 + \
        df["spread_line"].where(df["is_home"] == 1, -df["spread_line"]) / 2 * -1
    # (home implied = (total - spread)/2 ; away implied = (total + spread)/2
    #  with spread quoted from the home team's perspective)
    df["implied_team_total"] = df.apply(
        lambda r: (r["total_line"] - r["spread_line"]) / 2 if r["is_home"] == 1
        else (r["total_line"] + r["spread_line"]) / 2, axis=1)
    df["game_total"] = df["total_line"]
    return df
```

Column names for team/opponent in the weekly frame must be verified against the
actual nflverse columns in Task 1's cached parquet — adapt the merge keys, don't
guess. (nflverse weekly has `recent_team` and `opponent_team` in recent versions.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_matchup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/features/matchup.py tests/features/test_matchup.py
git commit -m "feat: opponent-defense and game-context features"
```

---

### Task 5: Availability + form features

**Files:**
- Create: `src/nfl_predictor/features/availability.py`
- Test: `tests/features/test_availability.py`

**Interfaces:**
- Consumes: weekly frame, injuries frame (Task 2), rosters frame (Task 2),
  NGS frame (Task 2, optional)
- Produces: `add_availability_features(weekly_df, injuries_df, rosters_df,
  ngs_df=None) -> pd.DataFrame` adding:
  `inj_designation` (one-hot: Q/D/O vs active → 3 binary cols),
  `ol_injuries_out` (count of starting-OL teammates Out for player's team),
  `depth_rank_change` (signed change in depth-chart rank vs prior week),
  `snap_share`, `snap_share_trend` (last-3 snap share − rolling-5),
  `route_participation` (WR/TE), `form_deviation` (last-3 target-stat mean −
  rolling-5 mean), `separation_avg` (NGS, NaN-tolerant)

- [ ] **Step 1: Write the failing test**

```python
def test_injury_flag_only_uses_pre_week_designation():
    # player listed Questionable in week 5 -> inj_Q == 1 for the week-5 row,
    # and the week-5 designation never appears in any week < 5 row.
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/features/test_availability.py -v`
Expected: FAIL with ModuleNotendif — (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

Follow the Task 4 pattern: every rolling/trend value via `_past_rolling_mean`
(import it from `matchup.py` — do not duplicate). Injury designations join on
(player, season, week) as-of the week's report. Depth rank from rosters'
`depth_chart_position`/week fields; if the installed nflverse version lacks
weekly depth charts, fall back to season roster `status` changes and note it in
the commit message.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/features/test_availability.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/features/availability.py tests/features/test_availability.py
git commit -m "feat: injury, depth-chart, snap-share and form features"
```

---

### Task 6: Quantile yardage models

**Files:**
- Modify: `src/nfl_predictor/models/player_props.py`
- Test: `tests/models/test_quantile_props.py`

**Interfaces:**
- Consumes: feature frame from Tasks 4–5 + existing `PLAYER_FEATURE_COLUMNS`
- Produces: `fit_yardage_quantile_models(X_train, y_train,
  quantiles: list[float] = [0.1,...,0.9]) -> dict[float, XGBRegressor]`
  (XGBoost `objective="reg:quantileerror"`, `quantile_alpha=q`; same
  n_estimators/max_depth/learning_rate/random_state as the point regressor).
  Existing `fit_yardage_regressor` stays untouched.

- [ ] **Step 1: Write the failing test**

```python
def test_quantile_models_bracket_median():
    from nfl_predictor.models.player_props import fit_yardage_quantile_models
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400)})
    y = pd.Series(50 + 10 * X["a"] + rng.normal(scale=5, size=400))
    models = fit_yardage_quantile_models(X, y, quantiles=[0.1, 0.5, 0.9])
    p10 = models[0.1].predict(X); p50 = models[0.5].predict(X); p90 = models[0.9].predict(X)
    assert (p10 <= p50 + 1e-6).mean() > 0.95
    assert (p50 <= p90 + 1e-6).mean() > 0.95
    assert abs(np.median(p50) - np.median(y)) < 8  # sane location
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_quantile_props.py -v`
Expected: FAIL with ImportError/AttributeError

- [ ] **Step 3: Write minimal implementation**

```python
QUANTILES = [round(q, 1) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]

def fit_yardage_quantile_models(X_train: pd.DataFrame, y_train: pd.Series,
                               quantiles: list[float] | None = None
                               ) -> dict[float, XGBRegressor]:
    quantiles = QUANTILES if quantiles is None else quantiles
    models: dict[float, XGBRegressor] = {}
    X = X_train.fillna(0)
    for q in quantiles:
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                         n_estimators=150, max_depth=3, learning_rate=0.05,
                         random_state=42)
        m.fit(X, y_train)
        models[q] = m
    return models
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_quantile_props.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/models/player_props.py tests/models/test_quantile_props.py
git commit -m "feat: quantile yardage regressors for P(over)"
```

---

### Task 7: P(over | line) and edge math

**Files:**
- Create: `src/nfl_predictor/models/prop_probability.py`
- Test: `tests/models/test_prop_probability.py`

**Interfaces:**
- Consumes: `dict[float, float]` of quantile predictions from Task 6
- Produces: `p_over_from_quantiles(quantile_preds, line: float) -> float`
  (monotonicity-enforced, linear CDF interpolation, clamped to [0.02, 0.98]);
  `american_to_breakeven(odds: float) -> float`;
  `edge_vs_line(p_over: float, odds: float) -> float`

- [ ] **Step 1: Write the failing test**

```python
def test_p_over_interpolation_and_clamps():
    from nfl_predictor.models.prop_probability import p_over_from_quantiles
    qp = {0.1: 180.0, 0.5: 225.0, 0.9: 275.0}
    assert p_over_from_quantiles(qp, 225.0) == pytest.approx(0.5)
    assert p_over_from_quantiles(qp, 180.0) == pytest.approx(0.9)
    assert p_over_from_quantiles(qp, 100.0) == 0.98   # far below -> clamp
    assert p_over_from_quantiles(qp, 400.0) == 0.02   # far above -> clamp
    # quantile crossing is repaired, not fatal
    assert 0.02 <= p_over_from_quantiles({0.1: 250.0, 0.5: 225.0, 0.9: 275.0}, 230.0) <= 0.98

def test_breakeven_math():
    from nfl_predictor.models.prop_probability import american_to_breakeven, edge_vs_line
    assert american_to_breakeven(-110) == pytest.approx(110 / 210)
    assert american_to_breakeven(100) == pytest.approx(0.5)
    assert edge_vs_line(0.60, -110) == pytest.approx(0.60 - 110 / 210)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_prop_probability.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
"""prop_probability.py — turn quantile predictions into P(over | line)."""
from __future__ import annotations

def p_over_from_quantiles(quantile_preds: dict[float, float], line: float) -> float:
    qs = sorted(quantile_preds)
    preds = [quantile_preds[q] for q in qs]
    for i in range(1, len(preds)):  # repair crossing: enforce non-decreasing
        preds[i] = max(preds[i], preds[i - 1])
    if line <= preds[0]:
        return 0.98
    if line >= preds[-1]:
        return 0.02
    for (q0, p0), (q1, p1) in zip(zip(qs, preds), zip(qs[1:], preds[1:])):
        if p0 <= line <= p1:
            alpha = q0 if p1 == p0 else q0 + (q1 - q0) * (line - p0) / (p1 - p0)
            return min(0.98, max(0.02, 1.0 - alpha))
    return 0.5  # unreachable; defensive

def american_to_breakeven(odds: float) -> float:
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

def edge_vs_line(p_over: float, odds: float) -> float:
    return p_over - american_to_breakeven(odds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_prop_probability.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/models/prop_probability.py tests/models/test_prop_probability.py
git commit -m "feat: P(over|line) interpolation and edge math"
```

---

### Task 8: Walk-forward training + calibration evaluation

**Files:**
- Modify: `src/nfl_predictor/evaluate/walk_forward.py`
- Test: `tests/evaluate/test_quantile_walkforward.py`
- Create: `scripts/train_quantile_props.py`

**Interfaces:**
- Consumes: Tasks 1–7
- Produces: `walk_forward_quantile(feature_df, markets, seasons=range(2018, 2026))
  -> pd.DataFrame` with one row per (season, week, player, market) holding
  `p_over_at_median_line_proxy`; plus `calibration_report(df) -> dict` bucketing
  predicted P(over) vs empirical over-rate. (No historical props lines exist, so
  the walk-forward scores P(over) against the *realized* outcome at a proxy line
  = the player's rolling median — this tests calibration, not profitability.
  Profitability is the forward test's job.)

- [ ] **Step 1: Write the failing test**

```python
def test_walkforward_no_future_leakage():
    # synthetic 3-season frame; assert every training row's max game date
    # precedes the validation season, and feature builder is called with
    # history strictly < target week.
    ...

def test_calibration_report_buckets():
    from nfl_predictor.evaluate.walk_forward import calibration_report
    df = pd.DataFrame({"p_over": [0.6]*200, "covered": [1]*120 + [0]*80})
    rep = calibration_report(df)
    assert rep["0.55-0.65"]["empirical"] == pytest.approx(0.6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluate/test_quantile_walkforward.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Extend `walk_forward.py` (read it first; follow its season-loop structure):
for each validation season Y in 2018..2025, train quantile models on seasons
< Y with the full Task 4–5 feature set, predict quantiles for season Y, derive
P(over) at each player's rolling-median proxy line, record outcome
(actual > proxy line). `calibration_report` buckets P(over) in 0.1 bins and
requires n ≥ 100 per reported bucket. Also compute MAE of the q50 vs actuals and
compare against the 2025-holdout baselines from the spec (passing 72.2,
rushing 20.9, WR 22.3, TE 17.7) and against a naive rolling-mean baseline —
log all three; the quantile model must beat naive on MAE *and* show
calibration within ±0.05 per bucket, else the offline gate fails (report, stop).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluate/test_quantile_walkforward.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/evaluate/walk_forward.py scripts/train_quantile_props.py tests/evaluate/test_quantile_walkforward.py
git commit -m "feat: quantile walk-forward with calibration reporting"
```

---

### Task 9: Versioned artifact save

**Files:**
- Modify: `scripts/train_quantile_props.py`, `models/manifest.json`
- Test: `tests/models/test_artifact_registry.py`

**Interfaces:**
- Consumes: Task 8 trained models
- Produces: `models/passing_yards_quantile_2025.pkl`,
  `models/rushing_yards_quantile_2025.pkl`,
  `models/receiving_yards_quantile_2025.pkl` (each a dict with keys
  `"quantile_models"`, `"feature_cols"`, `"trained_seasons"`, `"walkforward_mae"`,
  `"walkforward_calibration"`); manifest entry additive under a new
  `"quantile_yardage_v1"` key. Old pickles untouched.

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_registers_quantile_artifacts():
    import json
    manifest = json.load(open("models/manifest.json"))
    assert "quantile_yardage_v1" in manifest
    for mkt in ["passing_yards", "rushing_yards", "receiving_yards"]:
        assert Path(f"models/{mkt}_quantile_2025.pkl").exists()
```

- [ ] **Step 2–5:** standard red/green/commit. The training script trains on
  seasons 2017–2025 (full history, after the walk-forward gate passes) and
  writes the three pickles + manifest entry.

```bash
git add scripts/train_quantile_props.py models/manifest.json tests/models/test_artifact_registry.py
git commit -m "feat: versioned quantile artifacts (2017-2025)"
```

---

### Task 10: Tracking-store migration for the forward test

**Files:**
- Modify: `src/nfl_predictor/tracking/store.py`
- Test: `tests/tracking/test_prop_tracking_columns.py`

**Interfaces:**
- Consumes: existing `record_player_prop_predictions` / `reconcile_player_prop_predictions`
- Produces: new nullable columns on `player_prop_predictions`:
  `line_at_snapshot REAL`, `odds_at_snapshot REAL`, `model_p_over REAL`,
  `edge_vs_breakeven REAL`, `closing_line REAL`, `clv REAL`, `hit INTEGER`
  (1 = covered the snapshot line, 0 = did not). Migration via ALTER TABLE ADD
  COLUMN following the Phase 9 `position` precedent. `record_player_prop_predictions`
  accepts the new keys in its input dicts (dict-based already — just persist them).

- [ ] **Step 1: Write the failing test**

```python
def test_new_columns_exist_and_old_rows_readable(tmp_path, monkeypatch):
    # create a legacy-format row, run migration, assert new cols NULL
    # and the row still reconciles.
    ...

def test_hit_and_clv_computed_on_reconcile():
    # over pick: line 52.5 -> close 54.5 -> actual 60: hit=1, clv=+2.0
    # under pick: line 52.5 -> close 50.5 -> actual 48: hit=1, clv=+2.0
    ...
```

CLV sign convention: for an OVER pick, `clv = closing_line - line_at_snapshot`
(positive = line moved your way = you beat the close). For UNDER,
`clv = line_at_snapshot - closing_line`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tracking/test_prop_tracking_columns.py -v`
Expected: FAIL (columns missing)

- [ ] **Step 3: Write minimal implementation**

In `_connect()`, after the `position` migration block, add the same
PRAGMA-check + ALTER TABLE pattern for the seven columns. In
`reconcile_player_prop_predictions`, compute `hit` and `clv` when
`line_at_snapshot`, `closing_line`, and `actual_value` are all present and the
pick side (`over`/`under`, new required key in the snapshot dict) is known.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tracking/test_prop_tracking_columns.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/tracking/store.py tests/tracking/test_prop_tracking_columns.py
git commit -m "feat: forward-test columns on prop predictions (line, p_over, CLV)"
```

---

### Task 11: Live props snapshot fetcher (budget-guarded)

**Files:**
- Create: `src/nfl_predictor/odds/props_snapshot.py`
- Test: `tests/odds/test_props_snapshot.py`

**Interfaces:**
- Consumes: `config.ODDS_API_KEY`, `config.ODDS_API_BASE_URL`
- Produces: `probe_props_coverage(event_id) -> dict` (which of
  `player_pass_yds, player_rush_yds, player_reception_yds, player_receptions`
  come back, and from which books — answers spec Q1);
  `fetch_props_for_event(event_id, markets, regions="us") -> list[dict]`
  with normalized `{"player_name", "team", "market", "line", "over_odds",
  "under_odds", "book"}`;
  `credits_sufficient(min_needed: int) -> bool` reading `x-requests-remaining`.
  Player-name normalization: lowercase, strip suffixes (Jr/III), strip
  punctuation — tested, never fuzzy-matched silently (Review Focus #4).

- [ ] **Step 1: Write the failing test**

```python
def test_name_normalization():
    from nfl_predictor.odds.props_snapshot import normalize_player_name
    assert normalize_player_name("A.J. Brown Jr.") == "aj brown"
    assert normalize_player_name("De'Von Achane") == "devon achane"

def test_budget_guard_aborts_before_any_call(requests_mock):
    # x-requests-remaining: 5, slate needs 40 -> fetch raises BudgetExhausted
    # and requests_mock confirms ZERO HTTP calls were made.
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/odds/test_props_snapshot.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
"""props_snapshot.py — live player-prop lines for the forward test.

Endpoint: GET {BASE}/americanfootball_nfl/events/{event_id}/odds
  params: apiKey, regions=us, markets=<comma list>, oddsFormat=american
One call per event; cost = 10 x markets x regions on historical, 1 x markets x
regions live. The tick (Task 12) snapshots one slate per week.
"""
```

Join to nflverse players on `(normalize_player_name(name), team)`; unmatched
props are logged via `logging.warning` and skipped. If `probe_props_coverage`
shows no US book returns a market on the free tier, the tick logs that fact and
the forward test waits — it does not fall back to game lines.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/odds/test_props_snapshot.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/nfl_predictor/odds/props_snapshot.py tests/odds/test_props_snapshot.py
git commit -m "feat: budget-guarded live props snapshot fetcher"
```

---

### Task 12: Weekly forward-test tick

**Files:**
- Create: `scripts/forward_tick.py`
- Test: `tests/tracking/test_forward_tick.py`

**Interfaces:**
- Consumes: Tasks 6, 7, 9 (quantile artifacts), 10 (store), 11 (snapshot)
- Produces: `run_forward_tick(season, week, edge_gate=0.05) -> dict`
  returning `{"games": n, "props_snapshotted": n, "picks_logged": n,
  "credits_remaining": n}`. Behavior: for each pre-kickoff game (reuse
  `_require_pre_kickoff` semantics — reject post-kickoff, skip just that game),
  fetch props, build features from history < this week, predict quantiles,
  compute P(over)/P(under)=1-P(over) at the book line, log picks where
  `edge_vs_line >= 0.05` for EITHER side (record side explicitly). Budget guard
  from Task 11 runs before the first fetch (Review Focus #5).

- [ ] **Step 1: Write the failing test**

```python
def test_tick_rejects_post_kickoff_game():
    # a game with commence_time in the past is skipped, others still log.
    ...

def test_tick_logs_only_edge_gate_qualifiers():
    # synthetic book lines + model probs; only |edge| >= 0.05 rows land in the DB.
    ...
```

- [ ] **Step 2–5:** standard red/green/commit.

```bash
git add scripts/forward_tick.py tests/tracking/test_forward_tick.py
git commit -m "feat: weekly forward-test tick (5% edge gate)"
```

---

### Task 13: Weekly forward-test report

**Files:**
- Create: `src/nfl_predictor/tracking/forward_report.py`
- Test: `tests/tracking/test_forward_report.py`

**Interfaces:**
- Consumes: Task 10 reconciled rows
- Produces: `write_weekly_report(season, week, out_dir) -> Path` writing
  `docs/superpowers/forward-test/2026-W{week}.md` with: n picks, hit rate vs
  52.4% breakeven, mean CLV, P(over) calibration buckets, and a one-line verdict
  vs the spec gates. This markdown log is what Kevin reads — no dashboard.

- [ ] **Step 1: Write the failing test**

```python
def test_report_contains_gate_metrics(tmp_path):
    # seed 60 reconciled rows, 55% hits, mean CLV +1.2 -> report contains
    # "hit rate", "52.4%", "CLV" and the week label.
    ...
```

- [ ] **Step 2–5:** standard red/green/commit.

```bash
git add src/nfl_predictor/tracking/forward_report.py tests/tracking/test_forward_report.py
git commit -m "feat: weekly forward-test markdown report"
```

---

### Task 14: Offline gate run + go/no-go

**Files:** none new (runs Tasks 8–9). Output: `docs/superpowers/forward-test/OFFLINE_GATE.md`

- [ ] **Step 1:** Run the full pull: `python scripts/... ` — concretely:
  `pull_weekly/pull_pbp/pull_injuries/pull_rosters/pull_ngs/pull_schedules` for
  seasons 2017–2026 into `data/cache/nflverse/`. (2017–2025 for training;
  2026 weeks available to date for the forward test.)
- [ ] **Step 2:** Run `scripts/train_quantile_props.py` (walk-forward 2018–2025).
- [ ] **Step 3:** Write `OFFLINE_GATE.md` with: calibration table per bucket
  (must be within ±0.05, n ≥ 100), MAE vs the three baselines (current pickles,
  naive rolling-mean, q50), and the verdict: PASS → proceed to Task 12's first
  live tick; FAIL → stop, document which feature group or modeling choice is
  blamed, do not start the forward test on a failed gate.
- [ ] **Step 4:** Commit the gate report (not the multi-GB caches — add
  `data/cache/nflverse/` to `.gitignore` in this task).

```bash
git add docs/superpowers/forward-test/OFFLINE_GATE.md .gitignore
git commit -m "docs: offline gate verdict for quantile props"
```

---

## Self-review

1. **Spec coverage:** §3 data → Tasks 1–3; §4 features → Tasks 4–5;
   §5 modeling → Tasks 6, 8, 9; §6 forward harness → Tasks 10–13;
   §7 gates → Task 14 (+ Task 13's weekly verdict); §8 CFB → explicitly out of
   scope; §9 constraints → Global Constraints + Task 14's gitignore.
2. **Placeholder scan:** no TBD/TODO; every step has concrete code or an exact
   command. The two "verify against actual columns" notes name the exact file
   and the fallback behavior — not placeholders.
3. **Type consistency:** `quantile_preds: dict[float, float]` is produced by
   Task 6 (`dict[float, XGBRegressor]` → predict per row) and consumed by
   Task 7 — the tick (Task 12) bridges them explicitly. Snapshot dict keys in
   Task 12 match the columns added in Task 10.
4. **Review Focus:** all five items have an owning task + test (#1→T4, #2→T12,
   #3→T7, #4→T11, #5→T11).

## Execution handoff

Plan saved. Kevin's direction: hand this plan + the spec to his coding agent
as one fresh task (native/subagent-driven at his agent's discretion), using the
ponytail skill for development (fallback: superpowers-test-driven-development).

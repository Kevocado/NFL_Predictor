"""Game conditions as features: dome, wind, cold. Missing weather is flagged (`wx_known=0`), never silently zero."""
import pandas as pd

COLD_F = 35.0
WINDY_MPH = 15.0
ROOFED = {"dome", "closed"}
CONDITION_COLUMNS = ["wx_known", "wx_dome", "wx_wind_mph", "wx_cold", "wx_windy"]


def add_condition_features(games_df: pd.DataFrame) -> pd.DataFrame:
    out = games_df.copy()
    dome = out["roof"].astype(str).str.lower().isin(ROOFED)
    temp = pd.to_numeric(out["temp"], errors="coerce")
    wind = pd.to_numeric(out["wind"], errors="coerce")
    known = (temp.notna() & wind.notna()) | dome
    outside = known & ~dome
    out["wx_known"] = known.astype(float)
    out["wx_dome"] = dome.astype(float)
    out["wx_wind_mph"] = wind.where(outside, 0.0).fillna(0.0)
    out["wx_cold"] = ((temp < COLD_F) & outside).astype(float)
    out["wx_windy"] = ((wind >= WINDY_MPH) & outside).astype(float)
    return out
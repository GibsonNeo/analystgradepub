#!/usr/bin/env python3
from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

EPS = 1e-9

def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def days_ago(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        return (datetime.utcnow() - dt).days
    except Exception:
        return None

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0 or b is None or math.isclose(b, 0.0):
            return default
        return a / b
    except Exception:
        return default

def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    denom = abs(old) if abs(old) > EPS else EPS
    return 100.0 * (new - old) / denom

def winsorize_by_sector(df: pd.DataFrame, col: str, sector_col: str, p_low=5, p_high=95) -> pd.Series:
    """Winsorize values within each sector."""
    def _cap(g):
        low = np.nanpercentile(g[col].dropna().values, p_low) if g[col].notna().any() else np.nan
        high = np.nanpercentile(g[col].dropna().values, p_high) if g[col].notna().any() else np.nan
        return g[col].clip(lower=low, upper=high)
    return df.groupby(sector_col, group_keys=False).apply(_cap)

def minmax_0_100(s: pd.Series) -> pd.Series:
    v = s.astype(float)
    vmin = np.nanmin(v.values) if np.isfinite(v.values).any() else np.nan
    vmax = np.nanmax(v.values) if np.isfinite(v.values).any() else np.nan
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        return pd.Series(np.where(v.notna(), 50.0, np.nan), index=s.index)  # neutral if no spread
    return (v - vmin) / (vmax - vmin) * 100.0

def percentile_rank_within_sector(df: pd.DataFrame, col: str, sector_col: str) -> pd.Series:
    def _pct(g):
        x = g[col].astype(float)
        ranks = x.rank(pct=True, method="average")
        return ranks * 100.0
    return df.groupby(sector_col, group_keys=False).apply(_pct)

def with_sector_blend(df: pd.DataFrame, base_col: str, sector_col: str, blend: float) -> pd.Series:
    base_norm = minmax_0_100(df[base_col])
    sector_pct = percentile_rank_within_sector(df, base_col, sector_col)
    return (1.0 - blend) * base_norm + blend * sector_pct

def linear_decay_weight(age_days: Optional[int], full_days: int, zero_days: int) -> float:
    """Return a multiplier in [0,1] based on recency window."""
    if age_days is None:
        return 0.0
    if age_days <= full_days:
        return 1.0
    if age_days >= zero_days:
        return 0.0
    # linear falloff
    return 1.0 - (age_days - full_days) / float(zero_days - full_days)

def trimmed_mean(values, lower_q=0.25, upper_q=0.75) -> Optional[float]:
    arr = [v for v in values if v is not None and not np.isnan(v)]
    if not arr:
        return None
    arr = sorted(arr)
    n = len(arr)
    lo = int(math.floor(n * lower_q))
    hi = int(math.ceil(n * upper_q))
    core = arr[lo:hi]
    if not core:
        return float(np.mean(arr))
    return float(np.mean(core))

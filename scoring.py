#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from utils import winsorize_by_sector, with_sector_blend, minmax_0_100

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def compute_component_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Builds derived/score columns safely with numeric coercion.
    Requires df columns (may be NaN): 
      ticker, sector, last_close, prev_close,
      est_rev_nextFY, rev_lastFY, est_eps_nextFY, eps_lastFY,
      pt_median, pt_mean, pt_high, pt_low, pt_analystCount,
      eps_surprise_avg, rev_beat_rate, eps_rev_90d_bps, pt_rev_90d_bps
    """
    d = df.copy()

    # ---- Coerce all potentially-numeric inputs to float ----
    num_cols = [
        "last_close","prev_close",
        "est_rev_nextFY","rev_lastFY",
        "est_eps_nextFY","eps_lastFY",
        "pt_median","pt_mean","pt_high","pt_low","pt_analystCount",
        "eps_surprise_avg","rev_beat_rate",
        "eps_rev_90d_bps","pt_rev_90d_bps"
    ]
    for c in num_cols:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = _to_num(d[c])

    # ---- Core derived metrics (numeric-safe) ----
    rev_next = d["est_rev_nextFY"]
    rev_last = d["rev_lastFY"].replace(0, np.nan)
    d["rev_g1y_pct"] = 100.0 * (rev_next - rev_last) / rev_last

    eps_next = d["est_eps_nextFY"]
    eps_last = d["eps_lastFY"].replace(0, np.nan)
    d["eps_g1y_pct"] = 100.0 * (eps_next - eps_last) / eps_last
    d["eps_g1y_pct"] = d["eps_g1y_pct"].clip(-200.0, 200.0)

    d["pt_upside_pct"] = 100.0 * (d["pt_median"] / d["last_close"] - 1.0)
    denom = np.where(d["pt_median"] == 0, np.nan, d["pt_median"])
    d["dispersion_pct"] = 100.0 * (d["pt_high"] - d["pt_low"]) / denom

    # ---- Winsorize + sector-blend scaling for biased metrics ----
    p_lo = cfg["normalization"]["winsorize_p_low"]
    p_hi = cfg["normalization"]["winsorize_p_high"]
    blend = cfg["normalization"]["sector_blend"]

    for col in ["rev_g1y_pct", "eps_g1y_pct", "pt_upside_pct"]:
        d[f"{col}_w"] = winsorize_by_sector(d, col, "sector", p_lo, p_hi)
        d[f"{col}_score"] = with_sector_blend(d, f"{col}_w", "sector", blend)

    # Dispersion: higher dispersion is worse => invert
    d["dispersion_w"] = winsorize_by_sector(d, "dispersion_pct", "sector", p_lo, p_hi)
    disp_norm = with_sector_blend(d, "dispersion_w", "sector", blend)
    d["dispersion_score"] = 100.0 - disp_norm

    # Coverage: soft boost using log scale; neutral if missing
    cov = d["pt_analystCount"].fillna(0.0)
    denom_cov = np.log(1.0 + max(1.0, float(cov.max()) if np.isfinite(cov.max()) else 1.0))
    d["coverage_score"] = 100.0 * np.log1p(cov) / denom_cov if denom_cov > 0 else 50.0
    d["coverage_score"] = d["coverage_score"].fillna(50.0)

    # Revisions: normalize both, then 60/40 blend (EPS/PT)
    d["eps_rev_score"] = minmax_0_100(_to_num(d["eps_rev_90d_bps"]))
    d["pt_rev_score"]  = minmax_0_100(_to_num(d["pt_rev_90d_bps"]))
    d["revisions_score"] = 0.6 * d["eps_rev_score"] + 0.4 * d["pt_rev_score"]

    # Surprise momentum: EPS surprises (70) + revenue beat rate (30)
    d["eps_surprise_score"] = minmax_0_100(_to_num(d["eps_surprise_avg"]))
    d["rev_beat_score"]     = minmax_0_100(_to_num(d["rev_beat_rate"]))
    d["surprise_score"]     = 0.7 * d["eps_surprise_score"] + 0.3 * d["rev_beat_score"]

    # Growth component (revenue heavier per your spec)
    w_rev = cfg["weights"]["growth_revenue"]
    w_eps = cfg["weights"]["growth_eps"]
    total = max(1, w_rev + w_eps)
    d["growth_score"] = (w_rev * d["rev_g1y_pct_score"] + w_eps * d["eps_g1y_pct_score"]) / total

    return d

def apply_weights_and_gates(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    d = df.copy()
    w = cfg["weights"]

    d["raw_weighted"] = (
        w["growth_total"] * d["growth_score"] / 100.0 +
        w["upside"]       * d["pt_upside_pct_score"] / 100.0 +
        w["revisions"]    * d["revisions_score"] / 100.0 +
        w["surprises"]    * d["surprise_score"] / 100.0 +
        w["coverage"]     * d["coverage_score"] / 100.0 +
        w["dispersion"]   * d["dispersion_score"] / 100.0
    ).clip(0, 100)

    # Gates: negative forward EPS caps; lower cap if weak rev growth too
    cap_a = cfg["gates"]["cap_eps_le_zero"]
    cap_b = cfg["gates"]["cap_eps_le_zero_lowrev"]
    rev_low = cfg["gates"]["rev_growth_low_threshold_pct"]

    neg_eps = _to_num(d["est_eps_nextFY"]).fillna(0.0) <= 0.0
    low_rev = _to_num(d["rev_g1y_pct"]).fillna(0.0) < rev_low

    d["final_grade"] = d["raw_weighted"]
    d.loc[neg_eps, "final_grade"] = np.minimum(d.loc[neg_eps, "final_grade"], cap_a)
    d.loc[neg_eps & low_rev, "final_grade"] = np.minimum(d.loc[neg_eps & low_rev, "final_grade"], cap_b)

    # Small missing-data penalty (max as configured)
    penalty_max = cfg["freshness"]["data_coverage_penalty_max"]
    missing_components = (
        d["pt_median"].isna().astype(int) +
        d["est_rev_nextFY"].isna().astype(int) +
        d["est_eps_nextFY"].isna().astype(int)
    )
    d["final_grade"] = (d["final_grade"] - np.minimum(missing_components, 3) * (penalty_max / 3.0)).clip(0, 100)

    return d

def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    # final_grade immediately after sector, then score columns, then everything else
    score_cols = [
        "growth_score","pt_upside_pct_score","revisions_score","surprise_score",
        "coverage_score","dispersion_score","rev_g1y_pct_score","eps_g1y_pct_score"
    ]
    base = ["ticker","sector","final_grade"]
    remainder = [c for c in df.columns if c not in (base + score_cols)]
    ordered = [c for c in (base + score_cols + remainder) if c in df.columns]
    return df[ordered]

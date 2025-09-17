#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from utils import winsorize_by_sector, with_sector_blend, minmax_0_100, safe_div

def compute_component_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    df must contain columns at least:
      ticker, sector,
      last_close, prev_close,
      est_rev_nextFY, rev_lastFY,
      est_eps_nextFY, eps_lastFY,
      pt_median, pt_mean, pt_high, pt_low, pt_analystCount,
      eps_surprise_avg, rev_beat_rate,
      eps_rev_90d_bps, pt_rev_90d_bps
    and data-age multipliers already applied where needed.
    """
    d = df.copy()

    # Core derived metrics
    d["rev_g1y_pct"] = 100.0 * (d["est_rev_nextFY"] - d["rev_lastFY"]) / d["rev_lastFY"].replace(0, np.nan)
    d.loc[~np.isfinite(d["rev_g1y_pct"]), "rev_g1y_pct"] = np.nan

    d["eps_g1y_pct"] = 100.0 * (d["est_eps_nextFY"] - d["eps_lastFY"]) / d["eps_lastFY"].replace(0, np.nan)
    d["eps_g1y_pct"] = d["eps_g1y_pct"].clip(-200.0, 200.0)

    d["pt_upside_pct"] = 100.0 * (d["pt_median"] / d["last_close"] - 1.0)

    d["dispersion_pct"] = 100.0 * (d["pt_high"] - d["pt_low"]) / np.where(d["pt_median"]==0, np.nan, d["pt_median"])

    # Winsorize biased metrics by sector then scale 0-100 with sector blend
    p_lo = cfg["normalization"]["winsorize_p_low"]
    p_hi = cfg["normalization"]["winsorize_p_high"]
    blend = cfg["normalization"]["sector_blend"]

    for col in ["rev_g1y_pct", "eps_g1y_pct", "pt_upside_pct"]:
        d[f"{col}_w"] = winsorize_by_sector(d, col, "sector", p_lo, p_hi)
        d[f"{col}_score"] = with_sector_blend(d, f"{col}_w", "sector", blend)

    # Dispersion: higher dispersion -> worse score; invert after scaling
    d["dispersion_w"] = winsorize_by_sector(d, "dispersion_pct", "sector", p_lo, p_hi)
    disp_norm = with_sector_blend(d, "dispersion_w", "sector", blend)
    d["dispersion_score"] = 100.0 - disp_norm

    # Coverage: gentle boost (log scale). If pt_analystCount missing, set neutral.
    cov = d["pt_analystCount"].fillna(0.0).astype(float)
    d["coverage_score"] = 100.0 * np.log1p(cov) / np.log(1.0 + max(1.0, cov.max()))
    d["coverage_score"] = d["coverage_score"].fillna(50.0)

    # Revisions (EPS is primary; PT secondary). Normalize both, then combine.
    # We expect eps_rev_90d_bps & pt_rev_90d_bps already as percent/bps style values
    d["eps_rev_score"] = minmax_0_100(d["eps_rev_90d_bps"].astype(float))
    d["pt_rev_score"] = minmax_0_100(d["pt_rev_90d_bps"].astype(float))
    d["revisions_score"] = 0.6 * d["eps_rev_score"] + 0.4 * d["pt_rev_score"]

    # Surprise momentum: primarily EPS surprises blended with revenue beat rate
    d["eps_surprise_score"] = minmax_0_100(d["eps_surprise_avg"].astype(float))
    d["rev_beat_score"] = minmax_0_100(d["rev_beat_rate"].astype(float))
    d["surprise_score"] = 0.7 * d["eps_surprise_score"] + 0.3 * d["rev_beat_score"]

    # Growth component as weighted combo of revenue and EPS
    w_rev = cfg["weights"]["growth_revenue"]
    w_eps = cfg["weights"]["growth_eps"]
    total = max(1, w_rev + w_eps)
    d["growth_score"] = (w_rev * d["rev_g1y_pct_score"] + w_eps * d["eps_g1y_pct_score"]) / total

    return d

def apply_weights_and_gates(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    d = df.copy()
    # Weighted sum
    w = cfg["weights"]
    d["raw_weighted"] = (
        w["growth_total"] * d["growth_score"] / 100.0 +
        w["upside"] * d["pt_upside_pct_score"] / 100.0 +
        w["revisions"] * d["revisions_score"] / 100.0 +
        w["surprises"] * d["surprise_score"] / 100.0 +
        w["coverage"] * d["coverage_score"] / 100.0 +
        w["dispersion"] * d["dispersion_score"] / 100.0
    )
    # Normalize to 0-100
    d["raw_weighted"] = d["raw_weighted"].clip(0, 100)

    # Apply gates
    cap_a = cfg["gates"]["cap_eps_le_zero"]
    cap_b = cfg["gates"]["cap_eps_le_zero_lowrev"]
    rev_low = cfg["gates"]["rev_growth_low_threshold_pct"]
    # next-FY EPS <= 0 caps
    neg_eps = d["est_eps_nextFY"].fillna(0.0) <= 0.0
    low_rev = d["rev_g1y_pct"].fillna(0.0) < rev_low
    d["final_grade"] = d["raw_weighted"]
    d.loc[neg_eps, "final_grade"] = np.minimum(d.loc[neg_eps, "final_grade"], cap_a)
    d.loc[neg_eps & low_rev, "final_grade"] = np.minimum(d.loc[neg_eps & low_rev, "final_grade"], cap_b)

    # Data coverage penalty (missingness)
    penalty_max = cfg["freshness"]["data_coverage_penalty_max"]
    missing_components = (
        d["pt_median"].isna().astype(int) +
        d["est_rev_nextFY"].isna().astype(int) +
        d["est_eps_nextFY"].isna().astype(int)
    )
    d["final_grade"] = d["final_grade"] - np.minimum(missing_components, 3) * (penalty_max / 3.0)
    d["final_grade"] = d["final_grade"].clip(0, 100)

    return d

def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Put final_grade right after sector, followed by score columns, then raw fields
    score_cols = [
        "growth_score","pt_upside_pct_score","revisions_score","surprise_score","coverage_score","dispersion_score",
        "rev_g1y_pct_score","eps_g1y_pct_score"
    ]
    raw_cols = [c for c in df.columns if c not in (["ticker","sector","final_grade"] + score_cols)]
    ordered = ["ticker","sector","final_grade"] + score_cols + raw_cols
    ordered = [c for c in ordered if c in df.columns]
    return df[ordered]

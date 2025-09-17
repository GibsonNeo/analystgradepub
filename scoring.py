#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import pandas as pd

def _to_num(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(np.nan, index=s.index)

def _winsorize(s: pd.Series, low=5, high=95):
    ql, qh = np.nanpercentile(s, low), np.nanpercentile(s, high)
    return s.clip(lower=ql, upper=qh)

def _minmax_0_100(s: pd.Series):
    lo, hi = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo == 0:
        return pd.Series(np.nan, index=s.index)
    return 100.0 * (s - lo) / (hi - lo)

def compute_component_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = df.copy()

    # numeric coercions
    for c in ["est_rev_nextFY","rev_lastFY","est_ni_nextFY","ni_lastFY","pt_median","pt_high","pt_low","pt_analystCount","last_close","pt_rev_90d_bps","rec_index","rec_mom_90d_bps","eps_surprise_avg","rev_beat_rate"]:
        if c in d.columns:
            d[c] = _to_num(d[c])

    # ---- Forward growth (revenue, net income) ----
    d["rev_g1y_pct"] = (d["est_rev_nextFY"] - d["rev_lastFY"]) / d["rev_lastFY"]
    d["ni_g1y_pct"]  = (d["est_ni_nextFY"] - d["ni_lastFY"]) / d["ni_lastFY"]
    d.loc[~np.isfinite(d["rev_g1y_pct"]), "rev_g1y_pct"] = np.nan
    d.loc[~np.isfinite(d["ni_g1y_pct"]),  "ni_g1y_pct"]  = np.nan

    # Winsorize growth
    d["rev_g1y_pct_w"] = _winsorize(d["rev_g1y_pct"])
    d["ni_g1y_pct_w"]  = _winsorize(d["ni_g1y_pct"])

    # ---- Upside vs PT ----
    d["upside_pct"] = (d["pt_median"] - d["last_close"]) / d["last_close"]
    d.loc[~np.isfinite(d["upside_pct"]), "upside_pct"] = np.nan

    # ---- Dispersion & Coverage ----
    d["pt_dispersion_abs"] = (d["pt_high"] - d["pt_low"])
    d.loc[~np.isfinite(d["pt_dispersion_abs"]), "pt_dispersion_abs"] = np.nan

    # ---- Recommendations ----
    # rec_index is already on ~[-100,+100]. Normalize across universe.
    d["rec_level_score"] = _minmax_0_100(d["rec_index"])
    d["rec_mom_score"]   = _minmax_0_100(d["rec_mom_90d_bps"].fillna(0.0))

    # ---- Revisions (keep PT momentum only here, EPS momentum may be missing) ----
    d["revisions_raw"] = d["pt_rev_90d_bps"]
    d["revisions_score"] = _minmax_0_100(d["revisions_raw"].fillna(0.0))

    # ---- Surprises (blend EPS surprise avg and rev beat rate when available) ----
    # Scale: eps_surprise_avg is already %, rev_beat_rate is [0,1]
    z = []
    if "eps_surprise_avg" in d:
        z.append((d["eps_surprise_avg"]).fillna(0.0))
    if "rev_beat_rate" in d:
        z.append((100.0 * d["rev_beat_rate"]).fillna(0.0))
    d["surprises_raw"] = sum(z) / len(z) if z else pd.Series(0.0, index=d.index)
    d["surprises_score"] = _minmax_0_100(d["surprises_raw"])

    # ---- Score components (0..100) ----
    d["growth_revenue_score"]   = _minmax_0_100(d["rev_g1y_pct_w"])
    d["growth_netincome_score"] = _minmax_0_100(d["ni_g1y_pct_w"])
    d["upside_score"]           = _minmax_0_100(d["upside_pct"])
    d["dispersion_score"]       = _minmax_0_100((-1.0) * d["pt_dispersion_abs"])  # narrower better
    d["coverage_score"]         = _minmax_0_100(_to_num(d["pt_analystCount"]).fillna(0.0))

    return d

def apply_weights_and_gates(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    w = cfg["weights"]
    # Default missing -> 0 component contribution
    comp_cols = [
        ("growth_revenue_score",   w["growth_revenue"]),
        ("growth_netincome_score", w["growth_netincome"]),
        ("upside_score",           w["upside"]),
        ("rec_level_score",        w["recs_level"]),
        ("rec_mom_score",          w["recs_momentum"]),
        ("revisions_score",        w["revisions"]),
        ("surprises_score",        w["surprises"]),
        ("dispersion_score",       w["dispersion"]),
        ("coverage_score",         w["coverage"]),
    ]
    for c,_ in comp_cols:
        if c not in d.columns:
            d[c] = 0.0

    # Weighted sum
    total_w = sum(wt for _, wt in comp_cols)
    d["raw_weighted"] = 0.0
    for c, wt in comp_cols:
        d["raw_weighted"] += (d[c].fillna(0.0) * (wt / 100.0))
    d["raw_weighted"] = d["raw_weighted"].clip(lower=0.0, upper=100.0)

    # Gates / penalties: strong penalty if forward net income <= 0 or revenue growth negative
    penalties = np.zeros(len(d))
    if "est_ni_nextFY" in d.columns:
        neg_ni = (d["est_ni_nextFY"] <= 0).fillna(False)
        penalties = penalties + neg_ni.astype(float) * 20.0  # heavy cut
    if "rev_g1y_pct" in d.columns:
        weak_rev = (d["rev_g1y_pct"] < 0).fillna(False)
        penalties = penalties + weak_rev.astype(float) * 10.0

    d["penalties"] = penalties
    d["final_grade"] = (d["raw_weighted"] - d["penalties"]).clip(lower=0.0, upper=100.0)

    return d

def order_columns(d: pd.DataFrame) -> pd.DataFrame:
    # Put final_grade right after sector, then components, then raw
    first = ["ticker", "sector", "final_grade"]
    comps = [
        "growth_revenue_score","growth_netincome_score","upside_score",
        "rec_level_score","rec_mom_score","revisions_score","surprises_score",
        "dispersion_score","coverage_score","raw_weighted","penalties"
    ]
    raw = [c for c in d.columns if c not in set(first+comps)]
    cols = first + comps + raw
    cols_unique = [c for i,c in enumerate(cols) if c in d.columns and c not in cols[:i]]
    return d[cols_unique]

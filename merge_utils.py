#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
from utils import trimmed_mean, days_ago

def pick_freshest(*candidates: Tuple[Optional[float], Optional[str]]) -> Tuple[Optional[float], Optional[str]]:
    """
    Given tuples of (value, asof_ts) choose the freshest (smallest days_ago).
    """
    best = (None, None)
    best_age = 10**9
    for val, ts in candidates:
        if val is None or ts is None:
            continue
        age = days_ago(ts)
        if age is None:
            continue
        if age < best_age:
            best = (val, ts)
            best_age = age
    return best

def merge_targets(finnhub: Dict[str, Any], fmp: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge price target fields from vendors, using freshest or trimmed mean if close in time."""
    res = {}
    # Extract candidates
    fh = finnhub.get("price_targets", {})
    fm = fmp.get("price_targets", {})
    cands = []
    if fh:
        cands.append((fh.get("median"), fh.get("asof")))
    if fm:
        cands.append((fm.get("median"), fm.get("asof")))
    # Freshest rule or trimmed mean if within 7 days both present
    if len(cands) == 2 and cands[0][1] and cands[1][1]:
        age0 = days_ago(cands[0][1])
        age1 = days_ago(cands[1][1])
        if age0 is not None and age1 is not None and abs(age0 - age1) <= 7 and config["fmp_harvest"].get("merge_trimmed_mean", True):
            vals = [cands[0][0], cands[1][0]]
            res["pt_median"] = trimmed_mean(vals, 0.25, 0.75)
            res["pt_source"] = "mixed_trimmed"
        else:
            v, ts = pick_freshest(*cands)
            res["pt_median"] = v
            res["pt_source"] = "freshest"
    elif len(cands) >= 1:
        v, ts = pick_freshest(*cands)
        res["pt_median"] = v
        res["pt_source"] = "single"
    else:
        res["pt_median"] = None
        res["pt_source"] = None

    # carry other fields opportunistically
    if fh:
        for k in ("mean", "high", "low", "analystCount"):
            if fh.get(k) is not None:
                res[f"pt_{k}"] = fh.get(k)
    if fm:
        # prefer to fill missing fields
        res.setdefault("pt_high", fm.get("high"))
        res.setdefault("pt_low", fm.get("low"))
        res.setdefault("pt_mean", fm.get("mean"))
        res.setdefault("pt_analystCount", fm.get("analystCount"))
    return res

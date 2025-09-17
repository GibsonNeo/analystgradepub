#!/usr/bin/env python3
from __future__ import annotations
import os, json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

def _ensure_dirs(cache_dir: str, state_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

def _vendor_path(cache_dir: str, vendor: str, ticker: str) -> str:
    return os.path.join(cache_dir, vendor, f"{ticker}.json")

def load_vendor_cache(cache_dir: str, vendor: str, ticker: str) -> Dict[str, Any]:
    path = _vendor_path(cache_dir, vendor, ticker)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_vendor_cache(cache_dir: str, vendor: str, ticker: str, data: Dict[str, Any]):
    path = _vendor_path(cache_dir, vendor, ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _failure_path(state_dir: str) -> str:
    return os.path.join(state_dir, "vendor_failures.json")

def _load_failures(state_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    p = _failure_path(state_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_failures(state_dir: str, data: Dict[str, Dict[str, Dict[str, Any]]]):
    p = _failure_path(state_dir)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def mark_vendor_failure(state_dir: str, vendor: str, ticker: str, reason: str, backoff_days: int):
    data = _load_failures(state_dir)
    vendor_map = data.get(vendor, {})
    until_dt = datetime.utcnow() + timedelta(days=backoff_days)
    vendor_map[ticker] = {"reason": reason, "failure_until": until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")}
    data[vendor] = vendor_map
    _save_failures(state_dir, data)

def is_vendor_failed(state_dir: str, vendor: str, ticker: str) -> bool:
    data = _load_failures(state_dir)
    vendor_map = data.get(vendor, {})
    rec = vendor_map.get(ticker)
    if not rec:
        return False
    try:
        until = datetime.strptime(rec["failure_until"], "%Y-%m-%dT%H:%M:%SZ")
        return datetime.utcnow() < until
    except Exception:
        return False

def clear_vendor_failure(state_dir: str, vendor: str, ticker: str):
    data = _load_failures(state_dir)
    if vendor in data and ticker in data[vendor]:
        del data[vendor][ticker]
        _save_failures(state_dir, data)

def fmp_state_path(state_dir: str) -> str:
    return os.path.join(state_dir, "fmp_state.json")

def load_fmp_state(state_dir: str) -> Dict[str, Any]:
    p = fmp_state_path(state_dir)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_index": 0, "last_run": None}

def save_fmp_state(state_dir: str, st: Dict[str, Any]):
    with open(fmp_state_path(state_dir), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def append_history(cache: Dict[str, Any], field: str, value, asof_utc: str):
    """Keep a short time series history per field to compute revisions (90d/60d/30d)."""
    hist = cache.get("_history", {})
    series = hist.get(field, [])
    series.append({"ts": asof_utc, "value": value})
    # Keep last ~12 entries to avoid bloat
    hist[field] = series[-12:]
    cache["_history"] = hist

def get_history_series(cache: Dict[str, Any], field: str) -> List[Dict[str, Any]]:
    hist = cache.get("_history", {})
    return hist.get(field, [])

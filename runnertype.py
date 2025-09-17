#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml

from utils import now_utc_str, days_ago, linear_decay_weight, pct_change
from vendors import FinnhubClient, FMPClient, yfinance_last_and_prev_close
from data_cache import (
    load_vendor_cache, save_vendor_cache, append_history, get_history_series,
    mark_vendor_failure, is_vendor_failed, clear_vendor_failure,
    load_fmp_state, save_fmp_state
)
from scoring import compute_component_columns, apply_weights_and_gates, order_columns
from merge_utils import merge_targets

def write_debug(cfg, lines):
    try:
        if not cfg.get("debug", {}).get("enabled", False):
            return
        log_file = cfg.get("debug", {}).get("log_file", "output/debug_run.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")
    except Exception:
        pass

def summarize_missing(df: pd.DataFrame, cfg: dict):
    try:
        if not cfg.get("debug", {}).get("enabled", False):
            return
        mm = df.isna().sum().sort_values(ascending=False)
        total = df.shape[0]
        lines = []
        lines.append(f"=== Missing summary (rows={total}) ===")
        topn = int(cfg.get("debug", {}).get("top_missing_columns", 15))
        for col, cnt in mm.head(topn).items():
            pct = (cnt/total*100.0) if total else 0.0
            lines.append(f"{col}: {cnt} missing ({pct:.1f}%)")
        write_debug(cfg, lines)
        if cfg.get("debug", {}).get("write_missing_summary_csv", True):
            outcsv = cfg.get("debug", {}).get("missing_summary_csv", "output/missing_summary.csv")
            pd.DataFrame({"column": mm.index, "missing": mm.values,
                          "pct_missing": (mm.values/total*100.0) if total else np.zeros_like(mm.values)
                         }).to_csv(outcsv, index=False)
    except Exception:
        pass


def log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}Z] {msg}", flush=True)

def load_config() -> Dict[str, Any]:
    with open("config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_tickers() -> pd.DataFrame:
    df = pd.read_csv("tickers.csv")
    if df.columns.size < 2:
        # headerless: assume 2 columns
        df.columns = ["ticker", "sector"]
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df[["ticker","sector"]]

def ensure_dirs(cfg: Dict[str, Any]):
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["cache_dir"], "finnhub"), exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["cache_dir"], "fmp"), exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["cache_dir"], "yf"), exist_ok=True)
    os.makedirs(cfg["paths"]["state_dir"], exist_ok=True)

def select_fmp_batch(tickers: List[str], cfg: Dict[str, Any]) -> List[str]:
    quota = int(cfg["fmp_harvest"]["daily_quota"])
    ttl = int(cfg["fmp_harvest"]["ttl_days"])
    backoff_days = int(cfg["fmp_harvest"]["failure_backoff_days"])
    state = load_fmp_state(cfg["paths"]["state_dir"])
    last_idx = int(state.get("last_index", 0))

    # Simple ring buffer with staleness preference
    selected = []
    n = len(tickers)
    i = last_idx
    scanned = 0
    while len(selected) < quota and scanned < n:
        t = tickers[i]
        # Skip if failure backoff active
        if not is_vendor_failed(cfg["paths"]["state_dir"], "fmp", t):
            # Check staleness: if FMP cache missing or older than ttl days, select
            fmp_cache = load_vendor_cache(cfg["paths"]["cache_dir"], "fmp", t)
            asof = fmp_cache.get("asof")
            age = days_ago(asof) if asof else 10**6
            if age is None or age > ttl:
                selected.append(t)
        i = (i + 1) % n
        scanned += 1

    # Update state pointer for next run
    state["last_index"] = i
    state["last_run"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    save_fmp_state(cfg["paths"]["state_dir"], state)

    return selected

def fetch_finnhub_for_ticker(fh: FinnhubClient, ticker: str, cfg: Dict[str, Any], cache_dir: str) -> Dict[str, Any]:
    cache = load_vendor_cache(cache_dir, "finnhub", ticker)
    asof = now_utc_str()

    # Price targets
    pt = fh.price_targets(ticker)
    if pt:
        cache["price_targets"] = {
            "median": pt.get("targetMedian"),
            "mean": pt.get("targetMean"),
            "high": pt.get("targetHigh"),
            "low": pt.get("targetLow"),
            "analystCount": pt.get("analystCount"),
            "asof": asof,
        }

    # Recommendation trends (timeseries list)
    recs = fh.recommendation_trends(ticker)
    if recs:
        # Keep last entry (most recent month)
        rec0 = recs[0] if isinstance(recs, list) and recs else {}
        cache["recommendations"] = {
            "buy": rec0.get("buy"),
            "hold": rec0.get("hold"),
            "sell": rec0.get("sell"),
            "asof": asof,
        }

    # Earnings surprises
    er = fh.earnings_surprises(ticker)
    if er and isinstance(er, dict) and "earningsCalendar" in er:
        items = er["earningsCalendar"]
        # Compute average EPS surprise% over last 4, and revenue beat rate
        surprises = []
        rev_beats = []
        for it in items[-8:]:  # look back a bit; take last 4 with valid numbers
            try:
                est = it.get("epsEstimate")
                act = it.get("epsActual")
                if est is not None and act is not None:
                    denom = abs(est) if abs(est) > 1e-9 else 1e-6
                    surprises.append(100.0 * (act - est) / denom)
                # revenue beat if actual > estimate (if both present)
                rev_est = it.get("revenueEstimate")
                rev_act = it.get("revenueActual")
                if rev_est is not None and rev_act is not None:
                    rev_beats.append(1.0 if rev_act > rev_est else 0.0)
            except Exception:
                pass
        surprises = [s for s in surprises if np.isfinite(s)]
        avg_surprise = float(np.mean(surprises[-4:])) if surprises else None
        rev_beat_rate = float(np.mean(rev_beats[-4:])) if rev_beats else None
        cache["earnings"] = {
            "eps_surprise_avg": avg_surprise,
            "rev_beat_rate": rev_beat_rate,
            "asof": asof,
        }

    # Estimates (EPS & revenue) - use next FY and last FY actuals if returned
    eps = fh.eps_estimates(ticker)
    rev = fh.revenue_estimates(ticker)
    # Endpoints differ across plans; store raw for downstream parsing
    if eps:
        cache["eps_estimates"] = {"raw": eps, "asof": asof}
    if rev:
        cache["rev_estimates"] = {"raw": rev, "asof": asof}

    # NOTE: Deriving "nextFY" vs "lastFY" may need parsing by vendor schema.
    # We'll parse in the merge stage to keep fetch thin.

    cache["asof"] = asof
    save_vendor_cache(cache_dir, "finnhub", ticker, cache)
    return cache

def fetch_fmp_for_ticker(fmp: FMPClient, ticker: str, cfg: Dict[str, Any], cache_dir: str, state_dir: str) -> Optional[Dict[str, Any]]:
    # Skip if in failure backoff
    if is_vendor_failed(state_dir, "fmp", ticker):
        return None
    asof = now_utc_str()
    cache = load_vendor_cache(cache_dir, "fmp", ticker) or {}

    try:
        pt = fmp.price_target_consensus(ticker)
        est = fmp.analyst_estimates(ticker)
        # Minimal normalization, store raw and summarized fields for merge
        if pt:
            # consensus could be a list/dict; keep flexible
            median = None; high=None; low=None; mean=None; n=None
            try:
                if isinstance(pt, list) and pt:
                    row = pt[0]
                elif isinstance(pt, dict):
                    row = pt
                else:
                    row = {}
                median = row.get("median") or row.get("priceTargetMedian")
                high = row.get("high") or row.get("priceTargetHigh")
                low = row.get("low") or row.get("priceTargetLow")
                mean = row.get("mean") or row.get("priceTargetAverage")
                n = row.get("analystCount") or row.get("numberAnalystOpinions")
            except Exception:
                pass
            cache["price_targets"] = {
                "median": median, "mean": mean, "high": high, "low": low,
                "analystCount": n, "asof": asof
            }
        if est:
            cache["analyst_estimates"] = {"raw": est, "asof": asof}

        cache["asof"] = asof
        save_vendor_cache(cache_dir, "fmp", ticker, cache)
        clear_vendor_failure(state_dir, "fmp", ticker)
        return cache
    except Exception as e:
        # mark failure backoff
        mark_vendor_failure(state_dir, "fmp", ticker, reason=str(e), backoff_days=int(cfg["fmp_harvest"]["failure_backoff_days"]))
        return None

def fetch_yf_for_ticker(ticker: str, cache_dir: str) -> Optional[Dict[str, Any]]:
    asof = now_utc_str()
    data = yfinance_last_and_prev_close(ticker)
    if data:
        cache = {"last_close": data.get("last_close"), "prev_close": data.get("prev_close"), "asof": asof}
        from data_cache import save_vendor_cache
        save_vendor_cache(cache_dir, "yf", ticker, cache)
        return cache
    return None

def extract_estimates(ven_cache: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Parse raw vendor estimates to extract:
      est_rev_nextFY, rev_lastFY, est_eps_nextFY, eps_lastFY
    This is vendor-schema dependent; we keep it heuristic and robust.
    """
    rev_next = None; rev_last = None; eps_next = None; eps_last = None
    # Finnhub structure assumed in cache["rev_estimates"]["raw"] etc.
    try:
        rev_raw = ven_cache.get("rev_estimates", {}).get("raw")
        if isinstance(rev_raw, dict):
            # Try keys like "data" or "estimates"
            rows = rev_raw.get("data") or rev_raw.get("estimates") or []
        elif isinstance(rev_raw, list):
            rows = rev_raw
        else:
            rows = []
        # Find next FY and last FY rows by period naming
        # period formats vary, e.g., "2025", "FY2025", "2025FY"
        fy_values = {}
        for r in rows:
            period = str(r.get("period") or r.get("fiscalYear") or "")
            val = r.get("estimate") or r.get("revenueAvg")
            if val is None:
                continue
            fy = "".join(ch for ch in period if ch.isdigit())
            if len(fy) == 4:
                fy_values[fy] = float(val)
        years = sorted([int(y) for y in fy_values.keys()])
        if len(years) >= 2:
            rev_last = fy_values.get(str(years[-2]))
            rev_next = fy_values.get(str(years[-1]))
    except Exception:
        pass
    try:
        eps_raw = ven_cache.get("eps_estimates", {}).get("raw")
        rows = []
        if isinstance(eps_raw, dict):
            rows = eps_raw.get("data") or eps_raw.get("estimates") or []
        elif isinstance(eps_raw, list):
            rows = eps_raw
        for r in rows:
            period = str(r.get("period") or r.get("fiscalYear") or "")
            val = r.get("estimate") or r.get("epsAvg")
            if val is None:
                continue
            fy = "".join(ch for ch in period if ch.isdigit())
            if len(fy) == 4:
                fy = int(fy)
                # heuristic: store both last and next by ordering
        # Fallbacks deliberately minimal; vendor diversity makes exact parsing fragile.
    except Exception:
        pass
    # If EPS parsing failed, keep None; downstream handles missing.
    return rev_next, rev_last, eps_next, eps_last

def derive_revisions_from_history(hist: List[dict], days_window: int) -> Optional[float]:
    """Compute % change vs value ~window days ago from history entries (ts,value)."""
    if not hist:
        return None
    # Assume hist sorted by time asc
    try:
        from datetime import datetime
        entries = [(datetime.strptime(h["ts"], "%Y-%m-%dT%H:%M:%SZ"), h["value"]) for h in hist if h.get("value") is not None]
        if not entries:
            return None
        entries = sorted(entries, key=lambda x: x[0])
        # pick the closest at or before target date
        target = entries[-1][0] - timedelta(days=days_window)
        past = None
        latest = entries[-1][1]
        for dt, val in entries:
            if dt <= target:
                past = val
            else:
                break
        if past is None:
            # fallback to earliest if we have at least ~2/3 window
            if len(entries) >= 2:
                past = entries[0][1]
            else:
                return None
        denom = abs(past) if abs(past) > 1e-6 else 1e-6
        return 100.0 * (latest - past) / denom
    except Exception:
        return None

def extract_estimates2(fh_cache: dict, fmp_cache: dict) -> tuple[float|None,float|None,float|None,float|None]:
    """
    Return: (est_rev_nextFY, rev_lastFY, est_eps_nextFY, eps_lastFY)
    Tries Finnhub first, falls back to FMP if needed. Heuristic parsing across common shapes.
    """
    def _dig(raw):
        rows = []
        if isinstance(raw, dict):
            rows = raw.get("data") or raw.get("estimates") or raw.get("result") or []
        elif isinstance(raw, list):
            rows = raw
        return rows

    def _fy(s: str) -> int|None:
        if not s: return None
        digits = "".join(ch for ch in str(s) if ch.isdigit())
        return int(digits) if len(digits) == 4 else None

    # ---- revenue ----
    rev_next = rev_last = None
    for source in ("rev_estimates",):
        raw = fh_cache.get(source, {}).get("raw")
        rows = _dig(raw)
        if rows:
            tmp = {}
            for r in rows:
                yr = _fy(r.get("period") or r.get("fiscalYear"))
                val = r.get("estimate") or r.get("revenueAvg") or r.get("revenueEstimate")
                if yr and val is not None:
                    tmp[yr] = float(val)
            if tmp:
                years = sorted(tmp)
                if len(years) >= 2:
                    rev_last = tmp.get(years[-2])
                    rev_next = tmp.get(years[-1])
                    break

    if rev_next is None or rev_last is None:
        # FMP fallback (analyst_estimates payload can include revenue by year)
        raw = fmp_cache.get("analyst_estimates", {}).get("raw")
        rows = _dig(raw)
        tmp_r = {}
        for r in rows:
            yr = _fy(r.get("year") or r.get("fiscalYear") or r.get("period"))
            val = r.get("revenueAvg") or r.get("estimatedRevenueAvg") or r.get("revenueEstimate")
            if yr and val is not None:
                tmp_r[yr] = float(val)
        if tmp_r:
            years = sorted(tmp_r)
            if len(years) >= 2:
                rev_last = rev_last or tmp_r.get(years[-2])
                rev_next = rev_next or tmp_r.get(years[-1])

    # ---- EPS ----
    eps_next = eps_last = None
    for source in ("eps_estimates",):
        raw = fh_cache.get(source, {}).get("raw")
        rows = _dig(raw)
        if rows:
            tmp = {}
            for r in rows:
                yr = _fy(r.get("period") or r.get("fiscalYear"))
                val = r.get("estimate") or r.get("epsAvg") or r.get("estimatedEpsAvg") or r.get("epsEstimate")
                if yr and val is not None:
                    tmp[yr] = float(val)
            if tmp:
                years = sorted(tmp)
                if len(years) >= 2:
                    eps_last = tmp.get(years[-2])
                    eps_next = tmp.get(years[-1])
                    break

    if eps_next is None or eps_last is None:
        raw = fmp_cache.get("analyst_estimates", {}).get("raw")
        rows = _dig(raw)
        tmp_e = {}
        for r in rows:
            yr = _fy(r.get("year") or r.get("fiscalYear") or r.get("period"))
            val = r.get("epsAvg") or r.get("estimatedEpsAvg") or r.get("epsEstimate")
            if yr and val is not None:
                tmp_e[yr] = float(val)
        if tmp_e:
            years = sorted(tmp_e)
            if len(years) >= 2:
                eps_last = eps_last or tmp_e.get(years[-2])
                eps_next = eps_next or tmp_e.get(years[-1])

    return rev_next, rev_last, eps_next, eps_last


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    tick = load_tickers()
    symbols = tick["ticker"].tolist()
    log(f"Loaded {len(symbols)} tickers.")

    use_fh = cfg["vendors"].get("use_finnhub", True)
    use_yf = cfg["vendors"].get("use_yfinance", True)
    use_fmp = cfg["vendors"].get("use_fmp", True)

    fh = FinnhubClient() if use_fh else None
    fmp = FMPClient() if use_fmp else None

    # FMP batch selection (ring buffer + staleness + failure backoff)
    fmp_batch = []
    if use_fmp:
        fmp_batch = select_fmp_batch(symbols, cfg)
        log(f"FMP will refresh up to {len(fmp_batch)} tickers this run.")

    rows: List[Dict[str, Any]] = []
    progress_every = int(cfg["progress"]["every"])

    for idx, row in tick.iterrows():
        symbol = row["ticker"]
        sector = row["sector"]

        # Fetch vendors
        if use_fh:
            fetch_finnhub_for_ticker(fh, symbol, cfg, cfg["paths"]["cache_dir"])
        if use_yf:
            fetch_yf_for_ticker(symbol, cfg["paths"]["cache_dir"])
        if use_fmp and symbol in fmp_batch:
            fetch_fmp_for_ticker(fmp, symbol, cfg, cfg["paths"]["cache_dir"], cfg["paths"]["state_dir"])

        # Load caches for merge/derive
        fh_cache = load_vendor_cache(cfg["paths"]["cache_dir"], "finnhub", symbol)
        fmp_cache = load_vendor_cache(cfg["paths"]["cache_dir"], "fmp", symbol)
        yf_cache  = load_vendor_cache(cfg["paths"]["cache_dir"], "yf", symbol)

        # Merge targets
        target_fields = merge_targets(fh_cache, fmp_cache, cfg)
        pt_median = target_fields.get("pt_median")
        pt_mean = target_fields.get("pt_mean")
        pt_high = target_fields.get("pt_high")
        pt_low = target_fields.get("pt_low")
        pt_n = target_fields.get("pt_analystCount")

        # Quotes
        last_close = yf_cache.get("last_close") if yf_cache else None
        prev_close = yf_cache.get("prev_close") if yf_cache else None

        # Estimates (very heuristic parsing; vendor diversity)
        est_rev_nextFY, rev_lastFY, est_eps_nextFY, eps_lastFY = extract_estimates2(fh_cache, fmp_cache)

        # Revisions: use cached history of EPS nextFY and PT median over time (self-maintained)
        # We append today's snapshot and compute ~90d deltas later.
        # Append history to finnhub cache (preferred) or fmp if only source
        if est_eps_nextFY is not None:
            append_history(fh_cache, "est_eps_nextFY", est_eps_nextFY, now_utc_str())
            save_vendor_cache(cfg["paths"]["cache_dir"], "finnhub", symbol, fh_cache)
        if pt_median is not None:
            append_history(fh_cache, "pt_median", pt_median, now_utc_str())
            save_vendor_cache(cfg["paths"]["cache_dir"], "finnhub", symbol, fh_cache)

        eps_hist = get_history_series(fh_cache, "est_eps_nextFY")
        pt_hist  = get_history_series(fh_cache, "pt_median")
        eps_rev_90d = derive_revisions_from_history(eps_hist, 90)
        pt_rev_90d  = derive_revisions_from_history(pt_hist, 90)

        # Freshness weights for estimates & PTs
        est_age = None
        if fh_cache.get("rev_estimates", {}).get("asof"):
            from utils import days_ago
            est_age = days_ago(fh_cache["rev_estimates"]["asof"])
        pt_age = None
        if pt_median is not None:
            pt_age = days_ago(fh_cache.get("price_targets", {}).get("asof") or fmp_cache.get("price_targets", {}).get("asof"))
        est_mul = linear_decay_weight(est_age, cfg["freshness"]["estimates_full_days"], cfg["freshness"]["estimates_zero_days"])
        pt_mul  = linear_decay_weight(pt_age,  cfg["freshness"]["pt_full_days"],        cfg["freshness"]["pt_zero_days"])

        # Apply multipliers (if 0, we keep raw value but score modules will downweight by missingness if needed)
        # We'll store raw and the multipliers separately; scores use raw fields and multipliers implicitly via revision & decay policies.

        # Surprises
        eps_surprise_avg = fh_cache.get("earnings", {}).get("eps_surprise_avg")
        rev_beat_rate    = fh_cache.get("earnings", {}).get("rev_beat_rate")

        
        # Debug: capture missing fields to help troubleshoot vendor gaps
        debug_lines = []
        missing = []
        if pt_median is None: missing.append("pt_median")
        if est_rev_nextFY is None: missing.append("est_rev_nextFY")
        if rev_lastFY is None: missing.append("rev_lastFY")
        if est_eps_nextFY is None: missing.append("est_eps_nextFY")
        if eps_lastFY is None: missing.append("eps_lastFY")
        if last_close is None: missing.append("last_close")
        if missing:
            debug_lines.append(f"DEBUG: ticker={symbol} missing={','.join(missing)}")
        write_debug(cfg, debug_lines)

        rows.append({
            "ticker": symbol, "sector": sector,
            # Quotes
            "last_close": last_close, "prev_close": prev_close,
            # Estimates
            "est_rev_nextFY": est_rev_nextFY, "rev_lastFY": rev_lastFY,
            "est_eps_nextFY": est_eps_nextFY, "eps_lastFY": eps_lastFY,
            # Price targets
            "pt_median": pt_median, "pt_mean": pt_mean, "pt_high": pt_high, "pt_low": pt_low, "pt_analystCount": pt_n,
            # Revisions windows
            "eps_rev_90d_bps": eps_rev_90d, "pt_rev_90d_bps": pt_rev_90d,
            # Surprises
            "eps_surprise_avg": eps_surprise_avg, "rev_beat_rate": rev_beat_rate,
            # Freshness multipliers (for audit)
            "est_age_days": est_age, "est_decay_mult": est_mul,
            "pt_age_days": pt_age, "pt_decay_mult": pt_mul,
            # As-of stamps
            "asof": now_utc_str()
        })

        if (idx + 1) % progress_every == 0:
            log(f"Processed {idx + 1} / {len(symbols)} tickers so far.")

    df = pd.DataFrame(rows)

    # Compute components & scores
    df_scored = compute_component_columns(df, cfg)
    df_scored = apply_weights_and_gates(df_scored, cfg)
    df_final = order_columns(df_scored)

    out_path = os.path.join(cfg["paths"]["output_dir"], "analyst_grade.csv")
    df_final.to_csv(out_path, index=False)
    log(f"Wrote {out_path} with {df_final.shape[0]} rows.")

if __name__ == "__main__":
    main()

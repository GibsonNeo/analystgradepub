#!/usr/bin/env python3
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import yaml

from utils import now_utc_str, days_ago, linear_decay_weight
from vendors import FinnhubClient, FMPClient, yfinance_last_and_prev_close, yfinance_targets
from data_cache import (
    load_vendor_cache, save_vendor_cache, append_history, get_history_series,
    mark_vendor_failure, is_vendor_failed, clear_vendor_failure,
    load_fmp_state, save_fmp_state
)
from scoring import compute_component_columns, apply_weights_and_gates, order_columns

# -------------------------
# Debug / logging
# -------------------------

def log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}Z] {msg}", flush=True)

def write_debug(cfg, lines):
    try:
        if not cfg.get("debug", {}).get("enabled", False):
            return
        log_file = cfg.get("debug", {}).get("log_file", "output/debuglog.txt")
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

SCHEMA_PEEKS = {"finnhub": 0, "fmp": 0}
SCHEMA_PEEK_LIMIT = 3
def peek_schema(cfg, vendor_name: str, symbol: str, payload: dict):
    try:
        if not cfg.get("debug", {}).get("enabled", False):
            return
        if SCHEMA_PEEKS[vendor_name] >= SCHEMA_PEEK_LIMIT:
            return
        keys = sorted(list(payload.keys())) if isinstance(payload, dict) else str(type(payload))
        write_debug(cfg, [f"SCHEMA[{vendor_name}] {symbol}: {keys}"])
        SCHEMA_PEEKS[vendor_name] += 1
    except Exception:
        pass

# -------------------------
# Config / IO helpers
# -------------------------

def load_config() -> Dict[str, Any]:
    with open("config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_tickers() -> pd.DataFrame:
    df = pd.read_csv("tickers.csv")
    if df.columns.size < 2:
        df.columns = ["ticker", "sector"]
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df[["ticker","sector"]]

def ensure_dirs(cfg: Dict[str, Any]):
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["cache_dir"], "finnhub"), exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["cache_dir"], "fmp"), exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["cache_dir"], "yf"), exist_ok=True)
    os.makedirs(cfg["paths"]["state_dir"], exist_ok=True)

# -------------------------
# Vendor pacing / classification
# -------------------------

def fh_pause(cfg):
    try:
        ms = int(cfg.get("finnhub_rate", {}).get("sleep_ms_between_calls", 120))
        if ms > 0:
            time.sleep(ms / 1000.0)
    except Exception:
        pass

def classify_fmp_402(text: str | None) -> str:
    if not text:
        return "FMP_402_unknown"
    t = text.lower()
    if "special endpoint" in t and "not available under your current subscription" in t:
        return "FMP_402_plan_symbol"
    if "legacy endpoints" in t:
        return "FMP_402_legacy"
    if "reached your" in t and "limit" in t:
        return "FMP_402_quota"
    return "FMP_402_other"

# -------------------------
# FMP batch selection
# -------------------------

def _fmp_cache_usable(cache: dict) -> bool:
    if not cache: return False
    est = cache.get("analyst_estimates_stable") or {}
    has_est = bool(est.get("raw"))
    return has_est

def select_fmp_batch(tickers: List[str], cfg: Dict[str, Any]) -> List[str]:
    quota = int(cfg["fmp_harvest"]["daily_quota"])
    ttl = int(cfg["fmp_harvest"]["ttl_days"])
    state = load_fmp_state(cfg["paths"]["state_dir"])
    last_idx = int(state.get("last_index", 0))

    selected = []
    n = len(tickers)
    i = last_idx
    scanned = 0
    while len(selected) < quota and scanned < n:
        t = tickers[i]
        if not is_vendor_failed(cfg["paths"]["state_dir"], "fmp", t):
            fmp_cache = load_vendor_cache(cfg["paths"]["cache_dir"], "fmp", t)
            asof = fmp_cache.get("asof")
            usable = _fmp_cache_usable(fmp_cache)
            age = days_ago(asof) if asof else 10**6
            if (age is None or age > ttl) or (not usable):
                selected.append(t)
        i = (i + 1) % n
        scanned += 1

    state["last_index"] = i
    state["last_run"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    save_fmp_state(cfg["paths"]["state_dir"], state)
    return selected

# -------------------------
# Fetchers
# -------------------------

FH_RATELIMITED = False
FMP_QUOTA_TRIPPED = False  # only for actual quota wording, not plan gating

def fetch_finnhub_for_ticker(fh: FinnhubClient, ticker: str, cfg: Dict[str, Any], cache_dir: str) -> Dict[str, Any]:
    global FH_RATELIMITED
    cache = load_vendor_cache(cache_dir, "finnhub", ticker)
    asof = now_utc_str()

    fh_pause(cfg); pt, st_pt = fh.price_targets(ticker)  # may be 403 on free tier
    write_debug(cfg, [f'FH PT {ticker} status={st_pt}'])
    if pt:
        cache["price_targets"] = {
            "median": pt.get("targetMedian"),
            "mean": pt.get("targetMean"),
            "high": pt.get("targetHigh"),
            "low": pt.get("targetLow"),
            "analystCount": pt.get("analystCount"),
            "asof": asof,
        }

    fh_pause(cfg); recs, st_rec = fh.recommendation_trends(ticker)
    write_debug(cfg, [f'FH REC {ticker} status={st_rec}'])
    if st_rec == 429:
        FH_RATELIMITED = True
    if recs and isinstance(recs, list):
        rec0 = recs[0]
        cache["recommendations"] = {
            "strongBuy": rec0.get("strongBuy"),
            "buy": rec0.get("buy"),
            "hold": rec0.get("hold"),
            "sell": rec0.get("sell"),
            "strongSell": rec0.get("strongSell"),
            "asof": asof,
        }
        # rec index [-100, +100] and keep a daily-ish history
        sb, b = rec0.get("strongBuy") or 0, rec0.get("buy") or 0
        h, s, ss = rec0.get("hold") or 0, rec0.get("sell") or 0, rec0.get("strongSell") or 0
        tot = max((sb + b + h + s + ss), 1)
        rec_index = 100.0 * ((2*sb + b) - (s + 2*ss)) / (2.0 * tot)
        append_history(cache, "rec_index", rec_index, asof)

    fh_pause(cfg); er, st_er = fh.earnings_surprises(ticker)
    write_debug(cfg, [f'FH ER {ticker} status={st_er}'])
    if er and isinstance(er, dict) and "earningsCalendar" in er:
        items = er["earningsCalendar"]
        surprises, rev_beats = [], []
        for it in items[-8:]:
            try:
                est = it.get("epsEstimate"); act = it.get("epsActual")
                if est is not None and act is not None:
                    denom = abs(est) if abs(est) > 1e-9 else 1e-6
                    surprises.append(100.0 * (act - est) / denom)
                rev_est = it.get("revenueEstimate"); rev_act = it.get("revenueActual")
                if rev_est is not None and rev_act is not None:
                    rev_beats.append(1.0 if rev_act > rev_est else 0.0)
            except Exception:
                pass
        surprises = [s for s in surprises if np.isfinite(s)]
        avg_surprise = float(np.mean(surprises[-4:])) if surprises else None
        rev_beat_rate = float(np.mean(rev_beats[-4:])) if rev_beats else None
        cache["earnings"] = {"eps_surprise_avg": avg_surprise, "rev_beat_rate": rev_beat_rate, "asof": asof}

    cache["asof"] = asof
    save_vendor_cache(cache_dir, "finnhub", ticker, cache)
    return cache

def fetch_fmp_for_ticker(fmp: FMPClient, ticker: str, cfg: Dict[str, Any], cache_dir: str, state_dir: str) -> Optional[Dict[str, Any]]:
    global FMP_QUOTA_TRIPPED
    if FMP_QUOTA_TRIPPED:
        return None
    if not os.getenv('FMP_API_KEY'):
        write_debug(cfg, [f'FMP {ticker} skipped: no API key'])
        return None
    if is_vendor_failed(state_dir, "fmp", ticker):
        return None

    asof = now_utc_str()
    cache = load_vendor_cache(cache_dir, "fmp", ticker) or {}
    try:
        retries = int(cfg.get("fmp_harvest", {}).get("retries", 3))
        est, st_est, body = fmp.analyst_estimates_stable(ticker, retries=retries)
        write_debug(cfg, [f'FMP stable estimates {ticker} status={st_est}'])
        if st_est == 200 and est:
            cache["analyst_estimates_stable"] = {"raw": est, "asof": asof}
            cache["asof"] = asof
            save_vendor_cache(cache_dir, "fmp", ticker, cache)
            clear_vendor_failure(state_dir, "fmp", ticker)
            return cache
        elif st_est == 402:
            reason = classify_fmp_402(body)
            write_debug(cfg, [f'FMP 402 for {ticker}: {reason}'])
            if reason == "FMP_402_quota":
                FMP_QUOTA_TRIPPED = True  # stop remaining FMP calls this run
            # don't mark failure; try again on later days / skip for plan gating
            return None
        else:
            # other codes (403 legacy, 404 unknown symbol, timeouts exhausted, etc.)
            write_debug(cfg, [f'FMP ERR {ticker}: status={st_est} body={str(body)[:140]}'])
            return None
    except Exception as e:
        mark_vendor_failure(state_dir, "fmp", ticker, reason=str(e),
                            backoff_days=int(cfg["fmp_harvest"]["failure_backoff_days"]))
        return None

# -------------------------
# FMP parsing helpers
# -------------------------

def extract_growth_from_fmp_stable(fmp_cache: dict) -> Tuple[Optional[float],Optional[float],Optional[float],Optional[float]]:
    """
    From FMP stable analyst-estimates (annual), derive:
      (est_rev_nextFY, rev_lastFY, est_netinc_nextFY, netinc_lastFY)
    Uses previous row as "last" when next is future-dated.
    """
    try:
        raw = (fmp_cache or {}).get("analyst_estimates_stable", {}).get("raw") or []
        rows = []
        for r in raw:
            d = r.get("date")
            if not d: continue
            try:
                dt = date.fromisoformat(str(d)[:10])
            except Exception:
                continue
            rows.append((dt, r))
        if not rows:
            return None, None, None, None
        rows.sort(key=lambda x: x[0])

        today = date.today()
        next_idx = None
        for i,(dt0,_) in enumerate(rows):
            if dt0 >= today:
                next_idx = i
                break
        if next_idx is None:
            # all past: last two rows as proxies
            last_dt, last_r = rows[-2] if len(rows) >= 2 else rows[-1]
            next_dt, next_r = rows[-1]
        else:
            next_dt, next_r = rows[next_idx]
            last_dt, last_r = rows[next_idx-1] if next_idx > 0 else rows[next_idx]

        def _f(x):
            try: return float(x)
            except Exception: return None

        rev_next = _f(next_r.get("revenueAvg"))
        rev_last = _f(last_r.get("revenueAvg"))
        ni_next  = _f(next_r.get("netIncomeAvg"))
        ni_last  = _f(last_r.get("netIncomeAvg"))
        return rev_next, rev_last, ni_next, ni_last
    except Exception:
        return None, None, None, None

# -------------------------
# Main
# -------------------------

def main():
    cfg = load_config()
    ensure_dirs(cfg)
    tick = load_tickers()
    symbols = tick["ticker"].tolist()
    log(f"Loaded {len(symbols)} tickers.")

    fh_key = bool(os.getenv('FINNHUB_API_KEY'))
    fmp_key = bool(os.getenv('FMP_API_KEY'))
    write_debug(cfg, [f'APIKEY Finnhub set={fh_key}', f'APIKEY FMP set={fmp_key}'])

    use_fh = cfg["vendors"].get("use_finnhub", True)
    use_yf = cfg["vendors"].get("use_yfinance", True)
    use_fmp = cfg["vendors"].get("use_fmp", True)

    fh = FinnhubClient() if use_fh else None
    fmp = FMPClient() if use_fmp else None

    fmp_batch = []
    if use_fmp:
        fmp_batch = select_fmp_batch(symbols, cfg)
        log(f"FMP will refresh up to {len(fmp_batch)} tickers this run.")

    rows: List[Dict[str, Any]] = []
    progress_every = int(cfg["progress"]["every"])

    for idx, row in tick.iterrows():
        symbol = row["ticker"]; sector = row["sector"]

        # ---- Finnhub (with light throttle) ----
        if use_fh:
            fetch_finnhub_for_ticker(fh, symbol, cfg, cfg["paths"]["cache_dir"])

        # ---- Yahoo quotes ----
        if use_yf:
            q = yfinance_last_and_prev_close(symbol)
            if q:
                save_vendor_cache(cfg["paths"]["cache_dir"], "yf", symbol,
                                  {"last_close": q["last_close"], "prev_close": q["prev_close"], "asof": now_utc_str()})

        # ---- FMP (stable estimates) ----
        if use_fmp and (symbol in fmp_batch) and not FMP_QUOTA_TRIPPED:
            fetch_fmp_for_ticker(fmp, symbol, cfg, cfg["paths"]["cache_dir"], cfg["paths"]["state_dir"])

        # ---- Load caches ----
        fh_cache = load_vendor_cache(cfg["paths"]["cache_dir"], "finnhub", symbol)
        fmp_cache = load_vendor_cache(cfg["paths"]["cache_dir"], "fmp", symbol)
        yf_cache  = load_vendor_cache(cfg["paths"]["cache_dir"], "yf", symbol)

        peek_schema(cfg, 'finnhub', symbol, fh_cache)
        if fmp_cache:
            peek_schema(cfg, 'fmp', symbol, fmp_cache)

        # ----- PRICE TARGETS (prefer vendors; fallback to yfinance) -----
        pt = (fh_cache.get("price_targets") or {})
        pt_median = pt.get("median")
        pt_mean   = pt.get("mean")
        pt_high   = pt.get("high")
        pt_low    = pt.get("low")
        pt_n      = pt.get("analystCount")
        pt_asof   = pt.get("asof")   # may be None

        if pt_median is None:
            ypt = yfinance_targets(symbol)
            if ypt:
                pt_median = ypt.get("median")
                pt_mean   = ypt.get("mean")
                pt_high   = ypt.get("high")
                pt_low    = ypt.get("low")
                pt_n      = ypt.get("analystCount")
                pt_asof   = ypt.get("asof")  # capture asof for age calc
                write_debug(cfg, [f"YF PT fallback used for {symbol}"])

        # ----- QUOTES -----
        last_close = yf_cache.get("last_close") if yf_cache else None
        prev_close = yf_cache.get("prev_close") if yf_cache else None

        # ----- FORWARD GROWTH from FMP stable (single call) -----
        est_rev_nextFY, rev_lastFY, est_ni_nextFY, ni_lastFY = extract_growth_from_fmp_stable(fmp_cache)

        # ----- REVISIONS (keep PT history for momentum) -----
        if pt_median is not None:
            append_history(fh_cache, "pt_median", float(pt_median), now_utc_str())
            save_vendor_cache(cfg["paths"]["cache_dir"], "finnhub", symbol, fh_cache)

        pt_hist  = get_history_series(fh_cache, "pt_median")
        def derive_rev_from_hist(hist, days_window):
            if not hist: return None
            try:
                entries = [(datetime.strptime(h["ts"], "%Y-%m-%dT%H:%M:%SZ"), h["value"]) for h in hist if h.get("value") is not None]
                entries.sort(key=lambda x: x[0])
                target = entries[-1][0] - timedelta(days=days_window)
                past = None; latest = entries[-1][1]
                for dt0, val in entries:
                    if dt0 <= target:
                        past = val
                    else:
                        break
                if past is None:
                    if len(entries) >= 2:
                        past = entries[0][1]
                    else:
                        return None
                denom = abs(past) if abs(past) > 1e-6 else 1e-6
                return 100.0 * (latest - past) / denom
            except Exception:
                return None

        pt_rev_90d  = derive_rev_from_hist(pt_hist, 90)

        # ----- RECOMMENDATION momentum -----
        rec_hist = get_history_series(fh_cache, "rec_index")
        rec_mom_90d = derive_rev_from_hist(rec_hist, 90)
        rec_index_latest = rec_hist[-1]["value"] if rec_hist else None

        # ----- Freshness multipliers (audit only) -----
        est_age = days_ago((fmp_cache.get("analyst_estimates_stable") or {}).get("asof"))

        # prefer the asof from whichever PT source actually populated the data
        pt_asof_cache = (fh_cache.get("price_targets") or {}).get("asof")
        pt_age = days_ago(pt_asof or pt_asof_cache)

        est_mul = linear_decay_weight(est_age, cfg["freshness"]["estimates_full_days"], cfg["freshness"]["estimates_zero_days"])
        pt_mul  = linear_decay_weight(pt_age,  cfg["freshness"]["pt_full_days"],        cfg["freshness"]["pt_zero_days"])

        # ----- Reasons for missing (for audit) -----
        reason_growth = None
        if (est_rev_nextFY is None or rev_lastFY is None or est_ni_nextFY is None or ni_lastFY is None):
            has_raw = bool((fmp_cache or {}).get("analyst_estimates_stable", {}).get("raw"))
            reason_growth = "FMP_no_data" if not has_raw else "parse_fail"

        reason_rec = None
        if rec_index_latest is None:
            reason_rec = "FH_no_data_or_429"

        # ----- Row -----
        rows.append({
            "ticker": symbol, "sector": sector,
            "last_close": last_close, "prev_close": prev_close,
            "est_rev_nextFY": est_rev_nextFY, "rev_lastFY": rev_lastFY,
            "est_ni_nextFY": est_ni_nextFY, "ni_lastFY": ni_lastFY,
            "pt_median": pt_median, "pt_mean": pt_mean, "pt_high": pt_high, "pt_low": pt_low, "pt_analystCount": pt_n,
            "pt_rev_90d_bps": pt_rev_90d,
            "rec_index": rec_index_latest, "rec_mom_90d_bps": rec_mom_90d,
            "eps_surprise_avg": (fh_cache.get("earnings") or {}).get("eps_surprise_avg"),
            "rev_beat_rate": (fh_cache.get("earnings") or {}).get("rev_beat_rate"),
            "est_age_days": est_age, "est_decay_mult": est_mul,
            "pt_age_days": pt_age, "pt_decay_mult": pt_mul,
            "reason_missing_growth": reason_growth,
            "reason_missing_rec": reason_rec,
            "asof": now_utc_str()
        })

        if (idx + 1) % progress_every == 0:
            log(f"Processed {idx + 1} / {len(symbols)} tickers so far.")

    # ---- scoring & output ----
    df = pd.DataFrame(rows)
    df_scored = compute_component_columns(df, cfg)
    df_scored = apply_weights_and_gates(df_scored, cfg)
    df_final = order_columns(df_scored)

    out_path = os.path.join(cfg["paths"]["output_dir"], "analyst_grade.csv")
    df_final.to_csv(out_path, index=False)

    # audit
    audit_rows = []
    for r in rows:
        audit_rows.append({
            'ticker': r['ticker'],
            'has_pt': r.get('pt_median') is not None,
            'has_rev': (r.get('est_rev_nextFY') is not None and r.get('rev_lastFY') is not None),
            'has_ni': (r.get('est_ni_nextFY') is not None and r.get('ni_lastFY') is not None),
            'has_quote': r.get('last_close') is not None,
            'rec_index': r.get('rec_index'),
            'pt_age_days': r.get('pt_age_days'),
            'reason_missing_growth': r.get('reason_missing_growth'),
            'reason_missing_rec': r.get('reason_missing_rec'),
        })
    pd.DataFrame(audit_rows).to_csv(os.path.join(cfg['paths']['output_dir'], 'audit_fetch.csv'), index=False)

    log(f"Wrote {out_path} with {df_final.shape[0]} rows.")
    summarize_missing(df_final, cfg)

if __name__ == "__main__":
    main()
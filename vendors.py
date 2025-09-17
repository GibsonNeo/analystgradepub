#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from typing import Tuple, Dict, Any, List
import requests
import yfinance as yf

# -------------------------
# Low-level HTTP helpers
# -------------------------

def _http_get_json(url: str, params: dict | None, timeout_s: int = 12) -> tuple[dict | list | None, int, str | None]:
    """
    Returns (json_or_none, status_code, error_text_or_none).
    Handles non-JSON error bodies by returning text in the third slot.
    """
    try:
        r = requests.get(url, params=params, timeout=timeout_s)
        ct = (r.headers.get("content-type") or "").lower()
        text = None if r.ok else r.text
        if "application/json" in ct:
            try:
                return r.json(), r.status_code, text
            except Exception:
                return None, r.status_code, r.text
        else:
            # Some providers send text/plain on errors
            try:
                return r.json(), r.status_code, r.text
            except Exception:
                return None, r.status_code, r.text
    except requests.exceptions.RequestException as e:
        return None, 0, str(e)

def _retryable(status: int) -> bool:
    # transient classes worth retrying
    return status in (0, 408, 429, 500, 502, 503, 504)

def _sleep_backoff(attempt: int, base_ms: int = 300, max_ms: int = 3000):
    ms = min(max_ms, base_ms * (2 ** (attempt - 1)))
    time.sleep(ms / 1000.0)

# -------------------------
# Finnhub client
# -------------------------

class FinnhubClient:
    BASE = "https://finnhub.io/api/v1"
    def __init__(self):
        self.key = os.environ.get("FINNHUB_API_KEY", "")

    def _get(self, path: str, params: dict | None = None) -> Tuple[Dict[str, Any] | List[Dict[str, Any]] | None, int]:
        p = dict(params or {})
        if self.key:
            p["token"] = self.key
        data, status, _ = _http_get_json(f"{self.BASE}/{path.lstrip('/')}", p)
        return data, status

    def price_targets(self, symbol: str):
        # NOTE: plan-gated on free tiers -> likely 403
        return self._get("stock/price-target", {"symbol": symbol})

    def recommendation_trends(self, symbol: str):
        # Works on free tier
        return self._get("stock/recommendation", {"symbol": symbol})

    def earnings_surprises(self, symbol: str):
        # Calendar endpoint (free), sparse between events
        return self._get("calendar/earnings", {"symbol": symbol})

# -------------------------
# FMP client
# -------------------------

class FMPClient:
    BASE = "https://financialmodelingprep.com"
    def __init__(self):
        self.key = os.environ.get("FMP_API_KEY", "")

    def analyst_estimates_stable(self, symbol: str, retries: int = 3) -> tuple[list | dict | None, int, str | None]:
        """
        Returns (json, status_code, error_text)
          - On success: json is list, status=200, error_text=None
          - On error:   json None, status is HTTP, error_text contains body
        Retries on transient status (429/5xx/timeout).
        402/403 etc. are not retried.
        """
        url = f"{self.BASE}/stable/analyst-estimates"
        params = {"symbol": symbol, "period": "annual", "limit": 10, "apikey": self.key}

        attempt = 1
        while True:
            data, status, body = _http_get_json(url, params)
            if status == 200 and data:
                return data, status, None
            if _retryable(status) and attempt < retries:
                _sleep_backoff(attempt)
                attempt += 1
                continue
            return data, status, body

# -------------------------
# yfinance helpers
# -------------------------

def yfinance_last_and_prev_close(symbol: str) -> Dict[str, float] | None:
    """
    Returns {'last_close': float, 'prev_close': float} using 2 days of daily data.
    """
    try:
        t = yf.Ticker(symbol.replace(".", "-"))
        # Pull last ~3 trading days for safety
        df = t.history(period="5d", interval="1d", auto_adjust=False)
        closes = df["Close"].dropna().tolist()
        if len(closes) == 0:
            return None
        last_close = float(closes[-1])
        prev_close = float(closes[-2]) if len(closes) >= 2 else None
        return {"last_close": last_close, "prev_close": prev_close}
    except Exception:
        return None

def yfinance_targets(symbol: str) -> Dict[str, Any] | None:
    """
    Returns consensus target fields from Yahoo 'info'.
    {
      'median': float, 'mean': float, 'high': float, 'low': float,
      'analystCount': int, 'asof': ISO8601
    }
    """
    try:
        t = yf.Ticker(symbol.replace(".", "-"))
        info = t.info or {}
        out = {
            "median": info.get("targetMedianPrice"),
            "mean": info.get("targetMeanPrice"),
            "high": info.get("targetHighPrice"),
            "low": info.get("targetLowPrice"),
            "analystCount": info.get("numberOfAnalystOpinions"),
            "asof": datetime_utc_iso()
        }
        # If everything is None, treat as missing
        if all(out[k] is None for k in ("median","mean","high","low","analystCount")):
            return None
        return out
    except Exception:
        return None

def datetime_utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

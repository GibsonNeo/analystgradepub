#!/usr/bin/env python3
from __future__ import annotations
import os, time, requests, traceback
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import yfinance as yf

USER_AGENT = "AnalystGrader/1.0 (free research; contact: local user)"

class FinnhubClient:
    def __init__(self, api_key: Optional[str] = None, throttle_s: float = 0.0):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self.base = "https://finnhub.io/api/v1"
        self.throttle_s = throttle_s

    def _get(self, path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if not self.api_key:
                return None
            url = self.base + path
            headers = {"User-Agent": USER_AGENT}
            params = {**params, "token": self.api_key}
            r = requests.get(url, params=params, headers=headers, timeout=30)
            time.sleep(self.throttle_s)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def price_targets(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._get("/stock/price-target", {"symbol": symbol})

    def recommendation_trends(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._get("/stock/recommendation", {"symbol": symbol})

    def earnings_surprises(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Finnhub /calendar/earnings?symbol=; contains surprise
        return self._get("/calendar/earnings", {"symbol": symbol})

    def eps_estimates(self, symbol: str) -> Optional[Dict[str, Any]]:
        # EPS estimates: /stock/eps?symbol= (note: exact endpoint names may vary by plan)
        return self._get("/stock/eps-estimate", {"symbol": symbol})

    def revenue_estimates(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._get("/stock/revenue-estimate", {"symbol": symbol})

class FMPClient:
    def __init__(self, api_key: Optional[str] = None, throttle_s: float = 0.0):
        self.api_key = api_key or os.getenv("FMP_API_KEY", "")
        self.base = "https://financialmodelingprep.com/api/v4"
        self.v3 = "https://financialmodelingprep.com/api/v3"
        self.throttle_s = throttle_s

    def _get(self, url: str, params: Dict[str, Any]) -> Optional[Any]:
        try:
            if not self.api_key:
                return None
            headers = {"User-Agent": USER_AGENT}
            params = {**params, "apikey": self.api_key}
            r = requests.get(url, params=params, headers=headers, timeout=30)
            time.sleep(self.throttle_s)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def price_target_consensus(self, symbol: str) -> Optional[Any]:
        # v4 price target consensus (structure depends on FMP; keep flexible)
        return self._get(f"{self.base}/price-target-consensus", {"symbol": symbol})

    def analyst_estimates(self, symbol: str) -> Optional[Any]:
        # v3/analyst-estimates?symbol=
        return self._get(f"{self.v3}/analyst-estimates/{symbol}", {})

def yfinance_last_and_prev_close(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        t = yf.Ticker(symbol)
        # Prefer history to derive last & previous close reliably
        hist = t.history(period="5d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        closes = hist["Close"].dropna()
        if closes.shape[0] == 0:
            return None
        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if closes.shape[0] >= 2 else None
        return {"last_close": last_close, "prev_close": prev_close, "asof": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
    except Exception:
        return None

# agents/market_data_agent.py
"""
Stateless MarketDataAgent microservice that batches ticker requests and
uses yfinance for all price, volume, and fundamentals retrieval. Returns
structured MarketDataResponse objects containing latest price, intraday
deltas, volume, and basic fundamentals.

Usage:
    pip install yfinance pandas
    from agents.market_data_agent import MarketDataAgent
    agent = MarketDataAgent(concurrency=6, per_ticker_sleep=0.05)
    out = agent.fetch(tickers, fields, portfolio=portfolio_rows)
"""
from __future__ import annotations
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd

logger = logging.getLogger("MarketDataAgent")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

INDIA_TZ = ZoneInfo("Asia/Kolkata")


@dataclass
class MarketDataResponse:
    ticker: str
    price: Optional[float] = None
    price_time: Optional[str] = None
    change_1h: Optional[float] = None
    pct_change_1h: Optional[float] = None
    change_24h: Optional[float] = None
    pct_change_24h: Optional[float] = None
    change_7d: Optional[float] = None
    pct_change_7d: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    fundamentals: Optional[Dict[str, Any]] = None
    holding_value: Optional[float] = None
    position_pct: Optional[float] = None
    data_source: str = "yfinance"
    as_of: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Drop keys with None to keep payload compact (optional)
        return {k: v for k, v in data.items() if v is not None or k in {"ticker", "data_source"}}


@dataclass
class MarketDataAgent:
    per_ticker_sleep: float = 0.02   # small pause after each fetch in worker
    concurrency: int = 4             # number of parallel workers (tune: 3-8 recommended)
    request_timeout_seconds: int = 15
    # history strategy tuned for reliable 1h/24h/7d computations
    hist_attempts: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.hist_attempts is None:
            self.hist_attempts = [
                {"interval": "1m", "period": "2d"},   # best for 1h
                {"interval": "60m", "period": "8d"},  # for 24h/7d lookbacks with hourly resolution
                {"interval": "1d", "period": "30d"},  # daily fallback for 7d and fundamentals
            ]

    def fetch(
        self,
        tickers: List[str],
        fields: Optional[List[str]] = None,
        portfolio: Optional[List[Dict[str, Any]]] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch market data for an arbitrary-length tickers list.
        Returns list[dict] – one dict per ticker (order follows unique normalized tickers).
        """
        now = as_of or datetime.now(timezone.utc)
        normalized_fields = self._normalize_fields(fields or [])
        tickers_norm = self._normalize_tickers(tickers)
        portfolio_map = self._build_portfolio_map(portfolio) if portfolio else {}
        portfolio_total = self._compute_portfolio_total(portfolio_map)

        results_map: Dict[str, Dict[str, Any]] = {}
        # Use ThreadPoolExecutor for concurrency
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            future_to_ticker = {
                ex.submit(self._fetch_one, t, normalized_fields, portfolio_map, portfolio_total, now): t
                for t in tickers_norm
            }
            for fut in as_completed(future_to_ticker):
                t = future_to_ticker[fut]
                try:
                    res = fut.result(timeout=self.request_timeout_seconds)
                except Exception as e:
                    logger.debug("fetch failed for %s: %s", t, e, exc_info=True)
                    holding = portfolio_map.get(self._portfolio_key(t), {})
                    res = MarketDataResponse(
                        ticker=t,
                        holding_value=holding.get("holding_value"),
                        data_source="yfinance",
                        as_of=now.isoformat(),
                        error=str(e),
                    )
                results_map[t] = res.to_dict()

        # Preserve original deduped order
        ordered_results = [results_map[t] for t in tickers_norm]
        return ordered_results

    # -------------------------
    # single-ticker worker
    # -------------------------
    def _fetch_one(self, t: str, fields: set, portfolio_map: Dict[str, Dict[str, Any]],
                   portfolio_total: Optional[float], as_of: datetime) -> MarketDataResponse:
        item = MarketDataResponse(ticker=t, data_source="yfinance", as_of=as_of.isoformat())

        try:
            tk = yf.Ticker(t)

            try:
                info = tk.get_info()
            except Exception:
                info = {}

            if info:
                if "volume" in fields:
                    item.volume = item.volume or self._safe_int(info.get("volume"))
                item.market_cap = item.market_cap or self._safe_float(info.get("marketCap") or info.get("market_cap"))
                if "fundamentals" in fields:
                    fundamentals: Dict[str, Any] = {}
                    if info.get("trailingPE") is not None:
                        fundamentals["pe"] = self._safe_float(info.get("trailingPE"))
                    if info.get("epsTrailingTwelveMonths") is not None:
                        fundamentals["eps"] = self._safe_float(info.get("epsTrailingTwelveMonths"))
                    if info.get("trailingEps") is not None and "eps" not in fundamentals:
                        fundamentals["eps"] = self._safe_float(info.get("trailingEps"))
                    if fundamentals:
                        item.fundamentals = fundamentals

            hist = None
            for attempt in self.hist_attempts:
                try:
                    hist = tk.history(interval=attempt["interval"], period=attempt["period"], auto_adjust=False)
                    if isinstance(hist, pd.DataFrame) and not hist.empty and len(hist.index) >= 2:
                        break
                except Exception:
                    hist = None
                    continue

            if isinstance(hist, pd.DataFrame) and not hist.empty:
                hist = hist.sort_index()
                last_idx = hist["Close"].last_valid_index()
                last_price = float(hist.loc[last_idx, "Close"])
                if isinstance(last_idx, pd.Timestamp):
                    ts = last_idx.tz_convert("UTC") if last_idx.tzinfo is not None else last_idx.tz_localize(timezone.utc)
                    try:
                        india_ts = ts.tz_convert(INDIA_TZ)
                    except Exception:
                        india_ts = ts
                    item.price_time = india_ts.isoformat()
                else:
                    item.price_time = str(last_idx)
                item.price = round(last_price, 4)

                if "volume" in fields and item.volume is None:
                    try:
                        if "Volume" in hist.columns:
                            item.volume = int(hist.loc[last_idx, "Volume"])
                    except Exception:
                        pass

                def price_at_lookback(seconds_back: int) -> Optional[Tuple[float, pd.Timestamp]]:
                    target = last_idx - pd.Timedelta(seconds=seconds_back)
                    earlier = hist[hist.index <= target]
                    if not earlier.empty:
                        idx = earlier.index[-1]
                        return float(hist.loc[idx, "Close"]), idx
                    return None

                if "1h" in fields:
                    res = price_at_lookback(3600)
                    if res:
                        p_then, _ = res
                        item.change_1h = round(item.price - p_then, 6)
                        item.pct_change_1h = round((item.change_1h / p_then) * 100, 6) if p_then != 0 else None
                if "24h" in fields:
                    res = price_at_lookback(86400)
                    if res:
                        p_then, _ = res
                        item.change_24h = round(item.price - p_then, 6)
                        item.pct_change_24h = round((item.change_24h / p_then) * 100, 6) if p_then != 0 else None
                if "7d" in fields:
                    res = price_at_lookback(7 * 86400)
                    if res:
                        p_then, _ = res
                        item.change_7d = round(item.price - p_then, 6)
                        item.pct_change_7d = round((item.change_7d / p_then) * 100, 6) if p_then != 0 else None
            else:
                if info and info.get("regularMarketPrice") is not None:
                    item.price = self._safe_float(info.get("regularMarketPrice"))
                    item.error = "insufficient_history_for_deltas"
                else:
                    item.error = "no_price_data"

            if item.price is not None and item.price_time is None:
                try:
                    item.price_time = as_of.astimezone(INDIA_TZ).isoformat()
                except Exception:
                    item.price_time = as_of.isoformat()

            if info:
                if item.market_cap is None:
                    item.market_cap = self._safe_float(info.get("marketCap") or info.get("market_cap"))
                if "volume" in fields and item.volume is None:
                    item.volume = self._safe_int(info.get("volume"))

            p = portfolio_map.get(self._portfolio_key(t))
            if p:
                qty = p.get("qty")
                hv = p.get("holding_value")
                if hv is not None:
                    try:
                        item.holding_value = round(float(hv), 4)
                    except Exception:
                        item.holding_value = None
                elif qty is not None and item.price is not None:
                    try:
                        item.holding_value = round(float(qty) * float(item.price), 4)
                    except Exception:
                        item.holding_value = None
                if item.holding_value is not None and portfolio_total:
                    try:
                        item.position_pct = round(float(item.holding_value) / float(portfolio_total), 6)
                    except Exception:
                        item.position_pct = None

        except Exception as exc:
            logger.debug("yfinance error for %s: %s", t, exc, exc_info=True)
            item.error = str(exc)

        # gentle pause to avoid bursts
        time.sleep(self.per_ticker_sleep)
        return item

    # -------------------------
    # helpers
    # -------------------------
    def _normalize_fields(self, fields: List[str]) -> set:
        default_fields = {"price", "1h", "24h", "7d", "volume", "fundamentals"}
        if not fields:
            return default_fields
        alias_map = {
            "price": "price",
            "regularmarketprice": "price",
            "1h": "1h",
            "change_1h": "1h",
            "pct_change_1h": "1h",
            "24h": "24h",
            "change_24h": "24h",
            "pct_change_24h": "24h",
            "7d": "7d",
            "change_7d": "7d",
            "pct_change_7d": "7d",
            "volume": "volume",
            "fundamentals": "fundamentals",
        }
        normalized = set()
        for field in fields:
            if not field:
                continue
            key = str(field).strip().lower().replace(" ", "")
            normalized.add(alias_map.get(key, key))
        normalized.add("price")  # price always required for downstream calcs
        return default_fields.intersection(normalized)

    def _normalize_tickers(self, tickers: List[str]) -> List[str]:
        out = []
        seen = set()
        for t in tickers:
            if not t:
                continue
            key = str(t).strip().upper()
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def _portfolio_key(self, ticker: str) -> str:
        k = ticker.strip().upper()
        if ":" in k:
            k = k.split(":", 1)[1]
        if k.endswith(".NS") or k.endswith(".NSE"):
            k = k.split(".")[0]
        return k

    def _build_portfolio_map(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        m: Dict[str, Dict[str, Any]] = {}
        for row in portfolio:
            tk = (row.get("ticker") or "").strip().upper()
            if not tk:
                continue
            key = self._portfolio_key(tk)
            m[key] = {"qty": row.get("qty"), "holding_value": row.get("holding_value")}
        return m

    def _compute_portfolio_total(self, portfolio_map: Dict[str, Dict[str, Any]]) -> Optional[float]:
        total = 0.0
        found = False
        for v in portfolio_map.values():
            hv = v.get("holding_value")
            if hv is not None:
                try:
                    total += float(hv)
                    found = True
                except:
                    continue
        return total if found else None

    def _safe_float(self, v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    def _safe_int(self, v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            return int(float(v))
        except Exception:
            return None

MarketDataAgentYFinance = MarketDataAgent  # backwards compatibility alias


if __name__ == "__main__":
    # Minimal CLI runner for demonstration / manual smoke test
    agent = MarketDataAgent(concurrency=2, per_ticker_sleep=0.05)

    # Load tickers from the portfolio file used in ingest_agent
    portfolio_file = "data/samples/large_portfolio.xlsx"
    tickers: List[str] = []
    try:
        df = pd.read_excel(portfolio_file)
        ticker_col = None
        for col in df.columns:
            if isinstance(col, str) and col.strip().lower() in ["ticker", "symbol"]:
                ticker_col = col
                break
        if ticker_col:
            tickers = df[ticker_col].dropna().astype(str).str.strip().tolist()
        else:
            # Some sample files have instructions in the header row; retry with header=None
            df_raw = pd.read_excel(portfolio_file, header=None)
            first_col = df_raw.iloc[:, 0].dropna().astype(str).str.strip()
            tickers = [
                val for val in first_col
                if val and not val.lower().startswith("python ")
                and val.upper() != "QTY"
                and not val.lower().startswith("enter ")
            ]
    except Exception as cli_err:
        logger.warning("Could not auto-detect tickers from %s: %s", portfolio_file, cli_err)
        tickers = []

    fields = ["price", "1h", "24h", "7d", "volume", "fundamentals"]
    print(f"Fetching market data for: {tickers}")
    result = agent.fetch(tickers, fields)
    import json
    print(json.dumps(result, indent=2, default=str))

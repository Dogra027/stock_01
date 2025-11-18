# file: agents/market_data_agent_yfinance_parallel.py
"""
Scalable yfinance MarketDataAgent (Option A, parallelized).

Usage:
    pip install yfinance pandas
    from agents.market_data_agent_yfinance_parallel import MarketDataAgentYFinance
    agent = MarketDataAgentYFinance(concurrency=6, per_ticker_sleep=0.05)
    out = agent.fetch(tickers, fields, portfolio=portfolio_rows)
"""
from __future__ import annotations
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
import pandas as pd

# Import Alpha Vantage provider (handle both direct and package imports)
try:
    from agents.alpha_vantage_provider import AlphaVantageProvider
except ImportError:
    from alpha_vantage_provider import AlphaVantageProvider

logger = logging.getLogger("MarketDataAgentYFinanceParallel")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


@dataclass
class MarketDataAgentYFinance:
    per_ticker_sleep: float = 0.02   # small pause after each fetch in worker
    concurrency: int = 4             # number of parallel workers (tune: 3-8 recommended)
    request_timeout_seconds: int = 15
    # history strategy tuned for reliable 1h/24h/7d computations
    hist_attempts: List[Dict[str, str]] = None
    use_alpha_vantage: bool = True   # Enable Alpha Vantage fallback

    def __post_init__(self):
        if self.hist_attempts is None:
            self.hist_attempts = [
                {"interval": "1m", "period": "2d"},   # best for 1h
                {"interval": "60m", "period": "8d"},  # for 24h/7d lookbacks with hourly resolution
                {"interval": "1d", "period": "30d"},  # daily fallback for 7d and fundamentals
            ]
        # Initialize Alpha Vantage provider if enabled
        if self.use_alpha_vantage:
            self.alpha_vantage = AlphaVantageProvider()
        else:
            self.alpha_vantage = None

    def fetch(
        self,
        tickers: List[str],
        fields: List[str],
        portfolio: Optional[List[Dict[str, Any]]] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch market data for an arbitrary-length tickers list.
        Returns list[dict] – one dict per ticker (order follows unique normalized tickers).
        """
        now = as_of or datetime.now(timezone.utc)
        tickers_norm = self._normalize_tickers(tickers)
        portfolio_map = self._build_portfolio_map(portfolio) if portfolio else {}
        portfolio_total = self._compute_portfolio_total(portfolio_map)

        results_map: Dict[str, Dict[str, Any]] = {}
        # Use ThreadPoolExecutor for concurrency
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            future_to_ticker = {
                ex.submit(self._fetch_one, t, fields, portfolio_map, portfolio_total, now): t
                for t in tickers_norm
            }
            for fut in as_completed(future_to_ticker):
                t = future_to_ticker[fut]
                try:
                    res = fut.result(timeout=self.request_timeout_seconds)
                except Exception as e:
                    logger.debug("fetch failed for %s: %s", t, e, exc_info=True)
                    res = {
                        "ticker": t,
                        "price": None,
                        "price_time": None,
                        "change_1h": None,
                        "pct_change_1h": None,
                        "change_24h": None,
                        "pct_change_24h": None,
                        "change_7d": None,
                        "pct_change_7d": None,
                        "volume": None,
                        "market_cap": None,
                        "fundamentals": None,
                        "holding_value": portfolio_map.get(self._portfolio_key(t), {}).get("holding_value"),
                        "position_pct": None,
                        "data_source": "yfinance",
                        "as_of": now.isoformat(),
                        "error": str(e),
                    }
                results_map[t] = res

        # Preserve original deduped order
        ordered_results = [results_map[t] for t in tickers_norm]
        return ordered_results

    # -------------------------
    # single-ticker worker
    # -------------------------
    def _fetch_one(self, t: str, fields: List[str], portfolio_map: Dict[str, Dict[str, Any]],
                   portfolio_total: Optional[float], as_of: datetime) -> Dict[str, Any]:
        item = {
            "ticker": t,
            "price": None,
            "price_time": None,
            "change_1h": None,
            "pct_change_1h": None,
            "change_24h": None,
            "pct_change_24h": None,
            "change_7d": None,
            "pct_change_7d": None,
            "volume": None,
            "market_cap": None,
            "fundamentals": None,
            "holding_value": None,
            "position_pct": None,
            "data_source": "yfinance",
            "as_of": as_of.isoformat(),
            "error": None,
        }

        try:
            # instantiate Ticker
            tk = yf.Ticker(t)

            # get info (best-effort)
            try:
                info = tk.get_info()
            except Exception:
                info = {}

            # Map info -> fundamentals/market values
            if info:
                item["volume"] = self._safe_int(info.get("volume"))
                item["market_cap"] = self._safe_float(info.get("marketCap") or info.get("market_cap"))
                fundamentals = {}
                if info.get("trailingPE") is not None:
                    fundamentals["pe"] = self._safe_float(info.get("trailingPE"))
                if info.get("epsTrailingTwelveMonths") is not None:
                    fundamentals["eps"] = self._safe_float(info.get("epsTrailingTwelveMonths"))
                if info.get("trailingEps") is not None and "eps" not in fundamentals:
                    fundamentals["eps"] = self._safe_float(info.get("trailingEps"))
                if fundamentals:
                    item["fundamentals"] = fundamentals

            # fetch history using prioritized attempts
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
                # price_time: ensure ISO UTC
                if isinstance(last_idx, pd.Timestamp):
                    ts = last_idx.tz_convert("UTC") if last_idx.tzinfo is not None else last_idx.tz_localize(timezone.utc)
                    item["price_time"] = ts.isoformat()
                else:
                    item["price_time"] = str(last_idx)
                item["price"] = round(last_price, 4)

                # fill volume if present
                if item["volume"] is None:
                    try:
                        if "Volume" in hist.columns:
                            item["volume"] = int(hist.loc[last_idx, "Volume"])
                    except Exception:
                        pass

                # helper: price at lookback seconds
                def price_at_lookback(seconds_back: int) -> Optional[Tuple[float, pd.Timestamp]]:
                    target = last_idx - pd.Timedelta(seconds=seconds_back)
                    earlier = hist[hist.index <= target]
                    if not earlier.empty:
                        idx = earlier.index[-1]
                        return float(hist.loc[idx, "Close"]), idx
                    # if no earlier candle, treat as insufficient history -> return None
                    return None

                # compute deltas only if requested
                if "1h" in fields:
                    res = price_at_lookback(3600)
                    if res:
                        p_then, _ = res
                        item["change_1h"] = round(item["price"] - p_then, 6)
                        item["pct_change_1h"] = round((item["change_1h"] / p_then) * 100, 6) if p_then != 0 else None
                if "24h" in fields:
                    res = price_at_lookback(86400)
                    if res:
                        p_then, _ = res
                        item["change_24h"] = round(item["price"] - p_then, 6)
                        item["pct_change_24h"] = round((item["change_24h"] / p_then) * 100, 6) if p_then != 0 else None
                if "7d" in fields:
                    res = price_at_lookback(7 * 86400)
                    if res:
                        p_then, _ = res
                        item["change_7d"] = round(item["price"] - p_then, 6)
                        item["pct_change_7d"] = round((item["change_7d"] / p_then) * 100, 6) if p_then != 0 else None
            else:
                # fallback to regularMarketPrice from info if history missing
                if info and info.get("regularMarketPrice") is not None:
                    item["price"] = self._safe_float(info.get("regularMarketPrice"))
                    item["error"] = "insufficient_history_for_deltas"
                else:
                    item["error"] = "no_price_data"
            
            # Try Alpha Vantage as fallback if yfinance couldn't provide price changes
            if self.alpha_vantage and item.get("price") is not None:
                if (item.get("change_1h") is None or item.get("change_24h") is None or item.get("change_7d") is None):
                    try:
                        av_data = self.alpha_vantage.extract_price_changes(t, interval="5min")
                        if av_data:
                            # Fill missing fields from Alpha Vantage
                            if item.get("change_1h") is None and av_data.get("change_1h") is not None:
                                item["change_1h"] = av_data["change_1h"]
                                item["pct_change_1h"] = av_data.get("pct_change_1h")
                            if item.get("change_24h") is None and av_data.get("change_24h") is not None:
                                item["change_24h"] = av_data["change_24h"]
                                item["pct_change_24h"] = av_data.get("pct_change_24h")
                            if item.get("change_7d") is None and av_data.get("change_7d") is not None:
                                item["change_7d"] = av_data["change_7d"]
                                item["pct_change_7d"] = av_data.get("pct_change_7d")
                            # Update data source to indicate fallback was used
                            if item.get("data_source") == "yfinance":
                                item["data_source"] = "yfinance + alpha_vantage"
                    except Exception as e:
                        logger.debug(f"Alpha Vantage fallback failed for {t}: {e}")

            # final fill from info
            if info:
                if item.get("market_cap") is None:
                    item["market_cap"] = self._safe_float(info.get("marketCap") or info.get("market_cap"))
                if item.get("volume") is None:
                    item["volume"] = self._safe_int(info.get("volume"))

            # compute holding_value & position_pct
            p = portfolio_map.get(self._portfolio_key(t))
            if p:
                qty = p.get("qty")
                hv = p.get("holding_value")
                if hv is not None:
                    try:
                        item["holding_value"] = round(float(hv), 4)
                    except:
                        item["holding_value"] = None
                elif qty is not None and item.get("price") is not None:
                    try:
                        item["holding_value"] = round(float(qty) * float(item["price"]), 4)
                    except:
                        item["holding_value"] = None
                if item.get("holding_value") is not None and portfolio_total:
                    try:
                        item["position_pct"] = round(float(item["holding_value"]) / float(portfolio_total), 6)
                    except:
                        item["position_pct"] = None

        except Exception as exc:
            logger.debug("yfinance error for %s: %s", t, exc, exc_info=True)
            item["error"] = str(exc)

        # gentle pause to avoid bursts
        time.sleep(self.per_ticker_sleep)
        return item

    # -------------------------
    # helpers
    # -------------------------
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

if __name__ == "__main__":
    # Minimal CLI runner for demonstration
    agent = MarketDataAgentYFinance(concurrency=2, per_ticker_sleep=0.05)

    # Load tickers from the portfolio file used in ingest_agent
    portfolio_file = "data/samples/large_portfolio.xlsx"
    df = pd.read_excel(portfolio_file)
    ticker_col = None
    for col in df.columns:
        if col.lower() in ["ticker", "symbol"]:
            ticker_col = col
            break
    if ticker_col:
        tickers = df[ticker_col].dropna().unique().tolist()
    else:
        tickers = []  # fallback if no ticker column found

    fields = ["regularMarketPrice", "regularMarketChange", "regularMarketChangePercent"]
    print(f"Fetching market data for: {tickers}")
    result = agent.fetch(tickers, fields)
    import json
    print(json.dumps(result, indent=2, default=str))

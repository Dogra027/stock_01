# file: agents/filings.py
"""
FilingsAgent - fetch regulatory filings (US + stubbed India) and extract bullets.

Usage (with portfolio + LLM bullets):
    python agents/filings.py \
        --portfolio-file "/home/sahildogra/Desktop/ingest_agent/data/samples/portfolio_sample (1).xlsx" \
        --lookback-days 7 \
        --use-llm

Environment:
    GROQ_API_KEY=<your_key_here>

This script:
    - Loads tickers from an Excel/CSV portfolio file (expects a 'ticker' column)
    - For US tickers (AAPL, MSFT, GOOGL) fetches recent SEC EDGAR filings
    - For Indian tickers (e.g. RELIANCE.NS) uses a stub (no real regulator integration yet)
    - Optionally calls Groq LLM to produce bullets and refine impact tags
    - Outputs JSON:
        {
          "filings": [ ... ],
          "clusters": [ ... ]
        }
"""

from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv
load_dotenv()


# ---------- Data models ----------

@dataclass
class Filing:
    id: str
    ticker: str
    type: str
    filing_date: str           # YYYY-MM-DD
    filing_url: str
    short_summary: str
    impact_tag: str            # material | moderate | immaterial
    source: str                # sec_edgar / sebi / etc.
    bullets: Optional[List[str]] = None
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FilingCluster:
    id: str
    representative_headline: str
    filing_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- LLM helper (Groq) ----------

def _call_groq_chat(
    system_prompt: str,
    user_prompt: str,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> Optional[str]:
    """
    Low-level helper to call Groq Chat Completions API.

    Requires:
        export GROQ_API_KEY="sk_..."
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[llm] GROQ_API_KEY not set; skipping LLM call.")
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text
    except Exception as e:
        print(f"[llm] Error calling Groq API: {e}")
        return None


def llm_summarize_filing_bullets(
    filing: Filing,
) -> List[str]:
    """
    Use Groq LLM to produce 2–4 bullet points for a filing.
    If LLM call fails, fall back to simple template bullets.
    """
    # Fallback bullets if LLM is disabled/failed
    fallback = [
        f"{filing.ticker} filed {filing.type} on {filing.filing_date} (impact: {filing.impact_tag}).",
        f"Source: {filing.source}",
    ]

    # If we have no raw_text, just summarise from metadata
    text_for_model = filing.raw_text or ""
    meta = (
        f"Ticker: {filing.ticker}\n"
        f"Form type: {filing.type}\n"
        f"Filing date: {filing.filing_date}\n"
        f"Impact tag (heuristic): {filing.impact_tag}\n"
        f"URL: {filing.filing_url}\n\n"
        f"Raw filing text (may be empty):\n{text_for_model[:8000]}"
    )

    system_prompt = (
        "You are a financial analyst summarizing SEC/regulatory filings for a market news dashboard. "
        "Given filing metadata and optionally raw text, produce 2-4 SHORT bullet points for a portfolio manager.\n"
        "- Focus on: earnings, guidance, major corporate actions, management changes, legal/regulatory issues.\n"
        "- If the text has no details (e.g. Form 4 with no narrative), keep bullets generic but accurate.\n"
        "- Output JSON: {\"bullets\": [\"...\", \"...\"]} and nothing else."
    )

    user_prompt = meta

    resp_text = _call_groq_chat(system_prompt, user_prompt)
    if not resp_text:
        return fallback

    # Try to parse JSON with "bullets"
    try:
        parsed = json.loads(resp_text)
        bullets = parsed.get("bullets")
        if isinstance(bullets, list) and bullets:
            # Ensure strings and strip
            cleaned = [str(b).strip() for b in bullets if str(b).strip()]
            return cleaned or fallback
        return fallback
    except Exception:
        # Model might respond with text, not JSON; fallback
        return fallback


# ---------- Portfolio loading ----------

def _load_portfolio_tickers(portfolio_path: Optional[str]) -> List[str]:
    """
    Load tickers from a portfolio Excel/CSV file.

    Expects at least one column that looks like 'ticker'.
    If not found, uses the first column as tickers.

    Returns:
        tickers: list[str] (uppercased)
    """
    if not portfolio_path:
        print("[filings.py] No portfolio path provided.")
        return []

    path = Path(portfolio_path).expanduser()
    if not path.exists():
        print(f"[filings.py] Portfolio file not found: {path}")
        return []

    try:
        import pandas as pd
    except Exception as e:
        print(f"[filings.py] pandas not available: {e}")
        return []

    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        print(f"[filings.py] Error reading portfolio file: {e}")
        return []

    print(f"[filings.py] Portfolio shape: {df.shape}, columns: {list(df.columns)}")

    if df.empty:
        return []

    # Normalize column names
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    ticker_col = None

    for key, orig in lower_cols.items():
        if key in {"ticker", "symbol", "ticker_symbol"}:
            ticker_col = orig
            break
        if "ticker" in key or "symbol" in key:
            ticker_col = orig
            break

    if ticker_col is None:
        ticker_col = df.columns[0]

    tickers: List[str] = []
    for _, row in df.iterrows():
        tk = row.get(ticker_col)
        if tk is None:
            continue
        tk_str = str(tk).strip()
        if not tk_str:
            continue
        tickers.append(tk_str.upper())

    # Deduplicate but keep order
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    print(f"[filings.py] Loaded {len(uniq)} tickers from portfolio "
          f"(sample: {uniq[:8]})")
    return uniq


# ---------- FilingsAgent ----------

class FilingsAgent:
    def __init__(
        self,
        lookback_days: int = 30,
        max_per_ticker: int = 20,
        use_llm: bool = False,
    ):
        self.lookback_days = lookback_days
        self.max_per_ticker = max_per_ticker
        self.use_llm = use_llm

        # SEC requires a descriptive User-Agent
        self.sec_user_agent = (
            "FilingsAgent/1.0 (contact: your_email@example.com)"
        )

        # Very small built-in mapping: ticker -> CIK (str, zero-padded)
        self.us_ticker_to_cik: Dict[str, str] = {
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "GOOGL": "0001652044",
            "GOOG": "0001652044",
        }

    # ----- HTTP helpers -----

    def _http_get_json(self, url: str) -> Optional[Dict[str, Any]]:
        headers = {"User-Agent": self.sec_user_agent}
        try:
            print(f"[FilingsAgent._http_get_json] GET {url}")
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[FilingsAgent._http_get_json] Error: {e}")
            return None

    # ----- Fetch US filings (SEC EDGAR) -----

    def _fetch_us_filings(self, ticker: str) -> List[Filing]:
        ticker_u = ticker.upper()
        cik = self.us_ticker_to_cik.get(ticker_u)
        if not cik:
            print(f"[FilingsAgent._fetch_us_filings] No CIK mapping for {ticker_u}")
            return []

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = self._http_get_json(url)
        if not data:
            return []

        filings_raw = data.get("filings", {}).get("recent", {})
        forms = filings_raw.get("form", [])
        dates = filings_raw.get("filingDate", [])
        access_nums = filings_raw.get("accessionNumber", [])

        base_url = "https://www.sec.gov/ixviewer/doc?action=load&doc="
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)

        out: List[Filing] = []

        for form, date_str, acc in zip(forms, dates, access_nums):
            # Convert filingDate to datetime and filter by lookback
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                continue

            if dt < cutoff:
                # older than window
                continue

            # Build URL to the filing
            # EDGAR path: /Archives/edgar/data/CIK/ACCESSION_NUM(no dashes)/...
            acc_nodash = acc.replace("-", "")
            doc_path = f"/Archives/edgar/data/{int(cik)}/{acc_nodash}/"
            # Many filings expose a primary document at /.../primary_doc.xml or .htm etc.
            # But the ixviewer can figure it out from just the directory in many cases.
            filing_url = base_url + doc_path

            form_type = form
            impact_tag = self._score_impact(form_type)

            short_summary = form_type  # we will refine with LLM bullets
            fid = f"{ticker_u}-{form_type}-{date_str}-{hashlib.sha1(acc.encode()).hexdigest()[:10]}"

            filing = Filing(
                id=fid,
                ticker=ticker_u,
                type=form_type,
                filing_date=date_str,
                filing_url=filing_url,
                short_summary=short_summary,
                impact_tag=impact_tag,
                source="sec_edgar",
                raw_text=None,   # could be fetched & parsed later
            )
            out.append(filing)

            if len(out) >= self.max_per_ticker:
                break

        print(f"[FilingsAgent._fetch_us_filings] {ticker_u}: total={len(forms)}, "
              f"within_window={len(out)}")
        return out

    # ----- Fetch India filings (stub) -----

    def _fetch_india_filings(self, ticker: str) -> List[Filing]:
        """
        Fetch recent NSE corporate announcements for a given ticker using direct API.
        """
        import datetime
        symbol = ticker.replace('.NS', '').upper()
        url = "https://www.nseindia.com/api/corporate-filings-announcements?index=equities"
        headers = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) Chrome/142.0.0.0 Mobile Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
        }
        session = requests.Session()
        session.headers.update(headers)
        try:
            session.get("https://www.nseindia.com")
            resp = session.get(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[FilingsAgent._fetch_india_filings] NSE fetch error for {ticker}: {e}")
            return []
        entries = data if isinstance(data, list) else data.get("data") or data.get("rows") or []
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=self.lookback_days)
        filings = []
        for entry in entries:
            if entry.get("symbol") != symbol:
                continue
            sort_date = entry.get("sort_date")  # 'YYYY-MM-DD HH:MM:SS'
            try:
                dt = datetime.datetime.strptime(sort_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
            except Exception:
                continue
            if dt < cutoff:
                continue
            summary = entry.get("desc") or entry.get("attchmntText") or "NSE Announcement"
            impact = self._score_impact("announcement")
            filing_id = f"{symbol}-NSE-{sort_date}-{hashlib.sha1(str(entry).encode('utf-8')).hexdigest()[:10]}"
            filings.append(Filing(
                id=filing_id,
                ticker=ticker,
                type="NSE_ANNOUNCEMENT",
                filing_date=dt.strftime("%Y-%m-%d"),
                filing_url=entry.get("attchmntFile") or "",
                short_summary=summary,
                impact_tag=impact,
                source="nse",
                bullets=None,
                raw_text=None
            ))
            if len(filings) >= self.max_per_ticker:
                break
        return filings

    # ----- Impact scoring heuristics -----

    def _score_impact(self, form_type: str) -> str:
        """
        Very simple heuristic:
          - 8-K, 6-K, 10-K, 10-Q => material
          - 13D/G, proxy statements => moderate
          - 4, 144, misc => immaterial
        """
        ft = (form_type or "").upper()
        if ft.startswith(("8-K", "6-K", "10-K", "10-Q")):
            return "material"
        if ft.startswith(("SC 13D", "SC 13G", "DEFR14", "DFAN14", "PX14A6G")):
            return "moderate"
        return "immaterial"

    # ----- Clustering -----

    def _cluster_filings(self, filings: List[Filing]) -> List[FilingCluster]:
        """
        Simple clustering:
          - Group by (ticker, type, filing_date).
        """
        clusters_map: Dict[str, List[Filing]] = {}
        for f in filings:
            key = f"{f.ticker}::{f.type}::{f.filing_date}"
            clusters_map.setdefault(key, []).append(f)

        clusters: List[FilingCluster] = []
        for key, group in clusters_map.items():
            ticker, form_type, filing_date = key.split("::")
            headline = f"{ticker} {form_type} filed on {filing_date}"
            cid = hashlib.sha1(key.encode()).hexdigest()[:40]
            clusters.append(
                FilingCluster(
                    id=cid,
                    representative_headline=headline,
                    filing_ids=[g.id for g in group],
                )
            )
        return clusters

    # ----- Main run -----

    def run(
        self,
        tickers: List[str],
        lookback_days: Optional[int] = None,
        use_llm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if lookback_days is not None:
            self.lookback_days = lookback_days
        if use_llm is not None:
            self.use_llm = use_llm

        if not tickers:
            return {"filings": [], "clusters": []}

        universe = sorted(set(tickers))
        print(f"[FilingsAgent.run] Ticker universe ({len(universe)}): {universe[:10]}...")

        all_filings: List[Filing] = []

        for tk in universe:
            print(f"[FilingsAgent.run] Processing ticker: {tk}")
            if tk.endswith(".NS") or tk.endswith(".BO"):
                # assume Indian
                fs = self._fetch_india_filings(tk)
            else:
                # assume US
                fs = self._fetch_us_filings(tk)

            # Optional LLM summarization
            if self.use_llm and fs:
                for f in fs:
                    bullets = llm_summarize_filing_bullets(f)
                    f.bullets = bullets

            all_filings.extend(fs)
            # Be gentle with EDGAR
            time.sleep(0.2)

        clusters = self._cluster_filings(all_filings)

        return {
            "filings": [f.to_dict() for f in all_filings],
            "clusters": [c.to_dict() for c in clusters],
        }


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FilingsAgent on portfolio tickers")
    parser.add_argument(
        "--portfolio-file",
        help="Portfolio Excel/CSV file path (with 'ticker' column)",
        default=None,
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Lookback window in days (default: 30)",
    )
    parser.add_argument(
        "--max-per-ticker",
        type=int,
        default=20,
        help="(Optional) Limit number of filings per ticker (default: 20)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Groq LLM to generate bullets for filings",
    )

    args = parser.parse_args()

    # Auto-discover portfolio if not provided
    portfolio_file = args.portfolio_file
    if not portfolio_file:
        project_root = Path(__file__).parent.parent
        candidates = [
            project_root / "data" / "samples" / "portfolio_sample.xlsx",
            project_root / "data" / "samples" / "portfolio_sample (1).xlsx",
        ]
        resolved = None
        for c in candidates:
            if c.exists():
                resolved = str(c)
                break
        print(f"[filings.py] Resolved portfolio file: {resolved}")
        portfolio_file = resolved

    if not portfolio_file:
        print(json.dumps({"filings": [], "clusters": []}, indent=2))
    else:
        tickers = _load_portfolio_tickers(portfolio_file)
        agent = FilingsAgent(
            lookback_days=args.lookback_days,
            max_per_ticker=args.max_per_ticker,
            use_llm=args.use_llm,
        )
        out = agent.run(
            tickers=tickers,
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))



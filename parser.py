# ingest/parser.py
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import re

from .models import PortfolioRow
from .column_detector import detect_columns
from .symbol_mapper import normalize_ticker
from .sector_mapper import assign_sector


# Helpers for cleaning numeric fields
def _parse_number(x: Any) -> float | None:
    """Try to coerce strings like '1,000', '₹10,000', '10k' into float. Return None if empty / invalid."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "n/a", "-"}:
        return None

    # remove currency symbols and parentheses
    s = re.sub(r"[^\d\.\-kKmM]", "", s)

    # handle shorthand like 10k or 2.5M
    m = re.match(r"^(-?\d+(\.\d+)?)([kKmM])?$", s)
    if not m:
        try:
            return float(s)
        except Exception:
            return None

    num = float(m.group(1))
    suffix = m.group(3)
    if suffix:
        if suffix.lower() == "k":
            num *= 1_000
        elif suffix.lower() == "m":
            num *= 1_000_000
    return num


def _normalize_ticker(t: Any) -> str | None:
    if t is None:
        return None
    s = str(t).strip()
    if s == "":
        return None
    return s.upper()


def _normalize_sector(s: Any) -> str | None:
    if s is None:
        return None
    t = str(s).strip()
    return t.title() if t != "" else None


def row_to_portfolio_row(raw: Dict[str, Any]) -> PortfolioRow:
    """
    Accepts a raw dict (from pandas row) and returns a validated PortfolioRow.
    Expected raw keys: ticker, qty, value, sector, cost, notes (but tolerant to other keys)
    """
    # tolerate different column names by lowercasing keys
    low = {k.strip().lower(): v for k, v in raw.items()}

    ticker = _normalize_ticker(low.get("ticker") or low.get("symbol") or low.get("tickr"))
    if not ticker:
        # Let pydantic raise error if ticker missing
        ticker = ""

    qty = _parse_number(low.get("qty") or low.get("quantity") or low.get("shares"))
    value = _parse_number(low.get("value") or low.get("holding_value") or low.get("amount"))
    cost = _parse_number(low.get("cost") or low.get("price") or low.get("buy_price"))
    sector = _normalize_sector(low.get("sector") or low.get("industry"))
    notes = low.get("notes") or low.get("note") or ""

    # Create validated model (pydantic will coerce types)
    validated = PortfolioRow.model_validate({
        "ticker": ticker,
        "qty": qty,
        "value": value,
        "sector": sector,
        "cost": cost,
        "notes": notes,
    })

    return validated


def parse_file(file_path: str, use_symbol_mapping: bool = True, 
               use_sector_mapping: bool = True) -> Tuple[List[PortfolioRow], Dict[str, Any], List[str]]:
    """
    Read a CSV or Excel file, detect columns heuristically, and return validated PortfolioRow objects.
    
    Args:
        file_path: Path to CSV or Excel file
        use_symbol_mapping: Whether to normalize ticker symbols
        use_sector_mapping: Whether to assign sectors automatically
        
    Returns:
        Tuple of (rows, column_mapping) where column_mapping shows detected columns
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(file_path)

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p, dtype=str)  # read as strings, we'll coerce
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(p, dtype=str)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx/.xls")

    # Detect columns heuristically
    column_mapping = detect_columns(list(df.columns))
    
    # Map raw rows using detected columns
    rows: List[PortfolioRow] = []
    parse_errors = []  # Track parse errors for validation report
    
    for idx, raw in enumerate(df.fillna("").to_dict(orient="records"), start=1):
        # Skip completely empty rows
        if not any(str(v).strip() for v in raw.values()):
            continue
        
        try:
            # Map detected columns to field names
            mapped_raw = {}
            for field, col_name in column_mapping.items():
                if col_name and col_name in raw:
                    mapped_raw[field] = raw[col_name]
            
            # Get ticker early to check if it's empty
            ticker_col = column_mapping.get("ticker")
            raw_ticker = raw.get(ticker_col or "ticker") or mapped_raw.get("ticker") or ""
            
            # Skip rows with empty ticker (required field)
            if not str(raw_ticker).strip():
                parse_errors.append(f"Row {idx}: Empty ticker - skipped")
                continue
            
            # Create row with column mapping
            pr = row_to_portfolio_row_from_mapped(mapped_raw, raw, use_symbol_mapping, use_sector_mapping)
            rows.append(pr)
            
        except Exception as e:
            # For robustness, try to create a fallback row if we can extract a ticker
            mapped_raw = {}
            for field, col_name in column_mapping.items():
                if col_name and col_name in raw:
                    mapped_raw[field] = raw[col_name]
            
            # Try to extract ticker from various possible locations
            ticker_guess = (
                _normalize_ticker(raw.get(column_mapping.get("ticker") or "ticker")) or
                _normalize_ticker(mapped_raw.get("ticker")) or
                _normalize_ticker(raw.get("ticker")) or
                _normalize_ticker(raw.get("symbol")) or
                ""
            )
            
            # Only create fallback row if we have a valid ticker
            if ticker_guess and len(ticker_guess) > 0:
                try:
                    fallback = PortfolioRow.model_validate({
                        "ticker": ticker_guess,
                        "qty": _parse_number(mapped_raw.get("qty")),
                        "value": _parse_number(mapped_raw.get("value")),
                        "sector": None,
                        "cost": _parse_number(mapped_raw.get("cost")),
                        "notes": f"PARSE_ERROR: {str(e)} | raw_notes:{raw.get('notes', '')}"
                    })
                    rows.append(fallback)
                    parse_errors.append(f"Row {idx} ({ticker_guess}): Partial parse - {str(e)}")
                except Exception as fallback_error:
                    # Even fallback failed, just log the error
                    parse_errors.append(f"Row {idx} ({ticker_guess if ticker_guess else 'unknown'}): Parse failed - {str(e)} | Fallback failed - {str(fallback_error)}")
            else:
                # No valid ticker, skip this row
                parse_errors.append(f"Row {idx}: Empty ticker - skipped (error: {str(e)})")
    
    return rows, column_mapping, parse_errors


def row_to_portfolio_row_from_mapped(mapped: Dict[str, Any], raw: Dict[str, Any],
                                     use_symbol_mapping: bool = True,
                                     use_sector_mapping: bool = True) -> PortfolioRow:
    """
    Create PortfolioRow from mapped fields, with symbol and sector normalization.
    """
    # Get raw ticker
    raw_ticker = mapped.get("ticker") or raw.get("ticker") or raw.get("symbol") or raw.get("tickr")
    raw_ticker_str = str(raw_ticker).strip() if raw_ticker else ""
    
    # Normalize ticker if enabled
    if use_symbol_mapping and raw_ticker_str:
        normalized_ticker, error = normalize_ticker(raw_ticker_str)
        ticker = normalized_ticker if normalized_ticker else raw_ticker_str.upper()
    else:
        ticker = raw_ticker_str.upper() if raw_ticker_str else ""
    
    if not ticker:
        ticker = ""
    
    # Parse fields
    qty = _parse_number(mapped.get("qty"))
    value = _parse_number(mapped.get("value"))
    cost = _parse_number(mapped.get("cost"))
    notes = str(mapped.get("notes") or raw.get("notes") or raw.get("note") or "").strip()
    
    # Assign sector (priority: user provided > mapping > unknown)
    user_sector = _normalize_sector(mapped.get("sector"))
    if use_sector_mapping:
        sector = assign_sector(ticker, user_sector)
    else:
        sector = user_sector or "unknown"
    
    # If only ticker provided, set shares/holding_value to null (rely on MarketDataAgent)
    # This is already the case since qty and value default to None
    
    validated = PortfolioRow.model_validate({
        "ticker": ticker,
        "qty": qty,
        "value": value,
        "sector": sector,
        "cost": cost,
        "notes": notes,
    })
    
    return validated


def to_canonical_json(rows: List[PortfolioRow]) -> List[dict]:
    """
    Convert validated PortfolioRow objects to a canonical JSON structure:
    {
        "ticker": "AAPL",
        "sector": "Technology",
        "qty": 10.0,
        "cost": 120.0,
        "value": 1200.0,            # if value provided, use it; else compute qty*cost if possible
        "holding_value": 1200.0,    # always populated when derivable
        "notes": "..."
    }
    """
    out = []
    for r in rows:
        holding_value = None
        if r.value is not None:
            holding_value = r.value
        elif r.qty is not None and r.cost is not None:
            holding_value = r.qty * r.cost

        obj = {
            "ticker": r.ticker,
            "sector": r.sector,
            "qty": r.qty,
            "cost": r.cost,
            "value": r.value,
            "holding_value": holding_value,
            "notes": r.notes,
        }
        out.append(obj)
    return out


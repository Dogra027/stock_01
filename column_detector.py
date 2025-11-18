# ingest/column_detector.py
"""
Heuristic column detection to identify ticker, qty, value, sector, cost columns
from various possible column names in CSV/Excel files.
"""

from typing import Dict, List, Optional


# Possible column name variations for each field
TICKER_VARIANTS = {
    "ticker", "symbol", "tickr", "tkr", "stock", "stock_symbol", 
    "stock_ticker", "equity", "equity_symbol", "sym"
}

QTY_VARIANTS = {
    "qty", "quantity", "shares", "share_count", "num_shares", 
    "units", "number_of_shares", "position_size"
}

VALUE_VARIANTS = {
    "value", "holding_value", "amount", "total_value", "position_value",
    "market_value", "current_value", "worth"
}

SECTOR_VARIANTS = {
    "sector", "industry", "sector_name", "industry_name", "category",
    "sector_type", "industry_type", "business_sector"
}

COST_VARIANTS = {
    "cost", "price", "buy_price", "purchase_price", "avg_price", 
    "average_cost", "cost_per_share", "price_per_share", "unit_price"
}

NOTES_VARIANTS = {
    "notes", "note", "comments", "comment", "remarks", "remark", 
    "description", "desc", "memo"
}


def detect_columns(column_names: List[str]) -> Dict[str, Optional[str]]:
    """
    Heuristically detect which columns correspond to ticker, qty, value, sector, cost, notes.
    
    Args:
        column_names: List of column names from the file
        
    Returns:
        Dictionary mapping field names to detected column names (or None if not found)
        Example: {"ticker": "Symbol", "qty": "Quantity", "value": None, ...}
    """
    # Normalize column names for matching (lowercase, strip whitespace)
    normalized = {col.strip().lower(): col for col in column_names}
    
    # Detect each field
    ticker_col = _find_column(normalized, TICKER_VARIANTS)
    qty_col = _find_column(normalized, QTY_VARIANTS)
    value_col = _find_column(normalized, VALUE_VARIANTS)
    sector_col = _find_column(normalized, SECTOR_VARIANTS)
    cost_col = _find_column(normalized, COST_VARIANTS)
    notes_col = _find_column(normalized, NOTES_VARIANTS)
    
    return {
        "ticker": ticker_col,
        "qty": qty_col,
        "value": value_col,
        "sector": sector_col,
        "cost": cost_col,
        "notes": notes_col,
    }


def _find_column(normalized_cols: Dict[str, str], variants: set) -> Optional[str]:
    """
    Find a column that matches any of the given variants.
    
    Args:
        normalized_cols: Dictionary of normalized_col_name -> original_col_name
        variants: Set of possible column name variations to match
        
    Returns:
        Original column name if found, None otherwise
    """
    for norm_name, orig_name in normalized_cols.items():
        if norm_name in variants:
            return orig_name
    return None


def map_columns_to_fields(df_columns: List[str], detected: Dict[str, Optional[str]]) -> Dict[str, str]:
    """
    Map detected columns back to field names for row processing.
    
    Args:
        df_columns: Original DataFrame column names
        detected: Dictionary from detect_columns()
        
    Returns:
        Dictionary mapping field names to DataFrame column names
    """
    mapping = {}
    for field, detected_col in detected.items():
        if detected_col:
            mapping[field] = detected_col
    return mapping


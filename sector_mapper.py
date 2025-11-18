# ingest/sector_mapper.py
"""
Sector/industry mapping using static mapping or external APIs (e.g., IEX).
"""

from typing import Optional


# Static sector mapping for common tickers
STATIC_SECTOR_MAP = {
    # US Tech
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "AMZN": "E-Commerce",
    "META": "Technology",
    "NVDA": "Semiconductors",
    "INTC": "Semiconductors",
    "AMD": "Semiconductors",
    
    # US Finance
    "JPM": "Finance",
    "BAC": "Finance",
    "WFC": "Finance",
    "GS": "Finance",
    
    # US Consumer
    "TSLA": "Automotive",
    "NFLX": "Entertainment",
    "DIS": "Entertainment",
    
    # Indian Stocks
    "HDFCBANK": "Finance",
    "ICICIBANK": "Finance",
    "INFY": "IT",
    "TCS": "IT",
    "RELIANCE": "Energy",
    "ASIANPAINT": "Consumer",
    "MARUTI": "Automotive",
    "HUL": "FMCG",
    "ITC": "FMCG",
    
    # Crypto
    "BTC-USD": "Crypto",
    "ETH-USD": "Crypto",
    "BTC": "Crypto",
    "ETH": "Crypto",
}

# Sector aliases for common industry names
INDUSTRY_TO_SECTOR = {
    "information technology": "Technology",
    "it": "Technology",
    "software": "Technology",
    "tech": "Technology",
    "financial services": "Finance",
    "banking": "Finance",
    "consumer goods": "Consumer",
    "fmcg": "FMCG",
    "fast moving consumer goods": "FMCG",
    "consumer staples": "FMCG",
    "automotive": "Automotive",
    "auto": "Automotive",
    "energy": "Energy",
    "oil & gas": "Energy",
    "telecommunications": "Telecom",
    "telecom": "Telecom",
    "healthcare": "Healthcare",
    "pharmaceuticals": "Healthcare",
    "pharma": "Healthcare",
    "entertainment": "Entertainment",
    "media": "Entertainment",
    "cryptocurrency": "Crypto",
    "crypto": "Crypto",
}


def get_sector_for_ticker(ticker: str, use_iex: bool = False) -> Optional[str]:
    """
    Get sector for a ticker using static mapping or IEX API.
    
    Args:
        ticker: Normalized ticker symbol
        use_iex: Whether to use IEX Cloud API (requires API key)
        
    Returns:
        Sector name or None if not found
    """
    if not ticker:
        return None
    
    ticker_upper = ticker.upper()
    
    # Check static mapping first (exact match)
    if ticker_upper in STATIC_SECTOR_MAP:
        return STATIC_SECTOR_MAP[ticker_upper]
    
    # Try without exchange suffix (e.g., INFY.NS -> INFY)
    for suffix in [".NS", ".BO", ".L", ".T"]:
        if ticker_upper.endswith(suffix):
            base_ticker = ticker_upper[:-len(suffix)]
            if base_ticker in STATIC_SECTOR_MAP:
                return STATIC_SECTOR_MAP[base_ticker]
            break
    
    # Try IEX API if enabled (placeholder - would need API integration)
    if use_iex:
        sector = _get_sector_via_iex(ticker_upper)
        if sector:
            return sector
    
    # If not found, return None (will be set to "unknown" by caller)
    return None


def normalize_sector_name(sector_input: Optional[str]) -> Optional[str]:
    """
    Normalize a user-provided sector/industry name.
    
    Args:
        sector_input: Raw sector or industry name from input
        
    Returns:
        Normalized sector name or None if invalid
    """
    if not sector_input or not sector_input.strip():
        return None
    
    sector_lower = sector_input.strip().lower()
    
    # Check if it's a known alias
    if sector_lower in INDUSTRY_TO_SECTOR:
        return INDUSTRY_TO_SECTOR[sector_lower]
    
    # Title case if it looks like a valid sector name
    if len(sector_lower) > 0:
        return sector_input.strip().title()
    
    return None


def _get_sector_via_iex(ticker: str) -> Optional[str]:
    """
    Get sector via IEX Cloud API (placeholder implementation).
    
    In production, this would:
    1. Call IEX Cloud API to get company information
    2. Extract sector/industry
    3. Map to standard sector name
    
    For now, returns None.
    """
    # TODO: Implement IEX Cloud API integration
    # Would require: requests library, API key management
    return None


def assign_sector(ticker: str, user_sector: Optional[str] = None, 
                  use_iex: bool = False) -> str:
    """
    Assign sector to a ticker with fallback logic.
    
    Priority:
    1. User-provided sector (normalized)
    2. Static mapping by ticker
    3. IEX API lookup
    4. "unknown" if all else fails
    
    Args:
        ticker: Normalized ticker symbol
        user_sector: User-provided sector/industry name (optional)
        use_iex: Whether to try IEX API
        
    Returns:
        Sector name (or "unknown" if not found)
    """
    # Try user-provided sector first
    if user_sector:
        normalized = normalize_sector_name(user_sector)
        if normalized:
            return normalized
    
    # Try static mapping
    sector = get_sector_for_ticker(ticker, use_iex=False)
    if sector:
        return sector
    
    # Try IEX if enabled
    if use_iex:
        sector = _get_sector_via_iex(ticker)
        if sector:
            return sector
    
    # Fallback to unknown
    return "unknown"


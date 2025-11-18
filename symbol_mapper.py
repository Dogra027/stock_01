# ingest/symbol_mapper.py
"""
Symbol/ticker normalization using OpenFIGI API or internal mapping.
"""

from typing import Dict, Optional, Tuple
import re


# Internal mapping for common ticker variations and exchanges
INTERNAL_TICKER_MAP = {
    # Common variations
    "HDFC": "HDFCBANK",
    "ICICI": "ICICIBANK",
    "RELIANCE": "RELIANCE.NS",  # NSE suffix
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    
    # Cryptocurrency
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHEREUM": "ETH-USD",
    
    # Add more mappings as needed
}

# Exchange suffixes mapping
EXCHANGE_SUFFIXES = {
    ".NS": "NSE",  # National Stock Exchange of India
    ".BO": "BSE",  # Bombay Stock Exchange
    ".L": "LSE",   # London Stock Exchange
    ".T": "TSE",   # Tokyo Stock Exchange
}


def normalize_ticker(ticker: str, use_openfigi: bool = False) -> Tuple[str, Optional[str]]:
    """
    Normalize a ticker symbol using internal mapping or OpenFIGI.
    
    Args:
        ticker: Raw ticker symbol from input
        use_openfigi: Whether to use OpenFIGI API (requires API key)
        
    Returns:
        Tuple of (normalized_ticker, error_message)
        - normalized_ticker: The normalized ticker symbol
        - error_message: None if successful, error message if unresolved
    """
    if not ticker or not ticker.strip():
        return "", "Empty ticker"
    
    ticker_upper = ticker.strip().upper()
    
    # Check internal mapping first
    if ticker_upper in INTERNAL_TICKER_MAP:
        return INTERNAL_TICKER_MAP[ticker_upper], None
    
    # Try OpenFIGI if enabled (placeholder - would need API integration)
    if use_openfigi:
        normalized, error = _normalize_via_openfigi(ticker_upper)
        if not error:
            return normalized, None
    
    # Clean up common formatting issues
    ticker_clean = re.sub(r'[^\w\-.]+', '', ticker_upper)
    
    # If it looks valid (alphanumeric with possible dash/dot), accept it
    if re.match(r'^[\w\-\.]+$', ticker_clean) and len(ticker_clean) > 0:
        return ticker_clean, None
    
    # Return original if we can't normalize
    return ticker_upper, None  # Accept as-is for now, could return error if strict


def _normalize_via_openfigi(ticker: str) -> Tuple[str, Optional[str]]:
    """
    Normalize ticker via OpenFIGI API (placeholder implementation).
    
    In production, this would:
    1. Call OpenFIGI API with the ticker
    2. Match against FIGI results
    3. Return standardized ticker
    
    For now, returns the ticker as-is.
    """
    # TODO: Implement OpenFIGI API integration
    # Would require: requests library, API key management
    return ticker, None


def batch_normalize_tickers(tickers: list, use_openfigi: bool = False) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    Normalize a batch of tickers.
    
    Args:
        tickers: List of ticker symbols
        use_openfigi: Whether to use OpenFIGI API
        
    Returns:
        Dictionary mapping original_ticker -> (normalized_ticker, error_message)
    """
    results = {}
    for ticker in tickers:
        results[ticker] = normalize_ticker(ticker, use_openfigi)
    return results


# ingest/models.py
from typing import Optional, List
from pydantic import BaseModel, Field


class PortfolioRow(BaseModel):
    """
    A single row as validated/normalized from CSV/Excel input.
    Using pydantic v2 style (BaseModel.model_validate is used when parsing dicts).
    """
    ticker: str = Field(..., min_length=1)
    qty: Optional[float] = None        # number of shares (may be None)
    value: Optional[float] = None      # total holding value (may be None)
    sector: Optional[str] = None
    cost: Optional[float] = None       # cost per share (may be None)
    notes: Optional[str] = None


class ValidationReport(BaseModel):
    """
    Report of validation issues and unresolved items.
    """
    unresolved_tickers: List[str] = Field(default_factory=list)  # Tickers that couldn't be normalized
    missing_sectors: List[str] = Field(default_factory=list)     # Tickers without sector assignment
    parse_errors: List[str] = Field(default_factory=list)        # Rows that had parsing errors
    total_rows: int = 0
    valid_rows: int = 0


class PortfolioJSON(BaseModel):
    """
    Final portfolio output structure with holdings and validation report.
    """
    portfolio: List[dict]  # Canonical JSON holdings
    validation_report: ValidationReport
    user_id: Optional[str] = None  # None if retention_mode=ephemeral


# agents/ingest_agent.py
"""
Integrated Ingest Agent with portfolio processing and UI.
Combines IngestAgent, parser, models, and file handling.
Supports both file upload and manual data entry.
"""

import json
from typing import Union, Optional, Literal, List, Dict, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.parser import parse_file, to_canonical_json, row_to_portfolio_row_from_mapped
from ingest.models import PortfolioJSON, ValidationReport, PortfolioRow
from ingest.symbol_mapper import normalize_ticker
from ingest.sector_mapper import assign_sector


def open_file_browser() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("⚠ GUI file browser not available (tkinter not installed)")
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select Portfolio File (CSV/Excel)",
            filetypes=[('All Supported', '*.csv *.xlsx *.xls'), ('CSV files','*.csv'), ('Excel files','*.xlsx *.xls'), ('All files','*.*')]
        )
        root.destroy()
        if file_path:
            return Path(file_path)
        return None
    except Exception as e:
        print(f"⚠ Error opening file browser: {e}")
        return None


def manual_data_entry(agent: 'IngestAgent') -> PortfolioJSON:
    print("\n" + "=" * 60)
    print("Manual Data Entry")
    print("=" * 60)
    print("Enter portfolio holdings one by one.")
    print("Press Enter on an empty ticker to finish.\n")

    portfolio_rows: List[PortfolioRow] = []
    entry_num = 1
    while True:
        print(f"\n--- Holding #{entry_num} ---")
        ticker = input("Ticker/Symbol (required): ").strip()
        if not ticker:
            print("Finishing data entry...")
            break
        qty_str = input("Quantity/Shares (optional, press Enter to skip): ").strip()
        cost_str = input("Cost per share/Price (optional, press Enter to skip): ").strip()
        value_str = input("Total holding value (optional, press Enter to skip): ").strip()
        sector = input("Sector/Industry (optional, press Enter to skip): ").strip()
        notes = input("Notes (optional, press Enter to skip): ").strip()

        def parse_num(s):
            if not s:
                return None
            try:
                s_clean = s.replace(',', '').replace('$','').replace('₹','').strip()
                return float(s_clean) if s_clean else None
            except Exception:
                return None

        qty = parse_num(qty_str)
        cost = parse_num(cost_str)
        value = parse_num(value_str)

        if agent.use_symbol_mapping:
            normalized_ticker, _ = normalize_ticker(ticker)
            ticker = normalized_ticker if normalized_ticker else ticker.upper()
        else:
            ticker = ticker.upper()

        if agent.use_sector_mapping:
            final_sector = assign_sector(ticker, sector if sector else None)
        else:
            final_sector = sector if sector else None

        try:
            row = PortfolioRow.model_validate({
                'ticker': ticker,
                'qty': qty,
                'value': value,
                'sector': final_sector,
                'cost': cost,
                'notes': notes if notes else None
            })
            portfolio_rows.append(row)
            print(f"✓ Added {ticker}")
            entry_num += 1
        except Exception as e:
            print(f"✗ Error adding holding: {e}")
            continue

    portfolio_json = to_canonical_json(portfolio_rows)
    validation_report = ValidationReport(
        unresolved_tickers=[],
        missing_sectors=[r.ticker for r in portfolio_rows if not r.sector or r.sector=='unknown'],
        parse_errors=[],
        total_rows=len(portfolio_rows),
        valid_rows=len(portfolio_rows)
    )
    return PortfolioJSON(portfolio=portfolio_json, validation_report=validation_report, user_id=agent.user_id)


def process_file_upload(agent: 'IngestAgent', file_path: Path) -> PortfolioJSON:
    print(f"Loading file: {file_path}")
    print(f"Retention mode: {agent.retention_mode}")
    print(f"User ID: {agent.user_id}\n")
    try:
        portfolio_json = agent.process_portfolio(file_path)
        print(f"✓ Processed portfolio successfully\n")
        return portfolio_json
    except FileNotFoundError:
        print(f"✗ Error: File not found: {file_path}")
        raise
    except Exception as e:
        print(f"✗ Error processing file: {e}")
        raise


def display_results(portfolio_json: PortfolioJSON, agent: 'IngestAgent'):
    print("=" * 60)
    print("Portfolio JSON Output")
    print("=" * 60)
    print(json.dumps(portfolio_json.portfolio, indent=2, default=str))
    print("\n" + "=" * 60)
    print("Validation Report")
    print("=" * 60)
    report = portfolio_json.validation_report
    print(f"Total rows: {report.total_rows}")
    print(f"Valid rows: {report.valid_rows}")
    if report.unresolved_tickers:
        print(f"Unresolved tickers ({len(report.unresolved_tickers)}): {report.unresolved_tickers}")
    if report.missing_sectors:
        print(f"Missing sectors ({len(report.missing_sectors)}): {report.missing_sectors[:10]}...")
    if report.parse_errors:
        print(f"Parse errors ({len(report.parse_errors)}):")
        for error in report.parse_errors[:5]:
            print(f"  - {error}")
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total holdings: {len(portfolio_json.portfolio)}")
    total_value = sum(item.get('holding_value',0) or 0 for item in portfolio_json.portfolio if item.get('holding_value'))
    print(f"Total portfolio value: ${total_value:,.2f}")
    computed_count = sum(1 for item in portfolio_json.portfolio if item.get('holding_value') is not None)
    print(f"Holdings with computed values: {computed_count}/{len(portfolio_json.portfolio)}")
    file_to_delete = agent.get_file_deletion_instruction()
    if file_to_delete:
        print(f"\n⚠ Ephemeral mode: File marked for deletion: {file_to_delete}")
        print("   (In production, Orchestrator would delete this file after processing)")


class IngestAgent:
    def __init__(self, retention_mode: Literal['persistent','ephemeral']='persistent', user_id: Optional[str]=None, use_symbol_mapping: bool=True, use_sector_mapping: bool=True):
        self.retention_mode = retention_mode
        self.user_id = None if retention_mode == 'ephemeral' else user_id
        self.use_symbol_mapping = use_symbol_mapping
        self.use_sector_mapping = use_sector_mapping
        self.file_to_delete = None

    def load_file(self, file_path: Union[str,Path]):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        import pandas as pd
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx','.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError('Unsupported file format. Use CSV or Excel.')
        return df.to_dict(orient='records')

    def process_portfolio(self, file_path: Union[str,Path]) -> PortfolioJSON:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if self.retention_mode == 'ephemeral':
            self.file_to_delete = file_path
        try:
            portfolio_rows, column_mapping, parse_errors_from_file = parse_file(str(file_path), use_symbol_mapping=self.use_symbol_mapping, use_sector_mapping=self.use_sector_mapping)
        except Exception as e:
            return PortfolioJSON(portfolio=[], validation_report=ValidationReport(parse_errors=[f"File parsing failed: {str(e)}"], total_rows=0, valid_rows=0), user_id=self.user_id)
        unresolved_tickers = []
        missing_sectors = []
        parse_errors = parse_errors_from_file.copy()
        valid_rows = 0
        tickers_seen = set()
        for row in portfolio_rows:
            if self.use_symbol_mapping:
                normalized, error = normalize_ticker(row.ticker)
                if error:
                    if row.ticker not in tickers_seen:
                        unresolved_tickers.append(row.ticker)
                        tickers_seen.add(row.ticker)
                elif not normalized or len(row.ticker) < 1:
                    if row.ticker not in tickers_seen:
                        unresolved_tickers.append(row.ticker)
                        tickers_seen.add(row.ticker)
            if not row.sector or row.sector == 'unknown':
                missing_sectors.append(row.ticker)
            if 'PARSE_ERROR' in (row.notes or ''):
                parse_errors.append(f"Ticker {row.ticker}: {row.notes}")
            if row.ticker and len(row.ticker) > 0:
                valid_rows += 1
        portfolio_json = to_canonical_json(portfolio_rows)
        validation_report = ValidationReport(unresolved_tickers=unresolved_tickers, missing_sectors=missing_sectors, parse_errors=parse_errors, total_rows=len(portfolio_rows), valid_rows=valid_rows)
        return PortfolioJSON(portfolio=portfolio_json, validation_report=validation_report, user_id=self.user_id)

    def get_file_deletion_instruction(self) -> Optional[Path]:
        return self.file_to_delete


def extract_tickers_from_portfolio(file_path):
    from pathlib import Path
    file_path = Path(file_path)
    import pandas as pd
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return []
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx','.xls']:
        df = pd.read_excel(file_path)
    else:
        print('Unsupported file format. Use CSV or Excel.')
        return []
    ticker_col = None
    for col in df.columns:
        if col.lower() in ['ticker','symbol']:
            ticker_col = col
            break
    if ticker_col:
        tickers = df[ticker_col].dropna().unique().tolist()
        print('Tickers found in portfolio file:')
        for t in tickers:
            print(t)
        return tickers
    else:
        print('No ticker column found in the file. Columns available:', df.columns.tolist())
        return []


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        extract_tickers_from_portfolio(sys.argv[1])
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        agent = IngestAgent(retention_mode='persistent', user_id='user123', use_symbol_mapping=True, use_sector_mapping=True)
        print('='*60)
        print('Portfolio Ingest Agent')
        print('='*60)
        print('\nChoose an option:')
        print('1. Upload file (CSV/Excel)')
        print('2. Manual data entry')
        print()
        choice = input('Enter choice (1 or 2): ').strip()
        try:
            if choice == '1':
                file_path_input = input("\nEnter file path (or 'browser' for GUI): ").strip()
                if file_path_input.lower() in ('browser','b'):
                    file_path = open_file_browser() or (project_root / 'data' / 'samples' / 'complex_portfolio.xlsx')
                elif file_path_input:
                    candidate = Path(file_path_input.strip('\'\"')).expanduser()
                    if not candidate.is_absolute():
                        candidate = candidate.resolve()
                    if candidate.is_dir():
                        csv_files = list(candidate.glob('*.csv'))
                        xls_files = list(candidate.glob('*.xlsx')) + list(candidate.glob('*.xls'))
                        all_files = sorted(csv_files + xls_files)
                        if all_files:
                            print('Found the following portfolio files:')
                            for idx,f in enumerate(all_files,1):
                                print(f'  {idx}. {f.name}')
                            sel = input('Enter file number (or press Enter for default): ').strip()
                            try:
                                sel_idx = int(sel)-1
                                if 0 <= sel_idx < len(all_files):
                                    file_path = all_files[sel_idx]
                                else:
                                    file_path = project_root / 'data' / 'samples' / 'complex_portfolio.xlsx'
                            except Exception:
                                file_path = project_root / 'data' / 'samples' / 'complex_portfolio.xlsx'
                        else:
                            print('No CSV/Excel files found in directory; using default sample file.')
                            file_path = project_root / 'data' / 'samples' / 'complex_portfolio.xlsx'
                    else:
                        file_path = candidate
                else:
                    file_path = project_root / 'data' / 'samples' / 'complex_portfolio.xlsx'
                if not file_path.exists() or file_path.is_dir():
                    print(f"\n✗ Error: Path invalid: {file_path}")
                    exit(1)
                if file_path.suffix.lower() not in ['.csv','.xlsx','.xls']:
                    confirm = input(f"Warning: extension {file_path.suffix} may be unsupported. Continue? (y/n): ").strip().lower()
                    if confirm != 'y':
                        print('Cancelled.')
                        exit(0)
                portfolio_json = process_file_upload(agent, file_path)
            elif choice == '2':
                portfolio_json = manual_data_entry(agent)
            else:
                print('Invalid choice. Please run again and select 1 or 2.')
                exit(1)
            display_results(portfolio_json, agent)
        except KeyboardInterrupt:
            print('\n\nOperation cancelled by user.')
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

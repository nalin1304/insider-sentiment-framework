"""
Corporate actions data collector for the insider-sentiment-framework.

Fetches historical earnings dates, dividends, and stock splits for each
configured stock from Yahoo Finance.
Saves data to data/raw/corporate_actions/.

Usage:
    python -m data_collection.corporate_actions
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from config.settings import (
        LOG_FILE,
        RAW_CORP_ACTIONS_DIR,
        REQUEST_DELAY_SECONDS,
        STOCKS,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config.settings import (
        LOG_FILE,
        RAW_CORP_ACTIONS_DIR,
        REQUEST_DELAY_SECONDS,
        STOCKS,
    )

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("corporate_actions")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger

log = _setup_logger()

def collect_corporate_actions_for_stock(stock_key: str, stock_info: dict, save_dir: Path) -> dict:
    """Collect actions for a single stock."""
    yf_ticker = stock_info["yfinance_ticker"]
    log.info("Collecting corporate actions for %s (%s)", stock_key, yf_ticker)
    
    ticker_obj = yf.Ticker(yf_ticker)
    results = {}
    
    # 1. Dividends and Splits
    try:
        actions_df = ticker_obj.actions
        if actions_df is not None and not actions_df.empty:
            actions_csv = save_dir / f"{stock_key}_actions.csv"
            actions_df.to_csv(actions_csv)
            results["actions_rows"] = len(actions_df)
        else:
            results["actions_rows"] = 0
    except Exception as e:
        log.warning("Failed to fetch actions for %s: %s", stock_key, e)
        results["actions_rows"] = 0

    # 2. Earnings Dates
    try:
        earnings_df = ticker_obj.earnings_dates
        if earnings_df is not None and not earnings_df.empty:
            earnings_csv = save_dir / f"{stock_key}_earnings.csv"
            earnings_df.to_csv(earnings_csv)
            results["earnings_rows"] = len(earnings_df)
        else:
            results["earnings_rows"] = 0
    except Exception as e:
        log.warning("Failed to fetch earnings for %s: %s", stock_key, e)
        # BSE/NSE Fallback logic could go here
        log.debug("Fallback to BSE/NSE for earnings not implemented.")
        results["earnings_rows"] = 0

    return results

def collect_all_corporate_actions(stocks_config: dict = None, save_dir: Path = None) -> dict:
    if stocks_config is None:
        stocks_config = STOCKS
    if save_dir is None:
        save_dir = RAW_CORP_ACTIONS_DIR

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(stocks_config)
    log.info("Starting corporate actions collection for %d stocks.", total)
    
    summary = {}
    for idx, (stock_key, stock_info) in enumerate(stocks_config.items(), 1):
        log.info("--- [%d/%d] %s ---", idx, total, stock_key)
        
        # Checkpointing
        actions_csv = save_dir / f"{stock_key}_actions.csv"
        earnings_csv = save_dir / f"{stock_key}_earnings.csv"
        if actions_csv.exists() and earnings_csv.exists() and actions_csv.stat().st_size > 10:
            log.info("Skipping %s, already exists.", stock_key)
            continue
            
        summary[stock_key] = collect_corporate_actions_for_stock(stock_key, stock_info, save_dir)
        time.sleep(REQUEST_DELAY_SECONDS)
        
    return summary

def main() -> None:
    log.info("=" * 60)
    log.info("Corporate Actions Collector - standalone run")
    log.info("=" * 60)
    collect_all_corporate_actions()
    log.info("Corporate Actions Collection Complete")

if __name__ == "__main__":
    main()

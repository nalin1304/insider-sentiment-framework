import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from config.settings import LOG_FILE, MAX_RETRIES, PRICE_LOOKBACK_YEARS, RAW_PRICES_DIR, RAW_INDICES_DIR, REQUEST_DELAY_SECONDS, RETRY_BACKOFF_FACTOR, INDICES
    from config.universe import STOCKS
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config.settings import LOG_FILE, MAX_RETRIES, PRICE_LOOKBACK_YEARS, RAW_PRICES_DIR, RAW_INDICES_DIR, REQUEST_DELAY_SECONDS, RETRY_BACKOFF_FACTOR, INDICES
    from config.universe import STOCKS

logger = logging.getLogger("prices_collector")

def download_stock_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    save_dir: Path = None,
    max_retries: int = MAX_RETRIES,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    delay: float = REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    logger.info("Downloading %s [%s to %s]", ticker, start_date, end_date)
    
    # Try .NS first, then .BO
    suffixes = [".NS", ".BO"] if not ticker.startswith("^") and "." not in ticker else [""]
    
    for suffix in suffixes:
        current_ticker = f"{ticker}{suffix}"
        df = pd.DataFrame()
        
        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(
                    tickers=current_ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                
                if df is None or df.empty:
                    raise ValueError(f"Empty result for {current_ticker}")
                    
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                expected_cols = ["Open", "High", "Low", "Close", "Volume"]
                df = df[[c for c in expected_cols if c in df.columns]]
                
                if len(df) > 50: # Arbitrary threshold for valid data
                    df.index.name = "Date"
                    logger.info("Successfully downloaded %d rows for %s", len(df), current_ticker)
                    
                    if save_dir is not None:
                        save_dir = Path(save_dir)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        filepath = save_dir / f"{ticker}.csv"
                        df.to_csv(filepath)
                    
                    time.sleep(delay)
                    return df
            except Exception as exc:
                wait_time = backoff_factor ** attempt
                if attempt < max_retries:
                    time.sleep(wait_time)
                    
        logger.warning(f"Failed to get data for {current_ticker}.")
        
    return pd.DataFrame()

def download_all_prices(
    symbols: list[str],
    indices_config: dict[str, str],
    lookback_years: int = PRICE_LOOKBACK_YEARS,
    save_dir_prices: Path = None,
    save_dir_indices: Path = None,
) -> dict[str, pd.DataFrame]:
    
    if save_dir_prices is None:
        save_dir_prices = RAW_PRICES_DIR
    if save_dir_indices is None:
        save_dir_indices = RAW_INDICES_DIR
        
    from dateutil.relativedelta import relativedelta
    end_dt = datetime.utcnow()
    start_dt = end_dt - relativedelta(years=lookback_years)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = {}
    
    logger.info("--- Downloading Indices ---")
    for idx_ticker, idx_name in indices_config.items():
        df = download_stock_prices(idx_ticker, start_date, end_date, save_dir_indices)
        results[idx_ticker] = df
        
    logger.info(f"--- Downloading {len(symbols)} Stocks ---")
    
    def _download_sym(sym):
        out_path = Path(save_dir_prices) / f"{sym}.csv"
        if out_path.exists():
            try:
                df_existing = pd.read_csv(out_path)
                if not df_existing.empty:
                    return sym, df_existing
            except Exception:
                pass
        return sym, download_stock_prices(sym, start_date, end_date, save_dir_prices)
        
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_download_sym, sym): sym for sym in symbols}
        completed = 0
        for future in as_completed(futures):
            sym, df = future.result()
            results[sym] = df
            completed += 1
            if completed % 100 == 0:
                logger.info(f"Progress: {completed}/{len(symbols)}")
                
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all_prices(list(STOCKS.keys()), INDICES)

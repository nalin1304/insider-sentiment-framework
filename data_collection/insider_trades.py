import requests
import pandas as pd
from pathlib import Path
import time
import logging
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DIR = Path('data/raw/insider_trades')
PROCESSED_DIR = Path('data/processed')

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def fetch_nse_insider_data():
    """Fetch insider trading data from NSE API across multiple years."""
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nseindia.com/companies-listing/corporate-filings-insider-trading',
    }
    
    # 1. Hit homepage to get cookies
    logger.info("Initializing session cookies from NSE homepage...")
    try:
        session.get('https://www.nseindia.com', headers=headers, timeout=15)
    except Exception as e:
        logger.error(f"Failed to get NSE homepage: {e}")
        return False
        
    time.sleep(2)
    
    # Dynamic multi-year windows
    from datetime import datetime, timedelta
    from config.settings import INSIDER_LOOKBACK_YEARS
    
    windows = []
    end_date = datetime.now()
    for _ in range(INSIDER_LOOKBACK_YEARS):
        start_date = end_date - timedelta(days=365)
        windows.append((start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y")))
        end_date = start_date - timedelta(days=1)
    
    all_dfs = []
    
    for start_dt, end_dt in windows:
        logger.info(f"Fetching data for window: {start_dt} to {end_dt}")
        url = f"https://www.nseindia.com/api/corporates-pit?index=equities&from_date={start_dt}&to_date={end_dt}&csv=true"
        
        try:
            r = session.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                # Read CSV
                # NSE CSVs have broken newlines inside headers like "SYMBOL \n"
                clean_text = r.text.replace(' \n\",', '\",').replace(' \n\"', '\"')
                df = pd.read_csv(io.StringIO(clean_text), sep=',', skipinitialspace=True, encoding='utf-8-sig')
                
                # Clean column names
                df.columns = [c.strip().replace('\n', '').replace('ï»¿', '').replace('"', '') for c in df.columns]
                
                # Save raw chunk
                raw_path = RAW_DIR / f"insider_{start_dt[-4:]}_{end_dt[-4:]}.csv"
                df.to_csv(raw_path, index=False)
                logger.info(f"  -> Downloaded {len(df)} rows. Saved to {raw_path}")
                
                all_dfs.append(df)
            else:
                logger.warning(f"  -> Failed. Status Code: {r.status_code}")
                # Wait longer and retry once?
                time.sleep(5)
                r2 = session.get(url, headers=headers, timeout=30)
                if r2.status_code == 200:
                    clean_text = r2.text.replace(' \n\",', '\",').replace(' \n\"', '\"')
                    df = pd.read_csv(io.StringIO(clean_text), sep=',', skipinitialspace=True, encoding='utf-8-sig')
                    df.columns = [c.strip().replace('\n', '').replace('ï»¿', '').replace('"', '') for c in df.columns]
                    all_dfs.append(df)
                    logger.info(f"  -> Downloaded {len(df)} rows on retry.")
                else:
                    logger.error(f"  -> Retry failed. Missing {start_dt} to {end_dt}")
                    
        except Exception as e:
            logger.error(f"  -> Exception: {e}")
            
        # Polite delay
        time.sleep(3)
        
    if not all_dfs:
        logger.error("No data fetched at all. NSE may be blocking the request.")
        return False
        
    # Combine and save to processed
    logger.info("Combining all chunks...")
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Basic cleaning before saving to processed/insider_trades_clean.csv
    # The original logic expected certain column names. 
    # NSE CSVs have columns like "SYMBOL", "DATE OF ALLOTMENT/ACQUISITION TO", etc.
    # We will strip them just in case.
    combined.columns = [c.strip() for c in combined.columns]
    
    # Remove duplicates
    combined = combined.drop_duplicates()
    
    out_path = PROCESSED_DIR / "insider_trades_clean.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Saved merged dataset: {len(combined)} rows to {out_path}")
    
    # Print date range covered
    if 'DATE OF INITMATION TO COMPANY' in combined.columns:
        dates = pd.to_datetime(combined['DATE OF INITMATION TO COMPANY'], dayfirst=True, errors='coerce').dropna()
        if not dates.empty:
            logger.info(f"Date range covered: {dates.min().date()} to {dates.max().date()}")
            
    return True

if __name__ == "__main__":
    fetch_nse_insider_data()

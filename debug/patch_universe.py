import re

file_path = "config/universe.py"
with open(file_path, "r") as f:
    content = f.read()

new_imports = """import csv
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, Any

from config.settings import SECTOR_MAPPING, RAW_PRICES_DIR, PROCESSED_DIR

logger = logging.getLogger("universe")
"""

content = re.sub(r"import csv\nfrom pathlib import Path\nfrom typing import Dict, Any", new_imports, content)

new_func = """
def build_universe(min_filings: int = 3) -> Dict[str, Dict[str, Any]]:
    '''
    Dynamically build the universe from the cleaned insider trades data
    and check for valid price history on yfinance.
    Saves the result to config/universe.csv.
    '''
    logger.info("Building universe from insider trades...")
    clean_trades_file = PROCESSED_DIR / "insider_trades_clean.csv"
    
    if not clean_trades_file.exists():
        logger.error("Cleaned insider trades file not found. Cannot build universe.")
        return {}
        
    df = pd.read_csv(clean_trades_file)
    
    # 1. Filter symbols with at least `min_filings`
    counts = df['SYMBOL'].value_counts()
    valid_symbols = counts[counts >= min_filings].index.tolist()
    logger.info(f"Found {len(valid_symbols)} symbols with >= {min_filings} filings.")
    
    universe = {}
    
    # 2. Check price history from already downloaded data in data/raw/prices
    # or query yfinance if not downloaded (we expect prices to be downloaded already)
    for sym in valid_symbols:
        price_file = RAW_PRICES_DIR / f"{sym}.csv"
        # If the price file exists and has data, we consider it valid.
        if price_file.exists() and price_file.stat().st_size > 50:
            
            # Map sector, default to 'Unknown'
            sector = SECTOR_MAPPING.get(sym, "Unknown")
            
            # For company name and aliases, try to extract from the df
            company_names = df[df['SYMBOL'] == sym]['COMPANY'].unique()
            company_name = company_names[0] if len(company_names) > 0 else sym
            
            # We can use the company name as an alias
            aliases = [company_name, sym]
            # Remove Ltd/Limited to create another alias
            short_name = company_name.replace(" Limited", "").replace(" Ltd.", "").replace(" Ltd", "").strip()
            if short_name not in aliases:
                aliases.append(short_name)
                
            universe[sym] = {
                "ticker": f"{sym}.NS",
                "nse_symbol": sym,
                "company_name": company_name,
                "name_aliases": "|".join(aliases),
                "sector": sector,
                "n_insider_filings": int(counts[sym])
            }
            
    logger.info(f"Universe built with {len(universe)} symbols having valid price data.")
    
    # 3. Save to universe.csv
    with open(UNIVERSE_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "nse_symbol", "company_name", "name_aliases", "sector", "n_insider_filings"])
        for sym, data in universe.items():
            writer.writerow([
                data["ticker"],
                data["nse_symbol"],
                data["company_name"],
                data["name_aliases"],
                data["sector"],
                data["n_insider_filings"]
            ])
            
    # Also update the in-memory STOCKS dictionary
    global STOCKS
    STOCKS.clear()
    for sym, data in universe.items():
        STOCKS[sym] = {
            "nse_ticker": data["nse_symbol"],
            "yfinance_ticker": data["ticker"],
            "bse_scrip": "", # No reliable way to map this from NSE data alone without a master list, leave empty
            "company_name": data["company_name"],
            "name_aliases": data["name_aliases"].split("|"),
            "sector": data["sector"]
        }
    
    return STOCKS

# Pre-load the universe on import for easy access, similar to the old config
STOCKS = {}
try:
    STOCKS = load_universe()
except FileNotFoundError:
    pass
"""

# Replace the old STOCKS loading at the bottom
content = re.sub(r"# Pre-load the universe on import for easy access, similar to the old config\nSTOCKS = load_universe\(\)", new_func, content)

# Modify load_universe to handle missing keys gracefully
old_load = """            stocks[base_ticker] = {
                "nse_ticker": base_ticker,
                "yfinance_ticker": yfinance_ticker,
                "bse_scrip": row["bse_scrip_code"].strip(),
                "company_name": row["company_name"].strip(),
                "name_aliases": aliases,
                "sector": row["sector"].strip()
            }"""
new_load = """            stocks[base_ticker] = {
                "nse_ticker": row.get("nse_symbol", base_ticker).strip(),
                "yfinance_ticker": yfinance_ticker,
                "bse_scrip": row.get("bse_scrip_code", "").strip(),
                "company_name": row["company_name"].strip(),
                "name_aliases": aliases,
                "sector": row["sector"].strip()
            }"""
content = content.replace(old_load, new_load)

with open(file_path, "w") as f:
    f.write(content)

print("config/universe.py updated.")

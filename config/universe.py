"""
Stock universe loader for insider-sentiment-framework.
Parses the config/universe.csv file to provide a programmatic list
of stocks to the data collection layer.
"""

import csv
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from config.settings import SECTOR_MAPPING, RAW_PRICES_DIR, PROCESSED_DIR

logger = logging.getLogger("universe")

UNIVERSE_CSV = Path(__file__).resolve().parent / "universe.csv"

def load_universe() -> Dict[str, Dict[str, Any]]:
    stocks = {}
    if not UNIVERSE_CSV.exists():
        return stocks
        
    with open(UNIVERSE_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yfinance_ticker = row["ticker"].strip()
            base_ticker = yfinance_ticker.replace(".NS", "")
            aliases = [a.strip() for a in row.get("name_aliases", "").split("|") if a.strip()]
            
            stocks[base_ticker] = {
                "nse_ticker": row.get("nse_symbol", base_ticker).strip(),
                "yfinance_ticker": yfinance_ticker,
                "bse_scrip": row.get("bse_scrip_code", "").strip(),
                "company_name": row.get("company_name", "").strip(),
                "name_aliases": aliases,
                "sector": row.get("sector", "").strip(),
                "n_insider_filings": int(row.get("n_insider_filings", 0)) if row.get("n_insider_filings") else 0
            }
            
    return stocks

def build_universe(min_filings: int = 75) -> Dict[str, Dict[str, Any]]:
    logger.info("Building universe from insider trades...")
    clean_trades_file = PROCESSED_DIR / "insider_trades_clean.csv"
    
    if not clean_trades_file.exists():
        logger.error("Cleaned insider trades file not found.")
        return {}
        
    df = pd.read_csv(clean_trades_file)
    
    counts = df['SYMBOL'].value_counts()
    valid_symbols = counts[counts >= min_filings].index.tolist()
    
    # Intersect with downloaded price CSVs
    price_files = list(RAW_PRICES_DIR.glob("*.csv"))
    symbols_with_prices = {f.stem.replace('.NS', '').replace('.BO', '') for f in price_files}
    
    valid_symbols = [s for s in valid_symbols if s in symbols_with_prices]
    
    logger.info(f"Found {len(valid_symbols)} symbols with >= {min_filings} filings AND existing price data.")
    
    universe = {}
    
    for sym in valid_symbols:
        sector = SECTOR_MAPPING.get(sym, "Unknown")
        company_names = df[df['SYMBOL'] == sym]['COMPANY'].unique()
        company_name = str(company_names[0]) if len(company_names) > 0 and pd.notna(company_names[0]) else sym
        
        aliases = [company_name, sym]
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
            
    logger.info(f"Universe built with {len(universe)} symbols.")
    
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
            
    global STOCKS
    STOCKS.clear()
    for sym, data in universe.items():
        STOCKS[sym] = {
            "nse_ticker": data["nse_symbol"],
            "yfinance_ticker": data["ticker"],
            "bse_scrip": "",
            "company_name": data["company_name"],
            "name_aliases": data["name_aliases"].split("|"),
            "sector": data["sector"]
        }
    
    return STOCKS

STOCKS = load_universe()

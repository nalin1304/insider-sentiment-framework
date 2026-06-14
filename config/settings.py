"""
Configuration settings for the insider-sentiment-framework data collection layer.

Contains paths, API parameters, logging setup, and orchestrates the universe.
"""

import os
from pathlib import Path

# We will import STOCKS at the bottom to avoid circular imports

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_PRICES_DIR = RAW_DIR / "prices"
RAW_INDICES_DIR = RAW_DIR / "indices"
RAW_INSIDER_DIR = RAW_DIR / "insider_trades"
RAW_NEWS_DIR = RAW_DIR / "news"
RAW_CORP_ACTIONS_DIR = RAW_DIR / "corporate_actions"
RAW_MACRO_DIR = RAW_DIR / "macro"

# Ensure directories exist on import
for _dir in [RAW_PRICES_DIR, RAW_INDICES_DIR, RAW_INSIDER_DIR, RAW_NEWS_DIR, RAW_CORP_ACTIONS_DIR, RAW_MACRO_DIR, PROCESSED_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Indices & VIX
# ---------------------------------------------------------------------------
# Added relevant indices and VIX as requested for macro/context
INDICES = {
    "^BSESN": "BSE SENSEX",
    "^NSEI": "NIFTY 50",
    "^INDIAVIX": "India VIX",
    "BSE-IT.BO": "BSE IT Sector",
    "BSE-BANK.BO": "BSE Bankex",
    "BSE-AUTO.BO": "BSE Auto",
    "BSE-FMCG.BO": "BSE FMCG",
    "BSE-HC.BO": "BSE Healthcare"
}

# ---------------------------------------------------------------------------
# Insider Categorization & Sector Mapping
# ---------------------------------------------------------------------------

INSIDER_ROLE_WEIGHTS = {
    "Promoters": 1.0,
    "Promoter Group": 0.9,
    "Director": 0.8,
    "Key Managerial Personnel": 0.8,
    "Employees/Designated Employees": 0.4,
    "Immediate relative": 0.6,
    "Other": 0.2,
    "-": 0.1
}

# Static mapping for demonstration (this can be expanded or scraped later)
SECTOR_MAPPING = {
    # IT
    "TCS": "Information Technology",
    "INFY": "Information Technology",
    "HCLTECH": "Information Technology",
    "WIPRO": "Information Technology",
    "TECHM": "Information Technology",
    
    # Financials
    "HDFCBANK": "Financial Services",
    "ICICIBANK": "Financial Services",
    "SBIN": "Financial Services",
    "KOTAKBANK": "Financial Services",
    "AXISBANK": "Financial Services",
    "BAJFINANCE": "Financial Services",
    
    # Energy / Oil & Gas
    "RELIANCE": "Oil, Gas & Consumable Fuels",
    "ONGC": "Oil, Gas & Consumable Fuels",
    "NTPC": "Power",
    "POWERGRID": "Power",
    
    # FMCG
    "ITC": "Fast Moving Consumer Goods",
    "HINDUNILVR": "Fast Moving Consumer Goods",
    "NESTLEIND": "Fast Moving Consumer Goods",
    
    # Auto
    "MARUTI": "Automobile and Auto Components",
    "M&M": "Automobile and Auto Components",
    "TATAMOTORS": "Automobile and Auto Components",
    "BAJAJ-AUTO": "Automobile and Auto Components",
    
    # Telecom
    "BHARTIARTL": "Telecommunication",
    
    # Healthcare
    "SUNPHARMA": "Healthcare",
    "DRREDDY": "Healthcare",
    "CIPLA": "Healthcare",
    "DIVISLAB": "Healthcare",
    
    # Construction / Infrastructure
    "LT": "Construction",
    
    # Metals & Mining
    "TATASTEEL": "Metals & Mining",
    "HINDALCO": "Metals & Mining",
    "JSWSTEEL": "Metals & Mining",
}

# ---------------------------------------------------------------------------
# Data collection parameters
# ---------------------------------------------------------------------------

# Price data: lookback period in years
PRICE_LOOKBACK_YEARS = 7

# Insider trading: lookback period in years
INSIDER_LOOKBACK_YEARS = 7

# News data: lookback period in years
NEWS_LOOKBACK_YEARS = 7

# HTTP request settings
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 0.05          # polite delay between scraping requests
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0          # exponential backoff multiplier

# User-Agent for web scraping (realistic browser string)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# NewsAPI key: read from environment / .env file (optional)
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")

# GDELT API base URL
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "collection.log"

# Circular import removed.

import re

file_path = "README.md"
with open(file_path, "r") as f:
    content = f.read()

# Update Tracked Stocks
old_tracked = """## Tracked Stocks

The framework dynamically tracks Nifty/BSE 200 constituents configured via `config/universe.csv`. This CSV maintains the mapping of:
- `ticker`: Base NSE ticker symbol
- `bse_scrip_code`: 6-digit BSE numerical code
- `company_name`: Full corporate name
- `name_aliases`: Pipe-separated aliases for news/social scraping
- `sector`: Macro sector categorisation"""

new_tracked = """## Tracked Stocks

The framework builds a dynamic universe by intersecting historical insider trading disclosures from NSE with valid price histories from Yahoo Finance. This guarantees that every stock in the working universe has both features available.

To generate the universe:
```bash
python run_collection.py --insider
python run_collection.py --universe
```

This populates `config/universe.csv` with:
- `ticker`: Yahoo Finance ticker symbol (e.g., RELIANCE.NS)
- `nse_symbol`: Base NSE symbol
- `company_name`: Full corporate name
- `name_aliases`: Pipe-separated aliases for news/social scraping
- `sector`: Macro sector categorisation
- `n_insider_filings`: Number of valid insider trading filings"""

content = content.replace(old_tracked, new_tracked)

# Update Insider Trading Disclosures
old_insider = """### 2. Insider Trading Disclosures (BSE API)

- **Primary source**: BSE India API for insider trading filings under SEBI (Prohibition of Insider Trading) Regulations.
- **API endpoint**: `https://api.bseindia.com/BseIndiaAPI/api/InsiderTrading/w?scripcode={code}&fromdate={dd/mm/yyyy}&todate={dd/mm/yyyy}`
- **Fallback source**: NSE insider trading disclosures via `https://www.nseindia.com/api/corporates-pit` (requires session cookie management).
- **Coverage**: Last 3 years of filings.
- **Fields extracted**: Company name, scrip code, insider name, category (Promoter, KMP, Designated Person, Director), security type, transaction type (Buy, Sell, Pledge, Revoke), number of securities, transaction value, transaction date, disclosure date, exchange receipt date."""

new_insider = """### 2. Insider Trading Disclosures (NSE Archive)

- **Primary source**: NSE India Bulk Archive for insider trading filings under SEBI (Prohibition of Insider Trading) Regulations.
- **API endpoint**: `https://www.nseindia.com/api/corporates-pit?index=equities&from_date={dd-mm-yyyy}&to_date={dd-mm-yyyy}&csv=true`
- **Coverage**: 10+ years (2015-2025) bulk downloaded in yearly chunks.
- **Fields extracted**: Company name, symbol, acquirer/disposer name, category, transaction type (Buy, Sell, Pledge, Revoke), number of securities, value, transaction date, intimation date."""

content = content.replace(old_insider, new_insider)

# Update Running Data Collection
old_run = """# Combine specific collectors
python run_collection.py --prices --insider"""

new_run = """# Generate dynamic universe
python run_collection.py --universe

# Run social media collection
python run_collection.py --social

# Combine specific collectors
python run_collection.py --prices --insider"""

content = content.replace(old_run, new_run)

with open(file_path, "w") as f:
    f.write(content)

print("README.md updated.")

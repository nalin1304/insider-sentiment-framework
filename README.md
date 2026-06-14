# Insider Sentiment Framework

An AI-driven research framework for analyzing the effects of **insider trading disclosures** and **news sentiment** on BSE SENSEX constituent stocks. The framework combines regulatory filing data (SEBI SAST/PIT insider disclosures), financial news sentiment (via GDELT and NewsAPI), and historical price data to build predictive features. A multi-objective optimization layer (planned) will balance prediction accuracy, risk-adjusted returns, and model interpretability.


## Current Status

- **Universe**: 396 NSE-listed tickers selected by having >=75 insider filings over 7 years with available price history.
- **Date Range**: 2019-06-17 to 2026-06-12 (7 years of daily data).
- **Implemented**: Insider trading features (original, role-split, relative-size, clustering), macro features, LightGBM panel models, **PyTorch TabularMLP neural networks**, event study analysis, portfolio tilt backtest, and FinBERT continuous news sentiment pipeline (GPU/MPS accelerated).
- **Key Finding**: Core insider buy disclosures are followed by a statistically significant ~1% abnormal price drift over 20 trading days. However, this effect is economically marginal (0.08 SD) and does not survive transaction costs when deployed as a portfolio tilt strategy. An NSGA-II optimizer powered by Deep Learning (TabularMLP) successfully reduced portfolio volatility and cut max drawdown out-of-sample, but failed to generate excess positive Alpha over the baseline due to the turnover penalty. The Sentiment Analysis integration was disabled after failing data coverage validations (sentiment data only covered 9/396 tickers). See `FINAL_REPORT.md` for details.

---

## Repository Structure

```
insider-sentiment-framework/
|
|-- config/                     # Configuration and universe definitions
|   |-- universe.csv            # Constituents list
|   |-- universe.py             # Universe parser
|   |-- settings.py             # Global paths, parameters, index definitions
|
|-- data_collection/            # Data collection modules
|   |-- __init__.py
|   |-- prices.py               # yfinance-based OHLCV price collector
|   |-- insider_trades.py       # BSE/NSE insider trading disclosure scraper
|   |-- corporate_actions.py    # Dividends, earnings collector
|   |-- macro.py                # FII/DII flow and RBI Repo rate stubs
|   |-- news.py                 # GDELT + NewsAPI news article collector
|   |-- news_sources/           # Dedicated scrapers
|
|-- data/
|   |-- raw/                    # Raw collected data (not version-controlled)
|   |   |-- prices/             # Daily OHLCV CSVs per stock + SENSEX
|   |   |-- insider_trades/     # Insider trading disclosures per stock
|   |   |-- news/               # News article metadata per stock
|   |-- processed/              # Cleaned, merged, feature-ready datasets
|
|-- features/                   # Feature engineering (planned)
|-- models/                     # Sentiment and prediction models (planned)
|-- optimization/               # Multi-objective optimization (planned)
|-- notebooks/                  # Exploratory analysis Jupyter notebooks (planned)
|-- logs/                       # Collection and processing logs
|
|-- run_collection.py           # CLI entry point for data collection
|-- requirements.txt            # Python dependencies
|-- README.md                   # This file
```

---

## Tracked Stocks

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
- `n_insider_filings`: Number of valid insider trading filings

Plus multiple benchmark indices including **BSE SENSEX** (`^BSESN`), **India VIX** (`^INDIAVIX`), and various BSE sectoral indices.

---

## Data Sources

### 1. Price Data (yfinance)

- **Source**: Yahoo Finance via the `yfinance` Python library.
- **Ticker format**: NSE tickers with `.NS` suffix (e.g., `RELIANCE.NS`), `^BSESN` for SENSEX.
- **Coverage**: Daily OHLCV data for the last 5 years.
- **Limitations**:
  - Yahoo Finance data may have occasional gaps on non-trading days or during data feed interruptions.
  - Adjusted close prices account for splits and dividends but may differ slightly from BSE official data.
  - Rate limiting is possible with rapid sequential requests; the collector adds delays and retries.

### 2. Insider Trading Disclosures (NSE Archive)

- **Primary source**: NSE India Bulk Archive for insider trading filings under SEBI (Prohibition of Insider Trading) Regulations.
- **API endpoint**: `https://www.nseindia.com/api/corporates-pit?index=equities&from_date={dd-mm-yyyy}&to_date={dd-mm-yyyy}&csv=true`
- **Coverage**: 10+ years (2015-2025) bulk downloaded in yearly chunks.
- **Fields extracted**: Company name, symbol, acquirer/disposer name, category, transaction type (Buy, Sell, Pledge, Revoke), number of securities, value, transaction date, intimation date.
- **Limitations**:
  - **BSE API access**: The BSE API may block requests from non-browser clients or require specific headers/cookies. The scraper sets appropriate headers (Referer, User-Agent) but access may be restricted from certain networks or cloud environments.
  - **Data completeness**: Not all insider transactions may be captured. Some filings may have missing fields (especially transaction value). Voluntary disclosures and amendments may appear as separate entries.
  - **Date parsing**: Date formats from BSE can be inconsistent. The parser handles multiple formats but edge cases may exist.
  - **Rate limiting**: The scraper implements polite delays (1.5s between requests) and exponential backoff on failures.
  - **NSE fallback**: NSE's API requires first establishing a session by visiting their homepage (cookie-based auth). This works but can be fragile.

> **Note**: If running in a sandboxed or restricted network environment, the scraper will log connection errors gracefully. Synthetic sample files (prefixed `sample_`) are provided in `data/raw/insider_trades/` for downstream development.

### 3. News Articles (GDELT + NewsAPI)

#### GDELT 2.0 DOC API (Primary)

- **Source**: GDELT Project's Document API.
- **API endpoint**: `https://api.gdeltproject.org/api/v2/doc/doc`
- **Coverage**: Last 3 years, filtered for India-sourced financial/business news.
- **Fields**: Title, URL, publication date, source domain, average tone score, language, source country.
- **Limitations**:
  - GDELT returns a maximum of 250 articles per query. The collector chunks the time range into monthly windows to maximize coverage.
  - Tone scores are GDELT's automated sentiment measure (positive = positive tone, negative = negative tone). They are not financial-sentiment-specific.
  - Some articles may be duplicated across time windows. Deduplication is performed on URL.
  - GDELT may not cover all Indian financial news sources, especially smaller regional outlets.

#### NewsAPI (Secondary, Optional)

- **Source**: NewsAPI.org v2 `everything` endpoint.
- **API key**: Required. Set `NEWSAPI_KEY` in a `.env` file or as an environment variable.
- **Coverage**: Free tier limited to approximately 1 month of historical articles.
- **Limitations**:
  - Very limited history on the free tier, making it useful only as a supplement for recent news.
  - Rate limited to 100 requests/day on the free plan.
  - The collector gracefully skips if no API key is configured.

---

## Setup and Installation

### Prerequisites

- Python 3.9 or later
- pip (Python package manager)
- Network access to Yahoo Finance, BSE India, GDELT, and optionally NewsAPI

### Installation

```bash
# Clone or navigate to the project directory
cd insider-sentiment-framework

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables (Optional)

Create a `.env` file in the project root for NewsAPI:

```
NEWSAPI_KEY=your_newsapi_key_here
```

---

## Running Data Collection

The `run_collection.py` script is the single entry point for all data collection:

```bash
# Run all collectors (prices + insider trades + GDELT news)
python run_collection.py --all

# Run all collectors including NewsAPI
python run_collection.py --all --newsapi

# Run only price data collection
python run_collection.py --prices

# Run only insider trading collection
python run_collection.py --insider

# Run only news collection (GDELT only)
python run_collection.py --news

# Run news collection with NewsAPI supplement
python run_collection.py --news --newsapi

# Generate dynamic universe
python run_collection.py --universe

# Run social media collection
python run_collection.py --social

# Combine specific collectors
python run_collection.py --prices --insider
```

### Output

- Collected data is saved to `data/raw/` subdirectories as CSV files.
- Logs are written to `logs/collection.log` and also printed to the console.
- Progress, warnings, and errors are all logged with timestamps.

### Synthetic Sample Data

For development and testing without network access, synthetic sample files are provided in each `data/raw/` subdirectory. These files are prefixed with `sample_` and contain clearly-labeled synthetic data that matches the real data schema. They should **not** be used for actual analysis.

---

## Network Requirements

Some collectors require unrestricted network access:

| Collector        | Endpoint                                      | Notes                                        |
|-----------------|-----------------------------------------------|----------------------------------------------|
| Prices          | Yahoo Finance (via yfinance)                  | Usually accessible                           |
| Insider Trades  | BSE India API / NSE API                       | May be blocked in some environments          |
| News (GDELT)    | api.gdeltproject.org                          | Usually accessible                           |
| News (NewsAPI)  | newsapi.org                                   | Requires API key                             |

If running in a restricted environment (cloud sandbox, corporate firewall, etc.), the collectors will log errors and continue. Use the synthetic sample data for development in such cases.

---

## TODO: Next Stages

### Stage 2: Feature Engineering (`features/`)
- [x] Merge price data with insider trading events (event windows: T-5 to T+20 days around disclosure)
- [x] Calculate abnormal returns (CAR, BHAR) relative to SENSEX benchmark
- [x] Compute insider trading features: transaction size relative to market cap, insider category encoding, buy/sell ratio rolling windows
- [x] Extract news sentiment features: FinBERT sentiment probabilities, daily aggregates
- [x] Build cross-feature interactions (e.g., insider buy + positive news sentiment confluence)

### Stage 3: Sentiment Modeling (`models/`)
- [x] Score Indian financial news using transformer-based models (ProsusAI/FinBERT)
- [x] Evaluate pipeline on large batch of downloaded text (24,000+ headlines)
- [x] Build integration pipeline with main feature dataset

### Stage 4: Prediction Models (`models/`)
- [x] Train models to predict post-disclosure abnormal returns using insider + sentiment features
- [x] Evaluate LightGBM architectures
- [x] Implement robust, non-overlapping train/val/test splits to avoid leakage

### Stage 5: Multi-Objective Optimization (`optimization/`)
- [x] Define objectives: prediction accuracy (Sharpe ratio), risk control (max drawdown), model complexity/interpretability
- [x] Implement NSGA-II multi-objective optimization for Pareto-optimal model selection
- [x] Run walk-forward simulation incorporating daily vs monthly rebalancing logic
- [x] Apply realistic friction models (15bps transaction cost)

### Stage 6: Analysis and Visualization (`notebooks/`)
- [ ] Exploratory data analysis notebooks for each data source
- [ ] Interactive dashboards for Pareto front exploration

---

## License

This project is for academic and research purposes.

---

## Contributing

This is an active research project. See the TODO list above for planned extensions.

"""
News data collector for the insider-sentiment-framework.

Provides two collection strategies:
  1. GDELT 2.0 DOC API (primary): fetches article metadata over a configurable
     multi-year lookback window, chunked into monthly queries to stay within
     GDELT's per-request limits.
  2. NewsAPI v2 (secondary, optional): fetches recent articles (~1 month)
     when a valid NEWSAPI_KEY is available in the environment.

Both collectors save per-company CSV files into data/raw/news/.

Usage:
    # As a module
    from data_collection.news import collect_all_gdelt_news, collect_all_newsapi_news

    # As a standalone script
    python -m data_collection.news
"""

import csv
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project-level configuration imports
# ---------------------------------------------------------------------------
try:
    from config.settings import (
        GDELT_DOC_API_URL,
        LOG_FILE,
        MAX_RETRIES,
        NEWS_LOOKBACK_YEARS,
        NEWSAPI_KEY,
        RAW_NEWS_DIR,
        REQUEST_DELAY_SECONDS,
        REQUEST_TIMEOUT_SECONDS,
        RETRY_BACKOFF_FACTOR,
        USER_AGENT,
    )
    from config.universe import STOCKS
except ImportError:
    from config import (  # type: ignore[no-redef]
        GDELT_DOC_API_URL,
        LOG_FILE,
        MAX_RETRIES,
        NEWS_LOOKBACK_YEARS,
        NEWSAPI_KEY,
        RAW_NEWS_DIR,
        REQUEST_DELAY_SECONDS,
        REQUEST_TIMEOUT_SECONDS,
        RETRY_BACKOFF_FACTOR,
        STOCKS,
        USER_AGENT,
    )

# ---------------------------------------------------------------------------
# Logger setup: console + file
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers if module is re-imported
if not logger.handlers:
    _fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(_fmt)
    logger.addHandler(_console_handler)

    _file_handler = logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_fmt)
    logger.addHandler(_file_handler)


# ===================================================================
# Helper utilities
# ===================================================================

def _month_ranges(start_date: datetime, end_date: datetime) -> list[tuple[datetime, datetime]]:
    """
    Split [start_date, end_date] into a list of (chunk_start, chunk_end) tuples
    where each chunk spans roughly one calendar month.

    Returns:
        A list of (start, end) datetime pairs covering the full range.
    """
    from dateutil.relativedelta import relativedelta
    ranges: list[tuple[datetime, datetime]] = []
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + relativedelta(months=1), end_date)
        ranges.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(seconds=1)
    return ranges


def _safe_request(
    url: str,
    params: dict[str, Any],
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    retries: int = MAX_RETRIES,
    backoff: float = RETRY_BACKOFF_FACTOR,
    delay: float = REQUEST_DELAY_SECONDS,
) -> requests.Response | None:
    """
    Perform an HTTP GET with exponential-backoff retries.

    Args:
        url: Target URL.
        params: Query string parameters.
        timeout: Per-request timeout in seconds.
        retries: Maximum number of retry attempts.
        backoff: Multiplier for exponential backoff.
        delay: Polite delay (seconds) before each request.

    Returns:
        A requests.Response on success, or None after all retries are exhausted.
    """
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, retries + 1):
        try:
            time.sleep(delay)
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            wait = backoff ** attempt
            logger.warning(
                "Timeout on attempt %d/%d for %s, retrying in %.1fs",
                attempt, retries, url, wait,
            )
            time.sleep(wait)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            if status == 429:
                wait = backoff ** attempt * 2
                logger.warning(
                    "Rate-limited (429) on attempt %d/%d, retrying in %.1fs",
                    attempt, retries, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "HTTP %s on attempt %d/%d for %s: %s",
                    status, attempt, retries, url, exc,
                )
                time.sleep(backoff ** attempt)
        except requests.exceptions.RequestException as exc:
            logger.error(
                "Request error on attempt %d/%d for %s: %s",
                attempt, retries, url, exc,
            )
            time.sleep(backoff ** attempt)

    logger.error("All %d retries exhausted for %s", retries, url)
    return None


# ===================================================================
# 1. GDELT-based collector (primary)
# ===================================================================

def fetch_gdelt_articles(
    query: str,
    start_date: datetime,
    end_date: datetime,
    source_country: str = "IN",
) -> list[dict]:
    """
    Query the GDELT 2.0 DOC API for a single time window.

    Args:
        query: Free-text search query (e.g., company name).
        start_date: Window start (inclusive).
        end_date: Window end (inclusive).
        source_country: ISO-2 country code filter (default "IN" for India).

    Returns:
        A list of dicts, each containing: title, url, date, source, tone,
        language, source_country.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": 250,
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
        "format": "json",
        "sourcelang": "english",
        "sourcecountry": source_country,
    }

    logger.debug(
        "GDELT query: '%s' | %s -> %s",
        query,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    resp = _safe_request(GDELT_DOC_API_URL, params)
    if resp is None:
        return []

    # GDELT returns a JSON object with an "articles" key
    try:
        data = resp.json()
    except ValueError:
        logger.error("Non-JSON response from GDELT for query '%s'", query)
        return []

    raw_articles = data.get("articles", [])
    if not raw_articles:
        logger.debug("No articles returned for query '%s' in this window", query)
        return []

    articles: list[dict] = []
    for art in raw_articles:
        articles.append(
            {
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "date": art.get("seendate", art.get("date", "")),
                "source": art.get("domain", art.get("source", "")),
                "tone": art.get("tone", ""),
                "language": art.get("language", "English"),
                "source_country": art.get("sourcecountry", source_country),
            }
        )
    return articles


def collect_gdelt_news_for_stock(
    stock_key: str,
    stock_info: dict,
    save_dir: Path,
    lookback_years: int = NEWS_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """
    Collect GDELT news articles for a single stock over the lookback window.

    The full date range is chunked into ~monthly windows so that each API call
    stays within GDELT's per-request record limit (250).

    Args:
        stock_key: Short identifier, e.g. "RELIANCE".
        stock_info: Dict with keys like company_name, nse_ticker.
        save_dir: Directory to write the output CSV.
        lookback_years: How many years of history to fetch.

    Returns:
        A DataFrame of collected article metadata.
    """
    company_name = stock_info["company_name"]
    ticker = stock_info["nse_ticker"]
    aliases = stock_info.get("name_aliases", [])
    
    all_terms = [f'"{company_name}"', f'"{ticker}"'] + [f'"{a}"' for a in aliases]
    all_terms = list(dict.fromkeys(all_terms))[:5]  # deduplicate and cap length
    query = " OR ".join(all_terms)

    from dateutil.relativedelta import relativedelta
    end_date = datetime.utcnow()
    start_date = end_date - relativedelta(years=lookback_years)

    logger.info(
        "Collecting GDELT news for %s (%s) from %s to %s",
        stock_key,
        company_name,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    all_articles: list[dict] = []
    chunks = _month_ranges(start_date, end_date)
    logger.info("Split into %d monthly chunks", len(chunks))

    for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        logger.debug(
            "  Chunk %d/%d: %s -> %s",
            idx, len(chunks),
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )
        batch = fetch_gdelt_articles(query, chunk_start, chunk_end)
        all_articles.extend(batch)
        logger.debug("  -> %d articles in this chunk, %d total", len(batch), len(all_articles))

    if not all_articles:
        logger.warning("No GDELT articles found for %s", stock_key)
        df = pd.DataFrame(
            columns=["title", "url", "date", "source", "tone", "language", "source_country"]
        )
    else:
        df = pd.DataFrame(all_articles)
        # Remove exact-duplicate URLs
        before = len(df)
        df.drop_duplicates(subset=["url"], inplace=True)
        after = len(df)
        if before != after:
            logger.info("Removed %d duplicate URLs for %s", before - after, stock_key)

    # Save CSV
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{stock_key}_news_gdelt.csv"
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved %d GDELT articles -> %s", len(df), out_path)

    return df


def collect_all_gdelt_news(
    stocks_config: dict | None = None,
    lookback_years: int = NEWS_LOOKBACK_YEARS,
    save_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Collect GDELT news for every stock in the configuration.

    Args:
        stocks_config: Mapping of stock keys to info dicts. Defaults to
            config.STOCKS.
        lookback_years: Number of years to look back.
        save_dir: Output directory. Defaults to config.RAW_NEWS_DIR.

    Returns:
        Dict mapping stock keys to their article DataFrames.
    """
    if stocks_config is None:
        stocks_config = STOCKS
    if save_dir is None:
        save_dir = RAW_NEWS_DIR

    results: dict[str, pd.DataFrame] = {}
    total = len(stocks_config)

    logger.info("=== Starting GDELT news collection for %d stocks ===", total)

    for idx, (key, info) in enumerate(stocks_config.items(), 1):
        logger.info("[%d/%d] %s", idx, total, key)
        
        out_path = save_dir / f"{key}_news_gdelt.csv"
        if out_path.exists():
            try:
                df_existing = pd.read_csv(out_path)
                if not df_existing.empty:
                    logger.info("Skipping %s, already exists", key)
                    results[key] = df_existing
                    continue
            except Exception:
                pass
            
        try:
            df = collect_gdelt_news_for_stock(key, info, save_dir, lookback_years)
            results[key] = df
        except Exception:
            logger.exception("Unexpected error collecting GDELT news for %s", key)
            results[key] = pd.DataFrame()
            
        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info("=== GDELT news collection complete ===")
    return results


# ===================================================================
# 2. NewsAPI-based collector (secondary, optional)
# ===================================================================

NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"


def fetch_newsapi_articles(
    query: str,
    api_key: str,
    from_date: str,
    to_date: str,
) -> list[dict]:
    """
    Fetch articles from the NewsAPI v2 /everything endpoint.

    The free tier limits history to roughly one month, so callers should
    set from_date accordingly.

    Args:
        query: Search query string.
        api_key: NewsAPI API key.
        from_date: Start date in YYYY-MM-DD format.
        to_date: End date in YYYY-MM-DD format.

    Returns:
        A list of dicts with keys: title, url, date, source, description.
    """
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }

    logger.debug("NewsAPI query: '%s' | %s -> %s", query, from_date, to_date)

    resp = _safe_request(NEWSAPI_EVERYTHING_URL, params)
    if resp is None:
        return []

    try:
        data = resp.json()
    except ValueError:
        logger.error("Non-JSON response from NewsAPI for query '%s'", query)
        return []

    if data.get("status") != "ok":
        logger.error(
            "NewsAPI error: %s - %s",
            data.get("code", "unknown"),
            data.get("message", "no message"),
        )
        return []

    raw_articles = data.get("articles", [])
    articles: list[dict] = []
    for art in raw_articles:
        articles.append(
            {
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "date": art.get("publishedAt", ""),
                "source": art.get("source", {}).get("name", ""),
                "description": art.get("description", ""),
            }
        )
    return articles


def collect_newsapi_news_for_stock(
    stock_key: str,
    stock_info: dict,
    save_dir: Path,
    api_key: str,
) -> pd.DataFrame:
    """
    Collect NewsAPI articles for a single stock.

    Only the most recent ~30 days are fetched because the free NewsAPI tier
    restricts historical access.

    Args:
        stock_key: Short identifier, e.g. "RELIANCE".
        stock_info: Dict with company_name and nse_ticker.
        save_dir: Directory to write the output CSV.
        api_key: Valid NewsAPI key.

    Returns:
        A DataFrame of collected article metadata.
    """
    company_name = stock_info["company_name"]
    ticker = stock_info["nse_ticker"]
    aliases = stock_info.get("name_aliases", [])
    
    all_terms = [f'"{company_name}"', f'"{ticker}"'] + [f'"{a}"' for a in aliases]
    all_terms = list(dict.fromkeys(all_terms))  # deduplicate
    query = " OR ".join(all_terms)

    to_date = datetime.utcnow().strftime("%Y-%m-%d")
    from_date = (datetime.utcnow() - timedelta(days=29)).strftime("%Y-%m-%d")

    logger.info(
        "Collecting NewsAPI articles for %s (%s), %s to %s",
        stock_key, company_name, from_date, to_date,
    )

    articles = fetch_newsapi_articles(query, api_key, from_date, to_date)

    if not articles:
        logger.warning("No NewsAPI articles found for %s", stock_key)
        df = pd.DataFrame(columns=["title", "url", "date", "source", "description"])
    else:
        df = pd.DataFrame(articles)
        df.drop_duplicates(subset=["url"], inplace=True)

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{stock_key}_news_newsapi.csv"
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved %d NewsAPI articles -> %s", len(df), out_path)

    return df


def collect_all_newsapi_news(
    stocks_config: dict | None = None,
    save_dir: Path | None = None,
    api_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Collect NewsAPI news for every stock in the configuration.

    If no valid API key is found, the function logs a warning and returns
    an empty dict without raising an error.

    Args:
        stocks_config: Stock configuration dict. Defaults to config.STOCKS.
        save_dir: Output directory. Defaults to config.RAW_NEWS_DIR.
        api_key: NewsAPI key. Defaults to config.NEWSAPI_KEY.

    Returns:
        Dict mapping stock keys to their article DataFrames.
    """
    if stocks_config is None:
        stocks_config = STOCKS
    if save_dir is None:
        save_dir = RAW_NEWS_DIR
    if api_key is None:
        api_key = NEWSAPI_KEY

    if not api_key:
        logger.warning(
            "NEWSAPI_KEY is not set or empty. Skipping NewsAPI collection. "
            "Set the NEWSAPI_KEY environment variable to enable this collector."
        )
        return {}

    results: dict[str, pd.DataFrame] = {}
    total = len(stocks_config)

    logger.info("=== Starting NewsAPI news collection for %d stocks ===", total)

    for idx, (key, info) in enumerate(stocks_config.items(), 1):
        logger.info("[%d/%d] %s", idx, total, key)
        try:
            df = collect_newsapi_news_for_stock(key, info, save_dir, api_key)
            results[key] = df
        except Exception:
            logger.exception("Unexpected error collecting NewsAPI news for %s", key)
            results[key] = pd.DataFrame()

    logger.info("=== NewsAPI news collection complete ===")
    return results


# ===================================================================
# Main entry point
# ===================================================================

def main() -> None:
    """
    Run both GDELT and NewsAPI collectors for all configured stocks.

    Intended to be called via: python -m data_collection.news
    """
    logger.info("=" * 60)
    logger.info("News data collection started at %s", datetime.utcnow().isoformat())
    logger.info("=" * 60)

    # 1. GDELT (primary)
    gdelt_results = collect_all_gdelt_news()
    total_gdelt = sum(len(df) for df in gdelt_results.values())
    logger.info("Total GDELT articles across all stocks: %d", total_gdelt)

    # 2. NewsAPI (secondary, optional)
    newsapi_results = collect_all_newsapi_news()
    total_newsapi = sum(len(df) for df in newsapi_results.values())
    logger.info("Total NewsAPI articles across all stocks: %d", total_newsapi)

    logger.info("=" * 60)
    logger.info("News data collection finished at %s", datetime.utcnow().isoformat())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

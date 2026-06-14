#!/usr/bin/env python3
"""
run_collection.py
=================
Top-level CLI entry point for running all data collectors in the
insider-sentiment-framework.

Usage examples:
    # Run all collectors
    python run_collection.py --all

    # Run only the price collector
    python run_collection.py --prices

    # Run only insider trading collector
    python run_collection.py --insider

    # Run only news collectors (GDELT + optional NewsAPI)
    python run_collection.py --news

    # Combine flags
    python run_collection.py --prices --insider

    # Run news with NewsAPI enabled (requires NEWSAPI_KEY in .env or environment)
    python run_collection.py --news --newsapi
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from config.settings import (
    LOG_FILE,
    PRICE_LOOKBACK_YEARS,
    INSIDER_LOOKBACK_YEARS,
    NEWS_LOOKBACK_YEARS,
    RAW_PRICES_DIR,
    RAW_INSIDER_DIR,
    RAW_NEWS_DIR,
    NEWSAPI_KEY,
)
from config.universe import STOCKS


def setup_logging() -> logging.Logger:
    """
    Configure root logger to write to both the console and a log file.

    Returns:
        logging.Logger: configured root logger.
    """
    logger = logging.getLogger("insider_sentiment")
    logger.setLevel(logging.DEBUG)

    # Console handler: INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    # File handler: DEBUG and above (captures everything)
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def run_price_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING PRICE DATA COLLECTION")
    logger.info("=" * 60)
    try:
        from data_collection.prices import download_all_prices
        from config.settings import INDICES
        
        # If STOCKS is empty, we can't collect prices!
        symbols = list(STOCKS.keys())
        if not symbols:
            logger.warning("STOCKS is empty. Run universe generation first or supply a valid universe.csv")
            return False
            
        results = download_all_prices(symbols=symbols, indices_config=INDICES)
        successful = [k for k, v in results.items() if v is not None and not v.empty]
        failed = [k for k, v in results.items() if v is None or v.empty]
        logger.info(f"Price collection complete: {len(successful)} succeeded, {len(failed)} failed")
        return len(failed) == 0
    except Exception as exc:
        logger.error("Price collection failed: %s", exc, exc_info=True)
        return False


def run_insider_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING INSIDER TRADING BULK DATA COLLECTION")
    logger.info("=" * 60)
    try:
        from data_collection.insider_trades import fetch_nse_insider_data
        from data_collection.insider_trades_clean import clean_insider_trades
        
        success = fetch_nse_insider_data()
        if not success:
            logger.warning("Some years failed to download for insider trades.")
            
        logger.info("Cleaning insider trades...")
        clean_insider_trades()
        return True
    except Exception as exc:
        logger.error("Insider trading collection failed: %s", exc, exc_info=True)
        return False


def run_news_collection(
    logger: logging.Logger, include_newsapi: bool = False
) -> bool:
    """
    Run news data collectors: always GDELT, optionally NewsAPI.

    Args:
        logger: Logger instance.
        include_newsapi: If True, also run the NewsAPI collector (requires key).

    Returns:
        bool: True if collection completed without critical errors.
    """
    logger.info("=" * 60)
    logger.info("STARTING NEWS DATA COLLECTION (GDELT)")
    logger.info("=" * 60)

    gdelt_ok = True
    newsapi_ok = True

    try:
        from data_collection.news import collect_all_gdelt_news

        results = collect_all_gdelt_news(
            stocks_config=STOCKS,
            lookback_years=NEWS_LOOKBACK_YEARS,
            save_dir=RAW_NEWS_DIR,
        )

        successful = [k for k, v in results.items() if v is not None and not v.empty]
        logger.info(
            "GDELT collection complete: %d stocks with articles", len(successful)
        )

    except Exception as exc:
        logger.error("GDELT news collection failed: %s", exc, exc_info=True)
        gdelt_ok = False

    # Optional: NewsAPI
    if include_newsapi:
        logger.info("=" * 60)
        logger.info("STARTING NEWS DATA COLLECTION (NewsAPI)")
        logger.info("=" * 60)

        if not NEWSAPI_KEY:
            logger.warning(
                "NEWSAPI_KEY not found in environment or .env file. "
                "Skipping NewsAPI collection. Set the key and retry."
            )
            newsapi_ok = True  # not a failure, just skipped
        else:
            try:
                from data_collection.news import collect_all_newsapi_news

                results = collect_all_newsapi_news(
                    stocks_config=STOCKS,
                    save_dir=RAW_NEWS_DIR,
                    api_key=NEWSAPI_KEY,
                )

                successful = [
                    k for k, v in results.items() if v is not None and not v.empty
                ]
                logger.info(
                    "NewsAPI collection complete: %d stocks with articles",
                    len(successful),
                )

            except Exception as exc:
                logger.error("NewsAPI collection failed: %s", exc, exc_info=True)
                newsapi_ok = False

    return gdelt_ok and newsapi_ok



def run_corporate_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING CORPORATE ACTIONS COLLECTION")
    logger.info("=" * 60)
    try:
        from data_collection.corporate_actions import collect_all_corporate_actions
        results = collect_all_corporate_actions()
        logger.info("Corporate actions collection complete: %d stocks processed.", len(results))
        return True
    except Exception as exc:
        logger.error("Corporate actions collection failed: %s", exc, exc_info=True)
        return False


def run_universe_generation(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("GENERATING UNIVERSE")
    logger.info("=" * 60)
    try:
        from config.universe import build_universe
        universe = build_universe()
        if not universe:
            logger.error("Failed to build universe.")
            return False
        logger.info(f"Universe built with {len(universe)} symbols.")
        return True
    except Exception as exc:
        logger.error("Universe generation failed: %s", exc, exc_info=True)
        return False

def run_social_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING SOCIAL DATA COLLECTION")
    logger.info("=" * 60)
    try:
        from data_collection.social import collect_reddit_sentiment
        from data_collection.social_twitter import collect_twitter_sentiment
        
        collect_reddit_sentiment()
        collect_twitter_sentiment()
        return True
    except Exception as exc:
        logger.error("Social data collection failed: %s", exc, exc_info=True)
        return False

def run_macro_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING MACRO DATA COLLECTION")
    logger.info("=" * 60)
    try:
        from data_collection.macro import collect_all_macro_data
        collect_all_macro_data()
        logger.info("Macro data collection complete.")
        return True
    except Exception as exc:
        logger.error("Macro data collection failed: %s", exc, exc_info=True)
        return False

def run_social_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING SOCIAL SENTIMENT COLLECTION")
    logger.info("=" * 60)
    logger.info("Social collection not yet fully implemented. Skipping.")
    return True

def generate_markdown_report(results: dict, elapsed: float, logger: logging.Logger):
    report_path = Path("data/raw/collection_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"# Insider Sentiment Framework - Data Collection Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Elapsed Time:** {elapsed:.1f} seconds\n\n")
        f.write("## Collection Summary\n\n")
        f.write("| Collector | Status |\n")
        f.write("|-----------|--------|\n")
        for collector_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            f.write(f"| {collector_name.capitalize()} | {status} |\n")
    logger.info("Saved summary report to %s", report_path)

def main():
    """Parse CLI arguments and run selected collectors."""
    parser = argparse.ArgumentParser(
        description="Insider Sentiment Framework - Data Collection Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_collection.py --all\n"
            "  python run_collection.py --prices --insider\n"
            "  python run_collection.py --news --newsapi\n"
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all collectors (prices, insider trades, news).",
    )
    parser.add_argument(
        "--prices",
        action="store_true",
        help="Run price data collector (yfinance).",
    )
    parser.add_argument(
        "--insider",
        action="store_true",
        help="Run insider trading data collector (BSE/NSE scraper).",
    )
    parser.add_argument(
        "--news",
        action="store_true",
        help="Run news data collector (GDELT, and optionally NewsAPI).",
    )
    parser.add_argument("--corporate", action="store_true", help="Run corporate actions collector.")
    parser.add_argument("--macro", action="store_true", help="Run macro data collector.")
    parser.add_argument("--social", action="store_true", help="Run social sentiment collector.")
    parser.add_argument("--universe", action="store_true", help="Build universe from insider trades.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (now default behavior for most collectors).")

    parser.add_argument(
        "--newsapi",
        action="store_true",
        help="Include NewsAPI collector when --news or --all is used. "
        "Requires NEWSAPI_KEY in environment or .env file.",
    )

    args = parser.parse_args()

    # If no flags provided, show help
    if not (args.all or args.prices or args.insider or args.news or args.universe or args.social or args.macro or args.corporate):
        parser.print_help()
        sys.exit(1)

    # Setup logging
    logger = setup_logging()
    logger.info("Insider Sentiment Framework - Data Collection")
    logger.info("Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Configured stocks: %s", ", ".join(STOCKS.keys()))

    # Try to load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
        logger.debug("Loaded .env file")
    except ImportError:
        logger.debug("python-dotenv not installed, skipping .env loading")

    start_time = time.time()
    results = {}

    # Run selected collectors
    if args.all or args.prices:
        results["prices"] = run_price_collection(logger)

    if args.all or args.insider:
        results["insider"] = run_insider_collection(logger)

    if args.all or args.news:
        results["news"] = run_news_collection(
            logger, include_newsapi=(args.newsapi or args.all)
        )

    if args.all or args.corporate:
        results["corporate"] = run_corporate_collection(logger)
    if args.all or args.macro:
        results["macro"] = run_macro_collection(logger)
    if args.all or args.social:
        results["social"] = run_social_collection(logger)
    if args.universe:
        results["universe"] = run_universe_generation(logger)


    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    for collector_name, success in results.items():
        status = "SUCCESS" if success else "COMPLETED WITH ERRORS"
        logger.info("  %-15s : %s", collector_name, status)

    logger.info("Total elapsed time: %.1f seconds", elapsed)

    generate_markdown_report(results, elapsed, logger)

    logger.info("Log file: %s", LOG_FILE)
    logger.info("=" * 60)

    # Exit with non-zero if any collector failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

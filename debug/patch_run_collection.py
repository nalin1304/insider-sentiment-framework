import re

file_path = "run_collection.py"
with open(file_path, "r") as f:
    content = f.read()

# Replace run_insider_collection
old_insider = """def run_insider_collection(logger: logging.Logger) -> bool:
    \"\"\"
    Run the insider trading data collector for all configured stocks.

    Returns:
        bool: True if collection completed without critical errors.
    \"\"\"
    logger.info("=" * 60)
    logger.info("STARTING INSIDER TRADING DATA COLLECTION")
    logger.info("=" * 60)

    try:
        from data_collection.insider_trades import collect_all_insider_trades

        results = collect_all_insider_trades(
            stocks_config=STOCKS,
            lookback_years=INSIDER_LOOKBACK_YEARS,
            save_dir=RAW_INSIDER_DIR,
        )

        successful = [k for k, v in results.items() if v is not None and not v.empty]
        empty = [k for k, v in results.items() if v is not None and v.empty]
        failed = [k for k, v in results.items() if v is None]

        logger.info(
            "Insider collection complete: %d with data, %d empty, %d failed",
            len(successful),
            len(empty),
            len(failed),
        )
        if failed:
            logger.warning("Failed scrips: %s", ", ".join(failed))

        return len(failed) == 0

    except Exception as exc:
        logger.error(
            "Insider trading collection failed with error: %s", exc, exc_info=True
        )
        return False"""

new_insider = """def run_insider_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING INSIDER TRADING BULK DATA COLLECTION")
    logger.info("=" * 60)
    try:
        from data_collection.insider_trades import collect_all_insider_trades
        from data_collection.insider_trades_clean import clean_insider_trades
        
        success = collect_all_insider_trades()
        if not success:
            logger.warning("Some years failed to download for insider trades.")
            
        logger.info("Cleaning insider trades...")
        clean_insider_trades()
        return True
    except Exception as exc:
        logger.error("Insider trading collection failed: %s", exc, exc_info=True)
        return False"""

content = content.replace(old_insider, new_insider)

# Replace run_price_collection
old_price = """def run_price_collection(logger: logging.Logger) -> bool:
    \"\"\"
    Run the price data collector for all configured stocks + SENSEX index.

    Returns:
        bool: True if collection completed without critical errors.
    \"\"\"
    logger.info("=" * 60)
    logger.info("STARTING PRICE DATA COLLECTION")
    logger.info("=" * 60)

    try:
        from data_collection.prices import download_all_prices

        results = download_all_prices(
            stocks_config=STOCKS,
            lookback_years=PRICE_LOOKBACK_YEARS,
            save_dir=RAW_PRICES_DIR,
        )

        successful = [k for k, v in results.items() if v is not None and not v.empty]
        failed = [k for k, v in results.items() if v is None or v.empty]

        logger.info(
            "Price collection complete: %d succeeded, %d failed",
            len(successful),
            len(failed),
        )
        if failed:
            logger.warning("Failed tickers: %s", ", ".join(failed))

        return len(failed) == 0

    except Exception as exc:
        logger.error("Price collection failed with error: %s", exc, exc_info=True)
        return False"""

new_price = """def run_price_collection(logger: logging.Logger) -> bool:
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
        return False"""

content = content.replace(old_price, new_price)

# Add run_universe
new_universe = """
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
"""

# Add run_social
new_social = """
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
"""

# Find place to insert
content = content.replace("def run_macro_collection", new_universe + new_social + "\ndef run_macro_collection")

# Add args
old_args = """    parser.add_argument(
        "--corporate",
        action="store_true",
        help="Run only the corporate actions collector",
    )"""

new_args = """    parser.add_argument(
        "--corporate",
        action="store_true",
        help="Run only the corporate actions collector",
    )
    parser.add_argument(
        "--universe",
        action="store_true",
        help="Build universe from insider trades",
    )
    parser.add_argument(
        "--social",
        action="store_true",
        help="Run only the social collector",
    )"""

content = content.replace(old_args, new_args)

# Add logic inside main()
main_logic_old = """    if args.macro:
        run_macro_collection(logger)
        ran_any = True"""

main_logic_new = """    if args.macro:
        run_macro_collection(logger)
        ran_any = True

    if args.universe:
        run_universe_generation(logger)
        ran_any = True

    if args.social:
        run_social_collection(logger)
        ran_any = True"""

content = content.replace(main_logic_old, main_logic_new)

# Add to --all
all_logic_old = """        run_news_collection(logger, include_newsapi=args.newsapi)
        run_corporate_collection(logger)
        run_macro_collection(logger)"""

all_logic_new = """        run_news_collection(logger, include_newsapi=args.newsapi)
        run_corporate_collection(logger)
        run_macro_collection(logger)
        run_social_collection(logger)"""

content = content.replace(all_logic_old, all_logic_new)

with open(file_path, "w") as f:
    f.write(content)

print("run_collection.py updated.")

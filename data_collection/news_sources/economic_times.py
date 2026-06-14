"""
Economic Times scraper stub.
"""
import pandas as pd
import logging

log = logging.getLogger("economic_times_scraper")

def collect_economic_times_news(stock_key: str, stock_info: dict) -> pd.DataFrame:
    """
    Placeholder for Economic Times scraper.
    """
    log.info("Economic Times scraper stub called for %s", stock_key)
    return pd.DataFrame(columns=["ticker", "date", "title", "summary", "url", "source"])

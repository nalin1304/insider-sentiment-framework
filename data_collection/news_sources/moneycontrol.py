"""
Moneycontrol scraper stub.
"""
import pandas as pd
import logging

log = logging.getLogger("moneycontrol_scraper")

def collect_moneycontrol_news(stock_key: str, stock_info: dict) -> pd.DataFrame:
    """
    Placeholder for Moneycontrol scraper.
    """
    log.info("Moneycontrol scraper stub called for %s", stock_key)
    return pd.DataFrame(columns=["ticker", "date", "title", "summary", "url", "source"])

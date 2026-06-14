import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_PATH = Path('data/processed/daily_sentiment.csv')
OUTPUT_PATH = Path('data/processed/sentiment_features.parquet')
PRICES_PATH = Path('data/processed/panel_prices.csv')

def build_sentiment_features():
    if not INPUT_PATH.exists():
        logger.error(f"Scored news not found at {INPUT_PATH}")
        return
        
    logger.info("Loading scored news...")
    news = pd.read_csv(INPUT_PATH)
    
    # Parse dates
    try:
        # GNews dates can be a bit messy, let's coerce and normalize
        news['date'] = pd.to_datetime(news['Date'], errors='coerce').dt.normalize()
    except Exception as e:
        logger.error(f"Failed to parse dates: {e}")
        return
        
    news = news.dropna(subset=['date'])
    news['ticker'] = news['Ticker']
    
    logger.info("Using pre-aggregated daily sentiment scores...")
    daily = news[['ticker', 'date', 'daily_sentiment_score', 'news_volume']].copy()
    
    logger.info("Aligning with trading calendar...")
    MASTER_PATH = Path('data/processed/master_dataset.parquet')
    if not MASTER_PATH.exists():
        logger.error("master_dataset.parquet not found. Cannot align calendar.")
        return
        
    prices = pd.read_parquet(MASTER_PATH, columns=['ticker', 'date'])
    prices['date'] = pd.to_datetime(prices['date']).dt.normalize()
    trading_calendar = prices[['ticker', 'date']].drop_duplicates().sort_values(['ticker', 'date'])
    
    # Merge daily sentiment onto the trading calendar
    merged = pd.merge(trading_calendar, daily, on=['ticker', 'date'], how='left')
    
    # For days with no news, volume is 0
    merged['news_volume'] = merged['news_volume'].fillna(0)
    
    # For daily_sentiment_score, if there's no news, we could fill 0 (neutral), 
    # but that pulls rolling averages down incorrectly if we just want average sentiment of actual news.
    # We will compute rolling averages ignoring NaNs.
    
    logger.info("Computing rolling metrics...")
    
    # Set index for rolling operations
    merged = merged.set_index('date').sort_index()
    
    results = []
    
    # Process each ticker
    for ticker, group in merged.groupby('ticker'):
        df = group.copy()
        
        # 3-day and 7-day rolling average of sentiment (ignoring NaNs)
        df['sentiment_3d_avg'] = df['daily_sentiment_score'].rolling(window=3, min_periods=1).mean()
        df['sentiment_7d_avg'] = df['daily_sentiment_score'].rolling(window=7, min_periods=1).mean()
        
        # 7-day rolling sum of volume
        df['news_volume_7d'] = df['news_volume'].rolling(window=7, min_periods=1).sum()
        
        # Fill remaining NaNs with 0 (neutral sentiment)
        df['daily_sentiment_score'] = df['daily_sentiment_score'].fillna(0)
        df['sentiment_3d_avg'] = df['sentiment_3d_avg'].fillna(0)
        df['sentiment_7d_avg'] = df['sentiment_7d_avg'].fillna(0)
        
        # Add boolean flag for has_news today
        df['has_news'] = (df['news_volume'] > 0).astype(int)
        
        results.append(df.reset_index())
        
    final_df = pd.concat(results, ignore_index=True)
    
    logger.info(f"Saving {len(final_df)} rows to {OUTPUT_PATH}")
    final_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    build_sentiment_features()

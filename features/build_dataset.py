import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from config.settings import RAW_DIR, PROCESSED_DIR, RAW_PRICES_DIR, RAW_INDICES_DIR, RAW_MACRO_DIR, RAW_CORP_ACTIONS_DIR
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config.settings import RAW_DIR, PROCESSED_DIR, RAW_PRICES_DIR, RAW_INDICES_DIR, RAW_MACRO_DIR, RAW_CORP_ACTIONS_DIR

from features.insider_features import load_and_preprocess_insider_trades, aggregate_daily_insider_trades, build_insider_features
from features.sentiment_features import build_sentiment_features
from features.macro_features import build_macro_features
from features.market_outcomes import load_all_prices, build_market_outcomes

logger = logging.getLogger("build_dataset")

def load_corporate_actions(corp_dir: Path) -> pd.DataFrame:
    """Load all corporate action dates for the proximity flag."""
    dfs = []
    if corp_dir.exists():
        for f in corp_dir.glob("*.csv"):
            ticker = f.stem.split('_')[0] + '.NS'
            try:
                df = pd.read_csv(f)
                if 'Date' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce').dt.tz_localize(None)
                    df['ticker'] = ticker
                    df['has_event'] = 1
                    dfs.append(df[['ticker', 'date', 'has_event']].dropna())
            except Exception as e:
                pass
    
    if dfs:
        events = pd.concat(dfs, ignore_index=True)
        # Drop duplicates if multiple events on same day
        events = events.drop_duplicates(subset=['ticker', 'date'])
        return events
    return pd.DataFrame(columns=['ticker', 'date', 'has_event'])

def build_master_dataset():
    """
    Orchestrate the creation of the master panel dataset.
    Strictly NO synthetic data. Left joins everything to the trading calendar.
    """
    logger.info("Building Master Dataset...")
    
    # 1. Load Universe
    universe_file = Path(__file__).resolve().parent.parent / "config" / "universe.csv"
    if not universe_file.exists():
        logger.error("Universe CSV not found. Cannot proceed.")
        return
        
    universe_df = pd.read_csv(universe_file)
    universe_df['ticker'] = universe_df['ticker'].astype(str).str.replace(r'\.NS$', '', regex=True)
    
    # 2. Load Prices & Market Outcomes
    logger.info("Loading prices and calculating market outcomes...")
    panel_prices, panel_indices = load_all_prices(RAW_PRICES_DIR, RAW_INDICES_DIR)
    
    if panel_prices.empty:
        logger.error("No real price data found. Halting build to ensure no synthetic data is used.")
        return
        
    outcomes_df = build_market_outcomes(panel_prices, panel_indices, universe_df)
    
    # Restrict to multi-year insider window (~7 years: 2018-05-31 to 2026-06-13)
    outcomes_df = outcomes_df[(outcomes_df['date'] >= '2018-05-31') & (outcomes_df['date'] <= '2026-06-13')]
    
    # Create the master trading calendar from the price data
    trading_calendar = outcomes_df[['ticker', 'date']].drop_duplicates().sort_values(['ticker', 'date'])
    
    # 3. Insider Features
    logger.info("Building insider features...")
    insider_file = PROCESSED_DIR / "insider_trades_clean.csv"
    if insider_file.exists():
        raw_insider = load_and_preprocess_insider_trades(insider_file)
        daily_insider = aggregate_daily_insider_trades(raw_insider)
        event_dates = load_corporate_actions(RAW_CORP_ACTIONS_DIR)
        insider_df = build_insider_features(daily_insider, trading_calendar, universe_df, event_dates)
    else:
        logger.warning("No clean insider data found.")
        insider_df = pd.DataFrame()
        
    # 4. Sentiment Features
    logger.info("Building sentiment features...")
    sentiment_file = PROCESSED_DIR / "sentiment_features.parquet"
    sentiment_df = pd.DataFrame()
    
    if sentiment_file.exists():
        logger.info("Found sentiment_features.parquet, integrating into features...")
        sentiment_df = pd.read_parquet(sentiment_file)
        # Drop trading calendar duplicates just in case
        sentiment_df = sentiment_df.drop_duplicates(subset=['ticker', 'date'])
    else:
        logger.info("No sentiment_features.parquet found. Skipping sentiment features.")
    
    # 5. Macro Features
    logger.info("Building macro features...")
    macro_df = build_macro_features(RAW_MACRO_DIR)
    
    # 6. Merge everything
    logger.info("Merging into master panel...")
    master = outcomes_df.copy()
    
    if not insider_df.empty:
        master = pd.merge(master, insider_df, on=['ticker', 'date'], how='left')
        
        # Calculate Liquidity Context (Trade Size vs. ADV)
        if 'net_insider_value_30d' in master.columns and 'adv_30d' in master.columns:
            avg_daily_value_30d = master['adv_30d'] * master['Close']
            master['insider_value_to_adv_30d'] = np.where(avg_daily_value_30d > 0, master['net_insider_value_30d'] / avg_daily_value_30d, 0)
        
        if 'net_insider_value_90d' in master.columns and 'adv_90d' in master.columns:
            avg_daily_value_90d = master['adv_90d'] * master['Close']
            master['insider_value_to_adv_90d'] = np.where(avg_daily_value_90d > 0, master['net_insider_value_90d'] / avg_daily_value_90d, 0)
            
        if 'open_market_net_insider_value_30d' in master.columns and 'adv_30d' in master.columns:
            avg_daily_value_30d = master['adv_30d'] * master['Close']
            master['open_market_insider_value_to_adv_30d'] = np.where(avg_daily_value_30d > 0, master['open_market_net_insider_value_30d'] / avg_daily_value_30d, 0)
            
        if 'open_market_net_insider_value_90d' in master.columns and 'adv_90d' in master.columns:
            avg_daily_value_90d = master['adv_90d'] * master['Close']
            master['open_market_insider_value_to_adv_90d'] = np.where(avg_daily_value_90d > 0, master['open_market_net_insider_value_90d'] / avg_daily_value_90d, 0)
        
    if not sentiment_df.empty:
        master = pd.merge(master, sentiment_df, on=['ticker', 'date'], how='left')
        
        # 6a. Compute Interaction Features
        if 'core_net_insider_value_30d' in master.columns and 'sentiment_7d_avg' in master.columns:
            # Interaction: Insider buy value * short-term sentiment
            master['insider_sentiment_interaction'] = master['core_net_insider_value_30d'] * master['sentiment_7d_avg']
        
    if not macro_df.empty:
        master = pd.merge(master, macro_df, on='date', how='left')
        
    # Write to processed dir
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = PROCESSED_DIR / "master_dataset.parquet"
    
    try:
        logger.info(f"Saving to {out_parquet}...")
        master.to_parquet(out_parquet, index=False)
    except Exception as e:
        logger.warning(f"Could not save parquet: {e}")
        
    logger.info(f"Done. Master dataset shape: {master.shape}")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    build_master_dataset()

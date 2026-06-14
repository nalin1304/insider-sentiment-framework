import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_all_prices(prices_dir: Path, indices_dir: Path) -> (pd.DataFrame, pd.DataFrame):
    """Load all individual price CSVs into panel dataframes."""
    price_dfs = []
    if prices_dir.exists():
        for f in prices_dir.glob("*.csv"):
            ticker = f.stem
            try:
                df = pd.read_csv(f)
                if not df.empty and 'Date' in df.columns:
                    df['ticker'] = ticker
                    price_dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")
                
    panel_prices = pd.concat(price_dfs, ignore_index=True) if price_dfs else pd.DataFrame()
    if not panel_prices.empty:
        panel_prices['Date'] = pd.to_datetime(panel_prices['Date'], utc=True).dt.tz_localize(None)
        
    idx_dfs = []
    if indices_dir.exists():
        for f in indices_dir.glob("*.csv"):
            ticker = f.stem
            try:
                df = pd.read_csv(f)
                if not df.empty and 'Date' in df.columns:
                    df['ticker'] = ticker
                    idx_dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")
                
    panel_indices = pd.concat(idx_dfs, ignore_index=True) if idx_dfs else pd.DataFrame()
    if not panel_indices.empty:
        panel_indices['Date'] = pd.to_datetime(panel_indices['Date'], utc=True).dt.tz_localize(None)
        
    return panel_prices, panel_indices

def build_market_outcomes(panel_prices: pd.DataFrame, panel_indices: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forward-looking labels:
    - abnormal_return_5d / 10d / 20d
    - sector_abnormal_return_5d / 10d / 20d
    - realized_volatility_10d (forward)
    - max_drawdown_20d (forward)
    """
    if panel_prices.empty:
        return pd.DataFrame()
        
    df = panel_prices.sort_values(['ticker', 'Date']).copy()
    
    # Filter out any indices or symbols not in the strict universe
    universe_tickers = set(universe_df['ticker'].astype(str))
    df = df[df['ticker'].isin(universe_tickers)]
    
    df = df.rename(columns={'Date': 'date'})
    
    # Calculate daily returns
    df['ret_1d'] = df.groupby('ticker')['Close'].pct_change()
    
    # --- Technical Indicators ---
    # 14-day RSI
    delta = df.groupby('ticker')['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use exponential moving average for RSI calculation (standard Wilders smoothing)
    avg_gain = gain.groupby(df['ticker']).transform(lambda x: x.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
    avg_loss = loss.groupby(df['ticker']).transform(lambda x: x.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
    
    rs = avg_gain / avg_loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    # Handle edge case where loss is 0 (rs is inf)
    df.loc[avg_loss == 0, 'rsi_14d'] = 100
    
    # 200-day MA distance
    ma_200 = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=200, min_periods=50).mean())
    df['dist_from_200d_ma'] = (df['Close'] - ma_200) / ma_200
    
    # Average Daily Volume (30d and 90d)
    df['adv_30d'] = df.groupby('ticker')['Volume'].transform(lambda x: x.rolling(window=21, min_periods=5).mean())
    df['adv_90d'] = df.groupby('ticker')['Volume'].transform(lambda x: x.rolling(window=63, min_periods=15).mean())
    
    # Forward Returns (5d, 10d, 20d, 40d, 60d)
    # Forward return over N days is: Close_{t+N} / Close_t - 1
    # We use shift(-N)
    grouped = df.groupby('ticker')
    df['fwd_ret_5d'] = grouped['Close'].shift(-5) / df['Close'] - 1
    df['fwd_ret_10d'] = grouped['Close'].shift(-10) / df['Close'] - 1
    df['fwd_ret_20d'] = grouped['Close'].shift(-20) / df['Close'] - 1
    df['fwd_ret_40d'] = grouped['Close'].shift(-40) / df['Close'] - 1
    df['fwd_ret_60d'] = grouped['Close'].shift(-60) / df['Close'] - 1
    
    # Realized Volatility 10d (forward)
    # Standard deviation of daily returns over the next 10 days
    # Since we need forward looking, shift backward by the window length after computing rolling over past
    fwd_vol = df.groupby('ticker')['ret_1d'].transform(lambda x: x.rolling(window=10, min_periods=5).std().shift(-10))
    df['realized_volatility_10d'] = fwd_vol * np.sqrt(252) # Annualized
    
    fwd_vol_40d = df.groupby('ticker')['ret_1d'].transform(lambda x: x.rolling(window=40, min_periods=20).std().shift(-40))
    df['realized_volatility_40d'] = fwd_vol_40d * np.sqrt(252) # Annualized

    # Max Drawdown 20d and 40d (forward)
    # Min(Close_{t+1...t+20}) / Close_t - 1
    fwd_min = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=20, min_periods=5).min().shift(-20))
    df['max_drawdown_20d'] = fwd_min / df['Close'] - 1
    df['max_drawdown_20d'] = df['max_drawdown_20d'].clip(upper=0)
    
    fwd_min_40d = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=40, min_periods=20).min().shift(-40))
    df['max_drawdown_40d'] = fwd_min_40d / df['Close'] - 1
    df['max_drawdown_40d'] = df['max_drawdown_40d'].clip(upper=0)
    
    # --- Benchmark Returns ---
    # Sensex
    sensex = None
    if not panel_indices.empty:
        sensex = panel_indices[panel_indices['ticker'] == '^BSESN'].copy()
        
    if sensex is not None and not sensex.empty:
        sensex = sensex.sort_values('Date').rename(columns={'Date': 'date'})
        sensex['sensex_fwd_5d'] = sensex['Close'].shift(-5) / sensex['Close'] - 1
        sensex['sensex_fwd_10d'] = sensex['Close'].shift(-10) / sensex['Close'] - 1
        sensex['sensex_fwd_20d'] = sensex['Close'].shift(-20) / sensex['Close'] - 1
        sensex['sensex_fwd_40d'] = sensex['Close'].shift(-40) / sensex['Close'] - 1
        sensex['sensex_fwd_60d'] = sensex['Close'].shift(-60) / sensex['Close'] - 1
        
        df = pd.merge(df, sensex[['date', 'sensex_fwd_5d', 'sensex_fwd_10d', 'sensex_fwd_20d', 'sensex_fwd_40d', 'sensex_fwd_60d']], on='date', how='left')
        
        df['abnormal_return_5d'] = df['fwd_ret_5d'] - df['sensex_fwd_5d']
        df['abnormal_return_10d'] = df['fwd_ret_10d'] - df['sensex_fwd_10d']
        df['abnormal_return_20d'] = df['fwd_ret_20d'] - df['sensex_fwd_20d']
        df['abnormal_return_40d'] = df['fwd_ret_40d'] - df['sensex_fwd_40d']
        df['abnormal_return_60d'] = df['fwd_ret_60d'] - df['sensex_fwd_60d']
    else:
        logger.warning("SENSEX index not found. Abnormal returns will be NaN.")
        df['abnormal_return_5d'] = np.nan
        df['abnormal_return_10d'] = np.nan
        df['abnormal_return_20d'] = np.nan
        df['abnormal_return_40d'] = np.nan
        df['abnormal_return_60d'] = np.nan
        
    # --- Sector Abnormal Returns ---
    # Merge sector
    df = pd.merge(df, universe_df[['ticker', 'sector']], on='ticker', how='left')
    
    # Calculate daily sector average forward returns (excluding self)
    sector_stats = df.groupby(['sector', 'date']).agg(
        sec_sum_5d=('fwd_ret_5d', 'sum'), sec_count_5d=('fwd_ret_5d', 'count'),
        sec_sum_10d=('fwd_ret_10d', 'sum'), sec_count_10d=('fwd_ret_10d', 'count'),
        sec_sum_20d=('fwd_ret_20d', 'sum'), sec_count_20d=('fwd_ret_20d', 'count'),
        sec_sum_40d=('fwd_ret_40d', 'sum'), sec_count_40d=('fwd_ret_40d', 'count'),
        sec_sum_60d=('fwd_ret_60d', 'sum'), sec_count_60d=('fwd_ret_60d', 'count')
    ).reset_index()
    
    df = pd.merge(df, sector_stats, on=['sector', 'date'], how='left')
    
    for d in [5, 10, 20, 40, 60]:
        # other_mean = (sum - self) / (count - 1)
        # if count <= 1, other_mean is nan
        other_mean = np.where(
            df[f'sec_count_{d}d'] > 1,
            (df[f'sec_sum_{d}d'] - df[f'fwd_ret_{d}d'].fillna(0)) / (df[f'sec_count_{d}d'] - 1),
            np.nan
        )
        df[f'sector_abnormal_return_{d}d'] = df[f'fwd_ret_{d}d'] - other_mean
        df = df.drop(columns=[f'sec_sum_{d}d', f'sec_count_{d}d'])
    
    # Clean up
    keep_cols = [
        'ticker', 'date', 'Close', 'Volume', 'adv_30d', 'adv_90d', 'rsi_14d', 'dist_from_200d_ma',
        'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d', 'fwd_ret_40d', 'fwd_ret_60d',
        'abnormal_return_5d', 'abnormal_return_10d', 'abnormal_return_20d', 'abnormal_return_40d', 'abnormal_return_60d',
        'sector_abnormal_return_5d', 'sector_abnormal_return_10d', 'sector_abnormal_return_20d', 'sector_abnormal_return_40d', 'sector_abnormal_return_60d',
        'realized_volatility_10d', 'realized_volatility_40d', 
        'max_drawdown_20d', 'max_drawdown_40d'
    ]
    return df[keep_cols]

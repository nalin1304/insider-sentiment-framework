import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Config mapping for insider role weight
INSIDER_ROLE_WEIGHTS = {
    "Promoters": 1.0,
    "Promoter Group": 0.9,
    "Director": 0.8,
    "Key Managerial Personnel": 0.8,
    "Employees/Designated Employees": 0.4,
    "Immediate relative": 0.6,
    "Other": 0.2,
    "-": 0.1
}

def load_and_preprocess_insider_trades(filepath: Path) -> pd.DataFrame:
    """Load and clean the raw insider trades for feature engineering."""
    df = pd.read_csv(filepath, low_memory=False)
    
    # Keep only relevant columns to save memory
    cols = {
        'SYMBOL': 'ticker',
        'CATEGORY OF PERSON': 'role',
        'NAME OF THE ACQUIRER/DISPOSER': 'acquirer_name',
        'NO. OF SECURITIES (ACQUIRED/DISPLOSED)': 'securities_acquired_disposed',
        'NO. OF SECURITY (PRIOR)': 'securities_prior',
        'VALUE OF SECURITY (ACQUIRED/DISPLOSED)': 'trade_value',
        'ACQUISITION/DISPOSAL TRANSACTION TYPE': 'trade_type',
        'DATE OF ALLOTMENT/ACQUISITION TO': 'transaction_date',
        'DATE OF INITMATION TO COMPANY': 'disclosure_date',
        'MODE OF ACQUISITION': 'trade_mode'
    }
    df = df[list(cols.keys())].rename(columns=cols)
    
    df['ticker'] = df['ticker'].astype(str).str.strip()
    
    # Parse dates
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True, errors='coerce')
    df['disclosure_date'] = pd.to_datetime(df['disclosure_date'], dayfirst=True, errors='coerce')
    
    # Clean trade value (remove commas, handle non-numeric)
    df['trade_value'] = df['trade_value'].astype(str).str.replace(',', '', regex=False).str.replace('-', '0', regex=False)
    df['trade_value'] = pd.to_numeric(df['trade_value'], errors='coerce').fillna(0)
    
    # Signed trade value (Buy = positive, Sell/Pledge = negative)
    df['trade_type'] = df['trade_type'].fillna('').str.lower()
    buy_mask = df['trade_type'].str.contains('buy|acquire|purchase')
    sell_mask = df['trade_type'].str.contains('sell|disposal')
    
    # Apply signs
    df.loc[sell_mask, 'trade_value'] *= -1
    df.loc[~buy_mask & ~sell_mask, 'trade_value'] = 0
    
    # Calculate disclosure lag
    df['disclosure_lag_days'] = (df['disclosure_date'] - df['transaction_date']).dt.days
    df['disclosure_lag_days'] = df['disclosure_lag_days'].clip(lower=0) # Handle negative anomalies
    
    # Map role weights
    df['role_weight'] = df['role'].map(INSIDER_ROLE_WEIGHTS).fillna(0.2)
    
    # Identify core insiders
    core_roles = ["Promoters", "Promoter Group", "Director", "Key Managerial Personnel"]
    df['is_core'] = df['role'].isin(core_roles)
    
    # Calculate pct_stake_change
    df['securities_acquired_disposed'] = df['securities_acquired_disposed'].astype(str).str.replace(',', '', regex=False).str.replace('-', '0', regex=False)
    df['securities_acquired_disposed'] = pd.to_numeric(df['securities_acquired_disposed'], errors='coerce').fillna(0)
    
    df['securities_prior'] = df['securities_prior'].astype(str).str.replace(',', '', regex=False).str.replace('-', '0', regex=False)
    df['securities_prior'] = pd.to_numeric(df['securities_prior'], errors='coerce').fillna(0)
    
    # Apply signs to acquired_disposed
    df.loc[sell_mask, 'securities_acquired_disposed'] *= -1
    df.loc[~buy_mask & ~sell_mask, 'securities_acquired_disposed'] = 0
    
    df['pct_stake_change'] = np.where(df['securities_prior'] > 0, 
                                      df['securities_acquired_disposed'] / df['securities_prior'], 
                                      0)
                                      
    # Core/Peripheral splits for values and stake change
    df['core_trade_value'] = np.where(df['is_core'], df['trade_value'], 0)
    df['peripheral_trade_value'] = np.where(~df['is_core'], df['trade_value'], 0)
    df['core_pct_stake_change'] = np.where(df['is_core'], df['pct_stake_change'], 0)
    df['peripheral_pct_stake_change'] = np.where(~df['is_core'], df['pct_stake_change'], 0)
    
    # Open Market trades filtering
    df['is_open_market'] = df['trade_mode'].fillna('').str.lower().str.contains('market purchase|market sale|market buy|open market')
    df['open_market_trade_value'] = np.where(df['is_open_market'], df['trade_value'], 0)
    
    # Counts
    df['core_trade_count'] = np.where((df['is_core']) & (df['trade_value'] != 0), 1, 0)
    df['peripheral_trade_count'] = np.where((~df['is_core']) & (df['trade_value'] != 0), 1, 0)
    df['open_market_trade_count'] = np.where((df['is_open_market']) & (df['trade_value'] != 0), 1, 0)
    
    # Drop rows without valid public disclosure dates
    df = df.dropna(subset=['disclosure_date'])
    df = df.sort_values(['ticker', 'disclosure_date'])
    
    return df

def aggregate_daily_insider_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple trades on the same day for the same ticker based on disclosure."""
    daily = df.groupby(['ticker', 'disclosure_date']).agg(
        net_trade_value=('trade_value', 'sum'),
        trade_count=('trade_value', 'count'),
        core_trade_value=('core_trade_value', 'sum'),
        peripheral_trade_value=('peripheral_trade_value', 'sum'),
        core_trade_count=('core_trade_count', 'sum'),
        peripheral_trade_count=('peripheral_trade_count', 'sum'),
        core_pct_stake_change=('core_pct_stake_change', 'sum'),
        peripheral_pct_stake_change=('peripheral_pct_stake_change', 'sum'),
        open_market_trade_value=('open_market_trade_value', 'sum'),
        open_market_trade_count=('open_market_trade_count', 'sum'),
        avg_role_weight=('role_weight', 'mean'),
        avg_disclosure_lag=('disclosure_lag_days', 'mean')
    ).reset_index()
    
    # Collect unique core buyers and sellers
    core_buys = df[df['is_core'] & (df['trade_value'] > 0)]
    core_sells = df[df['is_core'] & (df['trade_value'] < 0)]
    
    buyers_agg = core_buys.groupby(['ticker', 'disclosure_date'])['acquirer_name'].apply(lambda x: list(set(x.dropna()))).reset_index(name='core_buyers')
    sellers_agg = core_sells.groupby(['ticker', 'disclosure_date'])['acquirer_name'].apply(lambda x: list(set(x.dropna()))).reset_index(name='core_sellers')
    
    daily = pd.merge(daily, buyers_agg, on=['ticker', 'disclosure_date'], how='left')
    daily = pd.merge(daily, sellers_agg, on=['ticker', 'disclosure_date'], how='left')
    
    daily['core_buyers'] = daily['core_buyers'].apply(lambda x: x if isinstance(x, list) else [])
    daily['core_sellers'] = daily['core_sellers'].apply(lambda x: x if isinstance(x, list) else [])
    
    daily = daily.rename(columns={'disclosure_date': 'date'})
    return daily

def build_insider_features(daily_trades: pd.DataFrame, trading_calendar: pd.DataFrame, universe_df: pd.DataFrame, event_dates: pd.DataFrame = None) -> pd.DataFrame:
    """
    Given a daily aggregated insider trades dataframe and a trading calendar (all tickers x all dates),
    compute the rolling 30d/90d features.
    
    trading_calendar must have ['ticker', 'date'] and be sorted.
    universe_df must have ['ticker', 'sector'].
    event_dates must have ['ticker', 'date', 'has_event']
    """
    # Merge daily trades into full calendar
    df = pd.merge(trading_calendar, daily_trades, on=['ticker', 'date'], how='left')
    
    # Fill missing days with 0 for value and count
    cols_to_fill = ['net_trade_value', 'trade_count', 'core_trade_value', 'peripheral_trade_value', 
                    'core_trade_count', 'peripheral_trade_count', 'core_pct_stake_change', 'peripheral_pct_stake_change',
                    'open_market_trade_value', 'open_market_trade_count']
    for c in cols_to_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    
    if 'core_buyers' in df.columns:
        df['core_buyers'] = df['core_buyers'].apply(lambda x: x if isinstance(x, list) else [])
    if 'core_sellers' in df.columns:
        df['core_sellers'] = df['core_sellers'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Create rolling windows. Since data is daily, we can use rolling(window) if there are no gaps.
    # Because we use a trading calendar, there are weekends missing. 30 trading days ~ 42 calendar days.
    # Let's use 21 trading days for 1 month, and 63 for 3 months.
    
    df = df.sort_values(['ticker', 'date'])
    
    # Group by ticker for rolling
    grouped = df.groupby('ticker')
    
    df['net_insider_value_30d'] = grouped['net_trade_value'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['net_insider_value_90d'] = grouped['net_trade_value'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    
    df['insider_trade_count_30d'] = grouped['trade_count'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['insider_trade_count_90d'] = grouped['trade_count'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    
    # New rolling sums
    df['core_net_insider_value_30d'] = grouped['core_trade_value'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['core_net_insider_value_90d'] = grouped['core_trade_value'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    df['peripheral_net_insider_value_30d'] = grouped['peripheral_trade_value'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['peripheral_net_insider_value_90d'] = grouped['peripheral_trade_value'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    
    df['core_trade_count_30d'] = grouped['core_trade_count'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['core_trade_count_90d'] = grouped['core_trade_count'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    df['peripheral_trade_count_30d'] = grouped['peripheral_trade_count'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['peripheral_trade_count_90d'] = grouped['peripheral_trade_count'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    
    df['open_market_net_insider_value_30d'] = grouped['open_market_trade_value'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['open_market_net_insider_value_90d'] = grouped['open_market_trade_value'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    df['open_market_trade_count_30d'] = grouped['open_market_trade_count'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['open_market_trade_count_90d'] = grouped['open_market_trade_count'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    
    df['core_pct_stake_change_30d'] = grouped['core_pct_stake_change'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['core_pct_stake_change_90d'] = grouped['core_pct_stake_change'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    df['peripheral_pct_stake_change_30d'] = grouped['peripheral_pct_stake_change'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    df['peripheral_pct_stake_change_90d'] = grouped['peripheral_pct_stake_change'].rolling(window=63, min_periods=1).sum().reset_index(0, drop=True)
    
    # Distinct counting for clustering features
    def rolling_distinct_count(series, window):
        names = series.values
        res = np.zeros(len(names), dtype=int)
        for i in range(len(names)):
            start = max(0, i - window + 1)
            window_lists = [n for n in names[start:i+1] if isinstance(n, list) and len(n) > 0]
            s = set(item for sublist in window_lists for item in sublist)
            res[i] = len(s)
        return pd.Series(res, index=series.index)
        
    df['n_distinct_core_insiders_buying_30d'] = grouped['core_buyers'].transform(lambda x: rolling_distinct_count(x, 21))
    df['n_distinct_core_insiders_selling_30d'] = grouped['core_sellers'].transform(lambda x: rolling_distinct_count(x, 21))
    df['n_distinct_core_insiders_buying_90d'] = grouped['core_buyers'].transform(lambda x: rolling_distinct_count(x, 63))
    df['n_distinct_core_insiders_selling_90d'] = grouped['core_sellers'].transform(lambda x: rolling_distinct_count(x, 63))
    
    # Forward fill role weight and disclosure lag so we have the 'latest' values
    df['insider_role_weight'] = grouped['avg_role_weight'].ffill().fillna(0)
    df['disclosure_lag_days'] = grouped['avg_disclosure_lag'].ffill().fillna(0)
    
    # Days since last trade
    # Create a boolean mask where a trade occurred
    df['trade_occurred'] = df['trade_count'] > 0
    # Use cumsum to create groups of periods since last trade
    # We can get days since last trade by counting rows since last True
    # A fast vectorized way:
    trade_dates = df.loc[df['trade_occurred'], ['ticker', 'date']].copy()
    trade_dates['date_last_trade'] = trade_dates['date']
    trade_dates = trade_dates.sort_values('date')
    df = df.sort_values('date')
    df = pd.merge_asof(df, trade_dates, on='date', by='ticker')
    df = df.sort_values(['ticker', 'date'])
    df['days_since_last_insider_trade'] = (df['date'] - df['date_last_trade']).dt.days
    df['days_since_last_insider_trade'] = df['days_since_last_insider_trade'].fillna(9999) # 9999 means never
    df = df.drop(columns=['date_last_trade'])
    
    # Event proximity flag (if an insider trade occurred near a corporate action)
    df['event_proximity_flag'] = 0
    if event_dates is not None and not event_dates.empty:
        # event_dates has ['ticker', 'date', 'has_event']
        # We want to flag if there was a trade within e.g. 7 days of an event
        df = pd.merge(df, event_dates, on=['ticker', 'date'], how='left')
        df['has_event'] = df['has_event'].fillna(0)
        
        # We can find rolling sum of events
        grouped_events = df.groupby('ticker')
        # Check if any event happened in past 7 days
        events_past_7d = grouped_events['has_event'].rolling(window=7, min_periods=1).sum().reset_index(0, drop=True)
        
        df['event_proximity_flag'] = np.where(
            (df['trade_occurred']) & (events_past_7d > 0),
            1, 0
        )
        df = df.drop(columns=['has_event'])
        
    # Sector features
    # Merge sector
    df = pd.merge(df, universe_df[['ticker', 'sector']], on='ticker', how='left')
    
    # Aggregate net trade value by sector and date
    sector_daily = df.groupby(['sector', 'date'])['net_trade_value'].sum().reset_index()
    sector_daily = sector_daily.sort_values(['sector', 'date'])
    
    # Rolling 30d for sector
    sector_grouped = sector_daily.groupby('sector')
    sector_daily['sector_insider_sentiment'] = sector_grouped['net_trade_value'].rolling(window=21, min_periods=1).sum().reset_index(0, drop=True)
    
    # Merge back
    df = pd.merge(df, sector_daily[['sector', 'date', 'sector_insider_sentiment']], on=['sector', 'date'], how='left')
    
    # Cross-sectional z-score of net_insider_value_90d by sector/date
    sector_stats = df.groupby(['sector', 'date'])['net_insider_value_90d'].agg(['mean', 'std']).reset_index()
    sector_stats = sector_stats.rename(columns={'mean': 'sector_mean_90d', 'std': 'sector_std_90d'})
    
    df = pd.merge(df, sector_stats, on=['sector', 'date'], how='left')
    df['sector_std_90d'] = df['sector_std_90d'].replace(0, np.nan)
    df['sector_neutral_insider_value_90d'] = (df['net_insider_value_90d'] - df['sector_mean_90d']) / df['sector_std_90d']
    df['sector_neutral_insider_value_90d'] = df['sector_neutral_insider_value_90d'].fillna(0)
    
    # Cleanup intermediate columns
    cols_to_drop = [
        'net_trade_value', 'trade_count', 'avg_role_weight', 'avg_disclosure_lag', 'trade_occurred', 'date_last_trade',
        'core_trade_value', 'peripheral_trade_value', 'core_trade_count', 'peripheral_trade_count',
        'core_pct_stake_change', 'peripheral_pct_stake_change', 'core_buyers', 'core_sellers',
        'open_market_trade_value', 'open_market_trade_count', 'sector_mean_90d', 'sector_std_90d'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    return df

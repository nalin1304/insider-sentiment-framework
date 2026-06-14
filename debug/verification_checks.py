import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')

def main():
    print("=== 1. VERIFY THE DATE COLUMN ===")
    df_trades = pd.read_csv(DATA_DIR / 'processed' / 'insider_trades_clean.csv', low_memory=False)
    cols = df_trades.columns.tolist()
    
    date_col = 'DATE OF INITMATION TO COMPANY'
    if date_col in cols:
        print(f"Column '{date_col}' exists.")
        print("First 10 raw values:")
        print(df_trades[date_col].head(10).tolist())
        parsed_dates = pd.to_datetime(df_trades[date_col], format='%d-%b-%Y', errors='coerce')
        valid_dates = parsed_dates.dropna()
        print(f"Valid parsed dates: {len(valid_dates)} / {len(df_trades)}")
        print(f"Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
    else:
        print(f"Column '{date_col}' NOT found.")
        
    print("\n=== 2 & 3. CHECK FOR EVENT CLUSTERING & OVERLAPPING WINDOWS ===")
    CORE_CATEGORIES = ['Promoters', 'Promoter Group', 'Key Managerial Personnel']
    type_col = 'ACQUISITION/DISPOSAL TRANSACTION TYPE'
    df_trades[type_col] = df_trades[type_col].replace({'Acquisition': 'Buy', 'Disposal': 'Sell'})
    
    buys = df_trades[(df_trades[type_col] == 'Buy') & (df_trades['CATEGORY OF PERSON'].isin(CORE_CATEGORIES))].copy()
    buys['event_date'] = pd.to_datetime(buys[date_col], format='%d-%b-%Y', errors='coerce')
    buys = buys.dropna(subset=['event_date'])
    
    events = buys.groupby(['SYMBOL', 'event_date']).size().reset_index(name='count')
    events = events.sort_values(['SYMBOL', 'event_date'])
    
    n_distinct_tickers = events['SYMBOL'].nunique()
    events['week'] = events['event_date'].dt.isocalendar().week
    events['year'] = events['event_date'].dt.year
    n_distinct_weeks = events.groupby(['year', 'week']).ngroups
    
    print(f"(a) Distinct tickers with core buys: {n_distinct_tickers}")
    print(f"(b) Distinct calendar weeks with core buys: {n_distinct_weeks}")
    
    events_idx = events.set_index('event_date')
    max_cluster = events_idx.groupby('SYMBOL')['count'].rolling('28D').count().max()
    print(f"Max events sharing same ticker within 20 trading days (~28 cal days): {max_cluster}")
    
    events['prev_event_date'] = events.groupby('SYMBOL')['event_date'].shift(1)
    events['days_since_prev'] = (events['event_date'] - events['prev_event_date']).dt.days
    overlapping = events[events['days_since_prev'] <= 28]
    print(f"Fraction of core-buy events with another core-buy event in prior 28 days: {len(overlapping) / len(events):.2%}")
    
    print("\n=== 4. SEGMENTATION MULTIPLE-TESTING ===")
    seg = pd.read_csv('analysis/segmentation_metrics.csv')
    print(f"Total combinations tested: {len(seg)}")
    
if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path('data')
OUTPUT_DIR = Path('optimization')

def main():
    print("Loading data...")
    df_trades = pd.read_csv(DATA_DIR / 'processed' / 'insider_trades_clean.csv', low_memory=False)
    universe = pd.read_csv('config/universe.csv')
    
    symbol_to_ticker = dict(zip(universe['nse_symbol'], universe['ticker']))
    
    # 1. Filter for Core Buys
    CORE_CATEGORIES = ['Promoters', 'Promoter Group', 'Key Managerial Personnel']
    type_col = 'ACQUISITION/DISPOSAL TRANSACTION TYPE'
    df_trades[type_col] = df_trades[type_col].replace({'Acquisition': 'Buy', 'Disposal': 'Sell'})
    
    buys = df_trades[(df_trades[type_col] == 'Buy') & (df_trades['CATEGORY OF PERSON'].isin(CORE_CATEGORIES))].copy()
    buys['event_date'] = pd.to_datetime(buys['DATE OF INITMATION TO COMPANY'], errors='coerce')
    buys = buys.dropna(subset=['event_date'])
    
    events = buys.groupby(['SYMBOL', 'event_date']).size().reset_index()
    events['ticker'] = events['SYMBOL'].map(symbol_to_ticker)
    events = events.dropna(subset=['ticker'])
    
    print("Loading prices to establish trading calendar...")
    prices_dir = DATA_DIR / 'raw' / 'prices'
    prices_list = []
    
    for ticker_symbol in tqdm(universe['nse_symbol'].unique()):
        p_file = prices_dir / f"{ticker_symbol}.csv"
        if p_file.exists():
            try:
                p = pd.read_csv(p_file, usecols=['Date', 'Close'])
                p['ticker'] = symbol_to_ticker[ticker_symbol]
                p['Date'] = pd.to_datetime(p['Date'])
                prices_list.append(p)
            except Exception:
                pass
                
    df_prices = pd.concat(prices_list, ignore_index=True)
    df_prices = df_prices.sort_values(['ticker', 'Date']).drop_duplicates(['ticker', 'Date'])
    
    # Create wide dataframe of prices
    df_wide = df_prices.pivot(index='Date', columns='ticker', values='Close')
    trading_dates = df_wide.index
    
    print("Constructing signal matrix...")
    # Initialize boolean matrix
    signal_matrix = pd.DataFrame(False, index=trading_dates, columns=df_wide.columns)
    
    for _, row in tqdm(events.iterrows(), total=len(events)):
        ticker = row['ticker']
        edate = row['event_date']
        
        if ticker not in signal_matrix.columns:
            continue
            
        future_dates = trading_dates[trading_dates >= edate]
        if len(future_dates) == 0:
            continue
            
        start_idx = trading_dates.get_loc(future_dates[0]) + 1  # t+1 to avoid intraday look-ahead
        if start_idx >= len(trading_dates):
            continue
        # Mark True for the next 20 trading days
        end_idx = min(start_idx + 20, len(trading_dates))
        signal_matrix.iloc[start_idx:end_idx, signal_matrix.columns.get_loc(ticker)] = True

    # Reporting fraction
    total_cells = signal_matrix.shape[0] * signal_matrix.shape[1]
    active_cells = signal_matrix.sum().sum()
    fraction = active_cells / total_cells
    
    print(f"Total (date, ticker) combinations: {total_cells}")
    print(f"Active signal instances: {active_cells}")
    print(f"Fraction of time the signal is active: {fraction:.4%}")
    
    # Save the matrices
    signal_matrix.to_parquet(OUTPUT_DIR / 'signal_matrix.parquet')
    df_wide.to_parquet(OUTPUT_DIR / 'price_matrix.parquet')
    print("Saved signal_matrix.parquet and price_matrix.parquet")

if __name__ == '__main__':
    main()

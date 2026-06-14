import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm

DATA_DIR = Path('data')

def main():
    # Load events
    df_trades = pd.read_csv(DATA_DIR / 'processed' / 'insider_trades_clean.csv', low_memory=False)
    CORE_CATEGORIES = ['Promoters', 'Promoter Group', 'Key Managerial Personnel']
    type_col = 'ACQUISITION/DISPOSAL TRANSACTION TYPE'
    df_trades[type_col] = df_trades[type_col].replace({'Acquisition': 'Buy', 'Disposal': 'Sell'})
    valid_types = ['Buy', 'Sell']
    df_trades = df_trades[df_trades[type_col].isin(valid_types)].copy()
    df_trades['event_date'] = pd.to_datetime(df_trades['DATE OF INITMATION TO COMPANY'], format='%d-%b-%Y', errors='coerce')
    df_trades = df_trades.dropna(subset=['event_date'])
    df_trades['is_core'] = df_trades['CATEGORY OF PERSON'].isin(CORE_CATEGORIES)
    
    events = df_trades.groupby(['SYMBOL', 'event_date', type_col]).agg(has_core=('is_core', 'max')).reset_index()
    events['has_core'] = events['has_core'].astype(bool)
    
    # Load universe
    universe = pd.read_csv('config/universe.csv')
    symbol_to_sector = dict(zip(universe['nse_symbol'], universe['sector']))
    symbol_to_ticker = dict(zip(universe['nse_symbol'], universe['ticker']))
    ticker_to_sector = dict(zip(universe['ticker'], universe['sector']))
    
    events['sector'] = events['SYMBOL'].map(symbol_to_sector)
    events['ticker'] = events['SYMBOL'].map(symbol_to_ticker)
    events = events.dropna(subset=['sector', 'ticker'])
    
    # Load prices
    prices_list = []
    prices_dir = DATA_DIR / 'raw' / 'prices'
    for ticker_symbol in tqdm(events['SYMBOL'].unique()):
        p_file = prices_dir / f"{ticker_symbol}.csv"
        if p_file.exists():
            try:
                p = pd.read_csv(p_file, usecols=['Date', 'Close'])
                p['ticker'] = events[events['SYMBOL'] == ticker_symbol]['ticker'].iloc[0]
                p['Date'] = pd.to_datetime(p['Date'])
                p = p.sort_values('Date').drop_duplicates('Date')
                p['ret_1d'] = p['Close'].pct_change()
                prices_list.append(p)
            except Exception:
                continue
                
    df_prices = pd.concat(prices_list, ignore_index=True)
    df_prices['sector'] = df_prices['ticker'].map(ticker_to_sector)
    
    sector_rets = df_prices.groupby(['sector', 'Date'])['ret_1d'].mean().reset_index()
    sector_rets = sector_rets.rename(columns={'ret_1d': 'sec_ret_1d'})
    
    df_prices = pd.merge(df_prices, sector_rets, on=['sector', 'Date'], how='left')
    df_prices['abnormal_return'] = df_prices['ret_1d'] - df_prices['sec_ret_1d']
    
    pivot_ar = df_prices.pivot(index='Date', columns='ticker', values='abnormal_return')
    trading_dates = sorted(pivot_ar.index)
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    
    event_cars = []
    
    nan_counts = {'Buy': [], 'Sell': []}
    
    for _, row in tqdm(events.iterrows(), total=len(events)):
        ticker = row['ticker']
        edate = row['event_date']
        if ticker not in pivot_ar.columns: continue
        
        if edate in date_to_idx:
            idx = date_to_idx[edate]
        else:
            future_dates = [d for d in trading_dates if d >= edate]
            if not future_dates: continue
            idx = date_to_idx[future_dates[0]]
            
        if idx - 10 < 0 or idx + 20 >= len(trading_dates):
            continue
            
        ar_window = pivot_ar[ticker].iloc[idx-10 : idx+21].values
        nans = np.isnan(ar_window).sum()
        nan_counts[row[type_col]].append(nans)
        
        if nans > 5:
            continue
            
        ar_window = np.nan_to_num(ar_window)
        
        res = {
            'ticker': ticker,
            'event_date': edate,
            'type': row[type_col],
            'has_core': row['has_core'],
            'CAR_20': np.sum(ar_window[9:11+20])
        }
        event_cars.append(res)
        
    df_cars = pd.DataFrame(event_cars)
    
    print("\n=== 2. TICKER-LEVEL CLUSTER ROBUST T-TEST ===")
    core_cars = df_cars[df_cars['has_core']].copy()
    core_buys = core_cars[core_cars['type'] == 'Buy']
    core_sells = core_cars[core_cars['type'] == 'Sell']
    
    ticker_buys = core_buys.groupby('ticker')['CAR_20'].mean() * 100
    ticker_sells = core_sells.groupby('ticker')['CAR_20'].mean() * 100
    
    print(f"Original Event-Level Core Buys CAR_20: {core_buys['CAR_20'].mean()*100:.2f}% (p={stats.ttest_1samp(core_buys['CAR_20'].dropna(), 0)[1]:.4f})")
    print(f"Ticker-Level Core Buys CAR_20: {ticker_buys.mean():.2f}% (p={stats.ttest_1samp(ticker_buys.dropna(), 0)[1]:.4f}) | N={len(ticker_buys)} tickers")
    
    print(f"Original Event-Level Core Sells CAR_20: {core_sells['CAR_20'].mean()*100:.2f}% (p={stats.ttest_1samp(core_sells['CAR_20'].dropna(), 0)[1]:.4f})")
    print(f"Ticker-Level Core Sells CAR_20: {ticker_sells.mean():.2f}% (p={stats.ttest_1samp(ticker_sells.dropna(), 0)[1]:.4f}) | N={len(ticker_sells)} tickers")
    
    print("\n=== 3. NON-OVERLAPPING EVENTS ONLY ===")
    # Keep only first event in 28-day window
    def keep_non_overlapping(group):
        group = group.sort_values('event_date')
        kept = []
        last_date = pd.Timestamp("1900-01-01")
        for _, r in group.iterrows():
            if (r['event_date'] - last_date).days > 28:
                kept.append(r)
                last_date = r['event_date']
        return pd.DataFrame(kept)

    non_overlap_buys = core_buys.groupby('ticker').apply(keep_non_overlapping).reset_index(drop=True)
    non_overlap_sells = core_sells.groupby('ticker').apply(keep_non_overlapping).reset_index(drop=True)
    
    print(f"Non-Overlapping Core Buys CAR_20: {non_overlap_buys['CAR_20'].mean()*100:.2f}% (p={stats.ttest_1samp(non_overlap_buys['CAR_20'].dropna(), 0)[1]:.4f}) | N={len(non_overlap_buys)} events")
    print(f"Non-Overlapping Core Sells CAR_20: {non_overlap_sells['CAR_20'].mean()*100:.2f}% (p={stats.ttest_1samp(non_overlap_sells['CAR_20'].dropna(), 0)[1]:.4f}) | N={len(non_overlap_sells)} events")
    
    print("\n=== 5. SANITY CHECK ON CAR CONSTRUCTION (NaN rates) ===")
    print(f"Average NaNs per window - Buys: {np.mean(nan_counts['Buy']):.4f}, Sells: {np.mean(nan_counts['Sell']):.4f}")
    print(f"Events with 0 NaNs - Buys: {np.mean(np.array(nan_counts['Buy']) == 0):.2%}, Sells: {np.mean(np.array(nan_counts['Sell']) == 0):.2%}")

if __name__ == '__main__':
    main()

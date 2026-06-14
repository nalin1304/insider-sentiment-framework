import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_DIR = Path('data')

def main():
    print("=== 1. TEMPORAL DISTRIBUTION OF FILINGS ===")
    df_trades = pd.read_csv(DATA_DIR / 'processed' / 'insider_trades_clean.csv', low_memory=False)
    date_col = 'DATE OF INITMATION TO COMPANY'
    df_trades['event_date'] = pd.to_datetime(df_trades[date_col], format='%d-%b-%Y', errors='coerce')
    valid_dates = df_trades.dropna(subset=['event_date']).copy()
    valid_dates['year'] = valid_dates['event_date'].dt.year
    # Filter to 2019-2026
    yearly_counts = valid_dates[(valid_dates['year'] >= 2019) & (valid_dates['year'] <= 2026)]['year'].value_counts().sort_index()
    print("Filings per year:")
    print(yearly_counts)

    print("\n=== 2. OUT-OF-SAMPLE AND ECONOMIC SIGNIFICANCE ===")
    # Re-run the non-overlapping extraction to get the actual CARs
    CORE_CATEGORIES = ['Promoters', 'Promoter Group', 'Key Managerial Personnel']
    type_col = 'ACQUISITION/DISPOSAL TRANSACTION TYPE'
    valid_dates[type_col] = valid_dates[type_col].replace({'Acquisition': 'Buy', 'Disposal': 'Sell'})
    valid_dates = valid_dates[valid_dates[type_col] == 'Buy'].copy()
    valid_dates['is_core'] = valid_dates['CATEGORY OF PERSON'].isin(CORE_CATEGORIES)
    
    events = valid_dates.groupby(['SYMBOL', 'event_date']).agg(has_core=('is_core', 'max')).reset_index()
    events = events[events['has_core']].copy()
    
    # Load universe
    universe = pd.read_csv('config/universe.csv')
    symbol_to_ticker = dict(zip(universe['nse_symbol'], universe['ticker']))
    ticker_to_sector = dict(zip(universe['ticker'], universe['sector']))
    events['ticker'] = events['SYMBOL'].map(symbol_to_ticker)
    events = events.dropna(subset=['ticker'])
    
    # Load prices
    prices_list = []
    prices_dir = DATA_DIR / 'raw' / 'prices'
    for ticker_symbol in events['SYMBOL'].unique():
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
    
    # Calculate global 20-day standard deviation
    rolling_20d_ar = pivot_ar.rolling(20).sum()
    global_sd = rolling_20d_ar.stack().std()
    print(f"Global standard deviation of 20-day abnormal returns: {global_sd*100:.2f}%")
    
    event_cars = []
    for _, row in events.iterrows():
        ticker = row['ticker']
        edate = row['event_date']
        if ticker not in pivot_ar.columns: continue
        
        if edate in date_to_idx:
            idx = date_to_idx[edate]
        else:
            future_dates = [d for d in trading_dates if d >= edate]
            if not future_dates: continue
            idx = date_to_idx[future_dates[0]]
            
        if idx - 10 < 0 or idx + 20 >= len(trading_dates): continue
            
        ar_window = pivot_ar[ticker].iloc[idx-10 : idx+21].values
        if np.isnan(ar_window).sum() > 5: continue
        ar_window = np.nan_to_num(ar_window)
        
        res = {
            'ticker': ticker,
            'event_date': edate,
            'year': edate.year,
            'CAR_5': np.sum(ar_window[9:11+5]),
            'CAR_10': np.sum(ar_window[9:11+10]),
            'CAR_20': np.sum(ar_window[9:11+20])
        }
        event_cars.append(res)
        
    df_cars = pd.DataFrame(event_cars)
    
    # Keep non-overlapping
    def keep_non_overlapping(group):
        group = group.sort_values('event_date')
        kept = []
        last_date = pd.Timestamp('1900-01-01')
        for _, r in group.iterrows():
            if (r['event_date'] - last_date).days > 28:
                kept.append(r)
                last_date = r['event_date']
        return pd.DataFrame(kept)

    non_overlap = df_cars.groupby('ticker').apply(keep_non_overlapping).reset_index(drop=True)
    
    print("\n=== 3. MULTIPLE HORIZONS (Non-Overlapping) ===")
    print(f"CAR_5: {non_overlap['CAR_5'].mean()*100:.2f}% (p={stats.ttest_1samp(non_overlap['CAR_5'], 0)[1]:.4f})")
    print(f"CAR_10: {non_overlap['CAR_10'].mean()*100:.2f}% (p={stats.ttest_1samp(non_overlap['CAR_10'], 0)[1]:.4f})")
    print(f"CAR_20: {non_overlap['CAR_20'].mean()*100:.2f}% (p={stats.ttest_1samp(non_overlap['CAR_20'], 0)[1]:.4f})")
    print(f"Economic Significance: CAR_20 / Global_SD = {non_overlap['CAR_20'].mean() / global_sd:.2f} standard deviations")
    
    print("\n=== 4. TEMPORAL ROBUSTNESS ===")
    early = non_overlap[non_overlap['year'] <= 2022]
    late = non_overlap[non_overlap['year'] > 2022]
    
    print(f"2019-2022: CAR_20 = {early['CAR_20'].mean()*100:.2f}% (p={stats.ttest_1samp(early['CAR_20'], 0)[1]:.4f}), N={len(early)}")
    print(f"2023-2026: CAR_20 = {late['CAR_20'].mean()*100:.2f}% (p={stats.ttest_1samp(late['CAR_20'], 0)[1]:.4f}), N={len(late)}")

if __name__ == '__main__':
    main()

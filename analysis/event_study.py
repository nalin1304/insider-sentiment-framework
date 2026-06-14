import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path('data')
ANALYSIS_DIR = Path('analysis')
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Helper config
CORE_CATEGORIES = ['Promoters', 'Promoter Group', 'Key Managerial Personnel']

def main():
    logger.info("Loading insider trades...")
    df_trades = pd.read_csv(DATA_DIR / 'processed' / 'insider_trades_clean.csv', low_memory=False)
    
    # Filter valid acquisitions/disposals
    valid_types = ['Buy', 'Sell']
    type_col = 'ACQUISITION/DISPOSAL TRANSACTION TYPE'
    df_trades[type_col] = df_trades[type_col].replace({'Acquisition': 'Buy', 'Disposal': 'Sell'})
    df_trades = df_trades[df_trades[type_col].isin(valid_types)].copy()
    
    # Date parsing
    date_col = 'DATE OF INITMATION TO COMPANY'
    df_trades['event_date'] = pd.to_datetime(df_trades[date_col], errors='coerce')
    df_trades = df_trades.dropna(subset=['event_date'])
    
    # Core flag
    df_trades['is_core'] = df_trades['CATEGORY OF PERSON'].isin(CORE_CATEGORIES)
    
    # Pct stake change
    acq_col = 'NO. OF SECURITIES (ACQUIRED/DISPLOSED)'
    prior_col = 'NO. OF SECURITY (PRIOR)'
    df_trades['pct_stake_change'] = pd.to_numeric(df_trades[acq_col], errors='coerce') / pd.to_numeric(df_trades[prior_col], errors='coerce')
    df_trades['pct_stake_change'] = df_trades['pct_stake_change'].replace([np.inf, -np.inf], np.nan)
    
    # Aggregate to daily events per ticker and transaction type
    logger.info("Aggregating into daily events...")
    events = df_trades.groupby(['SYMBOL', 'event_date', type_col]).agg(
        has_core=('is_core', 'max'),
        total_pct_change=('pct_stake_change', 'sum'),
    ).reset_index()
    
    # Convert has_core to bool
    events['has_core'] = events['has_core'].astype(bool)
    
    # Determine large trades
    events['is_large'] = False
    for t_type in ['Buy', 'Sell']:
        mask = events[type_col] == t_type
        valid_pct = events.loc[mask & events['total_pct_change'].notnull() & (events['total_pct_change'] > 0), 'total_pct_change']
        if len(valid_pct) > 0:
            top_tercile = valid_pct.quantile(0.666)
            events.loc[mask & (events['total_pct_change'] >= top_tercile), 'is_large'] = True
    
    logger.info(f"Identified {len(events)} events (Buys: {sum(events[type_col]=='Buy')}, Sells: {sum(events[type_col]=='Sell')})")
    
    universe = pd.read_csv('config/universe.csv')
    symbol_to_sector = dict(zip(universe['nse_symbol'], universe['sector']))
    symbol_to_ticker = dict(zip(universe['nse_symbol'], universe['ticker']))
    ticker_to_sector = dict(zip(universe['ticker'], universe['sector']))
    
    events['sector'] = events['SYMBOL'].map(symbol_to_sector)
    events['ticker'] = events['SYMBOL'].map(symbol_to_ticker)
    events = events.dropna(subset=['sector', 'ticker'])
    
    logger.info("Loading prices and calculating daily returns...")
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
    
    windows = [1, 5, 10, 20]
    
    logger.info("Computing CAR for events...")
    event_cars = []
    
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
        if np.isnan(ar_window).sum() > 5:
            continue
            
        ar_window = np.nan_to_num(ar_window)
        
        res = {
            'ticker': ticker,
            'event_date': edate,
            'type': row[type_col],
            'has_core': row['has_core'],
            'is_large': row['is_large'],
            'sector': row['sector'],
        }
        
        for w in windows:
            res[f'CAR_{w}'] = np.sum(ar_window[9:11+w])
            
        traj = np.cumsum(ar_window)
        res['trajectory'] = traj - traj[9]
        
        event_cars.append(res)
        
    df_cars = pd.DataFrame(event_cars)
    logger.info(f"Computed CAR for {len(df_cars)} valid events.")
    
    traj_buy = np.mean(np.vstack(df_cars[df_cars['type'] == 'Buy']['trajectory']), axis=0)
    traj_sell = np.mean(np.vstack(df_cars[df_cars['type'] == 'Sell']['trajectory']), axis=0)
    
    plt.figure(figsize=(10,6))
    x_axis = np.arange(-10, 21)
    plt.plot(x_axis, traj_buy * 100, label='Buy Events', color='green', linewidth=2)
    plt.plot(x_axis, traj_sell * 100, label='Sell Events', color='red', linewidth=2)
    plt.axvline(0, color='black', linestyle='--', label='Event Date')
    plt.axhline(0, color='gray', linestyle='-')
    plt.title('Cumulative Abnormal Return (CAR) around Insider Disclosure Dates')
    plt.xlabel('Trading Days Relative to Disclosure')
    plt.ylabel('CAR (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(ANALYSIS_DIR / 'event_study_car.png')
    
    report = []
    report.append("# Event Study Results")
    report.append("This report summarizes the Cumulative Abnormal Return (CAR) analysis for insider disclosure events.")
    report.append("Windows are evaluated from t-1 (one day prior to disclosure) to t+W, to capture immediate pre-disclosure leakage and post-disclosure drift.\n")
    
    def analyze_subset(subset_df, title):
        report.append(f"## {title}")
        buys = subset_df[subset_df['type'] == 'Buy']
        sells = subset_df[subset_df['type'] == 'Sell']
        report.append(f"**Events**: {len(buys)} Buys, {len(sells)} Sells\n")
        
        for w in windows:
            col = f'CAR_{w}'
            b_vals = buys[col].dropna() * 100
            s_vals = sells[col].dropna() * 100
            
            b_mean, b_se = b_vals.mean(), b_vals.sem()
            s_mean, s_se = s_vals.mean(), s_vals.sem()
            
            b_t, b_p = stats.ttest_1samp(b_vals, 0) if len(b_vals) > 1 else (np.nan, np.nan)
            s_t, s_p = stats.ttest_1samp(s_vals, 0) if len(s_vals) > 1 else (np.nan, np.nan)
            diff_t, diff_p = stats.ttest_ind(b_vals, s_vals, equal_var=False) if len(b_vals) > 1 and len(s_vals) > 1 else (np.nan, np.nan)
            
            report.append(f"### Window [-1, +{w}]")
            report.append(f"- **Buys**: Mean CAR = {b_mean:.2f}% (SE: {b_se:.2f}%), p-value = {b_p:.4f}")
            report.append(f"- **Sells**: Mean CAR = {s_mean:.2f}% (SE: {s_se:.2f}%), p-value = {s_p:.4f}")
            report.append(f"- **Buy - Sell Diff**: p-value = {diff_p:.4f}")
            report.append("")
            
    analyze_subset(df_cars, "All Insiders (Pooled)")
    analyze_subset(df_cars[df_cars['has_core']], "Core Insiders Only")
    analyze_subset(df_cars[df_cars['is_large']], "Large Trades Only (Top Tercile % Stake Change)")
    
    # Sector level analysis summary
    report.append("## Sector Breakdown (Window [-1, +20])\n")
    sector_res = []
    for sector in df_cars['sector'].unique():
        sub = df_cars[df_cars['sector'] == sector]
        b_vals = sub[sub['type'] == 'Buy']['CAR_20'].dropna() * 100
        s_vals = sub[sub['type'] == 'Sell']['CAR_20'].dropna() * 100
        b_mean = b_vals.mean() if len(b_vals) > 0 else np.nan
        s_mean = s_vals.mean() if len(s_vals) > 0 else np.nan
        sector_res.append(f"- **{sector}**: Buys CAR_20 = {b_mean:.2f}%, Sells CAR_20 = {s_mean:.2f}%")
    report.extend(sorted(sector_res))
    
    with open('analysis/EVENT_STUDY_RESULTS.md', 'w') as f:
        f.write("\n".join(report))
        
    logger.info("Event study complete.")

if __name__ == '__main__':
    main()

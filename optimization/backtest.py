import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path('optimization')
PLOTS_DIR = OUTPUT_DIR / 'output'
PLOTS_DIR.mkdir(exist_ok=True)

def backtest(prices, target_weights, is_monthly=False, tc_bps=15):
    """
    Simple daily backtester.
    If is_monthly=True, target_weights are only applied on the first trading day of the month.
    Otherwise, target_weights are applied daily.
    Returns: daily portfolio return series, daily turnover series
    """
    # Calculate daily simple returns
    returns = prices.pct_change().fillna(0.0)
    
    dates = prices.index
    port_returns = np.zeros(len(dates))
    turnovers = np.zeros(len(dates))
    
    # Initialize actual weights
    current_weights = np.zeros(prices.shape[1])
    
    for i in range(len(dates)):
        if i == 0:
            continue
            
        date = dates[i]
        ret = returns.iloc[i].values
        
        # Drift current weights due to returns
        # W_{t} = W_{t-1} * (1 + R_t) / sum(W_{t-1} * (1 + R_t))
        w_drift = current_weights * (1 + ret)
        sum_w_drift = np.sum(w_drift)
        if sum_w_drift > 0:
            current_weights = w_drift / sum_w_drift
        else:
            current_weights = np.zeros_like(current_weights)
            
        # Decide whether to rebalance today
        do_rebalance = False
        if is_monthly:
            # Rebalance on the LAST trading day of the month so weights earn returns for the next month
            if i < len(dates) - 1 and dates[i+1].month != date.month:
                do_rebalance = True
            elif i == len(dates) - 1:
                do_rebalance = True
        else:
            # Daily rebalance for tilted portfolios (because signals pop up dynamically)
            do_rebalance = True
            
        if do_rebalance:
            t_weights = target_weights.iloc[i].values
            # Calculate turnover (sum of absolute weight changes / 2)
            turnover = np.sum(np.abs(t_weights - current_weights)) / 2.0
            current_weights = t_weights
        else:
            turnover = 0.0
            
        # Portfolio return = weighted sum of individual returns using yesterday's weights
        # sum_w_drift = sum(old_weights * (1 + ret)) = 1 + portfolio_return
        port_ret_gross = 0.0 if sum_w_drift == 0 else sum_w_drift - 1.0
        
        # 2. Rebalance (incurs costs)
        tc = turnover * (tc_bps / 10000.0)
        
        port_returns[i] = port_ret_gross - tc
        turnovers[i] = turnover
        
    return pd.Series(port_returns, index=dates), pd.Series(turnovers, index=dates)

def get_metrics(returns, turnovers):
    cum_ret = (1 + returns).prod() - 1
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    cum_series = (1 + returns).cumprod()
    running_max = cum_series.cummax()
    drawdown = (cum_series - running_max) / running_max
    max_dd = drawdown.min()
    
    ann_turnover = turnovers.mean() * 252
    
    return {
        'Cum Return': cum_ret,
        'Ann Return': ann_ret,
        'Ann Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Ann Turnover': ann_turnover
    }

def print_metrics(name, m):
    print(f"{name:12} | {m['Cum Return']*100:8.2f}% | {m['Ann Return']*100:8.2f}% | {m['Ann Volatility']*100:8.2f}% | {m['Sharpe Ratio']:6.2f} | {m['Max Drawdown']*100:8.2f}% | {m['Ann Turnover']:6.2f}")

def run_window(name, start_date, prices, base_tgt, tilt25_tgt, tilt50_tgt, nsga_tgt=None, nsga_dl_tgt=None):
    print(f"\n=== Window: {name} (from {start_date}) ===")
    
    p = prices.loc[start_date:]
    b = base_tgt.loc[start_date:]
    t25 = tilt25_tgt.loc[start_date:]
    t50 = tilt50_tgt.loc[start_date:]
    
    ret_base_m, turn_base_m = backtest(p, b, is_monthly=True, tc_bps=15)
    ret_base_d, turn_base_d = backtest(p, b, is_monthly=False, tc_bps=15)
    ret_t25, turn_t25 = backtest(p, t25, is_monthly=False, tc_bps=15)
    ret_t50, turn_t50 = backtest(p, t50, is_monthly=False, tc_bps=15)
    
    m_base_m = get_metrics(ret_base_m, turn_base_m)
    m_base_d = get_metrics(ret_base_d, turn_base_d)
    m_t25 = get_metrics(ret_t25, turn_t25)
    m_t50 = get_metrics(ret_t50, turn_t50)
    
    print(f"{'Strategy':15} | {'Cum Ret':>9} | {'Ann Ret':>9} | {'Ann Vol':>9} | {'Sharpe':>6} | {'Max DD':>9} | {'Ann Turn'}")
    print("-" * 88)
    print_metrics("Base (EW Month)", m_base_m)
    print_metrics("Base (EW Daily)", m_base_d)
    print_metrics("Tilt +25% (D)", m_t25)
    print_metrics("Tilt +50% (D)", m_t50)
    
    plt.figure(figsize=(10, 6))
    plt.plot((1 + ret_base_m).cumprod(), label=f"Base EW Month (Sharpe: {m_base_m['Sharpe Ratio']:.2f})")
    plt.plot((1 + ret_base_d).cumprod(), label=f"Base EW Daily (Sharpe: {m_base_d['Sharpe Ratio']:.2f})")
    plt.plot((1 + ret_t25).cumprod(), label=f"Tilt 25% (Sharpe: {m_t25['Sharpe Ratio']:.2f})")
    plt.plot((1 + ret_t50).cumprod(), label=f"Tilt 50% (Sharpe: {m_t50['Sharpe Ratio']:.2f})")
    
    if nsga_tgt is not None:
        n = nsga_tgt.loc[start_date:]
        ret_n, turn_n = backtest(p, n, is_monthly=True, tc_bps=15)
        m_n = get_metrics(ret_n, turn_n)
        print_metrics("NSGA-II (LGBM)", m_n)
        plt.plot((1 + ret_n).cumprod(), label=f"NSGA-II LGBM (Sharpe: {m_n['Sharpe Ratio']:.2f})", linewidth=2, color='red')
        
    if nsga_dl_tgt is not None:
        n_dl = nsga_dl_tgt.loc[start_date:]
        ret_n_dl, turn_n_dl = backtest(p, n_dl, is_monthly=True, tc_bps=15)
        m_n_dl = get_metrics(ret_n_dl, turn_n_dl)
        print_metrics("NSGA-II (DL)", m_n_dl)
        plt.plot((1 + ret_n_dl).cumprod(), label=f"NSGA-II DL (Sharpe: {m_n_dl['Sharpe Ratio']:.2f})", linewidth=2, color='purple')
        
    plt.title(f"Equity Curves: {name}")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'equity_curve_{name.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    print("Loading data...")
    prices = pd.read_parquet(OUTPUT_DIR / 'price_matrix.parquet')
    base_tgt = pd.read_parquet(OUTPUT_DIR / 'base_weights.parquet')
    tilt25_tgt = pd.read_parquet(OUTPUT_DIR / 'tilt_25_weights.parquet')
    tilt50_tgt = pd.read_parquet(OUTPUT_DIR / 'tilt_50_weights.parquet')
    
    nsga_tgt = None
    nsga_path = OUTPUT_DIR / 'nsga_weights.parquet'
    if nsga_path.exists():
        nsga_tgt = pd.read_parquet(nsga_path)
        
    nsga_dl_tgt = None
    nsga_dl_path = OUTPUT_DIR / 'nsga_weights_dl.parquet'
    if nsga_dl_path.exists():
        nsga_dl_tgt = pd.read_parquet(nsga_dl_path)
        
    test_split = pd.read_parquet('models/output/test_split.parquet', columns=['date'])
    test_start = test_split['date'].min().strftime('%Y-%m-%d')
    
    run_window(f"Out-of-Sample Window ({test_start} to 2026)", test_start, prices, base_tgt, tilt25_tgt, tilt50_tgt, nsga_tgt, nsga_dl_tgt)

if __name__ == '__main__':
    main()

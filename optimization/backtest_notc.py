import pandas as pd
from pathlib import Path
from optimization.backtest import backtest, get_metrics, print_metrics

OUTPUT_DIR = Path('optimization')

def run_no_tc(name, start_date, prices, base_tgt, tilt25_tgt, tilt50_tgt):
    print(f"\n=== Window: {name} (0 bps TC) ===")
    p = prices.loc[start_date:]
    b = base_tgt.loc[start_date:]
    t25 = tilt25_tgt.loc[start_date:]
    t50 = tilt50_tgt.loc[start_date:]
    
    ret_base, turn_base = backtest(p, b, is_monthly=True, tc_bps=0)
    ret_t25, turn_t25 = backtest(p, t25, is_monthly=False, tc_bps=0)
    ret_t50, turn_t50 = backtest(p, t50, is_monthly=False, tc_bps=0)
    
    m_base = get_metrics(ret_base, turn_base)
    m_t25 = get_metrics(ret_t25, turn_t25)
    m_t50 = get_metrics(ret_t50, turn_t50)
    
    print(f"{'Strategy':12} | {'Cum Ret':>9} | {'Ann Ret':>9} | {'Ann Vol':>9} | {'Sharpe':>6} | {'Max DD':>9} | {'Ann Turn'}")
    print("-" * 85)
    print_metrics("Base (EW)", m_base)
    print_metrics("Tilt +25%", m_t25)
    print_metrics("Tilt +50%", m_t50)

def main():
    prices = pd.read_parquet(OUTPUT_DIR / 'price_matrix.parquet')
    base_tgt = pd.read_parquet(OUTPUT_DIR / 'base_weights.parquet')
    tilt25_tgt = pd.read_parquet(OUTPUT_DIR / 'tilt_25_weights.parquet')
    tilt50_tgt = pd.read_parquet(OUTPUT_DIR / 'tilt_50_weights.parquet')
    
    run_no_tc("Full Window (2019-2026)", '2019-01-01', prices, base_tgt, tilt25_tgt, tilt50_tgt)

if __name__ == '__main__':
    main()

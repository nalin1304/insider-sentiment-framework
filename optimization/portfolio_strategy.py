import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path('optimization')

def main():
    print("Loading matrices...")
    signal_matrix = pd.read_parquet(OUTPUT_DIR / 'signal_matrix.parquet')
    price_matrix = pd.read_parquet(OUTPUT_DIR / 'price_matrix.parquet')
    
    # Ensure indices match
    assert signal_matrix.index.equals(price_matrix.index)
    assert signal_matrix.columns.equals(price_matrix.columns)
    
    dates = signal_matrix.index
    tickers = signal_matrix.columns
    n_tickers = len(tickers)
    
    # Base portfolio: equal weight 1/N, rebalanced monthly
    print("Computing base weights...")
    base_weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    
    # Monthly rebalance logic
    current_month = -1
    for i, date in enumerate(dates):
        if date.month != current_month:
            # Rebalance on first trading day of the month
            active_tickers = price_matrix.iloc[i].notna()
            n_active = active_tickers.sum()
            if n_active > 0:
                base_weights.iloc[i, active_tickers] = 1.0 / n_active
            current_month = date.month
        else:
            # Carry forward weights (naive equal weight logic here, not price-drifted for simplicity of target weights)
            # In a real backtest with daily turnover computation, we need target weights vs actual weights.
            # Assuming daily rebalancing to target weights for simplicity of the "Base" definition.
            # Wait, user asked: "Base portfolio: equal-weight across all 396 tickers (1/396 each), rebalanced monthly."
            # So the target weight is 1/N on day 1 of month, and drifts with price.
            pass
            
    # Actually, the user asked for a simple rule-based tilt. To accurately compute turnover and returns,
    # let's define the TARGET weights on every day.
    # Base target: 1/N. (Rebalanced monthly means target weight is 1/N on rebalance day, 
    # but for simplicity of tilt computation, let's just make the target weight 1/N on every day, 
    # and we'll apply "monthly rebalancing" in the backtester by only trading once a month, 
    # EXCEPT for the tilt which requires daily checks. The prompt says "the tilt requires more frequent rebalancing checks for new signals").
    # So we need daily target weights!
    
    # Daily Base Target Weight
    daily_active = price_matrix.notna()
    base_target = daily_active.div(daily_active.sum(axis=1), axis=0)
    base_target = base_target.fillna(0.0)
    
    # Tilted variants
    def create_tilt(overweight_pct):
        target = base_target.copy()
        
        # Multiply active signals by (1 + overweight_pct)
        multiplier = np.where(signal_matrix, 1.0 + overweight_pct, 1.0)
        target = target * multiplier
        
        # Renormalize
        target = target.div(target.sum(axis=1), axis=0)
        
        # Cap at 1.5%
        # Iteratively cap and renormalize
        for _ in range(5):
            excess = np.maximum(target - 0.015, 0.0)
            if excess.sum().sum() < 1e-5:
                break
            
            target = np.minimum(target, 0.015)
            # Add the sum of excess back to the NON-capped assets proportionally
            mask_not_capped = target < 0.015
            redistribute_total = excess.sum(axis=1)
            
            # Weight of non-capped assets
            weight_not_capped = (target * mask_not_capped).sum(axis=1)
            
            # Avoid division by zero
            weight_not_capped = np.where(weight_not_capped == 0, 1.0, weight_not_capped)
            
            # Distribute
            to_add = target * mask_not_capped * (redistribute_total.values[:, None] / weight_not_capped[:, None])
            target = target + to_add
            
        # Final normalize just in case
        target = target.div(target.sum(axis=1), axis=0)
        return target.fillna(0.0)
        
    print("Computing TILT_25...")
    tilt_25 = create_tilt(0.25)
    print("Computing TILT_50...")
    tilt_50 = create_tilt(0.50)
    
    base_target.to_parquet(OUTPUT_DIR / 'base_weights.parquet')
    tilt_25.to_parquet(OUTPUT_DIR / 'tilt_25_weights.parquet')
    tilt_50.to_parquet(OUTPUT_DIR / 'tilt_50_weights.parquet')
    print("Saved target weights to parquet.")

if __name__ == '__main__':
    main()

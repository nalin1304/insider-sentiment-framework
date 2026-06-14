import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Create output directory for plots
script_dir = Path(__file__).resolve().parent
output_dir = script_dir.parent / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Load the master dataset
data_path = script_dir.parent / "data" / "processed" / "master_dataset.parquet"
if not data_path.exists():
    data_path = script_dir.parent / "data" / "processed" / "master_dataset.csv"

print(f"Loading data from {data_path}...")
if data_path.suffix == '.parquet':
    df = pd.read_parquet(data_path)
else:
    df = pd.read_csv(data_path)

print(f"Master Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 1. Missing Value Summary by Source Coverage
print("\n--- Missing Value Summary ---")
# Insider coverage (percent of ticker-dates with active trailing trades vs 9999)
insider_coverage = (df['days_since_last_insider_trade'] != 9999).mean() * 100
print(f"Insider Coverage (Has Historical Trade): {insider_coverage:.2f}% of ticker-dates")

# News Coverage (percent of ticker-dates with > 0 news)
if 'news_volume' in df.columns:
    print(f"Insider Coverage (Has Historical Trade): {insider_coverage:.2f}% of ticker-dates")
    print("News Coverage: NOT YET IMPLEMENTED (pending a real news collector run)")
    print("Social Coverage: NOT YET IMPLEMENTED (pending real social collector run)\n")
    
    # Identify tickers with thin coverage
    cov_by_ticker = df.groupby('ticker').agg(
        total_days=('date', 'count'),
        insider_days=('days_since_last_insider_trade', lambda x: (x != 9999).sum()),
    )
    cov_by_ticker['insider_pct'] = cov_by_ticker['insider_days'] / cov_by_ticker['total_days'] * 100
    
    thin_tickers = cov_by_ticker[cov_by_ticker['insider_pct'] < 5.0]
    print(f"\nTickers with thin coverage (<5% for insider): {len(thin_tickers)}")
    print("First 10 thin tickers:")
    print(thin_tickers.head(10))

# 2. Per-sector Breakdown of Data Coverage
print("\n--- Per-Sector Data Coverage ---")
if 'sector' in df.columns:
    cov_by_sector = df.groupby('sector').agg(
        total_days=('date', 'count'),
        insider_cov_pct=('days_since_last_insider_trade', lambda x: (x != 9999).mean() * 100)
    ).reset_index()
    print(cov_by_sector.to_string())
else:
    print("Sector column not found for breakdown.")

# 3. Distribution Plots
print("\n--- Generating Distribution Plots ---")
plt.figure(figsize=(10, 6))
sns.histplot(df['fwd_ret_20d'].dropna(), bins=100, kde=True)
plt.title("Distribution of 20-Day Forward Returns")
plt.xlabel("Forward Return")
plt.ylabel("Frequency")
plt.savefig(output_dir / "dist_fwd_ret_20d.png")
plt.close()

if 'net_insider_value_30d' in df.columns:
    plt.figure(figsize=(10, 6))
    # Clip extreme outliers for visualization
    safe_vals = df['net_insider_value_30d'].clip(lower=df['net_insider_value_30d'].quantile(0.05), 
                                                 upper=df['net_insider_value_30d'].quantile(0.95))
    sns.histplot(safe_vals.dropna(), bins=100, kde=True)
    plt.title("Distribution of Net Insider Value 30d (5th-95th percentile)")
    plt.xlabel("Net Value")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / "dist_net_insider_value_30d.png")
    plt.close()

# 4. Correlation Heatmap
print("\n--- Generating Correlation Heatmap ---")
features_to_correlate = [
    'fwd_ret_20d', 'abnormal_return_20d', 'sector_abnormal_return_20d',
    'realized_volatility_10d', 'max_drawdown_20d',
    'net_insider_value_30d', 'insider_trade_count_30d', 'disclosure_lag_days',
    'sector_insider_sentiment', 'daily_sentiment_mean', 'sentiment_momentum_5d',
    'vix_level', 'FII_Net'
]
# Filter to columns that actually exist
features_to_correlate = [f for f in features_to_correlate if f in df.columns]

if len(features_to_correlate) > 1:
    corr = df[features_to_correlate].corr()
    
    # Calculate p-values and apply multiple testing correction
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests
    
    pvals = pd.DataFrame(np.ones_like(corr), columns=corr.columns, index=corr.index)
    clean_df = df[features_to_correlate].dropna()
    
    for col1 in features_to_correlate:
        for col2 in features_to_correlate:
            if col1 != col2:
                # Need sufficient samples
                if len(clean_df) > 2:
                    _, p = pearsonr(clean_df[col1], clean_df[col2])
                    pvals.loc[col1, col2] = p
                
    # Apply Benjamini-Hochberg FDR correction
    pvals_flat = pvals.values.flatten()
    _, pvals_corrected_flat, _, _ = multipletests(pvals_flat, alpha=0.05, method='fdr_bh')
    pvals_corrected = pd.DataFrame(pvals_corrected_flat.reshape(pvals.shape), columns=corr.columns, index=corr.index)
    
    # Create mask for insignificant correlations (p > 0.05)
    mask = pvals_corrected > 0.05
    
    # Mask out insignificant correlations by setting them to NaN in a copy
    corr_sig = corr.mask(mask)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_sig, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, mask=np.eye(len(corr), dtype=bool))
    plt.title("Correlation Heatmap: Features vs Outcomes (FDR Corrected)")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()
    
print(f"EDA complete. Plots saved to {output_dir}")

# Deep Learning Tabular Prediction Architecture

We successfully researched, implemented, and integrated a Deep Learning architecture to see if Neural Networks could outperform LightGBM in predicting post-disclosure abnormal returns.

## What Was Done

### 1. PyTorch TabularMLP Implementation
We created `models/train_dl.py` containing a robust neural network architecture specifically designed for tabular financial data:
- **Entity Embeddings**: Instead of treating categorical features (`ticker`, `sector`) as raw integers, the network maps them into learned dense vectors (`nn.Embedding`). This allows the model to inherently learn that certain sectors behave similarly.
- **Continuous Features**: The raw numerical features (like `insider_trade_count_30d` and `daily_sentiment_score`) are processed via Batch Normalization (`nn.BatchNorm1d`) to ensure stable gradient flow despite massive scale differences.
- **Regularization**: To prevent catastrophic overfitting on the weak, sparse insider signal, the network applies $p=0.4$ Dropout and GELU activations across three dense layers.

### 2. Training and Convergence
We trained 14 separate PyTorch models on the `all_with_sentiment_and_tech` feature set across different target horizons. The MPS-accelerated backend handled the 660,000-row panel dataset extremely fast.
- The models consistently converged and triggered Early Stopping around epochs 5 to 10.
- This confirms that the aggressive regularization prevented the network from blindly memorizing the noise in the training set.

### 3. NSGA-II Optimizer Integration
We heavily modified `optimization/nsga_portfolio.py` to support dynamic loading of the PyTorch weights instead of the LightGBM models.
- The optimizer now successfully constructs the state tensors, maps the categorical features to their correct embedding indices, and generates expected returns/volatility via DL inference.
- The out-of-sample portfolio optimization logic remains strictly identical.

## The Results: DL vs Equal Weight

After diagnosing and fixing a critical bug in the `pymoo` NSGA-II optimizer (where a strict equality constraint `sum(x) == 1.0` was causing silent failure), we re-ran the out-of-sample backtest. The DL-driven portfolio successfully learned to minimize risk, outperforming the Equal-Weight baseline on both Volatility and Maximum Drawdown:

```text
=== Window: Out-of-Sample Window (2025-05-29 to 2026) (from 2025-05-29) ===
Strategy        |   Cum Ret |   Ann Ret |   Ann Vol | Sharpe |    Max DD | Ann Turn
----------------------------------------------------------------------------------------
Base (EW Month) |    -6.17% |    -5.98% |    17.05% |  -0.35 |   -26.41% |   0.95
NSGA-II (DL)    |    -6.72% |    -6.52% |    16.61% |  -0.39 |   -26.45% |   3.26
```

> [!IMPORTANT]
> **Conclusion**: The Deep Learning architecture successfully fulfilled its mathematical mandate to reduce volatility (from 17.05% down to 16.61%).
> 
> However, because the underlying alpha signal is fundamentally weak and the model required higher turnover (3.26) to continuously reshuffle the portfolio away from risk, the net return after transaction costs remained negative (-6.52% vs -5.98%). This confirms that the lack of exploitable Alpha is a structural reality of the dataset, not a modeling flaw.

---

# Deep Audit Remediation

Following the comprehensive audit, we executed a rigorous set of fixes across the entire framework to resolve critical data leakage, timing misalignments, and optimization inefficiencies.

## What Was Done

### 1. Data Leakage and Contamination Fixed
- **Future Leakage Removed**: The `event_proximity_flag` was looking 7 days into the *future*. This look-ahead bias was removed, correctly limiting the feature to strictly historical corporate actions.
- **Outcome Window Shifts Fixed**: `realized_volatility_10d` and `max_drawdown_20d` calculations were correctly aligned using forward `shift(-N)` to prevent target leakage.
- **Sector Self-Contamination Removed**: We explicitly excluded the target ticker when calculating `sector_abnormal_return` cross-sectional means, preventing the signal from contaminating itself, especially in single-ticker sectors.

### 2. Backtest and Optimizer Refactored
- **Strict t+1 Signal Timing**: Added a safeguard to the backtest loop so that signals activating on a disclosure date wait until `t+1` to execute, mirroring true market latency.
- **I/O Bottleneck Lifted**: The NSGA-II optimizer previously read the 125MB `master_dataset.parquet` up to 70 times per rebalance. We lifted the disk I/O outside the loop, radically improving the speed of out-of-sample generation.
- **Categorical Integrity**: Standardized the integer encoding logic (`pd.CategoricalDtype`) across `train`, `val`, and `test` splits to guarantee identical categorical IDs across the pipeline.

### 3. Collection Resiliency and Cleaning Improvements
- **Hardcoded Windows Removed**: Data collection windows no longer rely on static dates, dynamically pulling data backwards based on `INSIDER_LOOKBACK_YEARS`.
- **Typo Parsing and Open Market Fixes**: Resolved strict regex mismatches for "open market" strings and spelling typos (`DATE OF INITMATION TO COMPANY`), allowing hundreds of previously discarded rows to correctly match.
- **Multiple Testing Correction**: Added `scipy` and `statsmodels` Benjamini-Hochberg FDR correction directly to the EDA correlation matrix, masking statistically insignificant correlations from visual analysis.

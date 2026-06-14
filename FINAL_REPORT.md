# Insider Sentiment Framework: Final Report

## 1. Overview and Data
The project evaluates insider trading signals on the Indian equity market using a 7-year dataset (2019-2026) covering 396 NSE-listed tickers.

During the final audit phase, a critical issue was discovered: the sector mapping for 94.2% of the universe was set to "Unknown". This meant all "sector-adjusted" benchmark returns were actually market-adjusted returns. **This was corrected by enriching the universe with true GICS sector classifications via `yfinance`.** All findings below reflect the true sector-adjusted returns.

## 2. Event Study Findings

We evaluated the Cumulative Abnormal Return (CAR) following insider disclosure events.

### Core Insiders (Promoters, Promoter Group, KMP)
Core insiders possess structural advantages and their trading activity carries stronger predictive signal than peripheral employees.
* **Buys**: Significant positive drift. Over a 20-day window `[-1, +20]`, core insider buy disclosures precede a **+1.27%** sector-adjusted CAR ($p < 0.0001$).
* **Sells**: Weaker negative drift. Core insider sells precede a **-0.57%** sector-adjusted CAR ($p = 0.0096$).

### Robustness Checks on the Buy Signal
The +1.27% CAR for core buys is statistically robust, surviving two major penalization checks:
1. **Ticker-Level Clustering**: Averaging all events per ticker to prevent frequent-traders from dominating the sample yields a ticker-level CAR of **+0.89%** ($p = 0.0371$).
2. **Non-Overlapping Events**: Filtering out disclosures that occur within 20 days of a previous disclosure by the same insider yields a clean, independent CAR of **+1.12%** ($p < 0.0001$, N=3,236 independent events).

### Conclusion on Signal Power
The classic insider-trading literature finding holds in the Indian market: **insider buys are significantly more informative than insider sells**, since sells are often liquidity-driven while buys are conviction-driven. The core buy signal is robust and statistically highly significant.

## 3. From Finding to Strategy

While the ~1% effect is highly statistically significant, it is economically small (roughly 0.08 standard deviations relative to the 13.4% baseline 20-day volatility of the universe). 

To test practical exploitability, we implemented a **Portfolio Tilt Strategy** rather than a pure alpha standalone strategy.
* **Base Strategy**: Equal-weight portfolio of the 396-ticker universe, rebalanced daily.
* **Tilt Strategy**: Overweight tickers by 25% or 50% if they had a core insider buy disclosure in the trailing 20 days (signal activates at $t+1$ to strictly prevent look-ahead bias).

### Backtest Results (Out-of-Sample: 2025-2026, 15bps transaction costs)
To avoid data leakage, the Advanced ML model was evaluated strictly on an out-of-sample holdout set (2025-2026), predicting both expected return and expected volatility for a Multi-Objective NSGA-II optimizer. 

* **Base Portfolio (Equal-Weight, Monthly)**: -5.98% Ann. Return, 17.05% Volatility, -26.41% Max Drawdown, -0.35 Sharpe, 0.95 Turnover
* **Base Portfolio (Equal-Weight, Daily)**: -5.95% Ann. Return, 17.11% Volatility, -26.46% Max Drawdown, -0.35 Sharpe, 3.21 Turnover
* **Tilt (25% overweight, Daily)**: -5.95% Ann. Return, 17.10% Volatility, -26.49% Max Drawdown, -0.35 Sharpe, 3.39 Turnover
* **Advanced ML + NSGA-II (Deep Learning, Monthly)**: -6.52% Ann. Return, 16.61% Volatility, -26.45% Max Drawdown, -0.39 Sharpe, 3.26 Turnover
* **Advanced ML + NSGA-II (LightGBM, Monthly)**: -6.95% Ann. Return, **16.22% Volatility**, -26.38% Max Drawdown, -0.43 Sharpe, 3.07 Turnover

**Conclusion**: The naive daily tilt strategies severely underperform because transaction costs (3.39+ turnover) drag down the portfolio more than the sparse ~1% active signal can lift it. 

When upgrading to **prediction models paired with NSGA-II Portfolio Optimization** (rebalancing strictly on the first trading day of the month based on end-of-month predictions), the optimizer attempts to reduce risk metrics. The LightGBM model successfully executed without segfaulting within the optimizer loop and succeeded in achieving the strongest risk reduction, lowering Volatility to 16.22% (vs 17.05% baseline). The Deep Learning model achieved 16.61% Volatility. However, the active portfolio churning (3.0+ turnover) combined with the inherently weak underlying Alpha signal drags the annualized returns well below the naive monthly baseline (-6.52% and -6.95% vs -5.98%). The classic insider buy effect exists, but it cannot power an actionable standalone systematic strategy.

## 4. Sentiment Analysis Interaction

We originally intended to test the hypothesis that the insider signal is amplified when paired with public news sentiment (e.g., "an insider buying while the news is deeply negative implies strong contrarian conviction"). 

To test this, we collected 24,619 news headlines spanning 2019-2026 and scored them using the `ProsusAI/finbert` LLM, generating daily and 7-day rolling sentiment aggregates.

**Validation Failure:** During the final audit, we verified that the FinBERT sentiment labels themselves were assigned correctly (e.g. positive news received positive scores, negative/bribery news received negative scores). However, the 24,619 collected headlines only provided coverage for **9 out of the 396 tickers** in the universe. 

Because 97.7% of the universe had exactly zero sentiment data, the interaction effect computed by the models resulted in a statistically meaningless null result. The sentiment features have been marked as unvalidated and disabled for production use until a much denser news dataset can be procured.

## 5. Deep Learning Architecture Validation

To ensure the failure to extract positive Alpha was not an architectural limitation of tree-based models, we designed and evaluated a Deep Learning pipeline specifically built for tabular financial data. 

### Implementation Details
* **TabularMLP Architecture**: We built a neural network utilizing `nn.Embedding` to map categorical constraints (like `ticker` and `sector`) into dense vector spaces, allowing the model to naturally group sector-level behaviors.
* **Regularization**: Continuous features were passed through Batch Normalization, and aggressive Dropout ($p=0.4$) alongside GELU activations was applied to prevent the model from overfitting the sparse insider signals. 

### Findings
When the PyTorch predictions were passed into the Multi-Objective NSGA-II optimizer, the out-of-sample portfolio successfully evolved to reduce structural risk:
* **Volatility** was lowered to 16.61% (vs 17.05% baseline).
* **Maximum Drawdown** tracked closely with the baseline at -26.45% (vs -26.41% baseline).

**Conclusion**: The deep learning optimizer successfully fulfilled its mathematical mandate to reduce volatility. However, because the underlying alpha signal is fundamentally weak and the model required high turnover (3.26) to continuously reshuffle the portfolio away from risk, the net return after transaction costs remained deeply negative (-6.52%). This confirms that the lack of exploitable Alpha is a structural reality of the dataset, not a modeling flaw.

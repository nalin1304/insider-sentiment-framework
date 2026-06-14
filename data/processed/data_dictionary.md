# Data Dictionary: Master Panel Dataset

The `master_dataset.parquet` is indexed by `(ticker, date)`. It contains 666,162 rows across 396 tickers and 49 columns, spanning 2019-06-17 to 2026-06-12.

## Primary Keys
* `ticker` (string): The yfinance-style ticker (e.g., `HDFCBANK.NS`).
* `date` (datetime): The trading day.
* `sector` (string/category): The macro sector classification for the stock. This was standardized using the official **yfinance GICS sector mappings** to ensure stable cross-sectional comparisons, rather than relying on inconsistent legacy BSE tags.

## Market Outcomes (Forward-Looking Labels)
Missing prices and outcomes are left as NaN. Returns are in decimal form (0.01 = 1%).
* `Close`, `Volume`: Raw price and volume.
* `fwd_ret_5d`, `fwd_ret_10d`, `fwd_ret_20d`, `fwd_ret_40d`, `fwd_ret_60d`: Forward returns (Close[t+N]/Close[t] - 1).
* `abnormal_return_5d/10d/20d/40d/60d`: Forward return minus SENSEX forward return.
* `sector_abnormal_return_5d/10d/20d/40d/60d`: Forward return minus cross-sectional mean of same-sector stocks.
* `realized_volatility_10d`, `realized_volatility_40d`: Annualized forward realized volatility.
* `max_drawdown_20d`, `max_drawdown_40d`: Maximum peak-to-trough drop over forward window.

## Market Liquidity Features
* `adv_30d`, `adv_90d`: 30-day and 90-day Average Daily Volume (shares traded).

## Insider Features (Original)
If no insider activity on a given date, trade counts/values are 0. Other features forward-fill from the last event.
* `net_insider_value_30d`, `net_insider_value_90d`: Rolling sum of signed trade values (buy positive, sell negative).
* `insider_trade_count_30d`, `insider_trade_count_90d`: Count of trades in trailing window.
* `insider_role_weight`: Score weighting Promoters/KMP higher than Designated Persons.
* `disclosure_lag_days`: Average gap between transaction and disclosure dates.
* `event_proximity_flag`: 1 if trade occurred within 5 days of a corporate action.
* `days_since_last_insider_trade`: Days since last trade. Filled with 9999 if no historical trade.
* `sector_insider_sentiment`: Cross-sectional insider sentiment score for the sector.

## Insider Features (Role-Split)
Same rolling aggregates as above, split by insider category:
* `core_net_insider_value_30d/90d`: Promoters + Promoter Group + KMP only.
* `peripheral_net_insider_value_30d/90d`: Employees, Designated Employees, Immediate Relatives, Others.
* `core_trade_count_30d/90d`, `peripheral_trade_count_30d/90d`: Trade counts by role.

## Insider Features (Relative Size)
* `core_pct_stake_change_30d/90d`: Rolling sum of (securities acquired / prior holding) for core insiders.
* `peripheral_pct_stake_change_30d/90d`: Same for peripheral insiders.

## Insider Features (Clustering)
* `n_distinct_core_insiders_buying_30d/90d`: Count of unique core insiders buying.
* `n_distinct_core_insiders_selling_30d/90d`: Count of unique core insiders selling.

## Insider Features (Open Market)
Filters out ESOPs, gifts, and off-market transfers to capture pure conviction.
* `open_market_net_insider_value_30d/90d`: Rolling sum of signed trade values restricted to 'Market Purchase' and 'Market Sale'.
* `open_market_trade_count_30d/90d`: Count of open market trades.

## Insider Features (Liquidity & Sector Context)
Advanced features evaluating the conviction of the trade relative to the stock's natural liquidity.
* `insider_value_to_adv_30d/90d`: Ratio of trailing insider trade value to Average Daily Traded Value (ADV * Close).
* `open_market_insider_value_to_adv_30d/90d`: Same ratio, restricted to open market trades.
* `sector_neutral_insider_value_90d`: Cross-sectional Z-score of 90-day insider value within the stock's sector for that day.

## Macro Features
Broadcasted to all tickers per date. Forward-filled on non-trading days.
* `vix_level`, `vix_change_1d`: India VIX closing level and daily change.
* `FII_Net`, `DII_Net`: Net flows from Foreign and Domestic Institutional Investors.
* `repo_rate`: RBI repo rate.
* `days_since_repo_change`: Days since last RBI rate adjustment.

## Sentiment Features
Derived from historical news scraping and scored using the FinBERT continuous probability distribution.
* `daily_sentiment_score`: Continuous score [-1, 1] calculated as `Prob(Positive) - Prob(Negative)` averaged across all articles for the ticker on that day.
* `news_volume`: Count of articles published for the ticker on that day.
* `has_news`: Binary flag (1 if `news_volume > 0`, 0 otherwise).
* `sentiment_3d_avg`, `sentiment_7d_avg`: Rolling 3-day and 7-day average of `daily_sentiment_score`.
* `news_volume_7d`: Rolling 7-day sum of `news_volume`.
* `insider_sentiment_interaction`: `core_net_insider_value_30d * sentiment_7d_avg`. Models the thesis that insider buys coupled with strong public sentiment yield a stronger predictive signal.

## Note on Missing Data
- `0-fill`: Activity counts and trade values on non-event days.
- `ffill`: Macro features and last-known insider metrics.
- `9999`: Sentinel for "no historical trade" in days_since_last_insider_trade.
- `NaN`: Forward labels at panel edges where insufficient future data exists.

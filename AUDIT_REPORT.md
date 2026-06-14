# Codebase Audit Report: insider-sentiment-framework

**Audit Date**: 2026-06-13
**Scope**: Full end-to-end audit of data, features, models, event study, backtest, code quality, and documentation.

---

## Summary of Findings

| # | Area | Issue | Severity | Status |
|---|------|-------|----------|--------|
| 1 | Data | 760 orphaned price files in `data/raw/prices/` not in 396-ticker universe | Minor | Needs decision |
| 2 | Data | 17 out-of-range dates in insider_trades_clean.csv (e.g., 1922, 2036) | Minor | False alarm (filtered by price index) |
| 3 | Data | `TYPE OF SECURITY (ACQUIRED/DISPLOSED)` column has only 1 unique value | Minor | Needs decision |
| 4 | Data | `REMARK` column has only 1 unique value | Minor | Needs decision |
| 5 | Features | **94.2% of tickers have sector = "Unknown"**. Sector-based features/benchmarks are effectively universe-wide averages, not sector benchmarks. | **Critical** | Needs decision |
| 6 | Features | Sector self-contamination: For single-ticker sectors (Construction, Oil&Gas, Telecom), sector_abnormal_return is always exactly 0 | Moderate | Needs decision |
| 7 | Features | Realized volatility forward window includes current day's return (off-by-one) | Minor | False alarm (negligible impact) |
| 8 | Modeling | 4 tickers in test set not present in training set (AADHARHFC, HDBFS, GROWW, IKS); categorical codes will mismatch | Moderate | Needs decision |
| 9 | Modeling | Independent `.astype('category')` on each split can produce mismatched integer codes even for shared tickers | Moderate | Needs decision |
| 10 | Event Study | **Report claims CAR_20 = +1.09% but reproduction gives +1.12%** | Moderate | Needs fix |
| 11 | Event Study | Report claims 2019-2022 CAR = +1.17% but reproduction gives +1.19% | Moderate | Needs fix |
| 12 | Event Study | Report claims 2023-2026 CAR = +0.96% but reproduction gives +1.00% | Moderate | Needs fix |
| 13 | Event Study | Report claims Global SD = 13.43% but reproduction gives 13.39% | Minor | Needs fix |
| 14 | Event Study | Sector benchmark in event study = universe average for 94% of events (same as Finding #5) | **Critical** | Needs decision |
| 15 | Backtest | Dead code on line 65 of backtest.py (overwritten by line 69) | Minor | Needs fix |
| 16 | Backtest | Possible 1-day look-ahead: signal activates on disclosure date, but disclosure may be published after market close | Moderate | Needs decision |
| 17 | Code Quality | 13 scratch/patch files at repo root (patch_insider.py, fix_indent.py, test_bse_api.py, etc.) | Minor | Needs cleanup |
| 18 | Code Quality | 2 stray CSV files at repo root (bse_scrips.csv, nifty200_raw.csv) | Minor | Needs cleanup |
| 19 | Code Quality | Log files at repo root and analysis/ (evaluate_output.log, checks1.log, checks2.log) | Minor | Needs cleanup |
| 20 | Code Quality | **requirements.txt is severely incomplete**: missing lightgbm, scipy, matplotlib, shap, scikit-learn, tqdm, seaborn, joblib, pymoo, selenium | **Critical** | Needs fix |
| 21 | Documentation | **data_dictionary.md claims 1-year window** ("2025-06-13 through 2026-06-13") but actual data is 7 years (2019-2026) | **Critical** | Needs fix |
| 22 | Documentation | data_dictionary.md missing all new features: core/peripheral splits, pct_stake_change, clustering features, 40d/60d targets | Moderate | Needs fix |
| 23 | Documentation | README.md does not mention universe size (396) or current date range | Moderate | Needs fix |
| 24 | Documentation | README.md does not note that sentiment/social features are not implemented (data_dictionary.md does note this, but README does not) | Moderate | Needs fix |
| 25 | Documentation | FINAL_REPORT.md and EVENT_STUDY_RESULTS.md have stale numbers that don't match current reproduction (see #10-13) | Moderate | Needs fix |

---

## Critical Issues Detail

### Finding #5 & #14: Sector Mapping is 94% "Unknown"

This is the single most important finding in this audit.

**What happened**: The `config/universe.csv` file assigns a real sector to only 23 out of 396 tickers. The remaining 373 are labeled "Unknown".

**Impact on panel model**: `sector_abnormal_return` for Unknown tickers = `stock_return - mean(return of all 373 Unknown tickers)`, which is essentially market-adjusted return, not sector-adjusted. The sector categorical feature is also meaningless since 94% share the same category.

**Impact on event study**: The "abnormal return" in the event study uses the same sector mapping. For 94% of events, CAR = stock cumulative return minus the equal-weight universe return. This means the headline "+1.12% CAR" is market-adjusted, not sector-adjusted. This is still a valid measure of abnormal return (market-model-adjusted), but it is NOT the sector-specific alpha the documentation implies. The effect could partly reflect insiders correctly timing market beta exposure.

**Impact on segmentation analysis**: The per-sector segmentation was run with most sectors having 1-5 tickers, making the per-sector LightGBM models train on tiny samples (~260-520 test rows). These were correctly identified as noise during the multiple-testing check, but the *reason* they were noise is that the sector labels themselves are almost entirely absent.

**Decision needed**: Should the sector mapping be enriched (e.g., using NSE sector classifications for all 396 tickers) to produce a true sector-adjusted benchmark? Or should the report explicitly state that "abnormal return" means "market-adjusted return"?

### Finding #20: requirements.txt Severely Incomplete

The requirements.txt was written during the initial data collection phase and was never updated as modeling, analysis, and optimization dependencies were added. It is missing 14 packages that are actively imported across the codebase, including core dependencies like lightgbm, scipy, matplotlib, and scikit-learn.

### Finding #21: data_dictionary.md Stale

The data dictionary still claims the master dataset covers a 1-year window ("2025-06-13 through 2026-06-13"), which was the scope of the original first session. The actual current dataset spans 7 years (2019-06-17 to 2026-06-12).

---

## Issues Fixed in This Audit

None. This audit was read-only and diagnostic. All fixes require user decision on scope/priority.

---

## Issues Requiring User Input

1. **Sector mapping (#5, #14)**: Enrich universe.csv with real sector labels for all 396 tickers, or reframe all documentation to say "market-adjusted" instead of "sector-adjusted"?
2. **Orphaned files (#1, #17, #18, #19)**: Clean up the 760 orphaned price CSVs and 13+ scratch scripts? Move to `debug/` or delete?
3. **Report numbers (#10-13, #25)**: Update FINAL_REPORT.md and EVENT_STUDY_RESULTS.md with reproduced figures (+1.12%, +1.19%, +1.00%, 13.39%)?
4. **requirements.txt (#20)**: Auto-generate from actual imports?
5. **data_dictionary.md (#21, #22)**: Full rewrite to reflect current 7-year, 49-column dataset?
6. **Categorical encoding (#8, #9)**: Fix train.py to use a shared CategoricalDtype across splits?
7. **Look-ahead (#16)**: Shift signal activation to t+1 in signal_construction.py?

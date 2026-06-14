import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from pathlib import Path
import logging
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path('models/output')
ANALYSIS_DIR = Path('analysis')
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ['abnormal_return_5d', 'abnormal_return_10d', 'abnormal_return_20d']

ORIGINAL_INSIDER_FEATURES = [
    'net_insider_value_30d', 'net_insider_value_90d',
    'insider_trade_count_30d', 'insider_trade_count_90d',
    'insider_role_weight', 'disclosure_lag_days',
    'days_since_last_insider_trade', 'event_proximity_flag',
    'sector_insider_sentiment'
]
ROLE_SPLIT_FEATURES = [
    'core_net_insider_value_30d', 'core_net_insider_value_90d',
    'peripheral_net_insider_value_30d', 'peripheral_net_insider_value_90d',
    'core_trade_count_30d', 'core_trade_count_90d',
    'peripheral_trade_count_30d', 'peripheral_trade_count_90d',
]
RELATIVE_SIZE_FEATURES = [
    'core_pct_stake_change_30d', 'core_pct_stake_change_90d',
    'peripheral_pct_stake_change_30d', 'peripheral_pct_stake_change_90d',
]
CLUSTERING_FEATURES = [
    'n_distinct_core_insiders_buying_30d', 'n_distinct_core_insiders_selling_30d',
    'n_distinct_core_insiders_buying_90d', 'n_distinct_core_insiders_selling_90d',
]

FEATURE_SETS = {
    'original': ORIGINAL_INSIDER_FEATURES,
    'role_split': ORIGINAL_INSIDER_FEATURES + ROLE_SPLIT_FEATURES,
    'relative_size': ORIGINAL_INSIDER_FEATURES + RELATIVE_SIZE_FEATURES,
    'clustering': ORIGINAL_INSIDER_FEATURES + CLUSTERING_FEATURES,
    'all_combined': ORIGINAL_INSIDER_FEATURES + ROLE_SPLIT_FEATURES + RELATIVE_SIZE_FEATURES + CLUSTERING_FEATURES
}

CATEGORICAL_FEATURES = ['ticker', 'sector']

def get_market_caps(universe_path):
    universe = pd.read_csv(universe_path)
    logger.info(f"Fetching market caps for {len(universe)} tickers...")
    mcaps = {}
    for t in universe['ticker']:
        try:
            info = yf.Ticker(t).info
            mcap = info.get('marketCap', np.nan)
            if pd.isna(mcap):
                mcap = info.get('previousClose', 0) * info.get('impliedSharesOutstanding', 0)
            if mcap == 0:
                mcap = np.nan
            mcaps[t] = mcap
        except Exception as e:
            mcaps[t] = np.nan
            
    universe['mcap'] = universe['ticker'].map(mcaps)
    
    # Calculate terciles based on available data
    valid_mcaps = universe.dropna(subset=['mcap'])
    terciles = pd.qcut(valid_mcaps['mcap'], 3, labels=['Small', 'Mid', 'Large'])
    universe['mcap_tier'] = pd.Series(dtype='object')
    universe.loc[valid_mcaps.index, 'mcap_tier'] = terciles.astype(str)
    
    tier_counts = universe['mcap_tier'].value_counts()
    logger.info("Market Cap Tier counts:")
    for tier, count in tier_counts.items():
        logger.info(f"  {tier}: {count}")
        
    return universe[['ticker', 'mcap_tier', 'sector']]

def train_lgb(features, target, train, val):
    mask_train = train[target].notnull()
    mask_val = val[target].notnull()
    
    X_train = train.loc[mask_train, features]
    y_train = train.loc[mask_train, target]
    
    X_val = val.loc[mask_val, features]
    y_val = val.loc[mask_val, target]
    
    if len(X_train) == 0 or len(X_val) == 0:
        return None
        
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'max_depth': 4,
        'num_leaves': 15,
        'learning_rate': 0.05,
        'verbose': -1,
        'random_state': 42
    }
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    return model

def main():
    universe_info = get_market_caps('config/universe.csv')
    
    logger.info("Loading splits...")
    train = pd.read_parquet(MODELS_DIR / 'train_split.parquet')
    val = pd.read_parquet(MODELS_DIR / 'val_split.parquet')
    test = pd.read_parquet(MODELS_DIR / 'test_split.parquet')
    
    # Merge universe info
    train = train.merge(universe_info[['ticker', 'mcap_tier']], on='ticker', how='left')
    val = val.merge(universe_info[['ticker', 'mcap_tier']], on='ticker', how='left')
    test = test.merge(universe_info[['ticker', 'mcap_tier']], on='ticker', how='left')
    
    for cat in CATEGORICAL_FEATURES:
        train[cat] = train[cat].astype('category')
        val[cat] = val[cat].astype('category')
        test[cat] = test[cat].astype('category')
        
    groupings = [
        ('mcap_tier', train['mcap_tier'].dropna().unique()),
        ('sector', train['sector'].dropna().unique())
    ]
    
    results = []
    
    for group_col, group_vals in groupings:
        for group_val in group_vals:
            train_sub = train[train[group_col] == group_val]
            val_sub = val[val[group_col] == group_val]
            test_sub = test[test[group_col] == group_val]
            
            total_test_rows = len(test_sub)
            if total_test_rows == 0:
                continue
                
            warning_flag = total_test_rows < 2000
            if warning_flag:
                logger.warning(f"Subgroup {group_val} has extremely small test set ({total_test_rows} rows).")
                
            for target in TARGETS:
                mask_test = test_sub[target].notnull()
                y_test = test_sub.loc[mask_test, target]
                
                if len(y_test) == 0:
                    continue
                    
                model_base = train_lgb(CATEGORICAL_FEATURES, target, train_sub, val_sub)
                if model_base is None:
                    continue
                    
                preds_base = model_base.predict(test_sub.loc[mask_test, CATEGORICAL_FEATURES])
                mae_base = mean_absolute_error(y_test, preds_base)
                
                for set_name, feature_subset in FEATURE_SETS.items():
                    features = CATEGORICAL_FEATURES + feature_subset
                    model_main = train_lgb(features, target, train_sub, val_sub)
                    
                    if model_main is None:
                        continue
                        
                    preds_main = model_main.predict(test_sub.loc[mask_test, features])
                    mae_main = mean_absolute_error(y_test, preds_main)
                    
                    improvement_mae = (mae_base - mae_main) / mae_base * 100
                    
                    result_entry = {
                        'Grouping': group_col,
                        'Subgroup': group_val,
                        'Target': target,
                        'Feature_Set': set_name,
                        'Baseline_MAE': mae_base,
                        'Ablation_MAE': mae_main,
                        'Improvement_%': improvement_mae,
                        'Test_Rows': total_test_rows,
                        'Warning_Small_Sample': warning_flag
                    }
                    results.append(result_entry)
                    logger.info(f"[{group_col}:{group_val}] {target} | {set_name} -> Improv: {improvement_mae:+.2f}%")
                    
                    if improvement_mae > 0.5:
                        logger.info(f"*** SIGNAL DETECTED: {improvement_mae:+.2f}% Improv for {group_val} on {target} ({set_name}) ***")
                        mask_train = train_sub[target].notnull()
                        X_train = train_sub.loc[mask_train, features]
                        if len(X_train) > 0:
                            explainer = shap.TreeExplainer(model_main)
                            sample = X_train.sample(n=min(2000, len(X_train)), random_state=42)
                            shap_values = explainer.shap_values(sample)
                            
                            plt.figure(figsize=(10, 8))
                            shap.summary_plot(shap_values, sample, show=False)
                            plt.title(f'SHAP: {group_val} | {target} | {set_name}')
                            plt.tight_layout()
                            
                            safe_val = str(group_val).replace('/', '_').replace(' ', '_')
                            plt.savefig(ANALYSIS_DIR / f'shap_{group_col}_{safe_val}_{target}_{set_name}.png')
                            plt.close()
                            
    res_df = pd.DataFrame(results)
    res_df.to_csv(ANALYSIS_DIR / 'segmentation_metrics.csv', index=False)
    logger.info("Saved raw metrics to analysis/segmentation_metrics.csv")

if __name__ == '__main__':
    main()

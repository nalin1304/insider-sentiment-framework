import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path('data/processed')
MODELS_DIR = Path('models/output')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    'abnormal_return_5d', 'abnormal_return_10d', 'abnormal_return_20d', 'abnormal_return_40d', 'abnormal_return_60d',
    'sector_abnormal_return_5d', 'sector_abnormal_return_10d', 'sector_abnormal_return_20d', 'sector_abnormal_return_40d', 'sector_abnormal_return_60d',
    'realized_volatility_10d', 'realized_volatility_40d',
    'max_drawdown_20d', 'max_drawdown_40d'
]

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

TECHNICAL_FEATURES = [
    'rsi_14d', 'dist_from_200d_ma'
]

# SENTIMENT_FEATURES = [
#     'daily_sentiment_score', 'news_volume', 'has_news',
#     'sentiment_3d_avg', 'sentiment_7d_avg', 'news_volume_7d',
#     'insider_sentiment_interaction'
# ]
# Sentiment is officially disabled due to 97.7% missing coverage.
SENTIMENT_FEATURES = []

OPEN_MARKET_FEATURES = [
    'open_market_net_insider_value_30d', 'open_market_net_insider_value_90d',
    'open_market_trade_count_30d', 'open_market_trade_count_90d',
]

ADVANCED_FEATURES = [
    'insider_value_to_adv_30d', 'insider_value_to_adv_90d',
    'open_market_insider_value_to_adv_30d', 'open_market_insider_value_to_adv_90d',
    'sector_neutral_insider_value_90d'
]

FEATURE_SETS = {
    'original': ORIGINAL_INSIDER_FEATURES,
    'role_split': ORIGINAL_INSIDER_FEATURES + ROLE_SPLIT_FEATURES,
    'relative_size': ORIGINAL_INSIDER_FEATURES + RELATIVE_SIZE_FEATURES,
    'clustering': ORIGINAL_INSIDER_FEATURES + CLUSTERING_FEATURES,
    'filtered_mode': OPEN_MARKET_FEATURES,
    'all_insider': ORIGINAL_INSIDER_FEATURES + ROLE_SPLIT_FEATURES + RELATIVE_SIZE_FEATURES + CLUSTERING_FEATURES + OPEN_MARKET_FEATURES + ADVANCED_FEATURES,
    'all_with_tech': ORIGINAL_INSIDER_FEATURES + ROLE_SPLIT_FEATURES + RELATIVE_SIZE_FEATURES + CLUSTERING_FEATURES + TECHNICAL_FEATURES + OPEN_MARKET_FEATURES + ADVANCED_FEATURES,
    'all_with_sentiment_and_tech': ORIGINAL_INSIDER_FEATURES + ROLE_SPLIT_FEATURES + RELATIVE_SIZE_FEATURES + CLUSTERING_FEATURES + SENTIMENT_FEATURES + TECHNICAL_FEATURES + OPEN_MARKET_FEATURES + ADVANCED_FEATURES
}

CATEGORICAL_FEATURES = ['ticker', 'sector']

def train_models():
    # 1. Load Data
    logger.info("Loading master dataset...")
    df = pd.read_parquet(PROCESSED_DIR / 'master_dataset.parquet')
    
    # Drop rows without dates
    df = df.dropna(subset=['date']).sort_values('date')
    
    unique_dates = df['date'].unique()
    total_dates = len(unique_dates)
    
    # 2. Time-series Purged Split (70/15/15) with 60-day embargo
    # A 60 trading-day gap prevents forward-looking labels from leaking into the validation/test sets
    train_split_idx = int(total_dates * 0.70)
    val_split_idx = int(total_dates * 0.85)
    
    embargo_days = 60
    
    train_dates = unique_dates[:train_split_idx - embargo_days]
    val_dates = unique_dates[train_split_idx:val_split_idx - embargo_days]
    test_dates = unique_dates[val_split_idx:]
    
    train = df[df['date'].isin(train_dates)].copy()
    val = df[df['date'].isin(val_dates)].copy()
    test = df[df['date'].isin(test_dates)].copy()
    
    logger.info("--- PURGED SPLIT SUMMARY (60-day embargo) ---")
    logger.info(f"Train: {len(train_dates)} dates, {len(train)} rows")
    logger.info(f"Val:   {len(val_dates)} dates, {len(val)} rows")
    logger.info(f"Test:  {len(test_dates)} dates, {len(test)} rows")
    
    if len(test) < 2000:
        logger.warning("WARNING: Test set is extremely small (<2000 rows). Results may be noisy/unreliable.")
        
    # Convert categoricals with SHARED category set to ensure consistent encoding
    for cat in CATEGORICAL_FEATURES:
        all_categories = pd.CategoricalDtype(categories=np.sort(df[cat].dropna().unique()), ordered=False)
        train[cat] = train[cat].astype(all_categories)
        val[cat] = val[cat].astype(all_categories)
        test[cat] = test[cat].astype(all_categories)
        
    # 3. Train models
    # Conservative parameters for 7yr dataset to prevent overfitting
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'max_depth': 4,            # Shallow trees to prevent overfitting
        'num_leaves': 15,          # Constrain leaves
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'lambda_l1': 1.0,          # Strong L1 regularization 
        'lambda_l2': 1.0,          # Strong L2 regularization
        'verbose': -1,
        'random_state': 42
    }
    
    trained_models = {}
    
    for target in TARGETS:
        for set_name, feature_subset in FEATURE_SETS.items():
            logger.info(f"Training LightGBM for {target} with feature set {set_name}...")
            
            features = CATEGORICAL_FEATURES + feature_subset
            
            # Drop NaNs in target
            mask_train = train[target].notnull()
            mask_val = val[target].notnull()
            
            X_train = train.loc[mask_train, features]
            y_train = train.loc[mask_train, target]
            
            X_val = val.loc[mask_val, features]
            y_val = val.loc[mask_val, target]
            
            if len(X_train) == 0:
                logger.warning(f"No valid training data for {target}. Skipping.")
                continue
                
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_train, lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            model_key = f"{target}_{set_name}"
            trained_models[model_key] = model
            
            # Save model
            joblib.dump(model, MODELS_DIR / f'lgbm_{model_key}.pkl')
            
            # 4. SHAP Feature Importance
            if set_name in ['all_with_sentiment_and_tech', 'all_with_tech']:
                explainer = shap.TreeExplainer(model)
                sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)
                
                # Check for categorical columns and convert to category type for SHAP if needed
                for cat in CATEGORICAL_FEATURES:
                    if cat in sample.columns and not isinstance(sample[cat].dtype, pd.CategoricalDtype):
                        sample[cat] = sample[cat].astype('category')
                        
                shap_values = explainer.shap_values(sample)
                
                # Global Summary Plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, sample, show=False)
                plt.title(f'SHAP Values: {target} (All Features)')
                plt.tight_layout()
                plt.savefig(MODELS_DIR / f'shap_{target}.png')
                plt.close()
                
                # SHAP Interaction (Dependence) Plot
                if 'core_net_insider_value_30d' in sample.columns and 'sentiment_7d_avg' in sample.columns:
                    try:
                        plt.figure(figsize=(8, 6))
                        shap.dependence_plot(
                            "core_net_insider_value_30d", 
                            shap_values, 
                            sample, 
                            interaction_index="sentiment_7d_avg",
                            show=False
                        )
                        plt.title(f'SHAP Interaction: Insider Value vs Sentiment ({target})')
                        plt.tight_layout()
                        plt.savefig(MODELS_DIR / f'shap_interaction_{target}.png')
                        plt.close()
                    except Exception as e:
                        logger.warning(f"Failed to generate SHAP dependence plot: {e}")
        
    # Save splits for evaluate and predict
    train.to_parquet(MODELS_DIR / 'train_split.parquet', index=False)
    val.to_parquet(MODELS_DIR / 'val_split.parquet', index=False)
    test.to_parquet(MODELS_DIR / 'test_split.parquet', index=False)
    
    logger.info("Training complete. Models and SHAP plots saved to models/output/")

if __name__ == '__main__':
    train_models()

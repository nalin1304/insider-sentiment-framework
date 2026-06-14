import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import joblib

def print_top_features(target):
    print(f"\nTop features for {target} (all_combined):")
    model = joblib.load(f'models/output/lgbm_{target}_all_combined.pkl')
    train = pd.read_parquet('models/output/train_split.parquet')
    mask = train[target].notnull()
    
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
    features = ['ticker', 'sector'] + ORIGINAL_INSIDER_FEATURES + ROLE_SPLIT_FEATURES + RELATIVE_SIZE_FEATURES + CLUSTERING_FEATURES
    
    X_train = train.loc[mask, features]
    sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame({'feature': features, 'importance': mean_abs_shap})
    feat_importance = feat_importance.sort_values('importance', ascending=False)
    
    print(feat_importance.head(15).to_string(index=False))

print_top_features('abnormal_return_20d')
print_top_features('realized_volatility_10d')

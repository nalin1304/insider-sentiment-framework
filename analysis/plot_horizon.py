import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import joblib

MODELS_DIR = Path('models/output')
ANALYSIS_DIR = Path('analysis')

def main():
    test = pd.read_parquet(MODELS_DIR / 'test_split.parquet')
    train = pd.read_parquet(MODELS_DIR / 'train_split.parquet')
    val = pd.read_parquet(MODELS_DIR / 'val_split.parquet')
    
    for cat in ['ticker', 'sector']:
        train[cat] = train[cat].astype('category')
        val[cat] = val[cat].astype('category')
        test[cat] = test[cat].astype('category')

    horizons = ['5d', '10d', '20d', '40d', '60d']
    targets_base = ['abnormal_return', 'sector_abnormal_return']
    
    ORIGINAL = ['net_insider_value_30d', 'net_insider_value_90d', 'insider_trade_count_30d', 'insider_trade_count_90d', 'insider_role_weight', 'disclosure_lag_days', 'days_since_last_insider_trade', 'event_proximity_flag', 'sector_insider_sentiment']
    ROLE_SPLIT = ['core_net_insider_value_30d', 'core_net_insider_value_90d', 'peripheral_net_insider_value_30d', 'peripheral_net_insider_value_90d', 'core_trade_count_30d', 'core_trade_count_90d', 'peripheral_trade_count_30d', 'peripheral_trade_count_90d']
    RELATIVE = ['core_pct_stake_change_30d', 'core_pct_stake_change_90d', 'peripheral_pct_stake_change_30d', 'peripheral_pct_stake_change_90d']
    CLUSTERING = ['n_distinct_core_insiders_buying_30d', 'n_distinct_core_insiders_selling_30d', 'n_distinct_core_insiders_buying_90d', 'n_distinct_core_insiders_selling_90d']
    
    FEATURE_SETS = {
        'original': ORIGINAL,
        'all_combined': ORIGINAL + ROLE_SPLIT + RELATIVE + CLUSTERING
    }
    
    CATEGORICAL = ['ticker', 'sector']
    
    def get_baseline_mae(target):
        features = CATEGORICAL
        mask_train = train[target].notnull()
        mask_val = val[target].notnull()
        mask_test = test[target].notnull()
        
        X_train = train.loc[mask_train, features]
        y_train = train.loc[mask_train, target]
        X_val = val.loc[mask_val, features]
        y_val = val.loc[mask_val, target]
        
        if len(X_train) == 0: return np.nan
        
        params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'max_depth': 4, 'num_leaves': 15, 'learning_rate': 0.05, 'verbose': -1, 'random_state': 42}
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        model = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_train, lgb_val], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        preds = model.predict(test.loc[mask_test, features])
        return mean_absolute_error(test.loc[mask_test, target], preds)

    results = []
    
    for tb in targets_base:
        for h in horizons:
            target = f"{tb}_{h}"
            mae_base = get_baseline_mae(target)
            
            mask_test = test[target].notnull()
            y_test = test.loc[mask_test, target]
            
            row = {'Target_Base': tb, 'Horizon': h, 'Baseline_MAE': mae_base}
            
            for set_name in FEATURE_SETS.keys():
                model_path = MODELS_DIR / f'lgbm_{target}_{set_name}.pkl'
                if model_path.exists():
                    model = joblib.load(model_path)
                    preds = model.predict(test.loc[mask_test, CATEGORICAL + FEATURE_SETS[set_name]])
                    mae_main = mean_absolute_error(y_test, preds)
                    improv = (mae_base - mae_main) / mae_base * 100
                    row[f'{set_name}_Improv'] = improv
                else:
                    row[f'{set_name}_Improv'] = np.nan
            results.append(row)
            
    df = pd.DataFrame(results)
    print(df.to_string())
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_labels = [5, 10, 20, 40, 60]
    
    for i, tb in enumerate(targets_base):
        sub = df[df['Target_Base'] == tb]
        axes[i].plot(x_labels, sub['original_Improv'], marker='o', label='Original Features')
        axes[i].plot(x_labels, sub['all_combined_Improv'], marker='s', label='Refined Features (All Combined)')
        axes[i].set_title(f"MAE Improvement (%)\n{tb}")
        axes[i].set_xlabel("Horizon (Days)")
        axes[i].set_ylabel("Improvement over Baseline (%)")
        axes[i].set_xticks(x_labels)
        axes[i].axhline(0, color='gray', linestyle='--')
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'horizon_improvement.png')

if __name__ == "__main__":
    main()

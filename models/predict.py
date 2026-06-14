import logging
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path('models/output')
PROCESSED_DIR = Path('data/processed')

TARGETS = [
    'abnormal_return_5d', 'abnormal_return_10d', 'abnormal_return_20d',
    'sector_abnormal_return_5d', 'sector_abnormal_return_10d', 'sector_abnormal_return_20d',
    'realized_volatility_10d', 'max_drawdown_20d'
]

INSIDER_FEATURES = [
    'net_insider_value_30d', 'net_insider_value_90d',
    'insider_trade_count_30d', 'insider_trade_count_90d',
    'insider_role_weight', 'disclosure_lag_days',
    'days_since_last_insider_trade', 'event_proximity_flag',
    'sector_insider_sentiment'
]

CATEGORICAL_FEATURES = ['ticker', 'sector']

def train_baseline(target, train, val):
    """Train a naive baseline model without insider features (only ticker, sector)."""
    features = CATEGORICAL_FEATURES
    
    mask_train = train[target].notnull()
    mask_val = val[target].notnull()
    
    X_train = train.loc[mask_train, features]
    y_train = train.loc[mask_train, target]
    
    X_val = val.loc[mask_val, features]
    y_val = val.loc[mask_val, target]
    
    if len(X_train) == 0:
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

def generate_predictions():
    logger.info("Loading train, test and validation splits...")
    train = pd.read_parquet(MODELS_DIR / 'train_split.parquet')
    val = pd.read_parquet(MODELS_DIR / 'val_split.parquet')
    test = pd.read_parquet(MODELS_DIR / 'test_split.parquet')
    
    for cat in CATEGORICAL_FEATURES:
        train[cat] = train[cat].astype('category')
        val[cat] = val[cat].astype('category')
        test[cat] = test[cat].astype('category')
        
    features = CATEGORICAL_FEATURES + INSIDER_FEATURES
    
    # We will output a dataframe of predictions for the test set
    pred_df = test[['ticker', 'date']].copy()
    
    # We will also compute historical signal-reliability (validation prediction error) per ticker
    # We can aggregate validation errors for abnormal_return_20d as the main reliability metric.
    val_pred_df = val[['ticker', 'date']].copy()
    
    for target in TARGETS:
        model_path = MODELS_DIR / f'lgbm_{target}.pkl'
        if not model_path.exists():
            continue
            
        model = joblib.load(model_path)
        
        # Predict on Test
        pred_df[f'pred_{target}'] = model.predict(test[features])
        
        # Train and predict baseline for all targets (optimizer needs baseline returns, and maybe baseline risk)
        model_base = train_baseline(target, train, val)
        if model_base is not None:
            pred_df[f'pred_{target}_base'] = model_base.predict(test[CATEGORICAL_FEATURES])
        
        # Predict on Val for reliability tracking
        val_pred_df[f'pred_{target}'] = model.predict(val[features])
        
        # Calculate validation error for this target
        if target in val.columns:
            val_pred_df[f'err_{target}'] = np.abs(val[target] - val_pred_df[f'pred_{target}'])
            
    # Compute average validation error per ticker
    # This serves as the 'signal-reliability term' to downweight models that perform poorly for specific tickers
    reliability_df = val_pred_df.groupby('ticker').agg(
        val_error_abnormal_return_20d=('err_abnormal_return_20d', 'mean')
    ).reset_index()
    
    # Merge reliability into test predictions
    pred_df = pd.merge(pred_df, reliability_df, on='ticker', how='left')
    
    out_path = PROCESSED_DIR / 'predictions.csv'
    pred_df.to_csv(out_path, index=False)
    logger.info(f"OOS predictions and ticker reliability scores saved to {out_path}")

if __name__ == '__main__':
    generate_predictions()

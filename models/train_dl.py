import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import copy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path('models/output')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    'abnormal_return_5d', 'abnormal_return_10d', 'abnormal_return_20d', 'abnormal_return_40d', 'abnormal_return_60d',
    'sector_abnormal_return_5d', 'sector_abnormal_return_10d', 'sector_abnormal_return_20d', 'sector_abnormal_return_40d', 'sector_abnormal_return_60d',
    'realized_volatility_10d', 'realized_volatility_40d',
    'max_drawdown_20d', 'max_drawdown_40d'
]

# Minimal set of features for DL tests to avoid training 98 models. We'll train on the richest set.
FEATURE_SETS = {
    'all_with_sentiment_and_tech': [
        'net_insider_value_30d', 'net_insider_value_90d',
        'insider_trade_count_30d', 'insider_trade_count_90d',
        'insider_role_weight', 'disclosure_lag_days',
        'days_since_last_insider_trade', 'event_proximity_flag',
        'sector_insider_sentiment',
        'core_net_insider_value_30d', 'core_net_insider_value_90d',
        'peripheral_net_insider_value_30d', 'peripheral_net_insider_value_90d',
        'core_trade_count_30d', 'core_trade_count_90d',
        'peripheral_trade_count_30d', 'peripheral_trade_count_90d',
        'core_pct_stake_change_30d', 'core_pct_stake_change_90d',
        'peripheral_pct_stake_change_30d', 'peripheral_pct_stake_change_90d',
        'n_distinct_core_insiders_buying_30d', 'n_distinct_core_insiders_selling_30d',
        'n_distinct_core_insiders_buying_90d', 'n_distinct_core_insiders_selling_90d',
        # 'daily_sentiment_score', 'news_volume', 'has_news',
        # 'sentiment_3d_avg', 'sentiment_7d_avg', 'news_volume_7d',
        # 'insider_sentiment_interaction',
        'rsi_14d', 'dist_from_200d_ma',
        'open_market_net_insider_value_30d', 'open_market_net_insider_value_90d',
        'open_market_trade_count_30d', 'open_market_trade_count_90d',
        'insider_value_to_adv_30d', 'insider_value_to_adv_90d',
        'open_market_insider_value_to_adv_30d', 'open_market_insider_value_to_adv_90d',
        'sector_neutral_insider_value_90d'
    ]
}

CATEGORICAL_FEATURES = ['ticker', 'sector']

class TabularDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target_col):
        # Continuous features
        self.cont_X = df[cont_cols].astype(np.float32).values
        
        # Missing value imputation (fill NaN with 0, typical for these specific engineered features like rolling counts)
        self.cont_X = np.nan_to_num(self.cont_X, nan=0.0)
        
        # Categorical features (assume they are already integer encoded via pandas cat.codes + 1, with 0 as unknown)
        self.cat_X = df[cat_cols].values.astype(np.int64)
        
        self.y = df[target_col].astype(np.float32).values
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.cat_X[idx], self.cont_X[idx], self.y[idx]

class TabularMLP(nn.Module):
    def __init__(self, embedding_sizes, num_cont, out_sz=1, layers=[128, 64, 32], p=0.4):
        super().__init__()
        # Embedding layers for categoricals
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_sizes])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(num_cont)
        
        n_emb = sum((nf for ni, nf in embedding_sizes))
        n_in = n_emb + num_cont
        
        # Dense layers
        layerlist = []
        for i in range(len(layers)):
            layerlist.append(nn.Linear(n_in, layers[i]))
            layerlist.append(nn.GELU())
            layerlist.append(nn.BatchNorm1d(layers[i]))
            layerlist.append(nn.Dropout(p))
            n_in = layers[i]
            
        layerlist.append(nn.Linear(layers[-1], out_sz))
        
        self.layers = nn.Sequential(*layerlist)
        
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x.squeeze(1)

def train_dl_models():
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    logger.info("Loading splits...")
    train = pd.read_parquet(MODELS_DIR / 'train_split.parquet')
    val = pd.read_parquet(MODELS_DIR / 'val_split.parquet')
    test = pd.read_parquet(MODELS_DIR / 'test_split.parquet')
    
    # 2. Prepare Categoricals
    logger.info("Encoding categoricals...")
    cat_dims = {}
    cat_mappings = {}
    for cat in CATEGORICAL_FEATURES:
        # We need a unified mapping from the ENTIRE dataset so test set doesn't get NaNs
        # Unknowns during inference will be 0, so we shift codes by +1.
        all_cats = np.sort(pd.concat([train[cat], val[cat], test[cat]]).dropna().unique())
        mapping = {val: i+1 for i, val in enumerate(all_cats)}
        cat_mappings[cat] = mapping
        
        train[cat] = train[cat].map(mapping).fillna(0).astype(int)
        val[cat] = val[cat].map(mapping).fillna(0).astype(int)
        
        # Dimension is len(mapping) + 1 (for 0 index)
        # Embedding size: min(50, (cardinality+1)//2)
        n_unique = len(mapping) + 1
        emb_sz = min(50, (n_unique + 1) // 2)
        cat_dims[cat] = (n_unique, emb_sz)
    
    emb_sizes = [cat_dims[cat] for cat in CATEGORICAL_FEATURES]
    
    # Save the mapping for inference later
    torch.save(cat_mappings, MODELS_DIR / 'dl_cat_mappings.pth')
    
    batch_size = 1024
    epochs = 50
    patience = 5
    
    for target in TARGETS:
        for set_name, feature_subset in FEATURE_SETS.items():
            logger.info(f"Training TabularMLP for {target} with feature set {set_name}...")
            
            # Drop NaNs in target
            mask_train = train[target].notnull()
            mask_val = val[target].notnull()
            
            t_df = train.loc[mask_train].copy()
            v_df = val.loc[mask_val].copy()
            
            if len(t_df) == 0:
                logger.warning(f"No valid training data for {target}. Skipping.")
                continue
                
            train_ds = TabularDataset(t_df, CATEGORICAL_FEATURES, feature_subset, target)
            val_ds = TabularDataset(v_df, CATEGORICAL_FEATURES, feature_subset, target)
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = TabularMLP(emb_sizes, num_cont=len(feature_subset), layers=[256, 128, 64], p=0.4).to(device)
            criterion = nn.MSELoss()
            # AdamW with weight decay
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for cat_x, cont_x, y in train_loader:
                    cat_x, cont_x, y = cat_x.to(device), cont_x.to(device), y.to(device)
                    
                    optimizer.zero_grad()
                    preds = model(cat_x, cont_x)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                train_loss /= len(train_loader)
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for cat_x, cont_x, y in val_loader:
                        cat_x, cont_x, y = cat_x.to(device), cont_x.to(device), y.to(device)
                        preds = model(cat_x, cont_x)
                        loss = criterion(preds, y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                    break
            
            # Save the best model
            model_key = f"{target}_{set_name}"
            model_path = MODELS_DIR / f'dl_{model_key}.pth'
            torch.save({
                'model_state_dict': best_model_state,
                'emb_sizes': emb_sizes,
                'num_cont': len(feature_subset),
                'cat_dims': cat_dims  # Saving dims to reconstruct easily
            }, model_path)
            
            logger.info(f"Saved DL model to {model_path}")

if __name__ == '__main__':
    train_dl_models()

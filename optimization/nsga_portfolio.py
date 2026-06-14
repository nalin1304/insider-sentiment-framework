import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import argparse
import torch
import torch.nn as nn
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.decomposition.asf import ASF

def get_knee_point(F):
    """Pseudo-weights method to select balanced solution from Pareto front."""
    if F.ndim == 1:
        return 0
    # Normalize objectives
    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-8)
    # Give equal preference to all objectives (3 objectives in NSGA-II)
    weights = np.array([1/3, 1/3, 1/3])
    decomp = ASF()
    i = decomp.do(F_norm, 1/weights).argmin()
    return i

import sys
sys.path.append('.') # Allow importing models
from models.train_dl import TabularMLP, FEATURE_SETS, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path('data/processed')
OUTPUT_DIR = Path('optimization')

class PortfolioOptimizationProblem(ElementwiseProblem):
    def __init__(self, n_assets, expected_returns, predicted_volatility, prev_weights):
        super().__init__(
            n_var=n_assets,
            n_obj=3,     
            n_ieq_constr=0, 
            xl=0.0,      
            xu=0.05       
        )
        self.expected_returns = expected_returns
        self.predicted_volatility = predicted_volatility
        self.prev_weights = prev_weights
        
    def _evaluate(self, x, out, *args, **kwargs):
        sum_x = np.sum(x)
        if sum_x > 0:
            w = x / sum_x
        else:
            w = np.ones_like(x) / len(x)
            
        exp_ret = np.dot(w, self.expected_returns)
        f1 = -exp_ret
        
        risk = np.dot(w, self.predicted_volatility)
        f2 = risk
        
        turnover = np.sum(np.abs(w - self.prev_weights))
        f3 = turnover
        
        out["F"] = [f1, f2, f3]

def load_dl_model(target, set_name, device):
    model_path = f'models/output/dl_{target}_{set_name}.pth'
    checkpoint = torch.load(model_path, map_location=device)
    
    # Needs matching architecture
    model = TabularMLP(checkpoint['emb_sizes'], checkpoint['num_cont'], layers=[256, 128, 64], p=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['cat_dims']

def optimize_nsga_portfolio(model_type='lgbm'):
    logger.info(f"Loading {model_type.upper()} predictions and models...")
    try:
        df = pd.read_parquet(PROCESSED_DIR / 'master_dataset.parquet')
    except Exception as e:
        logger.error(f"Cannot load dataset: {e}")
        return
        
    dates = np.sort(df['date'].unique())
    tickers = np.sort(df['ticker'].unique())
    n_assets = len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    df['original_ticker'] = df['ticker']
    
    device = torch.device('cpu')
    if model_type == 'dl':
        device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    try:
        if model_type == 'lgbm':
            model_ret = joblib.load('models/output/lgbm_abnormal_return_20d_all_with_sentiment_and_tech.pkl')
            features_ret = model_ret.feature_name()
            
            model_vol = joblib.load('models/output/lgbm_realized_volatility_10d_all_with_sentiment_and_tech.pkl')
            features_vol = model_vol.feature_name()
        else:
            model_ret, _ = load_dl_model('abnormal_return_20d', 'all_with_sentiment_and_tech', device)
            model_vol, _ = load_dl_model('realized_volatility_10d', 'all_with_sentiment_and_tech', device)
            
            features_ret = FEATURE_SETS['all_with_sentiment_and_tech']
            features_vol = FEATURE_SETS['all_with_sentiment_and_tech']
            cat_mappings = torch.load('models/output/dl_cat_mappings.pth')
            
            # Map categoricals globally
            for cat in CATEGORICAL_FEATURES:
                df[cat] = df[cat].map(cat_mappings[cat]).fillna(0).astype(int)

        test_split = pd.read_parquet('models/output/test_split.parquet', columns=['date'])
        test_start_date = test_split['date'].min()
    except Exception as e:
        logger.error(f"Cannot load models or test split: {e}")
        return

    df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
    rebalance_dates = df.groupby('year_month')['date'].max().values
    rebalance_dates = [d for d in rebalance_dates if pd.to_datetime(d) >= test_start_date]
    
    nsga_weights = pd.DataFrame(0.0, index=pd.to_datetime(dates), columns=tickers)
    prev_w = np.ones(n_assets) / n_assets
    
    algorithm = NSGA2(
        pop_size=50,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    logger.info(f"Running NSGA-II optimization out-of-sample for {len(rebalance_dates)} months (from {test_start_date})...")
    
    if model_type == 'lgbm':
        for cat in CATEGORICAL_FEATURES:
            if cat in df.columns:
                df[cat] = df[cat].astype('category')
            
    for r_date in rebalance_dates:
        r_date = pd.to_datetime(r_date)
        day_data = df[df['date'] == r_date].copy()
        
        if len(day_data) == 0:
            continue
            
        if model_type == 'lgbm':
            preds_ret = model_ret.predict(day_data[features_ret])
            preds_vol = model_vol.predict(day_data[features_vol])
        else:
            cat_x = torch.tensor(day_data[CATEGORICAL_FEATURES].values.astype(np.int64)).to(device)
            
            cont_ret = np.nan_to_num(day_data[features_ret].values.astype(np.float32), nan=0.0)
            cont_ret_t = torch.tensor(cont_ret).to(device)
            
            cont_vol = np.nan_to_num(day_data[features_vol].values.astype(np.float32), nan=0.0)
            cont_vol_t = torch.tensor(cont_vol).to(device)
            
            with torch.no_grad():
                preds_ret = model_ret(cat_x, cont_ret_t).cpu().numpy()
                preds_vol = model_vol(cat_x, cont_vol_t).cpu().numpy()
        
        day_data['signal_ret'] = preds_ret
        day_data['signal_vol'] = preds_vol
            
        exp_rets = np.zeros(n_assets)
        vols = np.ones(n_assets) * 0.20 
        
        for i, original_ticker in enumerate(day_data['original_ticker'].values):
            idx = ticker_to_idx.get(original_ticker)
            if idx is not None:
                exp_rets[idx] = preds_ret[i]
                vols[idx] = preds_vol[i]
                
        exp_rets = np.nan_to_num(exp_rets, 0)
        vols = np.nan_to_num(vols, 0.20)
        
        problem = PortfolioOptimizationProblem(n_assets, exp_rets, vols, prev_w)
        
        res = minimize(problem,
                       algorithm,
                       ('n_gen', 40), 
                       seed=1,
                       verbose=False)
                       
        if res.X is not None:
            if res.F.ndim == 1:
                best_x = res.X
            else:
                best_idx = get_knee_point(res.F)
                best_x = res.X[best_idx]
            sum_x = np.sum(best_x)
            if sum_x > 0:
                best_w = best_x / sum_x
            else:
                best_w = np.ones(n_assets) / n_assets
        else:
            best_w = prev_w
            
        month_mask = (nsga_weights.index.year == r_date.year) & (nsga_weights.index.month == r_date.month)
        nsga_weights.loc[month_mask, :] = best_w
        
        prev_w = best_w
        logger.info(f"Optimized portfolio for {r_date.strftime('%Y-%m')}")
        
    out_filename = 'nsga_weights.parquet' if model_type == 'lgbm' else 'nsga_weights_dl.parquet'
    out_path = OUTPUT_DIR / out_filename
    nsga_weights.to_parquet(out_path)
    logger.info(f"Saved NSGA-II weights to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lgbm', choices=['lgbm', 'dl'])
    args = parser.parse_args()
    optimize_nsga_portfolio(args.model)

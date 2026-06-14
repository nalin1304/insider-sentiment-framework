import pandas as pd
import numpy as np
from pathlib import Path
import logging
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.decomposition.asf import ASF
import sys
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

from pymoo.core.repair import Repair

class PortfolioRepair(Repair):
    def _do(self, problem, Z, **kwargs):
        # Z is (pop_size, n_var)
        Z_repaired = np.zeros_like(Z)
        
        for i in range(len(Z)):
            w = Z[i].copy()
            w = np.clip(w, 0, 0.05)
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones_like(w) / len(w)
                
            # Iteratively enforce max constraints (5% per ticker, 25% per sector)
            for _ in range(5):
                # Sector constraints
                for s in problem.unique_sectors:
                    mask = (problem.sectors == s)
                    s_sum = w[mask].sum()
                    if s_sum > 0.25:
                        w[mask] = w[mask] * (0.25 / s_sum)
                
                # Ticker constraints
                w = np.clip(w, 0, 0.05)
                
                # Normalize again
                w_sum = w.sum()
                if w_sum > 0:
                    w = w / w_sum
            
            Z_repaired[i] = w
            
        return Z_repaired

class PortfolioProblem(Problem):
    def __init__(self, ret, vol, mdd, penalty, sectors):
        self.ret = ret
        self.vol = vol
        self.mdd = mdd
        self.penalty = penalty
        self.sectors = sectors
        self.unique_sectors = np.unique(sectors)
        
        super().__init__(
            n_var=len(ret), 
            n_obj=4, 
            n_ieq_constr=0, # All handled by repair
            xl=0.0, 
            xu=0.05
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # x is already repaired by PortfolioRepair
        f1 = -np.dot(x, self.ret)
        f2 = np.dot(x, self.vol)
        f3 = -np.dot(x, self.mdd) # MDD is negative, we want it closer to 0, so minimize -mdd
        f4 = np.dot(x, self.penalty)
        
        out["F"] = np.column_stack([f1, f2, f3, f4])

def get_knee_point(F):
    """Pseudo-weights method to select balanced solution from Pareto front."""
    # Normalize objectives
    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-8)
    # Give equal preference to all objectives
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    decomp = ASF()
    i = decomp.do(F_norm, 1/weights).argmin()
    return i

def run_optimization(mode='insider'):
    logger.info(f"Running optimizer in mode: {mode}")
    
    # Load universe to map tickers to sectors
    universe_csv = Path('config/universe.csv')
    univ_df = pd.read_csv(universe_csv)
    sector_map = dict(zip(univ_df['ticker'].str.replace('.NS', '').str.replace('.BO', ''), univ_df['sector']))
    
    pred_path = Path('data/processed/predictions.csv')
    df = pd.read_csv(pred_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # True month end rebalancing dates
    rebalance_dates = df.groupby(df['date'].dt.to_period('M'))['date'].max().values
    
    all_weights = []
    
    # Set target columns based on mode
    if mode == 'insider':
        col_ret = 'pred_abnormal_return_20d'       # Maximize insider return
        col_vol = 'pred_realized_volatility_10d'   # Insider risk
        col_mdd = 'pred_max_drawdown_20d'          # Insider risk
    else:
        col_ret = 'pred_abnormal_return_20d_base'
        col_vol = 'pred_realized_volatility_10d_base'
        col_mdd = 'pred_max_drawdown_20d_base'
        
    # Validation error penalty (we only have the baseline val_error or overall val_error)
    # The file has val_error_abnormal_return_20d. We will use that.
    col_pen = 'val_error_abnormal_return_20d'

    for rd in rebalance_dates:
        logger.info(f"Optimizing for date: {rd.strftime('%Y-%m-%d')}")
        day_df = df[df['date'] == rd].copy()
        
        # Drop rows with NaNs in required columns
        day_df = day_df.dropna(subset=[col_ret, col_vol, col_mdd, col_pen])
        
        if len(day_df) < 10:
            logger.warning(f"Not enough valid tickers on {rd}. Skipping.")
            continue
            
        sectors = np.array([sector_map.get(t, "Unknown") for t in day_df['ticker']])
        ret = day_df[col_ret].values
        vol = day_df[col_vol].values
        mdd = day_df[col_mdd].values
        pen = day_df[col_pen].values
        
        problem = PortfolioProblem(ret, vol, mdd, pen, sectors)
        algorithm = NSGA2(pop_size=100, repair=PortfolioRepair())
        termination = get_termination("n_gen", 150)
        
        res = minimize(problem, algorithm, termination, seed=42, verbose=False)
        
        if res.X is None:
            logger.warning(f"No feasible solution found for {rd}.")
            continue
            
        # Select knee point
        if res.F.ndim == 1:
            best_x = res.X
        else:
            best_idx = get_knee_point(res.F)
            best_x = res.X[best_idx]
            
        # Clean weights (enforce exactly sum to 1 after normalization)
        w = best_x / np.sum(best_x)
        w = np.clip(w, 0, 0.05)
        w = w / np.sum(w)
        
        for ticker, weight in zip(day_df['ticker'], w):
            if weight > 1e-4:
                all_weights.append({
                    'date': rd,
                    'ticker': ticker,
                    'weight': weight
                })
                
    weights_df = pd.DataFrame(all_weights)
    out_dir = Path('data/processed')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'portfolio_weights_{mode}.csv'
    weights_df.to_csv(out_path, index=False)
    logger.info(f"Saved optimized weights for {mode} mode to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='insider', choices=['insider', 'base'])
    args = parser.parse_args()
    
    Path("optimization/output").mkdir(parents=True, exist_ok=True)
    run_optimization(args.mode)

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def build_macro_features(macro_dir: Path) -> pd.DataFrame:
    """
    Load macro features like FII/DII net flows, India VIX level, and repo rate.
    Strictly NO synthetic data. If data is absent, returns an empty dataframe.
    """
    df_macro = pd.DataFrame()
    
    if not macro_dir.exists():
        logger.warning(f"Macro directory {macro_dir} does not exist. No macro features will be built.")
        return df_macro
        
    macro_dfs = []
    
    # Check for VIX (could be in indices or macro)
    # Just an example of how we'd load if the user provided it:
    vix_file = macro_dir / "INDIAVIX.csv"
    if vix_file.exists():
        vix = pd.read_csv(vix_file)
        if 'Date' in vix.columns and 'Close' in vix.columns:
            vix['date'] = pd.to_datetime(vix['Date'])
            vix['vix_level'] = vix['Close']
            vix['vix_change_1d'] = vix['vix_level'].pct_change()
            vix = vix[['date', 'vix_level', 'vix_change_1d']]
            macro_dfs.append(vix)
            
    # Check for FII/DII
    fii_file = macro_dir / "FII_DII.csv"
    if fii_file.exists():
        fii = pd.read_csv(fii_file)
        if 'Date' in fii.columns:
            fii['date'] = pd.to_datetime(fii['Date'])
            # Ensure columns exist, else we can't extract
            keep = ['date']
            if 'FII_Net' in fii.columns: keep.append('FII_Net')
            if 'DII_Net' in fii.columns: keep.append('DII_Net')
            macro_dfs.append(fii[keep])
            
    # Check for Repo Rate
    repo_file = macro_dir / "REPO_RATE.csv"
    if repo_file.exists():
        repo = pd.read_csv(repo_file)
        if 'Date' in repo.columns and 'Rate' in repo.columns:
            repo['date'] = pd.to_datetime(repo['Date'])
            repo['repo_rate'] = repo['Rate']
            # Calculate days since last change
            repo = repo.sort_values('date')
            repo['rate_changed'] = repo['repo_rate'].diff() != 0
            
            # Forward fill the date of the last change
            repo['last_change_date'] = repo['date'].where(repo['rate_changed']).ffill()
            repo['days_since_repo_change'] = (repo['date'] - repo['last_change_date']).dt.days
            
            repo = repo[['date', 'repo_rate', 'days_since_repo_change']]
            macro_dfs.append(repo)

    if not macro_dfs:
        logger.warning("No valid macro CSVs found. Returning empty dataframe.")
        return df_macro
        
    # Merge all macro features on date
    df_macro = macro_dfs[0]
    for m_df in macro_dfs[1:]:
        df_macro = pd.merge(df_macro, m_df, on='date', how='outer')
        
    df_macro = df_macro.sort_values('date').fillna(method='ffill')
    return df_macro

"""
Cleaner for insider trading data.
Loads raw NSE bulk exports, cleans columns, parses dates, maps categories,
and generates the final processed dataset used to derive the universe and features.
"""

import logging
import pandas as pd
from pathlib import Path
from glob import glob

try:
    from config.settings import RAW_INSIDER_DIR, PROCESSED_DIR, INSIDER_ROLE_WEIGHTS
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config.settings import RAW_INSIDER_DIR, PROCESSED_DIR, INSIDER_ROLE_WEIGHTS

logger = logging.getLogger("insider_clean")

def clean_insider_trades(raw_dir: Path = None, output_dir: Path = None) -> pd.DataFrame:
    """
    Load raw NSE insider trading CSVs, clean and standardize them.
    """
    if raw_dir is None:
        raw_dir = RAW_INSIDER_DIR
    if output_dir is None:
        output_dir = PROCESSED_DIR
        
    csv_files = glob(str(raw_dir / "*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {raw_dir}")
        return pd.DataFrame()
        
    dfs = []
    for f in csv_files:
        if "sample_" in Path(f).name:
            continue
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            
    if not dfs:
        logger.warning("Only sample files found or all failed. Using sample files if available...")
        sample_files = glob(str(raw_dir / "sample_*.csv"))
        for f in sample_files:
            dfs.append(pd.read_csv(f))
            
    if not dfs:
        logger.error("No data could be loaded.")
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    initial_rows = len(df)
    
    # 1. Standardize columns
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = df.columns.str.strip().str.replace('\n', '')
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            
    # 2. Filter invalid rows
    # Drop rows where EXCHANGE is "-" or transaction type is "-"
    mask = (df['EXCHANGE'] != "-") & (df['ACQUISITION/DISPOSAL TRANSACTION TYPE'] != "-")
    df = df[mask].copy()
    dropped = initial_rows - len(df)
    logger.info(f"Dropped {dropped} rows missing exchange or transaction type.")
    
    # 3. Parse dates
    # Normalize correct spellings to the misspellings expected by the rest of the pipeline
    col_mapping = {
        'DATE OF INTIMATION TO COMPANY': 'DATE OF INITMATION TO COMPANY',
        'BROADCAST DATE AND TIME': 'BROADCASTE DATE AND TIME'
    }
    df = df.rename(columns=col_mapping)

    date_cols = {
        'DATE OF ALLOTMENT/ACQUISITION FROM': '%d-%b-%Y',
        'DATE OF ALLOTMENT/ACQUISITION TO': '%d-%b-%Y',
        'DATE OF INITMATION TO COMPANY': '%d-%b-%Y',
        'BROADCASTE DATE AND TIME': '%d-%b-%Y %H:%M'
    }
    
    for col, fmt in date_cols.items():
        if col in df.columns:
            # Replace "-" with NaT
            df[col] = df[col].replace("-", pd.NaT)
            df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
            
    # 4. Map Transaction Types
    if 'ACQUISITION/DISPOSAL TRANSACTION TYPE' in df.columns:
        # Flag pledges
        pledge_terms = ['Pledge', 'Pledge Revoke', 'Pledge Invoke', 'Revoke', 'Invoke']
        df['is_pledge'] = df['ACQUISITION/DISPOSAL TRANSACTION TYPE'].apply(
            lambda x: any(term.lower() in str(x).lower() for term in pledge_terms)
        )
        # Normalize Buy/Sell
        def normalize_tx(x):
            x_lower = str(x).lower()
            if 'buy' in x_lower or 'acquisition' in x_lower: return 'Buy'
            if 'sell' in x_lower or 'disposal' in x_lower: return 'Sell'
            if 'pledge' in x_lower: return 'Pledge'
            return x
        df['clean_transaction_type'] = df['ACQUISITION/DISPOSAL TRANSACTION TYPE'].apply(normalize_tx)
        
    # 5. Map Category of Person to Insider Role Weight
    if 'CATEGORY OF PERSON' in df.columns:
        def map_weight(cat):
            # Attempt to match exact
            if cat in INSIDER_ROLE_WEIGHTS:
                return INSIDER_ROLE_WEIGHTS[cat]
            # Partial match
            for k, v in INSIDER_ROLE_WEIGHTS.items():
                if k != "-" and k.lower() in str(cat).lower():
                    return v
            return INSIDER_ROLE_WEIGHTS["-"]
            
        df['insider_role_weight'] = df['CATEGORY OF PERSON'].apply(map_weight)
        
    # Standardize company names slightly
    if 'COMPANY' in df.columns:
        df['COMPANY'] = df['COMPANY'].str.strip()
        
    # Remove rows where SYMBOL is "-"
    if 'SYMBOL' in df.columns:
        df = df[df['SYMBOL'] != "-"]
        
    # Deduplicate completely identical rows
    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    if before_dedup != len(df):
        logger.info(f"Removed {before_dedup - len(df)} duplicate rows.")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "insider_trades_clean.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} cleaned trades to {out_path}")
    
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clean_insider_trades()

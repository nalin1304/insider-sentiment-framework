"""
Macro data collector for the insider-sentiment-framework.

Fetches:
1. Historical FII/DII flow data
2. RBI Repo Rate changes

Currently implements stubs/placeholders since real APIs for these might require
paid access (e.g. NSE/BSE bulk download, or Moneycontrol scraping).
"""

import logging
import pandas as pd
from pathlib import Path

try:
    from config.settings import LOG_FILE, RAW_MACRO_DIR
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config.settings import LOG_FILE, RAW_MACRO_DIR

log = logging.getLogger("macro")

def collect_fii_dii_flows(save_dir: Path) -> pd.DataFrame:
    """
    Placeholder for FII/DII flows.
    """
    log.info("Collecting FII/DII flows (stub)")
    df = pd.DataFrame(columns=["date", "fii_buy", "fii_sell", "dii_buy", "dii_sell", "fii_net", "dii_net"])
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "fii_dii_flows.csv", index=False)
    return df

def collect_rbi_repo_rate(save_dir: Path) -> pd.DataFrame:
    """
    Placeholder for RBI Repo Rate.
    """
    log.info("Collecting RBI Repo Rate (stub)")
    df = pd.DataFrame(columns=["date", "repo_rate", "change_bps"])
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "rbi_repo_rate.csv", index=False)
    return df

def collect_all_macro_data(save_dir: Path = None):
    if save_dir is None:
        save_dir = RAW_MACRO_DIR
        
    save_dir = Path(save_dir)
    log.info("Starting Macro Data collection...")
    
    collect_fii_dii_flows(save_dir)
    collect_rbi_repo_rate(save_dir)
    
    log.info("Macro Data collection complete.")

def main():
    collect_all_macro_data()

if __name__ == "__main__":
    main()

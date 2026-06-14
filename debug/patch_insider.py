import re
from pathlib import Path

file_path = "data_collection/insider_trades.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Add checkpointing to collect_all_insider_trades
new_collect = """    if stocks_config is None:
        stocks_config = STOCKS

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    total = len(stocks_config)
    log.info("Starting insider trade collection for %d stocks.", total)
    
    # Simple Checkpointing: skip if file already exists with >0 bytes
    # To do a fresh run, the user should delete the raw folder.
"""
content = re.sub(
    r"    if stocks_config is None:\n        stocks_config = STOCKS\n\n    save_dir = Path\(save_dir\)\n    results: dict\[str, pd\.DataFrame\] = \{\}\n\n    total = len\(stocks_config\)\n    log\.info\(\"Starting insider trade collection for %d stocks\.\", total\)",
    new_collect,
    content,
    flags=re.DOTALL
)

new_loop = """        csv_path = save_dir / f"{stock_key}_insider_trades.csv"
        if csv_path.exists() and csv_path.stat().st_size > 50:
            log.info("--- [%d/%d] %s --- (SKIPPING: already exists)", idx, total, stock_key)
            try:
                results[stock_key] = pd.read_csv(csv_path)
            except Exception:
                pass
            continue

        log.info("--- [%d/%d] %s ---", idx, total, stock_key)

        try:
            df = collect_insider_trades_for_stock(stock_key, stock_info, save_dir)
            results[stock_key] = df
        except Exception:
            log.exception("Unexpected error collecting data for %s.", stock_key)
            results[stock_key] = pd.DataFrame(columns=OUTPUT_COLUMNS)

        # Per-scrip rate limit / polite delay
        if idx < total:
            delay = REQUEST_DELAY_SECONDS * 2 # Added explicit per-scrip delay
            log.debug("Waiting %.1fs before next stock to avoid per-scrip rate limits.", delay)
            time.sleep(delay)"""
content = re.sub(
    r"        log\.info\(\"--- \[%d/%d\] %s ---\", idx, total, stock_key\).*?time\.sleep\(REQUEST_DELAY_SECONDS\)",
    new_loop,
    content,
    flags=re.DOTALL
)

with open(file_path, "w") as f:
    f.write(content)

print("Patched insider_trades.py for checkpointing and per-scrip rate limit.")

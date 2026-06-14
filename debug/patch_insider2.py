import re

file_path = "data_collection/insider_trades.py"
with open(file_path, "r") as f:
    content = f.read()

# Replace collect_insider_trades_for_stock logic to include date pagination
new_collect_stock = """    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    scrip_code = stock_info["bse_scrip"]
    nse_ticker = stock_info["nse_ticker"]

    log.info(
        "Collecting insider trades for %s (BSE: %s, NSE: %s) with date pagination",
        stock_key, scrip_code, nse_ticker,
    )

    all_records = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=INSIDER_LOOKBACK_YEARS * 365)
    
    # Date Pagination: chunks of 365 days
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=365), end_date)
        
        bse_from = current_start.strftime("%d/%m/%Y")
        bse_to = current_end.strftime("%d/%m/%Y")
        nse_from = current_start.strftime("%d-%m-%Y")
        nse_to = current_end.strftime("%d-%m-%Y")

        log.debug("Fetching chunk: %s to %s", current_start.date(), current_end.date())
        
        # --- Attempt 1: BSE API ---
        records = fetch_bse_insider_trades(scrip_code, bse_from, bse_to)

        # --- Attempt 2: NSE API (fallback) ---
        if not records:
            time.sleep(REQUEST_DELAY_SECONDS)  # polite delay before fallback
            records = fetch_nse_insider_trades(nse_ticker, nse_from, nse_to)
            
        all_records.extend(records)
        
        current_start = current_end + timedelta(days=1)
        time.sleep(REQUEST_DELAY_SECONDS) # intra-scrip rate limiting between pages

    if not all_records:
        log.warning(
            "No insider trade data found for %s from either BSE or NSE.",
            stock_key,
        )
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Build DataFrame and enforce column order
    df = pd.DataFrame(all_records)

    # Deduplicate in case overlapping dates returned same records
    df = df.drop_duplicates()

    # Ensure all output columns exist (fill missing with None)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLUMNS]

    # Persist to CSV
    csv_path = save_dir / f"{stock_key}_insider_trades.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    log.info("Saved %d records to %s", len(df), csv_path)

    return df"""

content = re.sub(
    r"    save_dir = Path\(save_dir\)\n    save_dir\.mkdir\(parents=True, exist_ok=True\).*?return df",
    new_collect_stock,
    content,
    flags=re.DOTALL
)

with open(file_path, "w") as f:
    f.write(content)

print("Patched insider_trades.py for date pagination.")

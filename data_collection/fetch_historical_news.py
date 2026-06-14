import pandas as pd
from gnews import GNews
from datetime import datetime
import calendar
import time
import random
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

write_lock = threading.Lock()

# Paths
UNIVERSE_PATH = Path('config/universe.csv')
OUTPUT_PATH = Path('data/raw/news_sentiment_raw.csv')

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
]

def clean_company_name(name):
    """Clean the company name for better search results."""
    name = str(name).replace(' Limited', '').replace(' Ltd.', '').replace(' Ltd', '')
    name = name.replace(' Corporation', '').replace(' Corp', '')
    return name.strip()

def get_months_to_scrape(start_year=2019, end_year=2026):
    """Generate a list of (year, month, start_date, end_date) tuples."""
    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == datetime.now().year and month > datetime.now().month:
                break
            _, last_day = calendar.monthrange(year, month)
            start_date = (year, month, 1)
            end_date = (year, month, last_day)
            months.append((year, month, start_date, end_date))
    return months

def scrape_month(ticker, search_name, year, month, start_date, end_date):
    """Worker function to scrape a single month for a single ticker."""
    google_news = GNews(language='en', country='IN', max_results=50)
    # GNews doesn't expose headers easily, but setting proxy is possible if configured
    
    google_news.start_date = start_date
    google_news.end_date = end_date
    query = f'"{search_name}"'
    
    # Random sleep to avoid rapid spikes
    time.sleep(random.uniform(1.0, 5.0))
    
    try:
        articles = google_news.get_news(query)
        new_rows = []
        if articles:
            for a in articles:
                new_rows.append({
                    'Ticker': ticker,
                    'Company_Name': search_name,
                    'Date': a.get('published date'),
                    'Title': a.get('title'),
                    'Publisher': a.get('publisher', {}).get('title'),
                    'URL': a.get('url')
                })
        else:
            new_rows.append({
                'Ticker': ticker,
                'Company_Name': search_name,
                'Date': f"28 {calendar.month_abbr[month]} {year} 00:00:00 GMT",
                'Title': 'NO_NEWS_FOUND',
                'Publisher': 'NONE',
                'URL': 'NONE'
            })
        return new_rows
    except Exception as e:
        print(f" ERROR on {ticker} {year}-{month}: {e}")
        return []

def main():
    print("Loading universe...")
    universe = pd.read_csv(UNIVERSE_PATH)
    
    existing_records = set()
    if OUTPUT_PATH.exists():
        df_existing = pd.read_csv(OUTPUT_PATH)
        if not df_existing.empty:
            for _, row in df_existing.iterrows():
                try:
                    dt = pd.to_datetime(row['Date'])
                    existing_records.add((row['Ticker'], dt.year, dt.month))
                except:
                    pass
        print(f"Resuming from existing output file. Found data for {len(existing_records)} (ticker, month) combinations.")
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=['Ticker', 'Company_Name', 'Date', 'Title', 'Publisher', 'URL']).to_csv(OUTPUT_PATH, index=False)

    months_to_scrape = get_months_to_scrape(2019, 2026)
    
    # Prepare task queue
    tasks = []
    for idx, row in universe.iterrows():
        ticker = row['ticker']
        search_name = clean_company_name(row['company_name'])
        for year, month, start_date, end_date in months_to_scrape:
            if (ticker, year, month) not in existing_records:
                tasks.append((ticker, search_name, year, month, start_date, end_date))
                
    print(f"Total tasks to run: {len(tasks)}")
    
    # Run in parallel
    MAX_WORKERS = 5 # Keep low to avoid massive IP block, but faster than sequential
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(scrape_month, t, sn, y, m, sd, ed): (t, y, m) 
            for (t, sn, y, m, sd, ed) in tasks
        }
        
        for future in as_completed(future_to_task):
            ticker, year, month = future_to_task[future]
            try:
                results = future.result()
                if results:
                    with write_lock:
                        pd.DataFrame(results).to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
                completed += 1
                if completed % 50 == 0:
                    print(f"Progress: {completed}/{len(tasks)} tasks completed.")
            except Exception as exc:
                print(f"{ticker} {year}-{month} generated an exception: {exc}")

if __name__ == "__main__":
    main()

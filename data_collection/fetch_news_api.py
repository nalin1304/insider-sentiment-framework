import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Paths
UNIVERSE_PATH = Path('config/universe.csv')
OUTPUT_PATH = Path('data/raw/news_sentiment_raw_api.csv')

def fetch_eodhd_news(api_key, ticker, start_date='2019-01-01', end_date='2026-12-31', offset=0, limit=1000):
    """Fetch news from EODHD Financial News API."""
    url = f"https://eodhd.com/api/news"
    params = {
        'api_token': api_key,
        's': f"{ticker}.NS",  # EODHD uses .NS for NSE
        'from': start_date,
        'to': end_date,
        'limit': limit,
        'offset': offset,
        'fmt': 'json'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def main():
    load_dotenv()
    api_key = os.getenv('EODHD_API_KEY')
    
    if not api_key:
        print("ERROR: EODHD_API_KEY not found in .env file.")
        print("Please sign up at eodhd.com, get an API key, and add it to your .env file.")
        return

    print("Loading universe...")
    universe = pd.read_csv(UNIVERSE_PATH)
    
    existing_tickers = set()
    if OUTPUT_PATH.exists():
        df_existing = pd.read_csv(OUTPUT_PATH)
        if not df_existing.empty:
            existing_tickers = set(df_existing['Ticker'].unique())
        print(f"Resuming from existing output file. Found data for {len(existing_tickers)} tickers.")
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=['Ticker', 'Company_Name', 'Date', 'Title', 'Publisher', 'URL']).to_csv(OUTPUT_PATH, index=False)

    for idx, row in universe.iterrows():
        ticker = row['ticker']
        company_name = row['company_name']
        
        # We strip .NS for internal storage to match the master dataset
        clean_ticker = ticker.replace('.NS', '')
        
        if clean_ticker in existing_tickers:
            print(f"Skipping {clean_ticker} (already downloaded)")
            continue
            
        print(f"[{idx+1}/{len(universe)}] Fetching API news for {clean_ticker}...")
        
        offset = 0
        all_articles = []
        
        while True:
            try:
                articles = fetch_eodhd_news(api_key, clean_ticker, offset=offset)
                if not articles:
                    break
                    
                for a in articles:
                    all_articles.append({
                        'Ticker': clean_ticker,
                        'Company_Name': company_name,
                        'Date': a.get('date'),
                        'Title': a.get('title'),
                        'Publisher': a.get('source'), # The API returns the publisher under 'source'
                        'URL': a.get('link')
                    })
                
                # If we got less than the limit, we've reached the end
                if len(articles) < 1000:
                    break
                    
                offset += 1000
                time.sleep(0.5) # Be polite to the API
                
            except Exception as e:
                print(f"  Error fetching {clean_ticker} at offset {offset}: {e}")
                break
                
        if all_articles:
            pd.DataFrame(all_articles).to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
            print(f"  Saved {len(all_articles)} articles.")
        else:
            # Dummy record
            dummy = pd.DataFrame([{
                'Ticker': clean_ticker,
                'Company_Name': company_name,
                'Date': '2019-01-01T00:00:00+00:00',
                'Title': 'NO_NEWS_FOUND',
                'Publisher': 'NONE',
                'URL': 'NONE'
            }])
            dummy.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
            print("  0 articles found.")
            
if __name__ == "__main__":
    main()

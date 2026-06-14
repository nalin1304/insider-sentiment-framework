from gnews import GNews
import pandas as pd
from datetime import datetime
import time

def fetch_sample_news():
    print("Initializing GNews...")
    google_news = GNews(language='en', country='IN', start_date=(2024, 1, 1), end_date=(2024, 1, 31), max_results=20)
    
    print("Fetching news for Reliance Industries (Jan 2024)...")
    articles = google_news.get_news('Reliance Industries')
    
    if not articles:
        print("No articles found.")
        return
        
    data = []
    for a in articles:
        data.append({
            'Date': a.get('published date'),
            'Title': a.get('title'),
            'Publisher': a.get('publisher', {}).get('title')
        })
        
    df = pd.DataFrame(data)
    print(f"\nSuccessfully retrieved {len(df)} articles.")
    print("\nSample Output:")
    print(df.head(10).to_string())

if __name__ == "__main__":
    fetch_sample_news()

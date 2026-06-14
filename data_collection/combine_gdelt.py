import pandas as pd
from pathlib import Path

raw_news_dir = Path('data/raw/news')
output_file = Path('data/raw/news_sentiment_raw.csv')

all_dfs = []
for file in raw_news_dir.glob('*_news_gdelt.csv'):
    ticker = file.stem.replace('_news_gdelt', '')
    try:
        df = pd.read_csv(file)
        if not df.empty:
            df['Ticker'] = ticker
            # Ensure required columns are present. 'Date' needs to match what score_sentiment.py expects
            if 'date' in df.columns and 'Date' not in df.columns:
                df = df.rename(columns={'date': 'Date'})
            all_dfs.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"Combined {len(combined)} articles into {output_file}")
else:
    print("No articles found to combine.")

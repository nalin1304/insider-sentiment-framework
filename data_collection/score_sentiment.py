import pandas as pd
import numpy as np
import torch
from transformers import pipeline
import time
from pathlib import Path
import os

# Paths
INPUT_CSV = Path('data/raw/news_sentiment_raw_api.csv') # Defaults to the API one, change to news_sentiment_raw.csv if scraping
if not INPUT_CSV.exists():
    INPUT_CSV = Path('data/raw/news_sentiment_raw.csv')

OUTPUT_CSV = Path('data/processed/daily_sentiment.csv')
TEMP_OUTPUT = Path('data/processed/daily_sentiment_temp.csv')

def setup_pipeline():
    """Initialize the FinBERT pipeline with hardware acceleration if available."""
    device = -1
    
    # Check for Apple Silicon (MPS) or NVIDIA (CUDA)
    if torch.backends.mps.is_available():
        device = "mps"
        print("Hardware Acceleration: Apple Silicon (MPS) detected.")
    elif torch.cuda.is_available():
        device = 0
        print("Hardware Acceleration: NVIDIA CUDA detected.")
    else:
        print("Hardware Acceleration: CPU only.")

    print("Loading ProsusAI/finbert model...")
    # Initialize pipeline, requesting all scores to compute continuous sentiment
    sentiment_pipeline = pipeline(
        "text-classification", 
        model="ProsusAI/finbert", 
        device=device,
        top_k=None # Get probabilities for all classes (positive, negative, neutral)
    )
    return sentiment_pipeline

def compute_continuous_score(predictions):
    """
    Computes a continuous score from FinBERT probability distributions.
    Score = P(positive) - P(negative)
    """
    scores = []
    for pred in predictions:
        # pred is a list of dicts like [{'label': 'positive', 'score': 0.8}, ...]
        pos_score = 0.0
        neg_score = 0.0
        for p in pred:
            if p['label'] == 'positive':
                pos_score = p['score']
            elif p['label'] == 'negative':
                neg_score = p['score']
        
        # Continuous sentiment magnitude [-1, 1]
        magnitude = pos_score - neg_score
        scores.append(magnitude)
    return scores

def main():
    if not INPUT_CSV.exists():
        print(f"Input file {INPUT_CSV} not found. Please run data collection first.")
        return

    print("Loading raw news data...")
    df = pd.read_csv(INPUT_CSV)
    
    # Filter out dummy rows
    df = df[df['Title'] != 'NO_NEWS_FOUND'].copy()
    if df.empty:
        print("No actual news articles found in the input CSV.")
        return
        
    print(f"Total articles to score: {len(df)}")
    
    # Check for existing progress
    scored_indices = set()
    if TEMP_OUTPUT.exists():
        temp_df = pd.read_csv(TEMP_OUTPUT)
        scored_indices = set(temp_df.index.tolist()) # Assuming index aligns
        print(f"Resuming from checkpoint: {len(scored_indices)} articles already scored.")
        df = df.drop(index=list(scored_indices))
        print(f"Remaining articles to score: {len(df)}")

    if df.empty:
        print("All articles scored. Finalizing output...")
        finalize_output()
        return

    nlp = setup_pipeline()
    
    # Batch processing
    batch_size = 64
    titles = df['Title'].tolist()
    original_indices = df.index.tolist()
    
    all_scores = []
    
    print("Beginning batch inference...")
    start_time = time.time()
    
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i+batch_size]
        batch_indices = original_indices[i:i+batch_size]
        
        # Inference
        predictions = nlp(batch_titles)
        scores = compute_continuous_score(predictions)
        
        # Save checkpoint
        batch_results = df.loc[batch_indices].copy()
        batch_results['sentiment_score'] = scores
        
        # Append to temp file
        header = not TEMP_OUTPUT.exists()
        batch_results.to_csv(TEMP_OUTPUT, mode='a', header=header, index=False)
        
        if i % (batch_size * 10) == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            print(f"Processed {i}/{len(titles)} | Rate: {rate:.1f} articles/sec | ETA: {(len(titles)-i)/rate:.0f}s")
            
    print("Inference complete! Finalizing output...")
    finalize_output()

def finalize_output():
    """Aggregates the scored articles into daily metrics per ticker."""
    df = pd.read_csv(TEMP_OUTPUT)
    
    # Parse dates to daily level
    # Dates might have timezone info or "GMT"
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce').dt.tz_localize(None).dt.floor('D')
    df = df.dropna(subset=['Date', 'Ticker', 'sentiment_score'])
    
    # Group by Ticker and Date
    daily = df.groupby(['Ticker', 'Date']).agg(
        daily_sentiment_score=('sentiment_score', 'mean'),
        news_volume=('sentiment_score', 'count')
    ).reset_index()
    
    daily = daily.sort_values(['Ticker', 'Date'])
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved daily aggregated sentiment to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

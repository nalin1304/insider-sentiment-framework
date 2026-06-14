import pandas as pd
from transformers import pipeline
import torch
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_PATH = Path('data/raw/news_sentiment_raw.csv')
OUTPUT_PATH = Path('data/processed/news_scored.csv')

def get_device():
    if torch.cuda.is_available():
        return 0 # CUDA device index
    elif torch.backends.mps.is_available():
        return "mps"
    return -1 # CPU

def score_news(batch_size=256):
    if not INPUT_PATH.exists():
        logger.error(f"Input file not found at {INPUT_PATH}")
        return

    logger.info(f"Loading raw news from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # Drop rows without news
    df = df[df['Title'] != 'NO_NEWS_FOUND'].copy()
    
    if len(df) == 0:
        logger.warning("No valid news articles found to score.")
        return
        
    df = df.dropna(subset=['Title', 'Date'])
    logger.info(f"Loaded {len(df)} valid news articles for scoring.")
    
    # Check if already scored
    if OUTPUT_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_PATH)
        # Deduplicate using URL or Title+Date
        if 'URL' in existing_df.columns and 'URL' in df.columns:
            existing_urls = set(existing_df['URL'].dropna())
            df_to_score = df[~df['URL'].isin(existing_urls)].copy()
        else:
            df_to_score = df
            
        logger.info(f"Found {len(existing_df)} existing records. Scoring {len(df_to_score)} new records.")
    else:
        df_to_score = df
        existing_df = pd.DataFrame()
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
    if len(df_to_score) == 0:
        logger.info("Everything is already scored. Exiting.")
        return

    device = get_device()
    logger.info(f"Initializing ProsusAI/finbert pipeline on device: {device}")
    
    # Load model
    # transformers < 4.30 uses return_all_scores, newer uses top_k=None
    try:
        sentiment_pipeline = pipeline(
            "text-classification", 
            model="ProsusAI/finbert", 
            tokenizer="ProsusAI/finbert",
            device=device,
            top_k=None # Get pos/neg/neu probabilities
        )
    except Exception:
        sentiment_pipeline = pipeline(
            "text-classification", 
            model="ProsusAI/finbert", 
            tokenizer="ProsusAI/finbert",
            device=device,
            return_all_scores=True
        )
    
    titles = df_to_score['Title'].astype(str).tolist()
    
    logger.info(f"Scoring {len(titles)} headlines in batches of {batch_size}...")
    
    all_results = []
    
    # Use pipeline iterator for memory efficiency and speed
    for out in tqdm(sentiment_pipeline(titles, batch_size=batch_size, truncation=True, max_length=512), total=len(titles)):
        # out is a list of dicts: [{'label': 'positive', 'score': 0.1}, ...]
        score_dict = {d['label']: d['score'] for d in out}
        all_results.append(score_dict)
        
    # Build dataframe columns
    df_to_score['pos_score'] = [r.get('positive', 0.0) for r in all_results]
    df_to_score['neg_score'] = [r.get('negative', 0.0) for r in all_results]
    df_to_score['neu_score'] = [r.get('neutral', 0.0) for r in all_results]
    
    # Compute composite scalar score (positive - negative)
    df_to_score['sentiment_score'] = df_to_score['pos_score'] - df_to_score['neg_score']
    
    # Combine with existing and save
    if not existing_df.empty:
        final_df = pd.concat([existing_df, df_to_score], ignore_index=True)
    else:
        final_df = df_to_score
        
    logger.info(f"Saving {len(final_df)} scored records to {OUTPUT_PATH}")
    final_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    args = parser.parse_args()
    
    score_news(batch_size=args.batch_size)

#!/bin/bash
set -e

export PYTHONPATH=.

echo "1. Collecting news data..."
python3 data_collection/news.py

echo "2. Combining scraped news..."
python3 data_collection/combine_gdelt.py

echo "3. Scoring sentiment..."
python3 data_collection/score_sentiment.py

echo "4. Generating sentiment features..."
python3 features/sentiment_features.py

echo "5. Running downstream pipeline..."
bash scratch/run_pipeline_rest.sh

echo "Sentiment Pipeline Complete!"

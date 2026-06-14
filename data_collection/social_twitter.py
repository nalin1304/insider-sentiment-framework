"""
Twitter/X social sentiment collector.

Note: Twitter/X API access is currently excluded from this framework due to 
high API costs. This module serves as a well-documented stub ready to wire
in if a paid key is later provided.

Future Implementation:
- Should use `tweepy` or the official X API v2 endpoints.
- Require API Key, API Secret, Bearer Token in .env.
- Fetch cashtags (e.g. $RELIANCE) and keyword aliases.
"""

import logging
from pathlib import Path

logger = logging.getLogger("social_twitter")

def collect_twitter_sentiment(save_dir: Path = None):
    logger.info("Twitter collection is disabled due to API costs. Stub called.")
    return True

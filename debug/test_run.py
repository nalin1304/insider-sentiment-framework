import sys
import logging
import run_collection

# Patch STOCKS to only contain 2 for quick testing
run_collection.STOCKS = {
    "RELIANCE": run_collection.STOCKS["RELIANCE"],
    "TCS": run_collection.STOCKS["TCS"]
}

# Also patch INDICES to just 1 to save time
from config import settings
settings.INDICES = {
    "^BSESN": "BSE SENSEX",
}

sys.argv = ["run_collection.py", "--prices", "--insider", "--corporate", "--macro"]
run_collection.main()

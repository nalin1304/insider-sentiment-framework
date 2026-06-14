import re

with open('README.md') as f:
    content = f.read()

status_section = """
## Current Status

- **Universe**: 396 NSE-listed tickers selected by having >=75 insider filings over 7 years with available price history.
- **Date Range**: 2019-06-17 to 2026-06-12 (7 years of daily data).
- **Implemented**: Insider trading features (original, role-split, relative-size, clustering), macro features, LightGBM panel models, event study analysis, portfolio tilt backtest.
- **Not Implemented**: News sentiment features, social media sentiment features. These were planned but disabled after source data quality issues were identified.
- **Key Finding**: Core insider buy disclosures are followed by a statistically significant ~1% abnormal price drift over 20 trading days. However, this effect is economically marginal (0.08 SD) and does not survive transaction costs when deployed as a portfolio tilt strategy. See `FINAL_REPORT.md` for details.
"""

# Try to find a good place to insert this. After the first heading and description.
# A regex to match the first block of text
new_content = re.sub(
    r'(# Insider Sentiment Framework\n\n[^\n]+\n+)',
    r'\1' + status_section + '\n',
    content,
    count=1
)

with open('README.md', 'w') as f:
    f.write(new_content)

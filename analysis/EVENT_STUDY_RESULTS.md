# Event Study Results
This report summarizes the Cumulative Abnormal Return (CAR) analysis for insider disclosure events.
Windows are evaluated from t-1 (one day prior to disclosure) to t+W, to capture immediate pre-disclosure leakage and post-disclosure drift.

## All Insiders (Pooled)
**Events**: 17871 Buys, 28673 Sells

### Window [-1, +1]
- **Buys**: Mean CAR = 0.34% (SE: 0.04%), p-value = 0.0000
- **Sells**: Mean CAR = 0.08% (SE: 0.03%), p-value = 0.0038
- **Buy - Sell Diff**: p-value = 0.0000

### Window [-1, +5]
- **Buys**: Mean CAR = 0.48% (SE: 0.05%), p-value = 0.0000
- **Sells**: Mean CAR = -0.15% (SE: 0.05%), p-value = 0.0024
- **Buy - Sell Diff**: p-value = 0.0000

### Window [-1, +10]
- **Buys**: Mean CAR = 0.53% (SE: 0.07%), p-value = 0.0000
- **Sells**: Mean CAR = -0.39% (SE: 0.06%), p-value = 0.0000
- **Buy - Sell Diff**: p-value = 0.0000

### Window [-1, +20]
- **Buys**: Mean CAR = 0.36% (SE: 0.09%), p-value = 0.0001
- **Sells**: Mean CAR = -0.93% (SE: 0.07%), p-value = 0.0000
- **Buy - Sell Diff**: p-value = 0.0000

## Core Insiders Only
**Events**: 9015 Buys, 6122 Sells

### Window [-1, +1]
- **Buys**: Mean CAR = 0.60% (SE: 0.05%), p-value = 0.0000
- **Sells**: Mean CAR = -0.08% (SE: 0.06%), p-value = 0.2124
- **Buy - Sell Diff**: p-value = 0.0000

### Window [-1, +5]
- **Buys**: Mean CAR = 0.94% (SE: 0.08%), p-value = 0.0000
- **Sells**: Mean CAR = -0.10% (SE: 0.18%), p-value = 0.5596
- **Buy - Sell Diff**: p-value = 0.0000

### Window [-1, +10]
- **Buys**: Mean CAR = 1.19% (SE: 0.11%), p-value = 0.0000
- **Sells**: Mean CAR = -0.28% (SE: 0.19%), p-value = 0.1392
- **Buy - Sell Diff**: p-value = 0.0000

### Window [-1, +20]
- **Buys**: Mean CAR = 1.28% (SE: 0.14%), p-value = 0.0000
- **Sells**: Mean CAR = -0.55% (SE: 0.22%), p-value = 0.0138
- **Buy - Sell Diff**: p-value = 0.0000

## Large Trades Only (Top Tercile % Stake Change)
**Events**: 5508 Buys, 8371 Sells

### Window [-1, +1]
- **Buys**: Mean CAR = 0.08% (SE: 0.06%), p-value = 0.1971
- **Sells**: Mean CAR = 0.25% (SE: 0.05%), p-value = 0.0000
- **Buy - Sell Diff**: p-value = 0.0253

### Window [-1, +5]
- **Buys**: Mean CAR = 0.06% (SE: 0.09%), p-value = 0.5203
- **Sells**: Mean CAR = -0.03% (SE: 0.07%), p-value = 0.6713
- **Buy - Sell Diff**: p-value = 0.4430

### Window [-1, +10]
- **Buys**: Mean CAR = -0.06% (SE: 0.11%), p-value = 0.6091
- **Sells**: Mean CAR = -0.42% (SE: 0.09%), p-value = 0.0000
- **Buy - Sell Diff**: p-value = 0.0113

### Window [-1, +20]
- **Buys**: Mean CAR = -0.34% (SE: 0.15%), p-value = 0.0274
- **Sells**: Mean CAR = -1.05% (SE: 0.12%), p-value = 0.0000
- **Buy - Sell Diff**: p-value = 0.0003

## Sector Breakdown (Window [-1, +20])

- **Basic Materials**: Buys CAR_20 = -0.80%, Sells CAR_20 = -0.71%
- **Communication Services**: Buys CAR_20 = 2.03%, Sells CAR_20 = 2.29%
- **Consumer Cyclical**: Buys CAR_20 = 0.87%, Sells CAR_20 = -1.16%
- **Consumer Defensive**: Buys CAR_20 = 1.90%, Sells CAR_20 = -0.03%
- **Energy**: Buys CAR_20 = 0.08%, Sells CAR_20 = -0.90%
- **Financial Services**: Buys CAR_20 = -0.03%, Sells CAR_20 = -1.39%
- **Healthcare**: Buys CAR_20 = 0.58%, Sells CAR_20 = -0.80%
- **Industrials**: Buys CAR_20 = 0.69%, Sells CAR_20 = -1.66%
- **Real Estate**: Buys CAR_20 = 4.49%, Sells CAR_20 = -1.59%
- **Technology**: Buys CAR_20 = -0.31%, Sells CAR_20 = -1.22%
- **Unknown**: Buys CAR_20 = 0.78%, Sells CAR_20 = 3.89%
- **Utilities**: Buys CAR_20 = -0.99%, Sells CAR_20 = 0.45%
# REAL NSE 200-DMA SCANNER — PHASE 1

## Run this version

You already installed `yfinance`.

In Terminal, from the Setup folder:

```bash
python3 scanner_nse500.py
```

It will:
1. Download the current Nifty 500 constituent list from NSE.
2. Download 2 years of daily EOD data from Yahoo Finance.
3. Calculate 200 SMA, 50 SMA, RSI(14), and volume ratio.
4. Detect stocks that crossed below 200 DMA in the last 90 trading days.
5. Keep stocks currently within +/-3% of 200 DMA.
6. Score and rank the candidates.
7. Create:
   - `output/NSE_200DMA_Recovery_Scanner.xlsx`
   - `output/NSE_200DMA_Recovery_Scanner.csv`
   - `output/download_failures.csv` if any symbols fail.

## Important
- This version does NOT use Zerodha and does NOT place orders.
- Yahoo Finance is third-party EOD data, so verify any candidate in Kite before acting.
- The first full scan may take several minutes because it downloads hundreds of symbols.
- If Yahoo temporarily rate-limits requests, run the scanner again later; the script records failures.

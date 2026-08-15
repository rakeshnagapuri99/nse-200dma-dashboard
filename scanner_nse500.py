#!/usr/bin/env python3
"""
PHASE 1 — REAL NSE 200-DMA RECOVERY SCANNER
Uses free Yahoo Finance EOD data and the official NSE Nifty 500 constituent CSV.

Strategy:
1. Stock must have crossed from ABOVE its 200-SMA to BELOW it within the last 90 trading days.
2. Current price must be within +/-3% of the current 200-SMA.
3. Score recovery using distance to 200-SMA, recovery from the post-breakdown low,
   50-SMA, RSI(14), and volume ratio.

No Zerodha API and no orders are used.

Data caveat:
Yahoo Finance is a third-party EOD data source. Treat this as a screening/research
tool and verify candidates in Kite before trading.
"""

from pathlib import Path
import time
import warnings
import pandas as pd
import numpy as np
import requests
import yfinance as yf

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

PROXIMITY_PCT = 3.0
BREAK_LOOKBACK = 90
MIN_DAYS = 220
CHUNK_SIZE = 40
PAUSE_SECONDS = 1.0

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def get_nifty500_symbols():
    local = DATA_DIR / "nifty500_symbols.csv"
    try:
        # NSE can reject requests without browser-like headers.
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,*/*;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }
        r = requests.get(NIFTY500_URL, headers=headers, timeout=20)
        r.raise_for_status()
        local.write_bytes(r.content)
        df = pd.read_csv(local)
    except Exception as e:
        if local.exists():
            print(f"Using cached Nifty 500 list because NSE download failed: {e}")
            df = pd.read_csv(local)
        else:
            raise RuntimeError(
                "Could not download the Nifty 500 constituent list from NSE. "
                "Check your internet connection and run again."
            ) from e

    symbol_col = next((c for c in df.columns if c.strip().upper() == "SYMBOL"), None)
    if not symbol_col:
        raise RuntimeError("Nifty 500 CSV did not contain a SYMBOL column.")

    symbols = (
        df[symbol_col].astype(str).str.strip()
        .str.replace("&", "%26", regex=False)
        .tolist()
    )
    return [s for s in symbols if s and s.lower() != "nan"]

def clean_yf_frame(df, ticker):
    if df is None or df.empty:
        return None

    # yfinance can return MultiIndex columns for multi-ticker downloads.
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)

    df = df.copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    if "close" not in df.columns or "volume" not in df.columns:
        return None

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.reset_index()
    date_col = "date" if "date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df = df.sort_values("date").drop_duplicates("date")
    return df.dropna(subset=["close"])

def analyse(symbol, df):
    if df is None or len(df) < MIN_DAYS:
        return None

    df = df.copy()
    df["dma_200"] = df["close"].rolling(200).mean()
    df["dma_50"] = df["close"].rolling(50).mean()
    df["rsi_14"] = rsi(df["close"], 14)
    df["vol_avg_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["vol_avg_20"]

    cur = df.iloc[-1]
    if pd.isna(cur["dma_200"]):
        return None

    proximity = (cur["close"] / cur["dma_200"] - 1) * 100
    if abs(proximity) > PROXIMITY_PCT:
        return None

    cross = (
        (df["close"].shift(1) > df["dma_200"].shift(1))
        & (df["close"] <= df["dma_200"])
    )
    cross_candidates = df.loc[cross].tail(BREAK_LOOKBACK)
    if cross_candidates.empty:
        return None

    last_cross_idx = cross_candidates.index[-1]
    pos = df.index.get_loc(last_cross_idx)
    after = df.iloc[pos:]

    low_col = "low" if "low" in after.columns else "close"
    breakdown_low = after[low_col].min()
    recovery_pct = (cur["close"] / breakdown_low - 1) * 100

    # Is the price getting closer to the 200 DMA than it was 10 sessions ago?
    prev = df.iloc[-10]
    improving = (
        not pd.isna(prev["dma_200"])
        and abs(cur["close"]/cur["dma_200"] - 1)
        < abs(prev["close"]/prev["dma_200"] - 1)
    )

    score = 0
    reasons = []

    if proximity >= -1:
        score += 25
        reasons.append("at/above 200DMA")
    else:
        score += 15
        reasons.append("within 3% of 200DMA")

    if improving:
        score += 20
        reasons.append("moving toward 200DMA")

    if recovery_pct >= 10:
        score += 25
        reasons.append("recovery >=10%")
    elif recovery_pct >= 5:
        score += 20
        reasons.append("recovery >=5%")
    elif recovery_pct > 0:
        score += 10
        reasons.append("above breakdown low")

    if not pd.isna(cur["dma_50"]) and cur["close"] > cur["dma_50"]:
        score += 15
        reasons.append("above 50DMA")

    if not pd.isna(cur["volume_ratio"]) and cur["volume_ratio"] >= 1.2:
        score += 10
        reasons.append("volume >=1.2x")

    if not pd.isna(cur["rsi_14"]) and cur["rsi_14"] >= 50:
        score += 10
        reasons.append("RSI >=50")

    if score >= 75:
        setup = "STRONG RETEST"
    elif score >= 55:
        setup = "RECOVERY WATCH"
    else:
        setup = "NEAR 200DMA"

    return {
        "symbol": symbol,
        "date": cur["date"].date(),
        "close": round(float(cur["close"]), 2),
        "dma_200": round(float(cur["dma_200"]), 2),
        "distance_to_200dma_pct": round(float(proximity), 2),
        "dma_50": round(float(cur["dma_50"]), 2),
        "rsi_14": round(float(cur["rsi_14"]), 2),
        "volume_ratio": round(float(cur["volume_ratio"]), 2),
        "break_below_date": df.loc[last_cross_idx, "date"].date(),
        "breakdown_low": round(float(breakdown_low), 2),
        "recovery_from_breakdown_low_pct": round(float(recovery_pct), 2),
        "score": int(score),
        "setup": setup,
        "reasons": "; ".join(reasons),
    }

def download_one(symbol):
    ticker = symbol + ".NS"
    try:
        df = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )
        return clean_yf_frame(df, ticker)
    except Exception as e:
        print(f"  ! {symbol}: {e}")
        return None

def main():
    print("\n=== PHASE 1 — NSE 200 DMA RECOVERY SCANNER ===\n")
    symbols = get_nifty500_symbols()
    print(f"Nifty 500 universe loaded: {len(symbols)} stocks")
    print("Downloading 2 years of daily EOD data from Yahoo Finance...")
    print("This may take several minutes.\n")

    results = []
    failed = []

    for i, symbol in enumerate(symbols, 1):
        if i % 10 == 1:
            print(f"Progress: {i}/{len(symbols)}")

        df = download_one(symbol)
        if df is None:
            failed.append(symbol)
        else:
            row = analyse(symbol, df)
            if row:
                results.append(row)

        time.sleep(PAUSE_SECONDS)

    out = pd.DataFrame(results)

    if out.empty:
        print("\nNo stocks matched the current setup.")
    else:
        out = out.sort_values(
            ["score", "distance_to_200dma_pct"],
            ascending=[False, True]
        ).reset_index(drop=True)

        out.insert(0, "rank", range(1, len(out) + 1))

        csv_path = OUTPUT_DIR / "NSE_200DMA_Recovery_Scanner.csv"
        xlsx_path = OUTPUT_DIR / "NSE_200DMA_Recovery_Scanner.xlsx"

        out.to_csv(csv_path, index=False)
        out.to_excel(xlsx_path, index=False)

        print("\n=== CANDIDATES ===\n")
        print(out.to_string(index=False))
        print(f"\nSaved: {csv_path}")
        print(f"Saved: {xlsx_path}")

    if failed:
        pd.DataFrame({"symbol": failed}).to_csv(
            OUTPUT_DIR / "download_failures.csv", index=False
        )
        print(f"\nData unavailable for {len(failed)} symbols.")
        print("See output/download_failures.csv")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

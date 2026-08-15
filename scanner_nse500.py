#!/usr/bin/env python3

"""
PHASE 1 — NSE 200 DMA RECOVERY SCANNER

Optimized version:
- Uses official NSE Nifty 500 constituent list
- Downloads Yahoo Finance data in parallel batches
- Calculates 200 DMA, 50 DMA, RSI, volume ratio and recovery
- Produces CSV and Excel output
- No Zerodha API and no orders are used
"""

from pathlib import Path
import time
import warnings

import pandas as pd
import numpy as np
import requests
import yfinance as yf


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

NIFTY500_URL = (
    "https://archives.nseindia.com/"
    "content/indices/ind_nifty500list.csv"
)

PROXIMITY_PCT = 3.0
BREAK_LOOKBACK = 90
MIN_DAYS = 220

# Download 50 stocks concurrently at a time.
CHUNK_SIZE = 50

# Small pause between batches.
BATCH_PAUSE = 0.5


# =========================================================
# RSI
# =========================================================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    return 100 - (100 / (1 + rs))


# =========================================================
# NSE NIFTY 500 LIST
# =========================================================

def get_nifty500_symbols():

    local = DATA_DIR / "nifty500_symbols.csv"

    try:

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,*/*;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }

        response = requests.get(
            NIFTY500_URL,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        local.write_bytes(
            response.content
        )

        df = pd.read_csv(local)

    except Exception as e:

        if local.exists():

            print(
                "Using cached Nifty 500 list "
                f"because NSE download failed: {e}"
            )

            df = pd.read_csv(local)

        else:

            raise RuntimeError(
                "Could not download the Nifty 500 "
                "constituent list from NSE."
            ) from e

    symbol_col = next(
        (
            c for c in df.columns
            if c.strip().upper() == "SYMBOL"
        ),
        None
    )

    if not symbol_col:

        raise RuntimeError(
            "Nifty 500 CSV did not contain "
            "a SYMBOL column."
        )

    symbols = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .tolist()
    )

    symbols = [
        s for s in symbols
        if s and s.lower() != "nan"
    ]

    return symbols


# =========================================================
# CLEAN YAHOO DATA
# =========================================================

def clean_yf_frame(df, ticker):

    if df is None or df.empty:
        return None

    df = df.copy()

    # Handle MultiIndex returned by batch download.
    if isinstance(df.columns, pd.MultiIndex):

        try:

            if ticker in df.columns.get_level_values(0):

                df = df[ticker]

            elif ticker in df.columns.get_level_values(-1):

                df = df.xs(
                    ticker,
                    axis=1,
                    level=-1
                )

            else:

                return None

        except Exception:

            return None

    df.columns = [
        str(c)
        .lower()
        .replace(" ", "_")
        for c in df.columns
    ]

    required = [
        "close",
        "volume"
    ]

    if not all(
        c in df.columns
        for c in required
    ):

        return None

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        if c in df.columns:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    df = df.reset_index()

    date_col = (
        "date"
        if "date" in df.columns
        else df.columns[0]
    )

    df["date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    # Remove timezone safely.
    try:

        if df["date"].dt.tz is not None:

            df["date"] = (
                df["date"]
                .dt.tz_localize(None)
            )

    except Exception:
        pass

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date")
    )

    df = df.dropna(
        subset=["close"]
    )

    return df


# =========================================================
# ANALYSE STOCK
# =========================================================

def analyse(symbol, df):

    if df is None:
        return None

    if len(df) < MIN_DAYS:
        return None

    df = df.copy()

    # -----------------------------------------
    # Technical indicators
    # -----------------------------------------

    df["dma_200"] = (
        df["close"]
        .rolling(200)
        .mean()
    )

    df["dma_50"] = (
        df["close"]
        .rolling(50)
        .mean()
    )

    df["rsi_14"] = rsi(
        df["close"],
        14
    )

    df["vol_avg_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["vol_avg_20"]
    )

    # -----------------------------------------
    # Current values
    # -----------------------------------------

    cur = df.iloc[-1]

    if pd.isna(cur["dma_200"]):

        return None

    proximity = (
        cur["close"] /
        cur["dma_200"] -
        1
    ) * 100

    # -----------------------------------------
    # Must be within +/- 3%
    # -----------------------------------------

    if abs(proximity) > PROXIMITY_PCT:

        return None

    # -----------------------------------------
    # Find cross from above 200 DMA
    # -----------------------------------------

    cross = (

        (
            df["close"].shift(1)
            >
            df["dma_200"].shift(1)
        )

        &

        (
            df["close"]
            <=
            df["dma_200"]
        )

    )

    cross_candidates = (
        df.loc[cross]
        .tail(BREAK_LOOKBACK)
    )

    if cross_candidates.empty:

        return None

    last_cross_idx = (
        cross_candidates
        .index[-1]
    )

    pos = df.index.get_loc(
        last_cross_idx
    )

    after = df.iloc[pos:]

    low_col = (
        "low"
        if "low" in after.columns
        else "close"
    )

    breakdown_low = (
        after[low_col]
        .min()
    )

    recovery_pct = (
        cur["close"] /
        breakdown_low -
        1
    ) * 100

    # -----------------------------------------
    # Is price moving toward 200 DMA?
    # -----------------------------------------

    if len(df) >= 10:

        prev = df.iloc[-10]

        improving = (

            not pd.isna(
                prev["dma_200"]
            )

            and

            abs(
                cur["close"] /
                cur["dma_200"] -
                1
            )

            <

            abs(
                prev["close"] /
                prev["dma_200"] -
                1
            )

        )

    else:

        improving = False

    # -----------------------------------------
    # SCORE
    # -----------------------------------------

    score = 0

    reasons = []

    # Distance
    if proximity >= -1:

        score += 25

        reasons.append(
            "at/above 200DMA"
        )

    else:

        score += 15

        reasons.append(
            "within 3% of 200DMA"
        )

    # Improving
    if improving:

        score += 20

        reasons.append(
            "moving toward 200DMA"
        )

    # Recovery
    if recovery_pct >= 10:

        score += 25

        reasons.append(
            "recovery >=10%"
        )

    elif recovery_pct >= 5:

        score += 20

        reasons.append(
            "recovery >=5%"
        )

    elif recovery_pct > 0:

        score += 10

        reasons.append(
            "above breakdown low"
        )

    # 50 DMA
    if (

        not pd.isna(
            cur["dma_50"]
        )

        and

        cur["close"]
        >
        cur["dma_50"]

    ):

        score += 15

        reasons.append(
            "above 50DMA"
        )

    # Volume
    if (

        not pd.isna(
            cur["volume_ratio"]
        )

        and

        cur["volume_ratio"]
        >= 1.2

    ):

        score += 10

        reasons.append(
            "volume >=1.2x"
        )

    # RSI
    if (

        not pd.isna(
            cur["rsi_14"]
        )

        and

        cur["rsi_14"]
        >= 50

    ):

        score += 10

        reasons.append(
            "RSI >=50"
        )

    # -----------------------------------------
    # SETUP
    # -----------------------------------------

    if score >= 75:

        setup = "STRONG RETEST"

    elif score >= 55:

        setup = "RECOVERY WATCH"

    else:

        setup = "NEAR 200DMA"

    # -----------------------------------------
    # RESULT
    # -----------------------------------------

    return {

        "symbol": symbol,

        "date":
            cur["date"].date(),

        "close":
            round(
                float(cur["close"]),
                2
            ),

        "dma_200":
            round(
                float(cur["dma_200"]),
                2
            ),

        "distance_to_200dma_pct":
            round(
                float(proximity),
                2
            ),

        "dma_50":
            round(
                float(cur["dma_50"]),
                2
            ),

        "rsi_14":
            round(
                float(cur["rsi_14"]),
                2
            ),

        "volume_ratio":
            round(
                float(cur["volume_ratio"]),
                2
            ),

        "break_below_date":
            df.loc[
                last_cross_idx,
                "date"
            ].date(),

        "breakdown_low":
            round(
                float(breakdown_low),
                2
            ),

        "recovery_from_breakdown_low_pct":
            round(
                float(recovery_pct),
                2
            ),

        "score":
            int(score),

        "setup":
            setup,

        "reasons":
            "; ".join(reasons)

    }


# =========================================================
# BATCH DOWNLOAD
# =========================================================

def download_batch(symbols):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    print(
        f"Downloading batch of "
        f"{len(tickers)} stocks..."
    )

    try:

        data = yf.download(

            tickers=tickers,

            period="2y",

            interval="1d",

            auto_adjust=False,

            progress=False,

            threads=True,

            group_by="ticker",

            timeout=30

        )

        return data

    except Exception as e:

        print(
            f"Batch download failed: {e}"
        )

        return None


# =========================================================
# MAIN
# =========================================================

def main():

    start_time = time.time()

    print("")
    print(
        "=== FAST NSE 200 DMA "
        "RECOVERY SCANNER ==="
    )
    print("")

    symbols = (
        get_nifty500_symbols()
    )

    print(
        f"Nifty 500 universe loaded: "
        f"{len(symbols)} stocks"
    )

    print(
        f"Downloading in batches of "
        f"{CHUNK_SIZE}..."
    )

    print("")

    results = []

    failed = []

    total_batches = int(
        np.ceil(
            len(symbols) /
            CHUNK_SIZE
        )
    )

    for batch_number, start in enumerate(

        range(
            0,
            len(symbols),
            CHUNK_SIZE
        ),

        1

    ):

        batch_symbols = symbols[
            start:
            start + CHUNK_SIZE
        ]

        print(
            f"[Batch {batch_number}/"
            f"{total_batches}]"
        )

        data = download_batch(
            batch_symbols
        )

        if data is None:

            # ---------------------------------
            # Retry batch individually
            # ---------------------------------

            print(
                "Batch failed. "
                "Retrying stocks individually..."
            )

            for symbol in batch_symbols:

                try:

                    ticker = (
                        symbol + ".NS"
                    )

                    df = yf.download(

                        ticker,

                        period="2y",

                        interval="1d",

                        auto_adjust=False,

                        progress=False,

                        threads=False,

                        timeout=20

                    )

                    clean = (
                        clean_yf_frame(
                            df,
                            ticker
                        )
                    )

                    if clean is None:

                        failed.append(
                            symbol
                        )

                    else:

                        row = analyse(
                            symbol,
                            clean
                        )

                        if row:

                            results.append(
                                row
                            )

                except Exception:

                    failed.append(
                        symbol
                    )

            continue

        # -------------------------------------
        # Analyse each stock in batch
        # -------------------------------------

        for symbol in batch_symbols:

            ticker = (
                symbol + ".NS"
            )

            try:

                clean = (
                    clean_yf_frame(
                        data,
                        ticker
                    )
                )

                if clean is None:

                    failed.append(
                        symbol
                    )

                    continue

                row = analyse(
                    symbol,
                    clean
                )

                if row:

                    results.append(
                        row
                    )

            except Exception as e:

                print(
                    f"  ! {symbol}: {e}"
                )

                failed.append(
                    symbol
                )

        # Small pause between batches
        time.sleep(
            BATCH_PAUSE
        )

    # =====================================================
    # OUTPUT
    # =====================================================

    out = pd.DataFrame(
        results
    )

    if out.empty:

        print("")
        print(
            "No stocks matched "
            "the current setup."
        )

    else:

        # Preserve your original ranking logic.
        out = (
            out
            .sort_values(
                [
                    "score",
                    "distance_to_200dma_pct"
                ],
                ascending=[
                    False,
                    True
                ]
            )
            .reset_index(drop=True)
        )

        out.insert(
            0,
            "rank",
            range(
                1,
                len(out) + 1
            )
        )

        csv_path = (
            OUTPUT_DIR /
            "NSE_200DMA_Recovery_Scanner.csv"
        )

        xlsx_path = (
            OUTPUT_DIR /
            "NSE_200DMA_Recovery_Scanner.xlsx"
        )

        out.to_csv(
            csv_path,
            index=False
        )

        out.to_excel(
            xlsx_path,
            index=False
        )

        print("")
        print(
            "=== CANDIDATES ==="
        )
        print("")

        print(
            out.to_string(
                index=False
            )
        )

        print("")
        print(
            f"Saved: {csv_path}"
        )

        print(
            f"Saved: {xlsx_path}"
        )

    if failed:

        pd.DataFrame(
            {
                "symbol": failed
            }
        ).to_csv(

            OUTPUT_DIR /
            "download_failures.csv",

            index=False

        )

        print("")
        print(
            f"Data unavailable for "
            f"{len(failed)} symbols."
        )

    elapsed = (
        time.time() -
        start_time
    )

    print("")
    print(
        "======================================"
    )

    print(
        f"Scanner completed in "
        f"{elapsed / 60:.1f} minutes"
    )

    print(
        "======================================"
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore"
    )

    main()

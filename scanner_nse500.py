#!/usr/bin/env python3

"""
NSE 200 DMA + PRIOR HIGH BREAKOUT SCANNER

Scans the Nifty 500 universe using Yahoo Finance EOD data.

SETUPS:

1. 200 DMA RECOVERY
   - Recent cross from above 200 DMA to below 200 DMA
   - Current price within +/-3% of 200 DMA
   - Recovery from breakdown low
   - 50 DMA
   - RSI
   - Volume

2. 52 WEEK HIGH / HIGH LEVEL
   - Current price close to 52-week high
   - Current price above 200 DMA
   - Current price above 50 DMA
   - RSI
   - Volume

3. PRIOR HIGH BREAKOUT  <-- NEW
   - Identify the significant high BEFORE the 200 DMA breakdown
   - Stock subsequently falls toward/below 200 DMA
   - Stock recovers
   - Current price is above 200 DMA
   - Current price is above that PREVIOUS HIGH

Example:

Previous High        ₹530
200 DMA              ₹370
Correction Low       ₹350
Current Price        ₹540

=> PRIOR HIGH BREAKOUT

The scanner retains every valid downloaded stock.

No Zerodha API and no orders are used.

Yahoo Finance is a third-party EOD data source.
Use this as a research/screening tool and verify
candidates before trading.
"""

from pathlib import Path
import time
import warnings

import pandas as pd
import numpy as np
import requests
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

NIFTY500_URL = (
    "https://archives.nseindia.com/"
    "content/indices/ind_nifty500list.csv"
)


# ============================================================
# 200 DMA RECOVERY SETTINGS
# ============================================================

PROXIMITY_PCT = 3.0

# Look for a 200 DMA breakdown within last 90 sessions
BREAK_LOOKBACK = 90

MIN_DAYS = 220


# ============================================================
# PRIOR HIGH SETTINGS
# ============================================================

# Number of trading sessions before the 200 DMA breakdown
# used to identify the meaningful previous high.
#
# 180 sessions ≈ 9 months of trading.
#
# This avoids using an ancient all-time high.
PRIOR_HIGH_LOOKBACK = 180


# Near prior high means current price is within this
# percentage below the previous high.
PRIOR_HIGH_NEAR_PCT = 5.0


# ============================================================
# 52 WEEK HIGH SETTINGS
# ============================================================

HIGH_LEVEL_MAX_DISTANCE = 15.0


# ============================================================
# DOWNLOAD SETTINGS
# ============================================================

CHUNK_SIZE = 50
BATCH_PAUSE = 0.5


# ============================================================
# RSI
# ============================================================

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


# ============================================================
# GET NIFTY 500 SYMBOLS
# ============================================================

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

        local.write_bytes(response.content)

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

    return [
        s for s in symbols
        if s and s.lower() != "nan"
    ]


# ============================================================
# CLEAN YAHOO DATA
# ============================================================

def clean_yf_frame(df, ticker):

    if df is None or df.empty:
        return None

    df = df.copy()

    # Handle MultiIndex columns
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

    if "close" not in df.columns:
        return None

    if "volume" not in df.columns:
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

    return df.dropna(
        subset=["close"]
    )


# ============================================================
# ANALYSE ONE STOCK
# ============================================================

def analyse(symbol, df):

    if df is None:
        return None

    if len(df) < MIN_DAYS:
        return None

    df = df.copy()

    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================

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

    # ========================================================
    # 52 WEEK HIGH
    # ========================================================

    df["high_52w"] = (
        df["high"]
        .rolling(
            252,
            min_periods=20
        )
        .max()
    )

    # ========================================================
    # CURRENT VALUES
    # ========================================================

    cur = df.iloc[-1]

    current_price = float(
        cur["close"]
    )

    dma_200 = cur["dma_200"]
    dma_50 = cur["dma_50"]

    if pd.isna(dma_200):
        return None

    # ========================================================
    # DISTANCE FROM 200 DMA
    # ========================================================

    distance_to_200dma = (
        current_price /
        float(dma_200) -
        1
    ) * 100

    # ========================================================
    # DISTANCE FROM 52 WEEK HIGH
    # ========================================================

    high_52w = cur["high_52w"]

    if (
        pd.isna(high_52w)
        or high_52w <= 0
    ):

        distance_to_52w_high = np.nan

    else:

        distance_to_52w_high = (
            current_price /
            float(high_52w) -
            1
        ) * 100

    # ========================================================
    # COMMON INDICATORS
    # ========================================================

    rsi_value = cur["rsi_14"]
    volume_ratio = cur["volume_ratio"]

    # ========================================================
    # FIND 200 DMA BREAKDOWN
    # ========================================================

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

    # ========================================================
    # PRIOR HIGH VARIABLES
    # ========================================================

    prior_high = np.nan
    prior_high_date = None

    distance_to_prior_high = np.nan

    prior_high_breakout = False
    near_prior_high = False

    prior_high_setup = "NORMAL"

    prior_high_score = 0

    prior_high_reasons = []

    breakdown_found = False

    # ========================================================
    # FIND HIGH BEFORE BREAKDOWN
    # ========================================================

    if not cross_candidates.empty:

        last_cross_idx = (
            cross_candidates
            .index[-1]
        )

        cross_position = (
            df.index.get_loc(
                last_cross_idx
            )
        )

        breakdown_found = True

        # ----------------------------------------------------
        # Find the meaningful high BEFORE breakdown
        # ----------------------------------------------------

        start_position = max(
            0,
            cross_position -
            PRIOR_HIGH_LOOKBACK
        )

        prior_period = df.iloc[
            start_position:
            cross_position
        ]

        if not prior_period.empty:

            prior_high = (
                prior_period["high"]
                .max()
            )

            if not pd.isna(prior_high):

                prior_high_row = (
                    prior_period[
                        prior_period["high"]
                        ==
                        prior_high
                    ]
                    .iloc[-1]
                )

                prior_high_date = (
                    prior_high_row["date"]
                    .date()
                )

                # ------------------------------------------------
                # Current price vs previous high
                # ------------------------------------------------

                distance_to_prior_high = (

                    current_price /
                    float(prior_high) -
                    1

                ) * 100

                # ------------------------------------------------
                # BREAKOUT
                # ------------------------------------------------

                if (
                    current_price >
                    float(dma_200)
                    and
                    current_price >
                    float(prior_high)
                ):

                    prior_high_breakout = True

                    prior_high_score += 50

                    prior_high_reasons.append(
                        "above prior high"
                    )

                # ------------------------------------------------
                # NEAR PRIOR HIGH
                # ------------------------------------------------

                elif (
                    current_price >
                    float(dma_200)
                    and
                    distance_to_prior_high >=
                    -PRIOR_HIGH_NEAR_PCT
                ):

                    near_prior_high = True

                    prior_high_score += 35

                    prior_high_reasons.append(
                        "within 5% of prior high"
                    )

                # ------------------------------------------------
                # ABOVE 200 DMA
                # ------------------------------------------------

                if current_price > float(dma_200):

                    prior_high_score += 20

                    prior_high_reasons.append(
                        "above 200DMA"
                    )

                # ------------------------------------------------
                # ABOVE 50 DMA
                # ------------------------------------------------

                if (
                    not pd.isna(dma_50)
                    and
                    current_price >
                    float(dma_50)
                ):

                    prior_high_score += 10

                    prior_high_reasons.append(
                        "above 50DMA"
                    )

                # ------------------------------------------------
                # RSI
                # ------------------------------------------------

                if (
                    not pd.isna(rsi_value)
                    and
                    float(rsi_value) >= 50
                ):

                    prior_high_score += 10

                    prior_high_reasons.append(
                        "RSI >=50"
                    )

                # ------------------------------------------------
                # Volume
                # ------------------------------------------------

                if (
                    not pd.isna(volume_ratio)
                    and
                    float(volume_ratio) >= 1.2
                ):

                    prior_high_score += 10

                    prior_high_reasons.append(
                        "volume >=1.2x"
                    )

                # ------------------------------------------------
                # Setup classification
                # ------------------------------------------------

                if prior_high_breakout:

                    prior_high_setup = (
                        "PRIOR HIGH BREAKOUT"
                    )

                elif near_prior_high:

                    prior_high_setup = (
                        "NEAR PRIOR HIGH"
                    )

                elif (
                    current_price >
                    float(dma_200)
                ):

                    prior_high_setup = (
                        "ABOVE 200DMA"
                    )

                else:

                    prior_high_setup = "NORMAL"

    # ========================================================
    # 200 DMA RECOVERY
    # ========================================================

    recovery_score = 0
    recovery_reasons = []

    recovery_candidate = False

    break_below_date = None
    breakdown_low = np.nan
    recovery_pct = np.nan
    improving = False

    if (
        abs(distance_to_200dma)
        <=
        PROXIMITY_PCT
    ):

        if not cross_candidates.empty:

            last_cross_idx = (
                cross_candidates
                .index[-1]
            )

            pos = (
                df.index.get_loc(
                    last_cross_idx
                )
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

            if (
                not pd.isna(
                    breakdown_low
                )
                and
                breakdown_low > 0
            ):

                recovery_pct = (

                    current_price /
                    float(breakdown_low) -
                    1

                ) * 100

            break_below_date = (
                df.loc[
                    last_cross_idx,
                    "date"
                ].date()
            )

            # ------------------------------------------------
            # Improving toward 200 DMA
            # ------------------------------------------------

            if len(df) >= 10:

                prev = df.iloc[-10]

                if (
                    not pd.isna(
                        prev["dma_200"]
                    )
                    and
                    not pd.isna(
                        prev["close"]
                    )
                ):

                    improving = (

                        abs(
                            current_price /
                            float(dma_200) -
                            1
                        )

                        <

                        abs(
                            float(
                                prev["close"]
                            ) /
                            float(
                                prev["dma_200"]
                            ) -
                            1
                        )
                    )

            # ------------------------------------------------
            # Distance score
            # ------------------------------------------------

            if distance_to_200dma >= -1:

                recovery_score += 25

                recovery_reasons.append(
                    "at/above 200DMA"
                )

            else:

                recovery_score += 15

                recovery_reasons.append(
                    "within 3% of 200DMA"
                )

            # ------------------------------------------------
            # Improving
            # ------------------------------------------------

            if improving:

                recovery_score += 20

                recovery_reasons.append(
                    "moving toward 200DMA"
                )

            # ------------------------------------------------
            # Recovery
            # ------------------------------------------------

            if recovery_pct >= 10:

                recovery_score += 25

                recovery_reasons.append(
                    "recovery >=10%"
                )

            elif recovery_pct >= 5:

                recovery_score += 20

                recovery_reasons.append(
                    "recovery >=5%"
                )

            elif recovery_pct > 0:

                recovery_score += 10

                recovery_reasons.append(
                    "above breakdown low"
                )

            # ------------------------------------------------
            # 50 DMA
            # ------------------------------------------------

            if (
                not pd.isna(dma_50)
                and
                current_price >
                float(dma_50)
            ):

                recovery_score += 15

                recovery_reasons.append(
                    "above 50DMA"
                )

            # ------------------------------------------------
            # Volume
            # ------------------------------------------------

            if (
                not pd.isna(volume_ratio)
                and
                float(volume_ratio) >= 1.2
            ):

                recovery_score += 10

                recovery_reasons.append(
                    "volume >=1.2x"
                )

            # ------------------------------------------------
            # RSI
            # ------------------------------------------------

            if (
                not pd.isna(rsi_value)
                and
                float(rsi_value) >= 50
            ):

                recovery_score += 10

                recovery_reasons.append(
                    "RSI >=50"
                )

            recovery_candidate = True

    # ========================================================
    # RECOVERY SETUP
    # ========================================================

    if not recovery_candidate:

        recovery_setup = "NOT A RECOVERY"

    elif recovery_score >= 75:

        recovery_setup = "STRONG RETEST"

    elif recovery_score >= 55:

        recovery_setup = "RECOVERY WATCH"

    else:

        recovery_setup = "NEAR 200DMA"

    # ========================================================
    # 52 WEEK HIGH SETUP
    # ========================================================

    high_score = 0
    high_reasons = []

    if (
        not pd.isna(
            distance_to_52w_high
        )
        and
        distance_to_52w_high >= -5
    ):

        high_score += 40

        high_reasons.append(
            "within 5% of 52W high"
        )

    elif (
        not pd.isna(
            distance_to_52w_high
        )
        and
        distance_to_52w_high >= -10
    ):

        high_score += 30

        high_reasons.append(
            "within 10% of 52W high"
        )

    elif (
        not pd.isna(
            distance_to_52w_high
        )
        and
        distance_to_52w_high >= -15
    ):

        high_score += 20

        high_reasons.append(
            "within 15% of 52W high"
        )

    if current_price > float(dma_200):

        high_score += 20

        high_reasons.append(
            "above 200DMA"
        )

    if (
        not pd.isna(dma_50)
        and
        current_price >
        float(dma_50)
    ):

        high_score += 15

        high_reasons.append(
            "above 50DMA"
        )

    if (
        not pd.isna(rsi_value)
        and
        float(rsi_value) >= 50
    ):

        high_score += 10

        high_reasons.append(
            "RSI >=50"
        )

    if (
        not pd.isna(volume_ratio)
        and
        float(volume_ratio) >= 1.2
    ):

        high_score += 10

        high_reasons.append(
            "volume >=1.2x"
        )

    if (
        not pd.isna(
            distance_to_52w_high
        )
        and
        distance_to_52w_high >= -5
        and
        current_price > float(dma_200)
    ):

        high_setup = "BREAKOUT WATCH"

    elif (
        not pd.isna(
            distance_to_52w_high
        )
        and
        distance_to_52w_high >= -10
        and
        current_price > float(dma_200)
    ):

        high_setup = "HIGH LEVEL"

    elif (
        not pd.isna(
            distance_to_52w_high
        )
        and
        distance_to_52w_high >= -15
        and
        current_price > float(dma_200)
    ):

        high_setup = "APPROACHING HIGH"

    else:

        high_setup = "NORMAL"

    high_level_candidate = (
        high_setup != "NORMAL"
    )

    # ========================================================
    # COMBINED SETUP
    # ========================================================

    if prior_high_breakout:

        combined_setup = (
            "PRIOR HIGH BREAKOUT"
        )

    elif near_prior_high:

        combined_setup = (
            "NEAR PRIOR HIGH"
        )

    elif (
        recovery_candidate
        and
        high_level_candidate
    ):

        combined_setup = "BOTH"

    elif recovery_candidate:

        combined_setup = "200DMA RECOVERY"

    elif high_level_candidate:

        combined_setup = "HIGH LEVEL"

    else:

        combined_setup = "NONE"

    # ========================================================
    # RETURN
    # ========================================================

    return {

        # ====================================================
        # BASIC
        # ====================================================

        "symbol": symbol,

        "date": cur["date"].date(),

        "close": round(
            current_price,
            2
        ),

        # ====================================================
        # 200 DMA
        # ====================================================

        "dma_200": round(
            float(dma_200),
            2
        ),

        "distance_to_200dma_pct": round(
            float(distance_to_200dma),
            2
        ),

        # ====================================================
        # 50 DMA
        # ====================================================

        "dma_50": (
            round(
                float(dma_50),
                2
            )
            if not pd.isna(dma_50)
            else np.nan
        ),

        # ====================================================
        # RSI
        # ====================================================

        "rsi_14": (
            round(
                float(rsi_value),
                2
            )
            if not pd.isna(rsi_value)
            else np.nan
        ),

        # ====================================================
        # VOLUME
        # ====================================================

        "volume_ratio": (
            round(
                float(volume_ratio),
                2
            )
            if not pd.isna(volume_ratio)
            else np.nan
        ),

        # ====================================================
        # 200 DMA RECOVERY
        # ====================================================

        "break_below_date":
            break_below_date,

        "breakdown_low": (
            round(
                float(breakdown_low),
                2
            )
            if not pd.isna(breakdown_low)
            else np.nan
        ),

        "recovery_from_breakdown_low_pct": (
            round(
                float(recovery_pct),
                2
            )
            if not pd.isna(recovery_pct)
            else np.nan
        ),

        "score":
            int(recovery_score),

        "setup":
            recovery_setup,

        "recovery_candidate":
            bool(recovery_candidate),

        "reasons":
            "; ".join(
                recovery_reasons
            ),

        # ====================================================
        # 52 WEEK HIGH
        # ====================================================

        "high_52w": (
            round(
                float(high_52w),
                2
            )
            if not pd.isna(high_52w)
            else np.nan
        ),

        "distance_to_52w_high_pct": (
            round(
                float(distance_to_52w_high),
                2
            )
            if not pd.isna(
                distance_to_52w_high
            )
            else np.nan
        ),

        "high_level_score":
            int(high_score),

        "high_level_setup":
            high_setup,

        "high_level_candidate":
            bool(high_level_candidate),

        "high_level_reasons":
            "; ".join(
                high_reasons
            ),

        # ====================================================
        # NEW — PRIOR HIGH
        # ====================================================

        "prior_high_before_breakdown": (
            round(
                float(prior_high),
                2
            )
            if not pd.isna(prior_high)
            else np.nan
        ),

        "prior_high_date":
            prior_high_date,

        "distance_to_prior_high_pct": (
            round(
                float(
                    distance_to_prior_high
                ),
                2
            )
            if not pd.isna(
                distance_to_prior_high
            )
            else np.nan
        ),

        "prior_high_breakout":
            bool(prior_high_breakout),

        "near_prior_high":
            bool(near_prior_high),

        "prior_high_score":
            int(prior_high_score),

        "prior_high_setup":
            prior_high_setup,

        "prior_high_reasons":
            "; ".join(
                prior_high_reasons
            ),

        # ====================================================
        # COMBINED
        # ====================================================

        "combined_setup":
            combined_setup,

    }


# ============================================================
# BATCH DOWNLOAD
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("")
    print(
        "=============================================="
    )
    print(
        " FAST NSE 200 DMA + PRIOR HIGH SCANNER"
    )
    print(
        "=============================================="
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

    # ========================================================
    # BATCH LOOP
    # ========================================================

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

        # ====================================================
        # FALLBACK
        # ====================================================

        if data is None:

            print(
                "Batch failed. "
                "Retrying individually..."
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

                    clean = clean_yf_frame(
                        df,
                        ticker
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

                except Exception as e:

                    print(
                        f"  ! {symbol}: {e}"
                    )

                    failed.append(
                        symbol
                    )

            time.sleep(
                BATCH_PAUSE
            )

            continue

        # ====================================================
        # ANALYSE EACH STOCK
        # ====================================================

        for symbol in batch_symbols:

            ticker = (
                symbol + ".NS"
            )

            try:

                clean = clean_yf_frame(
                    data,
                    ticker
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

        time.sleep(
            BATCH_PAUSE
        )

    # ========================================================
    # CREATE OUTPUT
    # ========================================================

    out = pd.DataFrame(
        results
    )

    if out.empty:

        print("")
        print(
            "No valid stocks were downloaded."
        )

    else:

        # ====================================================
        # SORTING
        #
        # 1. PRIOR HIGH BREAKOUT
        # 2. NEAR PRIOR HIGH
        # 3. BOTH
        # 4. 200 DMA RECOVERY
        # 5. HIGH LEVEL
        # 6. OTHER
        # ====================================================

        out["_sort_group"] = np.select(

            [

                out[
                    "combined_setup"
                ].eq(
                    "PRIOR HIGH BREAKOUT"
                ),

                out[
                    "combined_setup"
                ].eq(
                    "NEAR PRIOR HIGH"
                ),

                out[
                    "combined_setup"
                ].eq(
                    "BOTH"
                ),

                out[
                    "combined_setup"
                ].eq(
                    "200DMA RECOVERY"
                ),

                out[
                    "combined_setup"
                ].eq(
                    "HIGH LEVEL"
                )

            ],

            [
                0,
                1,
                2,
                3,
                4
            ],

            default=5

        )

        out = (
            out
            .sort_values(

                [

                    "_sort_group",

                    "prior_high_score",

                    "score",

                    "high_level_score",

                    "distance_to_prior_high_pct"

                ],

                ascending=[

                    True,

                    False,

                    False,

                    False,

                    False

                ]

            )
            .reset_index(
                drop=True
            )
        )

        out.drop(
            columns=[
                "_sort_group"
            ],
            inplace=True
        )

        # ====================================================
        # RANK
        # ====================================================

        out.insert(
            0,
            "rank",
            range(
                1,
                len(out) + 1
            )
        )

        # ====================================================
        # SAVE CSV
        # ====================================================

        csv_path = (
            OUTPUT_DIR /
            "NSE_200DMA_Recovery_Scanner.csv"
        )

        out.to_csv(
            csv_path,
            index=False
        )

        # ====================================================
        # SAVE EXCEL
        # ====================================================

        xlsx_path = (
            OUTPUT_DIR /
            "NSE_200DMA_Recovery_Scanner.xlsx"
        )

        out.to_excel(
            xlsx_path,
            index=False
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        recovery_count = int(
            out[
                "recovery_candidate"
            ].sum()
        )

        high_count = int(
            out[
                "high_level_candidate"
            ].sum()
        )

        prior_breakout_count = int(
            out[
                "prior_high_breakout"
            ].sum()
        )

        near_prior_count = int(
            out[
                "near_prior_high"
            ].sum()
        )

        both_count = int(
            (
                out[
                    "combined_setup"
                ]
                .eq("BOTH")
            ).sum()
        )

        print("")
        print(
            "=============================================="
        )
        print(
            " SCANNER SUMMARY"
        )
        print(
            "=============================================="
        )

        print(
            f"Valid stocks: {len(out)}"
        )

        print(
            f"200 DMA recovery candidates: "
            f"{recovery_count}"
        )

        print(
            f"52W high-level candidates: "
            f"{high_count}"
        )

        print(
            f"Prior high breakouts: "
            f"{prior_breakout_count}"
        )

        print(
            f"Near prior high: "
            f"{near_prior_count}"
        )

        print(
            f"Both setups: "
            f"{both_count}"
        )

        print("")

        print(
            f"CSV: {csv_path}"
        )

        print(
            f"Excel: {xlsx_path}"
        )

    # ========================================================
    # DOWNLOAD FAILURES
    # ========================================================

    if failed:

        failure_file = (
            OUTPUT_DIR /
            "download_failures.csv"
        )

        pd.DataFrame(
            {
                "symbol": failed
            }
        ).to_csv(
            failure_file,
            index=False
        )

        print("")

        print(
            f"Data unavailable for "
            f"{len(failed)} stocks."
        )

        print(
            f"See: {failure_file}"
        )

    # ========================================================
    # TIME
    # ========================================================

    elapsed = (
        time.time() -
        start_time
    )

    print("")

    print(
        "=============================================="
    )

    print(
        f"Scanner completed in "
        f"{elapsed / 60:.1f} minutes"
    )

    print(
        "=============================================="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore"
    )

    main()

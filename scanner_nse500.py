#!/usr/bin/env python3

"""
CARJ.IN — NSE FULL EQUITY STOCK SCANNER

Universe:
    Full NSE actively tradable equity universe.

NOT restricted to:
    Nifty 50
    Nifty 100
    Nifty 200
    Nifty 500

Main signals:

1. 200 DMA Recovery
2. 52 Week High
3. Prior High Breakout
4. Near Prior High
5. Trend Strength
6. Momentum
7. Volume confirmation
8. Portfolio eligibility

IMPORTANT:
    52-week high = highest daily HIGH during approximately
    the last 252 trading sessions.

This is NOT lifetime high.

Data source:
    Yahoo Finance EOD data.

NSE universe:
    Official NSE equity securities list.

This is a research/screening system.
It is not investment advice.
"""

from pathlib import Path
import time
import warnings

import pandas as pd
import numpy as np
import requests
import yfinance as yf


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# NSE EQUITY UNIVERSE
# ============================================================

NSE_URLS = [

    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",

    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",

]


# ============================================================
# TECHNICAL SETTINGS
# ============================================================

MIN_DAYS = 220

DMA_200 = 200

DMA_50 = 50

RSI_PERIOD = 14

VOLUME_PERIOD = 20

WEEK_52 = 252


# ============================================================
# 200 DMA RECOVERY
# ============================================================

BREAK_LOOKBACK = 90

PROXIMITY_PCT = 3.0


# ============================================================
# PRIOR HIGH
# ============================================================

PRIOR_HIGH_LOOKBACK = 180

PRIOR_HIGH_NEAR_PCT = 5.0


# ============================================================
# DOWNLOAD
# ============================================================

CHUNK_SIZE = 75

BATCH_PAUSE = 0.5

DOWNLOAD_PERIOD = "2y"


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

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# DOWNLOAD NSE UNIVERSE
# ============================================================

def get_nse_equity_symbols():

    local_file = (
        DATA_DIR /
        "nse_equity_symbols.csv"
    )

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36",

        "Accept":
            "text/csv,application/octet-stream,*/*",

        "Referer":
            "https://www.nseindia.com/"

    }


    downloaded = False

    last_error = None


    for url in NSE_URLS:

        try:

            print(
                f"Downloading NSE equity universe..."
            )

            response = requests.get(
                url,
                headers=headers,
                timeout=30
            )

            response.raise_for_status()

            if len(response.content) < 1000:

                raise RuntimeError(
                    "NSE returned an unexpectedly small file."
                )

            local_file.write_bytes(
                response.content
            )

            downloaded = True

            print(
                f"NSE universe downloaded from: {url}"
            )

            break

        except Exception as e:

            last_error = e

            print(
                f"Universe download failed: {e}"
            )


    if not downloaded:

        if local_file.exists():

            print(
                "Using cached NSE equity universe."
            )

        else:

            raise RuntimeError(
                "Unable to download NSE equity universe "
                "and no cached file exists."
            ) from last_error


    df = pd.read_csv(
        local_file
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]


    symbol_column = None

    for column in df.columns:

        if column.upper() == "SYMBOL":

            symbol_column = column

            break


    if symbol_column is None:

        raise RuntimeError(
            "NSE equity CSV does not contain SYMBOL column."
        )


    symbols = (

        df[symbol_column]

        .astype(str)

        .str.strip()

        .str.upper()

        .tolist()

    )


    clean_symbols = []

    for symbol in symbols:

        if not symbol:
            continue

        if symbol == "NAN":
            continue

        clean_symbols.append(
            symbol
        )


    clean_symbols = sorted(
        list(
            set(
                clean_symbols
            )
        )
    )


    print(
        f"FULL NSE EQUITY UNIVERSE: "
        f"{len(clean_symbols)} securities"
    )


    return clean_symbols


# ============================================================
# CLEAN YAHOO FRAME
# ============================================================

def clean_yf_frame(
    df,
    ticker
):

    if df is None:
        return None

    if df.empty:
        return None


    df = df.copy()


    # -----------------------------------------
    # MultiIndex
    # -----------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        try:

            levels = [
                list(level)
                for level
                in df.columns.levels
            ]


            if ticker in df.columns.get_level_values(0):

                df = df[ticker]

            elif ticker in df.columns.get_level_values(-1):

                df = df.xs(
                    ticker,
                    axis=1,
                    level=-1
                )

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
        "high",
        "low",
        "volume"
    ]


    for column in required:

        if column not in df.columns:

            return None


    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


    df = df.reset_index()


    if "date" in df.columns:

        date_column = "date"

    else:

        date_column = df.columns[0]


    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )


    try:

        if df["date"].dt.tz is not None:

            df["date"] = (
                df["date"]
                .dt
                .tz_localize(None)
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


# ============================================================
# ANALYSE STOCK
# ============================================================

def analyse(
    symbol,
    df
):

    if df is None:
        return None


    if len(df) < MIN_DAYS:

        return None


    df = df.copy()


    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    df["dma_200"] = (
        df["close"]
        .rolling(DMA_200)
        .mean()
    )


    df["dma_50"] = (
        df["close"]
        .rolling(DMA_50)
        .mean()
    )


    # ========================================================
    # RSI
    # ========================================================

    df["rsi_14"] = rsi(
        df["close"],
        RSI_PERIOD
    )


    # ========================================================
    # VOLUME
    # ========================================================

    df["volume_avg_20"] = (
        df["volume"]
        .rolling(VOLUME_PERIOD)
        .mean()
    )


    df["volume_ratio"] = (

        df["volume"] /
        df["volume_avg_20"]

    )


    # ========================================================
    # 52 WEEK HIGH
    #
    # IMPORTANT:
    # 252 trading sessions.
    #
    # NOT lifetime high.
    # ========================================================

    df["high_52w"] = (

        df["high"]

        .rolling(
            WEEK_52,
            min_periods=20
        )

        .max()

    )


    # ========================================================
    # 52 WEEK LOW
    # ========================================================

    df["low_52w"] = (

        df["low"]

        .rolling(
            WEEK_52,
            min_periods=20
        )

        .min()

    )


    # ========================================================
    # CURRENT
    # ========================================================

    cur = df.iloc[-1]


    close = float(
        cur["close"]
    )


    dma200 = cur["dma_200"]

    dma50 = cur["dma_50"]

    rsi14 = cur["rsi_14"]

    volume_ratio = cur[
        "volume_ratio"
    ]

    high52 = cur[
        "high_52w"
    ]

    low52 = cur[
        "low_52w"
    ]


    if pd.isna(dma200):

        return None


    # ========================================================
    # DISTANCE TO 200 DMA
    # ========================================================

    distance_200 = (

        close /
        float(dma200)
        - 1

    ) * 100


    # ========================================================
    # DISTANCE TO 52W HIGH
    # ========================================================

    if (
        pd.isna(high52)
        or
        float(high52) <= 0
    ):

        distance_52w_high = np.nan

    else:

        distance_52w_high = (

            close /
            float(high52)
            - 1

        ) * 100


    # ========================================================
    # DISTANCE TO 52W LOW
    # ========================================================

    if (
        pd.isna(low52)
        or
        float(low52) <= 0
    ):

        distance_52w_low = np.nan

    else:

        distance_52w_low = (

            close /
            float(low52)
            - 1

        ) * 100


    # ========================================================
    # ABOVE 200 DMA
    # ========================================================

    above_200 = (
        close >
        float(dma200)
    )


    # ========================================================
    # ABOVE 50 DMA
    # ========================================================

    above_50 = (

        not pd.isna(dma50)
        and
        close >
        float(dma50)

    )


    # ========================================================
    # 200 DMA TREND
    # ========================================================

    dma200_rising = False

    if len(df) >= 210:

        old_dma = df.iloc[-10][
            "dma_200"
        ]

        if not pd.isna(old_dma):

            dma200_rising = (

                float(dma200)
                >
                float(old_dma)

            )


    # ========================================================
    # 50 DMA VS 200 DMA
    # ========================================================

    golden_trend = (

        not pd.isna(dma50)
        and
        float(dma50)
        >
        float(dma200)

    )


    # ========================================================
    # PRIOR HIGH BEFORE BREAKDOWN
    # ========================================================

    prior_high = np.nan

    prior_high_date = None

    distance_prior_high = np.nan

    prior_high_breakout = False

    near_prior_high = False


    # ========================================================
    # 200 DMA CROSS BELOW
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


    crosses = (
        df.loc[cross]
        .tail(BREAK_LOOKBACK)
    )


    if not crosses.empty:

        last_cross = crosses.index[-1]

        cross_position = (
            df.index.get_loc(
                last_cross
            )
        )


        start = max(
            0,
            cross_position -
            PRIOR_HIGH_LOOKBACK
        )


        before_breakdown = df.iloc[
            start:
            cross_position
        ]


        if not before_breakdown.empty:

            prior_high = (
                before_breakdown[
                    "high"
                ].max()
            )


            if not pd.isna(
                prior_high
            ):

                prior_row = (

                    before_breakdown[

                        before_breakdown[
                            "high"
                        ]
                        ==
                        prior_high

                    ]

                    .iloc[-1]

                )


                prior_high_date = (
                    prior_row[
                        "date"
                    ].date()
                )


                distance_prior_high = (

                    close /
                    float(prior_high)
                    - 1

                ) * 100


                prior_high_breakout = (

                    above_200
                    and
                    close >
                    float(prior_high)

                )


                near_prior_high = (

                    above_200

                    and

                    distance_prior_high >=
                    -PRIOR_HIGH_NEAR_PCT

                    and

                    not prior_high_breakout

                )


    # ========================================================
    # 200 DMA RECOVERY
    # ========================================================

    recovery_candidate = False

    recovery_score = 0

    recovery_reasons = []

    breakdown_low = np.nan

    recovery_pct = np.nan

    break_below_date = None

    improving = False


    if (

        abs(distance_200)
        <=
        PROXIMITY_PCT

        and

        not crosses.empty

    ):

        last_cross = crosses.index[-1]

        pos = df.index.get_loc(
            last_cross
        )


        after = df.iloc[pos:]


        breakdown_low = (
            after["low"].min()
        )


        if (

            not pd.isna(
                breakdown_low
            )

            and

            breakdown_low > 0

        ):

            recovery_pct = (

                close /
                float(breakdown_low)
                - 1

            ) * 100


        break_below_date = (
            df.loc[
                last_cross,
                "date"
            ].date()
        )


        # -----------------------------------------
        # Improving
        # -----------------------------------------

        if len(df) >= 10:

            previous = df.iloc[-10]

            if (

                not pd.isna(
                    previous["dma_200"]
                )

                and

                not pd.isna(
                    previous["close"]
                )

            ):

                improving = (

                    abs(
                        close /
                        float(dma200)
                        - 1
                    )

                    <

                    abs(
                        float(
                            previous["close"]
                        )
                        /
                        float(
                            previous["dma_200"]
                        )
                        - 1
                    )

                )


        # -----------------------------------------
        # Score
        # -----------------------------------------

        recovery_candidate = True


        if distance_200 >= -1:

            recovery_score += 25

            recovery_reasons.append(
                "at/above 200DMA"
            )

        else:

            recovery_score += 15

            recovery_reasons.append(
                "within 3% of 200DMA"
            )


        if improving:

            recovery_score += 20

            recovery_reasons.append(
                "moving toward 200DMA"
            )


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


        if above_50:

            recovery_score += 15

            recovery_reasons.append(
                "above 50DMA"
            )


        if (

            not pd.isna(
                volume_ratio
            )

            and

            float(volume_ratio) >= 1.2

        ):

            recovery_score += 10

            recovery_reasons.append(
                "volume >=1.2x"
            )


        if (

            not pd.isna(rsi14)

            and

            float(rsi14) >= 50

        ):

            recovery_score += 10

            recovery_reasons.append(
                "RSI >=50"
            )


    # ========================================================
    # RECOVERY SETUP
    # ========================================================

    if not recovery_candidate:

        recovery_setup = (
            "NOT A RECOVERY"
        )

    elif recovery_score >= 75:

        recovery_setup = (
            "STRONG RETEST"
        )

    elif recovery_score >= 55:

        recovery_setup = (
            "RECOVERY WATCH"
        )

    else:

        recovery_setup = (
            "NEAR 200DMA"
        )


    # ========================================================
    # 52 WEEK HIGH SCORE
    # ========================================================

    high_score = 0

    high_reasons = []


    if not pd.isna(
        distance_52w_high
    ):

        if distance_52w_high >= 0:

            high_score += 40

            high_reasons.append(
                "at/new 52W high"
            )

        elif distance_52w_high >= -2:

            high_score += 35

            high_reasons.append(
                "within 2% of 52W high"
            )

        elif distance_52w_high >= -5:

            high_score += 30

            high_reasons.append(
                "within 5% of 52W high"
            )

        elif distance_52w_high >= -10:

            high_score += 20

            high_reasons.append(
                "within 10% of 52W high"
            )

        elif distance_52w_high >= -15:

            high_score += 10

            high_reasons.append(
                "within 15% of 52W high"
            )


    if above_200:

        high_score += 20

        high_reasons.append(
            "above 200DMA"
        )


    if above_50:

        high_score += 15

        high_reasons.append(
            "above 50DMA"
        )


    if dma200_rising:

        high_score += 10

        high_reasons.append(
            "200DMA rising"
        )


    if (

        not pd.isna(rsi14)

        and

        50 <= float(rsi14) <= 75

    ):

        high_score += 10

        high_reasons.append(
            "healthy RSI"
        )


    if (

        not pd.isna(
            volume_ratio
        )

        and

        float(volume_ratio) >= 1.2

    ):

        high_score += 10

        high_reasons.append(
            "volume confirmation"
        )


    # ========================================================
    # 52W SETUP
    # ========================================================

    if (

        not pd.isna(
            distance_52w_high
        )

        and

        distance_52w_high >= 0

        and

        above_200

    ):

        high_setup = (
            "52W HIGH BREAKOUT"
        )


    elif (

        not pd.isna(
            distance_52w_high
        )

        and

        distance_52w_high >= -2

        and

        above_200

    ):

        high_setup = (
            "AT 52W HIGH"
        )


    elif (

        not pd.isna(
            distance_52w_high
        )

        and

        distance_52w_high >= -5

        and

        above_200

    ):

        high_setup = (
            "NEAR 52W HIGH"
        )


    elif (

        not pd.isna(
            distance_52w_high
        )

        and

        distance_52w_high >= -10

        and

        above_200

    ):

        high_setup = (
            "APPROACHING 52W HIGH"
        )


    else:

        high_setup = "NORMAL"


    high_level_candidate = (

        high_setup !=
        "NORMAL"

    )


    # ========================================================
    # TREND SCORE
    # ========================================================

    trend_score = 0

    trend_reasons = []


    if above_200:

        trend_score += 25

        trend_reasons.append(
            "above 200DMA"
        )


    if above_50:

        trend_score += 15

        trend_reasons.append(
            "above 50DMA"
        )


    if dma200_rising:

        trend_score += 20

        trend_reasons.append(
            "200DMA rising"
        )


    if golden_trend:

        trend_score += 15

        trend_reasons.append(
            "50DMA > 200DMA"
        )


    if (

        not pd.isna(rsi14)

        and

        float(rsi14) >= 50

    ):

        trend_score += 10

        trend_reasons.append(
            "RSI >=50"
        )


    if (

        not pd.isna(
            volume_ratio
        )

        and

        float(volume_ratio) >= 1.2

    ):

        trend_score += 15

        trend_reasons.append(
            "volume >=1.2x"
        )


    # ========================================================
    # PRIOR HIGH SCORE
    # ========================================================

    prior_score = 0

    prior_reasons = []


    if prior_high_breakout:

        prior_score += 50

        prior_reasons.append(
            "above prior high"
        )


    elif near_prior_high:

        prior_score += 35

        prior_reasons.append(
            "within 5% of prior high"
        )


    if above_200:

        prior_score += 20

        prior_reasons.append(
            "above 200DMA"
        )


    if above_50:

        prior_score += 10

        prior_reasons.append(
            "above 50DMA"
        )


    if (

        not pd.isna(rsi14)

        and

        float(rsi14) >= 50

    ):

        prior_score += 10

        prior_reasons.append(
            "RSI >=50"
        )


    if (

        not pd.isna(
            volume_ratio
        )

        and

        float(volume_ratio) >= 1.2

    ):

        prior_score += 10

        prior_reasons.append(
            "volume >=1.2x"
        )


    if prior_high_breakout:

        prior_setup = (
            "PRIOR HIGH BREAKOUT"
        )

    elif near_prior_high:

        prior_setup = (
            "NEAR PRIOR HIGH"
        )

    elif above_200:

        prior_setup = (
            "ABOVE 200DMA"
        )

    else:

        prior_setup = "NORMAL"


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

        high_setup ==
        "52W HIGH BREAKOUT"

    ):

        combined_setup = (
            "52W HIGH BREAKOUT"
        )

    elif (

        high_setup ==
        "AT 52W HIGH"

    ):

        combined_setup = (
            "AT 52W HIGH"
        )

    elif recovery_candidate:

        combined_setup = (
            "200DMA RECOVERY"
        )

    elif high_level_candidate:

        combined_setup = (
            "HIGH LEVEL"
        )

    elif above_200:

        combined_setup = (
            "ABOVE 200DMA"
        )

    else:

        combined_setup = (
            "NORMAL"
        )


    # ========================================================
    # PORTFOLIO ELIGIBILITY
    # ========================================================

    portfolio_eligible = (

        above_200

        and

        not pd.isna(
            distance_52w_high
        )

        and

        distance_52w_high >= -5

    )


    # ========================================================
    # PORTFOLIO SCORE
    #
    # This is the initial technical score.
    # Fundamentals will be added later.
    # ========================================================

    portfolio_technical_score = (

        trend_score * 0.45

        +

        high_score * 0.35

        +

        prior_score * 0.20

    )


    portfolio_technical_score = round(
        portfolio_technical_score,
        2
    )


    # ========================================================
    # TREND TARGET
    #
    # Initial conservative model:
    #
    # Target = entry + 2 ATR
    #
    # capped around a reasonable trend extension.
    # ========================================================

    df["tr"] = np.maximum(

        df["high"] -
        df["low"],

        np.maximum(

            abs(
                df["high"] -
                df["close"].shift(1)
            ),

            abs(
                df["low"] -
                df["close"].shift(1)
            )

        )

    )


    atr14 = (
        df["tr"]
        .rolling(14)
        .mean()
        .iloc[-1]
    )


    if pd.isna(atr14):

        trend_target = np.nan

        trend_upside_pct = np.nan

    else:

        trend_target = (
            close +
            (2 * float(atr14))
        )

        trend_upside_pct = (

            trend_target /
            close -
            1

        ) * 100


    # ========================================================
    # STOP LOSS
    # ========================================================

    stop_loss = (
        close * 0.90
    )


    # ========================================================
    # REWARD / RISK
    # ========================================================

    if (

        not pd.isna(
            trend_target
        )

        and

        trend_target > close

    ):

        reward = (
            trend_target -
            close
        )

        risk = (
            close -
            stop_loss
        )

        reward_risk = (
            reward /
            risk
        )

    else:

        reward_risk = np.nan


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "symbol":
            symbol,

        "date":
            cur["date"].date(),

        "close":
            round(
                close,
                2
            ),

        # -------------------------------------
        # MOVING AVERAGES
        # -------------------------------------

        "dma_200":
            round(
                float(dma200),
                2
            ),

        "dma_50":
            (
                round(
                    float(dma50),
                    2
                )
                if not pd.isna(dma50)
                else np.nan
            ),

        "distance_to_200dma_pct":
            round(
                distance_200,
                2
            ),

        "above_200dma":
            above_200,

        "above_50dma":
            above_50,

        "dma_200_rising":
            dma200_rising,

        "golden_trend":
            golden_trend,

        # -------------------------------------
        # MOMENTUM
        # -------------------------------------

        "rsi_14":
            (
                round(
                    float(rsi14),
                    2
                )
                if not pd.isna(rsi14)
                else np.nan
            ),

        "volume_ratio":
            (
                round(
                    float(
                        volume_ratio
                    ),
                    2
                )
                if not pd.isna(
                    volume_ratio
                )
                else np.nan
            ),

        # -------------------------------------
        # 52 WEEK
        # -------------------------------------

        "high_52w":
            (
                round(
                    float(high52),
                    2
                )
                if not pd.isna(high52)
                else np.nan
            ),

        "low_52w":
            (
                round(
                    float(low52),
                    2
                )
                if not pd.isna(low52)
                else np.nan
            ),

        "distance_to_52w_high_pct":
            (
                round(
                    distance_52w_high,
                    2
                )
                if not pd.isna(
                    distance_52w_high
                )
                else np.nan
            ),

        "distance_from_52w_low_pct":
            (
                round(
                    distance_52w_low,
                    2
                )
                if not pd.isna(
                    distance_52w_low
                )
                else np.nan
            ),

        "high_level_score":
            int(high_score),

        "high_level_setup":
            high_setup,

        "high_level_reasons":
            "; ".join(
                high_reasons
            ),

        # -------------------------------------
        # 200 DMA RECOVERY
        # -------------------------------------

        "break_below_date":
            break_below_date,

        "breakdown_low":
            (
                round(
                    float(
                        breakdown_low
                    ),
                    2
                )
                if not pd.isna(
                    breakdown_low
                )
                else np.nan
            ),

        "recovery_from_breakdown_low_pct":
            (
                round(
                    recovery_pct,
                    2
                )
                if not pd.isna(
                    recovery_pct
                )
                else np.nan
            ),

        "recovery_score":
            int(
                recovery_score
            ),

        "score":
            int(
                recovery_score
            ),

        "setup":
            recovery_setup,

        "recovery_candidate":
            recovery_candidate,

        "recovery_reasons":
            "; ".join(
                recovery_reasons
            ),

        # -------------------------------------
        # PRIOR HIGH
        # -------------------------------------

        "prior_high_before_breakdown":
            (
                round(
                    float(
                        prior_high
                    ),
                    2
                )
                if not pd.isna(
                    prior_high
                )
                else np.nan
            ),

        "prior_high_date":
            prior_high_date,

        "distance_to_prior_high_pct":
            (
                round(
                    distance_prior_high,
                    2
                )
                if not pd.isna(
                    distance_prior_high
                )
                else np.nan
            ),

        "prior_high_breakout":
            prior_high_breakout,

        "near_prior_high":
            near_prior_high,

        "prior_high_score":
            int(prior_score),

        "prior_high_setup":
            prior_setup,

        "prior_high_reasons":
            "; ".join(
                prior_reasons
            ),

        # -------------------------------------
        # TREND
        # -------------------------------------

        "trend_score":
            int(trend_score),

        "trend_reasons":
            "; ".join(
                trend_reasons
            ),

        # -------------------------------------
        # PORTFOLIO
        # -------------------------------------

        "portfolio_eligible":
            portfolio_eligible,

        "portfolio_technical_score":
            portfolio_technical_score,

        # -------------------------------------
        # TARGET / RISK
        # -------------------------------------

        "trend_target":
            (
                round(
                    float(
                        trend_target
                    ),
                    2
                )
                if not pd.isna(
                    trend_target
                )
                else np.nan
            ),

        "trend_upside_pct":
            (
                round(
                    float(
                        trend_upside_pct
                    ),
                    2
                )
                if not pd.isna(
                    trend_upside_pct
                )
                else np.nan
            ),

        "stop_loss_10pct":
            round(
                stop_loss,
                2
            ),

        "reward_risk":
            (
                round(
                    float(
                        reward_risk
                    ),
                    2
                )
                if not pd.isna(
                    reward_risk
                )
                else np.nan
            ),

        # -------------------------------------
        # FINAL
        # -------------------------------------

        "combined_setup":
            combined_setup

    }


# ============================================================
# BATCH DOWNLOAD
# ============================================================

def download_batch(
    symbols
):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]


    print(
        f"Downloading {len(tickers)} stocks..."
    )


    try:

        data = yf.download(

            tickers=tickers,

            period=DOWNLOAD_PERIOD,

            interval="1d",

            auto_adjust=False,

            progress=False,

            threads=True,

            group_by="ticker",

            timeout=60

        )


        return data


    except Exception as e:

        print(
            f"Batch download error: {e}"
        )

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()


    print("")
    print(
        "=================================================="
    )
    print(
        " CARJ.IN — FULL NSE EQUITY SCANNER"
    )
    print(
        "=================================================="
    )
    print("")


    symbols = (
        get_nse_equity_symbols()
    )


    print("")
    print(
        "IMPORTANT:"
    )

    print(
        "Scanning full NSE equity universe."
    )

    print(
        "Not limited to Nifty 500."
    )

    print(
        "52-week high = 252 trading-session high."
    )

    print("")


    results = []

    failed = []


    total = len(symbols)

    batches = int(
        np.ceil(
            total /
            CHUNK_SIZE
        )
    )


    for batch_number, start_index in enumerate(

        range(
            0,
            total,
            CHUNK_SIZE
        ),

        1

    ):

        batch_symbols = symbols[
            start_index:
            start_index +
            CHUNK_SIZE
        ]


        print(
            f"Batch "
            f"{batch_number}/{batches} "
            f"— "
            f"{start_index + 1}-"
            f"{min(start_index + CHUNK_SIZE,total)}"
        )


        data = download_batch(
            batch_symbols
        )


        if data is None:

            print(
                "Batch failed. "
                "Trying individual downloads..."
            )


            for symbol in batch_symbols:

                ticker = (
                    symbol +
                    ".NS"
                )


                try:

                    raw = yf.download(

                        ticker,

                        period=DOWNLOAD_PERIOD,

                        interval="1d",

                        auto_adjust=False,

                        progress=False,

                        threads=False,

                        timeout=30

                    )


                    clean = clean_yf_frame(
                        raw,
                        ticker
                    )


                    if clean is None:

                        failed.append(
                            symbol
                        )

                        continue


                    result = analyse(
                        symbol,
                        clean
                    )


                    if result:

                        results.append(
                            result
                        )


                except Exception as e:

                    print(
                        f"Failed {symbol}: {e}"
                    )

                    failed.append(
                        symbol
                    )


        else:

            for symbol in batch_symbols:

                ticker = (
                    symbol +
                    ".NS"
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


                    result = analyse(
                        symbol,
                        clean
                    )


                    if result:

                        results.append(
                            result
                        )


                except Exception as e:

                    print(
                        f"Analyse failed "
                        f"{symbol}: {e}"
                    )

                    failed.append(
                        symbol
                    )


        time.sleep(
            BATCH_PAUSE
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    out = pd.DataFrame(
        results
    )


    if out.empty:

        raise RuntimeError(
            "Scanner produced no valid stock data."
        )


    # ========================================================
    # SORT
    #
    # PRIORITY:
    # 1. 52W breakout
    # 2. Prior high breakout
    # 3. At 52W high
    # 4. Near 52W high
    # 5. Recovery
    # 6. Other
    # ========================================================

    def sort_group(row):

        setup = row[
            "combined_setup"
        ]

        if setup == "52W HIGH BREAKOUT":

            return 0

        if setup == "PRIOR HIGH BREAKOUT":

            return 1

        if setup == "AT 52W HIGH":

            return 2

        if setup == "NEAR 52W HIGH":

            return 3

        if setup == "NEAR PRIOR HIGH":

            return 4

        if setup == "200DMA RECOVERY":

            return 5

        return 6


    out["_sort_group"] = (
        out.apply(
            sort_group,
            axis=1
        )
    )


    out = (

        out

        .sort_values(

            [

                "_sort_group",

                "portfolio_technical_score",

                "high_level_score",

                "trend_score",

                "distance_to_52w_high_pct"

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


    out.insert(
        0,
        "rank",
        range(
            1,
            len(out) + 1
        )
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR /
        "NSE_200DMA_Recovery_Scanner.csv"
    )


    out.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # SAVE EXCEL
    # ========================================================

    xlsx_path = (
        OUTPUT_DIR /
        "NSE_200DMA_Recovery_Scanner.xlsx"
    )


    out.to_excel(
        xlsx_path,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    breakout_52w = int(

        (
            out[
                "combined_setup"
            ]

            ==

            "52W HIGH BREAKOUT"

        ).sum()

    )


    at_52w = int(

        (
            out[
                "combined_setup"
            ]

            ==

            "AT 52W HIGH"

        ).sum()

    )


    prior_breakouts = int(
        out[
            "prior_high_breakout"
        ].sum()
    )


    near_high = int(
        out[
            "near_prior_high"
        ].sum()
    )


    portfolio_count = int(
        out[
            "portfolio_eligible"
        ].sum()
    )


    recovery_count = int(
        out[
            "recovery_candidate"
        ].sum()
    )


    print("")
    print(
        "=================================================="
    )
    print(
        " SCANNER COMPLETE"
    )
    print(
        "=================================================="
    )

    print(
        f"Stocks successfully analysed: "
        f"{len(out)}"
    )

    print(
        f"52W High Breakouts: "
        f"{breakout_52w}"
    )

    print(
        f"At 52W High: "
        f"{at_52w}"
    )

    print(
        f"Prior High Breakouts: "
        f"{prior_breakouts}"
    )

    print(
        f"Near Prior High: "
        f"{near_high}"
    )

    print(
        f"200 DMA Recovery: "
        f"{recovery_count}"
    )

    print(
        f"Portfolio Eligible: "
        f"{portfolio_count}"
    )

    print("")
    print(
        f"CSV: {csv_path}"
    )

    print(
        f"Excel: {xlsx_path}"
    )


    # ========================================================
    # FAILURES
    # ========================================================

    if failed:

        failure_file = (
            OUTPUT_DIR /
            "download_failures.csv"
        )


        pd.DataFrame(
            {
                "symbol":
                    failed
            }
        ).to_csv(
            failure_file,
            index=False
        )


        print(
            f"Download failures: "
            f"{len(failed)}"
        )

        print(
            f"See: {failure_file}"
        )


    elapsed = (
        time.time() -
        start
    )


    print("")
    print(
        f"Total runtime: "
        f"{elapsed / 60:.1f} minutes"
    )

    print(
        "=================================================="
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore"
    )

    main()

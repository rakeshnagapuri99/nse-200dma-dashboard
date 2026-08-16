#!/usr/bin/env python3

"""
CARJ.IN — FULL NSE FUNDAMENTAL ENRICHMENT

Reads:
    output/NSE_200DMA_Recovery_Scanner.csv

Enriches EVERY stock in the scanner output.

Fundamentals:

VALUATION
    PE
    PB
    EPS
    Dividend Yield
    Market Cap

PROFITABILITY
    ROE
    ROCE
    Operating Margin
    Net Profit Margin

FINANCIAL HEALTH
    Debt / Equity
    Current Ratio
    Quick Ratio
    Interest Coverage

GROWTH
    Revenue Growth
    EPS Growth
    Profit Growth

FUNDAMENTAL SCORE
    0-100

PORTFOLIO SCORE
    Technical + Fundamental

Data source:
    Yahoo Finance.

Important:
    Yahoo Finance fundamentals can be unavailable for
    some securities. Missing values are left blank.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("output")

CSV_FILE = (
    OUTPUT_DIR /
    "NSE_200DMA_Recovery_Scanner.csv"
)

XLSX_FILE = (
    OUTPUT_DIR /
    "NSE_200DMA_Recovery_Scanner.xlsx"
)

MAX_WORKERS = 12


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_number(value):

    try:

        if value is None:
            return None

        value = float(value)

        if pd.isna(value):
            return None

        return value

    except Exception:

        return None


# ============================================================
# GET FUNDAMENTALS
# ============================================================

def get_fundamentals(
    symbol
):

    result = {

        "pe_ratio":
            None,

        "pb_ratio":
            None,

        "eps":
            None,

        "dividend_yield":
            None,

        "market_cap":
            None,

        "roe":
            None,

        "roce":
            None,

        "operating_margin":
            None,

        "net_profit_margin":
            None,

        "debt_to_equity":
            None,

        "current_ratio":
            None,

        "quick_ratio":
            None,

        "interest_coverage":
            None,

        "revenue_growth":
            None,

        "eps_growth":
            None,

        "profit_growth":
            None,

        "fundamental_score":
            None

    }


    try:

        ticker = yf.Ticker(
            f"{symbol}.NS"
        )

        info = ticker.info


        # ====================================================
        # VALUATION
        # ====================================================

        result["pe_ratio"] = safe_number(
            info.get(
                "trailingPE"
            )
        )


        result["pb_ratio"] = safe_number(
            info.get(
                "priceToBook"
            )
        )


        result["eps"] = safe_number(
            info.get(
                "trailingEps"
            )
        )


        dividend = safe_number(
            info.get(
                "dividendYield"
            )
        )


        if dividend is not None:

            result[
                "dividend_yield"
            ] = dividend * 100


        result["market_cap"] = safe_number(
            info.get(
                "marketCap"
            )
        )


        # ====================================================
        # PROFITABILITY
        # ====================================================

        roe = safe_number(
            info.get(
                "returnOnEquity"
            )
        )


        if roe is not None:

            result["roe"] = (
                roe * 100
            )


        roce = safe_number(
            info.get(
                "returnOnCapitalEmployed"
            )
        )


        if roce is not None:

            if abs(roce) <= 1:

                roce *= 100


            result[
                "roce"
            ] = roce


        operating_margin = safe_number(
            info.get(
                "operatingMargins"
            )
        )


        if operating_margin is not None:

            result[
                "operating_margin"
            ] = (
                operating_margin *
                100
            )


        net_margin = safe_number(
            info.get(
                "profitMargins"
            )
        )


        if net_margin is not None:

            result[
                "net_profit_margin"
            ] = (
                net_margin *
                100
            )


        # ====================================================
        # FINANCIAL HEALTH
        # ====================================================

        result[
            "debt_to_equity"
        ] = safe_number(
            info.get(
                "debtToEquity"
            )
        )


        result[
            "current_ratio"
        ] = safe_number(
            info.get(
                "currentRatio"
            )
        )


        result[
            "quick_ratio"
        ] = safe_number(
            info.get(
                "quickRatio"
            )
        )


        result[
            "interest_coverage"
        ] = safe_number(
            info.get(
                "interestCoverage"
            )
        )


        # ====================================================
        # GROWTH
        # ====================================================

        revenue_growth = safe_number(
            info.get(
                "revenueGrowth"
            )
        )


        if revenue_growth is not None:

            result[
                "revenue_growth"
            ] = (
                revenue_growth *
                100
            )


        earnings_growth = safe_number(
            info.get(
                "earningsGrowth"
            )
        )


        if earnings_growth is not None:

            result[
                "eps_growth"
            ] = (
                earnings_growth *
                100
            )

            result[
                "profit_growth"
            ] = (
                earnings_growth *
                100
            )


        # ====================================================
        # FUNDAMENTAL SCORE
        # ====================================================

        score = 0

        factors = 0


        # -----------------------------
        # PE
        # -----------------------------

        pe = result[
            "pe_ratio"
        ]


        if pe is not None and pe > 0:

            factors += 1

            if pe <= 20:

                score += 10

            elif pe <= 30:

                score += 7

            elif pe <= 50:

                score += 4


        # -----------------------------
        # ROE
        # -----------------------------

        roe = result[
            "roe"
        ]


        if roe is not None:

            factors += 1

            if roe >= 20:

                score += 10

            elif roe >= 15:

                score += 8

            elif roe >= 10:

                score += 5


        # -----------------------------
        # ROCE
        # -----------------------------

        roce = result[
            "roce"
        ]


        if roce is not None:

            factors += 1

            if roce >= 20:

                score += 10

            elif roce >= 15:

                score += 8

            elif roce >= 10:

                score += 5


        # -----------------------------
        # DEBT
        # -----------------------------

        debt = result[
            "debt_to_equity"
        ]


        if debt is not None:

            factors += 1

            if debt <= 0.5:

                score += 10

            elif debt <= 1:

                score += 7

            elif debt <= 2:

                score += 4


        # -----------------------------
        # CURRENT RATIO
        # -----------------------------

        current = result[
            "current_ratio"
        ]


        if current is not None:

            factors += 1

            if current >= 2:

                score += 10

            elif current >= 1.5:

                score += 8

            elif current >= 1:

                score += 5


        # -----------------------------
        # NET MARGIN
        # -----------------------------

        margin = result[
            "net_profit_margin"
        ]


        if margin is not None:

            factors += 1

            if margin >= 20:

                score += 10

            elif margin >= 10:

                score += 7

            elif margin > 0:

                score += 4


        # -----------------------------
        # REVENUE GROWTH
        # -----------------------------

        growth = result[
            "revenue_growth"
        ]


        if growth is not None:

            factors += 1

            if growth >= 20:

                score += 10

            elif growth >= 10:

                score += 7

            elif growth > 0:

                score += 4


        # -----------------------------
        # EPS GROWTH
        # -----------------------------

        eps_growth = result[
            "eps_growth"
        ]


        if eps_growth is not None:

            factors += 1

            if eps_growth >= 20:

                score += 10

            elif eps_growth >= 10:

                score += 7

            elif eps_growth > 0:

                score += 4


        # ====================================================
        # FINAL FUNDAMENTAL SCORE
        # ====================================================

        if factors > 0:

            result[
                "fundamental_score"
            ] = round(

                (
                    score /
                    (
                        factors *
                        10
                    )
                ) *

                100

            )


        return (
            symbol,
            result
        )


    except Exception as e:

        print(
            f"Fundamentals unavailable: "
            f"{symbol} — {e}"
        )

        return (
            symbol,
            result
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print(
        "=================================================="
    )
    print(
        " CARJ.IN — FULL FUNDAMENTAL ENRICHMENT"
    )
    print(
        "=================================================="
    )
    print("")


    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"{CSV_FILE} does not exist."
        )


    df = pd.read_csv(
        CSV_FILE
    )


    if df.empty:

        raise RuntimeError(
            "Scanner output is empty."
        )


    df = df.reset_index(
        drop=True
    )


    symbols = (

        df[
            "symbol"
        ]

        .astype(str)

        .str.strip()

        .tolist()

    )


    symbols = [
        s for s in symbols
        if s
        and
        s.lower() != "nan"
    ]


    print(
        f"Stocks requiring fundamentals: "
        f"{len(symbols)}"
    )


    # ========================================================
    # COLUMNS
    # ========================================================

    columns = [

        "pe_ratio",
        "pb_ratio",
        "eps",
        "dividend_yield",
        "market_cap",

        "roe",
        "roce",
        "operating_margin",
        "net_profit_margin",

        "debt_to_equity",
        "current_ratio",
        "quick_ratio",
        "interest_coverage",

        "revenue_growth",
        "eps_growth",
        "profit_growth",

        "fundamental_score"

    ]


    for column in columns:

        if column not in df.columns:

            df[column] = np.nan


    # ========================================================
    # PARALLEL FUNDAMENTALS
    # ========================================================

    results = {}


    completed = 0

    total = len(symbols)


    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                get_fundamentals,
                symbol
            ):
            symbol

            for symbol in symbols

        }


        for future in as_completed(
            futures
        ):

            symbol = futures[
                future
            ]


            try:

                _, result = (
                    future.result()
                )

                results[
                    symbol
                ] = result


            except Exception as e:

                print(
                    f"Worker error "
                    f"{symbol}: {e}"
                )

                results[
                    symbol
                ] = {}


            completed += 1


            if (

                completed % 50 == 0

                or

                completed == total

            ):

                print(
                    f"Fundamentals progress: "
                    f"{completed}/{total}"
                )


    # ========================================================
    # UPDATE DATAFRAME
    # ========================================================

    for index in df.index:

        symbol = str(
            df.loc[
                index,
                "symbol"
            ]
        ).strip()


        values = results.get(
            symbol,
            {}
        )


        for column in columns:

            value = values.get(
                column
            )


            if value is not None:

                df.loc[
                    index,
                    column
                ] = value


    # ========================================================
    # FINAL PORTFOLIO SCORE
    #
    # Technical 60%
    # Fundamentals 40%
    # ========================================================

    def calculate_overall(row):

        technical = safe_number(
            row.get(
                "portfolio_technical_score"
            )
        )


        fundamental = safe_number(
            row.get(
                "fundamental_score"
            )
        )


        if (
            technical is None
            and
            fundamental is None
        ):

            return np.nan


        if technical is None:

            return round(
                fundamental,
                2
            )


        if fundamental is None:

            return round(
                technical,
                2
            )


        return round(

            (
                technical *
                0.60
            )

            +

            (
                fundamental *
                0.40
            ),

            2

        )


    df[
        "overall_portfolio_score"
    ] = df.apply(
        calculate_overall,
        axis=1
    )


    # ========================================================
    # FINAL PORTFOLIO ELIGIBILITY
    #
    # Technical + Fundamental
    # ========================================================

    def portfolio_final(row):

        if str(
            row.get(
                "portfolio_eligible"
            )
        ).lower() != "true":

            return False


        score = safe_number(
            row.get(
                "overall_portfolio_score"
            )
        )


        if score is None:

            return False


        return score >= 55


    df[
        "portfolio_final_eligible"
    ] = df.apply(
        portfolio_final,
        axis=1
    )


    # ========================================================
    # SORT
    # ========================================================

    df = (

        df

        .sort_values(

            [
                "portfolio_final_eligible",
                "overall_portfolio_score",
                "portfolio_technical_score",
                "fundamental_score"

            ],

            ascending=[

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


    # ========================================================
    # SAVE CSV
    # ========================================================

    df.to_csv(
        CSV_FILE,
        index=False
    )


    # ========================================================
    # SAVE EXCEL
    # ========================================================

    df.to_excel(
        XLSX_FILE,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    eligible = int(

        df[
            "portfolio_final_eligible"
        ]

        .astype(bool)

        .sum()

    )


    print("")
    print(
        "=================================================="
    )
    print(
        " FUNDAMENTALS COMPLETE"
    )
    print(
        "=================================================="
    )


    print(
        f"Stocks processed: "
        f"{len(df)}"
    )


    print(
        f"Portfolio eligible: "
        f"{eligible}"
    )


    print(
        f"CSV: {CSV_FILE}"
    )


    print(
        f"Excel: {XLSX_FILE}"
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

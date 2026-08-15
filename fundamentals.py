#!/usr/bin/env python3

from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("output")

CSV_FILE = OUTPUT_DIR / "NSE_200DMA_Recovery_Scanner.csv"
XLSX_FILE = OUTPUT_DIR / "NSE_200DMA_Recovery_Scanner.xlsx"

# Number of Yahoo Finance requests running simultaneously.
# 6 is a reasonable balance between speed and reliability.
MAX_WORKERS = 6


# ============================================================
# SAFE NUMBER CONVERSION
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
# FUNDAMENTAL DATA
# ============================================================

def get_fundamentals(symbol):

    result = {

        # -----------------------------
        # VALUATION
        # -----------------------------

        "pe_ratio": None,
        "pb_ratio": None,
        "eps": None,
        "dividend_yield": None,
        "market_cap": None,

        # -----------------------------
        # PROFITABILITY
        # -----------------------------

        "roe": None,
        "roce": None,
        "operating_margin": None,
        "net_profit_margin": None,

        # -----------------------------
        # FINANCIAL HEALTH
        # -----------------------------

        "debt_to_equity": None,
        "current_ratio": None,
        "quick_ratio": None,
        "interest_coverage": None,

        # -----------------------------
        # GROWTH
        # -----------------------------

        "revenue_growth": None,
        "eps_growth": None,
        "profit_growth": None,

        # -----------------------------
        # SCORE
        # -----------------------------

        "fundamental_score": None,

    }

    try:

        print(f"→ {symbol}: fetching fundamentals...")

        ticker = yf.Ticker(f"{symbol}.NS")

        info = ticker.info

        if not info:

            print(
                f"⚠ {symbol}: Yahoo returned no fundamental data"
            )

            return result

        # ====================================================
        # VALUATION
        # ====================================================

        result["pe_ratio"] = safe_number(
            info.get("trailingPE")
        )

        result["pb_ratio"] = safe_number(
            info.get("priceToBook")
        )

        result["eps"] = safe_number(
            info.get("trailingEps")
        )

        dividend = safe_number(
            info.get("dividendYield")
        )

        if dividend is not None:

            result["dividend_yield"] = dividend * 100

        result["market_cap"] = safe_number(
            info.get("marketCap")
        )

        # ====================================================
        # PROFITABILITY
        # ====================================================

        roe = safe_number(
            info.get("returnOnEquity")
        )

        if roe is not None:

            result["roe"] = roe * 100

        roce = safe_number(
            info.get("returnOnCapitalEmployed")
        )

        if roce is not None:

            # Yahoo may return ROCE as decimal or percentage.
            if abs(roce) <= 1:

                roce *= 100

            result["roce"] = roce

        operating_margin = safe_number(
            info.get("operatingMargins")
        )

        if operating_margin is not None:

            result["operating_margin"] = (
                operating_margin * 100
            )

        net_margin = safe_number(
            info.get("profitMargins")
        )

        if net_margin is not None:

            result["net_profit_margin"] = (
                net_margin * 100
            )

        # ====================================================
        # FINANCIAL HEALTH
        # ====================================================

        result["debt_to_equity"] = safe_number(
            info.get("debtToEquity")
        )

        result["current_ratio"] = safe_number(
            info.get("currentRatio")
        )

        result["quick_ratio"] = safe_number(
            info.get("quickRatio")
        )

        result["interest_coverage"] = safe_number(
            info.get("interestCoverage")
        )

        # ====================================================
        # GROWTH
        # ====================================================

        revenue_growth = safe_number(
            info.get("revenueGrowth")
        )

        if revenue_growth is not None:

            result["revenue_growth"] = (
                revenue_growth * 100
            )

        earnings_growth = safe_number(
            info.get("earningsGrowth")
        )

        if earnings_growth is not None:

            result["eps_growth"] = (
                earnings_growth * 100
            )

            result["profit_growth"] = (
                earnings_growth * 100
            )

        # ====================================================
        # FUNDAMENTAL SCORE
        # ====================================================

        score = 0
        factors = 0

        # ----------------------------------------------------
        # P/E
        # ----------------------------------------------------

        pe = result["pe_ratio"]

        if pe is not None and pe > 0:

            factors += 1

            if pe <= 20:

                score += 10

            elif pe <= 30:

                score += 7

            elif pe <= 50:

                score += 4

        # ----------------------------------------------------
        # ROE
        # ----------------------------------------------------

        roe = result["roe"]

        if roe is not None:

            factors += 1

            if roe >= 20:

                score += 10

            elif roe >= 15:

                score += 8

            elif roe >= 10:

                score += 5

        # ----------------------------------------------------
        # ROCE
        # ----------------------------------------------------

        roce = result["roce"]

        if roce is not None:

            factors += 1

            if roce >= 20:

                score += 10

            elif roce >= 15:

                score += 8

            elif roce >= 10:

                score += 5

        # ----------------------------------------------------
        # DEBT / EQUITY
        # ----------------------------------------------------

        debt_equity = result["debt_to_equity"]

        if debt_equity is not None:

            factors += 1

            if debt_equity <= 0.5:

                score += 10

            elif debt_equity <= 1:

                score += 7

            elif debt_equity <= 2:

                score += 4

        # ----------------------------------------------------
        # CURRENT RATIO
        # ----------------------------------------------------

        current_ratio = result["current_ratio"]

        if current_ratio is not None:

            factors += 1

            if current_ratio >= 2:

                score += 10

            elif current_ratio >= 1.5:

                score += 8

            elif current_ratio >= 1:

                score += 5

        # ----------------------------------------------------
        # NET PROFIT MARGIN
        # ----------------------------------------------------

        margin = result["net_profit_margin"]

        if margin is not None:

            factors += 1

            if margin >= 20:

                score += 10

            elif margin >= 10:

                score += 7

            elif margin > 0:

                score += 4

        # ----------------------------------------------------
        # REVENUE GROWTH
        # ----------------------------------------------------

        growth = result["revenue_growth"]

        if growth is not None:

            factors += 1

            if growth >= 20:

                score += 10

            elif growth >= 10:

                score += 7

            elif growth > 0:

                score += 4

        # ----------------------------------------------------
        # EPS GROWTH
        # ----------------------------------------------------

        eps_growth = result["eps_growth"]

        if eps_growth is not None:

            factors += 1

            if eps_growth >= 20:

                score += 10

            elif eps_growth >= 10:

                score += 7

            elif eps_growth > 0:

                score += 4

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        if factors > 0:

            result["fundamental_score"] = round(
                (score / (factors * 10)) * 100
            )

        print(
            f"✓ {symbol}: Fundamental Score = "
            f"{result['fundamental_score']}"
        )

    except Exception as e:

        print(
            f"⚠ {symbol}: fundamentals unavailable - {e}"
        )

    return result


# ============================================================
# WORKER
# ============================================================

def fetch_one(index, symbol):

    fundamentals = get_fundamentals(symbol)

    return index, symbol, fundamentals


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("==============================================")
    print(" NSE 200 DMA — FUNDAMENTAL ENRICHMENT")
    print("==============================================")
    print("")

    # --------------------------------------------------------
    # Check CSV
    # --------------------------------------------------------

    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"{CSV_FILE} does not exist."
        )

    # --------------------------------------------------------
    # Load scanner output
    # --------------------------------------------------------

    df = pd.read_csv(CSV_FILE)

    if df.empty:

        print("No scanner results found.")

        return

    # --------------------------------------------------------
    # Validate symbol column
    # --------------------------------------------------------

    if "symbol" not in df.columns:

        raise RuntimeError(
            "Scanner CSV does not contain a 'symbol' column."
        )

    df = df.reset_index(drop=True)

    target_count = len(df)

    print(
        f"Scanner candidates found: {target_count}"
    )

    print(
        f"Fetching fundamentals for ALL "
        f"{target_count} stocks..."
    )

    print(
        f"Parallel workers: {MAX_WORKERS}"
    )

    print("")

    # --------------------------------------------------------
    # Fundamental columns
    # --------------------------------------------------------

    fundamental_columns = [

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

        "fundamental_score",

    ]

    # --------------------------------------------------------
    # Create columns if missing
    # --------------------------------------------------------

    for column in fundamental_columns:

        if column not in df.columns:

            df[column] = None

    # --------------------------------------------------------
    # Build jobs
    # --------------------------------------------------------

    jobs = []

    for index in range(target_count):

        symbol = str(
            df.loc[index, "symbol"]
        ).strip()

        if symbol and symbol.lower() != "nan":

            jobs.append(
                (
                    index,
                    symbol
                )
            )

    print(
        f"Valid stocks to enrich: {len(jobs)}"
    )

    print("")

    # --------------------------------------------------------
    # Parallel Yahoo Finance requests
    # --------------------------------------------------------

    completed = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                fetch_one,
                index,
                symbol
            ): (
                index,
                symbol
            )

            for index, symbol in jobs

        }

        for future in as_completed(futures):

            index, symbol = futures[future]

            try:

                (
                    result_index,
                    result_symbol,
                    fundamentals
                ) = future.result()

                # --------------------------------------------
                # Write fundamentals into dataframe
                # --------------------------------------------

                for column, value in fundamentals.items():

                    df.loc[
                        result_index,
                        column
                    ] = value

                completed += 1

                print(
                    f"[{completed}/{len(jobs)}] "
                    f"✓ {result_symbol}"
                )

            except Exception as e:

                completed += 1

                print(
                    f"[{completed}/{len(jobs)}] "
                    f"⚠ {symbol}: worker failed - {e}"
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

    available_scores = (
        df["fundamental_score"]
        .notna()
        .sum()
    )

    print("")
    print("==============================================")
    print(" FUNDAMENTALS UPDATED")
    print("==============================================")
    print("")

    print(
        f"Scanner candidates : {target_count}"
    )

    print(
        f"Stocks processed   : {completed}"
    )

    print(
        f"Scores available   : {available_scores}"
    )

    print(
        f"Scores unavailable : "
        f"{target_count - available_scores}"
    )

    print("")

    print(
        f"CSV saved : {CSV_FILE}"
    )

    print(
        f"Excel saved : {XLSX_FILE}"
    )

    print("")

    print("Fundamental enrichment completed.")

    print("")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    main()

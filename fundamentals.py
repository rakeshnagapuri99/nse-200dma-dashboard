#!/usr/bin/env python3

from pathlib import Path
import warnings
import pandas as pd
import yfinance as yf

OUTPUT_DIR = Path("output")
CSV_FILE = OUTPUT_DIR / "NSE_200DMA_Recovery_Scanner.csv"
XLSX_FILE = OUTPUT_DIR / "NSE_200DMA_Recovery_Scanner.xlsx"

# Only enrich the stocks actually displayed/needed.
TOP_N = 20


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


def get_fundamentals(symbol):

    result = {
        "pe_ratio": None,
        "pb_ratio": None,
        "eps": None,
        "dividend_yield": None,
        "market_cap": None,

        "roe": None,
        "roce": None,
        "operating_margin": None,
        "net_profit_margin": None,

        "debt_to_equity": None,
        "current_ratio": None,
        "quick_ratio": None,
        "interest_coverage": None,

        "revenue_growth": None,
        "eps_growth": None,
        "profit_growth": None,

        "fundamental_score": None,
    }

    try:

        ticker = yf.Ticker(f"{symbol}.NS")

        info = ticker.info

        # -----------------------------
        # VALUATION
        # -----------------------------

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

        # -----------------------------
        # PROFITABILITY
        # -----------------------------

        roe = safe_number(
            info.get("returnOnEquity")
        )

        if roe is not None:
            result["roe"] = roe * 100

        operating_margin = safe_number(
            info.get("operatingMargins")
        )

        if operating_margin is not None:
            result["operating_margin"] = operating_margin * 100

        net_margin = safe_number(
            info.get("profitMargins")
        )

        if net_margin is not None:
            result["net_profit_margin"] = net_margin * 100

        roce = safe_number(
            info.get("returnOnCapitalEmployed")
        )

        if roce is not None:

            if abs(roce) <= 1:
                roce *= 100

            result["roce"] = roce

        # -----------------------------
        # FINANCIAL HEALTH
        # -----------------------------

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

        # -----------------------------
        # GROWTH
        # -----------------------------

        revenue_growth = safe_number(
            info.get("revenueGrowth")
        )

        if revenue_growth is not None:
            result["revenue_growth"] = revenue_growth * 100

        earnings_growth = safe_number(
            info.get("earningsGrowth")
        )

        if earnings_growth is not None:

            result["eps_growth"] = earnings_growth * 100
            result["profit_growth"] = earnings_growth * 100

        # -----------------------------
        # FUNDAMENTAL SCORE
        # -----------------------------

        score = 0
        factors = 0

        pe = result["pe_ratio"]

        if pe is not None and pe > 0:

            factors += 1

            if pe <= 20:
                score += 10
            elif pe <= 30:
                score += 7
            elif pe <= 50:
                score += 4

        roe = result["roe"]

        if roe is not None:

            factors += 1

            if roe >= 20:
                score += 10
            elif roe >= 15:
                score += 8
            elif roe >= 10:
                score += 5

        roce = result["roce"]

        if roce is not None:

            factors += 1

            if roce >= 20:
                score += 10
            elif roce >= 15:
                score += 8
            elif roce >= 10:
                score += 5

        de = result["debt_to_equity"]

        if de is not None:

            factors += 1

            if de <= 0.5:
                score += 10
            elif de <= 1:
                score += 7
            elif de <= 2:
                score += 4

        current = result["current_ratio"]

        if current is not None:

            factors += 1

            if current >= 2:
                score += 10
            elif current >= 1.5:
                score += 8
            elif current >= 1:
                score += 5

        margin = result["net_profit_margin"]

        if margin is not None:

            factors += 1

            if margin >= 20:
                score += 10
            elif margin >= 10:
                score += 7
            elif margin > 0:
                score += 4

        growth = result["revenue_growth"]

        if growth is not None:

            factors += 1

            if growth >= 20:
                score += 10
            elif growth >= 10:
                score += 7
            elif growth > 0:
                score += 4

        eps_growth = result["eps_growth"]

        if eps_growth is not None:

            factors += 1

            if eps_growth >= 20:
                score += 10
            elif eps_growth >= 10:
                score += 7
            elif eps_growth > 0:
                score += 4

        if factors > 0:

            result["fundamental_score"] = round(
                (score / (factors * 10)) * 100
            )

        print(
            f"✓ {symbol}: "
            f"Fundamental Score = "
            f"{result['fundamental_score']}"
        )

    except Exception as e:

        print(
            f"⚠ {symbol}: fundamentals unavailable - {e}"
        )

    return result


def main():

    print("")
    print("======================================")
    print(" FAST FUNDAMENTAL ENRICHMENT")
    print("======================================")
    print("")

    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"{CSV_FILE} does not exist."
        )

    df = pd.read_csv(CSV_FILE)

    if df.empty:

        print("No scanner results.")
        return

    # ---------------------------------
    # IMPORTANT:
    # Scanner output is already ranked.
    # We only need the first 20.
    # ---------------------------------

    df = df.reset_index(drop=True)

    target_count = min(
        TOP_N,
        len(df)
    )

    print(
        f"Fetching fundamentals for "
        f"TOP {target_count} stocks only..."
    )

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

    for column in fundamental_columns:

        if column not in df.columns:

            df[column] = None

    # ---------------------------------
    # Fetch only Top 20
    # ---------------------------------

    for index in range(target_count):

        symbol = str(
            df.loc[index, "symbol"]
        ).strip()

        fundamentals = get_fundamentals(
            symbol
        )

        for column, value in fundamentals.items():

            df.loc[
                index,
                column
            ] = value

    # ---------------------------------
    # Save
    # ---------------------------------

    df.to_csv(
        CSV_FILE,
        index=False
    )

    df.to_excel(
        XLSX_FILE,
        index=False
    )

    print("")
    print("======================================")
    print(" FUNDAMENTALS UPDATED")
    print("======================================")
    print("")
    print(
        f"Stocks enriched: {target_count}"
    )
    print(
        f"CSV: {CSV_FILE}"
    )
    print(
        f"Excel: {XLSX_FILE}"
    )
    print("")
    print("Done.")


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore"
    )

    main()

#!/usr/bin/env python3

"""
NSE SMART PORTFOLIO BUILDER

Builds a research portfolio from the NSE scanner output.

Main logic:
1. Stock must be above 200 DMA.
2. Stock must be near / above its 52-week high.
3. Technical score is calculated.
4. Fundamental score is calculated.
5. Combined score = 60% Technical + 40% Fundamental.
6. Stocks are ranked.
7. Default portfolio = 10 stocks.
8. Default portfolio amount = ₹10,00,000.
9. Equal allocation by default.
10. Entry price = current price.
11. Stop loss = 10% below entry.
12. Trend-based target is estimated.
13. Expected gain and maximum loss are calculated.

This is a research/screening model and not investment advice.
"""

from pathlib import Path
import math
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

OUTPUT_DIR = Path("output")

INPUT_FILE = OUTPUT_DIR / "NSE_200DMA_Recovery_Scanner.csv"

PORTFOLIO_FILE = OUTPUT_DIR / "NSE_Portfolio_Recommendations.csv"

PORTFOLIO_XLSX = OUTPUT_DIR / "NSE_Portfolio_Recommendations.xlsx"


# =========================================================
# DEFAULT SETTINGS
# =========================================================

DEFAULT_PORTFOLIO_AMOUNT = 1_000_000

DEFAULT_STOCK_COUNT = 10

STOP_LOSS_PERCENT = 10.0

NEAR_HIGH_PERCENT = 5.0


# =========================================================
# SAFE NUMBER
# =========================================================

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


# =========================================================
# TECHNICAL SCORE
# =========================================================

def calculate_technical_score(row):

    score = 0.0
    factors = 0

    # -----------------------------------------------------
    # Existing scanner technical score
    # -----------------------------------------------------

    technical = safe_number(row.get("score"))

    if technical is not None:

        score += technical
        factors += 1

    # -----------------------------------------------------
    # Distance from 52-week high
    # -----------------------------------------------------

    high_gap = safe_number(
        row.get("distance_to_prior_high_pct")
    )

    if high_gap is not None:

        if high_gap >= 0:

            high_score = 100

        elif high_gap >= -1:

            high_score = 95

        elif high_gap >= -2:

            high_score = 90

        elif high_gap >= -3:

            high_score = 80

        elif high_gap >= -5:

            high_score = 70

        else:

            high_score = 40

        score += high_score
        factors += 1

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    rsi = safe_number(
        row.get("rsi_14")
    )

    if rsi is not None:

        if 55 <= rsi <= 70:

            rsi_score = 100

        elif 50 <= rsi < 55:

            rsi_score = 85

        elif 70 < rsi <= 75:

            rsi_score = 80

        elif 45 <= rsi < 50:

            rsi_score = 65

        else:

            rsi_score = 45

        score += rsi_score
        factors += 1

    # -----------------------------------------------------
    # Volume
    # -----------------------------------------------------

    volume = safe_number(
        row.get("volume_ratio")
    )

    if volume is not None:

        if volume >= 2:

            volume_score = 100

        elif volume >= 1.5:

            volume_score = 90

        elif volume >= 1.2:

            volume_score = 80

        elif volume >= 1:

            volume_score = 65

        else:

            volume_score = 45

        score += volume_score
        factors += 1

    if factors == 0:

        return 0.0

    return round(
        score / factors,
        2
    )


# =========================================================
# FUNDAMENTAL SCORE
# =========================================================

def calculate_fundamental_score(row):

    existing = safe_number(
        row.get("fundamental_score")
    )

    if existing is not None:

        return round(
            max(
                0.0,
                min(
                    100.0,
                    existing
                )
            ),
            2
        )

    score = 0.0
    factors = 0

    # -----------------------------------------------------
    # ROE
    # -----------------------------------------------------

    roe = safe_number(
        row.get("roe")
    )

    if roe is not None:

        factors += 1

        if roe >= 20:
            score += 100

        elif roe >= 15:
            score += 85

        elif roe >= 10:
            score += 70

        elif roe > 0:
            score += 50

        else:
            score += 20

    # -----------------------------------------------------
    # ROCE
    # -----------------------------------------------------

    roce = safe_number(
        row.get("roce")
    )

    if roce is not None:

        factors += 1

        if roce >= 20:
            score += 100

        elif roce >= 15:
            score += 85

        elif roce >= 10:
            score += 70

        elif roce > 0:
            score += 50

        else:
            score += 20

    # -----------------------------------------------------
    # Debt / Equity
    # -----------------------------------------------------

    debt = safe_number(
        row.get("debt_to_equity")
    )

    if debt is not None:

        factors += 1

        if debt <= 0.5:
            score += 100

        elif debt <= 1:
            score += 85

        elif debt <= 2:
            score += 65

        else:
            score += 30

    # -----------------------------------------------------
    # Revenue Growth
    # -----------------------------------------------------

    growth = safe_number(
        row.get("revenue_growth")
    )

    if growth is not None:

        factors += 1

        if growth >= 20:
            score += 100

        elif growth >= 10:
            score += 80

        elif growth > 0:
            score += 60

        else:
            score += 30

    if factors == 0:

        return 0.0

    return round(
        score / factors,
        2
    )


# =========================================================
# STOCK ELIGIBILITY
# =========================================================

def is_eligible(row):

    close = safe_number(
        row.get("close")
    )

    dma_200 = safe_number(
        row.get("dma_200")
    )

    high_gap = safe_number(
        row.get("distance_to_prior_high_pct")
    )

    # -----------------------------------------------------
    # Required data
    # -----------------------------------------------------

    if close is None:
        return False

    if dma_200 is None:
        return False

    if high_gap is None:
        return False

    # -----------------------------------------------------
    # Must be ABOVE 200 DMA
    # -----------------------------------------------------

    if close <= dma_200:
        return False

    # -----------------------------------------------------
    # Must be within 5% of 52-week high
    # OR above the high
    # -----------------------------------------------------

    if high_gap < -NEAR_HIGH_PERCENT:
        return False

    return True


# =========================================================
# TREND TARGET
# =========================================================

def calculate_trend_target(row):

    close = safe_number(
        row.get("close")
    )

    if close is None or close <= 0:
        return None

    dma_200 = safe_number(
        row.get("dma_200")
    )

    rsi = safe_number(
        row.get("rsi_14")
    )

    volume = safe_number(
        row.get("volume_ratio")
    )

    technical = safe_number(
        row.get("score")
    )

    fundamental = safe_number(
        row.get("fundamental_score")
    )

    # -----------------------------------------------------
    # Base expected trend
    # -----------------------------------------------------

    expected_return = 8.0

    # -----------------------------------------------------
    # Technical strength
    # -----------------------------------------------------

    if technical is not None:

        if technical >= 80:
            expected_return += 7

        elif technical >= 65:
            expected_return += 5

        elif technical >= 55:
            expected_return += 3

    # -----------------------------------------------------
    # Fundamental strength
    # -----------------------------------------------------

    if fundamental is not None:

        if fundamental >= 80:
            expected_return += 5

        elif fundamental >= 65:
            expected_return += 3

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if rsi is not None:

        if 55 <= rsi <= 70:
            expected_return += 3

        elif rsi > 75:
            expected_return -= 2

    # -----------------------------------------------------
    # Volume
    # -----------------------------------------------------

    if volume is not None:

        if volume >= 1.5:
            expected_return += 3

        elif volume >= 1.2:
            expected_return += 1

    # -----------------------------------------------------
    # Distance above 200 DMA
    # -----------------------------------------------------

    if dma_200 is not None and dma_200 > 0:

        dma_gap = (
            (close / dma_200) - 1
        ) * 100

        if dma_gap >= 20:

            expected_return += 2

        elif dma_gap >= 10:

            expected_return += 1

    # -----------------------------------------------------
    # Limit target expectation
    # -----------------------------------------------------

    expected_return = max(
        5.0,
        min(
            expected_return,
            30.0
        )
    )

    target = close * (
        1 + expected_return / 100
    )

    return round(
        target,
        2
    )


# =========================================================
# RISK CLASSIFICATION
# =========================================================

def risk_classification(
    combined_score,
    expected_return
):

    if (
        combined_score >= 80
        and expected_return >= 15
    ):

        return "HIGH CONVICTION"

    if (
        combined_score >= 65
        and expected_return >= 10
    ):

        return "MODERATE CONVICTION"

    if combined_score >= 50:

        return "WATCH"

    return "HIGH RISK"


# =========================================================
# BUILD PORTFOLIO
# =========================================================

def build_portfolio(
    df,
    portfolio_amount=DEFAULT_PORTFOLIO_AMOUNT,
    stock_count=DEFAULT_STOCK_COUNT
):

    print("")
    print("=================================================")
    print(" NSE SMART PORTFOLIO BUILDER")
    print("=================================================")
    print("")

    print(
        f"Portfolio amount : ₹{portfolio_amount:,.2f}"
    )

    print(
        f"Stocks required  : {stock_count}"
    )

    print(
        f"Near 52W high    : {NEAR_HIGH_PERCENT}%"
    )

    print(
        f"Stop loss        : {STOP_LOSS_PERCENT}%"
    )

    print("")

    # -----------------------------------------------------
    # Find eligible stocks
    # -----------------------------------------------------

    eligible_rows = []

    for _, row in df.iterrows():

        if is_eligible(row):

            eligible_rows.append(row)

    print(
        f"Eligible stocks: {len(eligible_rows)}"
    )

    if not eligible_rows:

        print(
            "No eligible stocks found."
        )

        return pd.DataFrame()

    eligible_df = pd.DataFrame(
        eligible_rows
    )

    # -----------------------------------------------------
    # Technical score
    # -----------------------------------------------------

    eligible_df[
        "calculated_technical_score"
    ] = eligible_df.apply(
        calculate_technical_score,
        axis=1
    )

    # -----------------------------------------------------
    # Fundamental score
    # -----------------------------------------------------

    eligible_df[
        "calculated_fundamental_score"
    ] = eligible_df.apply(
        calculate_fundamental_score,
        axis=1
    )

    # -----------------------------------------------------
    # Combined score
    #
    # 60% Technical
    # 40% Fundamental
    # -----------------------------------------------------

    eligible_df[
        "combined_score"
    ] = (
        eligible_df[
            "calculated_technical_score"
        ] * 0.60
        +
        eligible_df[
            "calculated_fundamental_score"
        ] * 0.40
    ).round(2)

    # -----------------------------------------------------
    # Rank
    # -----------------------------------------------------

    eligible_df = eligible_df.sort_values(
        [
            "combined_score",
            "calculated_technical_score",
            "calculated_fundamental_score"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Select requested number
    # -----------------------------------------------------

    selected = eligible_df.head(
        stock_count
    ).copy()

    if selected.empty:

        return pd.DataFrame()

    # -----------------------------------------------------
    # Equal allocation
    # -----------------------------------------------------

    allocation = (
        portfolio_amount /
        len(selected)
    )

    selected[
        "allocation_amount"
    ] = round(
        allocation,
        2
    )

    # -----------------------------------------------------
    # Entry price
    # -----------------------------------------------------

    selected[
        "entry_price"
    ] = selected[
        "close"
    ].apply(
        safe_number
    )

    # -----------------------------------------------------
    # Quantity
    # -----------------------------------------------------

    selected[
        "quantity"
    ] = selected.apply(
        lambda row:
        math.floor(
            row["allocation_amount"]
            /
            row["entry_price"]
        )
        if (
            row["entry_price"] is not None
            and row["entry_price"] > 0
        )
        else 0,
        axis=1
    )

    # -----------------------------------------------------
    # Actual investment
    # -----------------------------------------------------

    selected[
        "actual_investment"
    ] = (
        selected["quantity"]
        *
        selected["entry_price"]
    ).round(2)

    # -----------------------------------------------------
    # Stop loss
    # -----------------------------------------------------

    selected[
        "stop_loss"
    ] = (
        selected["entry_price"]
        *
        (
            1 -
            STOP_LOSS_PERCENT / 100
        )
    ).round(2)

    # -----------------------------------------------------
    # Maximum position loss
    # -----------------------------------------------------

    selected[
        "maximum_loss"
    ] = (
        selected["actual_investment"]
        *
        STOP_LOSS_PERCENT
        /
        100
    ).round(2)

    # -----------------------------------------------------
    # Trend target
    # -----------------------------------------------------

    selected[
        "trend_target"
    ] = selected.apply(
        calculate_trend_target,
        axis=1
    )

    # -----------------------------------------------------
    # Expected gain %
    # -----------------------------------------------------

    selected[
        "expected_gain_percent"
    ] = (
        (
            selected["trend_target"]
            /
            selected["entry_price"]
        )
        - 1
    ) * 100

    selected[
        "expected_gain_percent"
    ] = selected[
        "expected_gain_percent"
    ].round(2)

    # -----------------------------------------------------
    # Expected gain amount
    # -----------------------------------------------------

    selected[
        "expected_gain_amount"
    ] = (
        selected["actual_investment"]
        *
        selected["expected_gain_percent"]
        /
        100
    ).round(2)

    # -----------------------------------------------------
    # Risk classification
    # -----------------------------------------------------

    selected[
        "risk_classification"
    ] = selected.apply(
        lambda row:
        risk_classification(
            row["combined_score"],
            row["expected_gain_percent"]
        ),
        axis=1
    )

    # -----------------------------------------------------
    # Portfolio rank
    # -----------------------------------------------------

    selected.insert(
        0,
        "portfolio_rank",
        range(
            1,
            len(selected) + 1
        )
    )

    # -----------------------------------------------------
    # Portfolio weight
    # -----------------------------------------------------

    selected[
        "portfolio_weight_percent"
    ] = (
        selected["actual_investment"]
        /
        portfolio_amount
        *
        100
    ).round(2)

    # -----------------------------------------------------
    # Output columns
    # -----------------------------------------------------

    output_columns = [

        "portfolio_rank",

        "symbol",

        "date",

        "close",

        "dma_200",

        "distance_to_200dma_pct",

        "prior_high_before_breakdown",

        "distance_to_prior_high_pct",

        "rsi_14",

        "volume_ratio",

        "calculated_technical_score",

        "calculated_fundamental_score",

        "combined_score",

        "entry_price",

        "quantity",

        "allocation_amount",

        "actual_investment",

        "portfolio_weight_percent",

        "stop_loss",

        "maximum_loss",

        "trend_target",

        "expected_gain_percent",

        "expected_gain_amount",

        "risk_classification",

        "setup"
    ]

    # Only retain columns that actually exist.

    output_columns = [
        column
        for column in output_columns
        if column in selected.columns
    ]

    result = selected[
        output_columns
    ].copy()

    return result


# =========================================================
# SAVE OUTPUT
# =========================================================

def save_output(portfolio):

    if portfolio.empty:

        return

    portfolio.to_csv(
        PORTFOLIO_FILE,
        index=False
    )

    portfolio.to_excel(
        PORTFOLIO_XLSX,
        index=False
    )

    print("")

    print(
        "Portfolio files created:"
    )

    print(
        f"CSV  : {PORTFOLIO_FILE}"
    )

    print(
        f"Excel: {PORTFOLIO_XLSX}"
    )


# =========================================================
# PORTFOLIO SUMMARY
# =========================================================

def print_summary(
    portfolio,
    portfolio_amount
):

    if portfolio.empty:

        return

    total_invested = portfolio[
        "actual_investment"
    ].sum()

    total_expected_gain = portfolio[
        "expected_gain_amount"
    ].sum()

    total_maximum_loss = portfolio[
        "maximum_loss"
    ].sum()

    if total_invested > 0:

        expected_return = (
            total_expected_gain
            /
            total_invested
            *
            100
        )

        maximum_loss_percent = (
            total_maximum_loss
            /
            total_invested
            *
            100
        )

    else:

        expected_return = 0

        maximum_loss_percent = 0

    print("")

    print("=================================================")
    print(" PORTFOLIO SUMMARY")
    print("=================================================")
    print("")

    print(
        f"Stocks selected    : {len(portfolio)}"
    )

    print(
        f"Portfolio amount   : ₹{portfolio_amount:,.2f}"
    )

    print(
        f"Actual investment  : ₹{total_invested:,.2f}"
    )

    print(
        f"Expected gain      : ₹{total_expected_gain:,.2f}"
    )

    print(
        f"Expected return    : {expected_return:.2f}%"
    )

    print(
        f"Maximum loss       : ₹{total_maximum_loss:,.2f}"
    )

    print(
        f"Maximum loss %     : {maximum_loss_percent:.2f}%"
    )

    print("")

    print("=================================================")
    print(" SELECTED STOCKS")
    print("=================================================")
    print("")

    print(
        portfolio.to_string(
            index=False
        )
    )

    print("")


# =========================================================
# MAIN
# =========================================================

def main():

    print("")
    print("Loading scanner data...")

    # -----------------------------------------------------
    # Check input file
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # -----------------------------------------------------
    # Read scanner data
    # -----------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    if df.empty:

        print(
            "Scanner file is empty."
        )

        return

    print(
        f"Scanner rows loaded: {len(df)}"
    )

    # -----------------------------------------------------
    # Build portfolio
    # -----------------------------------------------------

    portfolio = build_portfolio(

        df=df,

        portfolio_amount=
            DEFAULT_PORTFOLIO_AMOUNT,

        stock_count=
            DEFAULT_STOCK_COUNT

    )

    # -----------------------------------------------------
    # No portfolio
    # -----------------------------------------------------

    if portfolio.empty:

        print(
            "Portfolio could not be generated."
        )

        return

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_output(
        portfolio
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print_summary(

        portfolio,

        DEFAULT_PORTFOLIO_AMOUNT

    )

    print("")
    print("Portfolio Builder completed successfully.")
    print("")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()

#!/usr/bin/env python3

"""
NSE SMART PORTFOLIO BUILDER

Purpose:
- Read the complete NSE scanner output
- Identify stocks near their 52-week high
- Require price to be above 200 DMA
- Combine technical + fundamental scores
- Rank eligible stocks
- Build an equal-allocation portfolio
- Calculate entry price
- Calculate 10% stop loss
- Estimate trend-based target
- Calculate expected gain and maximum loss

Research / screening tool only.
Not investment advice.
"""

from pathlib import Path
import math
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

OUTPUT_DIR = Path("output")

INPUT_FILE = (
    OUTPUT_DIR /
    "NSE_200DMA_Recovery_Scanner.csv"
)

PORTFOLIO_FILE = (
    OUTPUT_DIR /
    "NSE_Portfolio_Recommendations.csv"
)

PORTFOLIO_XLSX = (
    OUTPUT_DIR /
    "NSE_Portfolio_Recommendations.xlsx"
)


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
# NORMALIZE SCORE
# =========================================================

def score_value(value):

    value = safe_number(value)

    if value is None:
        return 0.0

    return max(
        0.0,
        min(
            100.0,
            value
        )
    )


# =========================================================
# TECHNICAL SCORE
# =========================================================

def calculate_technical_score(row):

    score = 0.0

    factors = 0

    # -----------------------------------------------------
    # Existing scanner technical score
    # -----------------------------------------------------

    technical = safe_number(
        row.get("score")
    )

    if technical is not None:

        score += technical

        factors += 1


    # -----------------------------------------------------
    # Distance from 52-week / prior high
    # -----------------------------------------------------

    high_gap = safe_number(
        row.get(
            "distance_to_prior_high_pct"
        )
    )

    if high_gap is not None:

        # Ideal zone:
        # 0% to 5% below high
        #
        # Also allow stocks already above
        # the previous high.

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
        row.get(
            "fundamental_score"
        )
    )

    if existing is not None:

        return round(
            score_value(existing),
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
# ELIGIBILITY
# =========================================================

def is_eligible(row):

    close = safe_number(
        row.get("close")
    )

    dma_200 = safe_number(
        row.get("dma_200")
    )

    high_gap = safe_number(
        row.get(
            "distance_to_prior_high_pct"
        )
    )


    # -----------------------------------------------------
    # Must have valid price and 200 DMA
    # -----------------------------------------------------

    if close is None:

        return False


    if dma_200 is None:

        return False


    # -----------------------------------------------------
    # Current price must be above 200 DMA
    # -----------------------------------------------------

    if close <= dma_200:

        return False


    # -----------------------------------------------------
    # Must be near or above prior high
    #
    # Example:
    #
    # Prior high = 530
    # Current price = 520
    #
    # Gap = -1.89%
    #
    # Eligible.
    # -----------------------------------------------------

    if high_gap is None:

        return False


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
    # Base trend potential
    #
    # This is deliberately conservative.
    # -----------------------------------------------------

    expected_return = 8.0


    # Technical strength

    if technical is not None:

        if technical >= 80:

            expected_return += 7

        elif technical >= 65:

            expected_return += 5

        elif technical >= 55:

            expected_return += 3


    # Fundamental strength

    if fundamental is not None:

        if fundamental >= 80:

            expected_return += 5

        elif fundamental >= 65:

            expected_return += 3


    # RSI confirmation

    if rsi is not None:

        if 55 <= rsi <= 70:

            expected_return += 3

        elif rsi > 75:

            expected_return -= 2


    # Volume confirmation

    if volume is not None:

        if volume >= 1.5:

            expected_return += 3

        elif volume >= 1.2:

            expected_return += 1


    # Strong distance above 200 DMA
    # gives additional trend confirmation.

    if dma_200 is not None:

        dma_gap = (
            close /
            dma_200
            - 1
        ) * 100


        if dma_gap >= 20:

            expected_return += 2

        elif dma_gap >= 10:

            expected_return += 1


    # Limit model target

    expected_return = max(
        5.0,
        min(
            expected_return,
            30.0
        )
    )


    target = (

        close *
        (
            1 +
            expected_return / 100
        )

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
    print(
        "================================================="
    )

    print(
        " NSE SMART PORTFOLIO BUILDER"
    )

    print(
        "================================================="
    )

    print("")

    print(
        f"Portfolio amount : ₹{portfolio_amount:,.2f}"
    )

    print(
        f"Stocks required  : {stock_count}"
    )

    print(
        f"Near high range  : {NEAR_HIGH_PERCENT}%"
    )

    print(
        f"Stop loss        : {STOP_LOSS_PERCENT}%"
    )

    print("")


    # -----------------------------------------------------
    # Eligibility
    # -----------------------------------------------------

    eligible = [

        row

        for _, row
        in df.iterrows()

        if is_eligible(row)

    ]


    print(
        f"Eligible stocks: {len(eligible)}"
    )


    if not eligible:

        print(
            "No eligible stocks found."
        )

        return pd.DataFrame()


    eligible_df =
        pd.DataFrame(eligible)


    # -----------------------------------------------------
    # Scores
    # -----------------------------------------------------

    eligible_df[
        "calculated_technical_score"
    ] = eligible_df.apply(
        calculate_technical_score,
        axis=1
    )


    eligible_df[
        "calculated_fundamental_score"
    ] = eligible_df.apply(
        calculate_fundamental_score,
        axis=1
    )


    eligible_df[
        "combined_score"
    ] = (

        eligible_df[
            "calculated_technical_score"
        ]

        * 0.60

        +

        eligible_df[
            "calculated_fundamental_score"
        ]

        * 0.40

    ).round(2)


    # -----------------------------------------------------
    # Rank
    # -----------------------------------------------------

    eligible_df = (

        eligible_df

        .sort_values(
            [
                "combined_score",
                "calculated_technical_score",
                "calculated_fundamental_score"
            ],

            ascending=False
        )

        .reset_index(drop=True)

    )


    # -----------------------------------------------------
    # Select stocks
    # -----------------------------------------------------

    selected = (

        eligible_df

        .head(stock_count)

        .copy()

    )


    # -----------------------------------------------------
    # Equal allocation
    # -----------------------------------------------------

    allocation = (

        portfolio_amount /
        len(selected)

    )


    selected[
        "allocation_amount"
    ] = allocation


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
    ] = (

        selected[
            "allocation_amount"
        ]

        /

        selected[
            "entry_price"
        ]

    ).apply(
        math.floor
    )


    # -----------------------------------------------------
    # Actual investment
    # -----------------------------------------------------

    selected[
        "actual_investment"
    ] = (

        selected[
            "quantity"
        ]

        *

        selected[
            "entry_price"
        ]

    ).round(2)


    # -----------------------------------------------------
    # Stop loss
    # -----------------------------------------------------

    selected[
        "stop_loss"
    ] = (

        selected[
            "entry_price"
        ]

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

        selected[
            "actual_investment"
        ]

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
    # Expected return
    # -----------------------------------------------------

    selected[
        "expected_gain_percent"
    ] = (

        (

            selected[
                "trend_target"
            ]

            /

            selected[
                "entry_price"
            ]

            - 1

        )

        *

        100

    ).round(2)


    # -----------------------------------------------------
    # Expected gain amount
    # -----------------------------------------------------

    selected[
        "expected_gain_amount"
    ] = (

        selected[
            "actual_investment"
        ]

        *

        selected[
            "expected_gain_percent"
        ]

        /

        100

    ).round(2)


    # -----------------------------------------------------
    # Risk classification
    # -----------------------------------------------------

    selected[
        "risk_classification"
    ] = selected.apply(

        lambda r:

        risk_classification(

            r[
                "combined_score"
            ],

            r[
                "expected_gain_percent"
            ]

        ),

        axis=1

    )


    # -----------------------------------------------------
    # Rank
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
    # Position weight
    # -----------------------------------------------------

    selected[
        "portfolio_weight_percent"
    ] = (

        selected[
            "actual_investment"
        ]

        /

        portfolio_amount

        *

        100

    ).round(2)


    # -----------------------------------------------------
    # Clean output
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


    output_columns = [

        c

        for c in output_columns

        if c in selected.columns

    ]


    result =
        selected[
            output_columns
        ].copy()


    return result


# =========================================================
# SAVE
# =========================================================

def save_output(
    portfolio
):

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
        PORTFOLIO_FILE
    )

    print(
        PORTFOLIO_XLSX
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(
    portfolio,
    portfolio_amount
):

    if portfolio.empty:

        return


    total_invested =
        portfolio[
            "actual_investment"
        ].sum()


    total_expected_gain =
        portfolio[
            "expected_gain_amount"
        ].sum()


    total_maximum_loss =
        portfolio[
            "maximum_loss"
        ].sum()


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


    print("")

    print(
        "================================================="
    )

    print(
        " PORTFOLIO SUMMARY"
    )

    print(
        "================================================="
    )

    print("")

    print(
        f"Stocks selected       : {len(portfolio)}"
    )

    print(
        f"Portfolio amount      : ₹{portfolio_amount:,.2f}"
    )

    print(
        f"Actual investment     : ₹{total_invested:,.2f}"
    )

    print(
        f"Expected gain         : ₹{total_expected_gain:,.2f}"
    )

    print(
        f"Expected return       : {expected_return:.2f}%"
    )

    print(
        f"Maximum loss          : ₹{total_maximum_loss:,.2f}"
    )

    print(
        f"Maximum loss %        : {maximum_loss_percent:.2f}%"
    )

    print("")

    print(
        "================================================="
    )

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

    print(
        "Loading scanner data..."
    )


    if not INPUT_FILE.exists():

        raise FileNotFoundError(

            f"Input file not found: "
            f"{INPUT_FILE}"

        )


    df =
        pd.read_csv(
            INPUT_FILE
        )


    if df.empty:

        print(
            "Scanner file is empty."
        )

        return


    portfolio =
        build_portfolio(

            df,

            portfolio_amount=
                DEFAULT_PORTFOLIO_AMOUNT,

            stock_count=
                DEFAULT_STOCK_COUNT

        )


    if portfolio.empty:

        print(
            "Portfolio could not be generated."
        )

        return


    save_output(
        portfolio
    )


    print_summary(

        portfolio,

        DEFAULT_PORTFOLIO_AMOUNT

    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()

print("======================================================================")
print("EGX WEEKLY SMART TREND PULLBACK")
print("ATR + RSI + EMA + PARTIAL EXIT BACKTEST")
print("RESEARCH ENGINE v1.0")
print("======================================================================")

import json
import os
import math
import pandas as pd
import numpy as np


# ======================================================================
# FILES
# ======================================================================

DB_FILE = "egx_weekly_database_v1.json"

RESULT_FILE = "backtest_results.json"

TRADES_FILE = "backtest_trades.json"


# ======================================================================
# STRATEGY CONFIGURATION
# ======================================================================

# ----------------------------------------------------------------------
# RSI
# ----------------------------------------------------------------------

RSI_PERIOD = 14

# Initial pullback zone
RSI_PULLBACK_MAX = 48

# Stronger pullback
RSI_DEEP_PULLBACK = 38

# Momentum recovery confirmation
RSI_RECOVERY_LEVEL = 42

# Partial profit
RSI_PARTIAL_EXIT = 65

# Final momentum exhaustion
RSI_FINAL_EXIT = 75


# ----------------------------------------------------------------------
# EMA TREND
# ----------------------------------------------------------------------

EMA_FAST = 20

EMA_MID = 40

EMA_SLOW = 80


# Minimum EMA80 slope over 8 weeks
EMA80_MIN_SLOPE_PERCENT = 1.0


# ----------------------------------------------------------------------
# Pullback
# ----------------------------------------------------------------------

# Price must not be excessively extended above EMA20
MAX_DISTANCE_FROM_EMA20_PERCENT = 12.0

# Price should be reasonably close to EMA20/40 during entry
MAX_DISTANCE_FROM_EMA40_PERCENT = 8.0


# ----------------------------------------------------------------------
# ATR
# ----------------------------------------------------------------------

ATR_PERIOD = 14

ATR_STOP_MULTIPLIER = 2.5

ATR_TRAILING_MULTIPLIER = 3.0


# ----------------------------------------------------------------------
# Position
# ----------------------------------------------------------------------

INITIAL_ENTRY_PERCENT = 0.50

SECOND_ENTRY_PERCENT = 0.50


# ----------------------------------------------------------------------
# Profit Management
# ----------------------------------------------------------------------

PARTIAL_EXIT_PERCENT = 0.50

# First target = entry + ATR multiple
FIRST_TARGET_ATR = 2.0

# Final target = entry + ATR multiple
FINAL_TARGET_ATR = 4.0


# ----------------------------------------------------------------------
# Break Even
# ----------------------------------------------------------------------

USE_BREAK_EVEN = True

BREAK_EVEN_TRIGGER_ATR = 2.0


# ----------------------------------------------------------------------
# Trend Exit
# ----------------------------------------------------------------------

USE_TREND_EXIT = True


# ----------------------------------------------------------------------
# Risk
# ----------------------------------------------------------------------

MAX_HOLDING_WEEKS = 52


# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------

INITIAL_CAPITAL = 100000.0

START_DATE = None

END_DATE = None


# ======================================================================
# LOAD DATABASE
# ======================================================================

print("\nLoading weekly database...")

if not os.path.exists(DB_FILE):

    raise FileNotFoundError(
        f"\nDatabase file not found:\n{DB_FILE}\n\n"
        f"Make sure {DB_FILE} is uploaded to the repository."
    )


with open(
    DB_FILE,
    "r",
    encoding="utf-8"
) as f:

    raw_database = json.load(f)


print(
    f"Database loaded successfully: "
    f"{len(raw_database)} symbols"
)


# ======================================================================
# RSI
# ======================================================================

def calculate_rsi(series, period=14):

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# ======================================================================
# ATR
# ======================================================================

def calculate_atr(df, period=14):

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return atr


# ======================================================================
# DATABASE -> DATAFRAME
# ======================================================================

def database_to_dataframe(symbol_data):

    if not symbol_data:

        return None


    # ------------------------------------------------------------------
    # Standard format
    # ------------------------------------------------------------------

    if isinstance(symbol_data, dict):

        columns = symbol_data.get(
            "columns"
        )

        data = symbol_data.get(
            "data"
        )

        if (
            columns is not None
            and isinstance(data, dict)
        ):

            rows = []

            for date, values in data.items():

                if isinstance(values, dict):

                    row = values.copy()

                    row["Date"] = date

                    rows.append(row)

                else:

                    if len(values) != len(columns):

                        continue

                    row = dict(
                        zip(
                            columns,
                            values
                        )
                    )

                    row["Date"] = date

                    rows.append(row)


            if not rows:

                return None


            df = pd.DataFrame(rows)


    elif isinstance(symbol_data, list):

        df = pd.DataFrame(
            symbol_data
        )


    else:

        return None


    # ------------------------------------------------------------------
    # Normalize columns
    # ------------------------------------------------------------------

    df.columns = [
        str(c).strip().capitalize()
        for c in df.columns
    ]


    if "Date" not in df.columns:

        return None


    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]


    for col in required:

        if col not in df.columns:

            return None


    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )


    for col in required:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


    df = df.dropna(
        subset=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )


    df = df.sort_values(
        "Date"
    )


    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )


    df = df.reset_index(
        drop=True
    )


    return df


# ======================================================================
# PREPARE INDICATORS
# ======================================================================

def prepare_dataframe(df):

    df = df.copy()


    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    df["RSI14"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )


    # ------------------------------------------------------------------
    # EMAs
    # ------------------------------------------------------------------

    df["EMA20"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()


    df["EMA40"] = df["Close"].ewm(
        span=EMA_MID,
        adjust=False
    ).mean()


    df["EMA80"] = df["Close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()


    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    df["ATR14"] = calculate_atr(
        df,
        ATR_PERIOD
    )


    # ------------------------------------------------------------------
    # EMA80 slope
    # ------------------------------------------------------------------

    df["EMA80_8W_AGO"] = (
        df["EMA80"].shift(8)
    )


    df["EMA80_SLOPE_PERCENT"] = (
        (
            df["EMA80"]
            -
            df["EMA80_8W_AGO"]
        )
        /
        df["EMA80_8W_AGO"]
        *
        100
    )


    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    df["TREND_UP"] = (
        (df["EMA20"] > df["EMA40"])
        &
        (df["EMA40"] > df["EMA80"])
        &
        (
            df["EMA80_SLOPE_PERCENT"]
            >= EMA80_MIN_SLOPE_PERCENT
        )
    )


    # ------------------------------------------------------------------
    # Price distance from EMA20
    # ------------------------------------------------------------------

    df["DIST_EMA20_PERCENT"] = (
        (
            df["Close"]
            -
            df["EMA20"]
        )
        /
        df["EMA20"]
        *
        100
    )


    # ------------------------------------------------------------------
    # Price distance from EMA40
    # ------------------------------------------------------------------

    df["DIST_EMA40_PERCENT"] = (
        (
            df["Close"]
            -
            df["EMA40"]
        )
        /
        df["EMA40"]
        *
        100
    )


    # ------------------------------------------------------------------
    # RSI previous
    # ------------------------------------------------------------------

    df["RSI_PREVIOUS"] = (
        df["RSI14"].shift(1)
    )


    return df


# ======================================================================
# DATE FILTER
# ======================================================================

def apply_date_filter(df):

    if START_DATE is not None:

        df = df[
            df["Date"]
            >=
            pd.Timestamp(START_DATE)
        ]


    if END_DATE is not None:

        df = df[
            df["Date"]
            <=
            pd.Timestamp(END_DATE)
        ]


    return df.reset_index(
        drop=True
    )


# ======================================================================
# ENTRY SIGNAL
# ======================================================================

def get_entry_signal(df, i):

    row = df.iloc[i]


    if i < 1:

        return False, ""


    if pd.isna(row["RSI14"]):

        return False, ""


    if pd.isna(row["ATR14"]):

        return False, ""


    # ------------------------------------------------------------------
    # Strong uptrend
    # ------------------------------------------------------------------

    if not bool(row["TREND_UP"]):

        return False, ""


    # ------------------------------------------------------------------
    # RSI pullback
    # ------------------------------------------------------------------

    rsi = float(
        row["RSI14"]
    )

    previous_rsi = float(
        df.iloc[i - 1]["RSI14"]
    )


    # ------------------------------------------------------------------
    # We want weakness followed by recovery.
    #
    # Example:
    #
    # previous RSI = 36
    # current RSI  = 43
    #
    # This is better than buying simply because RSI is low.
    # ------------------------------------------------------------------

    rsi_recovery = (
        previous_rsi
        <=
        RSI_PULLBACK_MAX
        and
        rsi
        >
        previous_rsi
        and
        rsi
        >=
        RSI_RECOVERY_LEVEL
    )


    # ------------------------------------------------------------------
    # Alternative deep oversold recovery
    # ------------------------------------------------------------------

    deep_recovery = (
        previous_rsi
        <=
        RSI_DEEP_PULLBACK
        and
        rsi
        >
        previous_rsi
    )


    if not (
        rsi_recovery
        or
        deep_recovery
    ):

        return False, ""


    # ------------------------------------------------------------------
    # Avoid buying extremely extended price
    # ------------------------------------------------------------------

    distance_ema20 = float(
        row["DIST_EMA20_PERCENT"]
    )


    distance_ema40 = float(
        row["DIST_EMA40_PERCENT"]
    )


    if (
        distance_ema20
        >
        MAX_DISTANCE_FROM_EMA20_PERCENT
    ):

        return False, ""


    if (
        distance_ema40
        >
        MAX_DISTANCE_FROM_EMA40_PERCENT
    ):

        return False, ""


    # ------------------------------------------------------------------
    # Bullish weekly close
    # ------------------------------------------------------------------

    weekly_bullish = (
        float(row["Close"])
        >=
        float(row["Open"])
    )


    if not weekly_bullish:

        return False, ""


    # ------------------------------------------------------------------
    # Signal type
    # ------------------------------------------------------------------

    if deep_recovery:

        return True, "DEEP_RSI_RECOVERY"


    return True, "RSI_PULLBACK_RECOVERY"


# ======================================================================
# SECOND ENTRY SIGNAL
# ======================================================================

def get_second_entry_signal(df, i):

    row = df.iloc[i]


    if pd.isna(row["RSI14"]):

        return False


    if pd.isna(row["ATR14"]):

        return False


    # Must remain in major uptrend
    if not bool(row["TREND_UP"]):

        return False


    rsi = float(
        row["RSI14"]
    )


    previous_rsi = float(
        df.iloc[i - 1]["RSI14"]
    )


    # ------------------------------------------------------------------
    # Deeper pullback
    # ------------------------------------------------------------------

    deep_pullback = (
        previous_rsi <= 35
        and
        rsi > previous_rsi
    )


    # ------------------------------------------------------------------
    # Price near EMA40
    # ------------------------------------------------------------------

    near_ema40 = (
        abs(
            float(row["DIST_EMA40_PERCENT"])
        )
        <=
        5.0
    )


    return (
        deep_pullback
        and
        near_ema40
    )


# ======================================================================
# TRADE PROFIT
# ======================================================================

def calculate_trade_profit(trade):

    total = 0.0


    for sale in trade["sales"]:

        weight = float(
            sale["position_sold"]
        )

        profit = float(
            sale["profit_pct"]
        )

        total += (
            weight
            *
            profit
        )


    return round(
        total,
        2
    )


# ======================================================================
# BACKTEST ONE SYMBOL
# ======================================================================

def backtest_symbol(symbol, df):

    closed_trades = []

    open_trade = None


    for i in range(len(df)):

        row = df.iloc[i]

        date = row["Date"]

        close = float(
            row["Close"]
        )

        high = float(
            row["High"]
        )

        low = float(
            row["Low"]
        )

        atr = row["ATR14"]

        rsi = row["RSI14"]


        if pd.isna(atr):

            continue


        if pd.isna(rsi):

            continue


        # ==============================================================
        # NO POSITION
        # ==============================================================

        if open_trade is None:

            signal, signal_type = (
                get_entry_signal(
                    df,
                    i
                )
            )


            if signal:

                entry_price = close


                initial_stop = (
                    entry_price
                    -
                    float(atr)
                    *
                    ATR_STOP_MULTIPLIER
                )


                open_trade = {

                    "symbol": symbol,

                    "status": "OPEN",

                    "entry_date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "entry_price":
                        entry_price,

                    "average_entry_price":
                        entry_price,

                    "position":
                        INITIAL_ENTRY_PERCENT,

                    "original_position":
                        1.0,

                    "first_entry_percent":
                        INITIAL_ENTRY_PERCENT,

                    "second_entry_done":
                        False,

                    "sales": [],

                    "entry_rsi":
                        float(rsi),

                    "entry_signal":
                        signal_type,

                    "initial_atr":
                        float(atr),

                    "initial_stop":
                        initial_stop,

                    "stop_price":
                        initial_stop,

                    "highest_close":
                        close,

                    "weeks_held":
                        0

                }


                continue


        # ==============================================================
        # EXISTING POSITION
        # ==============================================================

        if open_trade is not None:

            open_trade["weeks_held"] += 1


            entry_price = float(
                open_trade[
                    "average_entry_price"
                ]
            )


            current_position = float(
                open_trade[
                    "position"
                ]
            )


            current_atr = float(
                atr
            )


            # ----------------------------------------------------------
            # Highest close
            # ----------------------------------------------------------

            if close > open_trade[
                "highest_close"
            ]:

                open_trade[
                    "highest_close"
                ] = close


            highest_close = float(
                open_trade[
                    "highest_close"
                ]
            )


            # ==========================================================
            # SECOND ENTRY
            # ==========================================================

            if (
                not open_trade[
                    "second_entry_done"
                ]
                and
                current_position
                < 0.999
            ):

                second_signal = (
                    get_second_entry_signal(
                        df,
                        i
                    )
                )


                if second_signal:

                    second_entry_price = close


                    # Weighted average price
                    old_weight = (
                        current_position
                    )

                    new_weight = (
                        SECOND_ENTRY_PERCENT
                    )


                    new_average = (
                        (
                            entry_price
                            *
                            old_weight
                        )
                        +
                        (
                            second_entry_price
                            *
                            new_weight
                        )
                    )
                    /
                    (
                        old_weight
                        +
                        new_weight
                    )


                    open_trade[
                        "average_entry_price"
                    ] = new_average


                    open_trade[
                        "position"
                    ] = (
                        current_position
                        +
                        SECOND_ENTRY_PERCENT
                    )


                    open_trade[
                        "second_entry_done"
                    ] = True


                    # Recalculate stop from new average
                    open_trade[
                        "stop_price"
                    ] = (
                        new_average
                        -
                        current_atr
                        *
                        ATR_STOP_MULTIPLIER
                    )


                    open_trade.setdefault(
                        "entries",
                        []
                    )


                    open_trade[
                        "entries"
                    ].append({

                        "date":
                            date.strftime(
                                "%Y-%m-%d"
                            ),

                        "price":
                            second_entry_price,

                        "position_added":
                            SECOND_ENTRY_PERCENT,

                        "reason":
                            "DEEP_PULLBACK"

                    })


                    continue


            # ==========================================================
            # PROFIT CALCULATION
            # ==========================================================

            current_profit_pct = (
                (
                    close
                    -
                    entry_price
                )
                /
                entry_price
                *
                100
            )


            # ==========================================================
            # UPDATE TRAILING STOP
            # ==========================================================

            trailing_stop = (
                highest_close
                -
                current_atr
                *
                ATR_TRAILING_MULTIPLIER
            )


            if trailing_stop > float(
                open_trade["stop_price"]
            ):

                open_trade[
                    "stop_price"
                ] = trailing_stop


            # ==========================================================
            # BREAK EVEN
            # ==========================================================

            if USE_BREAK_EVEN:

                be_trigger = (
                    entry_price
                    +
                    (
                        open_trade[
                            "initial_atr"
                        ]
                        *
                        BREAK_EVEN_TRIGGER_ATR
                    )
                )


                if (
                    highest_close
                    >=
                    be_trigger
                ):

                    if entry_price > float(
                        open_trade[
                            "stop_price"
                        ]
                    ):

                        open_trade[
                            "stop_price"
                        ] = entry_price


            # ==========================================================
            # STOP LOSS
            # ==========================================================

            stop_price = float(
                open_trade[
                    "stop_price"
                ]
            )


            if low <= stop_price:

                exit_price = close

                remaining_position = float(
                    open_trade[
                        "position"
                    ]
                )


                profit_pct = (
                    (
                        exit_price
                        -
                        entry_price
                    )
                    /
                    entry_price
                    *
                    100
                )


                open_trade[
                    "sales"
                ].append({

                    "date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "price":
                        exit_price,

                    "position_sold":
                        remaining_position,

                    "profit_pct":
                        profit_pct,

                    "reason":
                        "ATR_STOP"

                })


                open_trade[
                    "position"
                ] = 0.0


                open_trade[
                    "status"
                ] = "CLOSED"


                open_trade[
                    "exit_date"
                ] = date.strftime(
                    "%Y-%m-%d"
                )


                open_trade[
                    "exit_price"
                ] = exit_price


                open_trade[
                    "profit_pct"
                ] = calculate_trade_profit(
                    open_trade
                )


                open_trade[
                    "exit_reason"
                ] = "ATR_STOP"


                closed_trades.append(
                    open_trade
                )


                open_trade = None

                continue


            # ==========================================================
            # FIRST PARTIAL EXIT
            # ==========================================================

            target_1 = (
                entry_price
                +
                (
                    open_trade[
                        "initial_atr"
                    ]
                    *
                    FIRST_TARGET_ATR
                )
            )


            if (
                current_position > 0.50
                and
                high >= target_1
                and
                rsi >= RSI_PARTIAL_EXIT
            ):

                sell_position = (
                    current_position
                    *
                    PARTIAL_EXIT_PERCENT
                )


                sell_price = close


                profit_pct = (
                    (
                        sell_price
                        -
                        entry_price
                    )
                    /
                    entry_price
                    *
                    100
                )


                open_trade[
                    "sales"
                ].append({

                    "date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "price":
                        sell_price,

                    "position_sold":
                        sell_position,

                    "profit_pct":
                        profit_pct,

                    "reason":
                        "PARTIAL_PROFIT"

                })


                open_trade[
                    "position"
                ] -= sell_position


                open_trade[
                    "first_partial_done"
                ] = True


                if USE_BREAK_EVEN:

                    open_trade[
                        "break_even_active"
                    ] = True


                continue


            # ==========================================================
            # FINAL TARGET
            # ==========================================================

            target_2 = (
                entry_price
                +
                (
                    open_trade[
                        "initial_atr"
                    ]
                    *
                    FINAL_TARGET_ATR
                )
            )


            if (
                open_trade[
                    "first_partial_done"
                ]
                if "first_partial_done"
                in open_trade
                else False
            ):

                if (
                    high >= target_2
                    and
                    rsi >= RSI_FINAL_EXIT
                ):

                    exit_price = close

                    remaining_position = float(
                        open_trade[
                            "position"
                        ]
                    )


                    profit_pct = (
                        (
                            exit_price
                            -
                            entry_price
                        )
                        /
                        entry_price
                        *
                        100
                    )


                    open_trade[
                        "sales"
                    ].append({

                        "date":
                            date.strftime(
                                "%Y-%m-%d"
                            ),

                        "price":
                            exit_price,

                        "position_sold":
                            remaining_position,

                        "profit_pct":
                            profit_pct,

                        "reason":
                            "FINAL_TARGET"

                    })


                    open_trade[
                        "position"
                    ] = 0.0


                    open_trade[
                        "status"
                    ] = "CLOSED"


                    open_trade[
                        "exit_date"
                    ] = date.strftime(
                        "%Y-%m-%d"
                    )


                    open_trade[
                        "exit_price"
                    ] = exit_price


                    open_trade[
                        "profit_pct"
                    ] = calculate_trade_profit(
                        open_trade
                    )


                    open_trade[
                        "exit_reason"
                    ] = "FINAL_TARGET"


                    closed_trades.append(
                        open_trade
                    )


                    open_trade = None

                    continue


            # ==========================================================
            # RSI FINAL EXIT
            # ==========================================================

            if (
                rsi >= RSI_FINAL_EXIT
                and
                current_profit_pct > 5.0
            ):

                exit_price = close

                remaining_position = float(
                    open_trade[
                        "position"
                    ]
                )


                profit_pct = (
                    (
                        exit_price
                        -
                        entry_price
                    )
                    /
                    entry_price
                    *
                    100
                )


                open_trade[
                    "sales"
                ].append({

                    "date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "price":
                        exit_price,

                    "position_sold":
                        remaining_position,

                    "profit_pct":
                        profit_pct,

                    "reason":
                        "RSI_FINAL"

                })


                open_trade[
                    "position"
                ] = 0.0


                open_trade[
                    "status"
                ] = "CLOSED"


                open_trade[
                    "exit_date"
                ] = date.strftime(
                    "%Y-%m-%d"
                )


                open_trade[
                    "exit_price"
                ] = exit_price


                open_trade[
                    "profit_pct"
                ] = calculate_trade_profit(
                    open_trade
                )


                open_trade[
                    "exit_reason"
                ] = "RSI_FINAL"


                closed_trades.append(
                    open_trade
                )


                open_trade = None

                continue


            # ==========================================================
            # TREND FAILURE
            # ==========================================================

            if USE_TREND_EXIT:

                ema20 = float(
                    row["EMA20"]
                )

                ema40 = float(
                    row["EMA40"]
                )


                # Major trend deterioration
                trend_failure = (
                    ema20
                    <
                    ema40
                    and
                    close
                    <
                    ema40
                )


                if (
                    trend_failure
                    and
                    current_profit_pct > 0
                ):

                    exit_price = close

                    remaining_position = float(
                        open_trade[
                            "position"
                        ]
                    )


                    profit_pct = (
                        (
                            exit_price
                            -
                            entry_price
                        )
                        /
                        entry_price
                        *
                        100
                    )


                    open_trade[
                        "sales"
                    ].append({

                        "date":
                            date.strftime(
                                "%Y-%m-%d"
                            ),

                        "price":
                            exit_price,

                        "position_sold":
                            remaining_position,

                        "profit_pct":
                            profit_pct,

                        "reason":
                            "TREND_FAILURE"

                    })


                    open_trade[
                        "position"
                    ] = 0.0


                    open_trade[
                        "status"
                    ] = "CLOSED"


                    open_trade[
                        "exit_date"
                    ] = date.strftime(
                        "%Y-%m-%d"
                    )


                    open_trade[
                        "exit_price"
                    ] = exit_price


                    open_trade[
                        "profit_pct"
                    ] = calculate_trade_profit(
                        open_trade
                    )


                    open_trade[
                        "exit_reason"
                    ] = "TREND_FAILURE"


                    closed_trades.append(
                        open_trade
                    )


                    open_trade = None

                    continue


            # ==========================================================
            # MAX HOLDING PERIOD
            # ==========================================================

            if (
                open_trade is not None
                and
                open_trade[
                    "weeks_held"
                ]
                >=
                MAX_HOLDING_WEEKS
            ):

                exit_price = close

                remaining_position = float(
                    open_trade[
                        "position"
                    ]
                )


                profit_pct = (
                    (
                        exit_price
                        -
                        entry_price
                    )
                    /
                    entry_price
                    *
                    100
                )


                open_trade[
                    "sales"
                ].append({

                    "date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "price":
                        exit_price,

                    "position_sold":
                        remaining_position,

                    "profit_pct":
                        profit_pct,

                    "reason":
                        "TIME_EXIT"

                })


                open_trade[
                    "position"
                ] = 0.0


                open_trade[
                    "status"
                ] = "CLOSED"


                open_trade[
                    "exit_date"
                ] = date.strftime(
                    "%Y-%m-%d"
                )


                open_trade[
                    "exit_price"
                ] = exit_price


                open_trade[
                    "profit_pct"
                ] = calculate_trade_profit(
                    open_trade
                )


                open_trade[
                    "exit_reason"
                ] = "TIME_EXIT"


                closed_trades.append(
                    open_trade
                )


                open_trade = None

                continue


    # ==================================================================
    # OPEN POSITION
    # ==================================================================

    if open_trade is not None:

        open_trade[
            "status"
        ] = "OPEN"


        open_trade[
            "exit_date"
        ] = None


        open_trade[
            "exit_price"
        ] = None


        open_trade[
            "profit_pct"
        ] = None


        closed_trades.append(
            open_trade
        )


    return closed_trades


# ======================================================================
# RUN BACKTEST
# ======================================================================

print("\nStarting weekly backtest...\n")


all_trades = []


symbol_statistics = {}


for symbol, symbol_data in raw_database.items():

    # ------------------------------------------------------------------
    # Ignore indexes
    # ------------------------------------------------------------------

    if symbol.upper() in [
        "EGX30",
        "EGX70",
        "EGX100"
    ]:

        continue


    df = database_to_dataframe(
        symbol_data
    )


    if df is None:

        print(
            f"⚠️ {symbol}: invalid data"
        )

        continue


    if len(df) < 100:

        print(
            f"⚠️ {symbol}: insufficient weekly data"
        )

        continue


    df = prepare_dataframe(
        df
    )


    df = apply_date_filter(
        df
    )


    if df.empty:

        continue


    trades = backtest_symbol(
        symbol,
        df
    )


    all_trades.extend(
        trades
    )


    closed_for_symbol = [
        t for t in trades
        if t["status"] == "CLOSED"
    ]


    symbol_profit = sum(
        t["profit_pct"]
        for t in closed_for_symbol
    )


    wins = [
        t for t in closed_for_symbol
        if t["profit_pct"] > 0
    ]


    symbol_statistics[
        symbol
    ] = {

        "trades":
            len(closed_for_symbol),

        "winning":
            len(wins),

        "win_rate":
            (
                len(wins)
                /
                len(closed_for_symbol)
                *
                100
            )
            if closed_for_symbol
            else 0,

        "profit":
            symbol_profit

    }


    print(
        f"✅ {symbol}: "
        f"{len(closed_for_symbol)} trades"
    )


# ======================================================================
# SORT
# ======================================================================

all_trades.sort(
    key=lambda x:
    x["entry_date"]
)


closed_trades = [
    t for t in all_trades
    if t["status"] == "CLOSED"
]


open_positions = [
    t for t in all_trades
    if t["status"] == "OPEN"
]


# ======================================================================
# BASIC STATISTICS
# ======================================================================

total_trades = len(
    closed_trades
)


winning_trades = [
    t for t in closed_trades
    if t["profit_pct"] > 0
]


losing_trades = [
    t for t in closed_trades
    if t["profit_pct"] <= 0
]


winning_count = len(
    winning_trades
)


losing_count = len(
    losing_trades
)


win_rate = (
    winning_count
    /
    total_trades
    *
    100
    if total_trades
    else 0
)


# ======================================================================
# PROFIT STATISTICS
# ======================================================================

average_win = (
    np.mean(
        [
            t["profit_pct"]
            for t in winning_trades
        ]
    )
    if winning_trades
    else 0
)


average_loss = (
    np.mean(
        [
            t["profit_pct"]
            for t in losing_trades
        ]
    )
    if losing_trades
    else 0
)


sum_trade_profit = sum(
    t["profit_pct"]
    for t in closed_trades
)


# ======================================================================
# PROFIT FACTOR
# ======================================================================

gross_profit = sum(
    t["profit_pct"]
    for t in winning_trades
)


gross_loss = abs(
    sum(
        t["profit_pct"]
        for t in losing_trades
    )
)


profit_factor = (
    gross_profit
    /
    gross_loss
    if gross_loss > 0
    else 0
)


# ======================================================================
# EXPECTANCY
# ======================================================================

expectancy = (
    (
        win_rate / 100
        *
        average_win
    )
    +
    (
        (1 - win_rate / 100)
        *
        average_loss
    )
)


# ======================================================================
# COMPOUND PORTFOLIO
# ======================================================================

portfolio_value = (
    INITIAL_CAPITAL
)


equity_curve = []


for trade in closed_trades:

    trade_return = (
        trade["profit_pct"]
        /
        100
    )


    portfolio_value *= (
        1
        +
        trade_return
    )


    equity_curve.append({

        "date":
            trade["exit_date"],

        "equity":
            portfolio_value

    })


compound_return = (
    (
        portfolio_value
        /
        INITIAL_CAPITAL
    )
    -
    1
) * 100


# ======================================================================
# MAX DRAWDOWN
# ======================================================================

peak = INITIAL_CAPITAL

max_drawdown = 0.0


for point in equity_curve:

    equity = point[
        "equity"
    ]


    if equity > peak:

        peak = equity


    drawdown = (
        (
            peak
            -
            equity
        )
        /
        peak
        *
        100
    )


    if drawdown > max_drawdown:

        max_drawdown = drawdown


# ======================================================================
# EXIT ANALYSIS
# ======================================================================

exit_reasons = {}


for trade in closed_trades:

    reason = trade.get(
        "exit_reason",
        "UNKNOWN"
    )


    exit_reasons[
        reason
    ] = (
        exit_reasons.get(
            reason,
            0
        )
        +
        1
    )


# ======================================================================
# ENTRY SIGNAL ANALYSIS
# ======================================================================

entry_signal_stats = {}


for trade in closed_trades:

    signal = trade.get(
        "entry_signal",
        "UNKNOWN"
    )


    if signal not in entry_signal_stats:

        entry_signal_stats[
            signal
        ] = {

            "trades": 0,

            "wins": 0,

            "losses": 0,

            "profit": 0.0

        }


    entry_signal_stats[
        signal
    ]["trades"] += 1


    entry_signal_stats[
        signal
    ]["profit"] += (
        trade["profit_pct"]
    )


    if trade["profit_pct"] > 0:

        entry_signal_stats[
            signal
        ]["wins"] += 1

    else:

        entry_signal_stats[
            signal
        ]["losses"] += 1


# ======================================================================
# BEST / WORST
# ======================================================================

best_trade = None

worst_trade = None


if closed_trades:

    best_trade = max(
        closed_trades,
        key=lambda x:
        x["profit_pct"]
    )


    worst_trade = min(
        closed_trades,
        key=lambda x:
        x["profit_pct"]
    )


# ======================================================================
# TOP STOCKS
# ======================================================================

sorted_symbols = sorted(
    symbol_statistics.items(),
    key=lambda x:
    x[1]["profit"],
    reverse=True
)


top_10 = sorted_symbols[
    :10
]


worst_10 = sorted(
    symbol_statistics.items(),
    key=lambda x:
    x[1]["profit"]
)[:10]


# ======================================================================
# RESULT OBJECT
# ======================================================================

result = {

    "strategy":
        "EGX Weekly Smart Trend Pullback v1.0",

    "description":
        (
            "Weekly trend-following strategy using "
            "EMA20/40/80 trend structure, RSI pullback "
            "recovery, ATR risk management, partial profit "
            "taking, break-even and trailing protection."
        ),

    "database":
        DB_FILE,

    "parameters": {

        "rsi_period":
            RSI_PERIOD,

        "rsi_pullback_max":
            RSI_PULLBACK_MAX,

        "rsi_deep_pullback":
            RSI_DEEP_PULLBACK,

        "rsi_recovery_level":
            RSI_RECOVERY_LEVEL,

        "rsi_partial_exit":
            RSI_PARTIAL_EXIT,

        "rsi_final_exit":
            RSI_FINAL_EXIT,

        "ema_fast":
            EMA_FAST,

        "ema_mid":
            EMA_MID,

        "ema_slow":
            EMA_SLOW,

        "ema80_min_slope_percent":
            EMA80_MIN_SLOPE_PERCENT,

        "max_distance_ema20_percent":
            MAX_DISTANCE_FROM_EMA20_PERCENT,

        "max_distance_ema40_percent":
            MAX_DISTANCE_FROM_EMA40_PERCENT,

        "atr_period":
            ATR_PERIOD,

        "atr_stop_multiplier":
            ATR_STOP_MULTIPLIER,

        "atr_trailing_multiplier":
            ATR_TRAILING_MULTIPLIER,

        "initial_entry_percent":
            INITIAL_ENTRY_PERCENT,

        "second_entry_percent":
            SECOND_ENTRY_PERCENT,

        "partial_exit_percent":
            PARTIAL_EXIT_PERCENT,

        "first_target_atr":
            FIRST_TARGET_ATR,

        "final_target_atr":
            FINAL_TARGET_ATR,

        "break_even":
            USE_BREAK_EVEN,

        "max_holding_weeks":
            MAX_HOLDING_WEEKS

    },

    "period": {

        "start":
            (
                min(
                    [
                        t["entry_date"]
                        for t in all_trades
                    ]
                )
                if all_trades
                else None
            ),

        "end":
            (
                max(
                    [
                        t["exit_date"]
                        for t in closed_trades
                        if t["exit_date"]
                    ]
                )
                if closed_trades
                else None
            )

    },

    "statistics": {

        "symbols_tested":
            len(symbol_statistics),

        "total_trades":
            total_trades,

        "winning_trades":
            winning_count,

        "losing_trades":
            losing_count,

        "win_rate_percent":
            round(
                win_rate,
                2
            ),

        "sum_trade_profit_percent":
            round(
                sum_trade_profit,
                2
            ),

        "compound_portfolio_return_percent":
            round(
                compound_return,
                2
            ),

        "average_winning_trade_percent":
            round(
                average_win,
                2
            ),

        "average_losing_trade_percent":
            round(
                average_loss,
                2
            ),

        "profit_factor":
            round(
                profit_factor,
                2
            ),

        "expectancy_percent":
            round(
                expectancy,
                2
            ),

        "maximum_drawdown_percent":
            round(
                max_drawdown,
                2
            )

    },

    "exit_analysis":
        exit_reasons,

    "entry_signal_analysis":
        entry_signal_stats,

    "best_trade":
        (
            {
                "symbol":
                    best_trade["symbol"],

                "entry_date":
                    best_trade["entry_date"],

                "exit_date":
                    best_trade["exit_date"],

                "profit_pct":
                    best_trade["profit_pct"]

            }
            if best_trade
            else None
        ),

    "worst_trade":
        (
            {
                "symbol":
                    worst_trade["symbol"],

                "entry_date":
                    worst_trade["entry_date"],

                "exit_date":
                    worst_trade["exit_date"],

                "profit_pct":
                    worst_trade["profit_pct"]

            }
            if worst_trade
            else None
        ),

    "top_10_stocks":
        [
            {
                "symbol":
                    symbol,

                **stats

            }
            for symbol, stats
            in top_10
        ],

    "worst_10_stocks":
        [
            {
                "symbol":
                    symbol,

                **stats

            }
            for symbol, stats
            in worst_10
        ],

    "open_positions":
        open_positions

}


# ======================================================================
# SAVE RESULTS
# ======================================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )


# ======================================================================
# SAVE TRADES
# ======================================================================

with open(
    TRADES_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_trades,
        f,
        ensure_ascii=False,
        indent=2
    )


# ======================================================================
# PRINT FINAL RESULTS
# ======================================================================

print("\n")

print(
    "======================================================================"
)

print(
    "FINAL WEEKLY BACKTEST RESULTS"
)

print(
    "======================================================================"
)


print(
    f"Symbols Tested           : "
    f"{len(symbol_statistics)}"
)


print(
    f"Total Trades             : "
    f"{total_trades}"
)


print(
    f"Winning Trades           : "
    f"{winning_count}"
)


print(
    f"Losing Trades            : "
    f"{losing_count}"
)


print(
    f"Win Rate                 : "
    f"{win_rate:.2f}%"
)


print(
    f"Sum Trade Profit         : "
    f"{sum_trade_profit:.2f}%"
)


print(
    f"Compound Return          : "
    f"{compound_return:.2f}%"
)


print(
    f"Average Win              : "
    f"{average_win:.2f}%"
)


print(
    f"Average Loss             : "
    f"{average_loss:.2f}%"
)


print(
    f"Profit Factor            : "
    f"{profit_factor:.2f}"
)


print(
    f"Expectancy               : "
    f"{expectancy:.2f}%"
)


print(
    f"Maximum Drawdown         : "
    f"{max_drawdown:.2f}%"
)


print(
    "\n"
    "----------------------------------------------------------------------"
)

print(
    "EXIT ANALYSIS"
)

print(
    "----------------------------------------------------------------------"
)


for reason, count in sorted(
    exit_reasons.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{reason:25} : {count}"
    )


print(
    "\n"
    "----------------------------------------------------------------------"
)

print(
    "TOP 10 STOCKS"
)

print(
    "----------------------------------------------------------------------"
)


for symbol, stats in top_10:

    print(
        f"{symbol:8} | "
        f"Trades: {stats['trades']:3} | "
        f"Win: {stats['win_rate']:6.2f}% | "
        f"Profit: {stats['profit']:8.2f}%"
    )


print(
    "\n"
    "----------------------------------------------------------------------"
)

print(
    "WORST 10 STOCKS"
)

print(
    "----------------------------------------------------------------------"
)


for symbol, stats in worst_10:

    print(
        f"{symbol:8} | "
        f"Trades: {stats['trades']:3} | "
        f"Win: {stats['win_rate']:6.2f}% | "
        f"Profit: {stats['profit']:8.2f}%"
    )


print(
    "\n"
    "----------------------------------------------------------------------"
)

print(
    "OPEN POSITIONS"
)

print(
    "----------------------------------------------------------------------"
)


for trade in open_positions:

    print(
        f"{trade['symbol']:8} | "
        f"Entry: {trade['entry_date']} | "
        f"Avg: {trade['average_entry_price']:.2f} | "
        f"Position: {trade['position']:.2f}"
    )


print(
    "\n"
    "======================================================================"
)

print(
    "BACKTEST COMPLETE"
)

print(
    "======================================================================"
)


print(
    f"\nResults saved to: {RESULT_FILE}"
)


print(
    f"Trades saved to : {TRADES_FILE}"
)

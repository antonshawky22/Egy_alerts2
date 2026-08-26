print("==============================================================")
print("EGX WEEKLY RSI 33/60/70 + EMA TREND + PULLBACK + BREAK-EVEN")
print("BACKTEST ENGINE v3.0")
print("==============================================================")

import json
import os
import pandas as pd
import numpy as np


# ==============================================================
# CONFIGURATION
# ==============================================================

DB_FILE = "egx_weekly_database_v1.json"

RESULT_FILE = "weekly_rsi_backtest_results.json"


# ==============================================================
# RSI
# ==============================================================

RSI_PERIOD = 14

RSI_BUY = 33

RSI_SELL_1 = 60

RSI_SELL_2 = 70


# ==============================================================
# POSITION MANAGEMENT
# ==============================================================

FIRST_SELL_PERCENT = 0.50

SECOND_SELL_PERCENT = 0.50

HARD_STOP_PERCENT = 5.0

BREAK_EVEN_AFTER_FIRST_SELL = True


# ==============================================================
# TREND FILTER
# ==============================================================

USE_TREND_FILTER = True

EMA_FAST = 20

EMA_SLOW = 40


# --------------------------------------------------------------
# NEW:
# EMA40 must be rising
# --------------------------------------------------------------

USE_EMA_SLOPE_FILTER = True

EMA_SLOPE_LOOKBACK = 4


# --------------------------------------------------------------
# NEW:
# Avoid buying stocks too far above EMA40.
#
# This is a pullback strategy, so we want the entry
# relatively close to the long trend line.
# --------------------------------------------------------------

USE_PULLBACK_FILTER = True

MAX_DISTANCE_FROM_EMA40 = 8.0


# ==============================================================
# COOLDOWN
# ==============================================================

USE_COOLDOWN = True

COOLDOWN_WEEKS = 4


# ==============================================================
# BACKTEST
# ==============================================================

INITIAL_CAPITAL = 100000.0

START_DATE = None

END_DATE = None


# ==============================================================
# LOAD DATABASE
# ==============================================================

print("\nLoading weekly database...")

if not os.path.exists(DB_FILE):

    raise FileNotFoundError(
        f"Database file not found: {DB_FILE}"
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


# ==============================================================
# RSI
# ==============================================================

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

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# ==============================================================
# DATABASE -> DATAFRAME
# ==============================================================

def database_to_dataframe(symbol_data):

    if not symbol_data:

        return None


    # ----------------------------------------------------------
    # FORMAT 1
    # ----------------------------------------------------------

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
                        zip(columns, values)
                    )

                    row["Date"] = date

                    rows.append(row)


            if not rows:

                return None


            df = pd.DataFrame(rows)


    # ----------------------------------------------------------
    # FORMAT 2
    # ----------------------------------------------------------

    elif isinstance(symbol_data, list):

        df = pd.DataFrame(
            symbol_data
        )


    else:

        return None


    # ----------------------------------------------------------
    # NORMALIZE COLUMNS
    # ----------------------------------------------------------

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


# ==============================================================
# PREPARE INDICATORS
# ==============================================================

def prepare_dataframe(df):

    df = df.copy()


    # ----------------------------------------------------------
    # RSI
    # ----------------------------------------------------------

    df["RSI14"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )


    # ----------------------------------------------------------
    # EMA20
    # ----------------------------------------------------------

    df["EMA20"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()


    # ----------------------------------------------------------
    # EMA40
    # ----------------------------------------------------------

    df["EMA40"] = df["Close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()


    # ----------------------------------------------------------
    # TREND
    # ----------------------------------------------------------

    df["TREND_UP"] = (
        df["EMA20"] >
        df["EMA40"]
    )


    # ----------------------------------------------------------
    # EMA40 SLOPE
    # ----------------------------------------------------------

    df["EMA40_SLOPE_UP"] = (
        df["EMA40"] >
        df["EMA40"].shift(
            EMA_SLOPE_LOOKBACK
        )
    )


    # ----------------------------------------------------------
    # DISTANCE FROM EMA40
    # ----------------------------------------------------------

    df["DISTANCE_FROM_EMA40"] = (
        (
            df["Close"] -
            df["EMA40"]
        )
        /
        df["EMA40"]
    ) * 100


    return df


# ==============================================================
# DATE FILTER
# ==============================================================

def apply_date_filter(df):

    if START_DATE is not None:

        df = df[
            df["Date"] >=
            pd.Timestamp(START_DATE)
        ]


    if END_DATE is not None:

        df = df[
            df["Date"] <=
            pd.Timestamp(END_DATE)
        ]


    return df.reset_index(
        drop=True
    )


# ==============================================================
# CALCULATE TRADE PROFIT
# ==============================================================

def calculate_trade_profit(trade):

    total_profit = 0.0


    for sale in trade["sales"]:

        weight = sale[
            "position_sold"
        ]

        profit = sale[
            "profit_pct"
        ]

        total_profit += (
            weight *
            profit
        )


    return round(
        total_profit,
        2
    )


# ==============================================================
# BACKTEST ONE SYMBOL
# ==============================================================

def backtest_symbol(symbol, df):

    trades = []

    position = None

    cooldown_until = None


    # ----------------------------------------------------------
    # Iterate weekly candles
    # ----------------------------------------------------------

    for i in range(len(df)):

        row = df.iloc[i]


        date = row["Date"]

        close = float(
            row["Close"]
        )

        open_price = float(
            row["Open"]
        )

        low = float(
            row["Low"]
        )

        rsi = row["RSI14"]

        trend_up = bool(
            row["TREND_UP"]
        )

        ema40_slope_up = bool(
            row["EMA40_SLOPE_UP"]
        )

        distance_from_ema40 = float(
            row["DISTANCE_FROM_EMA40"]
        )


        if pd.isna(rsi):

            continue


        # ======================================================
        # NO POSITION
        # ======================================================

        if position is None:


            # --------------------------------------------------
            # COOLDOWN
            # --------------------------------------------------

            if (
                USE_COOLDOWN
                and cooldown_until is not None
                and date <= cooldown_until
            ):

                continue


            # --------------------------------------------------
            # RSI ENTRY
            # --------------------------------------------------

            buy_signal = (
                rsi <= RSI_BUY
            )


            # --------------------------------------------------
            # TREND FILTER
            # --------------------------------------------------

            if USE_TREND_FILTER:

                buy_signal = (
                    buy_signal
                    and trend_up
                )


            # --------------------------------------------------
            # EMA40 SLOPE FILTER
            # --------------------------------------------------

            if USE_EMA_SLOPE_FILTER:

                buy_signal = (
                    buy_signal
                    and ema40_slope_up
                )


            # --------------------------------------------------
            # PULLBACK FILTER
            # --------------------------------------------------

            if USE_PULLBACK_FILTER:

                buy_signal = (
                    buy_signal
                    and
                    distance_from_ema40
                    <=
                    MAX_DISTANCE_FROM_EMA40
                )


            # --------------------------------------------------
            # ENTRY
            # --------------------------------------------------

            if buy_signal:

                position = {

                    "symbol":
                        symbol,

                    "status":
                        "OPEN",

                    "entry_date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "entry_price":
                        close,

                    "position":
                        1.0,

                    "original_position":
                        1.0,

                    "sales":
                        [],

                    "first_sell_done":
                        False,

                    "break_even_active":
                        False,

                    "entry_rsi":
                        float(rsi),

                    "entry_ema20":
                        float(
                            row["EMA20"]
                        ),

                    "entry_ema40":
                        float(
                            row["EMA40"]
                        ),

                    "entry_distance_ema40":
                        distance_from_ema40

                }

                continue


        # ======================================================
        # EXISTING POSITION
        # ======================================================

        if position is not None:

            entry_price = (
                position[
                    "entry_price"
                ]
            )


            # ==================================================
            # HARD STOP
            # ==================================================

            hard_stop_price = (
                entry_price
                *
                (
                    1 -
                    HARD_STOP_PERCENT /
                    100
                )
            )


            # --------------------------------------------------
            # STOP EXECUTION
            #
            # If weekly LOW touches the stop:
            #
            # 1) If OPEN is already below stop -> gap down
            #    and exit at OPEN.
            #
            # 2) Otherwise stop was reached during the week
            #    and exit at STOP price.
            #
            # This is more realistic than using weekly CLOSE.
            # --------------------------------------------------

            if low <= hard_stop_price:

                if open_price <= hard_stop_price:

                    exit_price = open_price

                else:

                    exit_price = (
                        hard_stop_price
                    )


                remaining_position = (
                    position["position"]
                )


                profit_pct = (
                    (
                        exit_price -
                        entry_price
                    )
                    /
                    entry_price
                ) * 100


                position["sales"].append({

                    "date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "price":
                        round(
                            exit_price,
                            4
                        ),

                    "position_sold":
                        remaining_position,

                    "profit_pct":
                        round(
                            profit_pct,
                            2
                        ),

                    "reason":
                        "HARD_STOP"

                })


                position["position"] = 0.0

                position["status"] = (
                    "CLOSED"
                )

                position["exit_date"] = (
                    date.strftime(
                        "%Y-%m-%d"
                    )
                )

                position["exit_price"] = (
                    exit_price
                )

                position["profit_pct"] = (
                    calculate_trade_profit(
                        position
                    )
                )

                position["exit_reason"] = (
                    "HARD_STOP"
                )


                trades.append(
                    position
                )


                # ------------------------------------------------
                # COOLDOWN
                # ------------------------------------------------

                if USE_COOLDOWN:

                    cooldown_until = (
                        date +
                        pd.Timedelta(
                            weeks=
                            COOLDOWN_WEEKS
                        )
                    )


                position = None

                continue


            # ==================================================
            # FIRST SELL - RSI 60
            # ==================================================

            if (
                not
                position[
                    "first_sell_done"
                ]
                and
                rsi >= RSI_SELL_1
            ):

                sell_position = (
                    position["position"]
                    *
                    FIRST_SELL_PERCENT
                )


                if sell_position > 0:

                    sell_price = close


                    profit_pct = (
                        (
                            sell_price -
                            entry_price
                        )
                        /
                        entry_price
                    ) * 100


                    position["sales"].append({

                        "date":
                            date.strftime(
                                "%Y-%m-%d"
                            ),

                        "price":
                            sell_price,

                        "position_sold":
                            sell_position,

                        "profit_pct":
                            round(
                                profit_pct,
                                2
                            ),

                        "reason":
                            "RSI_60"

                    })


                    position["position"] -= (
                        sell_position
                    )


                    position[
                        "first_sell_done"
                    ] = True


                    # ------------------------------------------
                    # BREAK EVEN
                    # ------------------------------------------

                    if (
                        BREAK_EVEN_AFTER_FIRST_SELL
                    ):

                        position[
                            "break_even_active"
                        ] = True


                continue


            # ==================================================
            # BREAK EVEN STOP
            # ==================================================

            if (
                position[
                    "first_sell_done"
                ]
                and
                position[
                    "break_even_active"
                ]
                and
                position["position"] > 0
            ):

                # ------------------------------------------------
                # Break-even is checked using LOW.
                #
                # If the weekly low reaches entry price,
                # remaining position exits at entry price,
                # unless the week opened below entry.
                # ------------------------------------------------

                if low <= entry_price:

                    if open_price <= entry_price:

                        sell_price = (
                            open_price
                        )

                    else:

                        sell_price = (
                            entry_price
                        )


                    remaining_position = (
                        position["position"]
                    )


                    profit_pct = (
                        (
                            sell_price -
                            entry_price
                        )
                        /
                        entry_price
                    ) * 100


                    position["sales"].append({

                        "date":
                            date.strftime(
                                "%Y-%m-%d"
                            ),

                        "price":
                            sell_price,

                        "position_sold":
                            remaining_position,

                        "profit_pct":
                            round(
                                profit_pct,
                                2
                            ),

                        "reason":
                            "BREAK_EVEN_STOP"

                    })


                    position["position"] = 0.0

                    position["status"] = (
                        "CLOSED"
                    )

                    position["exit_date"] = (
                        date.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    position["exit_price"] = (
                        sell_price
                    )

                    position["profit_pct"] = (
                        calculate_trade_profit(
                            position
                        )
                    )

                    position["exit_reason"] = (
                        "BREAK_EVEN_STOP"
                    )


                    trades.append(
                        position
                    )


                    if USE_COOLDOWN:

                        cooldown_until = (
                            date +
                            pd.Timedelta(
                                weeks=
                                COOLDOWN_WEEKS
                            )
                        )


                    position = None

                    continue


            # ==================================================
            # SECOND SELL - RSI 70
            # ==================================================

            if (
                position is not None
                and
                position[
                    "first_sell_done"
                ]
                and
                position["position"] > 0
                and
                rsi >= RSI_SELL_2
            ):

                sell_price = close

                remaining_position = (
                    position["position"]
                )


                profit_pct = (
                    (
                        sell_price -
                        entry_price
                    )
                    /
                    entry_price
                ) * 100


                position["sales"].append({

                    "date":
                        date.strftime(
                            "%Y-%m-%d"
                        ),

                    "price":
                        sell_price,

                    "position_sold":
                        remaining_position,

                    "profit_pct":
                        round(
                            profit_pct,
                            2
                        ),

                    "reason":
                        "RSI_70"

                })


                position["position"] = 0.0

                position["status"] = (
                    "CLOSED"
                )

                position["exit_date"] = (
                    date.strftime(
                        "%Y-%m-%d"
                    )
                )

                position["exit_price"] = (
                    sell_price
                )

                position["profit_pct"] = (
                    calculate_trade_profit(
                        position
                    )
                )

                position["exit_reason"] = (
                    "RSI_70"
                )


                trades.append(
                    position
                )


                position = None

                continue


    # ==========================================================
    # OPEN POSITION
    # ==========================================================

    if position is not None:

        position["status"] = (
            "OPEN"
        )

        position["exit_date"] = None

        position["exit_price"] = None

        position["profit_pct"] = None


        trades.append(
            position
        )


    return trades


# ==============================================================
# RUN ALL SYMBOLS
# ==============================================================

all_trades = []


print(
    "\nStarting backtest...\n"
)


for symbol, symbol_data in raw_database.items():

    # ----------------------------------------------------------
    # Ignore market indexes
    # ----------------------------------------------------------

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


    if len(df) < 60:

        print(
            f"⚠️ {symbol}: insufficient data"
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


    symbol_trades = backtest_symbol(
        symbol,
        df
    )


    all_trades.extend(
        symbol_trades
    )


    print(
        f"✅ {symbol}: "
        f"{len(symbol_trades)} trades"
    )


# ==============================================================
# SORT
# ==============================================================

all_trades.sort(
    key=lambda x:
    x["entry_date"]
)


# ==============================================================
# CLOSED / OPEN
# ==============================================================

closed_trades = [
    t
    for t in all_trades
    if t["status"] == "CLOSED"
]


open_positions = [
    t
    for t in all_trades
    if t["status"] == "OPEN"
]


# ==============================================================
# BASIC STATISTICS
# ==============================================================

total_trades = len(
    closed_trades
)


winning_trades = [
    t
    for t in closed_trades
    if t["profit_pct"] > 0
]


losing_trades = [
    t
    for t in closed_trades
    if t["profit_pct"] <= 0
]


winning_count = len(
    winning_trades
)


losing_count = len(
    losing_trades
)


win_rate = (
    winning_count /
    total_trades *
    100
    if total_trades > 0
    else 0
)


# ==============================================================
# AVERAGES
# ==============================================================

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


# ==============================================================
# SUM TRADE PROFIT
# ==============================================================

total_profit = sum(
    t["profit_pct"]
    for t in closed_trades
)


# ==============================================================
# COMPOUND RETURN
# ==============================================================

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
        1 +
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
        portfolio_value /
        INITIAL_CAPITAL
    )
    - 1

) * 100


# ==============================================================
# MAXIMUM DRAWDOWN
# ==============================================================

peak = (
    INITIAL_CAPITAL
)


max_drawdown = 0.0


for point in equity_curve:

    equity = point[
        "equity"
    ]


    if equity > peak:

        peak = equity


    drawdown = (
        (
            peak -
            equity
        )
        /
        peak
    ) * 100


    if drawdown > max_drawdown:

        max_drawdown = (
            drawdown
        )


# ==============================================================
# EXIT ANALYSIS
# ==============================================================

hard_stop_losses = 0

rsi_60_sales = 0

rsi_70_final_exits = 0

break_even_exits = 0


for trade in closed_trades:

    reason = trade.get(
        "exit_reason"
    )


    if reason == "HARD_STOP":

        hard_stop_losses += 1


    elif reason == "RSI_70":

        rsi_70_final_exits += 1


    elif reason == "BREAK_EVEN_STOP":

        break_even_exits += 1


    for sale in trade["sales"]:

        if sale["reason"] == "RSI_60":

            rsi_60_sales += 1


# ==============================================================
# ENTRY RSI ANALYSIS
# ==============================================================

rsi_below_25 = []

rsi_25_to_30 = []

rsi_30_to_33 = []


for trade in closed_trades:

    rsi = trade.get(
        "entry_rsi"
    )


    if rsi is None:

        continue


    if rsi < 25:

        rsi_below_25.append(
            trade
        )

    elif rsi < 30:

        rsi_25_to_30.append(
            trade
        )

    elif rsi <= 33:

        rsi_30_to_33.append(
            trade
        )


def rsi_group_stats(group):

    if not group:

        return {

            "trades": 0,

            "winning": 0,

            "losing": 0,

            "win_rate_percent": 0,

            "average_profit_percent": 0

        }


    wins = [
        t
        for t in group
        if t["profit_pct"] > 0
    ]


    losses = [
        t
        for t in group
        if t["profit_pct"] <= 0
    ]


    return {

        "trades":
            len(group),

        "winning":
            len(wins),

        "losing":
            len(losses),

        "win_rate_percent":
            round(
                len(wins) /
                len(group) *
                100,
                2
            ),

        "average_profit_percent":
            round(
                np.mean(
                    [
                        t["profit_pct"]
                        for t in group
                    ]
                ),
                2
            )

    }


# ==============================================================
# TRADE QUALITY ANALYSIS
# ==============================================================

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


# ==============================================================
# RESULT
# ==============================================================

result = {

    "strategy":
        "Weekly RSI 33/60/70 + EMA20/40 Trend + EMA40 Slope + Pullback + Break-Even",

    "version":
        "v3.0",

    "parameters": {

        "rsi_period":
            RSI_PERIOD,

        "rsi_buy":
            RSI_BUY,

        "rsi_sell_1":
            RSI_SELL_1,

        "rsi_sell_2":
            RSI_SELL_2,

        "first_sell_percent":
            FIRST_SELL_PERCENT * 100,

        "second_sell_percent":
            SECOND_SELL_PERCENT * 100,

        "hard_stop_percent":
            HARD_STOP_PERCENT,

        "break_even_after_first_sell":
            BREAK_EVEN_AFTER_FIRST_SELL,

        "trend_filter":
            USE_TREND_FILTER,

        "ema_fast":
            EMA_FAST,

        "ema_slow":
            EMA_SLOW,

        "ema_slope_filter":
            USE_EMA_SLOPE_FILTER,

        "ema_slope_lookback":
            EMA_SLOPE_LOOKBACK,

        "pullback_filter":
            USE_PULLBACK_FILTER,

        "max_distance_from_ema40":
            MAX_DISTANCE_FROM_EMA40,

        "cooldown":
            USE_COOLDOWN,

        "cooldown_weeks":
            COOLDOWN_WEEKS

    },

    "statistics": {

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
                total_profit,
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

        "maximum_drawdown_percent":
            round(
                max_drawdown,
                2
            )

    },

    "entry_rsi_analysis": {

        "rsi_below_25":
            rsi_group_stats(
                rsi_below_25
            ),

        "rsi_25_to_30":
            rsi_group_stats(
                rsi_25_to_30
            ),

        "rsi_30_to_33":
            rsi_group_stats(
                rsi_30_to_33
            )

    },

    "exit_analysis": {

        "hard_stop_losses":
            hard_stop_losses,

        "rsi_60_first_sales":
            rsi_60_sales,

        "rsi_70_final_exits":
            rsi_70_final_exits,

        "break_even_exits":
            break_even_exits

    },

    "best_trade":
        (
            {
                "symbol":
                    best_trade["symbol"],
                "profit_pct":
                    best_trade["profit_pct"],
                "entry_date":
                    best_trade["entry_date"],
                "exit_date":
                    best_trade["exit_date"]
            }
            if best_trade
            else None
        ),

    "worst_trade":
        (
            {
                "symbol":
                    worst_trade["symbol"],
                "profit_pct":
                    worst_trade["profit_pct"],
                "entry_date":
                    worst_trade["entry_date"],
                "exit_date":
                    worst_trade["exit_date"]
            }
            if worst_trade
            else None
        ),

    "open_positions":
        open_positions,

    "trades":
        all_trades

}


# ==============================================================
# SAVE RESULT
# ==============================================================

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


# ==============================================================
# PRINT FINAL RESULTS
# ==============================================================

print("\n")

print(
    "=============================================================="
)

print(
    "FINAL BACKTEST RESULTS"
)

print(
    "=============================================================="
)


print(
    f"Total Trades              : "
    f"{total_trades}"
)

print(
    f"Winning Trades            : "
    f"{winning_count}"
)

print(
    f"Losing Trades             : "
    f"{losing_count}"
)

print(
    f"Win Rate                  : "
    f"{win_rate:.2f}%"
)

print(
    f"Sum Trade Profit          : "
    f"{total_profit:.2f}%"
)

print(
    f"Compound Portfolio Return : "
    f"{compound_return:.2f}%"
)

print(
    f"Average Winning Trade     : "
    f"{average_win:.2f}%"
)

print(
    f"Average Losing Trade      : "
    f"{average_loss:.2f}%"
)

print(
    f"Maximum Drawdown          : "
    f"{max_drawdown:.2f}%"
)


print(
    "\n--------------------------------------------------------------"
)

print(
    "ENTRY RSI ANALYSIS"
)

print(
    "--------------------------------------------------------------"
)


stats_1 = rsi_group_stats(
    rsi_below_25
)

stats_2 = rsi_group_stats(
    rsi_25_to_30
)

stats_3 = rsi_group_stats(
    rsi_30_to_33
)


print(
    f"RSI < 25     : "
    f"{stats_1['trades']} trades | "
    f"Win {stats_1['win_rate_percent']:.2f}% | "
    f"Avg {stats_1['average_profit_percent']:.2f}%"
)


print(
    f"RSI 25-30    : "
    f"{stats_2['trades']} trades | "
    f"Win {stats_2['win_rate_percent']:.2f}% | "
    f"Avg {stats_2['average_profit_percent']:.2f}%"
)


print(
    f"RSI 30-33    : "
    f"{stats_3['trades']} trades | "
    f"Win {stats_3['win_rate_percent']:.2f}% | "
    f"Avg {stats_3['average_profit_percent']:.2f}%"
)


print(
    "\n--------------------------------------------------------------"
)

print(
    "EXIT ANALYSIS"
)

print(
    "--------------------------------------------------------------"
)


print(
    f"HARD STOP                : "
    f"{hard_stop_losses}"
)

print(
    f"RSI 60 FIRST SALES       : "
    f"{rsi_60_sales}"
)

print(
    f"RSI 70 FINAL EXITS       : "
    f"{rsi_70_final_exits}"
)

print(
    f"BREAK-EVEN EXITS         : "
    f"{break_even_exits}"
)


print(
    "\n--------------------------------------------------------------"
)

print(
    "BEST / WORST TRADE"
)

print(
    "--------------------------------------------------------------"
)


if best_trade:

    print(
        f"Best Trade  : "
        f"{best_trade['symbol']} | "
        f"{best_trade['profit_pct']:.2f}% | "
        f"{best_trade['entry_date']} -> "
        f"{best_trade['exit_date']}"
    )


if worst_trade:

    print(
        f"Worst Trade : "
        f"{worst_trade['symbol']} | "
        f"{worst_trade['profit_pct']:.2f}% | "
        f"{worst_trade['entry_date']} -> "
        f"{worst_trade['exit_date']}"
    )


print(
    "\n--------------------------------------------------------------"
)

print(
    f"Open Positions           : "
    f"{len(open_positions)}"
)


print(
    "\n=============================================================="
)

print(
    "BACKTEST COMPLETE"
)

print(
    "=============================================================="
)


print(
    f"\nResults saved to: "
    f"{RESULT_FILE}"
)

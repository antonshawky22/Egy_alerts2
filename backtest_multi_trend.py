print("==============================================================")
print("EGX WEEKLY RSI 33/60/70 + BREAK-EVEN + EMA TREND FILTER")
print("BACKTEST ENGINE v2.0")
print("==============================================================")

import json
import os
import math
import pandas as pd
import numpy as np


# ==============================================================
# CONFIGURATION
# ==============================================================

DB_FILE = "egx_weekly_database_v1.json"

RESULT_FILE = "weekly_rsi_backtest_results.json"

# --------------------------------------------------------------
# RSI
# --------------------------------------------------------------

RSI_PERIOD = 14

RSI_BUY = 33

RSI_SELL_1 = 60

RSI_SELL_2 = 70


# --------------------------------------------------------------
# Position Management
# --------------------------------------------------------------

FIRST_SELL_PERCENT = 0.50

SECOND_SELL_PERCENT = 0.50

HARD_STOP_PERCENT = 5.0

BREAK_EVEN_AFTER_FIRST_SELL = True


# --------------------------------------------------------------
# Trend Filter
# --------------------------------------------------------------

USE_TREND_FILTER = True

EMA_FAST = 20

EMA_SLOW = 40


# --------------------------------------------------------------
# Backtest
# --------------------------------------------------------------

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


with open(DB_FILE, "r", encoding="utf-8") as f:
    raw_database = json.load(f)


print(
    f"Database loaded successfully: "
    f"{len(raw_database)} symbols"
)


# ==============================================================
# RSI FUNCTION
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

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ==============================================================
# DATABASE -> DATAFRAME
# ==============================================================

def database_to_dataframe(symbol_data):

    if not symbol_data:
        return None

    # ----------------------------------------------------------
    # Format 1:
    # {
    #   "columns": [...],
    #   "data": {
    #       "2026-01-01": [...]
    #   }
    # }
    # ----------------------------------------------------------

    if isinstance(symbol_data, dict):

        columns = symbol_data.get("columns")

        data = symbol_data.get("data")

        if columns is not None and isinstance(data, dict):

            rows = []

            for date, values in data.items():

                if isinstance(values, dict):

                    row = values.copy()

                    row["Date"] = date

                    rows.append(row)

                else:

                    if len(values) != len(columns):
                        continue

                    row = dict(zip(columns, values))

                    row["Date"] = date

                    rows.append(row)

            if not rows:
                return None

            df = pd.DataFrame(rows)

    # ----------------------------------------------------------
    # Format 2:
    # list of records
    # ----------------------------------------------------------

    elif isinstance(symbol_data, list):

        df = pd.DataFrame(symbol_data)

    else:

        return None


    # ----------------------------------------------------------
    # Normalize columns
    # ----------------------------------------------------------

    df.columns = [
        str(c).strip().capitalize()
        for c in df.columns
    ]


    if "Date" not in df.columns:
        return None


    required = ["Open", "High", "Low", "Close"]

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
        subset=["Date", "Open", "High", "Low", "Close"]
    )


    df = df.sort_values("Date")

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    df = df.reset_index(drop=True)


    return df


# ==============================================================
# PREPARE INDICATORS
# ==============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["RSI14"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )


    df["EMA20"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()


    df["EMA40"] = df["Close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()


    # ----------------------------------------------------------
    # Trend
    # ----------------------------------------------------------

    df["TREND_UP"] = (
        df["EMA20"] > df["EMA40"]
    )


    return df


# ==============================================================
# DATE FILTER
# ==============================================================

def apply_date_filter(df):

    if START_DATE is not None:

        df = df[
            df["Date"] >= pd.Timestamp(START_DATE)
        ]


    if END_DATE is not None:

        df = df[
            df["Date"] <= pd.Timestamp(END_DATE)
        ]


    return df.reset_index(drop=True)


# ==============================================================
# BACKTEST ONE SYMBOL
# ==============================================================

def backtest_symbol(symbol, df):

    trades = []

    position = None

    # ----------------------------------------------------------
    # Iterate weekly candles
    # ----------------------------------------------------------

    for i in range(len(df)):

        row = df.iloc[i]

        date = row["Date"]

        close = float(row["Close"])

        low = float(row["Low"])

        rsi = row["RSI14"]

        trend_up = bool(row["TREND_UP"])


        if pd.isna(rsi):

            continue


        # ======================================================
        # NO POSITION
        # ======================================================

        if position is None:

            # --------------------------------------------------
            # ENTRY
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


            if buy_signal:

                position = {

                    "symbol": symbol,

                    "status": "OPEN",

                    "entry_date":
                        date.strftime("%Y-%m-%d"),

                    "entry_price": close,

                    "position": 1.0,

                    "original_position": 1.0,

                    "sales": [],

                    "first_sell_done": False,

                    "break_even_active": False,

                    "entry_rsi": float(rsi),

                }

                continue


        # ======================================================
        # EXISTING POSITION
        # ======================================================

        if position is not None:

            entry_price = position["entry_price"]


            # ==================================================
            # HARD STOP
            # ==================================================

            hard_stop_price = (
                entry_price
                * (1 - HARD_STOP_PERCENT / 100)
            )


            # --------------------------------------------------
            # IMPORTANT:
            #
            # We use LOW <= stop to detect that the stop was
            # touched during the weekly candle.
            #
            # But because weekly data has no intraday execution,
            # actual exit is represented using CLOSE.
            #
            # Therefore gap/slippage is preserved.
            # --------------------------------------------------

            if low <= hard_stop_price:

                exit_price = close

                remaining_position = position["position"]


                profit_pct = (
                    (exit_price - entry_price)
                    / entry_price
                    * 100
                )


                position["sales"].append({

                    "date":
                        date.strftime("%Y-%m-%d"),

                    "price":
                        exit_price,

                    "position_sold":
                        remaining_position,

                    "profit_pct":
                        profit_pct,

                    "reason":
                        "HARD_STOP"

                })


                position["status"] = "CLOSED"

                position["exit_date"] = (
                    date.strftime("%Y-%m-%d")
                )

                position["exit_price"] = exit_price

                position["profit_pct"] = (
                    calculate_trade_profit(
                        position
                    )
                )

                position["exit_reason"] = "HARD_STOP"


                trades.append(position)

                position = None

                continue


            # ==================================================
            # FIRST SELL - RSI 60
            # ==================================================

            if (
                not position["first_sell_done"]
                and rsi >= RSI_SELL_1
            ):

                sell_position = (
                    position["position"]
                    * FIRST_SELL_PERCENT
                )


                if sell_position > 0:

                    sell_price = close


                    profit_pct = (
                        (sell_price - entry_price)
                        / entry_price
                        * 100
                    )


                    position["sales"].append({

                        "date":
                            date.strftime("%Y-%m-%d"),

                        "price":
                            sell_price,

                        "position_sold":
                            sell_position,

                        "profit_pct":
                            profit_pct,

                        "reason":
                            "RSI_60"

                    })


                    position["position"] -= (
                        sell_position
                    )


                    position["first_sell_done"] = True


                    # ------------------------------------------
                    # BREAK EVEN
                    # ------------------------------------------

                    if BREAK_EVEN_AFTER_FIRST_SELL:

                        position[
                            "break_even_active"
                        ] = True


                continue


            # ==================================================
            # BREAK EVEN STOP
            # ==================================================

            if (
                position["first_sell_done"]
                and position["break_even_active"]
                and position["position"] > 0
            ):

                # ------------------------------------------------
                # We only activate BE after first partial sale.
                #
                # If price closes below entry, exit remaining.
                # ------------------------------------------------

                if close <= entry_price:

                    sell_price = close

                    remaining_position = (
                        position["position"]
                    )


                    profit_pct = (
                        (sell_price - entry_price)
                        / entry_price
                        * 100
                    )


                    position["sales"].append({

                        "date":
                            date.strftime("%Y-%m-%d"),

                        "price":
                            sell_price,

                        "position_sold":
                            remaining_position,

                        "profit_pct":
                            profit_pct,

                        "reason":
                            "BREAK_EVEN_STOP"

                    })


                    position["position"] = 0.0

                    position["status"] = "CLOSED"

                    position["exit_date"] = (
                        date.strftime("%Y-%m-%d")
                    )

                    position["exit_price"] = sell_price

                    position["profit_pct"] = (
                        calculate_trade_profit(
                            position
                        )
                    )

                    position["exit_reason"] = (
                        "BREAK_EVEN_STOP"
                    )


                    trades.append(position)

                    position = None

                    continue


            # ==================================================
            # SECOND SELL - RSI 70
            # ==================================================

            if (
                position is not None
                and position["first_sell_done"]
                and position["position"] > 0
                and rsi >= RSI_SELL_2
            ):

                sell_price = close

                remaining_position = (
                    position["position"]
                )


                profit_pct = (
                    (sell_price - entry_price)
                    / entry_price
                    * 100
                )


                position["sales"].append({

                    "date":
                        date.strftime("%Y-%m-%d"),

                    "price":
                        sell_price,

                    "position_sold":
                        remaining_position,

                    "profit_pct":
                        profit_pct,

                    "reason":
                        "RSI_70"

                })


                position["position"] = 0.0

                position["status"] = "CLOSED"

                position["exit_date"] = (
                    date.strftime("%Y-%m-%d")
                )

                position["exit_price"] = sell_price

                position["profit_pct"] = (
                    calculate_trade_profit(
                        position
                    )
                )

                position["exit_reason"] = "RSI_70"


                trades.append(position)

                position = None

                continue


    # ==========================================================
    # OPEN POSITION
    # ==========================================================

    if position is not None:

        position["status"] = "OPEN"

        position["exit_date"] = None

        position["exit_price"] = None

        position["profit_pct"] = None

        trades.append(position)


    return trades


# ==============================================================
# CALCULATE ACTUAL TRADE PROFIT
# ==============================================================

def calculate_trade_profit(trade):

    total_profit = 0.0


    for sale in trade["sales"]:

        weight = sale["position_sold"]

        profit = sale["profit_pct"]

        total_profit += (
            weight * profit
        )


    return round(total_profit, 2)


# ==============================================================
# RUN ALL SYMBOLS
# ==============================================================

all_trades = []

open_positions = []


print("\nStarting backtest...\n")


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


    df = prepare_dataframe(df)

    df = apply_date_filter(df)


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
# SORT TRADES
# ==============================================================

all_trades.sort(
    key=lambda x: x["entry_date"]
)


# ==============================================================
# SEPARATE CLOSED / OPEN
# ==============================================================

closed_trades = [
    t for t in all_trades
    if t["status"] == "CLOSED"
]


open_positions = [
    t for t in all_trades
    if t["status"] == "OPEN"
]


# ==============================================================
# BASIC STATISTICS
# ==============================================================

total_trades = len(closed_trades)


winning_trades = [
    t for t in closed_trades
    if t["profit_pct"] > 0
]


losing_trades = [
    t for t in closed_trades
    if t["profit_pct"] <= 0
]


winning_count = len(winning_trades)

losing_count = len(losing_trades)


win_rate = (
    winning_count
    / total_trades
    * 100
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
# SUM OF TRADE RETURNS
# ==============================================================

total_profit = sum(
    t["profit_pct"]
    for t in closed_trades
)


# ==============================================================
# COMPOUND PORTFOLIO RETURN
# ==============================================================

portfolio_value = INITIAL_CAPITAL

equity_curve = []


for trade in closed_trades:

    trade_return = (
        trade["profit_pct"]
        / 100
    )


    portfolio_value *= (
        1 + trade_return
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
        / INITIAL_CAPITAL
    )
    - 1
) * 100


# ==============================================================
# MAXIMUM DRAWDOWN
# ==============================================================

peak = INITIAL_CAPITAL

max_drawdown = 0.0


for point in equity_curve:

    equity = point["equity"]


    if equity > peak:

        peak = equity


    drawdown = (
        (peak - equity)
        / peak
        * 100
    )


    if drawdown > max_drawdown:

        max_drawdown = drawdown


# ==============================================================
# EXIT ANALYSIS
# ==============================================================

hard_stop_losses = 0

rsi_60_winners = 0

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

            rsi_60_winners += 1


# ==============================================================
# ENTRY RSI ANALYSIS
# ==============================================================

below_30 = []

between_30_33 = []


for trade in closed_trades:

    rsi = trade.get(
        "entry_rsi"
    )


    if rsi is None:
        continue


    if rsi < 30:

        below_30.append(trade)


    elif rsi <= 33:

        between_30_33.append(trade)


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
        t for t in group
        if t["profit_pct"] > 0
    ]


    losses = [
        t for t in group
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
                len(wins)
                / len(group)
                * 100,
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
# RESULT
# ==============================================================

result = {

    "strategy":
        "Weekly RSI 33/60/70 + Break-Even + EMA20/40 Trend Filter",

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
            EMA_SLOW

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

        "rsi_below_30":
            rsi_group_stats(
                below_30
            ),

        "rsi_30_to_33":
            rsi_group_stats(
                between_30_33
            )

    },

    "exit_analysis": {

        "hard_stop_losses":
            hard_stop_losses,

        "rsi_60_first_sales":
            rsi_60_winners,

        "rsi_70_final_exits":
            rsi_70_final_exits,

        "break_even_exits":
            break_even_exits

    },

    "open_positions":
        open_positions,

    "trades":
        all_trades

}


# ==============================================================
# SAVE JSON
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
# PRINT SUMMARY
# ==============================================================

print("\n")
print("==============================================================")
print("BACKTEST COMPLETE")
print("==============================================================")

print(
    f"Total Trades              : {total_trades}"
)

print(
    f"Winning Trades            : {winning_count}"
)

print(
    f"Losing Trades             : {losing_count}"
)

print(
    f"Win Rate                  : {win_rate:.2f}%"
)

print(
    f"Sum Trade Profit          : {total_profit:.2f}%"
)

print(
    f"Compound Portfolio Return : {compound_return:.2f}%"
)

print(
    f"Average Winning Trade    : {average_win:.2f}%"
)

print(
    f"Average Losing Trade     : {average_loss:.2f}%"
)

print(
    f"Maximum Drawdown         : {max_drawdown:.2f}%"
)

print("\n--------------------------------------------------------------")

print(
    f"HARD STOP                : {hard_stop_losses}"
)

print(
    f"RSI 60 SALES             : {rsi_60_winners}"
)

print(
    f"RSI 70 FINAL EXITS       : {rsi_70_final_exits}"
)

print(
    f"BREAK-EVEN EXITS         : {break_even_exits}"
)

print("\n--------------------------------------------------------------")

print(
    f"Open Positions           : {len(open_positions)}"
)

print("==============================================================")

print(
    f"\nResults saved to: {RESULT_FILE}"
)

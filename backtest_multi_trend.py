print("=" * 68)
print("EGX WEEKLY SIMPLE PULLBACK BACKTEST v6.1")
print("SIMPLE ENTRY + SIMPLE EXIT + PROFIT PROTECTION")
print("=" * 68)

import json
import os
import pandas as pd
import numpy as np


# ==============================================================
# CONFIG
# ==============================================================

DB_FILE = "egx_weekly_database_v1.json"

RESULT_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

INITIAL_CAPITAL = 100000.0

# Maximum simultaneous positions
MAX_POSITIONS = 8

# Equal portfolio allocation
POSITION_SIZE = 1.0 / MAX_POSITIONS


# ==============================================================
# STRATEGY PARAMETERS
# ==============================================================

RSI_PERIOD = 14

# Deeper weekly pullback
RSI_ENTRY = 32

# Simple exit
RSI_EXIT = 72


# Stronger long-term trend structure
EMA_FAST = 50
EMA_MID = 70
EMA_LONG = 100


# Initial protection
STOP_LOSS_PERCENT = 7.0


# Profit protection
TRAIL_START_PERCENT = 8.0
TRAIL_DISTANCE_PERCENT = 5.0


MIN_BARS = 100


# ==============================================================
# LOAD DATABASE
# ==============================================================

print("\nLoading database...")

if not os.path.exists(DB_FILE):

    raise FileNotFoundError(
        f"Database file not found: {DB_FILE}"
    )


with open(
    DB_FILE,
    "r",
    encoding="utf-8"
) as f:

    database = json.load(f)


print(
    f"Database loaded: {len(database)} symbols"
)


# ==============================================================
# RSI
# ==============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

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

def database_to_dataframe(data):

    # ----------------------------------------------------------
    # Format:
    #
    # {
    #   "data": {...},
    #   "columns": [...]
    # }
    # ----------------------------------------------------------

    if (
        isinstance(data, dict)
        and
        "data" in data
        and
        "columns" in data
    ):

        rows = []

        for date, values in data["data"].items():

            if isinstance(values, dict):

                row = values.copy()

            else:

                if len(values) != len(
                    data["columns"]
                ):

                    continue

                row = dict(
                    zip(
                        data["columns"],
                        values
                    )
                )

            row["Date"] = date

            rows.append(row)

        df = pd.DataFrame(rows)


    # ----------------------------------------------------------
    # Simple list format
    # ----------------------------------------------------------

    elif isinstance(data, list):

        df = pd.DataFrame(data)


    else:

        return None


    if df.empty:

        return None


    # ----------------------------------------------------------
    # Normalize column names
    # ----------------------------------------------------------

    df.columns = [
        str(c).strip().capitalize()
        for c in df.columns
    ]


    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close"
    ]


    if not all(
        c in df.columns
        for c in required
    ):

        return None


    # ----------------------------------------------------------
    # Convert data
    # ----------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )


    for column in required[1:]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # ----------------------------------------------------------
    # Clean
    # ----------------------------------------------------------

    df = df.dropna(
        subset=required
    )


    df = df.sort_values(
        "Date"
    )


    df = df.drop_duplicates(
        "Date",
        keep="last"
    )


    return df.reset_index(
        drop=True
    )


# ==============================================================
# PREPARE INDICATORS
# ==============================================================

def prepare_dataframe(df):

    df = df.copy()


    # ----------------------------------------------------------
    # RSI
    # ----------------------------------------------------------

    df["RSI"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )


    # ----------------------------------------------------------
    # EMAs
    #
    # Actual values:
    #
    # EMA50
    # EMA70
    # EMA100
    # ----------------------------------------------------------

    df["EMA_FAST"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()


    df["EMA_MID"] = df["Close"].ewm(
        span=EMA_MID,
        adjust=False
    ).mean()


    df["EMA_LONG"] = df["Close"].ewm(
        span=EMA_LONG,
        adjust=False
    ).mean()


    # ----------------------------------------------------------
    # SIMPLE UPTREND
    #
    # EMA50 > EMA70 > EMA100
    # ----------------------------------------------------------

    df["UPTREND"] = (

        (df["EMA_FAST"] > df["EMA_MID"])

        &

        (df["EMA_MID"] > df["EMA_LONG"])

    )


    return df


# ==============================================================
# TRADE PROFIT
# ==============================================================

def calculate_trade_profit(trade):

    total = 0.0


    for sale in trade["sales"]:

        total += sale["capital_return"]


    return round(
        total,
        2
    )


# ==============================================================
# BACKTEST ONE SYMBOL
# ==============================================================

def backtest_symbol(symbol, df):

    trades = []

    position = None


    for i in range(len(df)):

        row = df.iloc[i]


        date = row["Date"].strftime(
            "%Y-%m-%d"
        )


        close = float(
            row["Close"]
        )


        high = float(
            row["High"]
        )


        low = float(
            row["Low"]
        )


        rsi = row["RSI"]


        if pd.isna(rsi):

            continue


        rsi = float(rsi)


        uptrend = bool(
            row["UPTREND"]
        )


        # ======================================================
        # ENTRY
        # ======================================================

        if position is None:

            if (

                uptrend

                and

                rsi <= RSI_ENTRY

            ):

                position = {

                    "symbol":
                        symbol,

                    "status":
                        "OPEN",

                    "entry_date":
                        date,

                    "entry_price":
                        close,

                    "avg_price":
                        close,

                    "weight":
                        1.0,

                    "highest_price":
                        close,

                    "sales":
                        [],

                    "entry_rsi":
                        round(
                            rsi,
                            2
                        ),

                    "trail_active":
                        False
                }


                continue


        # ======================================================
        # NO POSITION
        # ======================================================

        if position is None:

            continue


        # ======================================================
        # PREVIOUS HIGH
        #
        # Current candle high is NOT used to create a trailing
        # stop for the same candle.
        #
        # This avoids look-ahead bias.
        # ======================================================

        if i > 0:

            previous_high = float(
                df.iloc[i - 1]["High"]
            )

        else:

            previous_high = (
                position["highest_price"]
            )


        previous_highest = max(

            position["highest_price"],
            previous_high

        )


        # ======================================================
        # CURRENT POSITION
        # ======================================================

        avg_price = float(
            position["avg_price"]
        )


        # ======================================================
        # INITIAL STOP
        # ======================================================

        initial_stop = (

            avg_price *

            (
                1 -
                STOP_LOSS_PERCENT / 100
            )

        )


        stop_price = initial_stop


        # ======================================================
        # PROFIT PROTECTION
        #
        # Activate after +8%.
        #
        # Protect 5% below the highest known price.
        # ======================================================

        previous_profit = (

            (
                previous_highest -
                avg_price
            )
            /
            avg_price

        ) * 100


        if previous_profit >= TRAIL_START_PERCENT:

            position["trail_active"] = True


            trailing_stop = (

                previous_highest *

                (
                    1 -
                    TRAIL_DISTANCE_PERCENT / 100
                )

            )


            stop_price = max(

                stop_price,
                trailing_stop

            )


        # ======================================================
        # STOP LOSS / TRAILING STOP
        # ======================================================

        if low <= stop_price:

            # --------------------------------------------------
            # Keep the same accounting convention:
            #
            # Exit is recorded at weekly CLOSE.
            # --------------------------------------------------

            exit_price = close


            profit_pct = (

                (
                    exit_price -
                    avg_price
                )
                /
                avg_price

            ) * 100


            weight = position["weight"]


            capital_return = (

                profit_pct / 100

            ) * weight


            if position["trail_active"]:

                reason = "TRAIL_STOP"

            else:

                reason = "STOP_LOSS"


            position["sales"].append({

                "date":
                    date,

                "price":
                    exit_price,

                "weight":
                    weight,

                "profit_pct":
                    round(
                        profit_pct,
                        2
                    ),

                "capital_return":
                    round(
                        capital_return * 100,
                        4
                    ),

                "reason":
                    reason

            })


            position["weight"] = 0.0


            position["status"] = "CLOSED"


            position["exit_date"] = date


            position["exit_price"] = exit_price


            position["exit_reason"] = reason


            position["profit_pct"] = (

                calculate_trade_profit(
                    position
                )

            )


            trades.append(
                position
            )


            position = None


            continue


        # ======================================================
        # RSI EXIT
        # ======================================================

        if rsi >= RSI_EXIT:

            exit_price = close


            profit_pct = (

                (
                    exit_price -
                    avg_price
                )
                /
                avg_price

            ) * 100


            weight = position["weight"]


            capital_return = (

                profit_pct / 100

            ) * weight


            position["sales"].append({

                "date":
                    date,

                "price":
                    exit_price,

                "weight":
                    weight,

                "profit_pct":
                    round(
                        profit_pct,
                        2
                    ),

                "capital_return":
                    round(
                        capital_return * 100,
                        4
                    ),

                "reason":
                    "RSI_EXIT"

            })


            position["weight"] = 0.0


            position["status"] = "CLOSED"


            position["exit_date"] = date


            position["exit_price"] = exit_price


            position["exit_reason"] = (
                "RSI_EXIT"
            )


            position["profit_pct"] = (

                calculate_trade_profit(
                    position
                )

            )


            trades.append(
                position
            )


            position = None


            continue


        # ======================================================
        # UPDATE HIGHEST PRICE
        #
        # Done only after all decisions.
        # ======================================================

        position["highest_price"] = max(

            position["highest_price"],
            high

        )


    # ==========================================================
    # OPEN POSITION
    # ==========================================================

    if position is not None:

        last_price = float(
            df.iloc[-1]["Close"]
        )


        position["status"] = "OPEN"


        position["last_price"] = (
            last_price
        )


        position["unrealized_pct"] = round(

            (

                (
                    last_price -
                    position["avg_price"]
                )
                /
                position["avg_price"]

            ) * 100,

            2

        )


        trades.append(
            position
        )


    return trades


# ==============================================================
# RUN BACKTEST
# ==============================================================

all_trades = []


print("\nStarting backtest...\n")


for symbol, data in database.items():

    # ----------------------------------------------------------
    # Ignore indexes
    # ----------------------------------------------------------

    if symbol.upper() in [

        "EGX30",
        "EGX70",
        "EGX100"

    ]:

        continue


    df = database_to_dataframe(
        data
    )


    if df is None:

        print(
            f"⚠️ {symbol}: invalid data"
        )

        continue


    if len(df) < MIN_BARS:

        print(

            f"⚠️ {symbol}: "
            f"{len(df)} bars"

        )

        continue


    df = prepare_dataframe(
        df
    )


    trades = backtest_symbol(
        symbol,
        df
    )


    all_trades.extend(
        trades
    )


    closed = sum(

        1
        for t in trades
        if t["status"] == "CLOSED"

    )


    print(

        f"{symbol:8} | "
        f"Closed Trades: {closed:3}"

    )


# ==============================================================
# SORT TRADES
# ==============================================================

all_trades.sort(

    key=lambda x:
    x["entry_date"]

)


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
# TRADE STATISTICS
# ==============================================================

profits = [

    float(t["profit_pct"])
    for t in closed_trades

]


winning = [

    p
    for p in profits
    if p > 0

]


losing = [

    p
    for p in profits
    if p <= 0

]


total_trades = len(
    profits
)


win_rate = (

    len(winning)
    /
    total_trades
    *
    100

    if total_trades

    else 0

)


sum_profit = sum(
    profits
)


average_win = (

    np.mean(winning)

    if winning

    else 0

)


average_loss = (

    np.mean(losing)

    if losing

    else 0

)


# ==============================================================
# REALISTIC PORTFOLIO RETURN
# ==============================================================
#
# Every closed trade receives 1/8 of portfolio.
#
# +10% trade
# =
# +1.25% portfolio impact
#
# before compounding.
# ==============================================================

portfolio = INITIAL_CAPITAL


equity_curve = []


portfolio_trades = []


for trade in closed_trades:

    trade_return = (

        trade["profit_pct"]
        /
        100

    )


    weighted_return = (

        trade_return *
        POSITION_SIZE

    )


    portfolio *= (

        1 +
        weighted_return

    )


    equity_curve.append(
        portfolio
    )


    portfolio_trades.append({

        "date":
            trade["exit_date"],

        "symbol":
            trade["symbol"],

        "trade_return_percent":
            round(
                trade["profit_pct"],
                2
            ),

        "portfolio_return_percent":
            round(
                weighted_return * 100,
                4
            ),

        "portfolio_value":
            round(
                portfolio,
                2
            )

    })


compound_return = (

    portfolio
    /
    INITIAL_CAPITAL
    -
    1

) * 100


# ==============================================================
# MAX DRAWDOWN
# ==============================================================

peak = INITIAL_CAPITAL


max_drawdown = 0.0


for value in equity_curve:

    if value > peak:

        peak = value


    drawdown = (

        (
            peak -
            value
        )
        /
        peak

    ) * 100


    max_drawdown = max(

        max_drawdown,
        drawdown

    )


# ==============================================================
# EXIT ANALYSIS
# ==============================================================

exit_analysis = {}


for trade in closed_trades:

    reason = trade.get(

        "exit_reason",
        "UNKNOWN"

    )


    exit_analysis[reason] = (

        exit_analysis.get(
            reason,
            0
        )
        +
        1

    )


# ==============================================================
# BEST TRADE
# ==============================================================

best_trade = (

    max(

        closed_trades,

        key=lambda x:
        x["profit_pct"]

    )

    if closed_trades

    else None

)


# ==============================================================
# WORST TRADE
# ==============================================================

worst_trade = (

    min(

        closed_trades,

        key=lambda x:
        x["profit_pct"]

    )

    if closed_trades

    else None

)


# ==============================================================
# RESULT
# ==============================================================

result = {

    "strategy":
        "Weekly Simple Pullback v6.1",

    "description":
        "Simple Entry + Simple Exit + Profit Protection",

    "parameters": {

        "rsi_period":
            RSI_PERIOD,

        "rsi_entry":
            RSI_ENTRY,

        "rsi_exit":
            RSI_EXIT,

        "ema_fast":
            EMA_FAST,

        "ema_mid":
            EMA_MID,

        "ema_long":
            EMA_LONG,

        "stop_loss_percent":
            STOP_LOSS_PERCENT,

        "trail_start_percent":
            TRAIL_START_PERCENT,

        "trail_distance_percent":
            TRAIL_DISTANCE_PERCENT,

        "max_positions":
            MAX_POSITIONS,

        "position_size_percent":
            POSITION_SIZE * 100

    },


    "statistics": {

        "total_trades":
            total_trades,

        "winning_trades":
            len(winning),

        "losing_trades":
            len(losing),

        "win_rate_percent":
            round(
                win_rate,
                2
            ),

        "sum_trade_profit_percent":
            round(
                sum_profit,
                2
            ),

        "realistic_compound_return_percent":
            round(
                compound_return,
                2
            ),

        "average_win_percent":
            round(
                average_win,
                2
            ),

        "average_loss_percent":
            round(
                average_loss,
                2
            ),

        "maximum_drawdown_percent":
            round(
                max_drawdown,
                2
            ),

        "open_positions":
            len(open_positions)

    },


    "exit_analysis":
        exit_analysis,


    "best_trade":
        best_trade,


    "worst_trade":
        worst_trade,


    "open_positions":
        open_positions,


    "portfolio_equity":
        portfolio_trades,


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
# SAVE TRADES
# ==============================================================

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


# ==============================================================
# OUTPUT
# ==============================================================

print("\n")


print("=" * 68)
print("FINAL BACKTEST RESULTS")
print("=" * 68)


print(

    f"Total Trades              : "
    f"{total_trades}"

)


print(

    f"Winning Trades            : "
    f"{len(winning)}"

)


print(

    f"Losing Trades             : "
    f"{len(losing)}"

)


print(

    f"Win Rate                  : "
    f"{win_rate:.2f}%"

)


print(

    f"Sum Trade Profit          : "
    f"{sum_profit:.2f}%"

)


print(

    f"REALISTIC COMPOUND RETURN : "
    f"{compound_return:.2f}%"

)


print(

    f"Average Win               : "
    f"{average_win:.2f}%"

)


print(

    f"Average Loss              : "
    f"{average_loss:.2f}%"

)


print(

    f"Maximum Drawdown          : "
    f"{max_drawdown:.2f}%"

)


print(

    f"Open Positions            : "
    f"{len(open_positions)}"

)


# ==============================================================
# EXIT ANALYSIS
# ==============================================================

print("\nEXIT ANALYSIS")
print("-" * 68)


for reason, count in exit_analysis.items():

    print(

        f"{reason:25} : {count}"

    )


# ==============================================================
# BEST TRADE
# ==============================================================

if best_trade:

    print("\nBEST TRADE")


    print(

        f"{best_trade['symbol']} | "
        f"{best_trade['profit_pct']:.2f}% | "
        f"{best_trade['entry_date']} -> "
        f"{best_trade['exit_date']}"

    )


# ==============================================================
# WORST TRADE
# ==============================================================

if worst_trade:

    print("\nWORST TRADE")


    print(

        f"{worst_trade['symbol']} | "
        f"{worst_trade['profit_pct']:.2f}% | "
        f"{worst_trade['entry_date']} -> "
        f"{worst_trade['exit_date']}"

    )


# ==============================================================
# FILES
# ==============================================================

print("\nFILES SAVED")


print(
    f"  {RESULT_FILE}"
)


print(
    f"  {TRADES_FILE}"
)


print("=" * 68)

print("BACKTEST COMPLETE")

print("=" * 68)

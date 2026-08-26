print("="*68)
print("EGX WEEKLY SMART PULLBACK BACKTEST v5.0")
print("NO LOOK-AHEAD + REAL PORTFOLIO ACCOUNTING")
print("="*68)

import json, os
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

# Capital allocated to each new position
POSITION_SIZE = 1.0 / MAX_POSITIONS

# ==============================================================
# STRATEGY PARAMETERS
# ==============================================================

RSI_PERIOD = 14

RSI_ENTRY = 42
RSI_ADD = 36

RSI_SELL_1 = 64
RSI_SELL_2 = 74

EMA_FAST = 20
EMA_MID = 40
EMA_LONG = 80

MIN_EMA_GAP = 0.003
MIN_EMA_SLOPE = 0.001

FIRST_ENTRY = 0.50
SECOND_ENTRY = 0.50

ATR_PERIOD = 14
ATR_STOP_MULT = 2.7
MAX_STOP_PERCENT = 7.0

TRAIL_START_PERCENT = 10.0
TRAIL_ATR_MULT = 3.5

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


# ==============================================================
# ATR
# ==============================================================

def calculate_atr(df, period=14):

    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"] -
                previous_close
            ).abs(),
            (
                df["Low"] -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ==============================================================
# DATABASE -> DATAFRAME
# ==============================================================

def database_to_dataframe(data):

    if (
        isinstance(data, dict)
        and "data" in data
        and "columns" in data
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

    elif isinstance(data, list):

        df = pd.DataFrame(data)

    else:

        return None

    if df.empty:
        return None

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

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    for c in required[1:]:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

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
# INDICATORS
# ==============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["RSI"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )

    df["EMA20"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["EMA40"] = df["Close"].ewm(
        span=EMA_MID,
        adjust=False
    ).mean()

    df["EMA80"] = df["Close"].ewm(
        span=EMA_LONG,
        adjust=False
    ).mean()

    df["ATR"] = calculate_atr(
        df,
        ATR_PERIOD
    )

    # 4-week EMA slope
    df["EMA20_SLOPE"] = (
        df["EMA20"] /
        df["EMA20"].shift(4)
        - 1
    )

    df["EMA40_SLOPE"] = (
        df["EMA40"] /
        df["EMA40"].shift(4)
        - 1
    )

    # Strong trend
    df["UPTREND"] = (

        (
            df["EMA20"] >
            df["EMA40"] *
            (1 + MIN_EMA_GAP)
        )

        &

        (
            df["EMA40"] >
            df["EMA80"]
        )

        &

        (
            df["EMA20_SLOPE"] >=
            MIN_EMA_SLOPE
        )

        &

        (
            df["EMA40_SLOPE"] >=
            MIN_EMA_SLOPE
        )
    )

    # Price not excessively extended
    df["PULLBACK_ZONE"] = (
        df["Close"] <=
        df["EMA20"] * 1.08
    )

    return df


# ==============================================================
# TRADE PROFIT
# ==============================================================

def calculate_trade_profit(trade):

    total = 0.0

    for sale in trade["sales"]:

        total += (
            sale["capital_return"]
        )

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

        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])

        rsi = row["RSI"]
        atr = row["ATR"]

        if pd.isna(rsi) or pd.isna(atr):
            continue

        rsi = float(rsi)
        atr = float(atr)

        uptrend = bool(
            row["UPTREND"]
        )

        pullback = bool(
            row["PULLBACK_ZONE"]
        )

        # ======================================================
        # ENTRY
        # ======================================================

        if position is None:

            if (
                uptrend
                and pullback
                and rsi <= RSI_ENTRY
            ):

                position = {

                    "symbol": symbol,

                    "status": "OPEN",

                    "entry_date": date,

                    "entry_price": close,

                    "avg_price": close,

                    "weight": FIRST_ENTRY,

                    "second_entry": False,

                    "highest_price": close,

                    "sales": [],

                    "entry_rsi": round(
                        rsi,
                        2
                    ),

                    "break_even": False,

                    "trail_active": False
                }

                continue

        if position is None:
            continue

        # ======================================================
        # IMPORTANT:
        #
        # Use PREVIOUS candle high for trailing calculation.
        # Current candle high is NOT allowed to create a
        # trailing stop for the same candle.
        # ======================================================

        previous_high = (
            float(df.iloc[i - 1]["High"])
            if i > 0
            else position["highest_price"]
        )

        previous_highest = max(
            position["highest_price"],
            previous_high
        )

        # ======================================================
        # STOP BASED ON PREVIOUS INFORMATION
        # ======================================================

        avg_price = position["avg_price"]

        atr_stop = (
            avg_price -
            ATR_STOP_MULT * atr
        )

        fixed_stop = (
            avg_price *
            (
                1 -
                MAX_STOP_PERCENT / 100
            )
        )

        stop_price = max(
            atr_stop,
            fixed_stop
        )

        # ======================================================
        # TRAILING STOP
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

            trail_stop = (
                previous_highest -
                TRAIL_ATR_MULT * atr
            )

            stop_price = max(
                stop_price,
                trail_stop
            )

        # Break-even protection
        if position["break_even"]:

            stop_price = max(
                stop_price,
                avg_price
            )

        # ======================================================
        # STOP EXECUTION
        # ======================================================

        if low <= stop_price:

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

            reason = (
                "TRAIL_STOP"
                if position["trail_active"]
                else "ATR_STOP"
            )

            position["sales"].append({

                "date": date,

                "price": exit_price,

                "weight": weight,

                "profit_pct": round(
                    profit_pct,
                    2
                ),

                "capital_return": round(
                    capital_return * 100,
                    4
                ),

                "reason": reason
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
        # SECOND ENTRY
        # ======================================================

        if (
            not position["second_entry"]
            and uptrend
            and rsi <= RSI_ADD
        ):

            old_weight = position["weight"]

            new_weight = SECOND_ENTRY

            second_price = close

            total_weight = (
                old_weight +
                new_weight
            )

            total_cost = (
                position["avg_price"] *
                old_weight
                +
                second_price *
                new_weight
            )

            position["avg_price"] = (
                total_cost /
                total_weight
            )

            position["weight"] = (
                total_weight
            )

            position["second_entry"] = True

            continue

        # ======================================================
        # FIRST PARTIAL SELL
        # ======================================================

        if (
            len(position["sales"]) == 0
            and
            rsi >= RSI_SELL_1
        ):

            sell_weight = min(
                FIRST_ENTRY,
                position["weight"]
            )

            profit_pct = (
                (
                    close -
                    avg_price
                )
                /
                avg_price
            ) * 100

            capital_return = (
                profit_pct / 100
            ) * sell_weight

            position["sales"].append({

                "date": date,

                "price": close,

                "weight": sell_weight,

                "profit_pct": round(
                    profit_pct,
                    2
                ),

                "capital_return": round(
                    capital_return * 100,
                    4
                ),

                "reason":
                    "RSI_PARTIAL"
            })

            position["weight"] -= (
                sell_weight
            )

            position["break_even"] = True

            continue

        # ======================================================
        # FINAL SELL
        # ======================================================

        if (
            position is not None
            and
            position["weight"] > 0
            and
            rsi >= RSI_SELL_2
        ):

            sell_weight = position["weight"]

            profit_pct = (
                (
                    close -
                    avg_price
                )
                /
                avg_price
            ) * 100

            capital_return = (
                profit_pct / 100
            ) * sell_weight

            position["sales"].append({

                "date": date,

                "price": close,

                "weight": sell_weight,

                "profit_pct": round(
                    profit_pct,
                    2
                ),

                "capital_return": round(
                    capital_return * 100,
                    4
                ),

                "reason":
                    "RSI_FINAL"
            })

            position["weight"] = 0.0

            position["status"] = "CLOSED"

            position["exit_date"] = date

            position["exit_price"] = close

            position["exit_reason"] = (
                "RSI_FINAL"
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
        # UPDATE HIGH ONLY AFTER ALL DECISIONS
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
# RUN
# ==============================================================

all_trades = []

print("\nStarting backtest...\n")

for symbol, data in database.items():

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
# SORT
# ==============================================================

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


# ==============================================================
# TRADE STATISTICS
# ==============================================================

profits = [
    float(t["profit_pct"])
    for t in closed_trades
]

winning = [
    p for p in profits
    if p > 0
]

losing = [
    p for p in profits
    if p <= 0
]

total_trades = len(
    profits
)

win_rate = (
    len(winning) /
    total_trades *
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
#
# Each complete trade receives POSITION_SIZE of capital.
# Therefore trade return is weighted by actual allocation.
# ==============================================================

portfolio = INITIAL_CAPITAL

equity_curve = []

portfolio_trades = []

for trade in closed_trades:

    trade_return = (
        trade["profit_pct"]
        / 100
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

    portfolio /
    INITIAL_CAPITAL -
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
        (peak - value)
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
        ) + 1
    )


# ==============================================================
# SECOND ENTRY ANALYSIS
# ==============================================================

second_entries = sum(
    1
    for t in all_trades
    if t.get("second_entry")
)


# ==============================================================
# BEST / WORST
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
        "Weekly Smart Pullback v5.0",

    "description":
        "No Look-Ahead + Realistic Portfolio Accounting",

    "parameters": {

        "rsi_period":
            RSI_PERIOD,

        "rsi_entry":
            RSI_ENTRY,

        "rsi_add":
            RSI_ADD,

        "rsi_sell_1":
            RSI_SELL_1,

        "rsi_sell_2":
            RSI_SELL_2,

        "ema_fast":
            EMA_FAST,

        "ema_mid":
            EMA_MID,

        "ema_long":
            EMA_LONG,

        "min_ema_gap":
            MIN_EMA_GAP,

        "min_ema_slope":
            MIN_EMA_SLOPE,

        "first_entry_percent":
            FIRST_ENTRY * 100,

        "second_entry_percent":
            SECOND_ENTRY * 100,

        "atr_period":
            ATR_PERIOD,

        "atr_stop_multiplier":
            ATR_STOP_MULT,

        "max_stop_percent":
            MAX_STOP_PERCENT,

        "trail_start_percent":
            TRAIL_START_PERCENT,

        "trail_atr_multiplier":
            TRAIL_ATR_MULT,

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

        "second_entries":
            second_entries,

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
# SAVE
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
print("="*68)
print("FINAL BACKTEST RESULTS")
print("="*68)

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
    f"Second Entries            : "
    f"{second_entries}"
)

print(
    f"Open Positions            : "
    f"{len(open_positions)}"
)

print("\nEXIT ANALYSIS")
print("-"*68)

for reason, count in exit_analysis.items():

    print(
        f"{reason:25} : {count}"
    )


if best_trade:

    print("\nBEST TRADE")

    print(
        f"{best_trade['symbol']} | "
        f"{best_trade['profit_pct']:.2f}% | "
        f"{best_trade['entry_date']} -> "
        f"{best_trade['exit_date']}"
    )


if worst_trade:

    print("\nWORST TRADE")

    print(
        f"{worst_trade['symbol']} | "
        f"{worst_trade['profit_pct']:.2f}% | "
        f"{worst_trade['entry_date']} -> "
        f"{worst_trade['exit_date']}"
    )


print("\nFILES SAVED")
print(
    f"  {RESULT_FILE}"
)
print(
    f"  {TRADES_FILE}"
)

print("="*68)
print("BACKTEST COMPLETE")
print("="*68)

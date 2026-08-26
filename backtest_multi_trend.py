print("="*68)
print("EGX WEEKLY SMART PULLBACK BACKTEST v4.0")
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
MAX_POSITIONS = 8

RSI_PERIOD = 14
RSI_ENTRY = 42
RSI_ADD = 34

EMA_FAST = 20
EMA_MID = 40
EMA_LONG = 80

MIN_EMA_GAP = 0.003
MIN_EMA_SLOPE = 0.001

RSI_SELL_1 = 64
RSI_SELL_2 = 74

FIRST_ENTRY = 0.50
SECOND_ENTRY = 0.50

ATR_PERIOD = 14
ATR_STOP_MULT = 2.7
MAX_STOP_PERCENT = 7.0

TRAIL_START_PERCENT = 6.0
TRAIL_ATR_MULT = 2.5

MIN_BARS = 100


# ==============================================================
# LOAD DATABASE
# ==============================================================

print("\nLoading database...")

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(f"Database file not found: {DB_FILE}")

with open(DB_FILE, "r", encoding="utf-8") as f:
    database = json.load(f)

print(f"Database loaded: {len(database)} symbols")


# ==============================================================
# INDICATORS
# ==============================================================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    ag = gain.ewm(
        alpha=1/period,
        adjust=False,
        min_periods=period
    ).mean()

    al = loss.ewm(
        alpha=1/period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = ag / al.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    pc = df["Close"].shift(1)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1/period,
        adjust=False,
        min_periods=period
    ).mean()


# ==============================================================
# DATABASE -> DATAFRAME
# ==============================================================

def database_to_dataframe(data):

    if isinstance(data, dict) and "data" in data and "columns" in data:

        rows = []

        for date, values in data["data"].items():

            if isinstance(values, dict):
                row = values.copy()
            else:
                if len(values) != len(data["columns"]):
                    continue
                row = dict(zip(data["columns"], values))

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
        "Date", "Open", "High", "Low", "Close"
    ]

    if not all(c in df.columns for c in required):
        return None

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    for c in required[1:]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce"
        )

    df = df.dropna(subset=required)
    df = df.sort_values("Date")
    df = df.drop_duplicates("Date", keep="last")
    df = df.reset_index(drop=True)

    return df


# ==============================================================
# PREPARE INDICATORS
# ==============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["RSI"] = calculate_rsi(
        df["Close"], RSI_PERIOD
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
        df, ATR_PERIOD
    )

    # EMA slopes
    df["EMA20_SLOPE"] = (
        df["EMA20"] /
        df["EMA20"].shift(4) - 1
    )

    df["EMA40_SLOPE"] = (
        df["EMA40"] /
        df["EMA40"].shift(4) - 1
    )

    # Strong weekly trend
    df["UPTREND"] = (
        (df["EMA20"] > df["EMA40"] * (1 + MIN_EMA_GAP))
        &
        (df["EMA40"] > df["EMA80"])
        &
        (df["EMA20_SLOPE"] >= MIN_EMA_SLOPE)
        &
        (df["EMA40_SLOPE"] >= MIN_EMA_SLOPE)
    )

    # Price must remain reasonably close to trend
    df["PULLBACK_ZONE"] = (
        df["Close"] <= df["EMA20"] * 1.08
    )

    return df


# ==============================================================
# TRADE PROFIT
# ==============================================================

def trade_profit(trade):

    total = 0.0

    for sale in trade["sales"]:
        total += (
            sale["weight"] *
            sale["profit_pct"]
        )

    return round(total, 2)


# ==============================================================
# BACKTEST ONE SYMBOL
# ==============================================================

def backtest_symbol(symbol, df):

    trades = []
    position = None

    for i in range(len(df)):

        row = df.iloc[i]

        date = row["Date"].strftime("%Y-%m-%d")

        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])

        rsi = row["RSI"]
        atr = row["ATR"]

        if pd.isna(rsi) or pd.isna(atr):
            continue

        rsi = float(rsi)
        atr = float(atr)

        uptrend = bool(row["UPTREND"])
        pullback = bool(row["PULLBACK_ZONE"])

        # ======================================================
        # ENTRY
        # ======================================================

        if position is None:

            entry_signal = (
                uptrend
                and
                pullback
                and
                rsi <= RSI_ENTRY
            )

            if entry_signal:

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
                    "entry_rsi": round(rsi, 2),
                    "break_even": False,
                    "trail_active": False
                }

                continue

        if position is None:
            continue

        # ======================================================
        # UPDATE HIGH
        # ======================================================

        position["highest_price"] = max(
            position["highest_price"],
            high
        )

        avg = position["avg_price"]

        # ======================================================
        # SECOND ENTRY
        # ======================================================

        if (
            not position["second_entry"]
            and
            uptrend
            and
            rsi <= RSI_ADD
        ):

            old_weight = position["weight"]
            new_weight = SECOND_ENTRY

            old_cost = avg * old_weight
            new_cost = close * new_weight

            total_weight = (
                old_weight + new_weight
            )

            position["avg_price"] = (
                old_cost + new_cost
            ) / total_weight

            position["weight"] = total_weight
            position["second_entry"] = True

            continue

        # ======================================================
        # STOP CALCULATION
        # ======================================================

        atr_stop = (
            avg -
            ATR_STOP_MULT * atr
        )

        fixed_stop = (
            avg *
            (1 - MAX_STOP_PERCENT / 100)
        )

        base_stop = max(
            atr_stop,
            fixed_stop
        )

        # ======================================================
        # TRAILING STOP
        # ======================================================

        profit_from_avg = (
            (position["highest_price"] - avg)
            / avg
        ) * 100

        stop_price = base_stop

        if profit_from_avg >= TRAIL_START_PERCENT:

            position["trail_active"] = True

            trail_stop = (
                position["highest_price"]
                -
                ATR_TRAIL_MULT * atr
                if False else
                position["highest_price"]
                -
                TRAIL_ATR_MULT * atr
            )

            stop_price = max(
                base_stop,
                trail_stop,
                avg if position["break_even"] else 0
            )

        # ======================================================
        # STOP
        # ======================================================

        if low <= stop_price:

            exit_price = close

            profit = (
                (exit_price - avg)
                / avg
            ) * 100

            position["sales"].append({
                "date": date,
                "price": exit_price,
                "weight": position["weight"],
                "profit_pct": round(profit, 2),
                "reason": (
                    "TRAIL_STOP"
                    if position["trail_active"]
                    else "ATR_STOP"
                )
            })

            position["weight"] = 0.0
            position["status"] = "CLOSED"
            position["exit_date"] = date
            position["exit_price"] = exit_price
            position["exit_reason"] = (
                "TRAIL_STOP"
                if position["trail_active"]
                else "ATR_STOP"
            )
            position["profit_pct"] = trade_profit(
                position
            )

            trades.append(position)
            position = None

            continue

        # ======================================================
        # FIRST SELL
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

            profit = (
                (close - avg)
                / avg
            ) * 100

            position["sales"].append({
                "date": date,
                "price": close,
                "weight": sell_weight,
                "profit_pct": round(profit, 2),
                "reason": "RSI_PARTIAL"
            })

            position["weight"] -= sell_weight

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

            profit = (
                (close - avg)
                / avg
            ) * 100

            position["sales"].append({
                "date": date,
                "price": close,
                "weight": position["weight"],
                "profit_pct": round(profit, 2),
                "reason": "RSI_FINAL"
            })

            position["weight"] = 0.0
            position["status"] = "CLOSED"
            position["exit_date"] = date
            position["exit_price"] = close
            position["exit_reason"] = "RSI_FINAL"
            position["profit_pct"] = trade_profit(
                position
            )

            trades.append(position)
            position = None

            continue

    # ==========================================================
    # OPEN POSITION
    # ==========================================================

    if position is not None:

        last_price = float(
            df.iloc[-1]["Close"]
        )

        position["status"] = "OPEN"
        position["last_price"] = last_price

        position["unrealized_pct"] = round(
            (
                (last_price - position["avg_price"])
                /
                position["avg_price"]
            ) * 100,
            2
        )

        trades.append(position)

    return trades


# ==============================================================
# RUN BACKTEST
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

    df = database_to_dataframe(data)

    if df is None:
        continue

    if len(df) < MIN_BARS:
        continue

    df = prepare_dataframe(df)

    trades = backtest_symbol(
        symbol,
        df
    )

    all_trades.extend(trades)

    closed = sum(
        1 for t in trades
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
    key=lambda x: x["entry_date"]
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
# BASIC STATISTICS
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

total_trades = len(profits)

win_rate = (
    len(winning) /
    total_trades *
    100
    if total_trades
    else 0
)

sum_profit = sum(profits)

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
# COMPOUND RETURN
# ==============================================================

portfolio = INITIAL_CAPITAL
equity_curve = []

for trade in closed_trades:

    portfolio *= (
        1 +
        trade["profit_pct"] / 100
    )

    equity_curve.append(
        portfolio
    )

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

    peak = max(
        peak,
        value
    )

    dd = (
        (peak - value)
        /
        peak
    ) * 100

    max_drawdown = max(
        max_drawdown,
        dd
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
# BEST / WORST
# ==============================================================

best_trade = (
    max(
        closed_trades,
        key=lambda x: x["profit_pct"]
    )
    if closed_trades
    else None
)

worst_trade = (
    min(
        closed_trades,
        key=lambda x: x["profit_pct"]
    )
    if closed_trades
    else None
)


# ==============================================================
# RESULT
# ==============================================================

result = {

    "strategy":
        "Weekly Smart Pullback v4.0",

    "parameters": {

        "rsi_period": RSI_PERIOD,
        "rsi_entry": RSI_ENTRY,
        "rsi_add": RSI_ADD,

        "rsi_sell_1": RSI_SELL_1,
        "rsi_sell_2": RSI_SELL_2,

        "ema_fast": EMA_FAST,
        "ema_mid": EMA_MID,
        "ema_long": EMA_LONG,

        "min_ema_gap": MIN_EMA_GAP,
        "min_ema_slope": MIN_EMA_SLOPE,

        "first_entry_percent":
            FIRST_ENTRY * 100,

        "second_entry_percent":
            SECOND_ENTRY * 100,

        "atr_period": ATR_PERIOD,

        "atr_stop_multiplier":
            ATR_STOP_MULT,

        "max_stop_percent":
            MAX_STOP_PERCENT,

        "trail_start_percent":
            TRAIL_START_PERCENT,

        "trail_atr_multiplier":
            TRAIL_ATR_MULT
    },

    "statistics": {

        "total_trades":
            total_trades,

        "winning_trades":
            len(winning),

        "losing_trades":
            len(losing),

        "win_rate_percent":
            round(win_rate, 2),

        "sum_trade_profit_percent":
            round(sum_profit, 2),

        "compound_return_percent":
            round(compound_return, 2),

        "average_win_percent":
            round(average_win, 2),

        "average_loss_percent":
            round(average_loss, 2),

        "maximum_drawdown_percent":
            round(max_drawdown, 2),

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
        open_positions
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
# FINAL OUTPUT
# ==============================================================

print("\n")
print("="*68)
print("FINAL BACKTEST RESULTS")
print("="*68)

print(
    f"Total Trades              : {total_trades}"
)

print(
    f"Winning Trades            : {len(winning)}"
)

print(
    f"Losing Trades             : {len(losing)}"
)

print(
    f"Win Rate                  : {win_rate:.2f}%"
)

print(
    f"Sum Trade Profit          : {sum_profit:.2f}%"
)

print(
    f"Compound Return           : {compound_return:.2f}%"
)

print(
    f"Average Win               : {average_win:.2f}%"
)

print(
    f"Average Loss              : {average_loss:.2f}%"
)

print(
    f"Maximum Drawdown          : {max_drawdown:.2f}%"
)

print(
    f"Open Positions            : {len(open_positions)}"
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
print(f"  {RESULT_FILE}")
print(f"  {TRADES_FILE}")

print("="*68)
print("BACKTEST COMPLETE")
print("="*68)

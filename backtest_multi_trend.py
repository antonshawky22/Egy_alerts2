print("="*68)
print("EGX WEEKLY SMART PULLBACK BACKTEST v3.0")
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

RSI_PERIOD = 14

# Trend
EMA_FAST = 20
EMA_SLOW = 40
EMA_LONG = 80

# Entry
RSI_ENTRY = 40
RSI_ADD = 32

# Momentum confirmation
MIN_EMA_GAP = 0.005          # EMA20 must be 0.5% above EMA40

# Position
FIRST_ENTRY = 0.50
SECOND_ENTRY = 0.50

# Exit
RSI_SELL_1 = 62
RSI_SELL_2 = 72

# Risk
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5
MAX_STOP_PERCENT = 8.0

# Minimum data
MIN_BARS = 100

print("\nLoading database...")

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(f"File not found: {DB_FILE}")

with open(DB_FILE, "r", encoding="utf-8") as f:
    database = json.load(f)

print(f"Database loaded: {len(database)} symbols")


# ==============================================================
# INDICATORS
# ==============================================================

def rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)

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


def atr(df, period=14):
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
# DATABASE CONVERTER
# ==============================================================

def to_df(x):

    if isinstance(x, dict) and "data" in x and "columns" in x:

        rows = []

        for date, values in x["data"].items():

            if isinstance(values, dict):
                row = values.copy()
            else:
                if len(values) != len(x["columns"]):
                    continue
                row = dict(zip(x["columns"], values))

            row["Date"] = date
            rows.append(row)

        df = pd.DataFrame(rows)

    elif isinstance(x, list):
        df = pd.DataFrame(x)

    else:
        return None

    df.columns = [
        str(c).strip().capitalize()
        for c in df.columns
    ]

    needed = ["Date","Open","High","Low","Close"]

    if not all(c in df.columns for c in needed):
        return None

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    for c in needed[1:]:
        df[c] = pd.to_numeric(
            df[c], errors="coerce"
        )

    df = df.dropna(
        subset=needed
    ).sort_values("Date")

    df = df.drop_duplicates(
        "Date", keep="last"
    ).reset_index(drop=True)

    return df


# ==============================================================
# PREPARE
# ==============================================================

def prepare(df):

    df = df.copy()

    df["RSI"] = rsi(
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

    df["EMA80"] = df["Close"].ewm(
        span=EMA_LONG,
        adjust=False
    ).mean()

    df["ATR"] = atr(
        df,
        ATR_PERIOD
    )

    # ----------------------------------------------------------
    # STRONG UPTREND
    # ----------------------------------------------------------

    df["UPTREND"] = (
        (df["EMA20"] > df["EMA40"] * (1 + MIN_EMA_GAP))
        &
        (df["EMA40"] > df["EMA80"])
        &
        (df["EMA20"] > df["EMA20"].shift(1))
        &
        (df["EMA40"] > df["EMA40"].shift(1))
    )

    return df


# ==============================================================
# TRADE PROFIT
# ==============================================================

def weighted_profit(sales):

    return round(
        sum(
            s["weight"] * s["profit_pct"]
            for s in sales
        ),
        2
    )


# ==============================================================
# BACKTEST ONE SYMBOL
# ==============================================================

def backtest(symbol, df):

    trades = []
    pos = None

    for i in range(len(df)):

        r = df.iloc[i]

        date = r["Date"].strftime("%Y-%m-%d")
        close = float(r["Close"])
        low = float(r["Low"])
        high = float(r["High"])
        rv = float(r["RSI"]) if not pd.isna(r["RSI"]) else np.nan
        atrv = float(r["ATR"]) if not pd.isna(r["ATR"]) else np.nan

        if pd.isna(rv) or pd.isna(atrv):
            continue

        up = bool(r["UPTREND"])

        # ======================================================
        # NO POSITION
        # ======================================================

        if pos is None:

            # First entry:
            # Strong uptrend + RSI pullback
            if up and rv <= RSI_ENTRY:

                pos = {
                    "symbol": symbol,
                    "entry_date": date,
                    "entry_price": close,
                    "avg_price": close,
                    "weight": FIRST_ENTRY,
                    "first_entry_price": close,
                    "second_entry": False,
                    "sales": [],
                    "highest_price": close
                }

                continue

        # ======================================================
        # EXISTING POSITION
        # ======================================================

        pos["highest_price"] = max(
            pos["highest_price"],
            high
        )

        avg = pos["avg_price"]

        # ------------------------------------------------------
        # SECOND ENTRY
        # ------------------------------------------------------

        if (
            not pos["second_entry"]
            and up
            and rv <= RSI_ADD
        ):

            old_weight = pos["weight"]
            new_weight = SECOND_ENTRY

            second_price = close

            pos["avg_price"] = (
                (avg * old_weight) +
                (second_price * new_weight)
            ) / (
                old_weight + new_weight
            )

            pos["weight"] += new_weight
            pos["second_entry"] = True

            continue

        # ------------------------------------------------------
        # DYNAMIC STOP
        # ------------------------------------------------------

        atr_stop = avg - (
            atrv * ATR_STOP_MULT
        )

        fixed_stop = avg * (
            1 - MAX_STOP_PERCENT / 100
        )

        stop_price = max(
            atr_stop,
            fixed_stop
        )

        # Stop only after position is established
        if low <= stop_price:

            profit = (
                (close - avg) / avg
            ) * 100

            pos["sales"].append({
                "date": date,
                "price": close,
                "weight": pos["weight"],
                "profit_pct": round(profit,2),
                "reason": "ATR_STOP"
            })

            pos["exit_date"] = date
            pos["exit_price"] = close
            pos["exit_reason"] = "ATR_STOP"
            pos["status"] = "CLOSED"
            pos["profit_pct"] = weighted_profit(
                pos["sales"]
            )

            trades.append(pos)
            pos = None
            continue

        # ------------------------------------------------------
        # FIRST PARTIAL SELL
        # ------------------------------------------------------

        if (
            len(pos["sales"]) == 0
            and rv >= RSI_SELL_1
        ):

            sell_weight = min(
                0.50,
                pos["weight"]
            )

            profit = (
                (close - avg) / avg
            ) * 100

            pos["sales"].append({
                "date": date,
                "price": close,
                "weight": sell_weight,
                "profit_pct": round(profit,2),
                "reason": "RSI_PARTIAL"
            })

            pos["weight"] -= sell_weight

            # Move remaining position protection
            pos["breakeven"] = True

            continue

        # ------------------------------------------------------
        # BREAK EVEN AFTER FIRST SELL
        # ------------------------------------------------------

        if (
            pos.get("breakeven", False)
            and pos["weight"] > 0
            and close < avg
        ):

            profit = (
                (close - avg) / avg
            ) * 100

            pos["sales"].append({
                "date": date,
                "price": close,
                "weight": pos["weight"],
                "profit_pct": round(profit,2),
                "reason": "BREAK_EVEN"
            })

            pos["weight"] = 0

            pos["exit_date"] = date
            pos["exit_price"] = close
            pos["exit_reason"] = "BREAK_EVEN"
            pos["status"] = "CLOSED"
            pos["profit_pct"] = weighted_profit(
                pos["sales"]
            )

            trades.append(pos)
            pos = None
            continue

        # ------------------------------------------------------
        # FINAL SELL
        # ------------------------------------------------------

        if (
            pos is not None
            and pos["weight"] > 0
            and rv >= RSI_SELL_2
        ):

            profit = (
                (close - avg) / avg
            ) * 100

            pos["sales"].append({
                "date": date,
                "price": close,
                "weight": pos["weight"],
                "profit_pct": round(profit,2),
                "reason": "RSI_FINAL"
            })

            pos["weight"] = 0

            pos["exit_date"] = date
            pos["exit_price"] = close
            pos["exit_reason"] = "RSI_FINAL"
            pos["status"] = "CLOSED"
            pos["profit_pct"] = weighted_profit(
                pos["sales"]
            )

            trades.append(pos)
            pos = None

    # ==========================================================
    # OPEN POSITION
    # ==========================================================

    if pos is not None:

        last = df.iloc[-1]
        last_price = float(last["Close"])

        pos["status"] = "OPEN"
        pos["last_price"] = last_price
        pos["unrealized_pct"] = round(
            (
                (last_price - pos["avg_price"])
                / pos["avg_price"]
            ) * 100,
            2
        )

        trades.append(pos)

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

    df = to_df(data)

    if df is None or len(df) < MIN_BARS:
        continue

    df = prepare(df)

    result = backtest(
        symbol,
        df
    )

    all_trades.extend(result)

    closed = sum(
        1 for t in result
        if t.get("status") == "CLOSED"
    )

    print(
        f"{symbol:8} | "
        f"Trades: {closed:3}"
    )


# ==============================================================
# STATISTICS
# ==============================================================

all_trades.sort(
    key=lambda x: x["entry_date"]
)

closed = [
    t for t in all_trades
    if t.get("status") == "CLOSED"
]

open_pos = [
    t for t in all_trades
    if t.get("status") == "OPEN"
]

profits = [
    t["profit_pct"]
    for t in closed
]

wins = [
    p for p in profits
    if p > 0
]

losses = [
    p for p in profits
    if p <= 0
]

total = len(profits)

win_rate = (
    len(wins) / total * 100
    if total else 0
)

sum_profit = sum(profits)

avg_win = (
    np.mean(wins)
    if wins else 0
)

avg_loss = (
    np.mean(losses)
    if losses else 0
)


# ==============================================================
# COMPOUND RETURN
# ==============================================================

equity = INITIAL_CAPITAL
curve = []

for t in closed:

    equity *= (
        1 + t["profit_pct"] / 100
    )

    curve.append(equity)

compound = (
    (equity / INITIAL_CAPITAL) - 1
) * 100


# ==============================================================
# DRAWDOWN
# ==============================================================

peak = INITIAL_CAPITAL
max_dd = 0

for value in curve:

    peak = max(
        peak,
        value
    )

    dd = (
        (peak - value)
        / peak
    ) * 100

    max_dd = max(
        max_dd,
        dd
    )


# ==============================================================
# EXIT ANALYSIS
# ==============================================================

exit_counts = {}

for t in closed:

    reason = t.get(
        "exit_reason",
        "UNKNOWN"
    )

    exit_counts[reason] = (
        exit_counts.get(reason, 0) + 1
    )


# ==============================================================
# BEST / WORST
# ==============================================================

best = (
    max(closed, key=lambda x: x["profit_pct"])
    if closed else None
)

worst = (
    min(closed, key=lambda x: x["profit_pct"])
    if closed else None
)


# ==============================================================
# SAVE RESULTS
# ==============================================================

result = {

    "strategy":
        "Weekly Smart Pullback v3.0",

    "parameters": {

        "rsi_period": RSI_PERIOD,

        "rsi_entry": RSI_ENTRY,

        "rsi_add": RSI_ADD,

        "rsi_sell_1": RSI_SELL_1,

        "rsi_sell_2": RSI_SELL_2,

        "ema_fast": EMA_FAST,

        "ema_slow": EMA_SLOW,

        "ema_long": EMA_LONG,

        "min_ema_gap": MIN_EMA_GAP,

        "atr_period": ATR_PERIOD,

        "atr_stop_multiplier": ATR_STOP_MULT,

        "max_stop_percent":
            MAX_STOP_PERCENT

    },

    "statistics": {

        "total_trades": total,

        "winning_trades": len(wins),

        "losing_trades": len(losses),

        "win_rate_percent":
            round(win_rate,2),

        "sum_trade_profit_percent":
            round(sum_profit,2),

        "compound_return_percent":
            round(compound,2),

        "average_win_percent":
            round(avg_win,2),

        "average_loss_percent":
            round(avg_loss,2),

        "maximum_drawdown_percent":
            round(max_dd,2),

        "open_positions":
            len(open_pos)

    },

    "exit_analysis":
        exit_counts,

    "best_trade":
        best,

    "worst_trade":
        worst

}

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
# FINAL PRINT
# ==============================================================

print("\n")
print("="*68)
print("FINAL BACKTEST RESULTS")
print("="*68)

print(
    f"Total Trades              : {total}"
)

print(
    f"Winning Trades            : {len(wins)}"
)

print(
    f"Losing Trades             : {len(losses)}"
)

print(
    f"Win Rate                  : {win_rate:.2f}%"
)

print(
    f"Sum Trade Profit          : {sum_profit:.2f}%"
)

print(
    f"Compound Return           : {compound:.2f}%"
)

print(
    f"Average Win               : {avg_win:.2f}%"
)

print(
    f"Average Loss              : {avg_loss:.2f}%"
)

print(
    f"Maximum Drawdown          : {max_dd:.2f}%"
)

print(
    f"Open Positions            : {len(open_pos)}"
)

print("\nEXIT ANALYSIS")
print("-"*68)

for reason, count in exit_counts.items():
    print(
        f"{reason:25} : {count}"
    )

if best:
    print("\nBEST TRADE")
    print(
        f"{best['symbol']} | "
        f"{best['profit_pct']:.2f}% | "
        f"{best['entry_date']} -> "
        f"{best['exit_date']}"
    )

if worst:
    print("\nWORST TRADE")
    print(
        f"{worst['symbol']} | "
        f"{worst['profit_pct']:.2f}% | "
        f"{worst['entry_date']} -> "
        f"{worst['exit_date']}"
    )

print("\nFiles saved:")
print(f"  {RESULT_FILE}")
print(f"  {TRADES_FILE}")

print("="*68)
print("BACKTEST COMPLETE")
print("="*68)

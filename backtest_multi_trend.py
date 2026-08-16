import json
import os
import pandas as pd
import numpy as np

# ============================================================
# Backtest - Multi Trend Strategy
# ============================================================

DB_FILE = "egx_history_database_v2.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

SIDE_CLOSE_PERCENT = 0.03
RSI_SELL = 79
EGX30_KEY = "EGX30"


# ============================================================
# RSI - Wilder
# ============================================================

def rsi(series, period=14):
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        com=period - 1,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        com=period - 1,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ============================================================
# Load Database
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(
        f"❌ Database file not found: {DB_FILE}"
    )

with open(DB_FILE, "r", encoding="utf-8") as f:
    raw_database = json.load(f)

print("✅ Historical database loaded.")

symbols = list(raw_database.keys())

if EGX30_KEY not in raw_database:
    raise ValueError("❌ EGX30 not found in database.")


# ============================================================
# Prepare Data
# ============================================================

prepared_data = {}

for symbol in symbols:

    content = raw_database.get(symbol, {})

    if "data" not in content or "columns" not in content:
        continue

    try:
        df = pd.DataFrame.from_dict(
            content["data"],
            orient="index",
            columns=content["columns"]
        )

        df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        if len(df) < 40:
            continue

        # -------------------------
        # Technical Indicators
        # -------------------------

        df["EMA20"] = df["Close"].ewm(
            span=20,
            adjust=False
        ).mean()

        df["EMA30"] = df["Close"].ewm(
            span=30,
            adjust=False
        ).mean()

        df["EMA40"] = df["Close"].ewm(
            span=40,
            adjust=False
        ).mean()

        df["EMA8"] = df["Close"].ewm(
            span=8,
            adjust=False
        ).mean()

        df["EMA12"] = df["Close"].ewm(
            span=12,
            adjust=False
        ).mean()

        df["EMA70"] = df["Close"].ewm(
            span=70,
            adjust=False
        ).mean()

        df["RSI14"] = rsi(
            df["Close"],
            14
        )

        # Cross with EMA40
        df["crossed"] = (
            (
                (df["Close"] > df["EMA40"]) &
                (df["Close"].shift(1) <= df["EMA40"])
            )
            |
            (
                (df["Close"] < df["EMA40"]) &
                (df["Close"].shift(1) >= df["EMA40"])
            )
        )

        prepared_data[symbol] = df

    except Exception as e:

        print(
            f"⚠️ Failed preparing {symbol}: {e}"
        )


print(
    f"📊 Prepared {len(prepared_data)} symbols."
)


# ============================================================
# Determine EGX30 Trend
# ============================================================

def get_egx_trend(df):

    if len(df) < 30:
        return "🔛"

    last = df.iloc[-1]

    prev_5 = (
        df.iloc[-14]
        if len(df) > 14
        else df.iloc[-2]
    )

    cross_count = (
        df["crossed"]
        .iloc[-30:]
        .sum()
    )

    if cross_count >= 7:

        return "🔛"

    elif (
        last["Close"] < last["EMA20"]
        and
        last["EMA20"] < prev_5["EMA20"]
    ):

        return "🔻"

    elif (
        last["Close"] > last["EMA20"]
        and
        last["EMA20"] > prev_5["EMA20"]
    ):

        return "↗️"

    return "🔛"


# ============================================================
# Strategy Logic
# ============================================================

def calculate_trend(
    df,
    index,
    egx_trend
):

    row = df.iloc[index]

    prev = df.iloc[index - 1]

    prev_5 = (
        df.iloc[index - 13]
        if index > 13
        else df.iloc[index - 1]
    )

    # --------------------------------------------------------
    # Market Filter
    # --------------------------------------------------------

    if egx_trend == "🔻":

        return "🔻"

    # --------------------------------------------------------
    # Cross Count
    # --------------------------------------------------------

    start = max(0, index - 29)

    cross_count = (
        df["crossed"]
        .iloc[start:index + 1]
        .sum()
    )

    # --------------------------------------------------------
    # Sideways
    # --------------------------------------------------------

    if cross_count >= 7:

        return "🔛"

    # --------------------------------------------------------
    # Up Trend
    # --------------------------------------------------------

    elif (
        row["Close"] > row["EMA40"]
        and
        row["EMA40"] > (
            prev_5["EMA40"] * 0.985
        )
    ):

        return "↗️"

    # --------------------------------------------------------
    # Down Trend
    # --------------------------------------------------------

    elif (
        row["Close"] < row["EMA40"]
        and
        row["EMA40"] < (
            prev_5["EMA40"] * 0.985
        )
    ):

        return "🔻"

    # --------------------------------------------------------
    # Sideways
    # --------------------------------------------------------

    return "🔛"


# ============================================================
# Get All Trading Dates
# ============================================================

all_dates = set()

for symbol, df in prepared_data.items():

    if symbol == EGX30_KEY:
        continue

    all_dates.update(df.index)


all_dates = sorted(all_dates)

print(
    f"📅 Backtest period: "
    f"{all_dates[0].strftime('%Y-%m-%d')} "
    f"→ "
    f"{all_dates[-1].strftime('%Y-%m-%d')}"
)


# ============================================================
# State
# ============================================================

states = {}

for symbol in prepared_data:

    if symbol == EGX30_KEY:
        continue

    states[symbol] = {
        "in_position": False,
        "entry_price": None,
        "entry_date": None,
        "entry_trend": None
    }


# ============================================================
# Trades
# ============================================================

trades = []


# ============================================================
# Equity Tracking
# ============================================================

closed_profit_percent = 0.0

equity_curve = []

peak_equity = 0.0
max_drawdown = 0.0


# ============================================================
# Main Backtest Loop
# ============================================================

for current_date in all_dates:

    # --------------------------------------------------------
    # EGX30 Data Up To Current Date
    # --------------------------------------------------------

    egx_df = prepared_data.get(EGX30_KEY)

    if egx_df is None:
        continue

    egx_available = egx_df[
        egx_df.index <= current_date
    ]

    if len(egx_available) < 30:
        continue

    egx_trend = get_egx_trend(
        egx_available
    )

    # --------------------------------------------------------
    # Process Every Stock
    # --------------------------------------------------------

    for symbol, df in prepared_data.items():

        if symbol == EGX30_KEY:
            continue

        if current_date not in df.index:
            continue

        current_index = df.index.get_loc(
            current_date
        )

        if current_index < 40:
            continue

        row = df.iloc[current_index]

        close = float(row["Close"])

        if pd.isna(close):
            continue

        state = states[symbol]

        in_position = state["in_position"]

        entry_price = state["entry_price"]

        # ----------------------------------------------------
        # Calculate Trend
        # ----------------------------------------------------

        trend = calculate_trend(
            df,
            current_index,
            egx_trend
        )

        buy_signal = False
        sell_signal = False
        sell_reason = ""

        # ====================================================
        # UP TREND
        # ====================================================

        if trend == "↗️":
           if (
              not in_position
              and
              row["RSI14"] < 65
              and
              close > row["EMA30"]
              and
              row["EMA12"] > row["EMA20"]
              and
              row["EMA20"] > row["EMA40"]
              and
              close > df.iloc[current_index - 1]["Close"]
            ):
                buy_signal = True

            elif in_position:

                prev = df.iloc[current_index - 1]

                cross_down = (
                    prev["EMA12"] >= prev["EMA20"]
                    and
                    row["EMA12"] < row["EMA20"]
                )

                stop_loss = (
                    close <
                    entry_price * 0.93
                )

                rsi_sell = (
                    row["RSI14"] > RSI_SELL
                )

                if stop_loss:

                    sell_signal = True
                    sell_reason = "STOP_LOSS"

                elif cross_down:

                    sell_signal = True
                    sell_reason = "EMA_CROSS"

                elif rsi_sell:

                    sell_signal = True
                    sell_reason = "RSI"

        # ====================================================
        # SIDEWAYS
        # ====================================================

        elif trend == "🔛":

            last_40 = df.iloc[
                current_index - 39:
                current_index + 1
            ]

            high = last_40["High"].max()

            low = last_40["Low"].min()

            if high == 0 or low == 0:
                continue

            from_high = (
                (high - close) / high
            )

            from_low = (
                (close - low) / low
            )

            if (
                not in_position
                and
                (
                    from_low <= SIDE_CLOSE_PERCENT
                    or
                    row["RSI14"] < 38
                )
            ):

                buy_signal = True

            elif in_position:

                if (
                    from_high <= SIDE_CLOSE_PERCENT
                    or
                    row["RSI14"] > 66
                ):

                    sell_signal = True
                    sell_reason = "SIDE_TARGET"

                elif (
                    close <
                    entry_price * 0.93
                ):

                    sell_signal = True
                    sell_reason = "STOP_LOSS"

        # ====================================================
        # DOWN TREND
        # ====================================================

        elif trend == "🔻":

            if in_position:

                sell_signal = True
                sell_reason = "DOWN_TREND"

        # ====================================================
        # BUY
        # ====================================================

        if buy_signal and not in_position:

            state["in_position"] = True
            state["entry_price"] = close
            state["entry_date"] = (
                current_date.strftime("%Y-%m-%d")
            )
            state["entry_trend"] = trend

        # ====================================================
        # SELL
        # ====================================================

        elif sell_signal and in_position:

            profit_pct = (
                (close - entry_price)
                / entry_price
            ) * 100

            trade = {
                "symbol": symbol,
                "entry_date": state["entry_date"],
                "exit_date": current_date.strftime(
                    "%Y-%m-%d"
                ),
                "entry_price": round(
                    entry_price,
                    4
                ),
                "exit_price": round(
                    close,
                    4
                ),
                "entry_trend": state["entry_trend"],
                "exit_trend": trend,
                "profit_pct": round(
                    profit_pct,
                    2
                ),
                "exit_reason": sell_reason
            }

            trades.append(trade)

            closed_profit_percent += profit_pct

            state["in_position"] = False
            state["entry_price"] = None
            state["entry_date"] = None
            state["entry_trend"] = None

    # --------------------------------------------------------
    # Equity
    # --------------------------------------------------------

    equity_curve.append(
        closed_profit_percent
    )

    if closed_profit_percent > peak_equity:

        peak_equity = closed_profit_percent

    drawdown = (
        peak_equity -
        closed_profit_percent
    )

    if drawdown > max_drawdown:

        max_drawdown = drawdown


# ============================================================
# Close Remaining Open Positions
# ============================================================

open_positions = []

for symbol, state in states.items():

    if state["in_position"]:

        df = prepared_data[symbol]

        last_date = df.index[-1]

        last_price = float(
            df.iloc[-1]["Close"]
        )

        open_positions.append({
            "symbol": symbol,
            "entry_date": state["entry_date"],
            "entry_price": state["entry_price"],
            "last_date": last_date.strftime(
                "%Y-%m-%d"
            ),
            "last_price": last_price
        })


# ============================================================
# Statistics
# ============================================================

total_trades = len(trades)

winning_trades = [
    t for t in trades
    if t["profit_pct"] > 0
]

losing_trades = [
    t for t in trades
    if t["profit_pct"] <= 0
]

wins = len(winning_trades)
losses = len(losing_trades)

win_rate = (
    (wins / total_trades) * 100
    if total_trades > 0
    else 0
)

average_profit = (
    np.mean([
        t["profit_pct"]
        for t in winning_trades
    ])
    if winning_trades
    else 0
)

average_loss = (
    np.mean([
        t["profit_pct"]
        for t in losing_trades
    ])
    if losing_trades
    else 0
)

best_trade = (
    max(
        trades,
        key=lambda x: x["profit_pct"]
    )
    if trades
    else None
)

worst_trade = (
    min(
        trades,
        key=lambda x: x["profit_pct"]
    )
    if trades
    else None
)


# ============================================================
# Results
# ============================================================

results = {

    "backtest_period": {
        "start": all_dates[0].strftime(
            "%Y-%m-%d"
        ),
        "end": all_dates[-1].strftime(
            "%Y-%m-%d"
        )
    },

    "statistics": {

        "total_trades": total_trades,

        "winning_trades": wins,

        "losing_trades": losses,

        "win_rate_percent": round(
            win_rate,
            2
        ),

        "total_profit_percent": round(
            closed_profit_percent,
            2
        ),

        "average_winning_trade_percent": round(
            float(average_profit),
            2
        ),

        "average_losing_trade_percent": round(
            float(average_loss),
            2
        ),

        "maximum_drawdown_percent": round(
            max_drawdown,
            2
        )
    },

    "best_trade": best_trade,

    "worst_trade": worst_trade,

    "open_positions": open_positions
}


# ============================================================
# Save Results
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


with open(
    TRADES_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        trades,
        f,
        indent=2,
        ensure_ascii=False
    )

# ============================================================
# Per Stock Summary
# ============================================================

stock_summary = {}

for trade in trades:

    symbol = trade["symbol"]
    profit = trade["profit_pct"]

    if symbol not in stock_summary:
        stock_summary[symbol] = {
            "symbol": symbol,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_percent": 0,
            "total_profit_percent": 0,
            "average_profit_percent": 0,
            "average_loss_percent": 0,
            "best_trade_percent": None,
            "worst_trade_percent": None
        }

    stock_summary[symbol]["total_trades"] += 1

    stock_summary[symbol]["total_profit_percent"] += profit

    if profit > 0:
        stock_summary[symbol]["winning_trades"] += 1
    else:
        stock_summary[symbol]["losing_trades"] += 1


# ============================================================
# Calculate Stock Statistics
# ============================================================

for symbol, summary in stock_summary.items():

    total = summary["total_trades"]
    wins = summary["winning_trades"]
    losses = summary["losing_trades"]

    symbol_trades = [
        t for t in trades
        if t["symbol"] == symbol
    ]

    winning_profits = [
        t["profit_pct"]
        for t in symbol_trades
        if t["profit_pct"] > 0
    ]

    losing_profits = [
        t["profit_pct"]
        for t in symbol_trades
        if t["profit_pct"] <= 0
    ]

    summary["win_rate_percent"] = round(
        (wins / total) * 100
        if total > 0
        else 0,
        2
    )

    summary["total_profit_percent"] = round(
        summary["total_profit_percent"],
        2
    )

    summary["average_profit_percent"] = round(
        float(np.mean(winning_profits))
        if winning_profits
        else 0,
        2
    )

    summary["average_loss_percent"] = round(
        float(np.mean(losing_profits))
        if losing_profits
        else 0,
        2
    )

    if symbol_trades:

        summary["best_trade_percent"] = round(
            max(
                t["profit_pct"]
                for t in symbol_trades
            ),
            2
        )

        summary["worst_trade_percent"] = round(
            min(
                t["profit_pct"]
                for t in symbol_trades
            ),
            2
        )


# ============================================================
# Sort Stocks By Total Profit
# ============================================================

stock_summary_list = list(
    stock_summary.values()
)

stock_summary_list.sort(
    key=lambda x: x["total_profit_percent"],
    reverse=True
)


# ============================================================
# Save Per Stock Summary
# ============================================================

STOCK_SUMMARY_FILE = "backtest_summary_by_stock.json"

with open(
    STOCK_SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        stock_summary_list,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"📊 Stock summary saved to: "
    f"{STOCK_SUMMARY_FILE}"
)


# ============================================================
# Print Top / Worst Stocks
# ============================================================

print()
print("=" * 60)
print("🏆 TOP 10 STOCKS")
print("=" * 60)

for stock in stock_summary_list[:10]:

    print(
        f"{stock['symbol']} | "
        f"Trades: {stock['total_trades']} | "
        f"Win Rate: {stock['win_rate_percent']:.2f}% | "
        f"Result: {stock['total_profit_percent']:.2f}%"
    )


print()
print("=" * 60)
print("💥 WORST 10 STOCKS")
print("=" * 60)

for stock in stock_summary_list[-10:]:

    print(
        f"{stock['symbol']} | "
        f"Trades: {stock['total_trades']} | "
        f"Win Rate: {stock['win_rate_percent']:.2f}% | "
        f"Result: {stock['total_profit_percent']:.2f}%"
    )
# ============================================================
# Console Report
# ============================================================

print()
print("=" * 60)
print("📊 MULTI-TREND BACKTEST RESULTS")
print("=" * 60)

print(
    f"📅 Period: "
    f"{results['backtest_period']['start']} "
    f"→ "
    f"{results['backtest_period']['end']}"
)

print(
    f"📈 Total Trades: {total_trades}"
)

print(
    f"🟢 Winning Trades: {wins}"
)

print(
    f"🔴 Losing Trades: {losses}"
)

print(
    f"🎯 Win Rate: {win_rate:.2f}%"
)

print(
    f"💰 Total Profit: "
    f"{closed_profit_percent:.2f}%"
)

print(
    f"📊 Average Win: "
    f"{average_profit:.2f}%"
)

print(
    f"📉 Average Loss: "
    f"{average_loss:.2f}%"
)

print(
    f"⚠️ Maximum Drawdown: "
    f"{max_drawdown:.2f}%"
)

if best_trade:

    print(
        f"🏆 Best Trade: "
        f"{best_trade['symbol']} "
        f"{best_trade['profit_pct']:.2f}%"
    )

if worst_trade:

    print(
        f"💥 Worst Trade: "
        f"{worst_trade['symbol']} "
        f"{worst_trade['profit_pct']:.2f}%"
    )

print(
    f"📂 Results saved to: {RESULTS_FILE}"
)

print(
    f"📂 Trades saved to: {TRADES_FILE}"
)

print("=" * 60)
print("🏁 Backtest Complete.")
print("=" * 60)

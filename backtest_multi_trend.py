import json
import os
import pandas as pd
import numpy as np

# ============================================================
# Backtest - Multi Trend Strategy
# Long Term EMA70 Version
# ============================================================

DB_FILE = "egx_history_database_v2.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"
STOCK_SUMMARY_FILE = "backtest_summary_by_stock.json"

# ============================================================
# Strategy Parameters
# ============================================================

# EMA70 clear uptrend:
# EMA70 current > EMA70 - 4 > EMA70 - 8 > EMA70 - 12
EMA70_UP_MIN_STEP_PERCENT = 0.30

# EMA70 sideways:
# Maximum distance between EMA70 levels
EMA70_SIDE_MAX_DISTANCE_PERCENT = 1.00

# EMA70 strong downtrend:
# EMA70 current must be lower than previous levels
# with a very clear difference
EMA70_DOWN_MIN_STEP_PERCENT = 1.00

# Up Trend
RSI_UP_BUY = 48
RSI_UP_SELL = 77

# Sideways
RSI_SIDE_BUY = 28
RSI_SIDE_SELL = 65

# Safety stop
STOP_LOSS_PERCENT = 7.0

# Market index is only monitored.
# It DOES NOT control the strategy.
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

        # EMA70 + safety history
        if len(df) < 80:
            continue

        # ----------------------------------------------------
        # Technical Indicators
        # ----------------------------------------------------

        df["EMA12"] = df["Close"].ewm(
            span=12,
            adjust=False
        ).mean()

        df["EMA20"] = df["Close"].ewm(
            span=20,
            adjust=False
        ).mean()

        df["EMA70"] = df["Close"].ewm(
            span=70,
            adjust=False
        ).mean()

        df["RSI14"] = rsi(df["Close"], 14)

        # ----------------------------------------------------
        # EMA12 / EMA20 Cross
        # Kept for analysis / future testing.
        # NOT used by current entry/exit logic.
        # ----------------------------------------------------

        df["cross_up"] = (
            (df["EMA12"] > df["EMA20"]) &
            (df["EMA12"].shift(1) <= df["EMA20"].shift(1))
        )

        df["cross_down"] = (
            (df["EMA12"] < df["EMA20"]) &
            (df["EMA12"].shift(1) >= df["EMA20"].shift(1))
        )

        prepared_data[symbol] = df

    except Exception as e:
        print(f"⚠️ Failed preparing {symbol}: {e}")

print(f"📊 Prepared {len(prepared_data)} symbols.")


# ============================================================
# EMA70 Trend Detection
# ============================================================

def calculate_trend(df, index):

    if index < 12:
        return "🔛"

    ema70_now = float(df.iloc[index]["EMA70"])
    ema70_4 = float(df.iloc[index - 4]["EMA70"])
    ema70_8 = float(df.iloc[index - 8]["EMA70"])
    ema70_12 = float(df.iloc[index - 12]["EMA70"])

    if (
        pd.isna(ema70_now) or
        pd.isna(ema70_4) or
        pd.isna(ema70_8) or
        pd.isna(ema70_12)
    ):
        return "🔛"

    # --------------------------------------------------------
    # Percentage change between EMA70 levels
    # --------------------------------------------------------

    step_1 = (
        (ema70_now - ema70_4) /
        ema70_4
    ) * 100

    step_2 = (
        (ema70_4 - ema70_8) /
        ema70_8
    ) * 100

    step_3 = (
        (ema70_8 - ema70_12) /
        ema70_12
    ) * 100

    # --------------------------------------------------------
    # STRONG UP TREND
    #
    # EMA70 now
    # >
    # EMA70 -4
    # >
    # EMA70 -8
    # >
    # EMA70 -12
    #
    # Every step must have clear positive movement.
    # --------------------------------------------------------

    if (
        step_1 >= EMA70_UP_MIN_STEP_PERCENT and
        step_2 >= EMA70_UP_MIN_STEP_PERCENT and
        step_3 >= EMA70_UP_MIN_STEP_PERCENT
    ):
        return "↗️"

    # --------------------------------------------------------
    # STRONG DOWN TREND
    #
    # EMA70 now
    # <
    # EMA70 -4
    # <
    # EMA70 -8
    # <
    # EMA70 -12
    #
    # Every step must have very clear negative movement.
    # --------------------------------------------------------

    if (
        step_1 <= -EMA70_DOWN_MIN_STEP_PERCENT and
        step_2 <= -EMA70_DOWN_MIN_STEP_PERCENT and
        step_3 <= -EMA70_DOWN_MIN_STEP_PERCENT
    ):
        return "🔻"

    # --------------------------------------------------------
    # SIDEWAYS / FLAT
    #
    # EMA70 levels are relatively close together.
    # --------------------------------------------------------

    max_ema70 = max(
        ema70_now,
        ema70_4,
        ema70_8,
        ema70_12
    )

    min_ema70 = min(
        ema70_now,
        ema70_4,
        ema70_8,
        ema70_12
    )

    if min_ema70 <= 0:
        return "🔛"

    distance_percent = (
        (max_ema70 - min_ema70) /
        min_ema70
    ) * 100

    if distance_percent <= EMA70_SIDE_MAX_DISTANCE_PERCENT:
        return "🔛"

    # Everything else = sideways for safety
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

if not all_dates:
    raise ValueError("❌ No trading dates available.")

print(
    f"📅 Backtest period: "
    f"{all_dates[0].strftime('%Y-%m-%d')} → "
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
# Trend Statistics
# ============================================================

trend_days = {
    "↗️": 0,
    "🔛": 0,
    "🔻": 0
}


# ============================================================
# Main Backtest Loop
# ============================================================

for current_date in all_dates:

    # ========================================================
    # EGX30
    #
    # IMPORTANT:
    # EGX30 DOES NOT CONTROL THE STRATEGY.
    # It is completely ignored here.
    # ========================================================

    # ========================================================
    # Process Every Stock
    # ========================================================

    for symbol, df in prepared_data.items():

        if symbol == EGX30_KEY:
            continue

        if current_date not in df.index:
            continue

        current_index = df.index.get_loc(current_date)

        # Safety history
        if current_index < 80:
            continue

        row = df.iloc[current_index]

        close = float(row["Close"])

        if pd.isna(close):
            continue

        if pd.isna(row["RSI14"]):
            continue

        state = states[symbol]

        in_position = state["in_position"]
        entry_price = state["entry_price"]

        # ====================================================
        # Calculate EMA70 Trend
        # ====================================================

        trend = calculate_trend(
            df,
            current_index
        )

        trend_days[trend] += 1

        buy_signal = False
        sell_signal = False
        sell_reason = ""

        # ====================================================
        # UP TREND
        #
        # EMA70 clearly rising
        #
        # Entry:
        # EMA70 clearly rising
        # +
        # RSI14 < 48
        #
        # Exit:
        # RSI14 > 77
        #
        # Stop Loss:
        # -7%
        # ====================================================

        if trend == "↗️":

            # ------------------------------------------------
            # BUY
            # ------------------------------------------------

            if (
                not in_position and
                row["RSI14"] < RSI_UP_BUY
            ):
                buy_signal = True

            # ------------------------------------------------
            # SELL
            # ------------------------------------------------

            elif in_position:

                rsi_sell = (
                    row["RSI14"] > RSI_UP_SELL
                )

                stop_loss = (
                    close <
                    entry_price * (
                        1 -
                        STOP_LOSS_PERCENT / 100
                    )
                )

                if rsi_sell:
                    sell_signal = True
                    sell_reason = "RSI_77"

                elif stop_loss:
                    sell_signal = True
                    sell_reason = "STOP_LOSS"

        # ====================================================
        # SIDEWAYS
        #
        # IMPORTANT:
        # Sideways strategy remains unchanged.
        #
        # Buy:
        # RSI < 28
        #
        # Sell:
        # RSI > 65
        # ====================================================

        elif trend == "🔛":

            if (
                not in_position and
                row["RSI14"] < RSI_SIDE_BUY
            ):
                buy_signal = True

            elif in_position:

                if row["RSI14"] > RSI_SIDE_SELL:

                    sell_signal = True
                    sell_reason = "SIDE_RSI_TARGET"

                elif (
                    close <
                    entry_price * (
                        1 -
                        STOP_LOSS_PERCENT / 100
                    )
                ):

                    sell_signal = True
                    sell_reason = "STOP_LOSS"

        # ====================================================
        # STRONG DOWN TREND
        #
        # No new trades.
        # Close any open position.
        # ====================================================

        elif trend == "🔻":

            if in_position:

                sell_signal = True
                sell_reason = "EMA70_STRONG_DOWN"

        # ====================================================
        # BUY
        # ====================================================

        if buy_signal and not in_position:

            state["in_position"] = True
            state["entry_price"] = close
            state["entry_date"] = current_date.strftime(
                "%Y-%m-%d"
            )
            state["entry_trend"] = trend

        # ====================================================
        # SELL
        # ====================================================

        elif sell_signal and in_position:

            profit_pct = (
                (close - entry_price) /
                entry_price
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

    # ========================================================
    # Equity
    # ========================================================

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
# Open Positions
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

    "strategy": {
        "primary_indicator": "EMA70",

        "ema70_levels": [
            "current",
            "4_candles_ago",
            "8_candles_ago",
            "12_candles_ago"
        ],

        "uptrend_rule": (
            "EMA70 current > EMA70-4 > "
            "EMA70-8 > EMA70-12 "
            "with clear positive steps"
        ),

        "sideways_rule": (
            "EMA70 levels are close"
        ),

        "downtrend_rule": (
            "EMA70 levels decrease "
            "with very clear negative steps"
        ),

        "up_entry": (
            "EMA70 clearly rising + RSI14 < 48"
        ),

        "up_exit": (
            "RSI14 > 77"
        ),

        "side_entry": (
            "RSI14 < 28"
        ),

        "side_exit": (
            "RSI14 > 65"
        ),

        "down_action": (
            "No buy + close open positions"
        ),

        "stop_loss_percent": STOP_LOSS_PERCENT,

        "egx30_market_filter": "DISABLED"
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

    "trend_days": trend_days,
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

    stock_summary[symbol][
        "total_profit_percent"
    ] += profit

    if profit > 0:

        stock_summary[symbol][
            "winning_trades"
        ] += 1

    else:

        stock_summary[symbol][
            "losing_trades"
        ] += 1


# ============================================================
# Calculate Stock Statistics
# ============================================================

for symbol, summary in stock_summary.items():

    total = summary["total_trades"]
    wins = summary["winning_trades"]

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
print("📊 MULTI-TREND EMA70 LONG-TERM BACKTEST RESULTS")
print("=" * 60)

print(
    f"📅 Period: "
    f"{results['backtest_period']['start']} → "
    f"{results['backtest_period']['end']}"
)

print(
    f"📈 Total Trades: "
    f"{total_trades}"
)

print(
    f"🟢 Winning Trades: "
    f"{wins}"
)

print(
    f"🔴 Losing Trades: "
    f"{losses}"
)

print(
    f"🎯 Win Rate: "
    f"{win_rate:.2f}%"
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

print()
print("📊 EMA70 Trend Distribution:")

print(
    f"↗️ Up Trend Days: "
    f"{trend_days['↗️']}"
)

print(
    f"🔛 Sideways Days: "
    f"{trend_days['🔛']}"
)

print(
    f"🔻 Down Trend Days: "
    f"{trend_days['🔻']}"
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
    f"📂 Results saved to: "
    f"{RESULTS_FILE}"
)

print(
    f"📂 Trades saved to: "
    f"{TRADES_FILE}"
)

print("=" * 60)
print("🏁 Backtest Complete.")
print("=" * 60)

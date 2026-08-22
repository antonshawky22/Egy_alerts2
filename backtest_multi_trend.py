import json
import os
import pandas as pd
import numpy as np

# ============================================================
# WEEKLY RSI STRATEGY - 33 / 60 / 70
# ============================================================

DB_FILE = "egx_weekly_database_v1.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

RSI_PERIOD = 14
RSI_BUY = 33
RSI_SELL_1 = 60
RSI_SELL_2 = 70

FIRST_SELL_PERCENT = 0.50
SECOND_SELL_PERCENT = 0.50

HARD_STOP_PERCENT = 5.0

symbols = {
    "OLFI":"OLFI","EMFD":"EMFD","ETEL":"ETEL","EAST":"EAST",
    "EFIH":"EFIH","ABUK":"ABUK","OIH":"OIH","SWDY":"SWDY",
    "ISPH":"ISPH","ATQA":"ATQA","MTIE":"MTIE","HRHO":"HRHO",
    "ORWE":"ORWE","JUFO":"JUFO","DSCW":"DSCW","SUGR":"SUGR",
    "ELSH":"ELSH","RMDA":"RMDA","RAYA":"RAYA","EEII":"EEII",
    "MPCO":"MPCO","GBCO":"GBCO","TMGH":"TMGH","ORHD":"ORHD",
    "AMOC":"AMOC","FWRY":"FWRY","COMI":"COMI","ADIB":"ADIB",
    "PHDC":"PHDC","MCQE":"MCQE","SKPC":"SKPC","EGAL":"EGAL"
}

# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(f"Database file not found: {DB_FILE}")

with open(DB_FILE, "r", encoding="utf-8") as f:
    raw_database = json.load(f)

prepared_data = {}

# ============================================================
# RSI FUNCTION
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi

# ============================================================
# PREPARE DATA
# ============================================================

for name in symbols:
    if name not in raw_database:
        continue

    content = raw_database[name]

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

        if len(df) < RSI_PERIOD + 5:
            continue

        df["Close"] = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        df = df.dropna(subset=["Close"])

        # RSI14
        df["RSI14"] = calculate_rsi(
            df["Close"],
            RSI_PERIOD
        )

        # EMA20 فقط لاستخدام اتجاهه
        df["EMA20"] = df["Close"].ewm(
            span=20,
            adjust=False
        ).mean()

        prepared_data[name] = df

    except Exception as e:
        print(f"Skipping {name}: {e}")

if not prepared_data:
    raise RuntimeError("No valid stock data found.")

all_dates = sorted(
    set().union(
        *[df.index for df in prepared_data.values()]
    )
)

# ============================================================
# STATES
# ============================================================

states = {
    name: {
        "position": 0.0,
        "entry_price": 0.0,
        "entry_date": None,
        "sales": []
    }
    for name in prepared_data
}

trades_history = []

# Portfolio accounting
portfolio_value = 100.0
portfolio_peak = 100.0
max_drawdown = 0.0

# ============================================================
# BACKTEST
# ============================================================

for current_date in all_dates:

    for name, df in prepared_data.items():

        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)

        if idx < RSI_PERIOD + 2:
            continue

        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]

        price = float(row["Close"])
        rsi = float(row["RSI14"])
        ema20 = float(row["EMA20"])
        ema20_prev = float(prev_row["EMA20"])

        if np.isnan(rsi) or np.isnan(ema20):
            continue

        date_str = current_date.strftime("%Y-%m-%d")

        s = states[name]

        # ====================================================
        # ENTRY
        # ====================================================

        if s["position"] == 0.0:

            ema20_up = ema20 > ema20_prev

            buy_signal = (
                rsi < RSI_BUY
                and ema20_up
            )

            if buy_signal:

                s["position"] = 1.0
                s["entry_price"] = price
                s["entry_date"] = date_str
                s["sales"] = []

                trades_history.append({
                    "symbol": name,
                    "status": "OPEN",
                    "entry_date": date_str,
                    "entry_price": round(price, 2),
                    "position": 1.0,
                    "sales": [],
                    "exit_date": None,
                    "exit_price": None,
                    "profit_pct": None,
                    "exit_reason": None
                })

            continue

        # ====================================================
        # CURRENT PROFIT
        # ====================================================

        entry_price = s["entry_price"]

        profit = (
            (price - entry_price)
            / entry_price
        ) * 100

        # ====================================================
        # HARD STOP
        # ====================================================

        if profit <= -HARD_STOP_PERCENT:

            sold_position = s["position"]

            trade_profit = profit * sold_position

            s["sales"].append({
                "date": date_str,
                "price": round(price, 2),
                "position_sold": round(sold_position, 2),
                "profit_pct": round(profit, 2),
                "weighted_profit_pct": round(
                    trade_profit, 2
                ),
                "reason": "HARD_STOP"
            })

            active = [
                t for t in trades_history
                if t["symbol"] == name
                and t["status"] == "OPEN"
            ]

            if active:

                trade = active[-1]

                trade["status"] = "CLOSED"
                trade["sales"] = s["sales"]
                trade["exit_date"] = date_str
                trade["exit_price"] = round(price, 2)

                total_profit = sum(
                    sale["weighted_profit_pct"]
                    for sale in s["sales"]
                )

                trade["profit_pct"] = round(
                    total_profit,
                    2
                )

                trade["exit_reason"] = "HARD_STOP"

                portfolio_value *= (
                    1 + total_profit / 100
                )

            s["position"] = 0.0
            s["entry_price"] = 0.0
            s["entry_date"] = None
            s["sales"] = []

            continue

        # ====================================================
        # FIRST SELL - RSI 60
        # ====================================================

        if (
            s["position"] == 1.0
            and rsi >= RSI_SELL_1
        ):

            sold = FIRST_SELL_PERCENT

            sale_profit = profit * sold

            s["sales"].append({
                "date": date_str,
                "price": round(price, 2),
                "position_sold": sold,
                "profit_pct": round(profit, 2),
                "weighted_profit_pct": round(
                    sale_profit,
                    2
                ),
                "reason": "RSI_60"
            })

            s["position"] -= sold

            active = [
                t for t in trades_history
                if t["symbol"] == name
                and t["status"] == "OPEN"
            ]

            if active:
                active[-1]["sales"] = s["sales"]

        # ====================================================
        # SECOND SELL - RSI 70
        # ====================================================

        if (
            s["position"] > 0
            and rsi >= RSI_SELL_2
        ):

            sold = s["position"]

            sale_profit = profit * sold

            s["sales"].append({
                "date": date_str,
                "price": round(price, 2),
                "position_sold": round(sold, 2),
                "profit_pct": round(profit, 2),
                "weighted_profit_pct": round(
                    sale_profit,
                    2
                ),
                "reason": "RSI_70"
            })

            s["position"] = 0.0

            active = [
                t for t in trades_history
                if t["symbol"] == name
                and t["status"] == "OPEN"
            ]

            if active:

                trade = active[-1]

                trade["status"] = "CLOSED"
                trade["sales"] = s["sales"]
                trade["exit_date"] = date_str
                trade["exit_price"] = round(price, 2)

                total_profit = sum(
                    sale["weighted_profit_pct"]
                    for sale in s["sales"]
                )

                trade["profit_pct"] = round(
                    total_profit,
                    2
                )

                trade["exit_reason"] = "RSI_70"

                portfolio_value *= (
                    1 + total_profit / 100
                )

            s["entry_price"] = 0.0
            s["entry_date"] = None
            s["sales"] = []

    # ========================================================
    # EQUITY / DRAWDOWN
    # ========================================================

    if portfolio_value > portfolio_peak:
        portfolio_peak = portfolio_value

    current_drawdown = (
        (portfolio_peak - portfolio_value)
        / portfolio_peak
    ) * 100

    if current_drawdown > max_drawdown:
        max_drawdown = current_drawdown

# ============================================================
# CLOSE OPEN POSITIONS AT LAST AVAILABLE PRICE
# ============================================================

for name, s in states.items():

    if s["position"] <= 0:
        continue

    df = prepared_data[name]

    price = float(df["Close"].iloc[-1])
    date_str = df.index[-1].strftime("%Y-%m-%d")

    profit = (
        (price - s["entry_price"])
        / s["entry_price"]
    ) * 100

    sold = s["position"]

    weighted_profit = profit * sold

    s["sales"].append({
        "date": date_str,
        "price": round(price, 2),
        "position_sold": round(sold, 2),
        "profit_pct": round(profit, 2),
        "weighted_profit_pct": round(
            weighted_profit,
            2
        ),
        "reason": "END_OF_DATA"
    })

    active = [
        t for t in trades_history
        if t["symbol"] == name
        and t["status"] == "OPEN"
    ]

    if active:

        trade = active[-1]

        trade["status"] = "CLOSED"
        trade["sales"] = s["sales"]
        trade["exit_date"] = date_str
        trade["exit_price"] = round(price, 2)

        total_profit = sum(
            sale["weighted_profit_pct"]
            for sale in s["sales"]
        )

        trade["profit_pct"] = round(
            total_profit,
            2
        )

        trade["exit_reason"] = "END_OF_DATA"

# ============================================================
# STATISTICS
# ============================================================

closed_trades = [
    t for t in trades_history
    if t["status"] == "CLOSED"
]

winning_trades = [
    t for t in closed_trades
    if t["profit_pct"] > 0
]

losing_trades = [
    t for t in closed_trades
    if t["profit_pct"] <= 0
]

total_count = len(closed_trades)
wins_count = len(winning_trades)
losses_count = len(losing_trades)

win_rate = (
    wins_count / total_count * 100
    if total_count > 0
    else 0
)

avg_win = (
    np.mean([
        t["profit_pct"]
        for t in winning_trades
    ])
    if winning_trades
    else 0
)

avg_loss = (
    np.mean([
        t["profit_pct"]
        for t in losing_trades
    ])
    if losing_trades
    else 0
)

# ============================================================
# RESULT
# ============================================================

results_summary = {
    "strategy": "Weekly RSI 33/60/70 + EMA20 Direction",
    "parameters": {
        "rsi_period": RSI_PERIOD,
        "rsi_buy": RSI_BUY,
        "rsi_sell_1": RSI_SELL_1,
        "rsi_sell_2": RSI_SELL_2,
        "first_sell_percent": 50,
        "second_sell_percent": 50,
        "ema20_direction_filter": "EMA20 > previous EMA20",
        "hard_stop_percent": HARD_STOP_PERCENT
    },
    "statistics": {
        "total_trades": total_count,
        "winning_trades": wins_count,
        "losing_trades": losses_count,
        "win_rate_percent": round(
            win_rate,
            2
        ),
        "compound_portfolio_return_percent": round(
            (portfolio_value - 100),
            2
        ),
        "average_winning_trade_percent": round(
            float(avg_win),
            2
        ),
        "average_losing_trade_percent": round(
            float(avg_loss),
            2
        ),
        "maximum_drawdown_percent": round(
            max_drawdown,
            2
        )
    }
}

# ============================================================
# SAVE
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results_summary,
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
        trades_history,
        f,
        indent=2,
        ensure_ascii=False
    )

# ============================================================
# PRINT
# ============================================================

print("=" * 50)
print("WEEKLY RSI 33/60/70 BACKTEST")
print("=" * 50)
print(f"Total Trades: {total_count}")
print(f"Winning Trades: {wins_count}")
print(f"Losing Trades: {losses_count}")
print(f"Win Rate: {win_rate:.2f}%")
print(
    f"Compound Return: "
    f"{portfolio_value - 100:.2f}%"
)
print(
    f"Average Win: "
    f"{float(avg_win):.2f}%"
)
print(
    f"Average Loss: "
    f"{float(avg_loss):.2f}%"
)
print(
    f"Max Drawdown: "
    f"{max_drawdown:.2f}%"
)
print("=" * 50)

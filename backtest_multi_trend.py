import json
import os
import pandas as pd
import numpy as np

# ============================================================
# WEEKLY RSI 33 / 60 / 70
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
    "OLFI": "OLFI",
    "EMFD": "EMFD",
    "ETEL": "ETEL",
    "EAST": "EAST",
    "EFIH": "EFIH",
    "ABUK": "ABUK",
    "OIH": "OIH",
    "SWDY": "SWDY",
    "ISPH": "ISPH",
    "ATQA": "ATQA",
    "MTIE": "MTIE",
    "HRHO": "HRHO",
    "ORWE": "ORWE",
    "JUFO": "JUFO",
    "DSCW": "DSCW",
    "SUGR": "SUGR",
    "ELSH": "ELSH",
    "RMDA": "RMDA",
    "RAYA": "RAYA",
    "EEII": "EEII",
    "MPCO": "MPCO",
    "GBCO": "GBCO",
    "TMGH": "TMGH",
    "ORHD": "ORHD",
    "AMOC": "AMOC",
    "FWRY": "FWRY",
    "COMI": "COMI",
    "ADIB": "ADIB",
    "PHDC": "PHDC",
    "MCQE": "MCQE",
    "SKPC": "SKPC",
    "EGAL": "EGAL"
}

# ============================================================
# RSI
# ============================================================

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

    return 100 - (100 / (1 + rs))

# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(
        f"Database file not found: {DB_FILE}"
    )

with open(DB_FILE, "r", encoding="utf-8") as f:
    raw_database = json.load(f)

prepared_data = {}

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

        if len(df) < 30:
            continue

        df["Close"] = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        df["RSI14"] = calculate_rsi(
            df["Close"],
            RSI_PERIOD
        )

        prepared_data[name] = df

    except Exception as e:
        print(f"Skipping {name}: {e}")

if not prepared_data:
    raise RuntimeError("No valid stock data found.")

# ============================================================
# DATES
# ============================================================

all_dates = sorted(
    set().union(
        *[df.index for df in prepared_data.values()]
    )
)

# ============================================================
# STATES
# ============================================================

states = {}

for name in prepared_data:

    states[name] = {
        "position": 0.0,
        "entry_price": 0.0,
        "entry_date": None,
        "sales": []
    }

trades_history = []

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

        if pd.isna(row["RSI14"]):
            continue

        price = float(row["Close"])
        rsi = float(row["RSI14"])

        date_str = current_date.strftime("%Y-%m-%d")

        s = states[name]

        # ====================================================
        # BUY
        # ====================================================

        if s["position"] == 0.0:

            if rsi < RSI_BUY:

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

        # ====================================================
        # MANAGE POSITION
        # ====================================================

        else:

            entry_price = s["entry_price"]

            profit = (
                (price - entry_price)
                / entry_price
            ) * 100

            # =================================================
            # HARD STOP
            # =================================================

            if profit <= -HARD_STOP_PERCENT:

                sold_position = s["position"]

                s["sales"].append({
                    "date": date_str,
                    "price": round(price, 2),
                    "position_sold": round(
                        sold_position,
                        2
                    ),
                    "profit_pct": round(
                        profit,
                        2
                    ),
                    "reason": "HARD_STOP"
                })

                active = [
                    t for t in trades_history
                    if (
                        t["symbol"] == name
                        and t["status"] == "OPEN"
                    )
                ]

                if active:

                    trade = active[-1]

                    trade["status"] = "CLOSED"
                    trade["sales"] = s["sales"]
                    trade["exit_date"] = date_str
                    trade["exit_price"] = round(
                        price,
                        2
                    )
                    trade["profit_pct"] = round(
                        profit,
                        2
                    )
                    trade["exit_reason"] = "HARD_STOP"

                s["position"] = 0.0
                s["entry_price"] = 0.0
                s["entry_date"] = None
                s["sales"] = []

                continue

            # =================================================
            # FIRST SALE - 50% AT RSI 60
            # =================================================

            if (
                s["position"] == 1.0
                and rsi >= RSI_SELL_1
            ):

                sale_profit = profit

                s["sales"].append({
                    "date": date_str,
                    "price": round(price, 2),
                    "position_sold":
                        FIRST_SELL_PERCENT,
                    "profit_pct":
                        round(sale_profit, 2),
                    "reason": "RSI_60"
                })

                s["position"] = 0.5

                active = [
                    t for t in trades_history
                    if (
                        t["symbol"] == name
                        and t["status"] == "OPEN"
                    )
                ]

                if active:
                    active[-1]["sales"] = s["sales"]

            # =================================================
            # SECOND SALE - REMAINING 50% AT RSI 70
            # =================================================

            elif (
                s["position"] == 0.5
                and rsi >= RSI_SELL_2
            ):

                sale_profit = profit

                s["sales"].append({
                    "date": date_str,
                    "price": round(price, 2),
                    "position_sold":
                        SECOND_SELL_PERCENT,
                    "profit_pct":
                        round(sale_profit, 2),
                    "reason": "RSI_70"
                })

                active = [
                    t for t in trades_history
                    if (
                        t["symbol"] == name
                        and t["status"] == "OPEN"
                    )
                ]

                if active:

                    trade = active[-1]

                    trade["status"] = "CLOSED"
                    trade["sales"] = s["sales"]
                    trade["exit_date"] = date_str
                    trade["exit_price"] = round(
                        price,
                        2
                    )

                    weighted_profit = (
                        s["sales"][0]["profit_pct"]
                        * FIRST_SELL_PERCENT
                        +
                        sale_profit
                        * SECOND_SELL_PERCENT
                    )

                    trade["profit_pct"] = round(
                        weighted_profit,
                        2
                    )

                    trade["exit_reason"] = "RSI_70"

                s["position"] = 0.0
                s["entry_price"] = 0.0
                s["entry_date"] = None
                s["sales"] = []

# ============================================================
# STATISTICS
# ============================================================

closed_trades = [
    t for t in trades_history
    if t["status"] == "CLOSED"
]

winning_trades = [
    t for t in closed_trades
    if t["profit_pct"] is not None
    and t["profit_pct"] > 0
]

losing_trades = [
    t for t in closed_trades
    if t["profit_pct"] is not None
    and t["profit_pct"] <= 0
]

total_count = len(closed_trades)
wins_count = len(winning_trades)
losses_count = len(losing_trades)

win_rate = (
    wins_count / total_count * 100
    if total_count > 0
    else 0.0
)

avg_win = (
    float(
        np.mean([
            t["profit_pct"]
            for t in winning_trades
        ])
    )
    if winning_trades
    else 0.0
)

avg_loss = (
    float(
        np.mean([
            t["profit_pct"]
            for t in losing_trades
        ])
    )
    if losing_trades
    else 0.0
)

# ============================================================
# SIMPLE TRADE EQUITY CURVE
# ============================================================

equity = 100.0
equity_curve = [equity]

for trade in closed_trades:

    profit = trade["profit_pct"]

    if profit is None:
        continue

    equity *= (
        1 + profit / 100
    )

    equity_curve.append(equity)

if equity_curve:

    equity_array = np.array(
        equity_curve,
        dtype=float
    )

    peak = np.maximum.accumulate(
        equity_array
    )

    drawdown = (
        (peak - equity_array)
        / peak
        * 100
    )

    max_drawdown = float(
        np.max(drawdown)
    )

else:

    max_drawdown = 0.0

compound_return = (
    (equity - 100)
    / 100
) * 100

# ============================================================
# RESULTS
# ============================================================

results_summary = {
    "strategy":
        "Weekly RSI 33/60/70",

    "parameters": {
        "rsi_period": RSI_PERIOD,
        "rsi_buy": RSI_BUY,
        "rsi_sell_1": RSI_SELL_1,
        "rsi_sell_2": RSI_SELL_2,
        "first_sell_percent": 50,
        "second_sell_percent": 50,
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
        "compound_portfolio_return_percent":
            round(
                compound_return,
                2
            ),
        "average_winning_trade_percent":
            round(
                avg_win,
                2
            ),
        "average_losing_trade_percent":
            round(
                avg_loss,
                2
            ),
        "maximum_drawdown_percent":
            round(
                max_drawdown,
                2
            )
    }
}

# ============================================================
# SAVE RESULTS
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

print("=" * 50)
print("WEEKLY RSI 33/60/70 BACKTEST")
print("=" * 50)
print(f"Total Trades: {total_count}")
print(f"Win Rate:     {win_rate:.2f}%")
print(f"Compound:     {compound_return:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("=" * 50)

import json
import os
import pandas as pd
import numpy as np

# ============================================================
# WEEKLY RSI 33/60/70 + EMA20 DIRECTION
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

    rsi = 100 - (100 / (1 + rs))

    return rsi

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

        df["EMA20"] = df["Close"].ewm(
            span=20,
            adjust=False
        ).mean()

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
# ALL DATES
# ============================================================

all_dates = sorted(
    set().union(
        *[df.index for df in prepared_data.values()]
    )
)

# ============================================================
# STATE
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

# Compound portfolio
portfolio_value = 100.0
portfolio_curve = []

# ============================================================
# BACKTEST
# ============================================================

for current_date in all_dates:

    for name, df in prepared_data.items():

        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)

        # Need enough history for RSI and EMA20 - 3 weeks
        if idx < 25:
            continue

        row = df.iloc[idx]

        if pd.isna(row["RSI14"]) or pd.isna(row["EMA20"]):
            continue

        price = float(row["Close"])
        rsi = float(row["RSI14"])
        ema20 = float(row["EMA20"])

        # ====================================================
        # EMA20 DIRECTION
        # Current EMA20 > EMA20 from 3 weeks ago
        # ====================================================

        ema20_3w_ago = float(
            df["EMA20"].iloc[idx - 3]
        )

        ema20_direction = ema20 > ema20_3w_ago

        date_str = current_date.strftime("%Y-%m-%d")

        s = states[name]

        # ====================================================
        # NO POSITION -> BUY
        # ====================================================

        if s["position"] == 0.0:

            buy_signal = (
                rsi < RSI_BUY
                and ema20_direction
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

        # ====================================================
        # POSITION MANAGEMENT
        # ====================================================

        elif s["position"] > 0:

            entry_price = s["entry_price"]

            profit = (
                (price - entry_price)
                / entry_price
            ) * 100

            # -----------------------------------------------
            # HARD STOP
            # -----------------------------------------------

            if profit <= -HARD_STOP_PERCENT:

                sold_position = s["position"]

                sales_profit = profit

                s["sales"].append({
                    "date": date_str,
                    "price": round(price, 2),
                    "position_sold": round(
                        sold_position,
                        2
                    ),
                    "profit_pct": round(
                        sales_profit,
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

            # -----------------------------------------------
            # SELL 50% AT RSI 60
            # -----------------------------------------------

            if (
                s["position"] == 1.0
                and rsi >= RSI_SELL_1
            ):

                sold = FIRST_SELL_PERCENT

                sale_profit = profit

                s["sales"].append({
                    "date": date_str,
                    "price": round(price, 2),
                    "position_sold": sold,
                    "profit_pct": round(
                        sale_profit,
                        2
                    ),
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

            # -----------------------------------------------
            # SELL REMAINING 50% AT RSI 70
            # -----------------------------------------------

            elif (
                s["position"] == 0.5
                and rsi >= RSI_SELL_2
            ):

                sold = SECOND_SELL_PERCENT

                sale_profit = profit

                s["sales"].append({
                    "date": date_str,
                    "price": round(price, 2),
                    "position_sold": sold,
                    "profit_pct": round(
                        sale_profit,
                        2
                    ),
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

                    # Weighted average profit
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

    # ========================================================
    # PORTFOLIO CURVE
    # ========================================================

    closed_today = [
        t for t in trades_history
        if (
            t["status"] == "CLOSED"
            and t["exit_date"] == current_date.strftime(
                "%Y-%m-%d"
            )
        )
    ]

    for trade in closed_today:

        if trade["profit_pct"] is not None:

            # Each completed trade is compounded
            portfolio_value *= (
                1 + trade["profit_pct"] / 100
            )

    portfolio_curve.append(portfolio_value)

# ============================================================
# CLOSE OPEN TRADES FOR REPORT ONLY
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

# ============================================================
# STATISTICS
# ============================================================

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
        np.mean(
            [
                t["profit_pct"]
                for t in winning_trades
            ]
        )
    )
    if winning_trades
    else 0.0
)

avg_loss = (
    float(
        np.mean(
            [
                t["profit_pct"]
                for t in losing_trades
            ]
        )
    )
    if losing_trades
    else 0.0
)

# ============================================================
# MAX DRAWDOWN
# ============================================================

if portfolio_curve:

    equity = np.array(
        portfolio_curve,
        dtype=float
    )

    peak = np.maximum.accumulate(equity)

    drawdown = (
        (peak - equity)
        / peak
        * 100
    )

    max_drawdown = float(
        np.max(drawdown)
    )

else:

    max_drawdown = 0.0

compound_return = (
    (portfolio_value - 100.0)
    / 100.0
) * 100

# ============================================================
# RESULTS
# ============================================================

results_summary = {

    "strategy":
        "Weekly RSI 33/60/70 + EMA20 Direction",

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
            50,

        "second_sell_percent":
            50,

        "ema20_direction_filter":
            "EMA20 > EMA20 from 3 weeks ago",

        "hard_stop_percent":
            HARD_STOP_PERCENT
    },

    "statistics": {

        "total_trades":
            total_count,

        "winning_trades":
            wins_count,

        "losing_trades":
            losses_count,

        "win_rate_percent":
            round(win_rate, 2),

        "compound_portfolio_return_percent":
            round(
                compound_return,
                2
            ),

        "average_winning_trade_percent":
            round(avg_win, 2),

        "average_losing_trade_percent":
            round(avg_loss, 2),

        "maximum_drawdown_percent":
            round(
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

print("=" * 50)
print("WEEKLY RSI 33/60/70 BACKTEST")
print("=" * 50)
print(f"Trades:       {total_count}")
print(f"Win Rate:     {win_rate:.2f}%")
print(f"Compound:     {compound_return:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("=" * 50)

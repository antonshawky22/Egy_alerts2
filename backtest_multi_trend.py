import json
import os
import numpy as np
import pandas as pd

# ============================================================
# WEEKLY RSI 33/60/70 BACKTEST
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

    rsi = rsi.where(
        avg_loss != 0,
        100
    )

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

        if len(df) < RSI_PERIOD + 5:
            continue

        df["Close"] = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        df = df.dropna(subset=["Close"])

        df["RSI14"] = calculate_rsi(
            df["Close"],
            RSI_PERIOD
        )

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
# STATE
# ============================================================

states = {}

for name in prepared_data:
    states[name] = {
        "position": 0.0,
        "entry_price": 0.0,
        "entry_date": None,
        "entry_rsi": None,
        "sales": [],
        "last_exit_date": None
    }

trades_history = []

# ============================================================
# PORTFOLIO TRACKING
# ============================================================

portfolio_returns = []
realized_profit = 0.0

# كل سهم يعامل كصفقة مستقلة.
# هذا يمنع تضخيم العائد بسبب تركيب أرباح صفقات متداخلة.

# ============================================================
# BACKTEST
# ============================================================

for current_date in all_dates:

    for name, df in prepared_data.items():

        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)

        if idx < RSI_PERIOD:
            continue

        row = df.iloc[idx]

        price = float(row["Close"])
        rsi = float(row["RSI14"])

        if np.isnan(rsi) or price <= 0:
            continue

        date_str = current_date.strftime("%Y-%m-%d")

        s = states[name]

        # ====================================================
        # POSITION OPEN
        # ====================================================

        if s["position"] > 0:

            entry_price = s["entry_price"]

            profit_pct = (
                (price - entry_price)
                / entry_price
            ) * 100

            # =================================================
            # HARD STOP
            # =================================================

            hard_stop = (
                price <=
                entry_price *
                (1 - HARD_STOP_PERCENT / 100)
            )

            if hard_stop:

                remaining_position = s["position"]

                sale_profit = profit_pct

                s["sales"].append({
                    "date": date_str,
                    "price": round(price, 2),
                    "position_sold": round(
                        remaining_position,
                        2
                    ),
                    "profit_pct": round(
                        sale_profit,
                        2
                    ),
                    "reason": "HARD_STOP"
                })

                total_profit = 0.0

                for sale in s["sales"]:

                    sale_weight = sale["position_sold"]

                    total_profit += (
                        sale["profit_pct"]
                        * sale_weight
                    )

                trades_history.append({
                    "symbol": name,
                    "status": "CLOSED",
                    "entry_date": s["entry_date"],
                    "entry_price": round(
                        s["entry_price"],
                        2
                    ),
                    "entry_rsi": round(
                        s["entry_rsi"],
                        2
                    ),
                    "sales": s["sales"].copy(),
                    "exit_date": date_str,
                    "exit_price": round(price, 2),
                    "profit_pct": round(
                        total_profit,
                        2
                    ),
                    "exit_reason": "HARD_STOP"
                })

                realized_profit += total_profit

                s["position"] = 0.0
                s["entry_price"] = 0.0
                s["entry_date"] = None
                s["entry_rsi"] = None
                s["sales"] = []
                s["last_exit_date"] = current_date

                continue

            # =================================================
            # FIRST SELL 50% AT RSI 60
            # =================================================

            if (
                s["position"] == 1.0
                and rsi >= RSI_SELL_1
            ):

                sell_position = FIRST_SELL_PERCENT

                sale_profit = profit_pct

                s["sales"].append({
                    "date": date_str,
                    "price": round(price, 2),
                    "position_sold": sell_position,
                    "profit_pct": round(
                        sale_profit,
                        2
                    ),
                    "reason": "RSI_60"
                })

                s["position"] = 0.5

                continue

            # =================================================
            # SECOND SELL 50% AT RSI 70
            # =================================================

            if (
                s["position"] == 0.5
                and rsi >= RSI_SELL_2
            ):

                sell_position = SECOND_SELL_PERCENT

                sale_profit = profit_pct

                s["sales"].append({
                    "date": date_str,
                    "price": round(price, 2),
                    "position_sold": sell_position,
                    "profit_pct": round(
                        sale_profit,
                        2
                    ),
                    "reason": "RSI_70"
                })

                total_profit = 0.0

                for sale in s["sales"]:

                    total_profit += (
                        sale["profit_pct"]
                        * sale["position_sold"]
                    )

                trades_history.append({
                    "symbol": name,
                    "status": "CLOSED",
                    "entry_date": s["entry_date"],
                    "entry_price": round(
                        s["entry_price"],
                        2
                    ),
                    "entry_rsi": round(
                        s["entry_rsi"],
                        2
                    ),
                    "sales": s["sales"].copy(),
                    "exit_date": date_str,
                    "exit_price": round(price, 2),
                    "profit_pct": round(
                        total_profit,
                        2
                    ),
                    "exit_reason": "RSI_70"
                })

                realized_profit += total_profit

                s["position"] = 0.0
                s["entry_price"] = 0.0
                s["entry_date"] = None
                s["entry_rsi"] = None
                s["sales"] = []
                s["last_exit_date"] = current_date

                continue

        # ====================================================
        # NEW ENTRY
        # ====================================================

        if s["position"] == 0.0:

            # منع الدخول في نفس تاريخ الخروج
            if s["last_exit_date"] == current_date:
                continue

            if rsi < RSI_BUY:

                s["position"] = 1.0
                s["entry_price"] = price
                s["entry_date"] = date_str
                s["entry_rsi"] = rsi
                s["sales"] = []

    # ========================================================
    # EQUITY TRACKING
    # ========================================================

    current_equity = 0.0

    for name, df in prepared_data.items():

        s = states[name]

        if s["position"] > 0:

            if current_date in df.index:

                current_price = float(
                    df.loc[current_date, "Close"]
                )

                if s["entry_price"] > 0:

                    unrealized = (
                        (current_price - s["entry_price"])
                        / s["entry_price"]
                    ) * s["position"] * 100

                    current_equity += unrealized

    portfolio_returns.append(
        realized_profit + current_equity
    )

# ============================================================
# CLOSE REMAINING OPEN POSITIONS FOR REPORT ONLY
# ============================================================

open_positions = []

for name, s in states.items():

    if s["position"] > 0:

        df = prepared_data[name]

        last_date = df.index[-1]
        last_price = float(
            df["Close"].iloc[-1]
        )

        open_positions.append({
            "symbol": name,
            "status": "OPEN",
            "entry_date": s["entry_date"],
            "entry_price": round(
                s["entry_price"],
                2
            ),
            "entry_rsi": round(
                s["entry_rsi"],
                2
            ),
            "current_date": last_date.strftime(
                "%Y-%m-%d"
            ),
            "current_price": round(
                last_price,
                2
            ),
            "current_profit_pct": round(
                (
                    (last_price - s["entry_price"])
                    / s["entry_price"]
                ) * 100,
                2
            ),
            "remaining_position": s["position"]
        })

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
# MAX DRAWDOWN
# ============================================================

if portfolio_returns:

    equity = np.array(
        portfolio_returns,
        dtype=float
    )

    equity_curve = 100 + equity

    peak = np.maximum.accumulate(
        equity_curve
    )

    drawdown = (
        (peak - equity_curve)
        / peak
    ) * 100

    max_drawdown = float(
        np.max(drawdown)
    )

else:
    max_drawdown = 0.0

# ============================================================
# ENTRY RSI ANALYSIS
# ============================================================

rsi_under_30 = [
    t for t in closed_trades
    if t["entry_rsi"] < 30
]

rsi_30_to_33 = [
    t for t in closed_trades
    if 30 <= t["entry_rsi"] < 33
]

rsi_under_30_wins = [
    t for t in rsi_under_30
    if t["profit_pct"] > 0
]

rsi_30_to_33_wins = [
    t for t in rsi_30_to_33
    if t["profit_pct"] > 0
]

def group_stats(group):

    count = len(group)

    wins = len([
        t for t in group
        if t["profit_pct"] > 0
    ])

    losses = count - wins

    avg_profit = (
        float(
            np.mean([
                t["profit_pct"]
                for t in group
            ])
        )
        if group
        else 0.0
    )

    win_rate_group = (
        wins / count * 100
        if count > 0
        else 0.0
    )

    return {
        "trades": count,
        "winning": wins,
        "losing": losses,
        "win_rate_percent": round(
            win_rate_group,
            2
        ),
        "average_profit_percent": round(
            avg_profit,
            2
        )
    }

# ============================================================
# LOSS ANALYSIS
# ============================================================

hard_stop_losses = [
    t for t in losing_trades
    if t["exit_reason"] == "HARD_STOP"
]

rsi60_winners = [
    t for t in winning_trades
    if any(
        s["reason"] == "RSI_60"
        for s in t["sales"]
    )
]

rsi70_winners = [
    t for t in winning_trades
    if t["exit_reason"] == "RSI_70"
]

# ============================================================
# RESULTS
# ============================================================

results_summary = {
    "strategy": "Weekly RSI 33/60/70",
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
        "total_profit_percent": round(
            realized_profit,
            2
        ),
        "average_winning_trade_percent": round(
            avg_win,
            2
        ),
        "average_losing_trade_percent": round(
            avg_loss,
            2
        ),
        "maximum_drawdown_percent": round(
            max_drawdown,
            2
        )
    },
    "entry_rsi_analysis": {
        "rsi_below_30": group_stats(
            rsi_under_30
        ),
        "rsi_30_to_33": group_stats(
            rsi_30_to_33
        )
    },
    "exit_analysis": {
        "hard_stop_losses": len(
            hard_stop_losses
        ),
        "rsi_60_winners": len(
            rsi60_winners
        ),
        "rsi_70_final_exits": len(
            rsi70_winners
        )
    },
    "open_positions": open_positions
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
# CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 55)
print("WEEKLY RSI 33/60/70 BACKTEST")
print("=" * 55)

print(f"Total Trades:       {total_count}")
print(f"Winning Trades:     {wins_count}")
print(f"Losing Trades:      {losses_count}")
print(f"Win Rate:           {win_rate:.2f}%")
print(f"Total Profit:       {realized_profit:.2f}%")
print(f"Average Win:        {avg_win:.2f}%")
print(f"Average Loss:       {avg_loss:.2f}%")
print(f"Max Drawdown:       {max_drawdown:.2f}%")

print("\n--- ENTRY RSI ANALYSIS ---")

print(
    f"RSI < 30:           "
    f"{len(rsi_under_30)} trades | "
    f"{len(rsi_under_30_wins)} wins"
)

print(
    f"RSI 30-33:          "
    f"{len(rsi_30_to_33)} trades | "
    f"{len(rsi_30_to_33_wins)} wins"
)

print("\n--- EXIT ANALYSIS ---")

print(
    f"Hard Stop Losses:   "
    f"{len(hard_stop_losses)}"
)

print(
    f"RSI 60 Winners:     "
    f"{len(rsi60_winners)}"
)

print(
    f"RSI 70 Final Exits: "
    f"{len(rsi70_winners)}"
)

print("\n--- OPEN POSITIONS ---")

for p in open_positions:

    print(
        f"{p['symbol']} | "
        f"Entry {p['entry_price']} | "
        f"RSI {p['entry_rsi']} | "
        f"P/L {p['current_profit_pct']}%"
    )

print("=" * 55)

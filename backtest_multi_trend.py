import json
import os
import pandas as pd
import numpy as np

# ============================================================
# STAGED ENTRY/EXIT SYSTEM (EMA 75 & EMA 100)
# ============================================================

DB_FILE = "egx_history_database_v2.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

EMA_PERIOD = 75  # يمكنك تغييرها إلى 100 لتجربة المتوسط الآخر

symbols = {
    "OLFI": "OLFI", "EMFD": "EMFD", "ETEL": "ETEL", "EAST": "EAST",
    "EFIH": "EFIH", "ABUK": "ABUK", "OIH": "OIH", "SWDY": "SWDY", "ISPH": "ISPH",
    "ATQA": "ATQA", "MTIE": "MTIE", "HRHO": "HRHO", "ORWE": "ORWE",
    "JUFO": "JUFO", "DSCW": "DSCW", "SUGR": "SUGR", "ELSH": "ELSH", "RMDA": "RMDA",
    "RAYA": "RAYA", "EEII": "EEII", "MPCO": "MPCO", "GBCO": "GBCO", "TMGH": "TMGH",
    "ORHD": "ORHD", "AMOC": "AMOC", "FWRY": "FWRY", "COMI": "COMI", "ADIB": "ADIB",
    "PHDC": "PHDC", "MCQE": "MCQE", "SKPC": "SKPC", "EGAL": "EGAL"
}

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(f"Database file not found: {DB_FILE}")

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
        df = pd.DataFrame.from_dict(content["data"], orient="index", columns=content["columns"])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)

        if len(df) < EMA_PERIOD + 10:
            continue

        close = df["Close"]
        df["EMA_MAIN"] = close.ewm(span=EMA_PERIOD, adjust=False).mean()
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()

        prepared_data[name] = df
    except Exception:
        pass

all_dates = sorted(set().union(*[df.index for df in prepared_data.values()]))

states = {name: {"stage": 0, "entry_1": 0.0, "entry_2": 0.0, "avg_price": 0.0, "peak_price": 0.0} for name in prepared_data}
trades_history = []
total_portfolio_profit = 0.0
portfolio_equity_curve = []

for current_date in all_dates:
    for name, df in prepared_data.items():
        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)
        if idx < EMA_PERIOD:
            continue

        df_slice = df.iloc[: idx + 1]
        if df_slice[["Close", "EMA_MAIN", "EMA20"]].iloc[-1].isna().any():
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        price = float(df_slice["Close"].iloc[-1])
        ema_main = float(df_slice["EMA_MAIN"].iloc[-1])
        ema20 = float(df_slice["EMA20"].iloc[-1])
        
        prev_price = float(df_slice["Close"].iloc[-2])
        prev_ema = float(df_slice["EMA_MAIN"].iloc[-2])

        s = states[name]

        # 🟢 مرحلة الدخول الأولى (50% من السيولة): اختراق المتوسط الرئيسي لأعلى
        if s["stage"] == 0 and prev_price <= prev_ema and price > ema_main:
            s["stage"] = 1
            s["entry_1"] = price
            s["avg_price"] = price
            s["peak_price"] = price
            
            trades_history.append({
                "symbol": name, "status": "OPEN_STAGE_1", "entry_date": date_str,
                "entry_price": round(price, 2), "exit_date": None, "exit_price": None, "profit_pct": None
            })

        # 🟢 مرحلة الدخول الثانية (50% المتبقية): استقرار فوق EMA20 لتأكيد الزخم
        elif s["stage"] == 1 and price > ema20 and price > s["entry_1"] * 1.02:
            s["stage"] = 2
            s["entry_2"] = price
            s["avg_price"] = (s["entry_1"] + s["entry_2"]) / 2.0

        # 🔴 إدارة الخروج والبيع على مراحل
        if s["stage"] > 0:
            if price > s["peak_price"]:
                s["peak_price"] = price

            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
            peak_profit = ((s["peak_price"] - s["avg_price"]) / s["avg_price"]) * 100

            # شروط الخروج: كسر المتوسط الرئيسي لأسفل أو وقف خسارة
            hard_stop = profit <= -5.0
            trailing_stop = (peak_profit >= 12.0) and ((s["peak_price"] - price) / s["peak_price"] * 100 >= 4.5)
            cross_below = price < ema_main

            if hard_stop or trailing_stop or cross_below:
                active = [t for t in trades_history if t["symbol"] == name and "OPEN" in t["status"]]
                if active:
                    active[-1]["status"] = "CLOSED"
                    active[-1]["exit_date"] = date_str
                    active[-1]["exit_price"] = round(price, 2)
                    active[-1]["profit_pct"] = round(profit, 2)

                total_portfolio_profit += profit
                s["stage"] = 0
                s["entry_1"] = 0.0
                s["entry_2"] = 0.0
                s["avg_price"] = 0.0
                s["peak_price"] = 0.0

    portfolio_equity_curve.append(total_portfolio_profit)

# ============================================================
# COMPUTE STATISTICS
# ============================================================

closed_trades = [t for t in trades_history if t["status"] == "CLOSED"]
winning_trades = [t for t in closed_trades if t["profit_pct"] is not None and t["profit_pct"] > 0]
losing_trades = [t for t in closed_trades if t["profit_pct"] is not None and t["profit_pct"] <= 0]

total_count = len(closed_trades)
wins_count = len(winning_trades)
losses_count = len(losing_trades)
win_rate = (wins_count / total_count) * 100 if total_count > 0 else 0.0

avg_win = float(np.mean([t["profit_pct"] for t in winning_trades])) if winning_trades else 0.0
avg_loss = float(np.mean([t["profit_pct"] for t in losing_trades])) if losing_trades else 0.0

equity = np.array(portfolio_equity_curve) + 100.0
peak = np.maximum.accumulate(equity)
drawdown = (peak - equity) / peak * 100.0
max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

results_summary = {
    "statistics": {
        "total_trades": total_count,
        "winning_trades": wins_count,
        "losing_trades": losses_count,
        "win_rate_percent": round(win_rate, 2),
        "total_profit_percent": round(total_portfolio_profit, 2),
        "average_winning_trade_percent": round(avg_win, 2),
        "average_losing_trade_percent": round(avg_loss, 2),
        "maximum_drawdown_percent": round(max_drawdown, 2)
    }
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

with open(TRADES_FILE, "w", encoding="utf-8") as f:
    json.dump(trades_history, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print(f"STAGED SYSTEM (EMA {EMA_PERIOD}) COMPLETE")
print("=" * 60)
print(f"Total Closed Trades: {total_count}")
print(f"Win Rate:            {win_rate:.2f}%")
print(f"Total Profit:        {total_portfolio_profit:.2f}%")
print(f"Max Drawdown:        {max_drawdown:.2f}%")
print("=" * 60)

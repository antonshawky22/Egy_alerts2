import json
import os
import pandas as pd
import numpy as np

# ============================================================
# LADDER SYSTEM V3.4 - STANDALONE BACKTEST ENGINE
# ============================================================

DB_FILE = "egx_history_database_v2.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"
STOCK_SUMMARY_FILE = "backtest_summary_by_stock.json"

symbols = {
    "OLFI": "OLFI", "EMFD": "EMFD", "ETEL": "ETEL", "EAST": "EAST",
    "EFIH": "EFIH", "ABUK": "ABUK", "OIH": "OIH", "SWDY": "SWDY", "ISPH": "ISPH",
    "ATQA": "ATQA", "MTIE": "MTIE", "HRHO": "HRHO", "ORWE": "ORWE",
    "JUFO": "JUFO", "DSCW": "DSCW", "SUGR": "SUGR", "ELSH": "ELSH", "RMDA": "RMDA",
    "RAYA": "RAYA", "EEII": "EEII", "MPCO": "MPCO", "GBCO": "GBCO", "TMGH": "TMGH",
    "ORHD": "ORHD", "AMOC": "AMOC", "FWRY": "FWRY", "COMI": "COMI", "ADIB": "ADIB",
    "PHDC": "PHDC", "MCQE": "MCQE", "SKPC": "SKPC", "EGAL": "EGAL"
}

# ============================================================
# HELPER FUNCTIONS (MATCHING LIVE CODE 100%)
# ============================================================

def rsi(series, period=14):
    if len(series) < period + 1:
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def update_avg(old_avg, old_pos, new_price, new_pos):
    if new_pos == 0:
        return 0.0
    added_pos = new_pos - old_pos
    if added_pos <= 0:
        return old_avg
    total_cost = (old_avg * old_pos) + (new_price * added_pos)
    return total_cost / new_pos

def calc_final_pnl(tracker, current_p, current_w):
    temp_tracker = list(tracker) + [(current_w, current_p)]
    w_sum = sum(w for w, _ in temp_tracker)
    return sum(p * w for w, p in temp_tracker) / w_sum if w_sum > 0 else current_p

# ============================================================
# LOAD DATABASE & PREPARE DATA
# ============================================================

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

        if len(df) < 80:
            continue

        close = df["Close"]
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA30"] = close.ewm(span=30, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()
        df["EMA75"] = close.ewm(span=75, adjust=False).mean()
        df["RSI"] = rsi(close)

        prepared_data[name] = df
    except Exception:
        pass

all_dates = set()
for df in prepared_data.values():
    all_dates.update(df.index)
all_dates = sorted(all_dates)

# ============================================================
# STATE TRACKING
# ============================================================

states = {
    name: {
        "cycle": 1,
        "position": 0.0,
        "avg_price": 0.0,
        "peak_profit": 0.0,
        "realized_pnl_tracker": []
    }
    for name in prepared_data
}

trades_history = []
stock_summaries = {}
total_portfolio_profit = 0.0

# ============================================================
# BACKTEST LOOP
# ============================================================

for current_date in all_dates:
    for name, df in prepared_data.items():
        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)
        if idx < 79:  # نحتاج 80 شمعة سابقة على الأقل للفلتر
            continue

        df_slice = df.iloc[: idx + 1]
        last = df_slice.iloc[-1]

        if df_slice[["Open", "Close", "EMA75", "RSI"]].iloc[-1].isna().any():
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        price = float(last["Close"])
        rsi_val = float(last["RSI"])

        s = states[name]
        s["position"] = round(float(s["position"]), 2)

        if s["position"] == 0.0:
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0

        # 1. Safe To Buy Filter (80-bars high/low)
        lookback = min(len(df_slice), 80)
        lowest_80 = float(df_slice["Low"].tail(lookback).min())
        highest_80 = float(df_slice["High"].tail(lookback).max())
        run_up_percent = ((highest_80 - lowest_80) / lowest_80) * 100 if lowest_80 > 0 else 0.0
        safe_to_buy = run_up_percent <= 60.0

        # 2. No Gap Down Filter (Last 3 candles)
        gap1 = ((df_slice["Open"].iloc[-1] - df_slice["Close"].iloc[-2]) / df_slice["Close"].iloc[-2]) * 100
        gap2 = ((df_slice["Open"].iloc[-2] - df_slice["Close"].iloc[-3]) / df_slice["Close"].iloc[-3]) * 100
        gap3 = ((df_slice["Open"].iloc[-3] - df_slice["Close"].iloc[-4]) / df_slice["Close"].iloc[-4]) * 100
        no_gap_down = (gap1 > -3.0) and (gap2 > -3.0) and (gap3 > -3.0)

        # 3. EMA75 Uptrend Condition
        ema_up = (
            df_slice["EMA75"].iloc[-1] > df_slice["EMA75"].iloc[-5]
            and df_slice["EMA75"].iloc[-5] > df_slice["EMA75"].iloc[-10]
            and df_slice["EMA75"].iloc[-1] > df_slice["EMA75"].iloc[-10] * 1.002
            and price <= df_slice["EMA75"].iloc[-1] * 1.08
        )

        buy1 = safe_to_buy and ema_up and no_gap_down and rsi_val <= 52
        buy2 = safe_to_buy and ema_up and no_gap_down and rsi_val <= 40
        buy3 = safe_to_buy and ema_up and no_gap_down and rsi_val <= 35

        profit = 0.0
        if s["avg_price"] > 0:
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100

        sell1 = 0.00 < s["position"] <= 0.33 and rsi_val >= 68 and profit > 4.0
        sell2 = 0.33 < s["position"] <= 0.66 and rsi_val >= 72 and profit > 5.0
        sell3 = s["position"] > 0.66 and rsi_val >= 76 and profit > 6.0

        initial_pos = s["position"]
        action = None

        # --- EXECUTE BUYS ---
        if s["position"] == 0 and buy1:
            s["position"] = 0.33
            s["avg_price"] = price
            s["peak_profit"] = 0.0
            s["realized_pnl_tracker"] = []
            profit = 0.0
            action = "BUY L1"

            trades_history.append({
                "symbol": name,
                "cycle": s["cycle"],
                "status": "OPEN",
                "first_entry": f"{date_str} with price {price:.2f}",
                "second_entry": None,
                "third_entry": None,
                "last_totally_average_price": round(price, 2),
                "exits": [],
                "exit_price": None,
                "exit_date": None,
                "profit_pct": None
            })

        elif 0.32 < s["position"] < 0.5 and buy2 and price < s["avg_price"] * 0.95:
            old_pos = s["position"]
            s["position"] = 0.66
            s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
            action = "BUY L2"

            active = [t for t in trades_history if t["symbol"] == name and t["status"] == "OPEN"]
            if active:
                active[-1]["second_entry"] = f"{date_str} with price {price:.2f}"
                active[-1]["last_totally_average_price"] = round(s["avg_price"], 2)

        elif 0.65 < s["position"] < 1 and buy3 and price < s["avg_price"] * 0.92:
            old_pos = s["position"]
            s["position"] = 1.0
            s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
            action = "BUY L3"

            active = [t for t in trades_history if t["symbol"] == name and t["status"] == "OPEN"]
            if active:
                active[-1]["third_entry"] = f"{date_str} with price {price:.2f}"
                active[-1]["last_totally_average_price"] = round(s["avg_price"], 2)

        if profit > s["peak_profit"]:
            s["peak_profit"] = profit

        # --- EXECUTE SELLS / STOPS ---
        if initial_pos > 0 and s["position"] > 0:
            stop_triggered = False

            if s["position"] <= 0.33 and profit <= -10:
                stop_triggered = True
            elif s["position"] <= 0.66 and profit <= -5:
                stop_triggered = True
            elif s["position"] == 1.0 and profit <= -4:
                stop_triggered = True

            if s["peak_profit"] > 10 and (s["peak_profit"] - profit) >= 4:
                stop_triggered = True

            active = [t for t in trades_history if t["symbol"] == name and t["status"] == "OPEN"]

            if stop_triggered or sell3:
                action = "STOP LOSS" if stop_triggered else "EXIT FULL"
                total_profit = calc_final_pnl(s["realized_pnl_tracker"], profit, s["position"])

                if active:
                    active[-1]["status"] = "CLOSED"
                    active[-1]["exit_price"] = round(price, 2)
                    active[-1]["exit_date"] = date_str
                    active[-1]["profit_pct"] = round(total_profit, 2)

                total_portfolio_profit += total_profit
                s["position"] = 0.0

            elif sell2:
                sell_amount = min(0.33, s["position"])
                s["realized_pnl_tracker"].append((sell_amount, profit))
                s["position"] = round(s["position"] - sell_amount, 2)
                action = "SELL L2"

                if active:
                    exit_log = f"{date_str}: Sold {sell_amount*100:.0f}% at price {price:.2f} (Profit: {profit:+.2f}%)"
                    active[-1]["exits"].append(exit_log)

                    if s["position"] == 0.0:
                        w_sum = sum(w for w, _ in s["realized_pnl_tracker"])
                        total_profit = sum(p * w for w, p in s["realized_pnl_tracker"]) / w_sum if w_sum > 0 else profit
                        active[-1]["status"] = "CLOSED"
                        active[-1]["exit_price"] = round(price, 2)
                        active[-1]["exit_date"] = date_str
                        active[-1]["profit_pct"] = round(total_profit, 2)
                        total_portfolio_profit += total_profit

            elif sell1:
                sell_amount = min(0.33, s["position"])
                s["realized_pnl_tracker"].append((sell_amount, profit))
                s["position"] = round(s["position"] - sell_amount, 2)
                action = "SELL L1"

                if active:
                    exit_log = f"{date_str}: Sold {sell_amount*100:.0f}% at price {price:.2f} (Profit: {profit:+.2f}%)"
                    active[-1]["exits"].append(exit_log)

                    if s["position"] == 0.0:
                        w_sum = sum(w for w, _ in s["realized_pnl_tracker"])
                        total_profit = sum(p * w for w, p in s["realized_pnl_tracker"]) / w_sum if w_sum > 0 else profit
                        active[-1]["status"] = "CLOSED"
                        active[-1]["exit_price"] = round(price, 2)
                        active[-1]["exit_date"] = date_str
                        active[-1]["profit_pct"] = round(total_profit, 2)
                        total_portfolio_profit += total_profit

            s["position"] = round(s["position"], 2)

        if action and s["position"] == 0.0 and ("SELL" in action or "EXIT" in action or "STOP" in action):
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["realized_pnl_tracker"] = []
            s["cycle"] += 1

# ============================================================
# COMPUTE STATISTICS & SAVE FILES
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

# تجميع ملخص كل سهم
for t in closed_trades:
    sym = t["symbol"]
    if sym not in stock_summaries:
        stock_summaries[sym] = {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "total_profit_pct": 0.0}
    
    stock_summaries[sym]["total_trades"] += 1
    pnl = t["profit_pct"] if t["profit_pct"] is not None else 0.0
    
    if pnl > 0:
        stock_summaries[sym]["winning_trades"] += 1
    else:
        stock_summaries[sym]["losing_trades"] += 1
    
    stock_summaries[sym]["total_profit_pct"] = round(stock_summaries[sym]["total_profit_pct"] + pnl, 2)

results_summary = {
    "statistics": {
        "total_trades": total_count,
        "winning_trades": wins_count,
        "losing_trades": losses_count,
        "win_rate_percent": round(win_rate, 2),
        "total_profit_percent": round(total_portfolio_profit, 2),
        "average_winning_trade_percent": round(avg_win, 2),
        "average_losing_trade_percent": round(avg_loss, 2)
    }
}

# 1. حفظ ملف النتائج الرئيسية
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

# 2. حفظ سجل الصفقات المفتوحة والمغلقة
with open(TRADES_FILE, "w", encoding="utf-8") as f:
    json.dump(trades_history, f, indent=2, ensure_ascii=False)

# 3. حفظ الملخص المجمع لكل سهم
with open(STOCK_SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(stock_summaries, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("EGX LADDER SYSTEM (v3.4) - BACKTEST COMPLETE")
print("=" * 60)
print(f"Total Closed Trades: {total_count}")
print(f"Win Rate:            {win_rate:.2f}%")
print(f"Total Profit:        {total_portfolio_profit:.2f}%")
print("=" * 60)
print(f" Saved stats to:    '{RESULTS_FILE}'")
print(f" Saved trades to:   '{TRADES_FILE}'")
print(f" Saved summary to:  '{STOCK_SUMMARY_FILE}'")

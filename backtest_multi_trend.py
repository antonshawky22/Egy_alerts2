import json
import os
import pandas as pd
import numpy as np

# ============================================================
# MARKET STRUCTURE SHIFT (MSS) ENGINE - FULL BACKTEST (v1.0)
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
# HELPER FUNCTIONS
# ============================================================

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

def find_pivots(df, window=5):
    """استخراج القمم والقيعان المحورية للهيكل السعري"""
    df_copy = df.copy()
    df_copy['pivot_high'] = np.nan
    df_copy['pivot_low'] = np.nan

    for i in range(window, len(df_copy) - window):
        high_range = df_copy['High'].iloc[i - window : i + window + 1]
        if df_copy['High'].iloc[i] == high_range.max():
            df_copy.loc[df_copy.index[i], 'pivot_high'] = df_copy['High'].iloc[i]

        low_range = df_copy['Low'].iloc[i - window : i + window + 1]
        if df_copy['Low'].iloc[i] == low_range.min():
            df_copy.loc[df_copy.index[i], 'pivot_low'] = df_copy['Low'].iloc[i]

    df_copy['last_pivot_high'] = df_copy['pivot_high'].ffill()
    df_copy['last_pivot_low'] = df_copy['pivot_low'].ffill()
    return df_copy

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
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()

        # حساب هيكل السوق (Pivot Highs & Lows)
        df = find_pivots(df, window=5)

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
        "structure_stop": 0.0,
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
        if idx < 60:
            continue

        df_slice = df.iloc[: idx + 1]
        last = df_slice.iloc[-1]

        if df_slice[["Open", "Close", "EMA20", "EMA50", "last_pivot_high", "last_pivot_low"]].iloc[-1].isna().any():
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        price = float(last["Close"])
        prev_close = float(df_slice["Close"].iloc[-2])

        s = states[name]
        s["position"] = round(float(s["position"]), 2)

        if s["position"] == 0.0:
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["structure_stop"] = 0.0

        # --- تحليل هيكل السوق (Market Structure Shift) ---
        last_ph = float(last["last_pivot_high"])
        last_pl = float(last["last_pivot_low"])

        # كسر القمة المحورية المباشرة (Market Structure Breakout)
        mss_breakout = (prev_close <= last_ph) and (price > last_ph)
        trend_aligned = float(last["EMA20"]) > float(last["EMA50"])

        # إعادة اختبار مستوى الهيكل المكسور (Structure Re-Test)
        retest_support = (price <= last_ph * 1.01) and (price >= last_ph * 0.985)

        # شروط الشراء الهيكلية
        buy1 = mss_breakout and trend_aligned
        buy2 = retest_support and trend_aligned and (price < s["avg_price"] * 0.97)
        buy3 = trend_support = (price > float(last["EMA50"])) and (price < s["avg_price"] * 0.93)

        profit = 0.0
        if s["avg_price"] > 0:
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100

        # أهداف البيع المبنية على امتداد الهيكل السعري
        sell1 = (0.00 < s["position"] <= 0.33) and profit > 8.0
        sell2 = (0.33 < s["position"] <= 0.66) and profit > 14.0
        sell3 = (s["position"] > 0.00) and profit > 22.0

        initial_pos = s["position"]
        action = None

        # --- EXECUTE BUYS ---
        if s["position"] == 0 and buy1:
            s["position"] = 0.33
            s["avg_price"] = price
            s["peak_profit"] = 0.0
            s["structure_stop"] = last_pl * 0.99  # وقف خسارة أسفل القاع المحوري السابق
            s["realized_pnl_tracker"] = []
            profit = 0.0
            action = "BUY L1 (MSS Breakout)"

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

        elif 0.32 < s["position"] < 0.5 and buy2:
            old_pos = s["position"]
            s["position"] = 0.66
            s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
            action = "BUY L2 (Re-Test)"

            active = [t for t in trades_history if t["symbol"] == name and t["status"] == "OPEN"]
            if active:
                active[-1]["second_entry"] = f"{date_str} with price {price:.2f}"
                active[-1]["last_totally_average_price"] = round(s["avg_price"], 2)

        elif 0.65 < s["position"] < 1 and buy3:
            old_pos = s["position"]
            s["position"] = 1.0
            s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
            action = "BUY L3 (Discount)"

            active = [t for t in trades_history if t["symbol"] == name and t["status"] == "OPEN"]
            if active:
                active[-1]["third_entry"] = f"{date_str} with price {price:.2f}"
                active[-1]["last_totally_average_price"] = round(s["avg_price"], 2)

        if profit > s["peak_profit"]:
            s["peak_profit"] = profit

        # --- EXECUTE SELLS / STOPS ---
        if initial_pos > 0 and s["position"] > 0:
            stop_triggered = False

            # وقف الخسارة الهيكلي (أو الوقف النسبي المحكم)
            if price < s["structure_stop"] or profit <= -5.0:
                stop_triggered = True

            # Trailing Stop ديناميكي لحماية الأرباح عند تسجيل قمة جديدة
            if s["peak_profit"] > 9.0 and (s["peak_profit"] - profit) >= 3.5:
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
            s["structure_stop"] = 0.0
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

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

with open(TRADES_FILE, "w", encoding="utf-8") as f:
    json.dump(trades_history, f, indent=2, ensure_ascii=False)

with open(STOCK_SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(stock_summaries, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("MARKET STRUCTURE SHIFT (MSS) ENGINE - BACKTEST COMPLETE")
print("=" * 60)
print(f"Total Closed Trades: {total_count}")
print(f"Win Rate:            {win_rate:.2f}%")
print(f"Total Profit:        {total_portfolio_profit:.2f}%")
print("=" * 60)

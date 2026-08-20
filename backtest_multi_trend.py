import json
import os
import pandas as pd
import numpy as np

# ============================================================
# HIGH-CONVICTION SYSTEM (V8.3 - EMA75 3-TRANCHE ENGINE)
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
# STRATEGY PARAMETERS (EMA75 & 3-TRANCHES CONFIG)
# ============================================================

EMA_PERIOD = 75

# Trend Thresholds
EMA_UP_MIN_STEP_PERCENT = 0.30
EMA_SIDE_MAX_DISTANCE_PERCENT = 1.00
EMA_DOWN_MIN_STEP_PERCENT = 1.00

# Entry RSI Thresholds (3 Buys)
RSI_BUY_1 = 55.0            # L1: دخول أول عند التراجع الصحي
RSI_BUY_2 = 48.0            # L2: دخول ثانٍ عند استمرار التهدئة
RSI_BUY_3 = 42.0            # L3: دخول ثالث عند التجميع القوي

# Exit RSI Thresholds (3 Sells)
RSI_SELL_1 = 70.0           # S1: جني أرباح أولي
RSI_SELL_2 = 75.0           # S2: جني أرباح متوسط
RSI_SELL_3 = 80.0           # S3: تصفية نهائية عند التشبع

# Stop Loss & Position Sizing
EMA_STOP_MIN_DROP_PERCENT = 1.00
TRANCHE_SIZE = 0.3333       # 3 شرائح متساوية
MAX_TRANCHES = 3

# ============================================================
# HELPER FUNCTIONS
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

        if len(df) < EMA_PERIOD + 20:
            continue

        close = df["Close"]
        df["EMA_TREND"] = close.ewm(span=EMA_PERIOD, adjust=False).mean()
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
portfolio_equity_curve = []

# ============================================================
# BACKTEST LOOP
# ============================================================

for current_date in all_dates:
    for name, df in prepared_data.items():
        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)
        if idx < EMA_PERIOD + 15:
            continue

        df_slice = df.iloc[: idx + 1]
        last = df_slice.iloc[-1]

        if df_slice[["Open", "Close", "EMA_TREND", "RSI"]].iloc[-1].isna().any():
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        price = float(last["Close"])
        rsi_val = float(last["RSI"])

        s = states[name]
        s["position"] = round(float(s["position"]), 4)

        if s["position"] == 0.0:
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0

        # --- 1. ENGINE DETECT TREND (EMA75 MESH) ---
        ema_curr = float(df_slice["EMA_TREND"].iloc[-1])
        ema_4 = float(df_slice["EMA_TREND"].iloc[-5])
        ema_8 = float(df_slice["EMA_TREND"].iloc[-9])
        ema_12 = float(df_slice["EMA_TREND"].iloc[-13])

        step1 = ((ema_8 - ema_12) / ema_12) * 100
        step2 = ((ema_4 - ema_8) / ema_8) * 100
        step3 = ((ema_curr - ema_4) / ema_4) * 100

        is_uptrend = (step1 >= EMA_UP_MIN_STEP_PERCENT) and (step2 >= EMA_UP_MIN_STEP_PERCENT) and (step3 >= EMA_UP_MIN_STEP_PERCENT)
        
        max_ema = max(ema_curr, ema_4, ema_8, ema_12)
        min_ema = min(ema_curr, ema_4, ema_8, ema_12)
        dist_percent = ((max_ema - min_ema) / min_ema) * 100
        is_sideways = dist_percent <= EMA_SIDE_MAX_DISTANCE_PERCENT

        is_sharp_downtrend = (step1 <= -EMA_DOWN_MIN_STEP_PERCENT) and (step2 <= -EMA_DOWN_MIN_STEP_PERCENT) and (step3 <= -EMA_DOWN_MIN_STEP_PERCENT)
        
        drop_from_4 = ((ema_curr - ema_4) / ema_4) * 100
        is_sharp_drop = drop_from_4 <= -EMA_STOP_MIN_DROP_PERCENT

        # --- 2. ENTRY LOGIC (3 BUY TRANCHES) ---
        buy1 = (s["position"] == 0.0) and is_uptrend and (rsi_val < RSI_BUY_1)
        buy2 = (0.30 <= s["position"] < 0.40) and (is_uptrend or is_sideways) and (rsi_val <= RSI_BUY_2) and (price < s["avg_price"])
        buy3 = (0.60 <= s["position"] < 0.70) and (is_uptrend or is_sideways) and (rsi_val <= RSI_BUY_3) and (price < s["avg_price"])

        profit = 0.0
        if s["avg_price"] > 0:
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100

        # --- 3. EXIT LOGIC (3 SELL TRANCHES) ---
        sell1 = (s["position"] > 0.60) and (rsi_val >= RSI_SELL_1) and (profit > 0)
        sell2 = (s["position"] > 0.30) and (rsi_val >= RSI_SELL_2) and (profit > 0)
        sell3 = (s["position"] > 0.00) and (rsi_val >= RSI_SELL_3)
        emergency_exit = (s["position"] > 0.0) and (is_sharp_downtrend or is_sharp_drop)

        initial_pos = s["position"]
        action = None

        # --- EXECUTE BUYS ---
        if buy1:
            s["position"] = TRANCHE_SIZE
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

        elif buy2:
            old_pos = s["position"]
            s["position"] = 0.6666
            s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
            action = "BUY L2"

            active = [t for t in trades_history if t["symbol"] == name and t["status"] == "OPEN"]
            if active:
                active[-1]["second_entry"] = f"{date_str} with price {price:.2f}"
                active[-1]["last_totally_average_price"] = round(s["avg_price"], 2)

        elif buy3:
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
            active = [t for t in trades_history if t["symbol"] == name and t["status"] == "OPEN"]

            if emergency_exit or sell3:
                action = "EMERGENCY STOP" if emergency_exit else "EXIT FULL (S3)"
                total_profit = calc_final_pnl(s["realized_pnl_tracker"], profit, s["position"])

                if active:
                    active[-1]["status"] = "CLOSED"
                    active[-1]["exit_price"] = round(price, 2)
                    active[-1]["exit_date"] = date_str
                    active[-1]["profit_pct"] = round(total_profit, 2)

                total_portfolio_profit += total_profit
                s["position"] = 0.0

            elif sell2:
                sell_amount = TRANCHE_SIZE
                s["realized_pnl_tracker"].append((sell_amount, profit))
                s["position"] = round(s["position"] - sell_amount, 4)
                action = "SELL S2"

                if active:
                    exit_log = f"{date_str}: Sold 33.3% at price {price:.2f} (Profit: {profit:+.2f}%)"
                    active[-1]["exits"].append(exit_log)

                    if s["position"] <= 0.05:
                        w_sum = sum(w for w, _ in s["realized_pnl_tracker"])
                        total_profit = sum(p * w for w, p in s["realized_pnl_tracker"]) / w_sum if w_sum > 0 else profit
                        active[-1]["status"] = "CLOSED"
                        active[-1]["exit_price"] = round(price, 2)
                        active[-1]["exit_date"] = date_str
                        active[-1]["profit_pct"] = round(total_profit, 2)
                        total_portfolio_profit += total_profit
                        s["position"] = 0.0

            elif sell1:
                sell_amount = TRANCHE_SIZE
                s["realized_pnl_tracker"].append((sell_amount, profit))
                s["position"] = round(s["position"] - sell_amount, 4)
                action = "SELL S1"

                if active:
                    exit_log = f"{date_str}: Sold 33.3% at price {price:.2f} (Profit: {profit:+.2f}%)"
                    active[-1]["exits"].append(exit_log)

                    if s["position"] <= 0.05:
                        w_sum = sum(w for w, _ in s["realized_pnl_tracker"])
                        total_profit = sum(p * w for w, p in s["realized_pnl_tracker"]) / w_sum if w_sum > 0 else profit
                        active[-1]["status"] = "CLOSED"
                        active[-1]["exit_price"] = round(price, 2)
                        active[-1]["exit_date"] = date_str
                        active[-1]["profit_pct"] = round(total_profit, 2)
                        total_portfolio_profit += total_profit
                        s["position"] = 0.0

            s["position"] = round(s["position"], 4)

        if action and s["position"] == 0.0 and ("SELL" in action or "EXIT" in action or "STOP" in action):
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["realized_pnl_tracker"] = []
            s["cycle"] += 1

    portfolio_equity_curve.append(total_portfolio_profit)

# ============================================================
# COMPUTE STATISTICS & DRAWDOWN
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
        "average_losing_trade_percent": round(avg_loss, 2),
        "maximum_drawdown_percent": round(max_drawdown, 2)
    }
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

with open(TRADES_FILE, "w", encoding="utf-8") as f:
    json.dump(trades_history, f, indent=2, ensure_ascii=False)

with open(STOCK_SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(stock_summaries, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("EMA75 3-TRANCHE ENGINE - BACKTEST COMPLETE")
print("=" * 60)
print(f"Total Closed Trades: {total_count}")
print(f"Win Rate:            {win_rate:.2f}%")
print(f"Total Profit:        {total_portfolio_profit:.2f}%")
print(f"Max Drawdown:        {max_drawdown:.2f}%")
print("=" * 60)

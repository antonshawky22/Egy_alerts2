import json
import os
import pandas as pd
import numpy as np

# ============================================================
# BACKTEST - EGX LADDER CYCLE SYSTEM (v3.4 FULLY AUDITED INTEGRATION)
# ============================================================

DB_FILE = "egx_history_database_v2.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

EGX30_KEY = "EGX30"

# ============================================================
# RSI - WILDER / EWM
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

# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(f"Database file not found: {DB_FILE}")

with open(DB_FILE, "r", encoding="utf-8") as f:
    raw_database = json.load(f)

print("Historical database loaded successfully.")

symbols = list(raw_database.keys())

# ============================================================
# PREPARE DATA
# ============================================================

prepared_data = {}

for symbol in symbols:
    content = raw_database.get(symbol, {})
    if "data" not in content or "columns" not in content:
        continue
    try:
        df = pd.DataFrame.from_dict(content["data"], orient="index", columns=content["columns"])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)

        if len(df) < 40:
            continue

        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA30"] = df["Close"].ewm(span=30, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA75"] = df["Close"].ewm(span=75, adjust=False).mean()
        df["RSI14"] = rsi(df["Close"], 14)

        prepared_data[symbol] = df
    except Exception as e:
        print(f"Failed preparing {symbol}: {e}")

print(f"Prepared {len(prepared_data)} symbols for backtest.")

# ============================================================
# ALL TRADING DATES
# ============================================================

all_dates = set()
for symbol, df in prepared_data.items():
    if symbol == EGX30_KEY:
        continue
    all_dates.update(df.index)

all_dates = sorted(all_dates)

if not all_dates:
    raise ValueError("No trading dates available.")

print(f"Backtest period: {all_dates[0].strftime('%Y-%m-%d')} -> {all_dates[-1].strftime('%Y-%m-%d')}")

# ============================================================
# STATE & TRACKING
# ============================================================

states = {}
for symbol in prepared_data:
    if symbol == EGX30_KEY:
        continue
    states[symbol] = {
        "cycle": 1,
        "position": 0.0,
        "avg_price": 0.0,
        "peak_profit": 0.0,
        "realized_pnl_tracker": [],
        "active_trade": None
    }

trades_history = []
closed_profit_percent = 0.0
equity_curve = []
peak_equity = 0.0
max_drawdown = 0.0

# ============================================================
# MAIN BACKTEST LOOP
# ============================================================

for current_date in all_dates:
    for symbol, df in prepared_data.items():
        if symbol == EGX30_KEY or current_date not in df.index:
            continue

        current_index = df.index.get_loc(current_date)
        if current_index < 40:
            continue

        # حماية الشمعات الفارغة
        if df[["Open", "Close", "EMA75", "RSI14"]].iloc[current_index].isna().any():
            continue

        row = df.iloc[current_index]
        price = float(row["Close"])
        rsi_val = float(row["RSI14"])

        s = states[symbol]
        
        s["position"] = round(float(s["position"]), 2)
        if s["position"] == 0.0:
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["realized_pnl_tracker"] = []

        # ----------------------------------------------------
        # FIlTERS
        # ----------------------------------------------------
        lookback = min(current_index + 1, 80)
        lowest_80 = float(df["Low"].iloc[current_index - lookback + 1 : current_index + 1].min())
        highest_80 = float(df["High"].iloc[current_index - lookback + 1 : current_index + 1].max())
        
        run_up_percent = ((highest_80 - lowest_80) / lowest_80) * 100 if lowest_80 > 0 else 0.0
        safe_to_buy = run_up_percent <= 60.0

        if current_index >= 3:
            gap1 = ((df["Open"].iloc[current_index] - df["Close"].iloc[current_index - 1]) / df["Close"].iloc[current_index - 1]) * 100
            gap2 = ((df["Open"].iloc[current_index - 1] - df["Close"].iloc[current_index - 2]) / df["Close"].iloc[current_index - 2]) * 100
            gap3 = ((df["Open"].iloc[current_index - 2] - df["Close"].iloc[current_index - 3]) / df["Close"].iloc[current_index - 3]) * 100
            no_gap_down = (gap1 > -3.0) and (gap2 > -3.0) and (gap3 > -3.0)
        else:
            no_gap_down = True

        near_ema = price <= df["EMA75"].iloc[current_index] * 1.08

        if current_index >= 13:
            ema75_now = df["EMA75"].iloc[current_index]
            ema75_4 = df["EMA75"].iloc[current_index - 4]
            ema75_8 = df["EMA75"].iloc[current_index - 8]
            ema75_12 = df["EMA75"].iloc[current_index - 12]

            ema_up = (ema75_now >= ema75_4 * 1.003) and (ema75_4 >= ema75_8 * 1.003) and (ema75_8 >= ema75_12 * 1.003)
            
            ema_vals = [ema75_now, ema75_4, ema75_8, ema75_12]
            ema_sideways = ((max(ema_vals) - min(ema_vals)) / min(ema_vals)) <= 0.01

            ema_down = (ema75_now < ema75_4) and (ema75_4 < ema75_8)
        else:
            ema_up = False
            ema_sideways = True
            ema_down = False

        # ----------------------------------------------------
        # STRATEGY CONDITIONS
        # ----------------------------------------------------
        buy1 = safe_to_buy and no_gap_down and near_ema and ema_up and rsi_val <= 55
        buy2 = safe_to_buy and no_gap_down and near_ema and ema_sideways and rsi_val <= 45
        buy3 = safe_to_buy and no_gap_down and near_ema and ema_sideways and rsi_val <= 38

        profit = 0.0
        if s["avg_price"] > 0:
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100

        sell1 = s["position"] > 0.70 and rsi_val >= 68 and profit > 3.0
        sell2 = 0.30 < s["position"] <= 0.70 and rsi_val >= 74 and profit > 5.0
        sell3 = s["position"] > 0.00 and rsi_val >= 80 and profit > 7.0

        initial_pos = s["position"]
        date_str = current_date.strftime("%Y-%m-%d")

        # ----------------------------------------------------
        # 🟢 BUY EXECUTION
        # ----------------------------------------------------
        if s["position"] == 0 and buy1:
            s["position"] = 0.33
            s["avg_price"] = price
            s["peak_profit"] = 0.0
            s["realized_pnl_tracker"] = []

            s["active_trade"] = {
                "symbol": symbol,
                "cycle": s["cycle"],
                "first_entry": f"{date_str} with price {price:.2f}",
                "second_entry": None,
                "third_entry": None,
                "last_totally_average_price": round(price, 2),
                "exits": []
            }

        elif 0.32 < s["position"] < 0.5 and buy2 and price < s["avg_price"] * 0.97:
            old_pos = s["position"]
            s["position"] = 0.66
            s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100

            if s["active_trade"]:
                s["active_trade"]["second_entry"] = f"{date_str} with price {price:.2f}"
                s["active_trade"]["last_totally_average_price"] = round(s["avg_price"], 2)

        elif 0.65 < s["position"] < 1.0 and buy3 and price < s["avg_price"] * 0.96:
            old_pos = s["position"]
            s["position"] = 1.0
            s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
            profit = ((price - s["avg_price"]) / s["avg_price"]) * 100

            if s["active_trade"]:
                s["active_trade"]["third_entry"] = f"{date_str} with price {price:.2f}"
                s["active_trade"]["last_totally_average_price"] = round(s["avg_price"], 2)

        if profit > s["peak_profit"]:
            s["peak_profit"] = profit

        # ----------------------------------------------------
        # 🔴 SELL EXECUTION
        # ----------------------------------------------------
        if initial_pos > 0 and s["position"] > 0:
            drop_from_ema = ((df["EMA75"].iloc[current_index] - price) / df["EMA75"].iloc[current_index]) * 100
            stop_triggered = (drop_from_ema >= 1.00) or ema_down

            if s["peak_profit"] > 10 and (s["peak_profit"] - profit) >= 4:
                stop_triggered = True

            def calc_final_pnl(current_p, current_w):
                temp_tracker = list(s["realized_pnl_tracker"]) + [(current_w, current_p)]
                w_sum = sum(w for w, _ in temp_tracker)
                return sum(p * w for w, p in temp_tracker) / w_sum if w_sum > 0 else current_p

            close_trade = False
            total_trade_pnl = 0.0

            # 1️⃣ STOP LOSS
            if stop_triggered:
                total_trade_pnl = calc_final_pnl(profit, s["position"])
                s["position"] = 0.0
                close_trade = True

            # 2️⃣ FULL EXIT
            elif sell3:
                total_trade_pnl = calc_final_pnl(profit, s["position"])
                s["position"] = 0.0
                close_trade = True

            # 3️⃣ PARTIAL SELL 2
            elif sell2:
                sell_amount = min(0.33, s["position"])
                s["realized_pnl_tracker"].append((sell_amount, profit))
                s["position"] = round(s["position"] - sell_amount, 2)

                if s["active_trade"]:
                    exit_log = f"{date_str}: Sold {sell_amount*100:.0f}% at price {price:.2f} (Profit: {profit:+.2f}%)"
                    s["active_trade"]["exits"].append(exit_log)

                if s["position"] == 0.0:
                    total_trade_pnl = calc_final_pnl(profit, 0.0)
                    close_trade = True

            # 4️⃣ PARTIAL SELL 1
            elif sell1:
                sell_amount = min(0.33, s["position"])
                s["realized_pnl_tracker"].append((sell_amount, profit))
                s["position"] = round(s["position"] - sell_amount, 2)

                if s["active_trade"]:
                    exit_log = f"{date_str}: Sold {sell_amount*100:.0f}% at price {price:.2f} (Profit: {profit:+.2f}%)"
                    s["active_trade"]["exits"].append(exit_log)

                if s["position"] == 0.0:
                    total_trade_pnl = calc_final_pnl(profit, 0.0)
                    close_trade = True

            # CLOSE TRADE LOGIC
            if close_trade and s["active_trade"]:
                trade_record = s["active_trade"]
                trade_record["status"] = "CLOSED"
                trade_record["exit_price"] = round(price, 2)
                trade_record["exit_date"] = date_str
                trade_record["profit_pct"] = round(total_trade_pnl, 2)

                trades_history.append(trade_record)
                closed_profit_percent += total_trade_pnl

                s["active_trade"] = None
                s["avg_price"] = 0.0
                s["peak_profit"] = 0.0
                s["realized_pnl_tracker"] = []
                s["cycle"] += 1

            s["position"] = round(s["position"], 2)

    # TRACK EQUITY & DRAWDOWN
    equity_curve.append(closed_profit_percent)
    if closed_profit_percent > peak_equity:
        peak_equity = closed_profit_percent
    drawdown = peak_equity - closed_profit_percent
    if drawdown > max_drawdown:
        max_drawdown = drawdown

# ============================================================
# RESULTS GENERATION & STATS
# ============================================================

total_trades = len(trades_history)
winning_trades = [t for t in trades_history if t["profit_pct"] > 0]
losing_trades = [t for t in trades_history if t["profit_pct"] <= 0]

wins = len(winning_trades)
losses = len(losing_trades)
win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
average_profit = np.mean([t["profit_pct"] for t in winning_trades]) if winning_trades else 0
average_loss = np.mean([t["profit_pct"] for t in losing_trades]) if losing_trades else 0

results = {
    "statistics": {
        "total_trades": total_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_percent": round(win_rate, 2),
        "total_profit_percent": round(closed_profit_percent, 2),
        "average_winning_trade_percent": round(float(average_profit), 2),
        "average_losing_trade_percent": round(float(average_loss), 2),
        "maximum_drawdown_percent": round(max_drawdown, 2)
    }
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open(TRADES_FILE, "w", encoding="utf-8") as f:
    json.dump(trades_history, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("EGX LADDER SYSTEM v3.4 - BACKTEST COMPLETE")
print("=" * 60)
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total Profit: {closed_profit_percent:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("=" * 60)

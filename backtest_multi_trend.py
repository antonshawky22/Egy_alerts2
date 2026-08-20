print("EGX LADDER CYCLE SYSTEM - BACKTEST ENGINE (v3.5 Radical Stop Synchronized)")

import json
import os
import numpy as np
import pandas as pd

DB_FILE = "egx_history_database_v2.json"

# قائمة الأسهم
symbols = [
    "OLFI", "EMFD", "ETEL", "EAST", "EFIH", "ABUK", "OIH", "SWDY", "ISPH",
    "ATQA", "MTIE", "HRHO", "ORWE", "JUFO", "DSCW", "SUGR", "ELSH", "RMDA",
    "RAYA", "EEII", "MPCO", "GBCO", "TMGH", "ORHD", "AMOC", "FWRY", "COMI",
    "ADIB", "PHDC", "MCQE", "SKPC", "EGAL"
]

# تحميل قاعدة البيانات المحلية
raw_database = {}
if os.path.exists(DB_FILE):
    try:
        with open(DB_FILE, "r") as f:
            raw_database = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load database file: {e}")
else:
    print(f"⚠️ Database file '{DB_FILE}' not found!")


def fetch_local_data(name):
    try:
        if name not in raw_database:
            return None
        content = raw_database[name]
        if "columns" in content and "data" in content:
            df_temp = pd.DataFrame.from_dict(
                content["data"], orient="index", columns=content["columns"]
            )
            df_temp.index.name = "Date"
            df_temp.index = pd.to_datetime(df_temp.index)
            df_temp = df_temp.sort_index(ascending=True)
            return df_temp
        return None
    except Exception as e:
        return None


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


all_trades = []

for name in symbols:
    df = fetch_local_data(name)
    if df is None or len(df) < 40:
        continue

    close = df["Close"]
    df["EMA75"] = close.ewm(span=75, adjust=False).mean()
    df["RSI"] = rsi(close)

    # حساب مؤشرات الفلاتر
    lookback = 80
    df["lowest_80"] = df["Low"].rolling(window=lookback, min_periods=40).min()
    df["highest_80"] = df["High"].rolling(window=lookback, min_periods=40).max()
    df["run_up"] = ((df["highest_80"] - df["lowest_80"]) / df["lowest_80"]) * 100

    df["gap1"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)) * 100
    df["gap2"] = ((df["Open"].shift(1) - df["Close"].shift(2)) / df["Close"].shift(2)) * 100
    df["gap3"] = ((df["Open"].shift(2) - df["Close"].shift(3)) / df["Close"].shift(3)) * 100

    # حالات السهم في الباك تست
    cycle = 1
    position = 0.0
    avg_price = 0.0
    peak_profit = 0.0
    realized_pnl_tracker = []
    current_trade = None

    for i in range(15, len(df)):
        sub_df = df.iloc[: i + 1]
        last = sub_df.iloc[-1]

        if sub_df[["Open", "Close", "EMA75", "RSI"]].iloc[-1].isna().any():
            continue

        current_date = str(sub_df.index[-1].strftime("%Y-%m-%d"))
        price = float(last["Close"])
        rsi_val = float(last["RSI"])

        safe_to_buy = float(last["run_up"]) <= 60.0
        no_gap_down = (float(last["gap1"]) > -3.0) and (float(last["gap2"]) > -3.0) and (float(last["gap3"]) > -3.0)
        near_ema = price <= float(last["EMA75"]) * 1.08

        ema75_now = float(sub_df["EMA75"].iloc[-1])
        ema75_4 = float(sub_df["EMA75"].iloc[-5])
        ema75_8 = float(sub_df["EMA75"].iloc[-9])
        ema75_12 = float(sub_df["EMA75"].iloc[-13])

        ema_up = (
            ema75_now >= ema75_4 * 1.003
            and ema75_4 >= ema75_8 * 1.003
            and ema75_8 >= ema75_12 * 1.003
        )

        ema_vals = [ema75_now, ema75_4, ema75_8, ema75_12]
        ema_sideways = ((max(ema_vals) - min(ema_vals)) / min(ema_vals)) <= 0.01

        buy1 = safe_to_buy and no_gap_down and near_ema and ema_up and rsi_val <= 55
        buy2 = safe_to_buy and no_gap_down and near_ema and ema_sideways and rsi_val <= 45
        buy3 = safe_to_buy and no_gap_down and near_ema and ema_sideways and rsi_val <= 38

        profit = 0.0
        if avg_price > 0:
            profit = ((price - avg_price) / avg_price) * 100

        sell1 = position > 0.70 and rsi_val >= 68 and profit > 3.0
        sell2 = 0.30 < position <= 0.70 and rsi_val >= 74 and profit > 5.0
        sell3 = position > 0.00 and rsi_val >= 80 and profit > 7.0

        initial_pos = position

        # 🟢 تنفيذ أومـر الـشـراء
        if position == 0 and buy1:
            position = 0.33
            avg_price = price
            peak_profit = 0.0
            realized_pnl_tracker = []
            profit = 0.0

            current_trade = {
                "symbol": name,
                "cycle": cycle,
                "status": "OPEN",
                "entry_date": current_date,
                "entry_price": round(price, 2),
                "exits": [],
                "exit_price": None,
                "exit_date": None,
                "profit_pct": None,
                "type": "NORMAL"
            }

        elif 0.32 < position < 0.5 and buy2 and price < avg_price * 0.97:
            old_pos = position
            position = 0.66
            avg_price = update_avg(avg_price, old_pos, price, position)
            profit = ((price - avg_price) / avg_price) * 100

        elif 0.65 < position < 1 and buy3 and price < avg_price * 0.96:
            old_pos = position
            position = 1.0
            avg_price = update_avg(avg_price, old_pos, price, position)
            profit = ((price - avg_price) / avg_price) * 100

        if profit > peak_profit:
            peak_profit = profit

        # 🔴 تنفيذ أومـر الـبـيـع وإغلاق الصفقات
        if initial_pos > 0 and position > 0:

            # 1. الوقف الجذري: هبوط القيم الأربعة لـ EMA75
            ema_down_radical = (
                ema75_now <= ema75_4 * 0.997
                and ema75_4 <= ema75_8 * 0.997
                and ema75_8 <= ema75_12 * 0.997
            )

            # 2. الوقف الديناميكي: حماية الأرباح عند التراجع من القمة
            trailing_stop = (peak_profit > 10 and (peak_profit - profit) >= 4)

            stop_triggered = ema_down_radical or trailing_stop

            def calc_final_pnl(current_p, current_w):
                temp_tracker = list(realized_pnl_tracker) + [(current_w, current_p)]
                w_sum = sum(w for w, _ in temp_tracker)
                return sum(p * w for w, p in temp_tracker) / w_sum if w_sum > 0 else current_p

            if stop_triggered:
                if current_trade:
                    total_profit = calc_final_pnl(profit, position)
                    current_trade["status"] = "CLOSED"
                    current_trade["exit_price"] = round(price, 2)
                    current_trade["exit_date"] = current_date
                    current_trade["profit_pct"] = round(total_profit, 2)
                    current_trade["type"] = "STOP_LOSS" if ema_down_radical else "TRAILING_STOP"
                    all_trades.append(current_trade)
                    current_trade = None

                position = 0.0

            elif sell3:
                if current_trade:
                    total_profit = calc_final_pnl(profit, position)
                    current_trade["status"] = "CLOSED"
                    current_trade["exit_price"] = round(price, 2)
                    current_trade["exit_date"] = current_date
                    current_trade["profit_pct"] = round(total_profit, 2)
                    current_trade["type"] = "FULL_TARGET"
                    all_trades.append(current_trade)
                    current_trade = None

                position = 0.0

            elif sell2 or sell1:
                sell_amount = min(0.33, position)
                realized_pnl_tracker.append((sell_amount, profit))
                position = round(position - sell_amount, 2)

                if position == 0.0 and current_trade:
                    total_profit = calc_final_pnl(profit, 0.0)
                    current_trade["status"] = "CLOSED"
                    current_trade["exit_price"] = round(price, 2)
                    current_trade["exit_date"] = current_date
                    current_trade["profit_pct"] = round(total_profit, 2)
                    current_trade["type"] = "PARTIAL_TARGET"
                    all_trades.append(current_trade)
                    current_trade = None

            position = round(position, 2)

            if position == 0.0:
                avg_price = 0.0
                peak_profit = 0.0
                realized_pnl_tracker = []
                cycle += 1

# ==========================================
# 📊 طباعة تقرير نتائج الباك تست
# ==========================================
if all_trades:
    trades_df = pd.DataFrame(all_trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["profit_pct"] > 0])
    losing_trades = len(trades_df[trades_df["profit_pct"] <= 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_profit = trades_df["profit_pct"].mean()
    total_cum_profit = trades_df["profit_pct"].sum()

    print("\n" + "="*45)
    print("      📊 BACKTEST RESULTS REPORT (v3.5)      ")
    print("="*45)
    print(f"Total Trades Executed:  {total_trades}")
    print(f"Winning Trades:         {winning_trades} ({win_rate:.1f}%)")
    print(f"Losing Trades:          {losing_trades}")
    print(f"Average Profit/Trade:   {avg_profit:.2f}%")
    print(f"Cumulative Return:      {total_cum_profit:.2f}%")
    print("="*45)
else:
    print("⚠️ No trades were executed in this period.")

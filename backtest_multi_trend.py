print("EGX LADDER CYCLE SYSTEM - ACCURATE BACKTEST ENGINE (v3.6 Optimized)")

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

    lookback = 80
    df["lowest_80"] = df["Low"].rolling(window=lookback, min_periods=40).min()
    df["highest_80"] = df["High"].rolling(window=lookback, min_periods=40).max()
    df["run_up"] = ((df["highest_80"] - df["lowest_80"]) / df["lowest_80"]) * 100

    df["gap1"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)) * 100
    df["gap2"] = ((df["Open"].shift(1) - df["Close"].shift(2)) / df["Close"].shift(2)) * 100
    df["gap3"] = ((df["Open"].shift(2) - df["Close"].shift(3)) / df["Close"].shift(3)) * 100

    cycle = 1
    position = 0.0
    avg_price = 0.0
    peak_profit = 0.0
    
    # تتبع الخروج الجزئي وحساب الربح الموزون للصفقة الحالية
    realized_exits = []  # قائمة لتخزين (حجم الشريحة المباعة, نسبة ربح الشريحة)
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

        # 🎯 فلتر دخول محدد ومركز (تم خفض RSI للشريحة الأولى لتقليل الصفقات العشوائية)
        buy1 = safe_to_buy and no_gap_down and near_ema and ema_up and rsi_val <= 48
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
            realized_exits = []

            current_trade = {
                "symbol": name,
                "cycle": cycle,
                "entry_date": current_date,
                "entry_price": round(price, 2),
                "max_position": 0.33,
                "exit_reason": None
            }

        elif 0.32 < position < 0.5 and buy2 and price < avg_price * 0.97:
            old_pos = position
            position = 0.66
            avg_price = update_avg(avg_price, old_pos, price, position)
            profit = ((price - avg_price) / avg_price) * 100
            if current_trade:
                current_trade["max_position"] = max(current_trade["max_position"], 0.66)

        elif 0.65 < position < 1 and buy3 and price < avg_price * 0.96:
            old_pos = position
            position = 1.0
            avg_price = update_avg(avg_price, old_pos, price, position)
            profit = ((price - avg_price) / avg_price) * 100
            if current_trade:
                current_trade["max_position"] = 1.0

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

            # دالة حساب الربح الصافي الموزون الحقيقي للصفقة
            def get_final_weighted_pnl(final_exit_profit, remaining_pos):
                temp_exits = list(realized_exits)
                if remaining_pos > 0:
                    temp_exits.append((remaining_pos, final_exit_profit))
                
                total_weight = sum(w for w, _ in temp_exits)
                if total_weight > 0:
                    return sum(w * p for w, p in temp_exits) / total_weight
                return final_exit_profit

            # 🛑 1. خروج كلي بسبب الستوب لوس
            if stop_triggered:
                final_pnl = get_final_weighted_pnl(profit, position)
                if current_trade:
                    current_trade["exit_date"] = current_date
                    current_trade["exit_price"] = round(price, 2)
                    current_trade["profit_pct"] = round(final_pnl, 2)
                    current_trade["exit_reason"] = "STOP_LOSS_RADICAL" if ema_down_radical else "TRAILING_PROFIT_STOP"
                    all_trades.append(current_trade)
                    current_trade = None

                position = 0.0

            # 🚨 2. خروج كلي لتحقيق الهدف الكامل
            elif sell3:
                final_pnl = get_final_weighted_pnl(profit, position)
                if current_trade:
                    current_trade["exit_date"] = current_date
                    current_trade["exit_price"] = round(price, 2)
                    current_trade["profit_pct"] = round(final_pnl, 2)
                    current_trade["exit_reason"] = "FULL_TARGET"
                    all_trades.append(current_trade)
                    current_trade = None

                position = 0.0

            # 🔴 3. بيع جزئي L2 أو L1 (تأمين الأرباح)
            elif sell2 or sell1:
                sell_amount = min(0.33, position)
                realized_exits.append((sell_amount, profit))
                position = round(position - sell_amount, 2)

                # إذا انتهت الكمية تماماً بعد البيع الجزئي
                if position == 0.0 and current_trade:
                    final_pnl = get_final_weighted_pnl(profit, 0.0)
                    current_trade["exit_date"] = current_date
                    current_trade["exit_price"] = round(price, 2)
                    current_trade["profit_pct"] = round(final_pnl, 2)
                    current_trade["exit_reason"] = "PARTIAL_TARGETS_COMPLETE"
                    all_trades.append(current_trade)
                    current_trade = None

            position = round(position, 2)

            if position == 0.0:
                avg_price = 0.0
                peak_profit = 0.0
                realized_exits = []
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
    
    avg_trade = trades_df["profit_pct"].mean()
    avg_win = trades_df[trades_df["profit_pct"] > 0]["profit_pct"].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df["profit_pct"] <= 0]["profit_pct"].mean() if losing_trades > 0 else 0
    total_cum_profit = trades_df["profit_pct"].sum()

    # حساب التراجع الأقصى Drawdown للصفقة التراكمية
    trades_df["cum_pnl"] = trades_df["profit_pct"].cumsum()
    trades_df["peak"] = trades_df["cum_pnl"].cummax()
    trades_df["drawdown"] = trades_df["cum_pnl"] - trades_df["peak"]
    max_drawdown = trades_df["drawdown"].min()

    print("\n" + "="*50)
    print("      📊 ACCURATE BACKTEST REPORT (v3.6)      ")
    print("="*50)
    print(f"Total Completed Trades:   {total_trades}")
    print(f"Winning Trades:           {winning_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades:            {losing_trades}")
    print(f"Average Win / Trade:      +{avg_win:.2f}%")
    print(f"Average Loss / Trade:     {avg_loss:.2f}%")
    print(f"Average PnL / Trade:      {avg_trade:.2f}%")
    print(f"Cumulative Profit:        +{total_cum_profit:.2f}%")
    print(f"Max Cumulative Drawdown:  {max_drawdown:.2f}%")
    print("="*50)
else:
    print("⚠️ No trades were executed in this period.")

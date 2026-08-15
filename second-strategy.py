print("EGX LADDER CYCLE SYSTEM - DATABASE SOURCED (v3.4 Fully Audited)")

import json
import os
import time
import numpy as np
import pandas as pd
import requests

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        print(text)
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)


# قائمة الأسهم
symbols = {
    "OLFI": "OLFI", "EMFD": "EMFD", "ETEL": "ETEL", "EAST": "EAST",
    "EFIH": "EFIH", "ABUK": "ABUK", "OIH": "OIH", "SWDY": "SWDY", "ISPH": "ISPH",
    "ATQA": "ATQA", "MTIE": "MTIE", "HRHO": "HRHO", "ORWE": "ORWE",
    "JUFO": "JUFO", "DSCW": "DSCW", "SUGR": "SUGR", "ELSH": "ELSH", "RMDA": "RMDA",
    "RAYA": "RAYA", "EEII": "EEII", "MPCO": "MPCO", "GBCO": "GBCO", "TMGH": "TMGH",
    "ORHD": "ORHD", "AMOC": "AMOC", "FWRY": "FWRY", "COMI": "COMI", "ADIB": "ADIB",
    "PHDC": "PHDC", "MCQE": "MCQE", "SKPC": "SKPC", "EGAL": "EGAL"
}

STATE_FILE = "last_signals_strat2.json"
DB_FILE = "egx_history_database_v2.json"
TRADES_FILE = "trades2.json"

# تحميل ملف الحالة
try:
    with open(STATE_FILE, "r") as f:
        state_data = json.load(f)
except Exception:
    state_data = {}

# تحميل ملف سجل الصفقات
try:
    with open(TRADES_FILE, "r") as f:
        trades_history = json.load(f)
except Exception:
    trades_history = {}

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
    """قراءة البيانات من قاعدة البيانات المحملة بالذاكرة."""
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
        print(f"💥 Error processing data for {name}: {e}")
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


def format_alert(title, name, price, position, avg, rsi_val, cycle, profit):
    return (
        f"{title} | {name}\n\n"
        f"💰 Price: {price:.2f}\n"
        f"📊 Position: {position*100:.0f}%\n"
        f"📉 Avg: {avg:.2f}\n\n"
        f"📈 RSI: {rsi_val:.1f}\n"
        f"🔁 Cycle: {cycle}\n"
        f"💵 P/L: {profit:.2f}%"
    )


alerts = []

for name, ticker in symbols.items():

    df = fetch_local_data(name)
    if df is None or len(df) < 40:
        continue

    close = df["Close"]
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA30"] = close.ewm(span=30, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA75"] = close.ewm(span=75, adjust=False).mean()
    df["RSI"] = rsi(close)

    last = df.iloc[-1]
    
    # حماية من القيم الفارغة في أحدث شمعة
    if df[["Open", "Close", "EMA75", "RSI"]].iloc[-1].isna().any():
        continue

    current_date = str(df.index[-1].strftime("%Y-%m-%d"))
    price = float(last["Close"])
    rsi_val = float(last["RSI"])

    if name not in state_data:
        state_data[name] = {
            "cycle": 1,
            "position": 0.0,
            "avg_price": 0.0,
            "peak_profit": 0.0,
            "realized_pnl_tracker": []
        }

    s = state_data[name]

    # حماية الدقة العائمة للأرقام (Float Precision)
    s["position"] = round(float(s.get("position", 0.0)), 2)
    s["avg_price"] = float(s.get("avg_price", 0.0))
    s["peak_profit"] = float(s.get("peak_profit", 0.0))
    s["cycle"] = int(s.get("cycle", 1))
    if "realized_pnl_tracker" not in s:
        s["realized_pnl_tracker"] = []

    if s["position"] == 0.0:
        s["avg_price"] = 0.0
        s["peak_profit"] = 0.0

    # فلتر منع قمم الصعود الصاروخي
    lookback = min(len(df), 80)
    lowest_80 = float(df["Low"].tail(lookback).min())
    highest_80 = float(df["High"].tail(lookback).max())
    
    run_up_percent = ((highest_80 - lowest_80) / lowest_80) * 100 if lowest_80 > 0 else 0.0
    safe_to_buy = run_up_percent <= 60.0

    # فلتر فجوة الهبوط: فحص آخر 3 جلسات متتالية
    gap1 = ((df["Open"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100
    gap2 = ((df["Open"].iloc[-2] - df["Close"].iloc[-3]) / df["Close"].iloc[-3]) * 100
    gap3 = ((df["Open"].iloc[-3] - df["Close"].iloc[-4]) / df["Close"].iloc[-4]) * 100

    no_gap_down = (gap1 > -3.0) and (gap2 > -3.0) and (gap3 > -3.0)

    ema_up = (
        df["EMA75"].iloc[-1] > df["EMA75"].iloc[-5]
        and df["EMA75"].iloc[-5] > df["EMA75"].iloc[-10]
        and df["EMA75"].iloc[-1] > df["EMA75"].iloc[-10] * 1.002
        and price <= df["EMA75"].iloc[-1] * 1.08
    )
    
    buy1 = safe_to_buy and ema_up and no_gap_down and rsi_val <= 58
    buy2 = safe_to_buy and ema_up and no_gap_down and rsi_val <= 50
    buy3 = safe_to_buy and ema_up and no_gap_down and rsi_val <= 42

    # حساب الربح اللحظي الحالي
    profit = 0.0
    if s["avg_price"] > 0:
        profit = ((price - s["avg_price"]) / s["avg_price"]) * 100

    sell1 = s["position"] > 0.70 and rsi_val >= 68 and profit > 3.0
    sell2 = 0.30 < s["position"] <= 0.70 and rsi_val >= 74 and profit > 5.0
    sell3 = s["position"] > 0.00 and rsi_val >= 80 and profit > 7.0
    action = None

    if name not in trades_history:
        trades_history[name] = []

    initial_pos = s["position"]

    # ==========================================
    # 🟢 تنفيذ أومـر الـشـراء وتسجيل الصفقات
    # ==========================================
    if s["position"] == 0 and buy1:
        s["position"] = 0.33
        s["avg_price"] = price
        s["peak_profit"] = 0.0
        s["realized_pnl_tracker"] = []
        profit = 0.0
        action = "🟢 BUY L1"

        new_trade = {
            "symbol": name,
            "cycle": s["cycle"],
            "status": "OPEN",
            "first_entry": f"{current_date} with price {price:.2f}",
            "second_entry": None,
            "third_entry": None,
            "last_totally_average_price": round(price, 2),
            "exits": [],
            "exit_price": None,
            "exit_date": None,
            "profit_pct": None
        }
        trades_history[name].append(new_trade)

    elif 0.32 < s["position"] < 0.5 and buy2 and price < s["avg_price"] * 0.97:
        old_pos = s["position"]
        s["position"] = 0.66
        s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
        profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
        action = "🟢 BUY L2"

        if trades_history[name] and trades_history[name][-1].get("status") == "OPEN":
            active_trade = trades_history[name][-1]
            active_trade["second_entry"] = f"{current_date} with price {price:.2f}"
            active_trade["last_totally_average_price"] = round(s["avg_price"], 2)

    elif 0.65 < s["position"] < 1 and buy3 and price < s["avg_price"] * 0.96:
        old_pos = s["position"]
        s["position"] = 1.0
        s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
        profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
        action = "🟢 BUY L3"

        if trades_history[name] and trades_history[name][-1].get("status") == "OPEN":
            active_trade = trades_history[name][-1]
            active_trade["third_entry"] = f"{current_date} with price {price:.2f}"
            active_trade["last_totally_average_price"] = round(s["avg_price"], 2)

    if profit > s["peak_profit"]:
        s["peak_profit"] = profit

    # ==========================================
    # 🔴 تنفيذ أومـر الـبـيـع وإغلاق الصفقات
    # ==========================================
    if initial_pos > 0 and s["position"] > 0:

        stop_triggered = False

        if s["position"] <= 0.33 and profit <= -8:
            stop_triggered = True
        elif s["position"] <= 0.66 and profit <= -5:
            stop_triggered = True
        elif s["position"] == 1.0 and profit <= -4:
            stop_triggered = True

        if s["peak_profit"] > 10 and (s["peak_profit"] - profit) >= 4:
            stop_triggered = True

        # دالة مساعدة لحساب إجمالي الربح الموزون
        def calc_final_pnl(current_p, current_w):
            temp_tracker = list(s["realized_pnl_tracker"]) + [(current_w, current_p)]
            w_sum = sum(w for w, _ in temp_tracker)
            return sum(p * w for w, p in temp_tracker) / w_sum if w_sum > 0 else current_p

        # 1️⃣ إغلاق كلي (وقف خسارة)
        if stop_triggered:
            action = "🛑 STOP LOSS"
            
            if trades_history[name] and trades_history[name][-1].get("status") == "OPEN":
                active_trade = trades_history[name][-1]
                total_profit = calc_final_pnl(profit, s["position"])

                active_trade["status"] = "CLOSED"
                active_trade["exit_price"] = round(price, 2)
                active_trade["exit_date"] = current_date
                active_trade["profit_pct"] = round(total_profit, 2)

            s["position"] = 0.0

        # 2️⃣ إغلاق كلي (تارجت كامل)
        elif sell3:
            action = "🚨 EXIT FULL"

            if trades_history[name] and trades_history[name][-1].get("status") == "OPEN":
                active_trade = trades_history[name][-1]
                total_profit = calc_final_pnl(profit, s["position"])

                active_trade["status"] = "CLOSED"
                active_trade["exit_price"] = round(price, 2)
                active_trade["exit_date"] = current_date
                active_trade["profit_pct"] = round(total_profit, 2)

            s["position"] = 0.0

        # 3️⃣ بيع جزئي مستوى ثاني (33%)
        elif sell2:
            sell_amount = min(0.33, s["position"])
            s["realized_pnl_tracker"].append((sell_amount, profit))

            s["position"] = round(s["position"] - sell_amount, 2)
            action = "🔴 SELL L2 (33%)"
            
            if trades_history[name] and trades_history[name][-1].get("status") == "OPEN":
                active_trade = trades_history[name][-1]
                exit_log = f"{current_date}: Sold {sell_amount*100:.0f}% at price {price:.2f} (Profit: {profit:+.2f}%)"
                
                if "exits" not in active_trade:
                    active_trade["exits"] = []
                active_trade["exits"].append(exit_log)

                if s["position"] == 0.0:
                    w_sum = sum(w for w, _ in s["realized_pnl_tracker"])
                    total_profit = sum(p * w for w, p in s["realized_pnl_tracker"]) / w_sum if w_sum > 0 else profit
                    active_trade["status"] = "CLOSED"
                    active_trade["exit_price"] = round(price, 2)
                    active_trade["exit_date"] = current_date
                    active_trade["profit_pct"] = round(total_profit, 2)

        # 4️⃣ بيع جزئي مستوى أول (33%)
        elif sell1:
            sell_amount = min(0.33, s["position"])
            s["realized_pnl_tracker"].append((sell_amount, profit))

            s["position"] = round(s["position"] - sell_amount, 2)
            action = "🔴 SELL L1 (33%)"
            
            if trades_history[name] and trades_history[name][-1].get("status") == "OPEN":
                active_trade = trades_history[name][-1]
                exit_log = f"{current_date}: Sold {sell_amount*100:.0f}% at price {price:.2f} (Profit: {profit:+.2f}%)"
                
                if "exits" not in active_trade:
                    active_trade["exits"] = []
                active_trade["exits"].append(exit_log)

                if s["position"] == 0.0:
                    w_sum = sum(w for w, _ in s["realized_pnl_tracker"])
                    total_profit = sum(p * w for w, p in s["realized_pnl_tracker"]) / w_sum if w_sum > 0 else profit
                    active_trade["status"] = "CLOSED"
                    active_trade["exit_price"] = round(price, 2)
                    active_trade["exit_date"] = current_date
                    active_trade["profit_pct"] = round(total_profit, 2)

        s["position"] = round(s["position"], 2)

    # ==========================================
    # 🔔 إرسال التنبيه وتحديث السايكل
    # ==========================================
    if action:
        alerts.append(
            format_alert(
                action,
                name,
                price,
                s["position"],
                s["avg_price"],
                rsi_val,
                s["cycle"],
                profit,
            )
        )
        
        # تصفير بيانات السهم عند الخروج التام
        if s["position"] == 0.0 and ("SELL" in action or "EXIT" in action or "STOP" in action):
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["realized_pnl_tracker"] = []
            s["cycle"] += 1


# حفظ ملف الحالة الحالية
with open(STATE_FILE, "w") as f:
    json.dump(state_data, f, indent=2)

# حفظ ملف سجل الصفقات التاريخي
with open(TRADES_FILE, "w") as f:
    json.dump(trades_history, f, indent=2)


if alerts:
    send_telegram("\n\n----------------------\n\n".join(alerts))
else:
    send_telegram("Ladder Strategy 😴 No new signals")

print("EGX LADDER CYCLE SYSTEM - DATABASE SOURCED (v3.2 Fully Audited)")

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


def fetch_local_data(name):
    """قراءة البيانات التاريخية والحديثة مباشرة من قاعدة البيانات المحلية المضغوطة."""
    try:
        if not os.path.exists(DB_FILE):
            print(f"⚠️ Database file '{DB_FILE}' not found!")
            return None

        with open(DB_FILE, "r") as f:
            raw_database = json.load(f)

        if name not in raw_database:
            print(f"⚠️ {name} not found in database.")
            return None

        content = raw_database[name]

        if "columns" in content and "data" in content:
            df_temp = pd.DataFrame.from_dict(
                content["data"], orient="index", columns=content["columns"]
            )
            df_temp.index.name = "Date"

            # تحويل حاسم وترتيب تصاعدي إجباري
            df_temp.index = pd.to_datetime(df_temp.index)
            df_temp = df_temp.sort_index(ascending=True)
            return df_temp
        else:
            return None
    except Exception as e:
        print(f"💥 Error reading local data for {name}: {e}")
        return None


# دالة الـ RSI الاحترافية
def rsi(series, period=14):
    if len(series) < period:
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
        return 0

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

    time.sleep(0.1)

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
    current_date = str(df.index[-1].strftime("%Y-%m-%d"))
    price = float(last["Close"])
    rsi_val = float(last["RSI"])

    if pd.isna(rsi_val):
        continue

    if name not in state_data:
        state_data[name] = {
            "cycle": 1,
            "position": 0.0,
            "avg_price": 0.0,
            "peak_profit": 0.0,
        }

    s = state_data[name]

    s["position"] = float(s.get("position", 0.0))
    s["avg_price"] = float(s.get("avg_price", 0.0))
    s["peak_profit"] = float(s.get("peak_profit", 0.0))
    s["cycle"] = int(s.get("cycle", 1))

    if s["position"] == 0.0:
        s["avg_price"] = 0.0
        s["peak_profit"] = 0.0

    # فلتر منع قمم الصعود الصاروخي
    lookback = min(len(df), 80)
    lowest_80 = float(df["Low"].tail(lookback).min())
    highest_80 = float(df["High"].tail(lookback).max())
    
    run_up_percent = ((highest_80 - lowest_80) / lowest_80) * 100 if lowest_80 > 0 else 0.0
    safe_to_buy = run_up_percent <= 60.0

    # شروط الشراء
    ema_up = (df["EMA75"].iloc[-1] > df["EMA75"].iloc[-10]) and (price <= df["EMA75"].iloc[-1] * 1.05)
    
    buy1 = safe_to_buy and ema_up and rsi_val <= 55
    buy2 = safe_to_buy and ema_up and rsi_val <= 43
    buy3 = safe_to_buy and ema_up and rsi_val <= 33

    profit = 0.0
    if s["avg_price"] > 0:
        profit = ((price - s["avg_price"]) / s["avg_price"]) * 100

    sell1 = rsi_val >= 65 and profit > 1
    sell2 = rsi_val >= 72 and profit > 2
    sell3 = rsi_val >= 78 and profit > 3

    action = None

    # تجهيز السجل للسهم في ملف الصفقات
    if name not in trades_history:
        trades_history[name] = []

    # ==========================================
    # 🟢 تنفيذ أومـر الـشـراء وتسجيل الصفقات
    # ==========================================
    
    # شراء المستوى الأول L1
    if s["position"] == 0 and buy1:
        s["position"] = 0.33
        s["avg_price"] = price
        s["peak_profit"] = 0.0
        profit = 0.0
        action = "🟢 BUY L1"

        # إنشاء سجل صفقة جديدة
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

    # شراء المستوى الثاني L2
    elif 0.32 < s["position"] < 0.5 and buy2 and price < s["avg_price"] * 0.98:
        old_pos = s["position"]
        s["position"] = 0.66
        s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
        profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
        action = "🟢 BUY L2"

        # تحديث الصفقة المفتوحة الحالية
        if trades_history[name]:
            active_trade = trades_history[name][-1]
            if active_trade.get("status") == "OPEN":
                active_trade["second_entry"] = f"{current_date} with price {price:.2f}"
                active_trade["last_totally_average_price"] = round(s["avg_price"], 2)

    # شراء المستوى الثالث L3
    elif 0.65 < s["position"] < 1 and buy3 and price < s["avg_price"] * 0.97:
        old_pos = s["position"]
        s["position"] = 1.0
        s["avg_price"] = update_avg(s["avg_price"], old_pos, price, s["position"])
        profit = ((price - s["avg_price"]) / s["avg_price"]) * 100
        action = "🟢 BUY L3"

        # تحديث الصفقة المفتوحة الحالية
        if trades_history[name]:
            active_trade = trades_history[name][-1]
            if active_trade.get("status") == "OPEN":
                active_trade["third_entry"] = f"{current_date} with price {price:.2f}"
                active_trade["last_totally_average_price"] = round(s["avg_price"], 2)

    if profit > s["peak_profit"]:
        s["peak_profit"] = profit

    # ==========================================
    # 🔴 تنفيذ أومـر الـبـيـع وإغلاق الصفقات
    # ==========================================
    if s["position"] > 0:

        stop_triggered = False

        if s["position"] <= 0.33 and profit <= -8:
            stop_triggered = True
        elif s["position"] <= 0.66 and profit <= -5:
            stop_triggered = True
        elif s["position"] == 1.0 and profit <= -4:
            stop_triggered = True

        if s["peak_profit"] > 10 and (s["peak_profit"] - profit) >= 4:
            stop_triggered = True

        # إغلاق كلي (وقف خسارة)
        if stop_triggered:
            action = "🛑 STOP LOSS"
            
            # تسجيل إغلاق الصفقة في الهستوري
            if trades_history[name]:
                active_trade = trades_history[name][-1]
                if active_trade.get("status") == "OPEN":
                    active_trade["status"] = "CLOSED"
                    active_trade["exit_price"] = round(price, 2)
                    active_trade["exit_date"] = current_date
                    active_trade["profit_pct"] = round(profit, 2)

            s["position"] = 0.0
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["cycle"] += 1

        # إغلاق كلي (تارجت أو خروج كامـل)
        elif sell3:
            action = "🚨 EXIT FULL"

            # تسجيل إغلاق الصفقة في الهستوري
            if trades_history[name]:
                active_trade = trades_history[name][-1]
                if active_trade.get("status") == "OPEN":
                    active_trade["status"] = "CLOSED"
                    active_trade["exit_price"] = round(price, 2)
                    active_trade["exit_date"] = current_date
                    active_trade["profit_pct"] = round(profit, 2)

            s["position"] = 0.0
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["cycle"] += 1

        # 🔴 بيع جزئي مستوى ثاني (33%)
        elif sell2:
            sell_amount = min(0.33, s["position"])
            s["position"] -= sell_amount
            action = "🔴 SELL L2 (33%)"
            
            # تسجيل عملية البيع الجزئي في السجل
            if trades_history[name] and trades_history[name][-1].get("status") == "OPEN":
                active_trade = trades_history[name][-1]
                exit_log = f"{current_date}: Sold 33% at price {price:.2f} (Profit: {profit:+.2f}%)"
                
                if "exits" not in active_trade:
                    active_trade["exits"] = []
                active_trade["exits"].append(exit_log)

                # إذا أدت عملية البيع الجزئي لإفراغ المحفظة تماماً (0%)
                if round(s["position"], 2) == 0.0:
                    active_trade["status"] = "CLOSED"
                    active_trade["exit_price"] = round(price, 2)
                    active_trade["exit_date"] = current_date
                    active_trade["profit_pct"] = round(profit, 2)

        # 🔴 بيع جزئي مستوى أول (33%)
        elif sell1:
            sell_amount = min(0.33, s["position"])
            s["position"] -= sell_amount
            action = "🔴 SELL L1 (33%)"
            
            # تسجيل عملية البيع الجزئي في السجل
            if trades_history[name] and trades_history[name][-1].get("status") == "OPEN":
                active_trade = trades_history[name][-1]
                exit_log = f"{current_date}: Sold 33% at price {price:.2f} (Profit: {profit:+.2f}%)"
                
                if "exits" not in active_trade:
                    active_trade["exits"] = []
                active_trade["exits"].append(exit_log)

                # إذا أدت عملية البيع الجزئي لإفراغ المحفظة تماماً (0%)
                if round(s["position"], 2) == 0.0:
                    active_trade["status"] = "CLOSED"
                    active_trade["exit_price"] = round(price, 2)
                    active_trade["exit_date"] = current_date
                    active_trade["profit_pct"] = round(profit, 2)

        s["position"] = round(s["position"], 2)

    # إرسال التنبيه أولاً قبل تصفير المتوسط إذا أغلقت الصفقة بالكامل
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
        
        # إذا أصبحت الكمية صفر بعد تنفيذ إشارات البيع، يتم إعادة ضبط قيم السهم
        if s["position"] == 0.0 and "SELL" in action:
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["cycle"] += 1


# حفظ ملف الحالة الحالية
with open(STATE_FILE, "w") as f:
    json.dump(state_data, f, indent=2)

# حفظ ملف سجل الصفقات التاريخي (تم تصحيح اسم المتغير)
with open(TRADES_FILE, "w") as f:
    json.dump(trades_history, f, indent=2)


if alerts:
    send_telegram("\n\n----------------------\n\n".join(alerts))
else:
    send_telegram("Ladder Strategy 😴 No new signals")

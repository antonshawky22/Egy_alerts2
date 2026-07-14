Print("EGX LADDER CYCLE SYSTEM - DATABASE SOURCED (v2.0 Clean Fix)")

import requests
import os
import json
import pandas as pd
import numpy as np
import time

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


# تم تحديث القائمة لـ 13 سهماً من أقوى أسهم السوق المصري بتناغم كامل
symbols = {
    "COMI": "COMI",
    "HRHO": "HRHO",
    "FWRY": "FWRY",
    "EFIH": "EFIH",
    "TMGH": "TMGH",
    "SWDY": "SWDY",
    "ABUK": "ABUK",
    "EAST": "EAST",
    "JUFO": "JUFO",
    "GBCO": "GBCO",
    "PHDC": "PHDC",
    "AMOC": "AMOC",
    "ETEL": "ETEL"
}

STATE_FILE = "last_signals_strat2.json"
DB_FILE = "egx_history_database_v2.json"


try:
    with open(STATE_FILE, "r") as f:
        state_data = json.load(f)
except:
    state_data = {}


def fetch_local_data(name):
    """
    قراءة البيانات التاريخية والحديثة مباشرة من قاعدة البيانات المحلية المضغوطة
    وتحويلها إلى DataFrame جاهز لحساب المؤشرات الفنية.
    """
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
            df_temp = pd.DataFrame.from_dict(content["data"], orient="index", columns=content["columns"])
            df_temp.index.name = "Date"
            
            # 🛡️ تحويل حاسم وترتيب تصاعدي إجباري (الأقدم فوق) لضمان حساب المتوسطات والـ RSI بشكل صحيح
            df_temp.index = pd.to_datetime(df_temp.index)
            df_temp = df_temp.sort_index(ascending=True)
            return df_temp
        else:
            return None
    except Exception as e:
        print(f"💥 Error reading local data for {name}: {e}")
        return None


# 🛡️ دالة الـ RSI الاحترافية المتطابقة مع TradingView بالملي
def rsi(series, period=14):
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # استخدام معادلة Wilder المعتمدة عالمياً في تريدنج فيو
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
    if df is None or len(df) < 40: # تأمين الحد الأدنى من الداتا للحساب الذكي
        continue

    close = df["Close"]
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA30"] = close.ewm(span=30, adjust=False).mean()
    df["EMA75"] = close.ewm(span=75, adjust=False).mean()
    df["RSI"] = rsi(close)

    last = df.iloc[-1]
    price = float(last["Close"])
    rsi_val = float(last["RSI"])

    if pd.isna(rsi_val):
        continue

    if name not in state_data:
        state_data[name] = {
            "cycle": 1,
            "position": 0.0,
            "avg_price": 0.0,
            "peak_profit": 0.0
        }

    s = state_data[name]
    
    # 🛡️ تأمين جودة ونوع البيانات المسترجعة من الـ JSON لمنع أخطاء الحسابات
    s["position"] = float(s.get("position", 0.0))
    s["avg_price"] = float(s.get("avg_price", 0.0))
    s["peak_profit"] = float(s.get("peak_profit", 0.0))
    s["cycle"] = int(s.get("cycle", 1))

    # 🚀 التعديل الجديد: تصفير البيانات التراكمية إذا كان المركز فارغاً لمنع الأخطاء البصرية والشوائب التراكمية
    if s["position"] == 0.0:
        s["avg_price"] = 0.0
        s["peak_profit"] = 0.0

    ema_up = df["EMA75"].iloc[-1] > df["EMA75"].iloc[-10]

    buy1 = ema_up and rsi_val <= 55
    buy2 = ema_up and rsi_val <= 45
    buy3 = ema_up and rsi_val <= 40

    profit = 0.0
    if s["avg_price"] > 0:
        profit = ((price - s["avg_price"]) / s["avg_price"]) * 100

    sell1 = rsi_val >= 65 and profit > 1
    sell2 = rsi_val >= 72 and profit > 2
    sell3 = rsi_val >= 78 and profit > 3

    action = None

    # شراء المستوى الأول
    if s["position"] == 0 and buy1:
        s["position"] = 0.33
        s["avg_price"] = price
        s["peak_profit"] = 0.0
        action = "🟢 BUY L1"

    # شراء المستوى الثاني
    elif 0.32 < s["position"] < 0.5 and buy2 and price < s["avg_price"]:
        old_pos = s["position"]
        s["position"] = 0.66
        s["avg_price"] = update_avg(
            s["avg_price"], old_pos, price, s["position"]
        )
        action = "🟢 BUY L2"

    # شراء المستوى الثالث
    elif 0.65 < s["position"] < 1 and buy3 and price < s["avg_price"]:
        old_pos = s["position"]
        s["position"] = 1.0
        s["avg_price"] = update_avg(
            s["avg_price"], old_pos, price, s["position"]
        )
        action = "🟢 BUY L3"

    if profit > s["peak_profit"]:
        s["peak_profit"] = profit

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

        if stop_triggered:
            action = "🛑 STOP LOSS"
            s["position"] = 0.0
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["cycle"] += 1

        elif sell3:
            action = "🚨 EXIT FULL"
            s["position"] = 0.0
            s["avg_price"] = 0.0
            s["peak_profit"] = 0.0
            s["cycle"] += 1

        elif sell2:
            sell_amount = min(0.33, s["position"])
            s["position"] -= sell_amount
            s["peak_profit"] = profit
            action = "🔴 SELL L2 (33%)"

        elif sell1:
            sell_amount = min(0.33, s["position"])
            s["position"] -= sell_amount
            s["peak_profit"] = profit
            action = "🔴 SELL L1 (33%)"

        s["position"] = round(s["position"], 2)

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
                profit
            )
        )


with open(STATE_FILE, "w") as f:
    json.dump(state_data, f, indent=2)


if alerts:
    send_telegram("\n\n----------------------\n\n".join(alerts))
else:
    send_telegram("Ladder Strategy 😴 No new signals")

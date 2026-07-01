print("EGX LADDER CYCLE SYSTEM - DATABASE SOURCED")

import requests
import os
import json
import pandas as pd
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


symbols = {
    "COMI": "COMI.CA",
    "HRHO": "HRHO.CA",
    "FWRY": "FWRY.CA",
    "EFIH": "EFIH.CA",
    "TMGH": "TMGH.CA"
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
            # إعادة بناء الجدول من صيغة الـ JSON المضغوطة
            df_temp = pd.DataFrame.from_dict(content["data"], orient="index", columns=content["columns"])
            df_temp.index.name = "Date"
            # تحويل الكشاف لنوع تاريخ لضمان حساب المتوسطات المتحركة والـ RSI بدقة زمنية صحيحة
            df_temp.index = pd.to_datetime(df_temp.index)
            return df_temp
        else:
            return None
    except Exception as e:
        print(f"💥 Error reading local data for {name}: {e}")
        return None


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def update_avg(old_avg, old_pos, new_price, new_pos):
    if new_pos == 0:
        return 0

    added_pos = new_pos - old_pos
    if added_pos <= 0:
        return old_avg  # البيع الجزئي لا يغير متوسط السعر للأسهم المتبقية
        
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

    # جلب البيانات النظيفة من الملف المحلى بدلاً من ياهو فاينانس
    df = fetch_local_data(name)
    if df is None or len(df) < 100:
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

    ema_up = df["EMA75"].iloc[-1] > df["EMA75"].iloc[-10]

    buy1 = ema_up and rsi_val <= 55
    buy2 = ema_up and rsi_val <= 45
    buy3 = ema_up and rsi_val <= 40

    profit = 0
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
        s["peak_profit"] = 0
        action = "🟢 BUY L1"

    # شراء المستوى الثاني (تمت إضافة شرط أن يكون السعر الحالي أقل من أول شراء لضمان التبريد الفعلي)
    elif 0.32 < s["position"] < 0.5 and buy2 and price < s["avg_price"]:
        old_pos = s["position"]
        s["position"] = 0.66
        s["avg_price"] = update_avg(
            s["avg_price"], old_pos, price, s["position"]
        )
        action = "🟢 BUY L2"

    # شراء المستوى الثالث (تمت إضافة شرط السعر)
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

        # نسب الستوب لوس المحدثة والمنطقية لتذبذب السوق المصري
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
            s["position"] = 0
            s["avg_price"] = 0
            s["peak_profit"] = 0
            s["cycle"] += 1

        elif sell3:
            action = "🚨 EXIT FULL"
            s["position"] = 0
            s["avg_price"] = 0
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
    json.dump(state_data, f)


if alerts:
    send_telegram("\n\n----------------------\n\n".join(alerts))
else:
    send_telegram("Ladder Strategy 😴 No new signals ")

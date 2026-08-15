import json
import os
import pandas as pd
import requests
import numpy as np

# =====================
# Telegram settings
# =====================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        print("Telegram credentials not set")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)


# =====================
# Symbols & Files
# =====================
DB_FILE = "egx_history_database_v2.json"
SIGNALS_FILE = "last_signals.json"
TRADES_FILE = "trades.json"  # ✅ ملف سجل الصفقات الجديد


# =====================
# 📒 Trade Logging (نسخة احترافية مأمنة بالكامل)
# =====================
def log_buy(symbol, price, date, entry_trend):
    trade = {
        "symbol": symbol,
        "entry_trend": entry_trend,
        "entry_price": price,
        "entry_date": date
    }

    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []
    else:
        data = []

    data.append(trade)

    with open(TRADES_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_sell(symbol, price, date):
    if not os.path.exists(TRADES_FILE):
        return

    with open(TRADES_FILE, "r") as f:
        try:
            data = json.load(f)
        except:
            data = []

    # البحث عن آخر صفقة مفتوحة لهذا السهم لتحديثها
    for trade in reversed(data):
        if trade["symbol"] == symbol and "exit_price" not in trade:
            entry_price = trade["entry_price"]
            if entry_price != 0:
                profit_pct = ((price - entry_price) / entry_price) * 100
            else:
                profit_pct = 0
                
            trade["exit_price"] = price
            trade["exit_date"] = date
            trade["profit_pct"] = round(profit_pct, 2)
            break 

    with open(TRADES_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# 1. قراءة قاعدة البيانات المحلية
try:
    with open(DB_FILE, "r") as f:
        raw_database = json.load(f)
    print("💾 Strategy Engine: Successfully loaded local compact database v2.")
except Exception as e:
    print(f"❌ Critical Error: Could not find or read {DB_FILE}. Error: {e}")
    raw_database = {}

symbols_keys = list(raw_database.keys())

# 2. تحميل الإشارات السابقة
try:
    with open(SIGNALS_FILE, "r") as f:
        last_signals = json.load(f)
except:
    last_signals = {}

new_signals = last_signals.copy()
data_failures = []
global_last_date = "Unknown Date"

section_up = []
section_side = []
section_down = []


# =====================
# 🛡️ دالة الـ RSI الاحترافية المتطابقة مع TradingView بالملي
# =====================
def rsi(series, period=14):
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # استخدام معادلة Wilder النظيفة المتطابقة مع المنصات العالمية
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# =====================
# Parameters
# =====================
SIDE_CLOSE_PERCENT = 0.03
RSI_SELL = 79
EGX30_KEY = "EGX30"

# =====================
# 🏛️ حساب وتحليل اتجاه المؤشر العام EGX30 أولاً
# =====================
egx_trend = "🔛"

if EGX30_KEY in raw_database:
    egx_content = raw_database[EGX30_KEY]
    if "data" in egx_content and "columns" in egx_content:
        df_egx = pd.DataFrame.from_dict(egx_content["data"], orient="index", columns=egx_content["columns"])
        
        # 🛡️ إصلاح الترتيب الحاسم للمؤشر
        df_egx.index = pd.to_datetime(df_egx.index)
        df_egx = df_egx.sort_index(ascending=True)
        
        if len(df_egx) >= 30:
            # 🛡️ تصحيح الـ adjust=False للتطابق مع الشارت
            df_egx["EMA20"] = df_egx["Close"].ewm(span=20, adjust=False).mean()
            egx_last = df_egx.iloc[-1]
            egx_prev_5 = df_egx.iloc[-14] if len(df_egx) > 14 else df_egx.iloc[-2]
            
            df_egx["crossed"] = (((df_egx["Close"] > df_egx["EMA20"]) & (df_egx["Close"].shift(1) <= df_egx["EMA20"])) | 
                                 ((df_egx["Close"] < df_egx["EMA20"]) & (df_egx["Close"].shift(1) >= df_egx["EMA20"])))
            egx_cross_count = df_egx["crossed"].iloc[-30:].sum()
            
            # 🔥 التعديل المعتمد: إعادة ترتيب منطق الـ if لتفادي الظلم البرمجي للمؤشر
            if egx_cross_count >= 7:
                egx_trend = "🔛"
            elif egx_last["Close"] < egx_last["EMA20"] and egx_last["EMA20"] < egx_prev_5["EMA20"]:
                egx_trend = "🔻"
            elif egx_last["Close"] > egx_last["EMA20"] and egx_last["EMA20"] > egx_prev_5["EMA20"]:
                egx_trend = "↗️"
            else:
                egx_trend = "🔛"

print(f"🏛️ Market Filter: EGX30 Trend determined as [{egx_trend}]")


# =====================
# Main Loop (قرارات الاستراتيجية)
# =====================
for name in symbols_keys:
    if name == EGX30_KEY:
        continue

    stock_content = raw_database.get(name, {})
    if not stock_content or "data" not in stock_content:
        data_failures.append(name)
        continue

    df = pd.DataFrame.from_dict(stock_content["data"], orient="index", columns=stock_content["columns"])
    
    # 🛡️ تحويل حاسم للتاريخ وترتيب تصاعدي أعمى للحسابات الفنية النظيفة
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)

    if len(df) < 40: # تأمين الحد الأدنى لحساب الـ RSI والـ EMA
        data_failures.append(name)
        continue

    # قراءة تاريخ آخر شمعة
    last_candle_date = df.index[-1].strftime('%Y-%m-%d')
    global_last_date = last_candle_date

    # 🛡️ تصحيح الـ adjust=False لجميع المتوسطات لتعطي نفس أرقام تريدنج فيو بالملي
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA30"] = df["Close"].ewm(span=30, adjust=False).mean()
    df["EMA40"] = df["Close"].ewm(span=40, adjust=False).mean()
    df["EMA8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA70"] = df["Close"].ewm(span=70, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)

    df["crossed"] = (((df["Close"] > df["EMA40"]) & (df["Close"].shift(1) <= df["EMA40"])) | 
                     ((df["Close"] < df["EMA40"]) & (df["Close"].shift(1) >= df["EMA40"])))
    cross_count = df["crossed"].iloc[-30:].sum()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev_5 = df.iloc[-14] if len(df) > 14 else df.iloc[-2]

    last_close = last["Close"]

    prev_data = last_signals.get(name, {})
    in_position = prev_data.get("in_position", False)
    entry_price = prev_data.get("entry_price", None)
    prev_trend = prev_data.get("trend", "")

    prev_in_position = in_position  # ✅ حفظ الحالة السابقة بدقة لتأمين دالة الـ Log

    buy_signal = False
    sell_signal = False
    side_signal = ""
    percent_side = None
    up_signal = ""

    if egx_trend == "🔻":
        trend = "🔻"
    else:
        if cross_count >= 7:
            trend = "🔛"
        elif last_close > last["EMA40"] and last["EMA40"] > prev_5["EMA40"]:
            trend = "↗️"
        elif last_close < last["EMA40"] and last["EMA40"] < prev_5["EMA40"]:
            trend = "🔻"
        else:
            trend = "🔛"

    trend_changed = trend != prev_trend

    # 🟢 UP TREND
    if trend == "↗️":
        if not in_position and last["RSI14"] < 68 and last_close >= last["EMA30"] and last_close > prev["Close"]:
            buy_signal = True
            up_signal = "🟢"
            in_position = True
            entry_price = last_close

        elif in_position:
            cross_down = (prev["EMA12"] >= prev["EMA20"] and last["EMA12"] < last["EMA20"])
            stop_loss = last_close < entry_price * 0.93
            rsi_sell = last["RSI14"] > RSI_SELL

            if stop_loss:
                sell_signal = True
                up_signal = "🔴💥"
            elif cross_down or rsi_sell:
                sell_signal = True
                up_signal = "🔴"

            if sell_signal:
                in_position = False
                entry_price = None

    # 🟡 SIDE TREND
    elif trend == "🔛":
        high = df["High"].iloc[-40:].max()
        low = df["Low"].iloc[-40:].min()

        from_high = (high - last_close) / high
        from_low = (last_close - low) / low

        if not in_position and (from_low <= SIDE_CLOSE_PERCENT or last["RSI14"] < 38):
            buy_signal = True
            side_signal = "🟢"
            percent_side = from_low * 100
            in_position = True
            entry_price = last_close

        elif in_position:
            if from_high <= SIDE_CLOSE_PERCENT or last["RSI14"] > 66:
                sell_signal = True
                side_signal = "🔴"
                percent_side = from_high * 100
                in_position = False
                entry_price = None

            elif last_close < entry_price * 0.93:
                sell_signal = True
                side_signal = "🔴💥"
                in_position = False
                entry_price = None

    # 🔴 DOWN TREND
    elif trend == "🔻":
        if in_position:
            sell_signal = True
            in_position = False
            entry_price = None

    # ✅ آلية حفظ الصفقات التلقائية المأمنة خارجياً
    if buy_signal and not prev_in_position:
        log_buy(name, last_close, last_candle_date, trend)
    elif sell_signal and prev_in_position:
        log_sell(name, last_close, last_candle_date)

    trend_mark = "🚧 " if trend_changed else ""

    if trend == "↗️":
        if up_signal:
            section_up.append(f"{trend_mark}{up_signal} {name} | {last_close:.2f} | {last_candle_date}")
        elif trend_changed:
            section_up.append(f"{trend_mark}{name} | {last_close:.2f} | {last_candle_date}")

    elif trend == "🔛":
        if side_signal:
            p = f"{percent_side:.2f}%" if percent_side else ""
            section_side.append(f"{trend_mark}{side_signal} {name} | {last_close:.2f} | {last_candle_date} | {p}")
        elif trend_changed:
            section_side.append(f"{trend_mark}{name} | {last_close:.2f} | {last_candle_date}")

    elif trend == "🔻":
        if trend_changed:
            section_down.append(f"{trend_mark}{name} | {last_close:.2f} | {last_candle_date}")

    new_signals[name] = {
        "trend": trend,
        "in_position": in_position,
        "entry_price": entry_price,
    }

# =====================
# Build & Send Telegram Message
# =====================
alerts = [f"🚦 EGX Alerts Trend Update (Market Filter: {egx_trend}):\n"]

if section_up:
    alerts.append("↗️ صاعد:")
    alerts.extend(["- " + s for s in section_up])

if section_side:
    alerts.append("\n🔛 عرضي:")
    alerts.extend(["- " + s for s in section_side])

if section_down:
    alerts.append("\n🔻 هابط:")
    alerts.extend(["- " + s for s in section_down])

if data_failures:
    alerts.append(f"\n⚠️ Database load failure for symbols:")
    alerts.extend(["- " + s for s in data_failures])
elif not section_up and not section_side and not section_down:
    alerts.append(f"ℹ️ No new symbols for today (last candle: {global_last_date})")

with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

send_telegram("\n".join(alerts))
print("🏁 Strategy Analysis Complete. Alerts Dispatched.")

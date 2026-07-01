import json
import os
import pandas as pd
import requests

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
# 🟢 تم التوجيه لقاعدة البيانات الخماسية الجديدة
DB_FILE = "egx_history_database_v2.json"
SIGNALS_FILE = "last_signals.json"

# 1. قراءة قاعدة البيانات المحلية بالهيكل الجديد
try:
    with open(DB_FILE, "r") as f:
        raw_database = json.load(f)
    print("💾 Strategy Engine: Successfully loaded local compact database v2.")
except Exception as e:
    print(f"❌ Critical Error: Could not find or read {DB_FILE}. Error: {e}")
    raw_database = {}

# استخراج الأسهم تلقائياً من مفاتيح ملف الداتا المحلي
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

# Containers للإشارات
section_up = []
section_side = []
section_down = []


# =====================
# Helpers
# =====================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# =====================
# Parameters
# =====================
SIDE_CLOSE_PERCENT = 0.04
RSI_SELL = 79
EGX30_KEY = "EGX30"  # رمز المؤشر العام في قاعدة البيانات

# =====================
# 🏛️ حساب وتحليل اتجاه المؤشر العام EGX30 أولاً
# =====================
egx_trend = "🔛"  # الحالة الافتراضية للمؤشر

if EGX30_KEY in raw_database:
    egx_content = raw_database[EGX30_KEY]
    if "data" in egx_content and "columns" in egx_content:
        df_egx = pd.DataFrame.from_dict(egx_content["data"], orient="index", columns=egx_content["columns"])
        df_egx.index.name = "Date"
        df_egx = df_egx.sort_index()
        
        if len(df_egx) >= 30:
            df_egx["EMA20"] = df_egx["Close"].ewm(span=20, adjust=True).mean()
            egx_last = df_egx.iloc[-1]
            egx_prev_5 = df_egx.iloc[-5] if len(df_egx) > 5 else df_egx.iloc[-2]
            
            # حساب عدد مرات تقاطع إغلاق المؤشر مع خط EMA20 لآخر 20 شمعة
            df_egx["crossed"] = (((df_egx["Close"] > df_egx["EMA20"]) & (df_egx["Close"].shift(1) <= df_egx["EMA20"])) | 
                                 ((df_egx["Close"] < df_egx["EMA20"]) & (df_egx["Close"].shift(1) >= df_egx["EMA20"])))
            egx_cross_count = df_egx["crossed"].iloc[-20:].sum()
            
            # تحديد الاتجاه الفعلي للمؤشر العام
            if egx_last["Close"] < egx_last["EMA20"] and egx_last["EMA20"] < egx_prev_5["EMA20"]:
                egx_trend = "🔻"  # هابط قوي إجباري
            elif egx_cross_count >= 3:
                egx_trend = "🔛"  # تذبذب عرضي رايح جاي
            elif egx_last["Close"] > egx_last["EMA20"] and egx_last["EMA20"] > egx_prev_5["EMA20"]:
                egx_trend = "↗️"  # صاعد قوي

print(f"🏛️ Market Filter: EGX30 Trend determined as [{egx_trend}]")


# =====================
# Main Loop (قرارات الاستراتيجية)
# =====================
for name in symbols_keys:
    # تخطي معالجة المؤشر العام كسهم فردي داخل الحلقة
    if name == EGX30_KEY:
        continue

    # جلب الداتا وقراءتها بناءً على الهيكل الجديد
    stock_content = raw_database.get(name, {})
    if not stock_content or "data" not in stock_content:
        data_failures.append(name)
        continue

    # 🟢 تحويل الداتا المضغوطة لـ DataFrame مع استيعاب عمود الفوليوم الجديد تلقائياً وبأمان
    df = pd.DataFrame.from_dict(stock_content["data"], orient="index", columns=stock_content["columns"])
    df.index.name = "Date"
    df = df.sort_index()

    if len(df) < 100:
        data_failures.append(name)
        continue

    # قراءة تاريخ آخر شمعة مسجلة ديناميكياً
    last_candle_date = str(df.index[-1])
    global_last_date = last_candle_date

    # حساب المؤشرات الفنية (تم الإبقاء على جميع المتوسطات الحالية دون مساس)
    df["EMA20"] = df["Close"].ewm(span=20, adjust=True).mean()
    df["EMA30"] = df["Close"].ewm(span=30, adjust=True).mean()
    df["EMA40"] = df["Close"].ewm(span=40, adjust=True).mean()
    df["EMA8"] = df["Close"].ewm(span=8, adjust=True).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=True).mean()
    df["EMA70"] = df["Close"].ewm(span=70, adjust=True).mean()
    df["RSI14"] = rsi(df["Close"], 14)

    # حساب عدد مرات تقاطع إغلاق السهم مع خط EMA40 لآخر 20 شمعة
    df["crossed"] = (((df["Close"] > df["EMA40"]) & (df["Close"].shift(1) <= df["EMA40"])) | 
                     ((df["Close"] < df["EMA40"]) & (df["Close"].shift(1) >= df["EMA40"])))
    cross_count = df["crossed"].iloc[-20:].sum()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev_5 = df.iloc[-5] if len(df) > 5 else df.iloc[-2]

    last_close = last["Close"]

    # قراءة الحالة السابقة للسهم
    prev_data = last_signals.get(name, {})
    in_position = prev_data.get("in_position", False)
    entry_price = prev_data.get("entry_price", None)
    prev_trend = prev_data.get("trend", "")

    buy_signal = False
    sell_signal = False
    side_signal = ""
    percent_side = None
    up_signal = ""

    # ===================================================
    # 🎯 تطبيق منطق تحديد الاتجاه المعدل الجديد (فلتر السوق)
    # ===================================================
    if egx_trend == "🔻":
        # المؤشر العام منهار وهابط قوي -> إجبار كافة الأسهم على المسار الهابط فوراً كصمام أمان
        trend = "🔻"
    else:
        # المؤشر العام آمن (صاعد أو عرضي) -> يحدد السهم مساره بالاعتماد على EMA40 والتقاطعات
        if cross_count >= 3:
            trend = "🔛"
        elif last_close > last["EMA40"] and last["EMA40"] > prev_5["EMA40"]:
            trend = "↗️"
        elif last_close < last["EMA40"] and last["EMA40"] < prev_5["EMA40"]:
            trend = "🔻"
        else:
            trend = "🔛"

    trend_changed = trend != prev_trend

    # =====================
    # STRATEGIES
    # =====================

    # 🟢 UP TREND
    if trend == "↗️":
        if not in_position and last["RSI14"] < 68 and last_close >= last["EMA30"] and last_close > prev["Close"]:
            buy_signal = True
            up_signal = "🟢"
            in_position = True
            entry_price = last_close

        elif in_position:
            cross_down = (prev["EMA12"] >= prev["EMA20"] and last["EMA12"] < last["EMA20"])
            stop_loss = last_close < entry_price * 0.94
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

    # 🟡 SIDE TREND (استراتيجيتك الحالية المعتمدة على الـ High/Low لآخر 40 شمعة)
    elif trend == "🔛":
        high = df["High"].iloc[-40:].max()
        low = df["Low"].iloc[-40:].min()

        from_high = (high - last_close) / high
        from_low = (last_close - low) / low

        if not in_position and from_low <= SIDE_CLOSE_PERCENT and last["RSI14"] < 33:
            buy_signal = True
            side_signal = "🟢"
            percent_side = from_low * 100
            in_position = True
            entry_price = last_close

        elif in_position:
            if from_high <= SIDE_CLOSE_PERCENT or last["RSI14"] > 68:
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

    # =====================
    # Formatting Messages
    # =====================
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

    # حفظ حالة التتبع
    new_signals[name] = {
        "trend": trend,
        "in_position": in_position,
        "entry_price": entry_price,
    }

# =====================
# Build & Send Telegram Message
# =====================
# دمج المؤشر العام للسوق في عنوان الرسالة بشكل ثابت
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
    alerts.append(f"ℹ️ No new signals for today (last candle: {global_last_date})")

with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

send_telegram("\n".join(alerts))
print("🏁 Strategy Analysis Complete. Alerts Dispatched.")

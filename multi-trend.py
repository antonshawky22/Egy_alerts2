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
symbols = {
    "OFH": "OFH",
    "OLFI": "OLFI",
    "EMFD": "EMFD",
    "ETEL": "ETEL",
    "EAST": "EAST",
    "EFIH": "EFIH",
    "ABUK": "ABUK",
    "OIH": "OIH",
    "SWDY": "SWDY",
    "ISPH": "ISPH",
    "ATQA": "ATQA",
    "MTIE": "MTIE",
    "ELEC": "ELEC",
    "HRHO": "HRHO",
    "ORWE": "ORWE",
    "JUFO": "JUFO",
    "DSCW": "DSCW",
    "SUGR": "SUGR",
    "ELSH": "ELSH",
    "RMDA": "RMDA",
    "RAYA": "RAYA",
    "EEII": "EEII",
    "MPCO": "MPCO",
    "GBCO": "GBCO",
    "TMGH": "TMGH",
    "ORHD": "ORHD",
    "AMOC": "AMOC",
    "FWRY": "FWRY",
    "COMI": "COMI",
    "ADIB": "ADIB",
    "PHDC": "PHDC",
    "MCQE": "MCQE",
    "SKPC": "SKPC",
    "EGAL": "EGAL",
}

DB_FILE = "egx_history_database.json"
SIGNALS_FILE = "last_signals.json"

# 1. قراءة قاعدة البيانات المحلية (التي تم إنشاؤها وتحديثها في الخطوة السابقة)
try:
    with open(DB_FILE, "r") as f:
        database = json.load(f)
    print("💾 Strategy Engine: Successfully loaded local price database.")
except Exception as e:
    print(f"❌ Critical Error: Could not find or read {DB_FILE}. Error: {e}")
    database = {}

# 2. تحميل الإشارات السابقة
try:
    with open(SIGNALS_FILE, "r") as f:
        last_signals = json.load(f)
except:
    last_signals = {}

new_signals = last_signals.copy()
data_failures = []
global_last_date = "Unknown Date"  # متغير احتياطي لتاريخ آخر شمعة

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

# =====================
# Main Loop (قرارات الاستراتيجية)
# =====================
for name, ticker in symbols.items():

    # جلب البيانات المخزنة محلياً للسهم
    stock_records = database.get(name, [])

    if not stock_records or len(stock_records) < 100:
        data_failures.append(name)
        continue

    # تحويل السجلات لـ DataFrame جاهز للحسابات فوراً بدون إنترنت
    df = pd.DataFrame(stock_records)
    df.set_index("Date", inplace=True)

    # تصحيح برميجي: قراءة تاريخ آخر شمعة مسجلة بعد تعريف الـ df بنجاح
    last_candle_date = list(df.index)[-1]
    global_last_date = last_candle_date

    # حساب المؤشرات الفنية (EMA & RSI) على البيانات الكاملة والمصححة
    df["EMA20"] = df["Close"].ewm(span=20, adjust=True).mean()
    df["EMA30"] = df["Close"].ewm(span=30, adjust=True).mean()
    df["EMA40"] = df["Close"].ewm(span=40, adjust=True).mean()
    df["EMA8"] = df["Close"].ewm(span=8, adjust=True).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=True).mean()
    df["EMA70"] = df["Close"].ewm(span=70, adjust=True).mean()
    df["RSI14"] = rsi(df["Close"], 14)

    last = df.iloc[-1]
    prev = df.iloc[-2]

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

    # تحديد الاتجاه الحالي (Trend Logic)
    if (
        last["EMA20"] > last["EMA30"] * 1.001
        and last["EMA30"] > last["EMA70"] * 1.001
    ):
        trend = "↗️"
    elif (
        last["EMA20"] < last["EMA30"] * 0.999
        and last["EMA30"] < last["EMA70"] * 0.999
    ):
        trend = "🔻"
    else:
        trend = "🔛"

    trend_changed = trend != prev_trend

    # =====================
    # STRATEGIES
    # =====================

    # 🟢 UP TREND
    if trend == "↗️":
        if not in_position and last["RSI14"] < 68 and last_close >= last["EMA30"]:
            buy_signal = True
            up_signal = "🟢"
            in_position = True
            entry_price = last_close

        elif in_position:
            cross_down = (
                prev["EMA12"] >= prev["EMA20"] and last["EMA12"] < last["EMA20"]
            )
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

    # 🟡 SIDE TREND
    elif trend == "🔛":
        high = df["High"].iloc[-40:].max()
        low = df["Low"].iloc[-40:].min()

        from_high = (high - last_close) / high
        from_low = (last_close - low) / low

        if (
            not in_position
            and from_low <= SIDE_CLOSE_PERCENT
            and last["RSI14"] < 33
        ):
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
            section_up.append(
                f"{trend_mark}{up_signal} {name} | {last_close:.2f} | {last_candle_date}"
            )
        elif trend_changed:
            section_up.append(
                f"{trend_mark}{name} | {last_close:.2f} | {last_candle_date}"
            )

    elif trend == "🔛":
        if side_signal:
            p = f"{percent_side:.2f}%" if percent_side else ""
            section_side.append(
                f"{trend_mark}{side_signal} {name} | {last_close:.2f} | {last_candle_date} | {p}"
            )
        elif trend_changed:
            section_side.append(
                f"{trend_mark}{name} | {last_close:.2f} | {last_candle_date}"
            )

    elif trend == "🔻":
        if trend_changed:
            section_down.append(
                f"{trend_mark}{name} | {last_close:.2f} | {last_candle_date}"
            )

    # حفظ حالة التتبع
    new_signals[name] = {
        "trend": trend,
        "in_position": in_position,
        "entry_price": entry_price,
    }

# =====================
# Build & Send Telegram Message
# =====================
alerts = ["🚦 EGX Alerts Trend Update:\n"]

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
    alerts.append(
        f"ℹ️ No new signals for today (last candle: {global_last_date})"
    )

with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

send_telegram("\n".join(alerts))
print("🏁 Strategy Analysis Complete. Alerts Dispatched.")


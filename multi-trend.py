print("EGX ALERTS - Corrected Stable Version with Side Trend Signals & RSI83 Sell")

import yfinance as yf
import requests
import os
import json
import pandas as pd

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
# EGX symbols
# =====================
symbols = {
    "OFH":"OFH.CA","OLFI":"OLFI.CA","EMFD":"EMFD.CA","ETEL":"ETEL.CA",
    "EAST":"EAST.CA","EFIH":"EFIH.CA","ABUK":"ABUK.CA","OIH":"OIH.CA",
    "SWDY":"SWDY.CA","ISPH":"ISPH.CA","ATQA":"ATQA.CA","MTIE":"MTIE.CA",
    "ELEC":"ELEC.CA","HRHO":"HRHO.CA","ORWE":"ORWE.CA","JUFO":"JUFO.CA",
    "DSCW":"DSCW.CA","SUGR":"SUGR.CA","ELSH":"ELSH.CA","RMDA":"RMDA.CA",
    "RAYA":"RAYA.CA","EEII":"EEII.CA","MPCO":"MPCO.CA","GBCO":"GBCO.CA",
    "TMGH":"TMGH.CA","ORHD":"ORHD.CA","AMOC":"AMOC.CA","FWRY":"FWRY.CA",
    "COMI":"COMI.CA","ADIB":"ADIB.CA","PHDC":"PHDC.CA",
    "MCQE":"MCQE.CA","SKPC":"SKPC.CA","EGAL":"EGAL.CA"
}

# =====================
# Load last signals
# =====================
SIGNALS_FILE = "last_signals.json"
try:
    with open(SIGNALS_FILE, "r") as f:
        last_signals = json.load(f)
except:
    last_signals = {}

new_signals = last_signals.copy()
data_failures = []
last_candle_date = None

# =====================
# Helpers
# =====================
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# =====================
# Parameters
# =====================
EMA_PERIOD = 40
TREND_LOOKBACK = 40
SIDE_LOOKBACK = 60

EMA_FAST = 20
EMA_SLOW = 40
EMA_TREND = 100

EMA_FORCED_SELL = 100

SIDE_CLOSE_PERCENT = 0.04
RSI_SELL = 83

# =====================
# Containers
# =====================
section_up = []
section_side = []
section_down = []

# =====================
# Main Loop
# =====================
for name, ticker in symbols.items():
    df = fetch_data(ticker)
    if df is None or len(df) < TREND_LOOKBACK:
        data_failures.append(name)
        continue

    last_candle_date = df.index[-1].date()

    # =====================
    # Indicators
    # =====================
    df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA40"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=EMA_TREND, adjust=False).mean()
    df["EMA100_forced"] = df["Close"].ewm(span=EMA_FORCED_SELL, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    last_close = last["Close"]
    prev_close = prev["Close"]

    last_ema4 = last["EMA4"]
    prev_ema4 = prev["EMA4"]

    last_ema9 = last["EMA9"]
    prev_ema9 = prev["EMA9"]

    buy_signal = False
    sell_signal = False
    side_signal = ""
    percent_side = None

    prev_data = last_signals.get(name, {})
    prev_signal = prev_data.get("last_signal", "")
    prev_trend = prev_data.get("trend", "")
    prev_forced = prev_data.get("last_forced_sell", False)
    prev_side_actual = prev_data.get("last_side_signal_actual", "")
    prev_side_buy_price = prev_data.get("prev_side_buy_price", None)

    # =====================
    # 🔥 تحديد الاتجاه (محسن)
    # =====================
    if last["EMA20"] > last["EMA40"] > last["EMA100"] and last_close > last["EMA20"]:
        trend = "↗️"

    elif last["EMA20"] < last["EMA40"] < last["EMA100"] and last_close < last["EMA20"]:
        trend = "🔻"

    else:
        trend = "🔛"

    # =====================
    # 📊 العرضي
    # =====================
    if trend == "🔛":
        high_lookback = df["High"].iloc[-SIDE_LOOKBACK:]
        low_lookback = df["Low"].iloc[-SIDE_LOOKBACK:]

        highest_high = high_lookback.max()
        lowest_low = low_lookback.min()

        percent_from_high = (highest_high - last_close) / highest_high * 100
        percent_from_low = (last_close - lowest_low) / lowest_low * 100

        if percent_from_high <= SIDE_CLOSE_PERCENT * 100:
            sell_signal = True
            side_signal = "🔴"
            percent_side = percent_from_high

        elif percent_from_low <= SIDE_CLOSE_PERCENT * 100:
            buy_signal = True
            side_signal = "🟢"
            percent_side = percent_from_low
            prev_side_buy_price = last_close

        # Stop loss عرضي
        if prev_side_buy_price and last_close < prev_side_buy_price * 0.96:
            sell_signal = True
            side_signal = "🔴💥"
            percent_side = None

    # =====================
    # 🔴 Reset عند بداية الهبوط
    # =====================
    if trend == "🔻" and prev_trend != "🔻":
        sell_signal = True
        buy_signal = False
        prev_side_buy_price = None
        prev_side_actual = ""
        prev_signal = ""

    # =====================
    # 🧹 تنظيف العرضي
    # =====================
    if trend != "🔛":
        prev_side_buy_price = None
        prev_side_actual = ""

    # =====================
    # Forced Sell
    # =====================
    forced_sell_mark = ""
    if last_close < last["EMA100_forced"] and not prev_forced:
        sell_signal = True
        buy_signal = False
        forced_sell_mark = "🚨"
        last_forced = True
    else:
        last_forced = prev_forced

    # =====================
    # استراتيجية الصاعد
    # =====================
    if trend == "↗️":
        if last["RSI14"] < 60 and last_close > last["EMA40"]:
            buy_signal = True

        elif prev_ema4 >= prev_ema9 and last_ema4 < last_ema9:
            if last["RSI14"] > RSI_SELL:
                sell_signal = True

    # =====================
    # منع التكرار
    # =====================
    if buy_signal and prev_signal == "BUY":
        buy_signal = False

    if sell_signal and prev_signal == "SELL":
        sell_signal = False

    if trend == "🔛":
        if side_signal == prev_side_actual:
            side_signal = ""
        else:
            prev_side_actual = side_signal

    # =====================
    # علامة تغيير الاتجاه
    # =====================
    trend_changed_mark = ""
    if prev_trend and prev_trend != trend:
        trend_changed_mark = "🚧 "

    # =====================
    # تجهيز الرسائل
    # =====================
    if trend == "↗️" and (buy_signal or sell_signal):
        mark = "🟢" if buy_signal else "🔴"
        section_up.append(f"{trend_changed_mark}{forced_sell_mark}{mark} {name} | {last_close:.2f} | {last_candle_date}")

    elif trend == "🔛" and side_signal:
        percent_display = f"{percent_side:.2f}%" if percent_side else ""
        section_side.append(f"{trend_changed_mark}{forced_sell_mark}{side_signal} {name} | {last_close:.2f} | {last_candle_date} | {percent_display}")

    elif trend == "🔻" and trend != prev_trend:
        section_down.append(f"{trend_changed_mark}{forced_sell_mark}{name} | {last_close:.2f} | {last_candle_date}")

    # =====================
    # حفظ الحالة
    # =====================
    new_signals[name] = {
        "last_signal": "BUY" if buy_signal else "SELL" if sell_signal else prev_signal,
        "trend": trend,
        "last_forced_sell": last_forced,
        "last_side_signal_actual": prev_side_actual,
        "prev_side_buy_price": None if sell_signal else prev_side_buy_price
    }

# =====================
# الرسالة
# =====================
alerts = ["🚦 EGX Alerts:\n"]

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
    alerts.append("\n⚠️ Failed:\n- " + "\n- ".join(data_failures))

# =====================
# حفظ + إرسال
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

send_telegram("\n".join(alerts))

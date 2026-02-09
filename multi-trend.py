print("EGX ALERTS - Phase 3: EMA60 Trend + Buy/Sell Rules with Direction Changes")

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
    "OFH": "OFH.CA","OLFI": "OLFI.CA","EMFD": "EMFD.CA","ETEL": "ETEL.CA",
    "EAST": "EAST.CA","EFIH": "EFIH.CA","ABUK": "ABUK.CA","OIH": "OIH.CA",
    "SWDY": "SWDY.CA","ISPH": "ISPH.CA","ATQA": "ATQA.CA","MTIE": "MTIE.CA",
    "ELEC": "ELEC.CA","HRHO": "HRHO.CA","ORWE": "ORWE.CA","JUFO": "JUFO.CA",
    "DSCW": "DSCW.CA","SUGR": "SUGR.CA","ELSH": "ELSH.CA","RMDA": "RMDA.CA",
    "RAYA": "RAYA.CA","EEII": "EEII.CA","MPCO": "MPCO.CA","GBCO": "GBCO.CA",
    "TMGH": "TMGH.CA","ORHD": "ORHD.CA","AMOC": "AMOC.CA","FWRY": "FWRY.CA",
    "COMI": "COMI.CA","ADIB": "ADIB.CA","PHDC": "PHDC.CA",
    "EGTS": "EGTS.CA","MCQE": "MCQE.CA","SKPC": "SKPC.CA",
    "EGAL": "EGAL.CA"
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
alerts = []
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
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# =====================
# Parameters
# =====================
EMA_PERIOD = 60
LOOKBACK = 50
THRESHOLD = 0.85  # 85% من الشموع

# =====================
# Main Logic
# =====================
for name, ticker in symbols.items():
    df = fetch_data(ticker)
    if df is None or len(df) < LOOKBACK:
        data_failures.append(name)
        continue

    last_candle_date = df.index[-1].date()
    last_close = df["Close"].iloc[-1]
    last_rsi = rsi(df["Close"], 14).iloc[-1]

    df["EMA60"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)

    recent_closes = df["Close"].iloc[-LOOKBACK:]
    recent_ema = df["EMA60"].iloc[-LOOKBACK:]

    bullish_ratio = (recent_closes > recent_ema).sum() / LOOKBACK
    bearish_ratio = (recent_closes < recent_ema).sum() / LOOKBACK

    # =====================
    # Trend classification
    # =====================
    if bullish_ratio >= THRESHOLD:
        trend = "↗️ صاعد"
        # شراء وبيع الصاعد
        df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        last_ema4 = df["EMA4"].iloc[-1]
        last_ema9 = df["EMA9"].iloc[-1]
        prev_ema4 = df["EMA4"].iloc[-2]
        prev_ema9 = df["EMA9"].iloc[-2]

        buy_signal = last_ema4 > last_ema9 and prev_ema4 <= prev_ema9 and last_rsi < 60
        sell_signal = (last_ema4 < last_ema9 and prev_ema4 >= prev_ema9) or last_rsi > 88

    elif bearish_ratio >= THRESHOLD:
        trend = "❌ هابط"
        buy_signal = sell_signal = False

    else:
        trend = "🔛 عرضي"
        # شراء وبيع العرضي
        df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        last_ema4 = df["EMA4"].iloc[-1]
        last_ema9 = df["EMA9"].iloc[-1]

        buy_signal = last_rsi < 40 and last_close < last_ema4
        sell_signal = last_rsi > 55 and last_close < last_ema9

    # =====================
    # Detect direction change
    # =====================
    prev_trend = last_signals.get(name, {}).get("trend")
    changed_mark = "📊 " if prev_trend and prev_trend != trend else ""

    # =====================
    # Prepare alerts
    # =====================
    prev_state = last_signals.get(name, {}).get("last_signal")

    if buy_signal and prev_state != "BUY":
        signal_text = f"{changed_mark}{name} | {last_close:.2f} | {last_candle_date} | {trend} | 🟢 BUY"
        new_signals[name] = {"last_signal": "BUY", "trend": trend}
    elif sell_signal and prev_state != "SELL":
        signal_text = f"{changed_mark}{name} | {last_close:.2f} | {last_candle_date} | {trend} | 🔴 SELL"
        new_signals[name] = {"last_signal": "SELL", "trend": trend}
    else:
        signal_text = f"{changed_mark}{name} | {last_close:.2f} | {last_candle_date} | {trend}"
        new_signals[name] = {"last_signal": prev_state, "trend": trend}

    alerts.append(signal_text)

# =====================
# Data failures alert
# =====================
if data_failures:
    alerts.append("⚠️ Failed to fetch data:\n- " + "\n- ".join(data_failures))

# =====================
# Save & Notify
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2)

if alerts:
    send_telegram("🚦 EGX Alerts:\n\n" + "\n".join(alerts))
else:
    send_telegram(f"ℹ️ No new signals\nLast candle: {last_candle_date}")

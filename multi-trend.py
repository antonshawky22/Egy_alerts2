print("EGX ALERTS - Phase 4: Complete Version with Full Symbols & Signals")

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
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =====================
# Parameters
# =====================
EMA_PERIOD = 60
LOOKBACK = 20
THRESHOLD = 0.60
EMA_FORCED_SELL = 25

# =====================
# Containers
# =====================
section_up = []
section_side = []
section_side_weak = []  
section_down = []

# =====================
# Main Logic
# =====================
for name, ticker in symbols.items():
    df = fetch_data(ticker)
    if df is None or len(df) < LOOKBACK:
        data_failures.append(name)
        continue

    last_candle_date = df.index[-1].date()

    # =====================
    # Indicators
    # =====================
    df["EMA60"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA25"] = df["Close"].ewm(span=EMA_FORCED_SELL, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)

    recent_closes = df["Close"].iloc[-LOOKBACK:]
    recent_ema60 = df["EMA60"].iloc[-LOOKBACK:]

    bullish_ratio = (recent_closes > recent_ema60).sum() / LOOKBACK
    bearish_ratio = (recent_closes < recent_ema60).sum() / LOOKBACK

    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    last_rsi = df["RSI14"].iloc[-1]
    prev_rsi = df["RSI14"].iloc[-2]
    last_ema4 = df["EMA4"].iloc[-1]
    prev_ema4 = df["EMA4"].iloc[-2]
    last_ema9 = df["EMA9"].iloc[-1]
    prev_ema9 = df["EMA9"].iloc[-2]
    last_volume = df["Volume"].iloc[-1]
    prev_volume = df["Volume"].iloc[-2]

    buy_signal = sell_signal = False
    changed_mark = ""
    target_section = None

    # =====================
    # Determine Trend
    # =====================
    if bullish_ratio >= THRESHOLD:
        trend = "↗️"  # Uptrend
    elif bearish_ratio >= THRESHOLD:
        trend = "🔻"  # Downtrend
    else:
        trend = "🔛"  # Sideways
        bullish_50 = (recent_closes > recent_ema60).sum() / LOOKBACK
        if bullish_50 < 0.5:
            target_section = section_side_weak
        else:
            target_section = section_side

    # =====================
    # Strategy by Trend
    # =====================
    if trend == "↗️":
        # EMA4/EMA9 crossover strategy
        if prev_ema4 <= prev_ema9 and last_ema4 > last_ema9:
            buy_signal = True
        elif prev_ema4 >= prev_ema9 and last_ema4 < last_ema9:
            sell_signal = True

    elif trend == "🔛":
        # Sideways: peaks/valleys ±5%
        high_lookback = df["Close"].iloc[-EMA_PERIOD:]
        low_lookback = df["Close"].iloc[-EMA_PERIOD:]
        high_threshold = high_lookback.max() * 0.95
        low_threshold = low_lookback.min() * 1.05
        near_peak = last_close >= high_threshold
        near_valley = last_close <= low_threshold

        if near_peak:
            section_side.append(f"🔴 {name} | {last_close:.2f} | {last_candle_date} | {last_close/high_lookback.max()*100:.2f}%")
        elif near_valley:
            section_side.append(f"🟢 {name} | {last_close:.2f} | {last_candle_date} | {last_close/low_lookback.min()*100:.2f}%")

    # Downtrend: show only
    elif trend == "🔻":
        section_down.append(f"{name} | {last_close:.2f} | {last_candle_date}")

    # =====================
    # Forced Sell
    # =====================
    if last_close < df["EMA25"].iloc[-1] and new_signals.get(name, {}).get("last_forced_sell") != "FORCED_SELL":
        sell_signal = True
        buy_signal = False
        changed_mark = "🚨"
        last_forced = "FORCED_SELL"
    else:
        last_forced = new_signals.get(name, {}).get("last_forced_sell", "")

    # =====================
    # Prevent repeated BUY/SELL
    # =====================
    prev_data = last_signals.get(name, {})
    prev_signal = prev_data.get("last_signal", "")
    if buy_signal and prev_signal == "BUY":
        buy_signal = False
    if sell_signal and prev_signal == "SELL":
        sell_signal = False

    # =====================
    # Prepare signals for Uptrend only (compact)
    # =====================
    if trend == "↗️" and (buy_signal or sell_signal):
        mark = "🟢" if buy_signal else "🔴"
        section_up.append(f"{mark} {name} | {last_close:.2f} | {last_candle_date}")

    # =====================
    # Update last signals
    # =====================
    new_signals[name] = {
        "last_signal": "BUY" if buy_signal else "SELL" if sell_signal else prev_signal,
        "trend": trend,
        "last_forced_sell": last_forced
    }

# =====================
# Compile Message
# =====================
alerts = ["🚦 EGX Alerts (Compact):\n"]

if section_up:
    alerts.append("↗️ صاعد (شراء/بيع):")
    alerts.extend(["- " + s for s in section_up])
if section_side:
    alerts.append("\n🔛 عرضي (قمم/قيعان):")
    alerts.extend(["- " + s for s in section_side])
if section_down:
    alerts.append("\n🔻 هابط:")
    alerts.extend(["- " + s for s in section_down])

if data_failures:
    alerts.append("\n⚠️ Failed to fetch data:\n- " + "\n- ".join(data_failures))

# =====================
# Save & Notify
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

if alerts:
    send_telegram("\n".join(alerts))
else:
    send_telegram(f"ℹ️ No new signals\nLast candle: {last_candle_date}")

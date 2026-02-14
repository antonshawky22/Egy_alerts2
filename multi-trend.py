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

# =====================
# RSI مطابق TradingView
# =====================
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
EMA_PERIOD = 60
LOOKBACK = 20
THRESHOLD = 0.60
NEAR_PERCENT = 5

# =====================
# Containers
# =====================
section_up = []
section_side = []
section_side_weak = []
section_down = []
section_peaks = []
section_valleys = []

# =====================
# Main Logic
# =====================
for name, ticker in symbols.items():
    df = fetch_data(ticker)
    if df is None or len(df) < EMA_PERIOD:
        data_failures.append(name)
        continue

    last_candle_date = df.index[-1].date()

    df["EMA60"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)

    recent_closes = df["Close"].iloc[-LOOKBACK:]
    recent_ema = df["EMA60"].iloc[-LOOKBACK:]

    bullish_ratio = (recent_closes > recent_ema).sum() / LOOKBACK
    bearish_ratio = (recent_closes < recent_ema).sum() / LOOKBACK

    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]

    # =====================
    # Trend classification
    # =====================
    if bullish_ratio >= THRESHOLD:
        trend = "↗️"
    elif bearish_ratio >= THRESHOLD:
        trend = "🔻"
    else:
        trend = "🔛"

    # =====================
    # 🔥 NEW PEAK / VALLEY LOGIC
    # =====================
    recent_high = df["Close"].iloc[-EMA_PERIOD:].max()
    recent_low = df["Close"].iloc[-EMA_PERIOD:].min()

    distance_from_high = ((recent_high - last_close) / recent_high) * 100
    distance_from_low = ((last_close - recent_low) / recent_low) * 100

    distance_from_high = round(distance_from_high, 2)
    distance_from_low = round(distance_from_low, 2)

    peak_signal = valley_signal = False

    if distance_from_high <= NEAR_PERCENT:
        peak_signal = True
        section_peaks.append(
            f"{name} | {last_close:.2f} | {last_candle_date} | 🔴SELL {distance_from_high}%"
        )

    elif distance_from_low <= NEAR_PERCENT:
        valley_signal = True
        section_valleys.append(
            f"{name} | {last_close:.2f} | {last_candle_date} | 🟢BUY {distance_from_low}%"
        )

    # =====================
    # Trend Sections
    # =====================
    base_text = f"{trend} {name} | {last_close:.2f} | {last_candle_date}"

    if trend == "↗️":
        section_up.append(base_text)
    elif trend == "🔛":
        section_side.append(base_text)
    else:
        section_down.append(base_text)

    new_signals[name] = {
        "trend": trend
    }

# =====================
# Compile message
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

if section_peaks:
    alerts.append("\n⛰️ قرب القمم:")
    alerts.extend(["- " + s for s in section_peaks])

if section_valleys:
    alerts.append("\n🏔️ قرب القيعان:")
    alerts.extend(["- " + s for s in section_valleys])

if data_failures:
    alerts.append("\n⚠️ Failed to fetch data:\n- " + "\n- ".join(data_failures))

with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

send_telegram("\n".join(alerts))

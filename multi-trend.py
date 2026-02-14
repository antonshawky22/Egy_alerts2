print("EGX ALERTS - Compact Version with Peaks/Valleys & EMA Signals")

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
EMA_PERIOD = 60  # للقمم/القيعان
LOOKBACK = 20
THRESHOLD = 0.60  # نسبة صعود/هبوط 60%
EMA_FORCED_SELL = 25  # متوسط للإغلاق القسري

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
    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]

    # =====================
    # Indicators
    # =====================
    df["EMA60"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["EMA25"] = df["Close"].ewm(span=EMA_FORCED_SELL, adjust=False).mean()
    df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    last_rsi = df["RSI14"].iloc[-1]

    recent_closes = df["Close"].iloc[-LOOKBACK:]
    recent_ema = df["EMA60"].iloc[-LOOKBACK:]
    bullish_ratio = (recent_closes > recent_ema).sum() / LOOKBACK
    bearish_ratio = (recent_closes < recent_ema).sum() / LOOKBACK

    buy_signal = sell_signal = False
    changed_mark = ""
    trend = ""

    # =====================
    # Trend classification
    # =====================
    if bullish_ratio >= THRESHOLD:
        trend = "↗️"  # صاعد
        # EMA4/EMA9 تقاطعات
        if df["EMA4"].iloc[-2] < df["EMA9"].iloc[-2] and df["EMA4"].iloc[-1] > df["EMA9"].iloc[-1]:
            buy_signal = True
        elif df["EMA4"].iloc[-2] > df["EMA9"].iloc[-2] and df["EMA4"].iloc[-1] < df["EMA9"].iloc[-1]:
            sell_signal = True

    elif bearish_ratio >= THRESHOLD:
        trend = "🔻"  # هابط
    else:
        trend = "🔛"  # عرضي
        # =====================
        # Peaks/Valleys 5%
        # =====================
        high_lookback = df["Close"].iloc[-EMA_PERIOD:]
        low_lookback = df["Close"].iloc[-EMA_PERIOD:]
        high_max = high_lookback.max()
        low_min = low_lookback.min()

        pct_from_peak = (last_close / high_max) * 100
        pct_from_valley = (last_close / low_min) * 100

        if pct_from_peak >= 95:  # قرب القمة
            sell_signal = True
            signal_text = f"🔴{name} | {last_close:.2f} | {last_candle_date} | {pct_from_peak:.2f}%"
            section_side.append(signal_text)
        elif pct_from_valley <= 105:  # قرب القاع
            buy_signal = True
            signal_text = f"🟢{name} | {last_close:.2f} | {last_candle_date} | {pct_from_valley:.2f}%"
            section_side.append(signal_text)

    # =====================
    # Forced Sell
    # =====================
    if trend != "🔛" and last_close < df["EMA25"].iloc[-1]:
        sell_signal = True
        buy_signal = False
        changed_mark = "🚨"

    # =====================
    # Detect trend change 🚧
    # =====================
    prev_data = last_signals.get(name, {})
    prev_trend = prev_data.get("trend", "")
    if trend != prev_trend:
        changed_mark = "🚧"

    # =====================
    # Prepare section text for ascending trend
    # =====================
    if trend == "↗️" and (buy_signal or sell_signal):
        mark = "🟢BUY" if buy_signal else "🔴SELL"
        signal_text = f"{changed_mark} {name} | {last_close:.2f} | {last_candle_date} | {mark}"
        section_up.append(signal_text)
    elif trend == "🔻":
        section_down.append(f"{name} | {last_close:.2f}")

    # =====================
    # Update last signals
    # =====================
    new_signals[name] = {
        "last_signal": "BUY" if buy_signal else "SELL" if sell_signal else prev_data.get("last_signal", ""),
        "trend": trend
    }

# =====================
# Compile message (compact)
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

send_telegram("\n".join(alerts))

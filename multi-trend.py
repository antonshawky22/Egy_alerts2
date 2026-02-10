٤print("EGX ALERTS - Phase 3: Final Version with Forced Sell")

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
THRESHOLD = 0.85  # 85%
EMA_FORCED_SELL = 25  # متوسط 25 للشروط القسرية

# =====================
# Containers
# =====================
section_up = []
section_side = []
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

    df["EMA60"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA25"] = df["Close"].ewm(span=EMA_FORCED_SELL, adjust=False).mean()

    recent_closes = df["Close"].iloc[-LOOKBACK:]
    recent_ema = df["EMA60"].iloc[-LOOKBACK:]

    bullish_ratio = (recent_closes > recent_ema).sum() / LOOKBACK
    bearish_ratio = (recent_closes < recent_ema).sum() / LOOKBACK

    last_close = df["Close"].iloc[-1]
    last_rsi = df["RSI14"].iloc[-1]
    last_ema4 = df["EMA4"].iloc[-1]
    last_ema9 = df["EMA9"].iloc[-1]
    last_ema25 = df["EMA25"].iloc[-1]
    prev_ema4 = df["EMA4"].iloc[-2]
    prev_ema9 = df["EMA9"].iloc[-2]
    prev_close = df["Close"].iloc[-2]

    buy_signal = sell_signal = False

    # =====================
    # Trend classification + Buy/Sell rules
    # =====================
    if bullish_ratio >= THRESHOLD:
        trend = "↗️صاعد"
        if last_rsi < 60:
            buy_signal = True
        if (last_ema4 < last_ema9 and prev_ema4 >= prev_ema9) or last_rsi > 88:
            sell_signal = True
    elif bearish_ratio >= THRESHOLD:
        trend = "❌ هابط"
        buy_signal = sell_signal = False
    else:
        trend = "🔛عرضي"
        if last_rsi < 32 and last_close < last_ema4:
            buy_signal = True
        if last_rsi > 55 and last_close < last_ema9:
            sell_signal = True

    # =====================
    # Check direction change
    # =====================
    prev_data = last_signals.get(name, {})
    prev_trend = prev_data.get("trend")
    prev_signal = prev_data.get("last_signal")
    prev_forced = prev_data.get("last_forced_sell", "")

    changed_mark = "🚧📢" if prev_trend and prev_trend != trend else ""

    # =====================
    # Forced Sell Rule (cross EMA25)
    # =====================
    if last_close < last_ema25 and prev_forced != "FORCED_SELL":
        sell_signal = True
        buy_signal = False
        changed_mark = "🚨⚠️ "  # forced sell alert
        last_forced = "FORCED_SELL"
    else:
        last_forced = prev_forced  # حافظ على القيمة السابقة

    # =====================
    # Prevent repeated normal BUY/SELL
    # =====================
    if buy_signal and prev_signal == "BUY":
        buy_signal = False
    if sell_signal and prev_signal == "SELL":
        sell_signal = False

    # =====================
    # Prepare signal text
    # =====================
    signal_text = f"{changed_mark}{name} | {last_close:.2f} | {last_candle_date}"
    if trend == "↗️صاعد":
        if buy_signal:
            signal_text += f"|{trend}|🟢BUY"
        elif sell_signal:
            signal_text += f"|{trend}|🔴SELL"
        else:
            signal_text += f" | {trend}"
        section_up.append(signal_text)
    elif trend == "🔛عرضي":
        if buy_signal:
            signal_text += f"|{trend}|🟢BUY"
        elif sell_signal:
            signal_text += f"|{trend}|🔴SELL"
        else:
            signal_text += f"|{trend}"
        section_side.append(signal_text)
    else:
        if sell_signal:
            signal_text += f"|{trend}|🔴SELL"
        else:
            signal_text += f"|{trend}"
        section_down.append(signal_text)

    # =====================
    # Update last signals
    # =====================
    new_signals[name] = {
        "last_signal": "BUY" if buy_signal else "SELL" if sell_signal else prev_signal,
        "trend": trend,
        "last_forced_sell": last_forced
    }

# =====================
# Compile final message
# =====================
alerts = []
alerts.append("🚦 EGX Alerts:\n")

if section_up:
    alerts.append("↗️صاعد:")
    alerts.extend(["- " + s for s in section_up])
if section_side:
    alerts.append("\n🔛عرضي:")
    alerts.extend(["- " + s for s in section_side])
if section_down:
    alerts.append("\n❌ هابط:")
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

print("EGX ALERTS - Safe RSI/EMA Strategy + Precise Reason + Trends")

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
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

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
# Main Logic
# =====================
for name, ticker in symbols.items():
    df = fetch_data(ticker)
    if df is None or len(df) < 60:
        data_failures.append(name)
        continue

    last_candle_date = df.index[-1].date()

    # EMAs
    for p in [3,4,5,9,25,35,45,55]:
        df[f"EMA{p}"] = df["Close"].ewm(span=p, adjust=False).mean()

    df["RSI14"] = rsi(df["Close"], 14)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # =====================
    # Trend detection
    # =====================
    if (
        last["EMA55"] > last["EMA45"] > last["EMA35"]
        and last["Close"] < last["EMA25"]
    ):
        trend = "⚪ اتجاه هابط"
        buy_signal = sell_signal = False
        reason = ""

    elif (
        last["EMA25"] > last["EMA35"] > last["EMA45"]
        and last["Close"] > last["EMA35"]
    ):
        trend = "🟢 ترند صاعد"

        buy_signal = False
        sell_signal = False
        reason = ""

        if last["RSI14"] >= 85:
            sell_signal = True
            reason = "RSI14 >= 85"

        elif last["EMA3"] < last["EMA5"] and last["Close"] < last["EMA9"]:
            sell_signal = True
            reason = "EMA3 < EMA5 & Price < EMA9"

        elif last["Close"] < last["EMA25"]:
            sell_signal = True
            reason = "Broke EMA25"

        elif last["EMA4"] > last["EMA9"] and prev["EMA4"] <= prev["EMA9"] and last["RSI14"] < 70:
            buy_signal = True
            reason = "EMA4 cross EMA9"

    else:
        trend = "🟡 اتجاه عرضي"

        buy_signal = False
        sell_signal = False
        reason = ""

        if last["RSI14"] < 40 and last["Close"] <= last["EMA3"]:
            buy_signal = True
            reason = "RSI < 40 & Price <= EMA3"

        elif last["EMA3"] < last["EMA9"] and prev["EMA3"] >= prev["EMA9"]:
            sell_signal = True
            reason = "EMA3 cross EMA9 down"

        elif last["RSI14"] > 60:
            sell_signal = True
            reason = "RSI > 60"

        elif last["Close"] >= last["EMA9"]:
            sell_signal = True
            reason = "Price >= EMA9"

    prev_state = last_signals.get(name, {}).get("last_signal")

    if buy_signal and prev_state != "BUY":
        alerts.append(
            f"🟢 BUY | {name} | {last['Close']:.2f} | {last_candle_date} | Reason: {reason} | Trend: {trend}"
        )
        new_signals[name] = {"last_signal": "BUY"}

    elif sell_signal and prev_state != "SELL":
        alerts.append(
            f"🔴 SELL | {name} | {last['Close']:.2f} | {last_candle_date} | Reason: {reason} | Trend: {trend}"
        )
        new_signals[name] = {"last_signal": "SELL"}
# =====================
# Data failures alert
# =====================
if data_failures:
    alerts.append(
        "⚠️ Failed to fetch data:\n- " + "\n- ".join(data_failures)
    )
# =====================
# Notify
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2)

if alerts:
    send_telegram("🚦 EGX Alerts 2:\n\n" + "\n".join(alerts))
else:
    send_telegram(f"ℹ️ No new signals\nLast candle: {last_candle_date}")

print("EGX ALERTS - Safe RSI/EMA Strategy + StopLoss + Updated Trends")

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
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False
        )
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
N = 60

for name, ticker in symbols.items():
    df = fetch_data(ticker)
    if df is None or len(df) < N:
        data_failures.append(name)
        continue

    last_candle_date = df.index[-1].date()

    # =====================
    # EMAs
    # =====================
    df["EMA3"]  = df["Close"].ewm(span=3, adjust=False).mean()
    df["EMA4"]  = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA5"]  = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA9"]  = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA25"] = df["Close"].ewm(span=25, adjust=False).mean()
    df["EMA35"] = df["Close"].ewm(span=35, adjust=False).mean()
    df["EMA45"] = df["Close"].ewm(span=45, adjust=False).mean()
    df["EMA55"] = df["Close"].ewm(span=55, adjust=False).mean()

    # =====================
    # RSI14
    # =====================
    df["RSI14"] = rsi(df["Close"], 14)
    last = df.iloc[-1]

    # =====================
    # Downtrend
    # =====================
    def is_downtrend(last):
        return (
            last["EMA55"] > last["EMA45"] > last["EMA35"] and
            last["Close"] < last["EMA25"] and
            last["Close"] < last["EMA35"]
        )

    # =====================
    # Uptrend (محسّن)
    # =====================
    def is_uptrend(last):
        return (
            last["EMA25"] > last["EMA35"] > last["EMA45"] and
            last["Close"] > last["EMA35"] and
            df["Close"].iloc[-2] > last["EMA35"]
        )

    def uptrend_signals(last):
        ema_cross = (
            last["EMA4"] > last["EMA9"] and
            df["EMA4"].iloc[-2] <= df["EMA9"].iloc[-2]
        )

        buy_signal = ema_cross and last["RSI14"] < 70

        sell_signal = (
            (last["EMA3"] < last["EMA5"] and last["Close"] < last["EMA9"]) or
            last["Close"] < last["EMA25"] or
            last["RSI14"] >= 85
        )

        return buy_signal, sell_signal

    # =====================
    # Sideways (New Strategy)
    # =====================
    def sideways_signals(df):
        last_N = df.tail(50)
        last_row = last_N.iloc[-1]
        price = last_row["Close"]
        rsi14 = last_row["RSI14"]
        ema3 = last_row["EMA3"]
        ema9 = last_row["EMA9"]

        # ---- Buy ----
        buy_signal = (rsi14 < 40) and (price <= ema3 or price <= ema9)

        # ---- Sell ----
        sell_signal = (rsi14 > 60) or (price >= ema3 or price >= ema9) or (ema3 < ema9 and df["EMA3"].iloc[-2] >= df["EMA9"].iloc[-2])

        return buy_signal, sell_signal

    # =====================
    # Decision
    # =====================
    if is_downtrend(last):
        buy_signal = sell_signal = False
        direction_text = "⚪ اتجاه هابط"
    elif is_uptrend(last):
        buy_signal, sell_signal = uptrend_signals(last)
        direction_text = "🟢 ترند صاعد"
    else:
        buy_signal, sell_signal = sideways_signals(df)
        direction_text = "🟡 اتجاه عرضي"

    prev_state = last_signals.get(name, {}).get("last_signal")

    if buy_signal and prev_state != "BUY":
        alerts.append(f"🟢 BUY | {name} | {last['Close']:.2f} | {last_candle_date}")
        new_signals[name] = {"last_signal": "BUY"}

    elif sell_signal and prev_state != "SELL":
        alerts.append(f"🔴 SELL | {name} | {last['Close']:.2f} | {last_candle_date}")
        new_signals[name] = {"last_signal": "SELL"}

# =====================
# Data failures alert
# =====================
if data_failures:
    alerts.append("⚠️ Failed to fetch data: " + ", ".join(data_failures))

# =====================
# Save & Notify
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2)

if alerts:
    send_telegram("🚦 EGX Alerts 2:\n\n" + "\n".join(alerts))
else:
    send_telegram(f"ℹ️ No new signals\nLast candle: {last_candle_date}")

print("EGX ALERTS - LuxAlgo Moving Average Converging + EMA50 FILTER")

import yfinance as yf
import requests
import os
import json
import numpy as np
import pandas as pd

# =====================
# Telegram settings
# =====================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(text):
    if not TOKEN or not CHAT_ID:
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
# LuxAlgo settings
# =====================
LENGTH = 80
INCR   = 12
FAST   = 12
K      = 1 / INCR

# =====================
# Load last signals
# =====================
SIGNALS_FILE = "last_signals.json"
try:
    with open(SIGNALS_FILE, "r") as f:
        last_signals = json.load(f)
except Exception:
    last_signals = {}

new_signals = last_signals.copy()
alerts = []
data_failures = []
last_candle_dates = []

# =====================
# Helpers
# =====================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

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
    except Exception:
        return None

# =====================
# Main Logic
# =====================
for name, ticker in symbols.items():
    df = fetch_data(ticker)

    if df is None or len(df) < LENGTH + 5:
        data_failures.append(name)
        continue

    last_candle_dates.append(df.index[-1].date())

    close = df["Close"].astype(float).to_numpy()
    high  = df["High"].astype(float).to_numpy()
    low   = df["Low"].astype(float).to_numpy()

    # ===== EMA 50 Trend Filter =====
    df["EMA50"] = ema(df["Close"], 50)
    ema50 = df["EMA50"].iloc[-1]

    ma  = np.zeros(len(close))
    fma = np.zeros(len(close))
    alpha = np.zeros(len(close))

    upper = np.maximum.accumulate(high)
    lower = np.minimum.accumulate(low)

    init_ma = np.full(len(close), np.nan)
    for i in range(LENGTH - 1, len(close)):
        init_ma[i] = np.mean(close[i - LENGTH + 1:i + 1])

    for i in range(len(close)):
        if i == 0 or np.isnan(init_ma[i]):
            ma[i] = close[i]
            fma[i] = close[i]
            alpha[i] = 0
            continue

        cross = (
            (close[i-1] <= ma[i-1] and close[i] > ma[i-1]) or
            (close[i-1] >= ma[i-1] and close[i] < ma[i-1])
        )

        if cross:
            alpha[i] = 2 / (LENGTH + 1)
        elif close[i] > ma[i-1] and upper[i] > upper[i-1]:
            alpha[i] = alpha[i-1] + K
        elif close[i] < ma[i-1] and lower[i] < lower[i-1]:
            alpha[i] = alpha[i-1] + K
        else:
            alpha[i] = alpha[i-1]

        ma[i] = ma[i-1] + alpha[i-1] * (close[i] - ma[i-1])

        if cross:
            fma[i] = (close[i] + fma[i-1]) / 2
        elif close[i] > ma[i]:
            fma[i] = max(close[i], fma[i-1]) + (close[i] - fma[i-1]) / FAST
        else:
            fma[i] = min(close[i], fma[i-1]) + (close[i] - fma[i-1]) / FAST

    prev_state = last_signals.get(name)

    # =====================
    # Final Signal Logic
    # =====================
    lux_buy  = fma[-1] > ma[-1]
    lux_sell = fma[-1] < ma[-1]

    buy_signal  = lux_buy and close[-1] > ema50
    sell_signal = lux_sell or close[-1] < ema50

    if buy_signal:
        curr_state = "BUY"
    elif sell_signal:
        curr_state = "SELL"
    else:
        continue

    if curr_state != prev_state:
        alerts.append(
            f"{'🟢 BUY' if curr_state == 'BUY' else '🔴 SELL'} | {name}\n"
            f"Price: {close[-1]:.2f}\n"
            f"Date: {df.index[-1].date()}"
        )
        new_signals[name] = curr_state

# =====================
# Data failure alert
# =====================
if data_failures:
    send_telegram("⚠️ فشل تحميل بيانات لبعض الأسهم: " + ", ".join(data_failures))

# =====================
# Save signals
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f)

# =====================
# Telegram output
# =====================
if alerts:
    send_telegram("🚨 EGX LuxAlgo + EMA50 Signals:\n\n" + "\n\n".join(alerts))
else:
    last_candle = max(last_candle_dates) if last_candle_dates else "غير متاح"
    send_telegram(
        "ℹ️ لا توجد إشارات جديدة\n\n"
        f"آخر شمعة محسوبة:\n📅 {last_candle}"
)

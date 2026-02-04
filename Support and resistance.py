print("EGX ALERTS - Auto Support/Resistance + RSI14 Sell + StopLoss Strategy")

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
N = 60  # last N days for support/resistance

for name, ticker in symbols.items():
    df = fetch_data(ticker)
    if df is None or len(df) < N:
        data_failures.append(name)
        continue

    last_candle_date = df.index[-1].date()
    last = df.iloc[-1]

    # حساب RSI14
    df["RSI14"] = rsi(df["Close"], 14)

    # ===== حساب الدعم والمقاومة تلقائي =====
    support = df["Low"].tail(N).min()
    resistance = df["High"].tail(N).max()

    # ===== SIGNALS =====
    buy_signal = last["Low"] <= support and last["Close"] > support
    sell_signal = (
        last["High"] >= resistance or
        last["RSI14"] >= 85 or
        last["Close"] < support  # كسر الدعم → SELL
    )

    prev_state = last_signals.get(name, {}).get("last_signal")

    if buy_signal and prev_state != "BUY":
        reason = f"Touched support ({support:.2f})"
        alerts.append(f"🟢 BUY | {name} | {last['Close']:.2f} | {last_candle_date} | Reason: {reason}")
        new_signals[name] = {
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "last_signal": "BUY"
        }

    elif sell_signal and prev_state != "SELL":
        if last["Close"] < support:
            reason = f"Stop Loss - broke support ({support:.2f})"
        elif last["High"] >= resistance:
            reason = f"Reached resistance ({resistance:.2f})"
        else:
            reason = "RSI14 >= 85"
        alerts.append(f"🔴 SELL | {name} | {last['Close']:.2f} | {last_candle_date} | Reason: {reason}")
        new_signals[name] = {
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "last_signal": "SELL"
        }

# =====================
# Data failure alert
# =====================
if data_failures:
    alerts.append("⚠️ Failed to fetch data: " + ", ".join(data_failures))

# =====================
# Save signals
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2)

# =====================
# Telegram output
# =====================
if alerts:
    send_telegram("📊 Egy supp&res:\n\n" + "\n".join(alerts))
else:
    send_telegram(f"ℹ️ No new signals\nLast candle: {last_candle_date}")

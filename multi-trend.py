print("TEST EMA CALCULATION - DSCW (Telegram)")

import yfinance as yf
import requests
import os

# =====================
# Telegram
# =====================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send(msg):
    if not TOKEN or not CHAT_ID:
        print(msg)
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# =====================
# Fetch Data
# =====================
ticker = "DSCW.CA"

df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False, progress=False)

if df is None or df.empty:
    send("❌ Failed to fetch data")
    exit()

# =====================
# Indicators
# =====================
df["EMA20_old"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA40_old"] = df["Close"].ewm(span=40, adjust=False).mean()
df["EMA100_old"] = df["Close"].ewm(span=100, adjust=False).mean()

df["EMA20_new"] = df["Close"].ewm(span=20, adjust=True).mean()
df["EMA40_new"] = df["Close"].ewm(span=40, adjust=True).mean()
df["EMA100_new"] = df["Close"].ewm(span=100, adjust=True).mean()

price = (df["High"] + df["Low"] + df["Close"]) / 3
df["EMA20_hlc"] = price.ewm(span=20, adjust=True).mean()
df["EMA40_hlc"] = price.ewm(span=40, adjust=True).mean()
df["EMA100_hlc"] = price.ewm(span=100, adjust=True).mean()

last = df.iloc[-1]

def trend(e20, e40, e100):
    if e20 > e40 > e100:
        return "↗️"
    elif e20 < e40 < e100:
        return "🔻"
    else:
        return "🔛"

msg = f"""
📊 DSCW EMA TEST

Close: {round(last["Close"], 4)}

=== OLD (adjust=False) ===
20: {round(last["EMA20_old"], 4)}
40: {round(last["EMA40_old"], 4)}
100: {round(last["EMA100_old"], 4)}
Trend: {trend(last["EMA20_old"], last["EMA40_old"], last["EMA100_old"])}

=== NEW (adjust=True) ===
20: {round(last["EMA20_new"], 4)}
40: {round(last["EMA40_new"], 4)}
100: {round(last["EMA100_new"], 4)}
Trend: {trend(last["EMA20_new"], last["EMA40_new"], last["EMA100_new"])}

=== HLC ===
20: {round(last["EMA20_hlc"], 4)}
40: {round(last["EMA40_hlc"], 4)}
100: {round(last["EMA100_hlc"], 4)}
Trend: {trend(last["EMA20_hlc"], last["EMA40_hlc"], last["EMA100_hlc"])}
"""

send(msg)

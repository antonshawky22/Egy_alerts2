print("TEST EMA CALCULATION - DSCW (FINAL)")

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

df = yf.download(
    ticker,
    period="6mo",
    interval="1d",
    auto_adjust=False,
    progress=False
)

if df is None or df.empty:
    send("❌ Failed to fetch data")
    exit()

# =====================
# FIX MultiIndex (مهم جدا)
# =====================
if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, "levels"):
    df.columns = df.columns.get_level_values(0)

# =====================
# Indicators
# =====================
close = df["Close"]

# الطريقة القديمة
df["EMA20_old"] = close.ewm(span=20, adjust=False).mean()
df["EMA40_old"] = close.ewm(span=40, adjust=False).mean()
df["EMA100_old"] = close.ewm(span=100, adjust=False).mean()

# الطريقة الجديدة (TradingView أقرب)
df["EMA20_new"] = close.ewm(span=20, adjust=True).mean()
df["EMA40_new"] = close.ewm(span=40, adjust=True).mean()
df["EMA100_new"] = close.ewm(span=100, adjust=True).mean()

# HLC3
hlc = (df["High"] + df["Low"] + df["Close"]) / 3
df["EMA20_hlc"] = hlc.ewm(span=20, adjust=True).mean()
df["EMA40_hlc"] = hlc.ewm(span=40, adjust=True).mean()
df["EMA100_hlc"] = hlc.ewm(span=100, adjust=True).mean()

# =====================
# Last Values (float 100%)
# =====================
e20_old = float(df["EMA20_old"].iloc[-1])
e40_old = float(df["EMA40_old"].iloc[-1])
e100_old = float(df["EMA100_old"].iloc[-1])

e20_new = float(df["EMA20_new"].iloc[-1])
e40_new = float(df["EMA40_new"].iloc[-1])
e100_new = float(df["EMA100_new"].iloc[-1])

e20_hlc = float(df["EMA20_hlc"].iloc[-1])
e40_hlc = float(df["EMA40_hlc"].iloc[-1])
e100_hlc = float(df["EMA100_hlc"].iloc[-1])

last_close = float(df["Close"].iloc[-1])

# =====================
# Trend Function (Safe)
# =====================
def trend(e20, e40, e100):
    if e20 > e40 > e100:
        return "↗️ صاعد"
    elif e20 < e40 < e100:
        return "🔻 هابط"
    else:
        return "🔛 عرضي"

# =====================
# Message
# =====================
msg = f"""
📊 DSCW EMA TEST

💰 Close: {round(last_close, 4)}

━━━━━━━━━━━━━━
🔹 OLD (adjust=False)
20: {round(e20_old, 4)}
40: {round(e40_old, 4)}
100: {round(e100_old, 4)}
📈 {trend(e20_old, e40_old, e100_old)}

━━━━━━━━━━━━━━
🔹 NEW (adjust=True)
20: {round(e20_new, 4)}
40: {round(e40_new, 4)}
100: {round(e100_new, 4)}
📈 {trend(e20_new, e40_new, e100_new)}

━━━━━━━━━━━━━━
🔹 HLC3 (TradingView style)
20: {round(e20_hlc, 4)}
40: {round(e40_hlc, 4)}
100: {round(e100_hlc, 4)}
📈 {trend(e20_hlc, e40_hlc, e100_hlc)}
"""

send(msg)

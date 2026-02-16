print("EGX ALERTS - Final Stable Version with Multi-Trend Logic")

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
    "OFH":"OFH.CA","OLFI":"OLFI.CA","EMFD":"EMFD.CA","ETEL":"ETEL.CA",
    "EAST":"EAST.CA","EFIH":"EFIH.CA","ABUK":"ABUK.CA","OIH":"OIH.CA",
    "SWDY":"SWDY.CA","ISPH":"ISPH.CA","ATQA":"ATQA.CA","MTIE":"MTIE.CA",
    "ELEC":"ELEC.CA","HRHO":"HRHO.CA","ORWE":"ORWE.CA","JUFO":"JUFO.CA",
    "DSCW":"DSCW.CA","SUGR":"SUGR.CA","ELSH":"ELSH.CA","RMDA":"RMDA.CA",
    "RAYA":"RAYA.CA","EEII":"EEII.CA","MPCO":"MPCO.CA","GBCO":"GBCO.CA",
    "TMGH":"TMGH.CA","ORHD":"ORHD.CA","AMOC":"AMOC.CA","FWRY":"FWRY.CA",
    "COMI":"COMI.CA","ADIB":"ADIB.CA","PHDC":"PHDC.CA",
    "MCQE":"MCQE.CA","SKPC":"SKPC.CA","EGAL":"EGAL.CA"
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
    return 100 - (100 / (1 + rs))

# =====================
# Parameters
# =====================
EMA_PERIOD = 60
LOOKBACK = 30
BULLISH_THRESHOLD = 0.65
BEARISH_THRESHOLD = 0.65
EMA_FORCED_SELL = 25

# =====================
# Containers
# =====================
section_up = []
section_side = []
section_down = []
anything_new = False

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
    df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA25"] = df["Close"].ewm(span=EMA_FORCED_SELL, adjust=False).mean()

    recent_closes = df["Close"].iloc[-LOOKBACK:]
    recent_ema60 = df["EMA60"].iloc[-LOOKBACK:]

    bullish_ratio = (recent_closes > recent_ema60).sum() / LOOKBACK
    bearish_ratio = (recent_closes < recent_ema60).sum() / LOOKBACK

    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    last_ema4 = df["EMA4"].iloc[-1]
    prev_ema4 = df["EMA4"].iloc[-2]
    last_ema9 = df["EMA9"].iloc[-1]
    prev_ema9 = df["EMA9"].iloc[-2]

    prev_data = last_signals.get(name, {})
    prev_signal = prev_data.get("last_signal", "")
    prev_trend = prev_data.get("trend", "")
    prev_forced = prev_data.get("last_forced_sell", False)
    prev_side = prev_data.get("last_side_signal", "")

    buy_signal = sell_signal = False
    side_signal = ""
    percent_side = None
    forced_sell_mark = ""
    trend_changed_mark = ""

    # =====================
    # Determine Trend
    # =====================
    if bullish_ratio >= BULLISH_THRESHOLD:
        trend = "↗️"
    elif bearish_ratio >= BEARISH_THRESHOLD:
        trend = "🔻"
    else:
        trend = "🔛"

    # =====================
    # Trend Change Mark (مرة واحدة)
    # =====================
    if prev_trend and prev_trend != trend:
        trend_changed_mark = "🚧 "
        anything_new = True

    # =====================
    # Forced Sell (مرة واحدة فقط)
    # =====================
    last_forced = False
    if last_close < df["EMA25"].iloc[-1]:
        if not prev_forced:
            forced_sell_mark = "🚨"
            sell_signal = True
            anything_new = True
            last_forced = True
    else:
        last_forced = False

    # =====================
    # Strategy
    # =====================
    if trend == "↗️":
        if prev_ema4 <= prev_ema9 and last_ema4 > last_ema9:
            if prev_signal != "BUY":
                buy_signal = True
                anything_new = True
        elif prev_ema4 >= prev_ema9 and last_ema4 < last_ema9:
            if prev_signal != "SELL":
                sell_signal = True
                anything_new = True

    elif trend == "🔛":
        high_lookback = df["Close"].iloc[-EMA_PERIOD:]
        low_lookback = df["Close"].iloc[-EMA_PERIOD:]
        high_threshold = high_lookback.max() * 0.95
        low_threshold = low_lookback.min() * 1.05

        if last_close >= high_threshold:
            if prev_side != "🔴":
                side_signal = "🔴"
                percent_side = (high_lookback.max() - last_close) / high_lookback.max() * 100
                anything_new = True
        elif last_close <= low_threshold:
            if prev_side != "🟢":
                side_signal = "🟢"
                percent_side = (last_close - low_lookback.min()) / low_lookback.min() * 100
                anything_new = True

    # =====================
    # Prepare messages
    # =====================
    if trend == "↗️" and (buy_signal or sell_signal):
        mark = "🟢" if buy_signal else "🔴"
        section_up.append(f"{trend_changed_mark}{forced_sell_mark}{mark} {name} | {last_close:.2f} | {last_candle_date}")

    if trend == "🔛" and side_signal:
        section_side.append(f"{trend_changed_mark}{forced_sell_mark}{side_signal} {name} | {last_close:.2f} | {last_candle_date} | {percent_side:.2f}%")

    if trend == "🔻" and prev_trend != "🔻":
        section_down.append(f"{trend_changed_mark}{forced_sell_mark}{name} | {last_close:.2f} | {last_candle_date}")
        anything_new = True

    # =====================
    # Save state
    # =====================
    new_signals[name] = {
        "last_signal": "BUY" if buy_signal else "SELL" if sell_signal else prev_signal,
        "trend": trend,
        "last_forced_sell": last_forced,
        "last_side_signal": side_signal
    }

# =====================
# Save File
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

# =====================
# Send only if new
# =====================
if anything_new:
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

    send_telegram("\n".join(alerts))
else:
    print("No new signals")

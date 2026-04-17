print("EGX ALERTS - Cycle Based Version (No Repetition + Smart Trend Handling)")

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
# Symbols
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
# Load signals
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
SIDE_CLOSE_PERCENT = 0.10
RSI_SELL = 83

# =====================
# Containers
# =====================
section_up = []
section_side = []
section_down = []

# =====================
# Main Loop
# =====================
for name, ticker in symbols.items():

    df = fetch_data(ticker)
    if df is None or len(df) < 100:
        data_failures.append(name)
        continue

    last_candle_date = df.index[-1].date()

    # =====================
    # EMA (UPDATED)
    # =====================
    df["EMA10"] = df["Close"].ewm(span=10, adjust=True).mean()
    df["EMA15"] = df["Close"].ewm(span=15, adjust=True).mean()
    df["EMA30"] = df["Close"].ewm(span=30, adjust=True).mean()

    df["EMA4"] = df["Close"].ewm(span=4, adjust=True).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=True).mean()

    df["RSI14"] = rsi(df["Close"], 14)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    last_close = last["Close"]

    # =====================
    # State
    # =====================
    prev_data = last_signals.get(name, {})

    in_position = prev_data.get("in_position", False)
    entry_price = prev_data.get("entry_price", None)
    prev_trend = prev_data.get("trend", "")

    buy_signal = False
    sell_signal = False
    side_signal = ""
    percent_side = None

    # =====================
    # Trend (UPDATED)
    # =====================
    if last["EMA10"] > last["EMA15"] > last["EMA30"]:
        trend = "↗️"
    elif last["EMA10"] < last["EMA15"] < last["EMA30"]:
        trend = "🔻"
    else:
        trend = "🔛"

    trend_changed = trend != prev_trend

    # =====================
    # SIDE → TREND CONVERSION
    # =====================
    converted_to_trend = (
        prev_trend == "🔛" and
        trend == "↗️" and
        in_position
    )

    # =====================
    # SIDE LOGIC
    # =====================
    if trend == "🔛" and not converted_to_trend:

        high = df["High"].iloc[-15:].max()
        low = df["Low"].iloc[-15:].min()

        from_high = (high - last_close) / high
        from_low = (last_close - low) / low

        if not in_position and from_low <= SIDE_CLOSE_PERCENT:
            buy_signal = True
            side_signal = "🟢"
            percent_side = from_low * 100
            in_position = True
            entry_price = last_close

        elif in_position and from_high <= SIDE_CLOSE_PERCENT:
            sell_signal = True
            side_signal = "🔴"
            percent_side = from_high * 100
            in_position = False
            entry_price = None

        elif in_position and last_close < entry_price * 0.96:
            sell_signal = True
            side_signal = "🔴💥"
            in_position = False
            entry_price = None

    # =====================
    # UP TREND LOGIC
    # =====================
    if trend == "↗️":

        if not in_position and last["RSI14"] < 60 and last_close > last["EMA15"]:
            buy_signal = True
            in_position = True
            entry_price = last_close

    # =====================
    # EXIT LOGIC (FIXED INDENTATION)
    # =====================
    if in_position:

        cross_down = prev["EMA4"] >= prev["EMA9"] and last["EMA4"] < last["EMA9"]

        stop_loss = last_close < entry_price * 0.96

        trend_flip = (prev_trend == "↗️" and trend in ["🔛", "🔻"])

        rsi_sell = last["RSI14"] > RSI_SELL

        if stop_loss or cross_down or trend_flip or rsi_sell:
            sell_signal = True
            in_position = False
            entry_price = None

    # =====================
    # DOWN TREND EXIT
    # =====================
    if trend == "🔻" and in_position:
        sell_signal = True
        in_position = False
        entry_price = None

    # =====================
    # Messages
    # =====================
    trend_mark = "🚧 " if trend_changed else ""

    if trend == "↗️" and (buy_signal or sell_signal):
        mark = "🟢" if buy_signal else "🔴"
        section_up.append(f"{trend_mark}{mark} {name} | {last_close:.2f} | {last_candle_date}")

    elif trend == "🔛" and side_signal:
        p = f"{percent_side:.2f}%" if percent_side else ""
        section_side.append(f"{trend_mark}{side_signal} {name} | {last_close:.2f} | {last_candle_date} | {p}")

    elif trend == "🔻" and trend_changed:
        section_down.append(f"{trend_mark}{name} | {last_close:.2f} | {last_candle_date}")

    # =====================
    # Save State
    # =====================
    new_signals[name] = {
        "trend": trend,
        "in_position": in_position,
        "entry_price": entry_price
    }

# =====================
# Message
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

if data_failures:
    alerts.append(f"\n⚠️ Data fail (last candle: {last_candle_date})")
    alerts.extend(["- " + s for s in data_failures])

elif not section_up and not section_side and not section_down:
    alerts.append(f"ℹ️ No new signals for today (last candle: {last_candle_date})")

# =====================
# Save + Send
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

send_telegram("\n".join(alerts))

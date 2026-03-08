print("EGX ALERTS - Market Structure Trend & Signals")

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
EMA_FORCED_SELL = 60

SIDE_CLOSE_PERCENT = 0.05
RSI_SELL = 82

MARKET_STRUCTURE_LOOKBACK = 120  # لتحديد أعلى قمتين وأقل قاعين
SWING_PERIOD = 20  # أقل عدد شموع بين القمم/القاع لتقليل ضوضاء السوق

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
    if df is None or len(df) < MARKET_STRUCTURE_LOOKBACK:
        data_failures.append(name)
        continue

    last_candle_date = df.index[-1].date()

    # =====================
    # Indicators
    # =====================
    df["EMA60"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA60_forced"] = df["Close"].ewm(span=EMA_FORCED_SELL, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)

    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    last_ema4 = df["EMA4"].iloc[-1]
    prev_ema4 = df["EMA4"].iloc[-2]
    last_ema9 = df["EMA9"].iloc[-1]
    prev_ema9 = df["EMA9"].iloc[-2]

    # =====================
    # Previous signals
    # =====================
    prev_data = last_signals.get(name, {})
    prev_signal = prev_data.get("last_signal", "")
    prev_trend = prev_data.get("trend", "")
    prev_forced = prev_data.get("last_forced_sell", False)
    prev_side_actual = prev_data.get("last_side_signal_actual", "")
    prev_side_buy_price = prev_data.get("prev_side_buy_price", None)

    buy_signal = sell_signal = False
    side_signal = ""
    percent_side = None

    # =====================
    # Market Structure Trend - Swing High/Low
    # =====================
    lookback_df = df.iloc[-MARKET_STRUCTURE_LOOKBACK:]

    # البحث عن القمم والقاع بفاصل SWING_PERIOD لتقليل الضوضاء
    highs = []
    lows = []

    for i in range(SWING_PERIOD, len(lookback_df) - SWING_PERIOD):
        window = lookback_df["Close"].iloc[i-SWING_PERIOD:i+SWING_PERIOD+1]
        if lookback_df["Close"].iloc[i] == window.max():
            highs.append((lookback_df.index[i], lookback_df["Close"].iloc[i]))
        if lookback_df["Close"].iloc[i] == window.min():
            lows.append((lookback_df.index[i], lookback_df["Close"].iloc[i]))

    if len(highs) >= 2 and len(lows) >= 2:
        high_prev_val = highs[-2][1]
        high_latest_val = highs[-1][1]
        low_prev_val = lows[-2][1]
        low_latest_val = lows[-1][1]

        if high_latest_val > high_prev_val and low_latest_val > low_prev_val:
            trend = "↗️"
        elif high_latest_val < high_prev_val and low_latest_val < low_prev_val:
            trend = "🔻"
        else:
            trend = "🔛"
    else:
        trend = "🔛"

    # =====================
    # إشارات داخل الاتجاه
    # =====================
    # الصاعد ↗️
    if trend == "↗️":
        if prev_ema4 <= prev_ema9 and last_ema4 > last_ema9:
            buy_signal = True
        elif prev_ema4 >= prev_ema9 and last_ema4 < last_ema9:
            if df["RSI14"].iloc[-1] > RSI_SELL:
                sell_signal = True

    # الهابط 🔻
    elif trend == "🔻":
        if prev_ema4 >= prev_ema9 and last_ema4 < last_ema9:
            sell_signal = True

    # العرضي 🔛
    else:
        high_threshold = lookback_df["Close"].max() * (1 - SIDE_CLOSE_PERCENT)
        low_threshold = lookback_df["Close"].min() * (1 + SIDE_CLOSE_PERCENT)
        if last_close <= low_threshold:
            buy_signal = True
            side_signal = "🟢"
            percent_side = (last_close - lookback_df["Close"].min()) / lookback_df["Close"].min() * 100
            prev_side_buy_price = last_close
        elif last_close >= high_threshold:
            sell_signal = True
            side_signal = "🔴"
            percent_side = (lookback_df["Close"].max() - last_close) / lookback_df["Close"].max() * 100
        if prev_side_buy_price and last_close < prev_side_buy_price:
            sell_signal = True
            side_signal = "🔴💥"

    # =====================
    # Forced Sell 🚨
    # =====================
    forced_sell_mark = ""
    if last_close < df["EMA60_forced"].iloc[-1] and not prev_forced:
        sell_signal = True
        buy_signal = False
        forced_sell_mark = "🚨"
        last_forced = True
    else:
        last_forced = prev_forced

    # =====================
    # Prevent repeated signals
    # =====================
    if trend == prev_trend and buy_signal and prev_signal == "BUY":
        buy_signal = False
    if trend == prev_trend and sell_signal and prev_signal == "SELL":
        sell_signal = False
    if trend == "🔛":
        if side_signal == prev_side_actual:
            side_signal = ""
        else:
            prev_side_actual = side_signal

    # =====================
    # Prepare messages
    # =====================
    if trend == "↗️" and (buy_signal or sell_signal):
        mark = "🟢" if buy_signal else "🔴"
        section_up.append(f"{mark} {name} | {last_close:.2f} | {last_candle_date}")
    elif trend == "🔛" and side_signal:
        section_side.append(f"{side_signal} {name} | {last_close:.2f} | {last_candle_date} | {percent_side:.2f}%")
    elif trend == "🔻" and sell_signal:
        section_down.append(f"🔴 {name} | {last_close:.2f} | {last_candle_date}")

    # =====================
    # Update last signals
    # =====================
    new_signals[name] = {
        "last_signal": "BUY" if buy_signal else "SELL" if sell_signal else prev_signal,
        "trend": trend,
        "last_forced_sell": last_forced,
        "last_side_signal_actual": prev_side_actual,
        "prev_side_buy_price": prev_side_buy_price
    }

# =====================
# Compile Message
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

if not section_up and not section_side and not section_down:
    alerts.append(f"ℹ️ No new signals\nlast candle: {last_candle_date}")

if data_failures:
    alerts.append("\n⚠️ Failed to fetch data:\n- " + "\n- ".join(data_failures))

# =====================
# Save & Notify
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

if alerts:
    send_telegram("\n".join(alerts))

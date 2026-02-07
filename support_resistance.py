print("EGX ALERTS - Safe Support/Resistance + RSI14 Sell + StopLoss Strategy + Updated Trends")

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

    # =====================
    # حساب كل الموفنجات المطلوبة
    # =====================
    df["EMA3"] = df["Close"].ewm(span=3, adjust=False).mean()
    df["EMA4"] = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA25"] = df["Close"].ewm(span=25, adjust=False).mean()
    df["EMA35"] = df["Close"].ewm(span=35, adjust=False).mean()
    df["EMA45"] = df["Close"].ewm(span=45, adjust=False).mean()
    df["EMA55"] = df["Close"].ewm(span=55, adjust=False).mean()

    # =====================
    # حساب RSI14
    # =====================
    df["RSI14"] = rsi(df["Close"], 14)

    last = df.iloc[-1]

    # =====================
    # اتجاه هابط
    # =====================
    def is_downtrend(last):
        ema25 = last["EMA25"]
        ema35 = last["EMA35"]
        ema45 = last["EMA45"]
        ema55 = last["EMA55"]
        price = last["Close"]

        if ema55 > ema45 > ema35 and price < ema25 and price < ema35:
            return True
        return False

    # =====================
    # اتجاه صاعد
    # =====================
    def is_uptrend(last):
        ema25 = last["EMA25"]
        ema35 = last["EMA35"]
        price = last["Close"]
        prev1_price = df["Close"].iloc[-2]

        if ema25 > ema35 and price > ema35 and prev1_price > ema35:
            return True
        return False

    def uptrend_signals(last):
        price = last["Close"]
        ema4 = last["EMA4"]
        ema9 = last["EMA9"]
        ema3 = last["EMA3"]
        ema5 = last["EMA5"]
        ema25 = last["EMA25"]
        rsi14 = last["RSI14"]

        # ---- شراء ----
        
    buy_signal = (
    prev["EMA4"] <= prev["EMA9"] and
    last["EMA4"] > last["EMA9"] and
    last["RSI14"] < 70
    )
    # ---- بيع ----
        sell_signal = ema3 < ema5 or price < ema25 or rsi14 >= 85

        return buy_signal, sell_signal

    # =====================
    # اتجاه عرضي
    # =====================
    SIDE_PERIOD = 50  # آخر 50 شمعة
    TOLERANCE = 0.01  # 1% حول الدعم/المقاومة
    STOPLOSS_PCT = 0.02  # كسر الدعم → وقف خسارة 2%

    def sideways_signals(df):
        last_N = df.tail(SIDE_PERIOD)
        current_price = last_N["Close"].iloc[-1]

        SUPPORT = last_N["Low"].min()
        RESISTANCE = last_N["High"].max()

        # ---- شراء ----
        buy_signal = abs(current_price - SUPPORT)/SUPPORT <= TOLERANCE

        # ---- بيع ----
        sell_signal = (
            abs(current_price - RESISTANCE)/RESISTANCE <= TOLERANCE or
            current_price < SUPPORT * (1 - STOPLOSS_PCT)
        )

        return buy_signal, sell_signal, SUPPORT, RESISTANCE

    # =====================
    # الحصول على الإشارات
    # =====================
    if is_downtrend(last):
        direction_text = "⚪ اتجاه هابط – تجاهل السهم"
        buy_signal, sell_signal, support, resistance = False, False, None, None
    elif is_uptrend(last):
        direction_text = "🟢 ترند صاعد"
        buy_signal, sell_signal = uptrend_signals(last)
        support, resistance = None, None
    else:
        direction_text = "🟡 اتجاه عرضي"
        buy_signal, sell_signal, support, resistance = sideways_signals(df)

    # =====================
    # حفظ الإشارات الجديدة
    # =====================
    prev_state = last_signals.get(name, {}).get("last_signal")

    if buy_signal and prev_state != "BUY":
        reason = f"Touched support ({support:.2f})" if support else "EMA4/EMA9 cross"
        alerts.append(f"🟢 BUY | {name} | {last['Close']:.2f} | {last_candle_date} | Reason: {reason}")
        new_signals[name] = {
            "support": round(support,2) if support else None,
            "resistance": round(resistance,2) if resistance else None,
            "last_signal": "BUY"
        }

    elif sell_signal and prev_state != "SELL":
        if direction_text == "🟢 ترند صاعد":
            if last["Close"] < last["EMA25"]:
                reason = f"Price < EMA25"
            elif last["RSI14"] >= 85:
                reason = "RSI14 >= 85"
            else:
                reason = "EMA3 < EMA5"
        else:
            if current_price < SUPPORT * (1 - STOPLOSS_PCT):
                reason = f"Stop Loss - broke support ({SUPPORT:.2f})"
            else:
                reason = f"Near resistance ({RESISTANCE:.2f})"
        alerts.append(f"🔴 SELL | {name} | {last['Close']:.2f} | {last_candle_date} | Reason: {reason}")
        new_signals[name] = {
            "support": round(support,2) if support else None,
            "resistance": round(resistance,2) if resistance else None,
            "last_signal": "SELL"
        }

# =====================
# إخطار فشل البيانات
# =====================
if data_failures:
    alerts.append("⚠️ Failed to fetch data: " + ", ".join(data_failures))

# =====================
# حفظ الإشارات
# =====================
with open(SIGNALS_FILE, "w") as f:
    json.dump(new_signals, f, indent=2)

# =====================
# إخطار Telegram
# =====================
if alerts:
    send_telegram("🚦🚦 EGX Alerts 2:\n\n" + "\n".join(alerts))
else:
    send_telegram(f"egx alerts 2 ℹ️ No new signals\nLast candle: {last_candle_date}")

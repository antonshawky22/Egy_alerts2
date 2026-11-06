import json
import os
import joblib
import pandas as pd
import numpy as np
import requests

# =====================
# Telegram Settings
# =====================
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        print("❌ Telegram credentials not set")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        if response.status_code != 200:
            print(f"❌ Telegram send failed: {response.text}")
    except Exception as e:
        print(f"❌ Telegram send exception: {e}")

# =====================
# Files & Model
# =====================
DB_FILE = "egx_history_database_v2.json"
MODEL_FILE = "ai_stock_model.pkl"
SIGNALS_FILE = "ai_last_signals.json"
TRADES_FILE = "ai_trades.json"

# إعدادات عتبة الثقة وقواعد حماية الأرباح
CONFIDENCE_THRESHOLD = 0.70  # نسبة الثقة المطلوبة للدخول (70%)
TRAILING_STOP_PCT = 0.05     # نسبة التراجع المسموحة من أعلى سعر قمة لحماية الأرباح (5%)

# =====================
# Trade Logging Functions
# =====================
def log_buy(symbol, price, date, confidence):
    trade = {
        "symbol": symbol,
        "entry_price": price,
        "entry_date": date,
        "entry_confidence": round(float(confidence) * 100, 2),
        "signal_type": "AI_BUY"
    }

    data = []
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                data = []

    data.append(trade)

    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def log_sell(symbol, price, date, exit_reason):
    if not os.path.exists(TRADES_FILE):
        return

    with open(TRADES_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except:
            data = []

    # البحث عن آخر صفقة مفتوحة لهذا السهم لإغلاقها
    for trade in reversed(data):
        if trade["symbol"] == symbol and "exit_price" not in trade:
            entry_price = trade["entry_price"]
            if entry_price and entry_price != 0:
                profit_pct = ((price - entry_price) / entry_price) * 100
            else:
                profit_pct = 0
                
            trade["exit_price"] = price
            trade["exit_date"] = date
            trade["exit_reason"] = exit_reason
            trade["profit_pct"] = round(profit_pct, 2)
            break 

    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =====================
# Load AI Model & Database
# =====================
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ ملف نموذج الذكاء الاصطناعي {MODEL_FILE} غير موجود!")

model = joblib.load(MODEL_FILE)
print(f"🤖 AI Model loaded successfully from {MODEL_FILE}")

try:
    with open(DB_FILE, "r", encoding="utf-8") as f:
        raw_database = json.load(f)
    print(f"💾 Market database loaded successfully from {DB_FILE}")
except Exception as e:
    raise FileNotFoundError(f"❌ تعذر قراءة قاعدة البيانات {DB_FILE}: {e}")

try:
    with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
        last_signals = json.load(f)
except:
    last_signals = {}

new_signals = last_signals.copy()
buy_alerts = []
sell_alerts = []
data_failures = []
global_last_date = "Unknown Date"

# مساعدة حساب الـ RSI
def calculate_rsi(series, period=14):
    if len(series) < period:
        return pd.Series(50, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# =====================
# Main Loop
# =====================
for symbol, stock_content in raw_database.items():
    if symbol == "EGX30":
        continue
        
    if not stock_content or "data" not in stock_content or "columns" not in stock_content:
        data_failures.append(symbol)
        continue

    df = pd.DataFrame.from_dict(stock_content["data"], orient="index", columns=stock_content["columns"])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)

    if len(df) < 50:
        data_failures.append(symbol)
        continue

    last_candle_date = df.index[-1].strftime('%Y-%m-%d')
    global_last_date = last_candle_date
    last_close = df["Close"].iloc[-1]
    last_high = df["High"].iloc[-1]

    # حساب الميزات التقنية
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["ATR14"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR14_Pct"] = df["ATR14"] / df["Close"]
    df["Volume_Ratio20"] = df["Volume"] / df["Volume"].rolling(20).mean()
    
    for p in [5, 10, 20, 50, 100]:
        df[f"Return_{p}"] = df["Close"].pct_change(p) * 100

    for span in [20, 50, 100]:
        ema = df["Close"].ewm(span=span, adjust=False).mean()
        df[f"Distance_EMA{span}_Pct"] = ((df["Close"] - ema) / ema) * 100

    rolling_high20 = df["High"].rolling(20).max()
    rolling_low20 = df["Low"].rolling(20).min()
    df["Distance_Resistance20_Pct"] = ((rolling_high20 - df["Close"]) / df["Close"]) * 100
    df["Distance_Support20_Pct"] = ((df["Close"] - rolling_low20) / rolling_low20) * 100

    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    ema100 = df["Close"].ewm(span=100, adjust=False).mean()
    df["Trend_20_50"] = (ema20 > ema50).astype(int)
    df["Trend_50_100"] = (ema50 > ema100).astype(int)

    latest_row = df.iloc[[-1]].copy()

    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
        for feat in expected_features:
            if feat not in latest_row.columns:
                latest_row[feat] = 0.0
        X_pred = latest_row[expected_features].select_dtypes(include=['number', 'bool'])
    else:
        exclude = ["date", "ticker", "symbol", "target", "Future_Return_Pct"]
        X_pred = latest_row[[c for c in latest_row.columns if c not in exclude]].select_dtypes(include=['number', 'bool'])

    # حساب احتمالية التنبؤ (Probabilities)
    try:
        probs = model.predict_proba(X_pred)[0]
        prob_buy = probs[1] if len(probs) > 1 else 0.5
    except Exception as e:
        print(f"⚠️ Prediction error for {symbol}: {e}")
        continue

    prev_data = last_signals.get(symbol, {})
    in_position = prev_data.get("in_position", False)
    entry_price = prev_data.get("entry_price", last_close)
    highest_price = prev_data.get("highest_price", last_high)

    # تحديث أعلى سعر وصل إليه السهم أثناء الصفقة
    if in_position:
        if last_high > highest_price:
            highest_price = last_high

    buy_signal = False
    sell_signal = False
    exit_reason = ""

    # منطق الدخول الصارم (برشاقة الثقة المطلوبة)
    if prob_buy >= CONFIDENCE_THRESHOLD and not in_position:
        buy_signal = True
        in_position = True
        entry_price = last_close
        highest_price = last_high
        buy_alerts.append(f"🟢 {symbol} | السعر: {last_close:.2f} | الثقة: {prob_buy*100:.1f}%")
        log_buy(symbol, last_close, last_candle_date, prob_buy)

    # منطق الخروج (إما بانقلاب إشارة النموذج < 50% أو بتفعيل حماية الأرباح Trailing Stop)
    elif in_position:
        # حساب نسبة التراجع من أعلى قمة وصل لها السهم
        drawdown_from_peak = (highest_price - last_close) / highest_price if highest_price > 0 else 0
        
        if prob_buy < 0.50:
            sell_signal = True
            in_position = False
            exit_reason = "AI_Model_Inversion (انعكاس النموذج)"
        elif drawdown_from_peak >= TRAILING_STOP_PCT:
            sell_signal = True
            in_position = False
            exit_reason = f"Trailing_Stop (حماية الأرباح بتراجع {drawdown_from_peak*100:.1f}%)"

        if sell_signal:
            profit_calc = ((last_close - entry_price) / entry_price) * 100
            emoji = "🟢" if profit_calc >= 0 else "🔴"
            sell_alerts.append(f"🔻 {symbol} | خروج عند: {last_close:.2f} | النتيجة: {emoji} {profit_calc:.2f}% | السبب: {exit_reason}")
            log_sell(symbol, last_close, last_candle_date, exit_reason)

    # حفظ الحالة الحالية
    new_signals[symbol] = {
        "in_position": in_position,
        "entry_price": float(entry_price) if in_position else 0.0,
        "highest_price": float(highest_price) if in_position else 0.0,
        "last_prediction": float(prob_buy),
        "last_price": float(last_close),
        "last_date": last_candle_date
    }

# =====================
# Telegram Report Dispatch
# =====================
report = [f"🤖 AI Stock Model Pro Report ({global_last_date}):\n"]

if buy_alerts:
    report.append("🚀 إشارات شراء جديدة (AI BUY - High Confidence):")
    report.extend(buy_alerts)

if sell_alerts:
    report.append("\n🔻 إشارات خروج / جني أرباح / Stop Loss:")
    report.extend(sell_alerts)

if not buy_alerts and not sell_alerts:
    report.append(f"ℹ️ لا توجد إشارات جديدة اليوم. السوق مستقر (آخر تاريخ: {global_last_date})")

if data_failures:
    report.append(f"\n⚠️ تعذر قراءة بيانات بعض الأسهم: {len(data_failures)}")

with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

final_message = "\n".join(report)
send_telegram(final_message)
print("🏁 AI Prediction & Alert Dispatch Complete with Pro Rules.")

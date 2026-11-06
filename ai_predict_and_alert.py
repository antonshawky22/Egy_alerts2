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

# =====================
# Trade Logging Functions (مأخوذة من استراتيجيتك)
# =====================
def log_buy(symbol, price, date):
    trade = {
        "symbol": symbol,
        "entry_price": price,
        "entry_date": date,
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

def log_sell(symbol, price, date):
    if not os.path.exists(TRADES_FILE):
        return

    with open(TRADES_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except:
            data = []

    # البحث عن آخر صفقة مفتوحة لهذا السهم لتحديثها وإغلاقها
    for trade in reversed(data):
        if trade["symbol"] == symbol and "exit_price" not in trade:
            entry_price = trade["entry_price"]
            if entry_price and entry_price != 0:
                profit_pct = ((price - entry_price) / entry_price) * 100
            else:
                profit_pct = 0
                
            trade["exit_price"] = price
            trade["exit_date"] = date
            trade["profit_pct"] = round(profit_pct, 2)
            break 

    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =====================
# Load AI Model & Database
# =====================
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"❌ ملف نموذج الذكاء الاصطناعي {MODEL_FILE} غير موجود! تأكد من تشغيل التدريب أولاً.")

model = joblib.load(MODEL_FILE)
print(f"🤖 AI Model loaded successfully from {MODEL_FILE}")

try:
    with open(DB_FILE, "r", encoding="utf-8") as f:
        raw_database = json.load(f)
    print(f"💾 Market database loaded successfully from {DB_FILE}")
except Exception as e:
    raise FileNotFoundError(f"❌ تعذر قراءة قاعدة البيانات {DB_FILE}: {e}")

# تحميل الإشارات السابقة لتفادي التكرار وتتبع الحالة
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

# مساعدة حساب الـ RSI للتطابق
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
# Main Loop (تحليل وتنبؤ الأسهم بالذكاء الاصطناعي)
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

    if len(df) < 50: # تأمين الحد الأدنى للمؤشرات
        data_failures.append(symbol)
        continue

    last_candle_date = df.index[-1].strftime('%Y-%m-%d')
    global_last_date = last_candle_date
    last_close = df["Close"].iloc[-1]

    # حساب نفس الميزات التقنية التي تدرب عليها النموذج لضمان التطابق التام
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["ATR14"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR14_Pct"] = df["ATR14"] / df["Close"]
    df["Volume_Ratio20"] = df["Volume"] / df["Volume"].rolling(20).mean()
    
    for p in [5, 10, 20, 50, 100]:
        df[f"Return_{p}"] = df["Close"].pct_change(p) * 100

    for span in [20, 50, 100]:
        ema = df["Close"].ewm(span=span, adjust=False).mean()
        df[f"Distance_EMA{span}_Pct"] = ((df["Close"] - ema) / ema) * 100

    # المستويات
    rolling_high20 = df["High"].rolling(20).max()
    rolling_low20 = df["Low"].rolling(20).min()
    df["Distance_Resistance20_Pct"] = ((rolling_high20 - df["Close"]) / df["Close"]) * 100
    df["Distance_Support20_Pct"] = ((df["Close"] - rolling_low20) / rolling_low20) * 100

    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    ema100 = df["Close"].ewm(span=100, adjust=False).mean()
    df["Trend_20_50"] = (ema20 > ema50).astype(int)
    df["Trend_50_100"] = (ema50 > ema100).astype(int)

    # أخذ أحدث شمعة فقط للتنبؤ
    latest_row = df.iloc[[-1]].copy()

    # استخراج الميزات بنفس ترتيب تدريب النموذج (model.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
        # التأكد من توفر جميع الأعمدة المطلوبة
        for feat in expected_features:
            if feat not in latest_row.columns:
                latest_row[feat] = 0.0
        X_pred = latest_row[expected_features].select_dtypes(include=['number', 'bool'])
    else:
        # احتياطياً لو الخاصية غير موجودة
        exclude = ["date", "ticker", "symbol", "target", "Future_Return_Pct"]
        X_pred = latest_row[[c for c in latest_row.columns if c not in exclude]].select_dtypes(include=['number', 'bool'])

    # توقع الذكاء الاصطناعي (1 = فرصة شراء صاعدة، 0 = لا توجد إشارة / خروج)
    try:
        ai_pred = model.predict(X_pred)[0]
    except Exception as e:
        print(f"⚠️ Prediction error for {symbol}: {e}")
        continue

    # حالة السهم السابقة من الملف
    prev_data = last_signals.get(symbol, {})
    in_position = prev_data.get("in_position", False)

    buy_signal = False
    sell_signal = False

    # منطق اتخاذ القرار المستقل للـ AI
    if ai_pred == 1 and not in_position:
        buy_signal = True
        in_position = True
        buy_alerts.append(f"🟢 {symbol} | السعر: {last_close:.2f} | التاريخ: {last_candle_date}")
        log_buy(symbol, last_close, last_candle_date)

    elif ai_pred == 0 and in_position:
        sell_signal = True
        in_position = False
        sell_alerts.append(f"🔴 {symbol} | خروج عند: {last_close:.2f} | التاريخ: {last_candle_date}")
        log_sell(symbol, last_close, last_candle_date)

    # تحديث حالة الإشارات لهذا السهم
    new_signals[symbol] = {
        "in_position": in_position,
        "last_prediction": int(ai_pred),
        "last_price": float(last_close),
        "last_date": last_candle_date
    }

# =====================
# Build & Send Telegram Report
# =====================
report = [f"🤖 AI Stock Model Daily Signals Report ({global_last_date}):\n"]

if buy_alerts:
    report.append("🚀 إشارات شراء جديدة (AI BUY):")
    report.extend(buy_alerts)

if sell_alerts:
    report.append("\n🔻 إشارات خروج/بيع (AI SELL):")
    report.extend(sell_alerts)

if not buy_alerts and not sell_alerts:
    report.append(f"ℹ️ لا توجد إشارات جديدة اليوم. السوق مستقر (آخر تاريخ: {global_last_date})")

if data_failures:
    report.append(f"\n⚠️ تعذر قراءة بيانات بعض الأسهم: {len(data_failures)}")

# حفظ الحالة الحالية في ملف الإشارات لتفادي التكرار غداً
with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
    json.dump(new_signals, f, indent=2, ensure_ascii=False)

# إرسال التقرير النهائي عبر التليجرام
final_message = "\n".join(report)
send_telegram(final_message)
print("🏁 AI Prediction & Alert Dispatch Complete.")

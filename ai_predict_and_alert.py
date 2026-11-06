from datetime import datetime
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
# Files & Model Configuration
# =====================
TARGET_HORIZON = 20
DB_FILE = "egx_history_database_v2.json"
FEATURES_FILE = "ai_features_database.json"
MARKET_FILE = "ai_market_regime_database.json"
MODEL_FILE = f"ai_stock_model_{TARGET_HORIZON}d.pkl"
META_FILE = f"ai_stock_model_{TARGET_HORIZON}d_meta.json"
SIGNALS_FILE = f"ai_last_signals_{TARGET_HORIZON}d.json"
TRADES_FILE = f"ai_trades_{TARGET_HORIZON}d.json"

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
# Load AI Model & Databases
# =====================
if not os.path.exists(MODEL_FILE) or not os.path.exists(META_FILE):
    raise FileNotFoundError(f"❌ نموذج أو ملف مواصفات الأفق {TARGET_HORIZON}d غير موجود! تأكد من تدريب النموذج أولاً.")

model = joblib.load(MODEL_FILE)
print(f"🤖 AI Model loaded successfully from {MODEL_FILE}")

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)
expected_features = metadata.get("feature_columns", [])
print(f"-> تم تحميل مواصفات النموذج بعدد ميزات: {len(expected_features)}")

try:
    with open(DB_FILE, "r", encoding="utf-8") as f:
        raw_database = json.load(f)
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        features_db = json.load(f)
    with open(MARKET_FILE, "r", encoding="utf-8") as f:
        market_db = json.load(f)
    print(f"💾 Databases (History, Features, Market) loaded successfully.")
except Exception as e:
    raise FileNotFoundError(f"❌ تعذر قراءة قواعد البيانات: {e}")

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

stock_features_db = features_db.get("symbols", {})
market_data_db = market_db.get("data", {})

# خرائط ترميز ميزات السوق (نفس التي استخدمناها في التدريب تماماً)
regime_mapping = {"BEARISH": -1, "NEUTRAL": 0, "BULLISH": 1, "INSUFFICIENT_DATA": 0}
trend_mapping = {"DOWN": -1, "MIXED": 0, "UP": 1}
momentum_mapping = {"NEGATIVE": -1, "MIXED": 0, "POSITIVE": 1}

# =====================
# Main Loop
# =====================
for symbol, stock_content in raw_database.items():
    if symbol == "EGX30":
        continue
        
    if symbol not in stock_features_db:
        data_failures.append(symbol)
        continue

    # استخراج أسعار الشموع من قاعدة التاريخ الأساسية لمعرفة السعر الحالي وأعلى سعر
    if not stock_content or "data" not in stock_content or "columns" not in stock_content:
        data_failures.append(symbol)
        continue

    df_price = pd.DataFrame.from_dict(stock_content["data"], orient="index", columns=stock_content["columns"])
    df_price.index = pd.to_datetime(df_price.index)
    df_price = df_price.sort_index(ascending=True)

    if len(df_price) < 10:
        data_failures.append(symbol)
        continue

    last_candle_date = df_price.index[-1].strftime('%Y-%m-%d')
    global_last_date = last_candle_date
    last_close = df_price["Close"].iloc[-1]
    last_high = df_price["High"].iloc[-1]

    # جلب ميزات هذا السهم الجاهزة لنفس التاريخ الأخير من ai_features_database.json لمنع Train-Serving Skew
    symbol_features_data = stock_features_db[symbol].get("data", {})
    if last_candle_date not in symbol_features_data:
        # البحث عن أقرب تاريخ متاح إذا لم يكن التاريخ متطابقاً بحذافيره
        available_dates = sorted(list(symbol_features_data.keys()))
        if not available_dates:
            data_failures.append(symbol)
            continue
        last_candle_date = available_dates[-1]

    latest_features = symbol_features_data[last_candle_date].copy()

    # جلب بيانات السوق لنفس التاريخ وربطها
    market_info = market_data_db.get(last_candle_date, {})
    latest_features["Market_Score"] = market_info.get("score", 0.0)
    latest_features["Market_Volatility"] = market_info.get("volatility", 0.0)
    
    # ترميز ميزات السوق النصية نفس التدريب
    regime_val = market_info.get("regime", "NEUTRAL")
    trend_val = market_info.get("trend", "MIXED")
    momentum_val = market_info.get("momentum", "MIXED")

    latest_features["Market_Regime"] = regime_mapping.get(regime_val, 0)
    latest_features["Market_Trend"] = trend_mapping.get(trend_val, 0)
    latest_features["Market_Momentum"] = momentum_mapping.get(momentum_val, 0)

    # تجهيز صف التنبؤ وترتيب الميزات بدقة حسب ما تدرّب عليه النموذج
    row_df = pd.DataFrame([latest_features])
    
    for feat in expected_features:
        if feat not in row_df.columns:
            row_df[feat] = 0.0

    X_pred = row_df[expected_features].select_dtypes(include=['number', 'bool'])

    # حساب احتمالية الشراء بدقة باستخدام model.classes_
    try:
        probs = model.predict_proba(X_pred)[0]
        classes = list(model.classes_)
        if 1 in classes:
            buy_class_idx = classes.index(1)
            prob_buy = probs[buy_class_idx]
        else:
            prob_buy = probs[1] if len(probs) > 1 else 0.5
    except Exception as e:
        print(f"⚠️ Prediction error for {symbol}: {e}")
        continue

    prev_data = last_signals.get(symbol, {})
    in_position = prev_data.get("in_position", False)
    entry_price = prev_data.get("entry_price", last_close)
    highest_price = prev_data.get("highest_price", last_high)

    # تحديث أعلى سعر وصل إليه السهم أثناء الصفقة (Trailing Stop tracking)
    if in_position:
        if last_high > highest_price:
            highest_price = last_high

    buy_signal = False
    sell_signal = False
    exit_reason = ""

    # منطق الدخول الصارم بعتبة الثقة المحددة
    if prob_buy >= CONFIDENCE_THRESHOLD and not in_position:
        buy_signal = True
        in_position = True
        entry_price = last_close
        highest_price = last_high
        buy_alerts.append(f"🟢 {symbol} | السعر: {last_close:.2f} | الثقة: {prob_buy*100:.1f}%")
        log_buy(symbol, last_close, last_candle_date, prob_buy)

    # منطق الخروج (انعكاس النموذج < 50% أو تفعيل Trailing Stop)
    elif in_position:
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
report = [f"🤖 AI Stock Model Pro Report ({TARGET_HORIZON} Days Horizon - {global_last_date}):\n"]

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
print("🏁 AI Prediction & Alert Dispatch Complete with Pro Rules & Feature Store Alignment.")

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
TARGET_THRESHOLD = 8.0

DB_FILE = "egx_history_database_v2.json"
FEATURES_FILE = "ai_features_database.json"
MARKET_FILE = "ai_market_regime_database.json"
MODEL_FILE = f"ai_stock_model_{TARGET_HORIZON}d.pkl"
META_FILE = f"ai_stock_model_{TARGET_HORIZON}d_meta.json"
SIGNALS_FILE = "ai_last_signals.json"
TRADES_FILE = "ai_trades.json"

# إعدادات عتبة الثقة وقواعد حماية الأرباح
# تظل هذه القيم مؤقتة لحين تقييم نتائج الاختبار
CONFIDENCE_THRESHOLD = 0.70
TRAILING_STOP_PCT = 0.05

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
                if not isinstance(data, list):
                    data = []
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
            if not isinstance(data, list):
                return
        except:
            return

    for trade in reversed(data):
        if (
            isinstance(trade, dict)
            and trade.get("symbol") == symbol
            and "exit_price" not in trade
        ):
            entry_price = trade.get("entry_price", 0)

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
    raise FileNotFoundError(
        f"❌ نموذج أو ملف مواصفات الأفق {TARGET_HORIZON}d غير موجود! "
        "تأكد من تدريب النموذج أولاً."
    )

model = joblib.load(MODEL_FILE)
print(f"🤖 AI Model loaded successfully from {MODEL_FILE}")

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

if not isinstance(metadata, dict):
    raise ValueError("❌ ملف Metadata غير صالح.")

# =====================
# Validate Model Configuration
# =====================
if metadata.get("horizon") != TARGET_HORIZON:
    raise ValueError(
        "❌ أفق النموذج المحفوظ لا يتطابق مع أفق التنبؤ."
    )

try:
    saved_threshold = float(metadata.get("profit_threshold"))
except (TypeError, ValueError):
    raise ValueError("❌ حد الربحية في Metadata غير صالح.")

if saved_threshold != TARGET_THRESHOLD:
    raise ValueError(
        f"❌ حد الربحية غير متطابق! "
        f"النموذج: {saved_threshold}% | "
        f"التوقعات: {TARGET_THRESHOLD}%. "
        "أعد تدريب النموذج."
    )

expected_features = metadata.get("feature_columns", [])

if (
    not isinstance(expected_features, list)
    or not expected_features
    or not all(isinstance(x, str) for x in expected_features)
):
    raise ValueError(
        "❌ قائمة خصائص النموذج غير موجودة أو غير صالحة."
    )

if len(expected_features) != len(set(expected_features)):
    raise ValueError("❌ توجد أسماء خصائص مكررة في Metadata.")

# التأكد من توافق عدد الخصائص مع النموذج
model_feature_count = getattr(model, "n_features_in_", None)

if (
    model_feature_count is not None
    and model_feature_count != len(expected_features)
):
    raise ValueError(
        f"❌ عدد خصائص النموذج غير متطابق! "
        f"النموذج: {model_feature_count} | "
        f"Metadata: {len(expected_features)}"
    )

print(f"-> Horizon: {TARGET_HORIZON}")
print(f"-> Target Return: {TARGET_THRESHOLD}%")
print(f"-> Model Features: {len(expected_features)}")

# التأكد من وجود احتمالية فئة النجاح
if not hasattr(model, "classes_"):
    raise ValueError("❌ النموذج لا يحتوي على classes_.")

model_classes = list(model.classes_)

if 1 not in model_classes:
    raise ValueError(
        "❌ النموذج لا يحتوي على فئة النجاح (1). "
        "يجب إعادة تدريب النموذج."
    )

buy_class_idx = model_classes.index(1)

# =====================
# Load Databases
# =====================
try:
    with open(DB_FILE, "r", encoding="utf-8") as f:
        raw_database = json.load(f)

    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        features_db = json.load(f)

    with open(MARKET_FILE, "r", encoding="utf-8") as f:
        market_db = json.load(f)

    print("💾 Databases (History, Features, Market) loaded successfully.")

except Exception as e:
    raise FileNotFoundError(
        f"❌ تعذر قراءة قواعد البيانات: {e}"
    )

if not all(
    isinstance(x, dict)
    for x in [raw_database, features_db, market_db]
):
    raise ValueError("❌ أحد ملفات قواعد البيانات ليس بصيغة Dictionary.")

# =====================
# Load Previous Signals
# =====================
try:
    with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
        last_signals = json.load(f)

    if not isinstance(last_signals, dict):
        raise ValueError("ملف الإشارات ليس بصيغة Dictionary.")

except Exception as e:
    print(f"⚠️ Previous signals could not be loaded: {e}")
    last_signals = {}

new_signals = last_signals.copy()

buy_alerts = []
sell_alerts = []
data_failures = []
processed_dates = []

stock_features_db = features_db.get("symbols", {})
market_data_db = market_db.get("data", {})

if not isinstance(stock_features_db, dict) or not isinstance(market_data_db, dict):
    raise ValueError(
        "❌ بنية ملفات الخصائص أو السوق غير صحيحة."
    )

# =====================
# Market Feature Mappings
# =====================
regime_mapping = {
    "BEARISH": -1,
    "NEUTRAL": 0,
    "BULLISH": 1,
    "INSUFFICIENT_DATA": 0
}

trend_mapping = {
    "DOWN": -1,
    "MIXED": 0,
    "UP": 1
}

momentum_mapping = {
    "NEGATIVE": -1,
    "MIXED": 0,
    "POSITIVE": 1
}

# =====================
# Main Loop
# =====================
for symbol, stock_content in raw_database.items():

    if symbol == "EGX30":
        continue

    if symbol not in stock_features_db:
        data_failures.append(f"{symbol}: Missing Features")
        continue

    # =====================
    # 1. Validate Price Data
    # =====================
    if (
        not stock_content
        or "data" not in stock_content
        or "columns" not in stock_content
    ):
        data_failures.append(
            f"{symbol}: Invalid Price Structure"
        )
        continue

    try:
        df_price = pd.DataFrame.from_dict(
            stock_content["data"],
            orient="index",
            columns=stock_content["columns"]
        )

        required_price_cols = ["Close", "High"]

        if not all(col in df_price.columns for col in required_price_cols):
            data_failures.append(
                f"{symbol}: Missing Price Columns"
            )
            continue

        df_price.index = pd.to_datetime(
            df_price.index,
            errors="coerce"
        )

        if df_price.index.isna().any():
            data_failures.append(
                f"{symbol}: Invalid Price Dates"
            )
            continue

        df_price = df_price.sort_index(ascending=True)

        if len(df_price) < 10:
            data_failures.append(
                f"{symbol}: Insufficient Price History"
            )
            continue

        last_candle_date = df_price.index[-1].strftime("%Y-%m-%d")

        last_close = pd.to_numeric(
            df_price["Close"].iloc[-1],
            errors="coerce"
        )

        last_high = pd.to_numeric(
            df_price["High"].iloc[-1],
            errors="coerce"
        )

        if (
            pd.isna(last_close)
            or pd.isna(last_high)
            or not np.isfinite(last_close)
            or not np.isfinite(last_high)
            or last_close <= 0
            or last_high <= 0
        ):
            data_failures.append(
                f"{symbol}: Invalid Last Price"
            )
            continue

        last_close = float(last_close)
        last_high = float(last_high)

        if last_high < last_close:
            data_failures.append(
                f"{symbol}: Invalid High/Close"
            )
            continue

    except Exception as e:
        data_failures.append(
            f"{symbol}: Price Error ({e})"
        )
        continue

    # =====================
    # 2. Exact Feature Date Matching
    # =====================
    symbol_features_data = stock_features_db[symbol].get("data", {})

    if not isinstance(symbol_features_data, dict):
        data_failures.append(
            f"{symbol}: Invalid Features Structure"
        )
        continue

    # لا نستخدم أي تاريخ بديل
    if last_candle_date not in symbol_features_data:
        data_failures.append(
            f"{symbol}: Feature Date Mismatch ({last_candle_date})"
        )
        continue

    latest_features = symbol_features_data[last_candle_date]

    if not isinstance(latest_features, dict):
        data_failures.append(
            f"{symbol}: Invalid Feature Row"
        )
        continue

    latest_features = latest_features.copy()

    # =====================
    # 3. Exact Market Date Matching
    # =====================
    if last_candle_date not in market_data_db:
        data_failures.append(
            f"{symbol}: Missing Market Data ({last_candle_date})"
        )
        continue

    market_info = market_data_db[last_candle_date]

    if not isinstance(market_info, dict):
        data_failures.append(
            f"{symbol}: Invalid Market Data"
        )
        continue

    required_market_fields = [
        "score",
        "volatility",
        "regime",
        "trend",
        "momentum"
    ]

    if not all(
        field in market_info and market_info[field] is not None
        for field in required_market_fields
    ):
        data_failures.append(
            f"{symbol}: Incomplete Market Data"
        )
        continue

    if market_info.get("regime") == "INSUFFICIENT_DATA":
        data_failures.append(
            f"{symbol}: Insufficient Market Regime"
        )
        continue

    # =====================
    # Validate Market Numeric Values
    # =====================
    try:
        market_score = float(market_info["score"])
        market_volatility = float(market_info["volatility"])

        if (
            not np.isfinite(market_score)
            or not np.isfinite(market_volatility)
        ):
            raise ValueError("Non-finite market values")

    except Exception:
        data_failures.append(
            f"{symbol}: Invalid Market Numeric Values"
        )
        continue

    regime_val = market_info["regime"]
    trend_val = market_info["trend"]
    momentum_val = market_info["momentum"]

    if (
        regime_val not in regime_mapping
        or trend_val not in trend_mapping
        or momentum_val not in momentum_mapping
    ):
        data_failures.append(
            f"{symbol}: Unknown Market Categories"
        )
        continue

    latest_features["Market_Score"] = market_score
    latest_features["Market_Volatility"] = market_volatility
    latest_features["Market_Regime"] = regime_mapping[regime_val]
    latest_features["Market_Trend"] = trend_mapping[trend_val]
    latest_features["Market_Momentum"] = momentum_mapping[momentum_val]

    # =====================
    # 4. Strict Feature Validation
    # =====================
    missing_features = [
        feat for feat in expected_features
        if feat not in latest_features
    ]

    if missing_features:
        data_failures.append(
            f"{symbol}: Missing Features ({','.join(missing_features)})"
        )
        continue

    try:
        feature_values = {}

        for feat in expected_features:
            value = latest_features[feat]

            if isinstance(value, (bool, np.bool_)):
                value = int(value)

            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f"Invalid feature: {feat}"
                )

            feature_values[feat] = value

        # ترتيب الخصائص مطابق تماماً للتدريب
        X_pred = pd.DataFrame(
            [[feature_values[feat] for feat in expected_features]],
            columns=expected_features
        )

    except Exception as e:
        data_failures.append(
            f"{symbol}: Feature Validation Error ({e})"
        )
        continue

    # =====================
    # 5. AI Prediction
    # =====================
    try:
        probs = model.predict_proba(X_pred)[0]
        prob_buy = float(probs[buy_class_idx])

        if not np.isfinite(prob_buy):
            raise ValueError("Invalid prediction probability")

    except Exception as e:
        data_failures.append(
            f"{symbol}: Prediction Error ({e})"
        )
        print(f"⚠️ Prediction error for {symbol}: {e}")
        continue

    # =====================
    # 6. Previous Trade State
    # =====================
    prev_data = last_signals.get(symbol, {})

    if not isinstance(prev_data, dict):
        data_failures.append(
            f"{symbol}: Invalid Previous Signal Structure"
        )
        continue

    in_position = prev_data.get("in_position", False)
    entry_price = prev_data.get("entry_price", last_close)
    highest_price = prev_data.get("highest_price", last_high)

    if in_position:
        try:
            entry_price = float(entry_price)
            highest_price = float(highest_price)

            if (
                not np.isfinite(entry_price)
                or not np.isfinite(highest_price)
                or entry_price <= 0
                or highest_price <= 0
            ):
                raise ValueError(
                    "Invalid saved trade state"
                )

        except Exception:
            data_failures.append(
                f"{symbol}: Invalid Previous Trade State"
            )
            continue

    # تحديث أعلى سعر وصل إليه السهم أثناء الصفقة
    if in_position:
        if last_high > highest_price:
            highest_price = last_high

    buy_signal = False
    sell_signal = False
    exit_reason = ""

    # =====================
    # 7. Buy Logic
    # =====================
    if prob_buy >= CONFIDENCE_THRESHOLD and not in_position:

        buy_signal = True
        in_position = True
        entry_price = last_close
        highest_price = last_high

        buy_alerts.append(
            f"🟢 {symbol} | السعر: {last_close:.2f} | "
            f"الثقة: {prob_buy*100:.1f}%"
        )

        log_buy(
            symbol,
            last_close,
            last_candle_date,
            prob_buy
        )

    # =====================
    # 8. Sell Logic
    # =====================
    elif in_position:

        drawdown_from_peak = (
            (highest_price - last_close) / highest_price
            if highest_price > 0 else 0
        )

        if prob_buy < 0.50:
            sell_signal = True
            in_position = False
            exit_reason = "AI_Model_Inversion (انعكاس النموذج)"

        elif drawdown_from_peak >= TRAILING_STOP_PCT:
            sell_signal = True
            in_position = False
            exit_reason = (
                f"Trailing_Stop (حماية الأرباح بتراجع "
                f"{drawdown_from_peak*100:.1f}%)"
            )

        if sell_signal:

            profit_calc = (
                ((last_close - entry_price) / entry_price) * 100
            )

            emoji = "🟢" if profit_calc >= 0 else "🔴"

            sell_alerts.append(
                f"🔻 {symbol} | خروج عند: {last_close:.2f} | "
                f"النتيجة: {emoji} {profit_calc:.2f}% | "
                f"السبب: {exit_reason}"
            )

            log_sell(
                symbol,
                last_close,
                last_candle_date,
                exit_reason
            )

    # =====================
    # 9. Save Current State
    # =====================
    new_signals[symbol] = {
        "in_position": in_position,
        "entry_price": float(entry_price) if in_position else 0.0,
        "highest_price": float(highest_price) if in_position else 0.0,
        "last_prediction": float(prob_buy),
        "last_price": float(last_close),
        "last_date": last_candle_date
    }

    processed_dates.append(last_candle_date)

# =====================
# Telegram Report Dispatch
# =====================
global_last_date = (
    max(processed_dates) if processed_dates else "Unknown Date"
)

report = [
    f"🤖 AI Stock Model Pro Report "
    f"({TARGET_HORIZON} Days Horizon - {global_last_date}):\n"
]

if buy_alerts:
    report.append(
        "🚀 إشارات شراء جديدة (AI BUY - High Confidence):"
    )
    report.extend(buy_alerts)

if sell_alerts:
    report.append(
        "\n🔻 إشارات خروج / جني أرباح / Stop Loss:"
    )
    report.extend(sell_alerts)

if not buy_alerts and not sell_alerts:
    report.append(
        f"ℹ️ لا توجد إشارات جديدة اليوم. "
        f"(آخر تاريخ تمت معالجته: {global_last_date})"
    )

if data_failures:
    report.append(
        f"\n⚠️ تعذر معالجة بيانات بعض الأسهم: {len(data_failures)}"
    )
    report.extend(data_failures[:15])

# =====================
# Save Signals
# =====================
with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        new_signals,
        f,
        indent=2,
        ensure_ascii=False
    )

# =====================
# Send Telegram Report
# =====================
final_message = "\n".join(report)
send_telegram(final_message)

print("🏁 AI Prediction & Alert Dispatch Complete.")
print(f"-> Horizon: {TARGET_HORIZON}")
print(f"-> Target Return: {TARGET_THRESHOLD}%")
print(f"-> Processed stocks: {len(processed_dates)}")
print(f"-> Data failures: {len(data_failures)}")
print(f"-> Buy signals: {len(buy_alerts)}")
print(f"-> Sell signals: {len(sell_alerts)}")

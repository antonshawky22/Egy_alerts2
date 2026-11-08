from datetime import datetime, timezone
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
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": text},
            timeout=10
        )

        if response.status_code != 200:
            print(f"❌ Telegram send failed: {response.text}")

    except Exception as e:
        print(f"❌ Telegram send exception: {e}")


# =====================
# Files & Model Configuration
# =====================
TARGET_HORIZON = 20
TARGET_THRESHOLD = 8.0
TARGET_COLUMN = "Target_Reach_8Pct"

TARGET_DEFINITION = {
    "type": "max_gain_anytime",
    "horizon": TARGET_HORIZON,
    "threshold_pct": TARGET_THRESHOLD,
    "target_column": TARGET_COLUMN,
    "condition": "Max_Gain_Pct > 8.0"
}

DB_FILE = "egx_history_database_v2.json"
FEATURES_FILE = "ai_features_database.json"
MARKET_FILE = "ai_market_regime_database.json"

MODEL_FILE = f"ai_stock_model_{TARGET_HORIZON}d.pkl"
META_FILE = f"ai_stock_model_{TARGET_HORIZON}d_meta.json"

SIGNALS_FILE = "ai_last_signals.json"
TRADES_FILE = "ai_trades.json"

# =====================
# Prediction Settings
# =====================
CONFIDENCE_THRESHOLD = 0.70
SELL_CONFIDENCE_THRESHOLD = 0.50
TRAILING_STOP_PCT = 0.05


# =====================
# Safe JSON Functions
# =====================
def load_json_file(file_path, default=None):
    if not os.path.exists(file_path):
        return default

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        raise ValueError(
            f"❌ تعذر قراءة الملف {file_path}: {e}"
        )


def save_json_file(file_path, data):
    temp_file = file_path + ".tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False
            )

        os.replace(temp_file, file_path)

    except Exception:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


# =====================
# Trade Logging Functions
# =====================
def load_trades():
    data = load_json_file(TRADES_FILE, [])

    if not isinstance(data, list):
        raise ValueError(
            "❌ ملف الصفقات ليس بصيغة List. "
            "لن يتم استبداله لحماية السجل."
        )

    return data


def log_buy(symbol, price, date, confidence):
    data = load_trades()

    trade = {
        "symbol": symbol,
        "entry_price": float(price),
        "entry_date": date,
        "entry_confidence": round(
            float(confidence) * 100, 2
        ),
        "signal_type": "AI_BUY"
    }

    data.append(trade)
    save_json_file(TRADES_FILE, data)


def log_sell(symbol, price, date, exit_reason):
    data = load_trades()

    for trade in reversed(data):
        if (
            trade.get("symbol") == symbol
            and "exit_price" not in trade
        ):
            entry_price = trade.get("entry_price")

            if entry_price is None or float(entry_price) <= 0:
                raise ValueError(
                    f"Invalid entry price in trade history: {symbol}"
                )

            entry_price = float(entry_price)

            profit_pct = (
                (float(price) - entry_price)
                / entry_price
            ) * 100

            trade["exit_price"] = float(price)
            trade["exit_date"] = date
            trade["exit_reason"] = exit_reason
            trade["profit_pct"] = round(profit_pct, 2)

            save_json_file(TRADES_FILE, data)
            return True

    print(
        f"⚠️ No open trade found in history for {symbol}"
    )

    return False


# =====================
# Load AI Model & Metadata
# =====================
if not os.path.exists(MODEL_FILE) or not os.path.exists(META_FILE):
    raise FileNotFoundError(
        f"❌ نموذج أو ملف مواصفات الأفق "
        f"{TARGET_HORIZON}d غير موجود! "
        "تأكد من تدريب النموذج أولاً."
    )

model = joblib.load(MODEL_FILE)

print(
    f"🤖 AI Model loaded successfully from {MODEL_FILE}"
)

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# =====================
# Validate Model Metadata
# =====================
if metadata.get("horizon") != TARGET_HORIZON:
    raise ValueError(
        "❌ أفق النموذج المحفوظ لا يتطابق مع أفق التنبؤ."
    )

if metadata.get("profit_threshold") != TARGET_THRESHOLD:
    raise ValueError(
        "❌ حد الربحية لا يتطابق مع إعدادات التنبؤ. "
        "أعد تدريب النموذج."
    )

if metadata.get("target_column") != TARGET_COLUMN:
    raise ValueError(
        "❌ عمود الهدف في Metadata غير متوافق."
    )

saved_definition = metadata.get("target_definition", {})

if saved_definition != TARGET_DEFINITION:
    raise ValueError(
        "❌ تعريف الهدف في النموذج غير متوافق. "
        "يجب إعادة التدريب باستخدام تعريف الهدف الجديد."
    )

expected_features = metadata.get("feature_columns", [])

if not expected_features or not isinstance(expected_features, list):
    raise ValueError(
        "❌ قائمة خصائص النموذج غير موجودة أو غير صالحة."
    )

if len(expected_features) != len(set(expected_features)):
    raise ValueError(
        "❌ توجد أسماء خصائص مكررة في Metadata."
    )

forbidden_features = {
    TARGET_COLUMN,
    "Future_Return_Pct",
    "Max_Gain_Pct",
    "Max_Drawdown_Pct",
    "Sessions_To_Max",
    "Sessions_To_Min",
    "date",
    "ticker",
    "symbol",
    "horizon",
    "target"
}

invalid_features = [
    feat for feat in expected_features
    if feat in forbidden_features
]

if invalid_features:
    raise ValueError(
        f"❌ Metadata تحتوي على ميزات مستقبلية ممنوعة: "
        f"{invalid_features}"
    )

print(
    f"-> تم تحميل مواصفات النموذج بعدد ميزات: "
    f"{len(expected_features)}"
)

# =====================
# Validate Model Classes
# =====================
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
    raw_database = load_json_file(DB_FILE)
    features_db = load_json_file(FEATURES_FILE)
    market_db = load_json_file(MARKET_FILE)

    if not isinstance(raw_database, dict):
        raise ValueError("بنية قاعدة الأسعار غير صحيحة.")

    if not isinstance(features_db, dict):
        raise ValueError("بنية قاعدة الميزات غير صحيحة.")

    if not isinstance(market_db, dict):
        raise ValueError("بنية قاعدة السوق غير صحيحة.")

    print(
        "💾 Databases (History, Features, Market) "
        "loaded successfully."
    )

except Exception as e:
    raise FileNotFoundError(
        f"❌ تعذر قراءة قواعد البيانات: {e}"
    )


# =====================
# Load Previous Signals
# =====================
last_signals = load_json_file(SIGNALS_FILE, {})

if not isinstance(last_signals, dict):
    raise ValueError(
        "❌ ملف الإشارات ليس بصيغة Dictionary. "
        "لن يتم استبداله."
    )

new_signals = last_signals.copy()

buy_alerts = []
sell_alerts = []
data_failures = []
processed_dates = []

already_processed = 0

stock_features_db = features_db.get("symbols", {})
market_data_db = market_db.get("data", {})

if (
    not isinstance(stock_features_db, dict)
    or not isinstance(market_data_db, dict)
):
    raise ValueError(
        "❌ بنية ملفات الخصائص أو السوق غير صحيحة."
    )


# =====================
# Market Encoding
# =====================
regime_mapping = {
    "BEARISH": -1,
    "NEUTRAL": 0,
    "BULLISH": 1
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
        data_failures.append(
            f"{symbol}: Missing Features"
        )
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

        required_price_cols = [
            "Close",
            "High"
        ]

        if not all(
            col in df_price.columns
            for col in required_price_cols
        ):
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

        df_price = df_price.sort_index(
            ascending=True
        )

        if len(df_price) < 10:
            data_failures.append(
                f"{symbol}: Insufficient Price History"
            )
            continue

        last_candle_date = (
            df_price.index[-1].strftime("%Y-%m-%d")
        )

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
    # 2. Load Previous State
    # =====================
    prev_data = last_signals.get(symbol, {})

    if not isinstance(prev_data, dict):
        data_failures.append(
            f"{symbol}: Invalid Previous Signal State"
        )
        continue

    # منع معالجة نفس الشمعة أكثر من مرة
    if prev_data.get("last_date") == last_candle_date:
        already_processed += 1
        processed_dates.append(last_candle_date)
        continue

    # =====================
    # 3. Exact Feature Date Matching
    # =====================
    symbol_features_data = (
        stock_features_db[symbol].get("data", {})
    )

    if not isinstance(symbol_features_data, dict):
        data_failures.append(
            f"{symbol}: Invalid Features Structure"
        )
        continue

    if last_candle_date not in symbol_features_data:
        data_failures.append(
            f"{symbol}: Feature Date Mismatch "
            f"({last_candle_date})"
        )
        continue

    latest_features = (
        symbol_features_data[last_candle_date]
    )

    if not isinstance(latest_features, dict):
        data_failures.append(
            f"{symbol}: Invalid Feature Row"
        )
        continue

    latest_features = latest_features.copy()

    # =====================
    # 4. Exact Market Date Matching
    # =====================
    if last_candle_date not in market_data_db:
        data_failures.append(
            f"{symbol}: Missing Market Data "
            f"({last_candle_date})"
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
        field in market_info
        and market_info[field] is not None
        for field in required_market_fields
    ):
        data_failures.append(
            f"{symbol}: Incomplete Market Data"
        )
        continue

    if (
        market_info.get("regime")
        == "INSUFFICIENT_DATA"
    ):
        data_failures.append(
            f"{symbol}: Insufficient Market Regime"
        )
        continue

    # =====================
    # 5. Validate Market Values
    # =====================
    try:
        market_score = float(
            market_info["score"]
        )

        market_volatility = float(
            market_info["volatility"]
        )

        if (
            not np.isfinite(market_score)
            or not np.isfinite(market_volatility)
        ):
            raise ValueError(
                "Non-finite market values"
            )

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
    latest_features["Market_Regime"] = (
        regime_mapping[regime_val]
    )
    latest_features["Market_Trend"] = (
        trend_mapping[trend_val]
    )
    latest_features["Market_Momentum"] = (
        momentum_mapping[momentum_val]
    )

    # =====================
    # 6. Strict Feature Validation
    # =====================
    missing_features = [
        feat for feat in expected_features
        if feat not in latest_features
    ]

    if missing_features:
        data_failures.append(
            f"{symbol}: Missing Features "
            f"({','.join(missing_features)})"
        )
        continue

    try:
        feature_values = {}

        for feat in expected_features:
            value = latest_features[feat]

            if isinstance(
                value,
                (bool, np.bool_)
            ):
                value = int(value)

            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f"Invalid feature: {feat}"
                )

            feature_values[feat] = value

        X_pred = pd.DataFrame(
            [[
                feature_values[feat]
                for feat in expected_features
            ]],
            columns=expected_features
        )

    except Exception as e:
        data_failures.append(
            f"{symbol}: Feature Validation Error ({e})"
        )
        continue

    # =====================
    # 7. AI Prediction
    # =====================
    try:
        probs = model.predict_proba(X_pred)[0]

        prob_buy = float(
            probs[buy_class_idx]
        )

        if (
            not np.isfinite(prob_buy)
            or prob_buy < 0
            or prob_buy > 1
        ):
            raise ValueError(
                "Invalid prediction probability"
            )

    except Exception as e:
        data_failures.append(
            f"{symbol}: Prediction Error ({e})"
        )
        print(
            f"⚠️ Prediction error for {symbol}: {e}"
        )
        continue

    # =====================
    # 8. Previous Trade State
    # =====================
    in_position = prev_data.get(
        "in_position",
        False
    )

    if not isinstance(in_position, bool):
        data_failures.append(
            f"{symbol}: Invalid Position State"
        )
        continue

    entry_price = prev_data.get(
        "entry_price",
        0.0
    )

    highest_price = prev_data.get(
        "highest_price",
        0.0
    )

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

    buy_signal = False
    sell_signal = False
    exit_reason = ""

    # =====================
    # 9. Buy Logic
    # =====================
    if (
        prob_buy >= CONFIDENCE_THRESHOLD
        and not in_position
    ):
        buy_signal = True
        in_position = True

        entry_price = last_close

        # لا نحتسب أعلى سعر في شمعة الدخول
        # لأنه قد يكون حدث قبل افتراض الدخول عند الإغلاق
        highest_price = last_close

        buy_alerts.append(
            f"🟢 {symbol} | السعر: {last_close:.2f} | "
            f"الثقة: {prob_buy * 100:.1f}%"
        )

        log_buy(
            symbol,
            last_close,
            last_candle_date,
            prob_buy
        )

    # =====================
    # 10. Sell Logic
    # =====================
    elif in_position:

        # تحديث أعلى سعر بعد الدخول
        if last_high > highest_price:
            highest_price = last_high

        drawdown_from_peak = (
            (highest_price - last_close)
            / highest_price
            if highest_price > 0
            else 0
        )

        if prob_buy < SELL_CONFIDENCE_THRESHOLD:
            sell_signal = True
            in_position = False

            exit_reason = (
                "AI_Model_Inversion "
                "(انعكاس النموذج)"
            )

        elif drawdown_from_peak >= TRAILING_STOP_PCT:
            sell_signal = True
            in_position = False

            exit_reason = (
                f"Trailing_Stop "
                f"(حماية الأرباح بتراجع "
                f"{drawdown_from_peak * 100:.1f}%)"
            )

        if sell_signal:

            profit_calc = (
                (last_close - entry_price)
                / entry_price
            ) * 100

            emoji = (
                "🟢"
                if profit_calc >= 0
                else "🔴"
            )

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
    # 11. Save Current State
    # =====================
    new_signals[symbol] = {
        "in_position": in_position,
        "entry_price": (
            float(entry_price)
            if in_position else 0.0
        ),
        "highest_price": (
            float(highest_price)
            if in_position else 0.0
        ),
        "last_prediction": float(prob_buy),
        "last_price": float(last_close),
        "last_date": last_candle_date
    }

    processed_dates.append(
        last_candle_date
    )


# =====================
# Save Signals
# =====================
save_json_file(
    SIGNALS_FILE,
    new_signals
)


# =====================
# Telegram Report
# =====================
global_last_date = (
    max(processed_dates)
    if processed_dates
    else "Unknown Date"
)

report = [
    f"🤖 AI Stock Model Pro Report "
    f"({TARGET_HORIZON} Days Horizon - "
    f"{global_last_date}):\n",

    f"🎯 Target: Max Gain > {TARGET_THRESHOLD}% "
    f"within {TARGET_HORIZON} sessions",

    f"📊 Buy Confidence: "
    f"{CONFIDENCE_THRESHOLD * 100:.0f}%",

    f"🛡️ Trailing Stop: "
    f"{TRAILING_STOP_PCT * 100:.0f}%\n"
]

if buy_alerts:
    report.append(
        "🚀 إشارات شراء جديدة (AI BUY):"
    )
    report.extend(buy_alerts)

if sell_alerts:
    report.append(
        "\n🔻 إشارات خروج / حماية أرباح:"
    )
    report.extend(sell_alerts)

if not buy_alerts and not sell_alerts:
    report.append(
        "ℹ️ لا توجد إشارات جديدة اليوم."
    )

if already_processed:
    report.append(
        f"\n🔁 أسهم سبق معالجتها في نفس اليوم: "
        f"{already_processed}"
    )

if data_failures:
    report.append(
        f"\n⚠️ تعذر معالجة بيانات بعض الأسهم: "
        f"{len(data_failures)}"
    )

    report.extend(
        data_failures[:15]
    )

final_message = "\n".join(report)

send_telegram(final_message)


# =====================
# Final Report
# =====================
print(
    "\n🏁 AI Prediction & Alert Dispatch Complete."
)

print(
    f"-> Processed stocks: {len(processed_dates)}"
)

print(
    f"-> Already processed: {already_processed}"
)

print(
    f"-> Data failures: {len(data_failures)}"
)

print(
    f"-> Buy signals: {len(buy_alerts)}"
)

print(
    f"-> Sell signals: {len(sell_alerts)}"
)

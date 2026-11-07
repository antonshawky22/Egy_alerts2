from datetime import datetime
import json
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False

# =====================
# Model Settings
# =====================
TARGET_HORIZON = 20
TARGET_THRESHOLD = 2.0  # عائد أكبر من 2% خلال الأفق المحدد
TRAIN_RATIO = 0.80


def load_dataset(file_path="ai_training_dataset.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"الملف {file_path} غير موجود. تأكد من تشغيل بناء البيانات أولاً."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "samples" in data:
        df = pd.DataFrame(data["samples"])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError("صيغة ملف الـ JSON غير متوافقة.")

    if df.empty:
        raise ValueError("ملف بيانات التدريب لا يحتوي على عينات.")

    return df


def train_model():
    print("-> جاري تحميل بيانات التدريب...")
    df = load_dataset()

    # 1. التأكد من وجود عمود الأفق الزمني
    if "horizon" not in df.columns:
        raise ValueError(
            "عمود horizon غير موجود. لا يمكن ضمان عدم خلط الآفاق الزمنية."
        )

    df = df[df["horizon"] == TARGET_HORIZON].copy()
    print(
        f"-> الأفق المستهدف: {TARGET_HORIZON} جلسة. عدد العينات: {len(df)}"
    )

    if df.empty:
        raise ValueError(
            f"لا توجد عينات متاحة للأفق الزمني {TARGET_HORIZON}."
        )

    # 2. التحقق من التاريخ وترتيب العينات
    if "date" not in df.columns:
        raise ValueError(
            "عمود date غير موجود. التقسيم الزمني يتطلب وجود التواريخ."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError(
            "توجد تواريخ غير صالحة في بيانات التدريب. يجب إصلاحها قبل التدريب."
        )

    df = df.sort_values("date").reset_index(drop=True)

    # 3. إنشاء الهدف بناءً على عائد أكبر من 2%
    if "Future_Return_Pct" not in df.columns:
        raise ValueError(
            "عمود Future_Return_Pct غير موجود في البيانات."
        )

    df["Future_Return_Pct"] = pd.to_numeric(
        df["Future_Return_Pct"], errors="coerce"
    )

    df = df.dropna(subset=["Future_Return_Pct"]).copy()

    if df.empty:
        raise ValueError(
            "لا توجد قيم صالحة في Future_Return_Pct."
        )

    df["target"] = (
        df["Future_Return_Pct"] > TARGET_THRESHOLD
    ).astype(int)

    target_col = "target"

    print(
        f"-> حد النجاح: عائد أكبر من {TARGET_THRESHOLD}% "
        f"خلال {TARGET_HORIZON} جلسة."
    )

    # 4. معالجة الميزات النصية للسوق
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

    if "Market_Regime" in df.columns:
        df["Market_Regime"] = df["Market_Regime"].map(
            regime_mapping
        )

    if "Market_Trend" in df.columns:
        df["Market_Trend"] = df["Market_Trend"].map(
            trend_mapping
        )

    if "Market_Momentum" in df.columns:
        df["Market_Momentum"] = df["Market_Momentum"].map(
            momentum_mapping
        )

    exclude_cols = [
        target_col,
        "Future_Return_Pct",
        "Max_Gain_Pct",
        "Max_Drawdown_Pct",
        "Sessions_To_Max",
        "Sessions_To_Min",
        "date",
        "ticker",
        "symbol",
        "horizon"
    ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
    ]

    X = df[feature_cols].select_dtypes(
        include=["number", "bool"]
    ).copy()

    y = df[target_col].copy()

    if X.empty:
        raise ValueError(
            "لم يتم العثور على ميزات رقمية صالحة للتدريب."
        )

    # التأكد من صلاحية الميزات الرقمية
    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    if X.isna().any().any():
        raise ValueError(
            "توجد قيم مفقودة أو غير صالحة في ميزات التدريب. "
            "يجب إصلاحها قبل التدريب."
        )

    print(f"-> عدد الميزات النهائية: {X.shape[1]}")
    print(f"-> قائمة الميزات: {list(X.columns)}")

    # 5. التقسيم الزمني حسب التواريخ الفريدة
    unique_dates = sorted(df["date"].unique())
    total_dates = len(unique_dates)

    if total_dates <= TARGET_HORIZON + 1:
        raise ValueError(
            f"عدد التواريخ ({total_dates}) غير كافٍ لتقسيم البيانات "
            f"مع استبعاد {TARGET_HORIZON} جلسة."
        )

    split_index = int(total_dates * TRAIN_RATIO)

    # استبعاد آخر 20 تاريخ تداول قبل بداية الاختبار
    train_end_index = split_index - TARGET_HORIZON

    if train_end_index <= 0 or split_index >= total_dates:
        raise ValueError(
            "عدد التواريخ غير كافٍ لإنشاء مجموعتي تدريب واختبار "
            "مع فترة الاستبعاد."
        )

    train_dates = unique_dates[:train_end_index]
    purged_dates = unique_dates[train_end_index:split_index]
    test_dates = unique_dates[split_index:]

    train_mask = df["date"].isin(train_dates)
    test_mask = df["date"].isin(test_dates)

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print("\n========== التقسيم الزمني ==========")
    print(f"-> إجمالي التواريخ: {total_dates}")
    print(f"-> تواريخ التدريب: {len(train_dates)}")
    print(f"-> تواريخ الاستبعاد (Purging): {len(purged_dates)}")
    print(f"-> تواريخ الاختبار: {len(test_dates)}")
    print(f"-> التدريب: {X_train.shape[0]} عينة")
    print(f"-> الاختبار: {X_test.shape[0]} عينة")
    print(
        f"-> بداية الاختبار: {pd.Timestamp(test_dates[0]).date()}"
    )
    print("====================================\n")

    if X_train.empty or X_test.empty:
        raise ValueError(
            "بيانات التدريب أو الاختبار فارغة بعد التقسيم الزمني."
        )

    # التأكد من وجود فئتي النجاح والخسارة في التدريب
    if y_train.nunique() < 2:
        raise ValueError(
            "بيانات التدريب تحتوي على فئة واحدة فقط. "
            "لا يمكن تدريب نموذج تصنيف موثوق."
        )

    print(
        f"-> توزيع الفئات في التدريب: {dict(y_train.value_counts())}"
    )

    print(
        f"-> توزيع الفئات في الاختبار: {dict(y_test.value_counts())}"
    )

    # 6. إنشاء النموذج
    if HAS_XGBOOST:
        print("-> استخدام نموذج XGBoost...")

        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

    else:
        print("-> استخدام نموذج RandomForest...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

    model.fit(X_train, y_train)

    # استخراج احتمالات النجاح الفعلية لكل عينة اختبار
    buy_class_idx = list(model.classes_).index(1)

    y_prob = model.predict_proba(X_test)[:, buy_class_idx]

    results = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Probability": y_prob
    })

    results.to_csv(
        "ai_test_results.csv",
        index=False
    )

    print(
        "-> تم حفظ احتمالات الاختبار: ai_test_results.csv"
    )

    # 7. تقييم النموذج
    y_pred = model.predict(X_test)

    print("\n========== نتائج تقييم النموذج ==========")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )
    )

    print("=========================================\n")

    # 8. حفظ النموذج وملف المواصفات
    model_filename = (
        f"ai_stock_model_{TARGET_HORIZON}d.pkl"
    )

    meta_filename = (
        f"ai_stock_model_{TARGET_HORIZON}d_meta.json"
    )

    joblib.dump(model, model_filename)

    metadata = {
        "model_file": model_filename,
        "horizon": TARGET_HORIZON,
        "profit_threshold": TARGET_THRESHOLD,
        "feature_columns": list(X.columns),
        "classes": [int(c) for c in model.classes_],
        "trained_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "model_type": (
            "XGBoost" if HAS_XGBOOST else "RandomForest"
        ),
        "train_ratio": TRAIN_RATIO,
        "purged_sessions": TARGET_HORIZON,
        "train_start_date": str(
            pd.Timestamp(train_dates[0]).date()
        ),
        "train_end_date": str(
            pd.Timestamp(train_dates[-1]).date()
        ),
        "test_start_date": str(
            pd.Timestamp(test_dates[0]).date()
        ),
        "test_end_date": str(
            pd.Timestamp(test_dates[-1]).date()
        ),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0])
    }

    with open(
        meta_filename,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"-> تم حفظ النموذج بنجاح: {model_filename}"
    )

    print(
        f"-> تم حفظ Metadata: {meta_filename}"
    )

    print(
        "-> اكتملت عملية التدريب والتقييم بنجاح."
    )


if __name__ == "__main__":
    train_model()

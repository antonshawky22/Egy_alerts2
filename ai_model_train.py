from datetime import datetime, timezone
import json
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

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
TARGET_THRESHOLD = 8.0
TARGET_COLUMN = "Target_Reach_8Pct"
TRAIN_RATIO = 0.80

CONFIDENCE_LEVELS = (0.50, 0.60, 0.70, 0.80)

# =====================
# Target Definition
# =====================
EXPECTED_TARGET_DEFINITION = {
    "type": "max_gain_anytime",
    "horizon": TARGET_HORIZON,
    "threshold_pct": TARGET_THRESHOLD,
    "target_column": TARGET_COLUMN,
    "condition": "Max_Gain_Pct > 8.0"
}

# =====================
# Load Dataset
# =====================
def load_dataset(file_path="ai_training_dataset.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"الملف {file_path} غير موجود. "
            "تأكد من تشغيل بناء البيانات أولاً."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "samples" not in data:
        raise ValueError(
            "صيغة Dataset غير متوافقة. "
            "يجب استخدام ملف Dataset الجديد."
        )

    definition = data.get("target_definition", {})

    for key, expected in EXPECTED_TARGET_DEFINITION.items():
        if definition.get(key) != expected:
            raise ValueError(
                f"تعريف الهدف غير متوافق في Dataset: {key}. "
                f"Expected={expected}, Found={definition.get(key)}"
            )

    feature_columns = (
        data.get("feature_columns", []) +
        data.get("market_feature_columns", [])
    )

    if not feature_columns:
        raise ValueError("قائمة Feature Columns غير موجودة.")

    df = pd.DataFrame(data["samples"])

    if df.empty:
        raise ValueError("ملف بيانات التدريب لا يحتوي على عينات.")

    return df, feature_columns


# =====================
# Market Encoding
# =====================
def encode_market_features(df):
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

    mappings = {
        "Market_Regime": regime_mapping,
        "Market_Trend": trend_mapping,
        "Market_Momentum": momentum_mapping
    }

    for column, mapping in mappings.items():
        if column not in df.columns:
            raise ValueError(
                f"عمود السوق {column} غير موجود."
            )

        original = df[column].copy()
        df[column] = original.map(mapping)

        if df[column].isna().any():
            raise ValueError(
                f"توجد قيم سوق غير معروفة في {column}."
            )

    return df


# =====================
# Model Evaluation
# =====================
def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)

    class_list = list(model.classes_)

    if 1 not in class_list:
        raise ValueError(
            "النموذج لم يتعلم فئة النجاح 1."
        )

    positive_index = class_list.index(1)
    y_probability = probabilities[:, positive_index]

    # Standard evaluation at 50%
    y_pred = (y_probability >= 0.50).astype(int)

    print("\n========== نتائج تقييم النموذج ==========")

    print("\n--- Classification Report (50%) ---")
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

    # Probability metrics
    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(y_test, y_probability)
        pr_auc = average_precision_score(y_test, y_probability)
        brier = brier_score_loss(y_test, y_probability)

        print(f"\nROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
    else:
        roc_auc = None
        pr_auc = None
        brier = None

        print(
            "\n⚠️ لا يمكن حساب AUC "
            "لأن الاختبار يحتوي على فئة واحدة فقط."
        )

    # Evaluation at different confidence thresholds
    threshold_results = {}

    print("\n--- Confidence Threshold Evaluation ---")

    for threshold in CONFIDENCE_LEVELS:
        predictions = (
            y_probability >= threshold
        ).astype(int)

        selected = int(predictions.sum())

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        threshold_results[str(threshold)] = {
            "selected_samples": selected,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4)
        }

        print(
            f"Threshold {threshold:.0%} | "
            f"Signals: {selected} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

    print("=========================================\n")

    return {
        "roc_auc": (
            round(float(roc_auc), 4)
            if roc_auc is not None else None
        ),
        "pr_auc": (
            round(float(pr_auc), 4)
            if pr_auc is not None else None
        ),
        "brier_score": (
            round(float(brier), 4)
            if brier is not None else None
        ),
        "threshold_results": threshold_results
    }


# =====================
# Training
# =====================
def train_model():
    print("-> جاري تحميل بيانات التدريب...")

    df, feature_columns = load_dataset()

    # 1. Validate horizon
    if "horizon" not in df.columns:
        raise ValueError(
            "عمود horizon غير موجود."
        )

    df = df[
        df["horizon"] == TARGET_HORIZON
    ].copy()

    print(
        f"-> الأفق المستهدف: {TARGET_HORIZON} جلسة. "
        f"عدد العينات: {len(df)}"
    )

    if df.empty:
        raise ValueError(
            "لا توجد عينات متاحة للأفق الزمني المطلوب."
        )

    # 2. Validate dates
    if "date" not in df.columns:
        raise ValueError(
            "عمود date غير موجود."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            "توجد تواريخ غير صالحة في Dataset."
        )

    # 3. Validate target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"عمود الهدف {TARGET_COLUMN} غير موجود."
        )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    df = df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    if df.empty:
        raise ValueError(
            "لا توجد أهداف صالحة للتدريب."
        )

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            "قيم الهدف يجب أن تكون 0 أو 1 فقط."
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # 4. Verify target against raw future data
    if "Max_Gain_Pct" not in df.columns:
        raise ValueError(
            "Max_Gain_Pct غير موجود للتحقق من صحة الهدف."
        )

    df["Max_Gain_Pct"] = pd.to_numeric(
        df["Max_Gain_Pct"],
        errors="coerce"
    )

    if df["Max_Gain_Pct"].isna().any():
        raise ValueError(
            "توجد قيم Max_Gain_Pct غير صالحة."
        )

    expected_target = (
        df["Max_Gain_Pct"] > TARGET_THRESHOLD
    ).astype(int)

    if not expected_target.equals(df[TARGET_COLUMN]):
        raise ValueError(
            "عدم تطابق Target_Reach_8Pct مع Max_Gain_Pct. "
            "يجب مراجعة بناء Dataset."
        )

    # 5. Sort chronologically
    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    print(
        f"-> تعريف الهدف: Max_Gain_Pct > {TARGET_THRESHOLD}%"
    )

    print(
        f"-> نسبة النجاح في البيانات: "
        f"{df[TARGET_COLUMN].mean() * 100:.2f}%"
    )

    # 6. Encode market categories
    df = encode_market_features(df)

    # 7. Build features without future information
    forbidden_columns = {
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

    feature_columns = [
        col for col in feature_columns
        if col not in forbidden_columns
    ]

    missing_features = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features مفقودة: {missing_features}"
        )

    X = df[feature_columns].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X.isna().any().any():
        invalid_columns = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            f"توجد Features غير صالحة: {invalid_columns}"
        )

    y = df[TARGET_COLUMN].copy()

    if X.empty:
        raise ValueError(
            "لم يتم العثور على ميزات صالحة."
        )

    print(
        f"-> عدد الميزات النهائية: {X.shape[1]}"
    )

    print(
        f"-> قائمة الميزات: {list(X.columns)}"
    )

    # 8. Chronological split with purging
    unique_dates = sorted(
        df["date"].unique()
    )

    total_dates = len(unique_dates)

    if total_dates <= TARGET_HORIZON + 1:
        raise ValueError(
            f"عدد التواريخ ({total_dates}) غير كافٍ "
            "للتقسيم الزمني."
        )

    split_index = int(
        total_dates * TRAIN_RATIO
    )

    train_end_index = (
        split_index - TARGET_HORIZON
    )

    if (
        train_end_index <= 0
        or split_index >= total_dates
    ):
        raise ValueError(
            "عدد التواريخ غير كافٍ لإنشاء "
            "مجموعتي التدريب والاختبار."
        )

    train_dates = unique_dates[
        :train_end_index
    ]

    purged_dates = unique_dates[
        train_end_index:split_index
    ]

    test_dates = unique_dates[
        split_index:
    ]

    train_mask = df["date"].isin(
        train_dates
    )

    test_mask = df["date"].isin(
        test_dates
    )

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print("\n========== التقسيم الزمني ==========")

    print(f"إجمالي التواريخ: {total_dates}")
    print(f"تواريخ التدريب: {len(train_dates)}")
    print(f"تواريخ الاستبعاد: {len(purged_dates)}")
    print(f"تواريخ الاختبار: {len(test_dates)}")

    print(f"عينات التدريب: {len(X_train)}")
    print(f"عينات الاختبار: {len(X_test)}")

    print(
        f"بداية التدريب: "
        f"{pd.Timestamp(train_dates[0]).date()}"
    )

    print(
        f"نهاية التدريب: "
        f"{pd.Timestamp(train_dates[-1]).date()}"
    )

    print(
        f"بداية الاختبار: "
        f"{pd.Timestamp(test_dates[0]).date()}"
    )

    print(
        f"نهاية الاختبار: "
        f"{pd.Timestamp(test_dates[-1]).date()}"
    )

    print("====================================\n")

    if X_train.empty or X_test.empty:
        raise ValueError(
            "بيانات التدريب أو الاختبار فارغة."
        )

    # 9. Validate classes
    if y_train.nunique() < 2:
        raise ValueError(
            "بيانات التدريب تحتوي على فئة واحدة فقط."
        )

    if y_test.nunique() < 2:
        print(
            "⚠️ تحذير: الاختبار يحتوي على فئة واحدة فقط."
        )

    print(
        f"توزيع التدريب: "
        f"{dict(y_train.value_counts().sort_index())}"
    )

    print(
        f"توزيع الاختبار: "
        f"{dict(y_test.value_counts().sort_index())}"
    )

    # 10. Create model
    if HAS_XGBOOST:
        print("-> استخدام نموذج XGBoost...")

        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric="logloss"
        )

    else:
        print("-> استخدام نموذج RandomForest...")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    # 11. Train
    print("\n-> بدء تدريب النموذج...")

    model.fit(
        X_train,
        y_train
    )

    # 12. Evaluate
    evaluation = evaluate_model(
        model,
        X_test,
        y_test
    )

    # 13. Save model and metadata
    model_filename = (
        f"ai_stock_model_{TARGET_HORIZON}d.pkl"
    )

    meta_filename = (
        f"ai_stock_model_{TARGET_HORIZON}d_meta.json"
    )

    joblib.dump(
        model,
        model_filename
    )

    metadata = {
        "model_file": model_filename,
        "horizon": TARGET_HORIZON,
        "profit_threshold": TARGET_THRESHOLD,
        "target_column": TARGET_COLUMN,
        "target_definition": EXPECTED_TARGET_DEFINITION,
        "feature_columns": list(X.columns),
        "classes": [
            int(c) for c in model.classes_
        ],
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
        "model_type": (
            "XGBoost"
            if HAS_XGBOOST
            else "RandomForest"
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
        "test_samples": int(X_test.shape[0]),
        "train_positive_rate": round(
            float(y_train.mean()), 4
        ),
        "test_positive_rate": round(
            float(y_test.mean()), 4
        ),
        "evaluation": evaluation
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

    print(f"-> تم حفظ النموذج: {model_filename}")
    print(f"-> تم حفظ Metadata: {meta_filename}")
    print("-> اكتملت عملية التدريب والتقييم بنجاح.")


if __name__ == "__main__":
    train_model()

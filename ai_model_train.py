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

TEST_RESULTS_FILE = "ai_test_results.csv"
FEATURE_IMPORTANCE_FILE = "ai_feature_importance_20d.csv"


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
# Positive Class Probability
# =====================

def get_positive_probabilities(model, X):
    probabilities = model.predict_proba(X)
    class_list = list(model.classes_)

    if 1 not in class_list:
        raise ValueError(
            "النموذج لم يتعلم فئة النجاح 1."
        )

    positive_index = class_list.index(1)
    y_probability = probabilities[:, positive_index]

    if not np.isfinite(y_probability).all():
        raise ValueError(
            "توجد احتمالات غير صالحة في نتائج النموذج."
        )

    if (
        (y_probability < 0).any()
        or (y_probability > 1).any()
    ):
        raise ValueError(
            "احتمالات النموذج خارج النطاق [0, 1]."
        )

    return y_probability


# =====================
# Model Evaluation
# =====================

def evaluate_model(model, X_test, y_test):
    y_probability = get_positive_probabilities(
        model,
        X_test
    )

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

    positive_rate = float(y_test.mean())

    baseline_brier = (
        positive_rate * (1 - positive_rate)
    )

    print("\n--- Baseline Comparison ---")
    print(f"Test positive rate: {positive_rate:.4f}")
    print(f"Baseline PR-AUC: {positive_rate:.4f}")
    print(f"Baseline Brier: {baseline_brier:.4f}")

    if brier is not None:
        if brier < baseline_brier:
            print("✅ النموذج أفضل من Baseline في Brier Score.")
        else:
            print("⚠️ النموذج لم يتفوق على Baseline في Brier Score.")

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
        "test_positive_rate": round(positive_rate, 4),
        "baseline_pr_auc": round(positive_rate, 4),
        "baseline_brier_score": round(baseline_brier, 4),
        "brier_better_than_baseline": (
            bool(brier < baseline_brier)
            if brier is not None else None
        ),
        "threshold_results": threshold_results
    }


# =====================
# Save Test Results CSV
# =====================

def save_test_results(
    df,
    test_mask,
    y_test,
    y_probability
):
    test_info = df.loc[test_mask].copy()

    if len(test_info) != len(y_test):
        raise ValueError(
            "عدم تطابق عدد عينات الاختبار مع بيانات CSV."
        )

    symbol_column = None

    for column in ("symbol", "ticker"):
        if column in test_info.columns:
            symbol_column = column
            break

    if symbol_column is None:
        symbols = ["UNKNOWN"] * len(test_info)
    else:
        symbols = test_info[symbol_column].astype(str).tolist()

    results = pd.DataFrame({
        "Date": test_info["date"].dt.strftime("%Y-%m-%d").values,
        "Symbol": symbols,
        "Market_Regime": test_info["Market_Regime_Original"].astype(str).values,
        "Market_Trend": test_info["Market_Trend_Original"].astype(str).values,
        "Market_Momentum": test_info["Market_Momentum_Original"].astype(str).values,
        "Actual_Target": y_test.to_numpy(),
        "Predicted_Probability": y_probability,
        "Predicted_50": (y_probability >= 0.50).astype(int),
        "Predicted_60": (y_probability >= 0.60).astype(int),
        "Predicted_70": (y_probability >= 0.70).astype(int),
        "Predicted_80": (y_probability >= 0.80).astype(int)
    })

    results["Correct_50"] = (
        results["Actual_Target"] == results["Predicted_50"]
    ).astype(int)

    results["Correct_70"] = (
        results["Actual_Target"] == results["Predicted_70"]
    ).astype(int)

    results.to_csv(
        TEST_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"✅ تم حفظ نتائج الاختبار: {TEST_RESULTS_FILE}"
    )

    print(
        f"عدد سجلات الاختبار المحفوظة: {len(results)}"
    )

    return results


# =====================
# Segment Analysis
# =====================

def analyze_segment(results, group_column, title):
    print(f"\n========== {title} ==========")

    if group_column not in results.columns:
        print("⚠️ العمود المطلوب غير موجود.")
        return []

    analysis = []

    grouped = results.groupby(
        group_column,
        dropna=False
    )

    for name, group in grouped:
        count = len(group)

        if count == 0:
            continue

        actual = group["Actual_Target"]
        predicted = group["Predicted_50"]

        precision = precision_score(
            actual,
            predicted,
            zero_division=0
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0
        )

        success_rate = float(actual.mean())

        avg_probability = float(
            group["Predicted_Probability"].mean()
        )

        item = {
            "group": str(name),
            "samples": int(count),
            "actual_success_rate": round(success_rate, 4),
            "average_probability": round(avg_probability, 4),
            "precision_50": round(float(precision), 4),
            "recall_50": round(float(recall), 4)
        }

        analysis.append(item)

        print(
            f"{name} | "
            f"Samples: {count} | "
            f"Actual Success: {success_rate:.2%} | "
            f"Avg Probability: {avg_probability:.2%} | "
            f"Precision: {precision:.2%} | "
            f"Recall: {recall:.2%}"
        )

    if not analysis:
        print("لا توجد مجموعات صالحة للتحليل.")

    return analysis


# =====================
# Feature Importance
# =====================

def analyze_feature_importance(model, feature_columns):
    if not hasattr(model, "feature_importances_"):
        print(
            "\n⚠️ النموذج لا يوفر Feature Importance."
        )
        return []

    importances = np.asarray(
        model.feature_importances_,
        dtype=float
    )

    if len(importances) != len(feature_columns):
        print(
            "\n⚠️ عدد قيم Feature Importance غير متوافق."
        )
        return []

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== Feature Importance ==========")

    for _, row in importance_df.iterrows():
        print(
            f"{row['Feature']}: "
            f"{row['Importance']:.6f}"
        )

    print(
        f"\n✅ تم حفظ أهمية المؤشرات: "
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    return [
        {
            "feature": str(row["Feature"]),
            "importance": round(float(row["Importance"]), 8)
        }
        for _, row in importance_df.iterrows()
    ]


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

    # Preserve original market categories for analysis
    for column in (
        "Market_Regime",
        "Market_Trend",
        "Market_Momentum"
    ):
        df[column + "_Original"] = df[column].copy()

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

    y_probability = get_positive_probabilities(
        model,
        X_test
    )

    # 13. Save detailed test results
    test_results = save_test_results(
        df,
        test_mask,
        y_test,
        y_probability
    )

    # 14. Analyze performance by market regime
    regime_analysis = analyze_segment(
        test_results,
        "Market_Regime",
        "تحليل الأداء حسب حالة السوق EGX30"
    )

    # 15. Analyze performance by stock
    stock_analysis = analyze_segment(
        test_results,
        "Symbol",
        "تحليل الأداء حسب كل سهم"
    )

    # 16. Feature importance
    feature_importance = analyze_feature_importance(
        model,
        list(X.columns)
    )

    # 17. Save model and metadata
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
        "evaluation": evaluation,
        "diagnostics": {
            "test_results_file": TEST_RESULTS_FILE,
            "feature_importance_file": FEATURE_IMPORTANCE_FILE,
            "market_regime_analysis": regime_analysis,
            "stock_analysis": stock_analysis,
            "feature_importance": feature_importance
        }
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

    print(f"\n-> تم حفظ النموذج: {model_filename}")
    print(f"-> تم حفظ Metadata: {meta_filename}")
    print(f"-> تم حفظ نتائج الاختبار: {TEST_RESULTS_FILE}")
    print(f"-> تم حفظ أهمية المؤشرات: {FEATURE_IMPORTANCE_FILE}")
    print("-> اكتملت عملية التدريب والتقييم والتحليل بنجاح.")


if __name__ == "__main__":
    train_model()

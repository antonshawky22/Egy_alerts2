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

FEATURES_FILE = "ai_features_database.json"

TEST_RESULTS_FILE = "ai_test_results.csv"
FEATURE_IMPORTANCE_FILE = "ai_feature_importance_20d.csv"
WALK_FORWARD_RESULTS_FILE = "ai_walk_forward_results.csv"

WALK_FORWARD_FOLDS = 3
MIN_TRAIN_SESSIONS = 100


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
# Load Actual EGX30 Trading Calendar
# =====================

def load_egx30_calendar(file_path=FEATURES_FILE):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"ملف المؤشرات {file_path} غير موجود. "
            "لا يمكن حساب Purging دون تقويم جلسات EGX30."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "صيغة ai_features_database.json غير صحيحة."
        )

    symbols = data.get("symbols")

    if not isinstance(symbols, dict):
        raise ValueError(
            "قسم symbols غير موجود في ملف المؤشرات."
        )

    egx30 = symbols.get("EGX30")

    if not isinstance(egx30, dict):
        raise ValueError(
            "بيانات EGX30 غير موجودة في ملف المؤشرات."
        )

    egx30_data = egx30.get("data")

    if not isinstance(egx30_data, dict) or not egx30_data:
        raise ValueError(
            "تقويم جلسات EGX30 فارغ أو غير صالح."
        )

    calendar = pd.to_datetime(
        list(egx30_data.keys()),
        errors="coerce"
    )

    calendar = pd.DatetimeIndex(
        calendar
    ).dropna().normalize().unique().sort_values()

    if len(calendar) <= TARGET_HORIZON + 1:
        raise ValueError(
            "عدد جلسات EGX30 غير كافٍ لحساب Purging."
        )

    print(
        f"-> تم تحميل تقويم EGX30: {len(calendar)} جلسة فعلية."
    )

    print(
        f"-> بداية التقويم: {calendar[0].date()}"
    )

    print(
        f"-> نهاية التقويم: {calendar[-1].date()}"
    )

    return calendar


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
# Model Factory
# =====================

def create_model():

    if HAS_XGBOOST:

        return xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric="logloss"
        )

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )


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

def evaluate_model(
    model,
    X_test,
    y_test,
    baseline_probability
):

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

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        pr_auc = average_precision_score(
            y_test,
            y_probability
        )

        brier = brier_score_loss(
            y_test,
            y_probability
        )

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

    baseline_probability = float(
        baseline_probability
    )

    if (
        not np.isfinite(baseline_probability)
        or not 0 <= baseline_probability <= 1
    ):
        raise ValueError(
            "قيمة Baseline Probability غير صالحة."
        )

    baseline_brier = brier_score_loss(
        y_test,
        np.full(
            len(y_test),
            baseline_probability,
            dtype=float
        )
    )

    print("\n--- Baseline Comparison ---")

    print(
        f"Train positive rate: {baseline_probability:.4f}"
    )

    print(
        f"Test positive rate: {positive_rate:.4f}"
    )

    print(
        f"Baseline PR-AUC: {positive_rate:.4f}"
    )

    print(
        f"Baseline Brier: {baseline_brier:.4f}"
    )

    if brier is not None:

        if brier < baseline_brier:
            print(
                "✅ النموذج أفضل من Baseline في Brier Score."
            )
        else:
            print(
                "⚠️ النموذج لم يتفوق على Baseline في Brier Score."
            )

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
        "baseline_probability": round(
            baseline_probability, 4
        ),
        "baseline_pr_auc": round(positive_rate, 4),
        "baseline_brier_score": round(
            float(baseline_brier), 4
        ),
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
        symbols = (
            test_info[symbol_column]
            .astype(str)
            .tolist()
        )

    results = pd.DataFrame({
        "Date": test_info["date"].dt.strftime(
            "%Y-%m-%d"
        ).values,
        "Symbol": symbols,
        "Market_Regime": test_info[
            "Market_Regime_Original"
        ].astype(str).values,
        "Market_Trend": test_info[
            "Market_Trend_Original"
        ].astype(str).values,
        "Market_Momentum": test_info[
            "Market_Momentum_Original"
        ].astype(str).values,
        "Actual_Target": y_test.to_numpy(),
        "Predicted_Probability": y_probability,
        "Predicted_50": (
            y_probability >= 0.50
        ).astype(int),
        "Predicted_60": (
            y_probability >= 0.60
        ).astype(int),
        "Predicted_70": (
            y_probability >= 0.70
        ).astype(int),
        "Predicted_80": (
            y_probability >= 0.80
        ).astype(int)
    })

    results["Correct_50"] = (
        results["Actual_Target"] ==
        results["Predicted_50"]
    ).astype(int)

    results["Correct_70"] = (
        results["Actual_Target"] ==
        results["Predicted_70"]
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

def analyze_segment(
    results,
    group_column,
    title
):

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

        success_rate = float(
            actual.mean()
        )

        avg_probability = float(
            group["Predicted_Probability"].mean()
        )

        item = {
            "group": str(name),
            "samples": int(count),
            "actual_success_rate": round(
                success_rate, 4
            ),
            "average_probability": round(
                avg_probability, 4
            ),
            "precision_50": round(
                float(precision), 4
            ),
            "recall_50": round(
                float(recall), 4
            )
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

def analyze_feature_importance(
    model,
    feature_columns
):

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
            "importance": round(
                float(row["Importance"]), 8
            )
        }
        for _, row in importance_df.iterrows()
    ]


# =====================
# Market Regime Diagnostics
# =====================

def analyze_market_distribution(
    df,
    train_mask,
    purged_mask,
    test_mask
):

    print("\n")
    print("=" * 65)
    print("========== Market Regime Distribution ==========")
    print("=" * 65)

    periods = [
        ("TRAIN", train_mask),
        ("PURGED", purged_mask),
        ("TEST", test_mask)
    ]

    regimes = [
        "BULLISH",
        "NEUTRAL",
        "BEARISH"
    ]

    diagnostic_results = {}

    for period_name, mask in periods:

        subset = df.loc[mask].copy()

        print(f"\n---------- {period_name} ----------")

        print(f"Total Samples: {len(subset)}")

        period_data = {}

        if subset.empty:

            print("⚠️ لا توجد عينات في هذه الفترة.")

            diagnostic_results[period_name] = period_data

            continue

        grouped = subset.groupby(
            "Market_Regime_Original",
            dropna=False
        )

        for regime in regimes:

            if regime in grouped.groups:

                group = grouped.get_group(regime)

                count = len(group)

                success_rate = float(
                    group[TARGET_COLUMN].mean()
                )

                positive_count = int(
                    group[TARGET_COLUMN].sum()
                )

                negative_count = count - positive_count

            else:

                count = 0
                success_rate = None
                positive_count = 0
                negative_count = 0

            period_data[regime] = {
                "samples": int(count),
                "positive": int(positive_count),
                "negative": int(negative_count),
                "success_rate": (
                    round(success_rate, 4)
                    if success_rate is not None else None
                )
            }

            if count > 0:

                print(
                    f"{regime} | "
                    f"Samples: {count} | "
                    f"Success: {positive_count} | "
                    f"Failure: {negative_count} | "
                    f"Success Rate: {success_rate:.2%}"
                )

            else:

                print(
                    f"{regime} | Samples: 0 | "
                    "لا توجد بيانات لهذه الحالة."
                )

        diagnostic_results[period_name] = period_data

    print("\n---------- DIAGNOSTIC SUMMARY ----------")

    train_bearish = diagnostic_results.get(
        "TRAIN", {}
    ).get("BEARISH", {}).get("samples", 0)

    test_bearish = diagnostic_results.get(
        "TEST", {}
    ).get("BEARISH", {}).get("samples", 0)

    print(
        f"BEARISH samples in TRAIN: {train_bearish}"
    )

    print(
        f"BEARISH samples in TEST: {test_bearish}"
    )

    if train_bearish == 0 and test_bearish > 0:

        print(
            "\n⚠️ WARNING: لا توجد عينات BEARISH في التدريب، "
            "بينما توجد عينات BEARISH في الاختبار."
        )

        print(
            "النموذج لم يتدرب على أمثلة فعلية من السوق الهابط."
        )

    elif train_bearish > 0:

        print(
            "\n✅ توجد عينات BEARISH داخل بيانات التدريب."
        )

    else:

        print(
            "\n⚠️ لا توجد عينات BEARISH في التدريب أو الاختبار."
        )

    print("=" * 65)
    print("========== End of Diagnostics ==========")
    print("=" * 65)

    return diagnostic_results


# =====================
# Walk-Forward Validation
# =====================

def run_walk_forward_validation(
    df,
    X,
    y,
    calendar,
    final_test_start_index
):

    print("\n")
    print("=" * 65)
    print("========== WALK-FORWARD VALIDATION ==========")
    print("=" * 65)

    development_calendar = calendar[
        :final_test_start_index
    ]

    development_sessions = len(
        development_calendar
    )

    fold_size = (
        development_sessions // 10
    )

    if fold_size <= 0:
        raise ValueError(
            "عدد جلسات التطوير غير كافٍ لإنشاء Walk-Forward."
        )

    fold_rows = []

    fold_results = []

    first_test_index = int(
        development_sessions * 0.50
    )

    test_starts = [
        first_test_index + i * fold_size
        for i in range(WALK_FORWARD_FOLDS)
    ]

    for fold_number, test_start in enumerate(
        test_starts,
        start=1
    ):

        test_end = min(
            test_start + fold_size,
            development_sessions
        )

        train_end = (
            test_start - TARGET_HORIZON
        )

        if train_end <= 0:
            print(
                f"⚠️ Fold {fold_number}: "
                "عدد جلسات التدريب غير كافٍ."
            )
            continue

        train_calendar = development_calendar[
            :train_end
        ]

        purged_calendar = development_calendar[
            train_end:test_start
        ]

        test_calendar = development_calendar[
            test_start:test_end
        ]

        if len(train_calendar) < MIN_TRAIN_SESSIONS:
            print(
                f"⚠️ Fold {fold_number}: "
                f"التدريب أقل من {MIN_TRAIN_SESSIONS} جلسة."
            )
            continue

        if len(test_calendar) == 0:
            continue

        train_mask = df["date"].isin(
            train_calendar
        )

        test_mask = df["date"].isin(
            test_calendar
        )

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        if X_train.empty or X_test.empty:
            print(
                f"⚠️ Fold {fold_number}: "
                "بيانات التدريب أو الاختبار فارغة."
            )
            continue

        if y_train.nunique() < 2:
            print(
                f"⚠️ Fold {fold_number}: "
                "التدريب يحتوي على فئة واحدة."
            )
            continue

        model = create_model()

        model.fit(
            X_train,
            y_train
        )

        y_probability = get_positive_probabilities(
            model,
            X_test
        )

        predictions = (
            y_probability >= 0.50
        ).astype(int)

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

        positive_rate = float(
            y_test.mean()
        )

        if y_test.nunique() == 2:

            roc_auc = roc_auc_score(
                y_test,
                y_probability
            )

            pr_auc = average_precision_score(
                y_test,
                y_probability
            )

            brier = brier_score_loss(
                y_test,
                y_probability
            )

        else:

            roc_auc = None
            pr_auc = None
            brier = brier_score_loss(
                y_test,
                y_probability
            )

        baseline_probability = float(
            y_train.mean()
        )

        baseline_brier = brier_score_loss(
            y_test,
            np.full(
                len(y_test),
                baseline_probability,
                dtype=float
            )
        )

        result = {
            "fold": fold_number,
            "train_start": str(
                train_calendar[0].date()
            ),
            "train_end": str(
                train_calendar[-1].date()
            ),
            "purged_sessions": len(
                purged_calendar
            ),
            "test_start": str(
                test_calendar[0].date()
            ),
            "test_end": str(
                test_calendar[-1].date()
            ),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_positive_rate": round(
                baseline_probability, 4
            ),
            "test_positive_rate": round(
                positive_rate, 4
            ),
            "roc_auc": (
                round(float(roc_auc), 4)
                if roc_auc is not None else None
            ),
            "pr_auc": (
                round(float(pr_auc), 4)
                if pr_auc is not None else None
            ),
            "brier_score": round(
                float(brier), 4
            ),
            "baseline_brier_score": round(
                float(baseline_brier), 4
            ),
            "brier_better_than_baseline": bool(
                brier < baseline_brier
            ),
            "precision_50": round(
                float(precision), 4
            ),
            "recall_50": round(
                float(recall), 4
            ),
            "f1_50": round(
                float(f1), 4
            ),
            "signals_50": int(
                predictions.sum()
            )
        }

        fold_results.append(result)

        fold_rows.append(result)

        print(
            f"\n---------- FOLD {fold_number} ----------"
        )

        print(
            f"Train: {result['train_start']} "
            f"-> {result['train_end']}"
        )

        print(
            f"Purged Sessions: {result['purged_sessions']}"
        )

        print(
            f"Test: {result['test_start']} "
            f"-> {result['test_end']}"
        )

        print(
            f"Train Samples: {result['train_samples']}"
        )

        print(
            f"Test Samples: {result['test_samples']}"
        )

        print(
            f"ROC-AUC: {result['roc_auc']}"
        )

        print(
            f"PR-AUC: {result['pr_auc']}"
        )

        print(
            f"Brier: {result['brier_score']}"
        )

        print(
            f"Baseline Brier: "
            f"{result['baseline_brier_score']}"
        )

        print(
            f"Precision 50%: {result['precision_50']}"
        )

        print(
            f"Recall 50%: {result['recall_50']}"
        )

    if not fold_rows:

        raise ValueError(
            "لم يتم إنشاء أي Fold صالح في Walk-Forward."
        )

    results_df = pd.DataFrame(
        fold_rows
    )

    results_df.to_csv(
        WALK_FORWARD_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    valid_roc = results_df[
        "roc_auc"
    ].dropna()

    valid_pr = results_df[
        "pr_auc"
    ].dropna()

    summary = {
        "folds_completed": int(
            len(results_df)
        ),
        "walk_forward_results_file": (
            WALK_FORWARD_RESULTS_FILE
        ),
        "average_roc_auc": (
            round(float(valid_roc.mean()), 4)
            if not valid_roc.empty else None
        ),
        "average_pr_auc": (
            round(float(valid_pr.mean()), 4)
            if not valid_pr.empty else None
        ),
        "average_brier_score": round(
            float(results_df["brier_score"].mean()),
            4
        ),
        "average_baseline_brier_score": round(
            float(
                results_df[
                    "baseline_brier_score"
                ].mean()
            ),
            4
        ),
        "folds_better_than_baseline": int(
            results_df[
                "brier_better_than_baseline"
            ].sum()
        )
    }

    print("\n========== WALK-FORWARD SUMMARY ==========")

    print(
        f"Completed Folds: {summary['folds_completed']}"
    )

    print(
        f"Average ROC-AUC: {summary['average_roc_auc']}"
    )

    print(
        f"Average PR-AUC: {summary['average_pr_auc']}"
    )

    print(
        f"Average Brier: {summary['average_brier_score']}"
    )

    print(
        f"Average Baseline Brier: "
        f"{summary['average_baseline_brier_score']}"
    )

    print(
        f"Folds Better Than Baseline: "
        f"{summary['folds_better_than_baseline']}"
    )

    print(
        f"✅ تم حفظ نتائج Walk-Forward: "
        f"{WALK_FORWARD_RESULTS_FILE}"
    )

    print("=" * 65)

    return summary


# =====================
# Training
# =====================

def train_model():

    print("-> جاري تحميل بيانات التدريب...")

    df, feature_columns = load_dataset()

    # 1. Load actual EGX30 calendar

    calendar = load_egx30_calendar()

    # 2. Validate horizon

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

    # 3. Validate dates

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

    df["date"] = df["date"].dt.normalize()

    # 4. Validate target column

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

    df[TARGET_COLUMN] = df[
        TARGET_COLUMN
    ].astype(int)

    # 5. Verify target against raw future data

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

    if not expected_target.equals(
        df[TARGET_COLUMN]
    ):
        raise ValueError(
            "عدم تطابق Target_Reach_8Pct مع Max_Gain_Pct. "
            "يجب مراجعة بناء Dataset."
        )

    # 6. Sort chronologically

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

    # Validate dataset dates against EGX30 calendar

    calendar_set = set(calendar)

    unknown_dates = sorted(
        set(df["date"].unique()) - calendar_set
    )

    if unknown_dates:

        raise ValueError(
            "توجد تواريخ في Dataset غير موجودة "
            "في تقويم EGX30. "
            f"عددها: {len(unknown_dates)}. "
            "يجب التأكد من تطابق مصادر التواريخ."
        )

    # Preserve original market categories

    for column in (
        "Market_Regime",
        "Market_Trend",
        "Market_Momentum"
    ):

        df[column + "_Original"] = df[
            column
        ].copy()

    # 7. Encode market categories

    df = encode_market_features(df)

    # 8. Build features without future information

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

    # 9. Final chronological split using EGX30 calendar

    total_calendar_sessions = len(calendar)

    split_index = int(
        total_calendar_sessions * TRAIN_RATIO
    )

    train_end_index = (
        split_index - TARGET_HORIZON
    )

    if (
        train_end_index <= 0
        or split_index >= total_calendar_sessions
    ):

        raise ValueError(
            "عدد جلسات EGX30 غير كافٍ "
            "لإنشاء مجموعتي التدريب والاختبار."
        )

    train_dates = calendar[
        :train_end_index
    ]

    purged_dates = calendar[
        train_end_index:split_index
    ]

    test_dates = calendar[
        split_index:
    ]

    train_mask = df["date"].isin(
        train_dates
    )

    purged_mask = df["date"].isin(
        purged_dates
    )

    test_mask = df["date"].isin(
        test_dates
    )

    X_train = X.loc[train_mask]

    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]

    y_test = y.loc[test_mask]

    print("\n========== التقسيم الزمني ==========")

    print(
        f"إجمالي جلسات EGX30: "
        f"{total_calendar_sessions}"
    )

    print(
        f"جلسات التدريب: {len(train_dates)}"
    )

    print(
        f"جلسات الاستبعاد Purged: "
        f"{len(purged_dates)}"
    )

    print(
        f"جلسات الاختبار: {len(test_dates)}"
    )

    print(
        f"عينات التدريب: {len(X_train)}"
    )

    print(
        f"عينات الاستبعاد: {int(purged_mask.sum())}"
    )

    print(
        f"عينات الاختبار: {len(X_test)}"
    )

    print(
        f"بداية التدريب: {train_dates[0].date()}"
    )

    print(
        f"نهاية التدريب: {train_dates[-1].date()}"
    )

    print(
        f"بداية الاختبار: {test_dates[0].date()}"
    )

    print(
        f"نهاية الاختبار: {test_dates[-1].date()}"
    )

    print(
        f"جلسات Purging الفعلية: {len(purged_dates)}"
    )

    print("====================================\n")

    if X_train.empty or X_test.empty:

        raise ValueError(
            "بيانات التدريب أو الاختبار فارغة."
        )

    # 10. Market Regime Diagnostics

    regime_diagnostics = analyze_market_distribution(
        df,
        train_mask,
        purged_mask,
        test_mask
    )

    # 11. Walk-Forward Validation

    walk_forward_summary = run_walk_forward_validation(
        df,
        X,
        y,
        calendar,
        split_index
    )

    # 12. Validate classes

    if y_train.nunique() < 2:

        raise ValueError(
            "بيانات التدريب تحتوي على فئة واحدة فقط."
        )

    if y_test.nunique() < 2:

        print(
            "⚠️ تحذير: الاختبار النهائي يحتوي على فئة واحدة فقط."
        )

    print(
        f"توزيع التدريب: "
        f"{dict(y_train.value_counts().sort_index())}"
    )

    print(
        f"توزيع الاختبار: "
        f"{dict(y_test.value_counts().sort_index())}"
    )

    # 13. Create final model

    if HAS_XGBOOST:

        print("-> استخدام نموذج XGBoost...")

    else:

        print("-> استخدام نموذج RandomForest...")

    model = create_model()

    # 14. Train final model

    print("\n-> بدء تدريب النموذج النهائي...")

    model.fit(
        X_train,
        y_train
    )

    # 15. Evaluate final holdout test

    evaluation = evaluate_model(
        model,
        X_test,
        y_test,
        baseline_probability=float(
            y_train.mean()
        )
    )

    y_probability = get_positive_probabilities(
        model,
        X_test
    )

    # 16. Save detailed test results

    test_results = save_test_results(
        df,
        test_mask,
        y_test,
        y_probability
    )

    # 17. Analyze performance by market regime

    regime_analysis = analyze_segment(
        test_results,
        "Market_Regime",
        "تحليل الأداء حسب حالة السوق EGX30"
    )

    # 18. Analyze performance by stock

    stock_analysis = analyze_segment(
        test_results,
        "Symbol",
        "تحليل الأداء حسب كل سهم"
    )

    # 19. Feature importance

    feature_importance = analyze_feature_importance(
        model,
        list(X.columns)
    )

    # 20. Save model and metadata

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
        "calendar_source": FEATURES_FILE,
        "calendar_symbol": "EGX30",
        "calendar_sessions": int(
            total_calendar_sessions
        ),
        "train_start_date": str(
            train_dates[0].date()
        ),
        "train_end_date": str(
            train_dates[-1].date()
        ),
        "test_start_date": str(
            test_dates[0].date()
        ),
        "test_end_date": str(
            test_dates[-1].date()
        ),
        "train_samples": int(
            X_train.shape[0]
        ),
        "test_samples": int(
            X_test.shape[0]
        ),
        "train_positive_rate": round(
            float(y_train.mean()), 4
        ),
        "test_positive_rate": round(
            float(y_test.mean()), 4
        ),
        "evaluation": evaluation,
        "walk_forward_validation": walk_forward_summary,
        "diagnostics": {
            "test_results_file": TEST_RESULTS_FILE,
            "feature_importance_file": FEATURE_IMPORTANCE_FILE,
            "walk_forward_results_file": WALK_FORWARD_RESULTS_FILE,
            "market_regime_distribution": regime_diagnostics,
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

    print(
        f"\n-> تم حفظ النموذج: {model_filename}"
    )

    print(
        f"-> تم حفظ Metadata: {meta_filename}"
    )

    print(
        f"-> تم حفظ نتائج الاختبار: {TEST_RESULTS_FILE}"
    )

    print(
        f"-> تم حفظ أهمية المؤشرات: "
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    print(
        f"-> تم حفظ نتائج Walk-Forward: "
        f"{WALK_FORWARD_RESULTS_FILE}"
    )

    print(
        "-> اكتملت عملية التدريب والتقييم والتحليل بنجاح."
    )


if __name__ == "__main__":
    train_model()

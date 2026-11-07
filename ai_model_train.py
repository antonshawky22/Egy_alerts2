from datetime import datetime
import json
import os
import joblib
import pandas as pd
import numpy as np
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
HORIZONS = {
    20: 8.0,
    40: 12.0,
    60: 18.0
}

TRAIN_RATIO = 0.80
DATASET_FILE = "ai_training_dataset.json"

# Explicit feature allowlist to prevent accidental target leakage
FEATURES = [
    "RSI14", "ATR14_Pct", "Volume_Ratio20",
    "Return_5", "Return_10", "Return_20",
    "Return_50", "Return_100",
    "Distance_EMA20_Pct", "Distance_EMA50_Pct",
    "Distance_EMA100_Pct",
    "Distance_Resistance20_Pct",
    "Distance_Support20_Pct",
    "Trend_20_50", "Trend_50_100",
    "Market_Score", "Market_Volatility",
    "Market_Regime", "Market_Trend", "Market_Momentum"
]

REGIME_MAPPING = {
    "BEARISH": -1,
    "NEUTRAL": 0,
    "BULLISH": 1,
    "INSUFFICIENT_DATA": 0
}

TREND_MAPPING = {
    "DOWN": -1,
    "MIXED": 0,
    "UP": 1
}

MOMENTUM_MAPPING = {
    "NEGATIVE": -1,
    "MIXED": 0,
    "POSITIVE": 1
}


def load_dataset(file_path=DATASET_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"الملف {file_path} غير موجود."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "samples" in data:
        df = pd.DataFrame(data["samples"])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        raise ValueError("صيغة ملف JSON غير متوافقة.")

    if df.empty:
        raise ValueError("ملف التدريب لا يحتوي على عينات.")

    return df


def prepare_features(df):
    df = df.copy()

    for col, mapping in [
        ("Market_Regime", REGIME_MAPPING),
        ("Market_Trend", TREND_MAPPING),
        ("Market_Momentum", MOMENTUM_MAPPING)
    ]:
        if col in df.columns:
            df[col] = df[col].map(mapping)

    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        raise ValueError(
            f"ميزات مطلوبة غير موجودة في Dataset: {missing}"
        )

    X = df[FEATURES].copy()

    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"توجد قيم مفقودة أو غير صالحة في الميزات: {bad}"
        )

    return X


def create_model():
    if HAS_XGBOOST:
        return xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )


def calculate_performance(data):
    if data.empty:
        return {
            "samples": 0,
            "success_rate": None,
            "average_return": None,
            "average_win": None,
            "average_loss": None,
            "profit_factor": None,
            "average_max_drawdown": None
        }

    returns = pd.to_numeric(
        data["Future_Return_Pct"], errors="coerce"
    ).dropna()

    if returns.empty:
        return {"samples": 0}

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else None
    )

    drawdowns = pd.to_numeric(
        data["Max_Drawdown_Pct"], errors="coerce"
    )

    return {
        "samples": int(len(returns)),
        "success_rate": round(
            float((data.loc[returns.index, "Actual"] == 1).mean()) * 100,
            2
        ),
        "average_return": round(float(returns.mean()), 2),
        "average_win": round(float(wins.mean()), 2) if len(wins) else None,
        "average_loss": round(float(losses.mean()), 2) if len(losses) else None,
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "average_max_drawdown": round(float(drawdowns.mean()), 2)
        if drawdowns.notna().any() else None
    }


def evaluate_ranking(results):
    output = {}

    ranked = results.sort_values(
        "Probability", ascending=False
    ).reset_index(drop=True)

    for percentage in (10, 20, 30, 50):
        count = max(1, int(len(ranked) * percentage / 100))
        selected = ranked.head(count)

        output[f"top_{percentage}_percent"] = {
            "selected": count,
            **calculate_performance(selected)
        }

    return output


def train_horizon(base_df, horizon, threshold):
    print("\n" + "=" * 55)
    print(f"AI MODEL - {horizon} SESSIONS")
    print(f"Target return: {threshold}%")
    print("=" * 55)

    df = base_df[
        base_df["horizon"] == horizon
    ].copy()

    if df.empty:
        raise ValueError(
            f"لا توجد عينات للأفق {horizon}."
        )

    required = [
        "date", "Future_Return_Pct",
        "Max_Drawdown_Pct"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"العمود {col} غير موجود.")

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df["Future_Return_Pct"] = pd.to_numeric(
        df["Future_Return_Pct"], errors="coerce"
    )

    df["Max_Drawdown_Pct"] = pd.to_numeric(
        df["Max_Drawdown_Pct"], errors="coerce"
    )

    df = df.dropna(
        subset=["date", "Future_Return_Pct"]
    ).copy()

    if df.empty:
        raise ValueError("لا توجد عينات صالحة بعد التنظيف.")

    df = df.sort_values("date").reset_index(drop=True)

    # Target: reaching the agreed return at the horizon close
    df["target"] = (
        df["Future_Return_Pct"] >= threshold
    ).astype(int)

    X = prepare_features(df)
    y = df["target"].copy()

    unique_dates = sorted(df["date"].unique())
    total_dates = len(unique_dates)

    split_index = int(total_dates * TRAIN_RATIO)

    # Purge the full prediction horizon
    train_end_index = split_index - horizon

    if train_end_index <= 0 or split_index >= total_dates:
        raise ValueError(
            f"التواريخ غير كافية للأفق {horizon}. "
            f"المتاح {total_dates} تاريخًا."
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

    if X_train.empty or X_test.empty:
        raise ValueError("مجموعة التدريب أو الاختبار فارغة.")

    if y_train.nunique() < 2:
        raise ValueError(
            "مجموعة التدريب تحتوي على فئة واحدة فقط."
        )

    if y_test.nunique() < 2:
        print(
            "WARNING: مجموعة الاختبار تحتوي على فئة واحدة فقط."
        )

    print(f"Total dates: {total_dates}")
    print(f"Training dates: {len(train_dates)}")
    print(f"Purged dates: {len(purged_dates)}")
    print(f"Testing dates: {len(test_dates)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training success rate: {y_train.mean() * 100:.2f}%")
    print(f"Testing success rate: {y_test.mean() * 100:.2f}%")

    # Train
    model = create_model()
    model.fit(X_train, y_train)

    buy_class_idx = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, buy_class_idx]

    # Full test results
    results = df.loc[test_mask, [
        "date", "Future_Return_Pct", "Max_Drawdown_Pct"
    ]].copy()

    results["Actual"] = y_test.to_numpy()
    results["Probability"] = probabilities

    if "symbol" in df.columns:
        results["Symbol"] = df.loc[test_mask, "symbol"].to_numpy()
    elif "ticker" in df.columns:
        results["Symbol"] = df.loc[test_mask, "ticker"].to_numpy()

    if "Max_Gain_Pct" in df.columns:
        results["Max_Gain_Pct"] = pd.to_numeric(
            df.loc[test_mask, "Max_Gain_Pct"],
            errors="coerce"
        ).to_numpy()

    if "Sessions_To_Max" in df.columns:
        results["Sessions_To_Max"] = df.loc[
            test_mask, "Sessions_To_Max"
        ].to_numpy()

    if "Sessions_To_Min" in df.columns:
        results["Sessions_To_Min"] = df.loc[
            test_mask, "Sessions_To_Min"
        ].to_numpy()

    results["Horizon"] = horizon
    results["Target_Threshold"] = threshold

    results = results.sort_values(
        "Probability", ascending=False
    )

    # Save results
    result_file = f"ai_test_results_{horizon}d.csv"
    results.to_csv(result_file, index=False)

    if horizon == 20:
        results.to_csv("ai_test_results.csv", index=False)

    print(f"Saved: {result_file}")

    # Classification evaluation
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # Performance of all test opportunities
    overall = calculate_performance(results)

    # Ranking performance
    ranking = evaluate_ranking(results)

    # Probability threshold evaluation
    probability_groups = {}

    for cutoff in (0.50, 0.60, 0.70, 0.80):
        selected = results[
            results["Probability"] >= cutoff
        ]

        probability_groups[str(cutoff)] = calculate_performance(
            selected
        )

    print("\nOverall test performance:")
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    print("\nTop ranking performance:")
    print(json.dumps(ranking, ensure_ascii=False, indent=2))

    # Feature importance
    importance = {}

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_

        importance = {
            name: round(float(value), 6)
            for name, value in sorted(
                zip(FEATURES, values),
                key=lambda x: x[1],
                reverse=True
            )
        }

    # Save model
    model_file = f"ai_stock_model_{horizon}d.pkl"
    meta_file = f"ai_stock_model_{horizon}d_meta.json"
    report_file = f"ai_model_report_{horizon}d.json"

    joblib.dump(model, model_file)

    metadata = {
        "model_file": model_file,
        "horizon": horizon,
        "profit_threshold": threshold,
        "target_definition": f"Future_Return_Pct >= {threshold}",
        "feature_columns": FEATURES,
        "model_type": "XGBoost" if HAS_XGBOOST else "RandomForest",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_ratio": TRAIN_RATIO,
        "purged_sessions": horizon,
        "train_start_date": str(pd.Timestamp(train_dates[0]).date()),
        "train_end_date": str(pd.Timestamp(train_dates[-1]).date()),
        "test_start_date": str(pd.Timestamp(test_dates[0]).date()),
        "test_end_date": str(pd.Timestamp(test_dates[-1]).date()),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    }

    report = {
        "metadata": metadata,
        "overall_performance": overall,
        "ranking_performance": ranking,
        "probability_groups": probability_groups,
        "feature_importance": importance
    }

    for filename, content in [
        (meta_file, metadata),
        (report_file, report)
    ]:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                content, f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False
            )

    print(f"\nModel saved: {model_file}")
    print(f"Metadata saved: {meta_file}")
    print(f"Report saved: {report_file}")
    print(f"Completed AI-{horizon}.")


def train_all():
    print("Loading AI training dataset...")

    df = load_dataset()

    if "horizon" not in df.columns:
        raise ValueError("عمود horizon غير موجود.")

    df["horizon"] = pd.to_numeric(
        df["horizon"], errors="coerce"
    )

    for horizon, threshold in HORIZONS.items():
        train_horizon(df, horizon, threshold)

    print("\n" + "=" * 55)
    print("ALL THREE AI MODELS TRAINED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    train_all()

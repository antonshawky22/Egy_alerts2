import json
import math
from datetime import datetime, timezone

FEATURES_FILE = "ai_features_database.json"
LABELS_FILE = "ai_labels_database.json"
MARKET_FILE = "ai_market_regime_database.json"
OUT_FILE = "ai_training_dataset.json"

TARGET_HORIZON = 20
TARGET_THRESHOLD = 8.0
HORIZONS = (TARGET_HORIZON,)
TARGET_COLUMN = "Target_Reach_8Pct"

STOCK_FEATURES = [
    "RSI14",
    "ATR14_Pct",
    "Volume_Ratio20",
    "Return_5",
    "Return_10",
    "Return_20",
    "Return_50",
    "Return_100",
    "Distance_EMA20_Pct",
    "Distance_EMA50_Pct",
    "Distance_EMA100_Pct",
    "Distance_Resistance20_Pct",
    "Distance_Support20_Pct",
    "Trend_20_50",
    "Trend_50_100"
]

RAW_TARGET_COLUMNS = [
    "Future_Return_Pct",
    "Max_Gain_Pct",
    "Max_Drawdown_Pct",
    "Sessions_To_Max",
    "Sessions_To_Min"
]

TARGET_COLUMNS = RAW_TARGET_COLUMNS + [TARGET_COLUMN]

MARKET_NUMERIC = ["Market_Score", "Market_Volatility"]

MARKET_CATEGORIES = {
    "Market_Regime": ("BULLISH", "NEUTRAL", "BEARISH"),
    "Market_Trend": ("UP", "DOWN", "MIXED"),
    "Market_Momentum": ("POSITIVE", "NEGATIVE", "MIXED")
}

def safe_num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def validate_target_definition(labels_db):
    definition = labels_db.get("target_definition", {})

    if definition.get("type") != "max_gain_anytime":
        raise ValueError(
            "Invalid labels target definition. "
            "Expected type: max_gain_anytime"
        )

    if definition.get("horizon") != TARGET_HORIZON:
        raise ValueError(
            f"Invalid target horizon. Expected {TARGET_HORIZON}"
        )

    threshold = safe_num(definition.get("threshold_pct"))

    if threshold != TARGET_THRESHOLD:
        raise ValueError(
            f"Invalid target threshold. Expected {TARGET_THRESHOLD}%"
        )

    print("Target definition: VALID")

def main():
    with open(FEATURES_FILE, encoding="utf-8") as f:
        features_db = json.load(f)

    with open(LABELS_FILE, encoding="utf-8") as f:
        labels_db = json.load(f)

    with open(MARKET_FILE, encoding="utf-8") as f:
        market_db = json.load(f)

    validate_target_definition(labels_db)

    stock_symbols = features_db["symbols"]
    stock_labels = labels_db["symbols"]
    market_data = market_db["data"]

    samples = []

    stats = {
        "dates_checked": 0,
        "processed": 0,
        "missing_label_dates": 0,
        "missing_label_symbols": 0,
        "missing_market_dates": 0,
        "invalid_market_dates": 0,
        "invalid_feature_dates": 0,
        "invalid_feature_values": 0,
        "invalid_features_by_name": {},
        "missing_market_values": 0,
        "incomplete_labels": {"20": 0},
        "complete_labels": {"20": 0},
        "invalid_targets": {"20": 0},
        "samples_by_horizon": {"20": 0},
        "missing_symbols_labels": [],
        "excluded_invalid_features": 0,
        "excluded_invalid_market": 0,
        "excluded_invalid_labels": 0,
        "target_counts": {"0": 0, "1": 0}
    }

    for symbol, block in stock_symbols.items():
        if symbol == "EGX30":
            continue

        if symbol not in stock_labels:
            print(f"⚠️ Missing labels: {symbol}")
            stats["missing_label_symbols"] += 1
            stats["missing_symbols_labels"].append(symbol)
            continue

        stock_data = block.get("data", {})
        symbol_labels = stock_labels[symbol]

        for date in sorted(stock_data):
            stats["dates_checked"] += 1

            row = stock_data[date]
            labels_by_date = symbol_labels.get(date)

            if not labels_by_date:
                stats["missing_label_dates"] += 1

            # 1. Validate stock features
            stock_features = {}
            invalid_names = []

            for name in STOCK_FEATURES:
                value = safe_num(row.get(name))

                if value is None:
                    invalid_names.append(name)
                    stats["invalid_feature_values"] += 1
                    stats["invalid_features_by_name"][name] = (
                        stats["invalid_features_by_name"].get(name, 0) + 1
                    )
                else:
                    stock_features[name] = value

            if invalid_names:
                stats["invalid_feature_dates"] += 1

            # 2. Validate EGX30 market data
            market = market_data.get(date)
            market_features = {}
            market_invalid = False

            if not market or market.get("regime") == "INSUFFICIENT_DATA":
                stats["missing_market_dates"] += 1
                market_invalid = True
            else:
                market_features = {
                    "Market_Score": safe_num(market.get("score")),
                    "Market_Volatility": safe_num(market.get("volatility")),
                    "Market_Regime": market.get("regime"),
                    "Market_Trend": market.get("trend"),
                    "Market_Momentum": market.get("momentum")
                }

                for name in MARKET_NUMERIC:
                    if market_features.get(name) is None:
                        stats["missing_market_values"] += 1
                        market_invalid = True

                for name, allowed in MARKET_CATEGORIES.items():
                    if market_features.get(name) not in allowed:
                        stats["missing_market_values"] += 1
                        market_invalid = True

                if market_invalid:
                    stats["invalid_market_dates"] += 1

            # 3. Validate labels for 20 sessions only
            valid_labels = None
            label = {}

            if labels_by_date:
                label = labels_by_date.get(
                    f"{TARGET_HORIZON}_sessions", {}
                )

                if not label.get("complete"):
                    stats["incomplete_labels"]["20"] += 1
                else:
                    targets = {
                        name: safe_num(label.get(source))
                        for name, source in {
                            "Future_Return_Pct": "future_return_pct",
                            "Max_Gain_Pct": "max_gain_pct",
                            "Max_Drawdown_Pct": "max_drawdown_pct",
                            "Sessions_To_Max": "sessions_to_max",
                            "Sessions_To_Min": "sessions_to_min"
                        }.items()
                    }

                    if any(v is None for v in targets.values()):
                        stats["invalid_targets"]["20"] += 1
                    else:
                        valid_labels = targets
                        stats["complete_labels"]["20"] += 1

            # 4. Exclude invalid samples
            if invalid_names:
                stats["excluded_invalid_features"] += 1

            if market_invalid:
                stats["excluded_invalid_market"] += 1

            if valid_labels is None:
                stats["excluded_invalid_labels"] += 1

            if invalid_names or market_invalid or valid_labels is None:
                continue

            # 5. Define the actual AI target
            max_gain = valid_labels["Max_Gain_Pct"]

            target_value = int(max_gain > TARGET_THRESHOLD)

            # 6. Build final sample
            sample = {
                "symbol": symbol,
                "date": date,
                "horizon": TARGET_HORIZON,
                **stock_features,
                **market_features,
                **valid_labels,
                TARGET_COLUMN: target_value
            }

            samples.append(sample)

            stats["processed"] += 1
            stats["samples_by_horizon"]["20"] += 1
            stats["target_counts"][str(target_value)] += 1

    # 7. Calculate target distribution
    total_samples = len(samples)
    positive_count = stats["target_counts"]["1"]
    negative_count = stats["target_counts"]["0"]

    positive_pct = (
        round(positive_count / total_samples * 100, 2)
        if total_samples else 0
    )

    negative_pct = (
        round(negative_count / total_samples * 100, 2)
        if total_samples else 0
    )

    target_definition = {
        "type": "max_gain_anytime",
        "horizon": TARGET_HORIZON,
        "threshold_pct": TARGET_THRESHOLD,
        "target_column": TARGET_COLUMN,
        "condition": "Max_Gain_Pct > 8.0"
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": [
            FEATURES_FILE,
            LABELS_FILE,
            MARKET_FILE
        ],
        "excluded_symbol": "EGX30",
        "horizons": list(HORIZONS),
        "target_definition": target_definition,
        "feature_columns": STOCK_FEATURES,
        "market_feature_columns": [
            "Market_Score",
            "Market_Volatility",
            "Market_Regime",
            "Market_Trend",
            "Market_Momentum"
        ],
        "raw_target_columns": RAW_TARGET_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "statistics": stats,
        "target_distribution": {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_pct": positive_pct,
            "negative_pct": negative_pct
        },
        "total_samples": total_samples,
        "samples": samples
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False
        )

    print("\n========== AI TRAINING DATASET ==========")
    print(f"Stocks: {len(stock_symbols) - 1}")
    print(f"Dates checked: {stats['dates_checked']}")
    print(f"Total samples: {total_samples}")
    print(f"Processed: {stats['processed']}")

    print("\n--- Target Definition ---")
    print(f"Horizon: {TARGET_HORIZON} sessions")
    print(f"Required gain: More than {TARGET_THRESHOLD}%")
    print(f"Target column: {TARGET_COLUMN}")

    print("\n--- Target Distribution ---")
    print(f"Positive (1): {positive_count} ({positive_pct}%)")
    print(f"Negative (0): {negative_count} ({negative_pct}%)")

    print("\n--- Data Quality ---")
    print(f"Missing label dates: {stats['missing_label_dates']}")
    print(f"Missing label symbols: {stats['missing_label_symbols']}")
    print(f"Missing market dates: {stats['missing_market_dates']}")
    print(f"Invalid market dates: {stats['invalid_market_dates']}")
    print(f"Invalid feature dates: {stats['invalid_feature_dates']}")
    print(f"Invalid feature values: {stats['invalid_feature_values']}")
    print(f"Missing market values: {stats['missing_market_values']}")

    print("\n--- Excluded Samples ---")
    print(f"Invalid features: {stats['excluded_invalid_features']}")
    print(f"Invalid market: {stats['excluded_invalid_market']}")
    print(f"Invalid labels: {stats['excluded_invalid_labels']}")

    print("\n--- Invalid Features ---")
    for name, count in sorted(stats["invalid_features_by_name"].items()):
        print(f"{name}: {count}")

    print("\n--- Labels ---")
    print(
        f"20 sessions: "
        f"Complete={stats['complete_labels']['20']} | "
        f"Incomplete={stats['incomplete_labels']['20']} | "
        f"Invalid targets={stats['invalid_targets']['20']}"
    )

    print(f"\nOutput: {OUT_FILE}")
    print("Status: SUCCESS")
    print("=========================================")

if __name__ == "__main__":
    main()

import json
import math
from datetime import datetime, timezone

FEATURES_FILE = "ai_features_database.json"
LABELS_FILE = "ai_labels_database.json"
MARKET_FILE = "ai_market_regime_database.json"
OUT_FILE = "ai_training_dataset.json"

HORIZONS = (20, 40, 60)

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

TARGET_COLUMNS = [
    "Future_Return_Pct",
    "Max_Gain_Pct",
    "Max_Drawdown_Pct",
    "Sessions_To_Max",
    "Sessions_To_Min"
]

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

def main():
    with open(FEATURES_FILE, encoding="utf-8") as f:
        features_db = json.load(f)

    with open(LABELS_FILE, encoding="utf-8") as f:
        labels_db = json.load(f)

    with open(MARKET_FILE, encoding="utf-8") as f:
        market_db = json.load(f)

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
        "incomplete_labels": {},
        "complete_labels": {},
        "invalid_targets": {},
        "samples_by_horizon": {},
        "missing_symbols_labels": []
    }

    for period in HORIZONS:
        key = str(period)
        stats["incomplete_labels"][key] = 0
        stats["complete_labels"][key] = 0
        stats["invalid_targets"][key] = 0
        stats["samples_by_horizon"][key] = 0

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

            # 1. Validate all stock features
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

            # 2. Validate market data
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

            # 3. Validate labels independently for each horizon
            valid_labels = {}

            if labels_by_date:
                for period in HORIZONS:
                    key = str(period)
                    label = labels_by_date.get(f"{period}_sessions", {})

                    if not label.get("complete"):
                        stats["incomplete_labels"][key] += 1
                        continue

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
                        stats["invalid_targets"][key] += 1
                        continue

                    stats["complete_labels"][key] += 1
                    valid_labels[period] = targets

            # 4. Exclude dates with invalid features or market data
            if invalid_names or market_invalid or not labels_by_date:
                continue

            # 5. Build samples
            for period in HORIZONS:
                if period not in valid_labels:
                    continue

                sample = {
                    "symbol": symbol,
                    "date": date,
                    "horizon": period,
                    **stock_features,
                    **market_features,
                    **valid_labels[period]
                }

                samples.append(sample)
                stats["processed"] += 1
                stats["samples_by_horizon"][str(period)] += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": [
            FEATURES_FILE,
            LABELS_FILE,
            MARKET_FILE
        ],
        "excluded_symbol": "EGX30",
        "horizons": list(HORIZONS),
        "feature_columns": STOCK_FEATURES,
        "market_feature_columns": [
            "Market_Score",
            "Market_Volatility",
            "Market_Regime",
            "Market_Trend",
            "Market_Momentum"
        ],
        "target_columns": TARGET_COLUMNS,
        "statistics": stats,
        "total_samples": len(samples),
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
    print(f"Total samples: {len(samples)}")
    print(f"Processed: {stats['processed']}")

    print("\n--- Samples by Horizon ---")
    for period in HORIZONS:
        key = str(period)
        print(
            f"{period} sessions: "
            f"{stats['samples_by_horizon'][key]}"
        )

    print("\n--- Data Quality ---")
    print(f"Missing label dates: {stats['missing_label_dates']}")
    print(f"Missing label symbols: {stats['missing_label_symbols']}")
    print(f"Missing market dates: {stats['missing_market_dates']}")
    print(f"Invalid market dates: {stats['invalid_market_dates']}")
    print(f"Invalid feature dates: {stats['invalid_feature_dates']}")
    print(f"Invalid feature values: {stats['invalid_feature_values']}")
    print(f"Missing market values: {stats['missing_market_values']}")

    print("\n--- Invalid Features ---")
    for name, count in sorted(stats["invalid_features_by_name"].items()):
        print(f"{name}: {count}")

    print("\n--- Labels by Horizon ---")
    for period in HORIZONS:
        key = str(period)
        print(
            f"{period}: Complete={stats['complete_labels'][key]} | "
            f"Incomplete={stats['incomplete_labels'][key]} | "
            f"Invalid targets={stats['invalid_targets'][key]}"
        )

    print(f"\nOutput: {OUT_FILE}")
    print("Status: SUCCESS")
    print("=========================================")

if __name__ == "__main__":
    main()

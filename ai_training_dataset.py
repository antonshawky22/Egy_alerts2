import json
import math
from datetime import datetime

FEATURES_FILE = "ai_features_database.json"
LABELS_FILE = "ai_labels_database.json"
MARKET_FILE = "ai_market_regime_database.json"
OUT_FILE = "ai_training_dataset.json"

HORIZONS = (20, 40, 60)

# Stock features selected for model training
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
        "processed": 0,
        "missing_labels": 0,
        "incomplete_labels": 0,
        "missing_market": 0,
        "invalid_features": 0
    }

    for symbol, block in stock_symbols.items():
        if symbol == "EGX30":
            continue

        if symbol not in stock_labels:
            print(f"⚠️ Missing labels: {symbol}")
            stats["missing_labels"] += 1
            continue

        stock_data = block.get("data", {})
        symbol_labels = stock_labels[symbol]

        for date in sorted(stock_data):
            row = stock_data[date]
            labels_by_date = symbol_labels.get(date)

            if not labels_by_date:
                stats["missing_labels"] += 1
                continue

            market = market_data.get(date)

            if not market or market.get("regime") == "INSUFFICIENT_DATA":
                stats["missing_market"] += 1
                continue

            stock_features = {}

            for name in STOCK_FEATURES:
                value = safe_num(row.get(name))
                if value is None:
                    break
                stock_features[name] = value
            else:
                market_features = {
                    "Market_Score": safe_num(market.get("score")),
                    "Market_Volatility": safe_num(market.get("volatility")),
                    "Market_Regime": market.get("regime"),
                    "Market_Trend": market.get("trend"),
                    "Market_Momentum": market.get("momentum")
                }

                if any(
                    v is None
                    for k, v in market_features.items()
                    if k in ("Market_Score", "Market_Volatility")
                ):
                    stats["missing_market"] += 1
                    continue

                for period in HORIZONS:
                    label = labels_by_date.get(f"{period}_sessions", {})

                    if not label.get("complete"):
                        stats["incomplete_labels"] += 1
                        continue

                    targets = {
                        "Future_Return_Pct": safe_num(label.get("future_return_pct")),
                        "Max_Gain_Pct": safe_num(label.get("max_gain_pct")),
                        "Max_Drawdown_Pct": safe_num(label.get("max_drawdown_pct")),
                        "Sessions_To_Max": safe_num(label.get("sessions_to_max")),
                        "Sessions_To_Min": safe_num(label.get("sessions_to_min"))
                    }

                    if any(v is None for v in targets.values()):
                        stats["incomplete_labels"] += 1
                        continue

                    sample = {
                        "symbol": symbol,
                        "date": date,
                        "horizon": period,
                        **stock_features,
                        **market_features,
                        **targets
                    }

                    samples.append(sample)
                    stats["processed"] += 1

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
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
        "target_columns": [
            "Future_Return_Pct",
            "Max_Gain_Pct",
            "Max_Drawdown_Pct",
            "Sessions_To_Max",
            "Sessions_To_Min"
        ],
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
    print(f"Samples: {len(samples)}")
    print(f"Processed: {stats['processed']}")
    print(f"Incomplete labels: {stats['incomplete_labels']}")
    print(f"Missing market: {stats['missing_market']}")
    print(f"Invalid features: {stats['invalid_features']}")
    print(f"Missing labels: {stats['missing_labels']}")
    print(f"Output: {OUT_FILE}")
    print("Status: SUCCESS")
    print("=========================================")

if __name__ == "__main__":
    main()

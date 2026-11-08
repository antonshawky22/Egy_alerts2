import json
import math
from collections import defaultdict
from datetime import datetime, timezone

INPUT_FILE = "ai_training_dataset.json"
OUTPUT_FILE = "ai_dataset_analysis_report.json"

TARGET_HORIZON = 20
TARGET_THRESHOLD = 8.0
TARGET_COLUMN = "Target_Reach_8Pct"
MIN_STOCK_SAMPLES = 20

EXPECTED_TARGET_DEFINITION = {
    "type": "max_gain_anytime",
    "horizon": TARGET_HORIZON,
    "threshold_pct": TARGET_THRESHOLD,
    "target_column": TARGET_COLUMN,
    "condition": "Max_Gain_Pct > 8.0"
}


# =====================
# Safe Number Conversion
# =====================

def number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# =====================
# Statistics
# =====================

def stats(values):
    values = sorted(v for v in values if v is not None)

    if not values:
        return {}

    n = len(values)

    if n % 2 == 0:
        median = (
            values[n // 2 - 1] + values[n // 2]
        ) / 2
    else:
        median = values[n // 2]

    return {
        "count": n,
        "mean": round(sum(values) / n, 3),
        "median": round(median, 3),
        "minimum": round(values[0], 3),
        "maximum": round(values[-1], 3),
        "positive_pct": round(
            sum(v > 0 for v in values) / n * 100, 2
        ),
        "negative_pct": round(
            sum(v < 0 for v in values) / n * 100, 2
        ),
        "zero_pct": round(
            sum(v == 0 for v in values) / n * 100, 2
        )
    }


# =====================
# Target Statistics
# =====================

def target_stats(rows):
    if not rows:
        return {}

    positives = sum(
        int(s[TARGET_COLUMN]) == 1
        for s in rows
    )

    total = len(rows)

    return {
        "samples": total,
        "positive": positives,
        "negative": total - positives,
        "success_rate_pct": round(
            positives / total * 100, 2
        )
    }


# =====================
# Dataset Validation
# =====================

def validate_dataset(dataset):
    if not isinstance(dataset, dict):
        raise ValueError("Invalid dataset format.")

    samples = dataset.get("samples")

    if not isinstance(samples, list):
        raise ValueError(
            "Dataset samples are missing or invalid."
        )

    definition = dataset.get("target_definition", {})

    if definition != EXPECTED_TARGET_DEFINITION:
        raise ValueError(
            "Target definition mismatch between dataset and analysis."
        )

    valid_samples = []
    invalid_samples = 0

    for sample in samples:
        if not isinstance(sample, dict):
            invalid_samples += 1
            continue

        if number(sample.get("horizon")) != TARGET_HORIZON:
            continue

        max_gain = number(sample.get("Max_Gain_Pct"))
        target = number(sample.get(TARGET_COLUMN))

        if max_gain is None or target not in (0, 1):
            invalid_samples += 1
            continue

        expected_target = int(
            max_gain > TARGET_THRESHOLD
        )

        if int(target) != expected_target:
            raise ValueError(
                "Target mismatch detected for symbol "
                f"{sample.get('symbol', 'UNKNOWN')} "
                f"on date {sample.get('date', 'UNKNOWN')}."
            )

        valid_samples.append(sample)

    return valid_samples, invalid_samples


# =====================
# Group Analysis
# =====================

def analyze_groups(rows, group_field):
    groups = defaultdict(list)

    for sample in rows:
        group = sample.get(group_field) or "UNKNOWN"
        groups[str(group)].append(sample)

    results = {}

    for group, group_rows in sorted(groups.items()):
        result = target_stats(group_rows)

        returns = [
            number(s.get("Future_Return_Pct"))
            for s in group_rows
        ]

        gains = [
            number(s.get("Max_Gain_Pct"))
            for s in group_rows
        ]

        drawdowns = [
            number(s.get("Max_Drawdown_Pct"))
            for s in group_rows
        ]

        result["future_return"] = stats(returns)
        result["maximum_gain"] = stats(gains)
        result["maximum_drawdown"] = stats(drawdowns)

        results[group] = result

    return results


# =====================
# Stock Analysis
# =====================

def analyze_stocks(rows):
    groups = defaultdict(list)

    for sample in rows:
        symbol = str(sample.get("symbol", "UNKNOWN"))
        groups[symbol].append(sample)

    stock_results = []

    for symbol, stock_rows in groups.items():
        if len(stock_rows) < MIN_STOCK_SAMPLES:
            continue

        result = target_stats(stock_rows)

        result["symbol"] = symbol

        result["future_return"] = stats([
            number(s.get("Future_Return_Pct"))
            for s in stock_rows
        ])

        result["maximum_gain"] = stats([
            number(s.get("Max_Gain_Pct"))
            for s in stock_rows
        ])

        result["maximum_drawdown"] = stats([
            number(s.get("Max_Drawdown_Pct"))
            for s in stock_rows
        ])

        stock_results.append(result)

    stock_results.sort(
        key=lambda x: (
            x["success_rate_pct"],
            x["samples"]
        ),
        reverse=True
    )

    return {
        "minimum_samples_required": MIN_STOCK_SAMPLES,
        "eligible_stocks": len(stock_results),
        "best_10": stock_results[:10],
        "worst_10": stock_results[-10:][::-1]
    }


# =====================
# Main
# =====================

def main():
    print("-> جاري تحميل Dataset...")

    with open(INPUT_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    samples, invalid_samples = validate_dataset(dataset)

    if not samples:
        raise ValueError(
            "No valid samples found for the 20-session target."
        )

    overall_target = target_stats(samples)

    returns = [
        number(s.get("Future_Return_Pct"))
        for s in samples
    ]

    gains = [
        number(s.get("Max_Gain_Pct"))
        for s in samples
    ]

    drawdowns = [
        number(s.get("Max_Drawdown_Pct"))
        for s in samples
    ]

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "target_definition": EXPECTED_TARGET_DEFINITION,
        "total_dataset_samples": len(dataset["samples"]),
        "analyzed_samples": len(samples),
        "invalid_samples": invalid_samples,
        "target_distribution": overall_target,
        "horizons": {
            str(TARGET_HORIZON): {
                "samples": len(samples),
                "future_return": stats(returns),
                "maximum_gain": stats(gains),
                "maximum_drawdown": stats(drawdowns),
                "target": overall_target
            }
        },
        "market_analysis": {
            str(TARGET_HORIZON): analyze_groups(
                samples,
                "Market_Regime"
            )
        },
        "stock_analysis": {
            str(TARGET_HORIZON): analyze_stocks(samples)
        }
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        )

    print("\n========== AI DATASET ANALYSIS ==========")

    print(f"Total dataset samples: {len(dataset['samples'])}")
    print(f"Analyzed samples: {len(samples)}")
    print(f"Invalid samples: {invalid_samples}")

    print("\n--- Target Definition ---")
    print(f"Horizon: {TARGET_HORIZON} sessions")
    print(f"Required gain: More than {TARGET_THRESHOLD}%")
    print(f"Target column: {TARGET_COLUMN}")

    print("\n--- Target Distribution ---")
    print(f"Positive: {overall_target['positive']}")
    print(f"Negative: {overall_target['negative']}")
    print(
        f"Success rate: "
        f"{overall_target['success_rate_pct']}%"
    )

    print("\n--- Return Statistics ---")

    r = stats(returns)

    print(f"Average return: {r.get('mean', 'N/A')}%")
    print(f"Median return: {r.get('median', 'N/A')}%")
    print(f"Positive returns: {r.get('positive_pct', 'N/A')}%")
    print(f"Negative returns: {r.get('negative_pct', 'N/A')}%")

    print("\n--- Market Success Rates ---")

    for regime, result in report["market_analysis"][
        str(TARGET_HORIZON)
    ].items():
        print(
            f"{regime}: "
            f"{result['success_rate_pct']}% "
            f"({result['positive']}/{result['samples']})"
        )

    stock_report = report["stock_analysis"][
        str(TARGET_HORIZON)
    ]

    print("\n--- Best 10 Stocks ---")

    for stock in stock_report["best_10"]:
        print(
            f"{stock['symbol']}: "
            f"{stock['success_rate_pct']}% "
            f"({stock['positive']}/{stock['samples']})"
        )

    print("\n--- Worst 10 Stocks ---")

    for stock in stock_report["worst_10"]:
        print(
            f"{stock['symbol']}: "
            f"{stock['success_rate_pct']}% "
            f"({stock['positive']}/{stock['samples']})"
        )

    print(f"\nOutput: {OUTPUT_FILE}")
    print("Status: SUCCESS")
    print("=========================================")


if __name__ == "__main__":
    main()

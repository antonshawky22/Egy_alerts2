import json
import math
from collections import defaultdict
from datetime import datetime, timezone

INPUT_FILE = "ai_training_dataset.json"
OUTPUT_FILE = "ai_dataset_analysis_report.json"
HORIZONS = (20, 40, 60)

def number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {}
    n = len(values)
    return {
        "count": n,
        "mean": round(sum(values) / n, 3),
        "median": round(values[n // 2], 3),
        "minimum": round(values[0], 3),
        "maximum": round(values[-1], 3),
        "positive_pct": round(sum(v > 0 for v in values) / n * 100, 2),
        "negative_pct": round(sum(v < 0 for v in values) / n * 100, 2),
        "zero_pct": round(sum(v == 0 for v in values) / n * 100, 2)
    }

def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset["samples"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(samples),
        "horizons": {},
        "market_analysis": {},
        "stock_analysis": {}
    }

    for horizon in HORIZONS:
        rows = [s for s in samples if s.get("horizon") == horizon]

        returns = [number(s.get("Future_Return_Pct")) for s in rows]
        gains = [number(s.get("Max_Gain_Pct")) for s in rows]
        drawdowns = [number(s.get("Max_Drawdown_Pct")) for s in rows]

        report["horizons"][str(horizon)] = {
            "samples": len(rows),
            "future_return": stats(returns),
            "maximum_gain": stats(gains),
            "maximum_drawdown": stats(drawdowns)
        }

        market_groups = defaultdict(list)
        for s in rows:
            market_groups[s.get("Market_Regime", "UNKNOWN")].append(
                number(s.get("Future_Return_Pct"))
            )

        report["market_analysis"][str(horizon)] = {
            regime: stats(values)
            for regime, values in sorted(market_groups.items())
        }

        stock_groups = defaultdict(list)
        for s in rows:
            stock_groups[s.get("symbol", "UNKNOWN")].append(
                number(s.get("Future_Return_Pct"))
            )

        stock_results = []
        for symbol, values in stock_groups.items():
            result = stats(values)
            if result:
                stock_results.append({
                    "symbol": symbol,
                    **result
                })

        stock_results.sort(
            key=lambda x: x["mean"],
            reverse=True
        )

        report["stock_analysis"][str(horizon)] = {
            "best_10": stock_results[:10],
            "worst_10": list(reversed(stock_results[-10:]))
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n========== AI DATASET ANALYSIS ==========")
    print(f"Total samples: {len(samples)}")

    for horizon in HORIZONS:
        result = report["horizons"][str(horizon)]
        r = result.get("future_return", {})

        print(f"\n--- {horizon} Sessions ---")
        print(f"Samples: {result['samples']}")
        print(f"Average return: {r.get('mean', 'N/A')}%")
        print(f"Median return: {r.get('median', 'N/A')}%")
        print(f"Positive returns: {r.get('positive_pct', 'N/A')}%")
        print(f"Negative returns: {r.get('negative_pct', 'N/A')}%")

    print(f"\nOutput: {OUTPUT_FILE}")
    print("Status: SUCCESS")
    print("=========================================")

if __name__ == "__main__":
    main()

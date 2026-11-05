import json
import math
from datetime import datetime

DB_FILE = "egx_history_database_v2.json"
OUT_FILE = "ai_labels_database.json"
HORIZONS = (20, 40, 60)

def round_value(x):
    return round(x, 4) if x is not None and math.isfinite(x) else None

def calculate_labels(symbol, block):
    data = block.get("data", {})
    dates = sorted(data.keys())

    rows = [data[d] for d in dates]
    close = [float(x["Close"]) for x in rows]
    high = [float(x["High"]) for x in rows]
    low = [float(x["Low"]) for x in rows]

    output = {}

    for i, date in enumerate(dates):
        entry = close[i]
        result = {}

        for period in HORIZONS:
            key = f"{period}_sessions"
            end = i + period

            if end >= len(dates):
                result[key] = {
                    "complete": False,
                    "future_return_pct": None,
                    "max_gain_pct": None,
                    "max_drawdown_pct": None,
                    "sessions_to_max": None,
                    "sessions_to_min": None
                }
                continue

            future_high = high[i+1:end+1]
            future_low = low[i+1:end+1]
            future_close = close[end]

            max_high = max(future_high)
            min_low = min(future_low)

            max_index = future_high.index(max_high) + 1
            min_index = future_low.index(min_low) + 1

            result[key] = {
                "complete": True,
                "future_return_pct": round_value(
                    (future_close / entry - 1) * 100
                ),
                "max_gain_pct": round_value(
                    (max_high / entry - 1) * 100
                ),
                "max_drawdown_pct": round_value(
                    (min_low / entry - 1) * 100
                ),
                "sessions_to_max": max_index,
                "sessions_to_min": min_index
            }

        output[date] = result

    return output

def main():
    with open(DB_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "source": DB_FILE,
        "method": "Future outcomes measured after each historical candle",
        "horizons": list(HORIZONS),
        "total_symbols": len(db),
        "symbols": {},
        "summary": {}
    }

    total_rows = 0
    complete_rows = {p: 0 for p in HORIZONS}

    for symbol, block in db.items():
        labels = calculate_labels(symbol, block)
        report["symbols"][symbol] = labels
        total_rows += len(labels)

        for date, values in labels.items():
            for p in HORIZONS:
                if values[f"{p}_sessions"]["complete"]:
                    complete_rows[p] += 1

        print(f"✅ {symbol}: {len(labels)} dates processed")

    report["summary"] = {
        "total_historical_rows": total_rows,
        "complete_samples": complete_rows,
        "incomplete_samples": {
            str(p): total_rows - complete_rows[p]
            for p in HORIZONS
        }
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            report, f, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False
        )

    print("\n========== AI LABEL ENGINE ==========")
    print(f"Symbols: {len(db)}")
    print(f"Historical rows: {total_rows}")

    for p in HORIZONS:
        print(
            f"{p} sessions: "
            f"{complete_rows[p]} complete samples"
        )

    print(f"Output: {OUT_FILE}")
    print("Status: SUCCESS ✅")
    print("=====================================")

if __name__ == "__main__":
    main()

import json
import math
from datetime import datetime
from pathlib import Path

DB_FILE = "egx_history_database_v2.json"
REPORT_FILE = "ai_data_quality_report.json"
REQUIRED = ["Open", "High", "Low", "Close", "Volume"]
MAX_ERRORS = 30

duplicate_keys = []

def object_hook(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            duplicate_keys.append(key)
        result[key] = value
    return result

def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def audit():
    path = Path(DB_FILE)
    if not path.exists():
        print(f"❌ Database not found: {DB_FILE}")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f, object_pairs_hook=object_hook)
    except Exception as e:
        print(f"❌ JSON loading error: {e}")
        return

    report = {
        "database": DB_FILE,
        "audit_date": datetime.now().isoformat(timespec="seconds"),
        "total_symbols": 0,
        "total_candles": 0,
        "valid_candles": 0,
        "invalid_candles": 0,
        "symbols": {},
        "errors": [],
        "warnings": [],
        "duplicate_json_keys": duplicate_keys[:MAX_ERRORS]
    }

    if not isinstance(db, dict):
        print("❌ Invalid database root")
        return

    if duplicate_keys:
        report["errors"].append(
            f"Duplicate JSON keys detected: {len(duplicate_keys)}"
        )

    for symbol, block in db.items():
        report["total_symbols"] += 1
        errors = []
        warnings = []

        if not isinstance(block, dict):
            report["errors"].append(f"{symbol}: invalid symbol structure")
            continue

        columns = block.get("columns")
        data = block.get("data")

        if not isinstance(columns, list):
            errors.append("columns is not a list")
        elif not all(col in columns for col in REQUIRED):
            errors.append("missing required columns")

        if not isinstance(data, dict):
            errors.append("data is not a dictionary")
            data = {}

        dates = list(data.keys())
        valid_dates = []

        for date in dates:
            try:
                datetime.strptime(date, "%Y-%m-%d")
                valid_dates.append(date)
            except (ValueError, TypeError):
                errors.append(f"Invalid date: {date}")

        if len(valid_dates) > 1:
            if valid_dates != sorted(valid_dates, reverse=True):
                warnings.append("Dates are not ordered newest to oldest")

        symbol_valid = 0
        symbol_invalid = 0

        for date, candle in data.items():
            report["total_candles"] += 1

            if not isinstance(candle, dict):
                symbol_invalid += 1
                errors.append(f"{date}: invalid candle structure")
                continue

            missing = [x for x in REQUIRED if x not in candle]
            if missing:
                symbol_invalid += 1
                errors.append(f"{date}: missing {missing}")
                continue

            values = {x: candle[x] for x in REQUIRED}

            if not all(valid_number(v) for v in values.values()):
                symbol_invalid += 1
                errors.append(f"{date}: non-numeric or invalid values")
                continue

            o, h, l, c, v = (
                values["Open"], values["High"], values["Low"],
                values["Close"], values["Volume"]
            )

            if min(o, h, l, c) <= 0 or v < 0:
                symbol_invalid += 1
                errors.append(f"{date}: non-positive price or negative volume")
                continue

            if h < max(o, c, l) or l > min(o, c, h):
                symbol_invalid += 1
                errors.append(f"{date}: invalid OHLC relationship")
                continue

            symbol_valid += 1

        report["valid_candles"] += symbol_valid
        report["invalid_candles"] += symbol_invalid

        if len(valid_dates) >= 2:
            sorted_dates = sorted(valid_dates, reverse=True)
            for i in range(len(sorted_dates) - 1):
                d1 = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
                d2 = datetime.strptime(sorted_dates[i + 1], "%Y-%m-%d")
                if (d1 - d2).days > 5:
                    warnings.append(
                        f"Large date gap: {sorted_dates[i + 1]} to {sorted_dates[i]}"
                    )

        latest = max(valid_dates) if valid_dates else None

        if len(data) < 100:
            warnings.append("Less than 100 candles")

        report["symbols"][symbol] = {
            "candles": len(data),
            "valid_candles": symbol_valid,
            "invalid_candles": symbol_invalid,
            "latest_date": latest,
            "errors": errors[:MAX_ERRORS],
            "warnings": warnings[:MAX_ERRORS]
        }

        if errors:
            report["errors"].append(
                f"{symbol}: {len(errors)} data issues"
            )
        if warnings:
            report["warnings"].append(
                f"{symbol}: {len(warnings)} warnings"
            )

    report["data_integrity_pass"] = (
        report["invalid_candles"] == 0
        and len(report["errors"]) == 0
    )

    report["training_readiness"] = (
        "REQUIRES FEATURE AND LABEL VALIDATION"
    )

    counts = [
        x["candles"] for x in report["symbols"].values()
    ]

    report["candle_statistics"] = {
        "minimum": min(counts) if counts else 0,
        "maximum": max(counts) if counts else 0,
        "average": round(sum(counts) / len(counts), 1) if counts else 0
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n========== EGX AI DATA AUDIT ==========")
    print(f"Symbols: {report['total_symbols']}")
    print(f"Total candles: {report['total_candles']}")
    print(f"Valid candles: {report['valid_candles']}")
    print(f"Invalid candles: {report['invalid_candles']}")
    print(f"Minimum candles per symbol: {report['candle_statistics']['minimum']}")
    print(f"Maximum candles per symbol: {report['candle_statistics']['maximum']}")
    print(f"Average candles per symbol: {report['candle_statistics']['average']}")
    print(f"Data integrity: {'PASS ✅' if report['data_integrity_pass'] else 'CHECK ⚠️'}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")
    print(f"Report saved: {REPORT_FILE}")
    print("=======================================\n")

if __name__ == "__main__":
    audit()

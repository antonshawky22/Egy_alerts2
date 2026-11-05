import json
import math
from datetime import datetime

FEATURES_FILE = "ai_features_database.json"
OUT_FILE = "ai_market_regime_database.json"

def safe_num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def classify_market(row):
    close = safe_num(row.get("Close"))
    ema20 = safe_num(row.get("EMA20"))
    ema50 = safe_num(row.get("EMA50"))
    ema100 = safe_num(row.get("EMA100"))
    rsi = safe_num(row.get("RSI14"))
    macd_hist = safe_num(row.get("MACD_Hist"))
    return20 = safe_num(row.get("Return_20"))
    atr_pct = safe_num(row.get("ATR14_Pct"))

    required = [close, ema20, ema50, ema100, rsi, macd_hist, return20]
    if any(x is None for x in required):
        return {
            "regime": "INSUFFICIENT_DATA",
            "score": None,
            "trend": None,
            "momentum": None,
            "volatility": atr_pct
        }

    score = 0

    # Price and moving-average structure
    if close > ema20:
        score += 1
    elif close < ema20:
        score -= 1

    if ema20 > ema50:
        score += 1
    elif ema20 < ema50:
        score -= 1

    if ema50 > ema100:
        score += 1
    elif ema50 < ema100:
        score -= 1

    # Momentum
    if rsi >= 55:
        score += 1
    elif rsi <= 45:
        score -= 1

    if macd_hist > 0:
        score += 1
    elif macd_hist < 0:
        score -= 1

    if return20 > 2:
        score += 1
    elif return20 < -2:
        score -= 1

    if score >= 3:
        regime = "BULLISH"
    elif score <= -3:
        regime = "BEARISH"
    else:
        regime = "NEUTRAL"

    trend = (
        "UP" if close > ema20 and ema20 > ema50 and ema50 > ema100
        else "DOWN" if close < ema20 and ema20 < ema50 and ema50 < ema100
        else "MIXED"
    )

    momentum = (
        "POSITIVE" if rsi >= 55 and macd_hist > 0
        else "NEGATIVE" if rsi <= 45 and macd_hist < 0
        else "MIXED"
    )

    return {
        "regime": regime,
        "score": score,
        "trend": trend,
        "momentum": momentum,
        "volatility": atr_pct
    }

def main():
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        database = json.load(f)

    if "EGX30" not in database.get("symbols", {}):
        raise ValueError("EGX30 not found in features database")

    market_data = database["symbols"]["EGX30"].get("data", {})
    dates = sorted(market_data.keys())

    output = {}
    counts = {
        "BULLISH": 0,
        "NEUTRAL": 0,
        "BEARISH": 0,
        "INSUFFICIENT_DATA": 0
    }

    for date in dates:
        result = classify_market(market_data[date])
        output[date] = result
        counts[result["regime"]] += 1

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "source": FEATURES_FILE,
        "symbol": "EGX30",
        "method": "Historical market regime classification using same-day features only",
        "total_dates": len(dates),
        "counts": counts,
        "data": output
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    print("\n========== AI MARKET REGIME ENGINE ==========")
    print(f"Symbol: EGX30")
    print(f"Historical dates: {len(dates)}")
    for regime, count in counts.items():
        print(f"{regime}: {count}")
    print(f"Output: {OUT_FILE}")
    print("Status: SUCCESS")
    print("=============================================")

if __name__ == "__main__":
    main()

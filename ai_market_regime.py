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

def calculate_ema_series(closes, period):
    result = [None] * len(closes)

    if len(closes) < period:
        return result

    initial_values = closes[:period]

    if any(x is None for x in initial_values):
        return result

    ema = sum(initial_values) / period
    result[period - 1] = ema

    alpha = 2 / (period + 1)

    for i in range(period, len(closes)):
        price = closes[i]

        if price is None:
            continue

        ema = (price * alpha) + (ema * (1 - alpha))
        result[i] = ema

    return result

def calculate_return(closes, index, period):
    if index < period:
        return None

    current = closes[index]
    previous = closes[index - period]

    if current is None or previous is None or previous == 0:
        return None

    return ((current / previous) - 1) * 100

def classify_market(row, ema5, ema10, return5, return10):
    close = safe_num(row.get("Close"))
    ema20 = safe_num(row.get("EMA20"))
    rsi = safe_num(row.get("RSI14"))
    macd_hist = safe_num(row.get("MACD_Hist"))
    return20 = safe_num(row.get("Return_20"))
    atr_pct = safe_num(row.get("ATR14_Pct"))

    ema5 = safe_num(ema5)
    ema10 = safe_num(ema10)
    return5 = safe_num(return5)
    return10 = safe_num(return10)

    required = [
        close, ema5, ema10, ema20,
        rsi, macd_hist, return5, return10,
        return20
    ]

    if any(x is None for x in required):
        return {
            "regime": "INSUFFICIENT_DATA",
            "score": None,
            "trend": None,
            "momentum": None,
            "volatility": atr_pct,
            "ema5": ema5,
            "ema10": ema10,
            "return5": return5,
            "return10": return10
        }

    score = 0

    # 1. Short-term price position
    if close > ema5:
        score += 2
    elif close < ema5:
        score -= 2

    # 2. Fast moving-average direction
    if ema5 > ema10:
        score += 2
    elif ema5 < ema10:
        score -= 2

    # 3. Short-term trend confirmation
    if ema10 > ema20:
        score += 1
    elif ema10 < ema20:
        score -= 1

    # 4. Recent market performance
    if return5 > 1:
        score += 1
    elif return5 < -1:
        score -= 1

    if return10 > 2:
        score += 1
    elif return10 < -2:
        score -= 1

    # 5. Momentum
    if rsi >= 55:
        score += 1
    elif rsi <= 45:
        score -= 1

    if macd_hist > 0:
        score += 1
    elif macd_hist < 0:
        score -= 1

    # 6. Medium-term confirmation
    if return20 > 2:
        score += 1
    elif return20 < -2:
        score -= 1

    # Market regime
    if score >= 4:
        regime = "BULLISH"
    elif score <= -4:
        regime = "BEARISH"
    else:
        regime = "NEUTRAL"

    # Short-term trend
    if close > ema5 > ema10 > ema20:
        trend = "UP"
    elif close < ema5 < ema10 < ema20:
        trend = "DOWN"
    else:
        trend = "MIXED"

    # Short-term momentum
    if rsi >= 52 and macd_hist > 0 and return5 > 0:
        momentum = "POSITIVE"
    elif rsi <= 48 and macd_hist < 0 and return5 < 0:
        momentum = "NEGATIVE"
    else:
        momentum = "MIXED"

    return {
        "regime": regime,
        "score": score,
        "trend": trend,
        "momentum": momentum,
        "volatility": atr_pct,
        "ema5": round(ema5, 4),
        "ema10": round(ema10, 4),
        "return5": round(return5, 4),
        "return10": round(return10, 4)
    }

def main():
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        database = json.load(f)

    if "EGX30" not in database.get("symbols", {}):
        raise ValueError("EGX30 not found in features database")

    market_data = database["symbols"]["EGX30"].get("data", {})
    dates = sorted(market_data.keys())

    closes = [
        safe_num(market_data[date].get("Close"))
        for date in dates
    ]

    ema5_series = calculate_ema_series(closes, 5)
    ema10_series = calculate_ema_series(closes, 10)

    output = {}

    counts = {
        "BULLISH": 0,
        "NEUTRAL": 0,
        "BEARISH": 0,
        "INSUFFICIENT_DATA": 0
    }

    for i, date in enumerate(dates):
        return5 = calculate_return(closes, i, 5)
        return10 = calculate_return(closes, i, 10)

        result = classify_market(
            market_data[date],
            ema5_series[i],
            ema10_series[i],
            return5,
            return10
        )

        output[date] = result
        counts[result["regime"]] += 1

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "source": FEATURES_FILE,
        "symbol": "EGX30",
        "method": "Short-term market regime classification using EMA5, EMA10, EMA20, recent returns and momentum",
        "total_dates": len(dates),
        "counts": counts,
        "data": output
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False
        )

    print("\n========== AI SHORT-TERM MARKET REGIME ==========")
    print("Symbol: EGX30")
    print(f"Historical dates: {len(dates)}")

    for regime, count in counts.items():
        print(f"{regime}: {count}")

    print(f"Output: {OUT_FILE}")
    print("Status: SUCCESS")
    print("=================================================")

if __name__ == "__main__":
    main()

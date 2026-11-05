import json, math
from datetime import datetime

DB_FILE = "egx_history_database_v2.json"
OUT_FILE = "ai_features_database.json"

def ema(values, period):
    result = [None] * len(values)
    if not values:
        return result
    alpha = 2 / (period + 1)
    current = values[0]
    result[0] = current
    for i in range(1, len(values)):
        current = values[i] * alpha + current * (1 - alpha)
        result[i] = current
    return result

def rsi(values, period=14):
    result = [None] * len(values)
    if len(values) <= period:
        return result
    gains = [max(values[i] - values[i-1], 0) for i in range(1, len(values))]
    losses = [max(values[i-1] - values[i], 0) for i in range(1, len(values))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    def calc(g, l):
        if l == 0:
            return 100.0 if g > 0 else 50.0
        return 100 - 100 / (1 + g / l)
    result[period] = calc(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        result[i] = calc(avg_gain, avg_loss)
    return result

def atr(high, low, close, period=14):
    result = [None] * len(close)
    if len(close) < period:
        return result
    tr = []
    for i in range(len(close)):
        if i == 0:
            tr.append(high[i] - low[i])
        else:
            tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    current = sum(tr[:period]) / period
    result[period-1] = current
    for i in range(period, len(close)):
        current = (current * (period - 1) + tr[i]) / period
        result[i] = current
    return result

def rolling_level(values, period, mode):
    result = [None] * len(values)
    for i in range(period, len(values)):
        window = values[i-period:i]
        result[i] = max(window) if mode == "max" else min(window)
    return result

def safe_round(value):
    if value is None or not math.isfinite(value):
        return None
    return round(value, 6)

def calculate(symbol, block):
    data = block.get("data", {})
    dates = sorted(data.keys())
    rows = [data[d] for d in dates]

    required = ["Open", "High", "Low", "Close", "Volume"]
    for date, row in zip(dates, rows):
        if not isinstance(row, dict) or any(k not in row for k in required):
            raise ValueError(f"{symbol}: invalid candle at {date}")
        for key in required:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{symbol}: invalid {key} at {date}")

    op = [float(x["Open"]) for x in rows]
    hi = [float(x["High"]) for x in rows]
    lo = [float(x["Low"]) for x in rows]
    cl = [float(x["Close"]) for x in rows]
    vol = [float(x["Volume"]) for x in rows]

    e20 = ema(cl, 20)
    e50 = ema(cl, 50)
    e100 = ema(cl, 100)
    r14 = rsi(cl, 14)
    a14 = atr(hi, lo, cl, 14)

    macd_fast = ema(cl, 12)
    macd_slow = ema(cl, 26)
    macd = [macd_fast[i] - macd_slow[i] for i in range(len(cl))]
    signal = ema(macd, 9)
    hist = [macd[i] - signal[i] for i in range(len(cl))]

    levels = {}
    for p in (20, 50, 100):
        levels[f"High_{p}"] = rolling_level(hi, p, "max")
        levels[f"Low_{p}"] = rolling_level(lo, p, "min")

    columns = [
        "Open", "High", "Low", "Close", "Volume",
        "EMA20", "EMA50", "EMA100",
        "RSI14", "MACD", "MACD_Signal", "MACD_Hist",
        "ATR14", "ATR14_Pct",
        "Volume_MA20", "Volume_Ratio20",
        "Return_5", "Return_10", "Return_20", "Return_50", "Return_100",
        "High_20", "Low_20", "High_50", "Low_50", "High_100", "Low_100",
        "Distance_EMA20_Pct", "Distance_EMA50_Pct", "Distance_EMA100_Pct",
        "Distance_Resistance20_Pct", "Distance_Support20_Pct",
        "Trend_20_50", "Trend_50_100"
    ]

    output = {}

    for i, date in enumerate(dates):
        c = cl[i]
        volume_avg = sum(vol[i-20:i]) / 20 if i >= 20 else None

        def ret(period):
            if i < period or cl[i-period] == 0:
                return None
            return (c / cl[i-period] - 1) * 100

        def distance(level):
            if level is None or level == 0:
                return None
            return (c / level - 1) * 100

        row = {
            "Open": op[i], "High": hi[i], "Low": lo[i],
            "Close": c, "Volume": vol[i],
            "EMA20": safe_round(e20[i]),
            "EMA50": safe_round(e50[i]),
            "EMA100": safe_round(e100[i]),
            "RSI14": safe_round(r14[i]),
            "MACD": safe_round(macd[i]),
            "MACD_Signal": safe_round(signal[i]),
            "MACD_Hist": safe_round(hist[i]),
            "ATR14": safe_round(a14[i]),
            "ATR14_Pct": safe_round(a14[i] / c * 100) if a14[i] is not None and c else None,
            "Volume_MA20": safe_round(volume_avg),
            "Volume_Ratio20": safe_round(vol[i] / volume_avg) if volume_avg else None,
            "Return_5": safe_round(ret(5)),
            "Return_10": safe_round(ret(10)),
            "Return_20": safe_round(ret(20)),
            "Return_50": safe_round(ret(50)),
            "Return_100": safe_round(ret(100)),
            "High_20": safe_round(levels["High_20"][i]),
            "Low_20": safe_round(levels["Low_20"][i]),
            "High_50": safe_round(levels["High_50"][i]),
            "Low_50": safe_round(levels["Low_50"][i]),
            "High_100": safe_round(levels["High_100"][i]),
            "Low_100": safe_round(levels["Low_100"][i]),
            "Distance_EMA20_Pct": safe_round(distance(e20[i])),
            "Distance_EMA50_Pct": safe_round(distance(e50[i])),
            "Distance_EMA100_Pct": safe_round(distance(e100[i])),
            "Distance_Resistance20_Pct": safe_round(distance(levels["High_20"][i])),
            "Distance_Support20_Pct": safe_round(distance(levels["Low_20"][i])),
            "Trend_20_50": int(e20[i] > e50[i]),
            "Trend_50_100": int(e50[i] > e100[i])
        }

        output[date] = row

    return {"columns": columns, "data": output}

def main():
    with open(DB_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)

    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "source": DB_FILE,
        "total_symbols": len(db),
        "symbols": {}
    }

    for symbol, block in db.items():
        result["symbols"][symbol] = calculate(symbol, block)
        print(f"✅ {symbol}: {len(result['symbols'][symbol]['data'])} candles processed")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    print("\n========== AI FEATURE ENGINE ==========")
    print(f"Symbols: {result['total_symbols']}")
    print(f"Features per candle: {len(next(iter(result['symbols'].values()))['columns'])}")
    print(f"Output: {OUT_FILE}")
    print("Status: SUCCESS ✅")
    print("=======================================")

if __name__ == "__main__":
    main()

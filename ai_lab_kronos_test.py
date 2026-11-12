import json
import os
import sys
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd

DB_FILE = "egx_history_database_v2.json"
OUTPUT_FILE = "ai_lab_kronos_results.csv"
MODEL_ID = "Ahmed-elbarbary/kronos-small-egx"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK = 90
PRED_LEN = 20
MAX_SYMBOLS = 34
SAMPLE_COUNT = 1

def load_database():
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Missing {DB_FILE}")
    with open(DB_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
    if not isinstance(db, dict) or not db:
        raise ValueError("Database is empty or has an invalid format.")
    return db

def prepare_symbol(symbol, block):
    data = block.get("data", {}) if isinstance(block, dict) else {}
    if not isinstance(data, dict):
        raise ValueError(f"{symbol}: data must be a date-keyed object.")
    dates = sorted(data.keys())
    rows = []
    valid_dates = []
    required = ("Open", "High", "Low", "Close")
    for date in dates:
        row = data[date]
        if not isinstance(row, dict) or any(k not in row for k in required):
            continue
        try:
            values = {k: float(row[k]) for k in required}
            values["volume"] = float(row.get("Volume", 0) or 0)
            if not all(np.isfinite(v) for v in values.values()):
                continue
            if min(values[k] for k in required) <= 0:
                continue
            rows.append(values)
            valid_dates.append(pd.to_datetime(date))
        except (TypeError, ValueError, OverflowError):
            continue
    if len(rows) < LOOKBACK:
        raise ValueError(f"{symbol}: only {len(rows)} valid candles; need {LOOKBACK}.")
    df = pd.DataFrame(rows).tail(LOOKBACK).reset_index(drop=True)
    timestamps = pd.Series(pd.to_datetime(valid_dates[-LOOKBACK:]))
    df.columns = [str(c).lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    return df, timestamps

def future_egx_dates(last_date, periods):
    # Placeholder session dates: EGX normally trades Sunday through Thursday.
    # Official exchange holidays are not included; dates are forecast indexes only.
    dates = []
    day = pd.Timestamp(last_date).normalize()
    while len(dates) < periods:
        day += pd.Timedelta(days=1)
        if day.weekday() in (6, 0, 1, 2, 3):  # Sunday-Thursday
            dates.append(day)
    return pd.Series(dates)

def main():
    print("EGX KRONOS LAB - INFERENCE SMOKE TEST")
    print(f"Model: {MODEL_ID}")
    print(f"Input: last {LOOKBACK} daily candles | Horizon: {PRED_LEN} sessions")
    db = load_database()

    from model import Kronos, KronosTokenizer, KronosPredictor

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading tokenizer and fine-tuned EGX model...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    candidates = []
    for symbol, block in db.items():
        if symbol.upper() == "EGX30":
            continue
        try:
            df, x_ts = prepare_symbol(symbol, block)
            candidates.append((symbol, df, x_ts))
        except Exception as exc:
            print(f"SKIP {symbol}: {exc}")

    if not candidates:
        raise RuntimeError("No eligible stocks found in the price database.")

    candidates = candidates[:MAX_SYMBOLS]
    results = []
    for symbol, df, x_ts in candidates:
        try:
            y_ts = future_egx_dates(x_ts.iloc[-1], PRED_LEN)
            forecast = predictor.predict(
                df=df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=PRED_LEN,
                T=1.0,
                top_p=0.9,
                sample_count=SAMPLE_COUNT
            )
            if not isinstance(forecast, pd.DataFrame) or "close" not in forecast.columns:
                raise ValueError("Kronos output has no close column.")
            predicted_close = pd.to_numeric(forecast["close"], errors="coerce").dropna()
            if len(predicted_close) != PRED_LEN:
                raise ValueError(f"Expected {PRED_LEN} predicted closes, got {len(predicted_close)}.")
            last_close = float(df["close"].iloc[-1])
            final_close = float(predicted_close.iloc[-1])
            max_forecast_high = float(pd.to_numeric(forecast["high"], errors="coerce").max()) if "high" in forecast else np.nan
            min_forecast_low = float(pd.to_numeric(forecast["low"], errors="coerce").min()) if "low" in forecast else np.nan
            results.append({
                "symbol": symbol,
                "last_date": x_ts.iloc[-1].strftime("%Y-%m-%d"),
                "last_close": last_close,
                "forecast_close_20": final_close,
                "forecast_return_20_pct": (final_close / last_close - 1) * 100,
                "forecast_max_high_pct": (max_forecast_high / last_close - 1) * 100 if np.isfinite(max_forecast_high) else np.nan,
                "forecast_min_low_pct": (min_forecast_low / last_close - 1) * 100 if np.isfinite(min_forecast_low) else np.nan,
                "status": "OK"
            })
            print(f"OK {symbol}: close {last_close:.4f} -> forecast {final_close:.4f}")
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")
            results.append({"symbol": symbol, "status": f"ERROR: {exc}"})

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    success = sum(1 for row in results if row.get("status") == "OK")
    print(f"Finished: {success}/{len(results)} symbols succeeded.")
    print(f"Output: {OUTPUT_FILE}")
    if success == 0:
        raise RuntimeError("Kronos inference failed for all selected stocks.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

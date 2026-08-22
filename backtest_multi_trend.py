import json
import os
import pandas as pd
import numpy as np

# ============================================================
# HIGH-PRECISION TREND CROSSOVER SYSTEM (EMA20/50 + VOL + RSI)
# ============================================================

DB_FILE = "egx_history_database_v2.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

symbols = {
    "OLFI": "OLFI", "EMFD": "EMFD", "ETEL": "ETEL", "EAST": "EAST",
    "EFIH": "EFIH", "ABUK": "ABUK", "OIH": "OIH", "SWDY": "SWDY", "ISPH": "ISPH",
    "ATQA": "ATQA", "MTIE": "MTIE", "HRHO": "HRHO", "ORWE": "ORWE",
    "JUFO": "JUFO", "DSCW": "DSCW", "SUGR": "SUGR", "ELSH": "ELSH", "RMDA": "RMDA",
    "RAYA": "RAYA", "EEII": "EEII", "MPCO": "MPCO", "GBCO": "GBCO", "TMGH": "TMGH",
    "ORHD": "ORHD", "AMOC": "AMOC", "FWRY": "FWRY", "COMI": "COMI", "ADIB": "ADIB",
    "PHDC": "PHDC", "MCQE": "MCQE", "SKPC": "SKPC", "EGAL": "EGAL"
}

def rsi(series, period=14):
    if len(series) < period + 1:
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ============================================================
# LOAD DATABASE & PREPARE DATA
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(f"Database file not found: {DB_FILE}")

with open(DB_FILE, "r", encoding="utf-8") as f:
    raw_database = json.load(f)

prepared_data = {}

for name in symbols:
    if name not in raw_database:
        continue
    content = raw_database[name]
    if "data" not in content or "columns" not in content:
        continue
    try:
        df = pd.DataFrame.from_dict(content["data"], orient="index", columns=content["columns"])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)

        if len(df) < 50:
            continue

        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)

        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()
        df["RSI"] = rsi(close)
        df["VOL_MA20"] = volume.rolling(window=20).mean()

        prepared_data[name] = df
    except Exception:
        pass

all_dates = sorted(set().union(*[df.index for df in prepared_data.values()]))

# ============================================================
# STATE TRACKING & BACKTEST LOOP
# ============================================================

states = {name: {"position": 0.0, "entry_price": 0.0, "peak_price": 0.0} for name in prepared_data}
trades_history = []
total_portfolio_profit = 0.0
portfolio_equity_curve = []

for current_date in all_dates:
    for name, df in prepared_data.items():
        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)
        if idx < 50:
            continue

        df_slice = df.iloc[: idx + 1]
        
        if df_slice[["Close", "EMA20", "EMA50", "RSI"]].iloc[-1].isna().any():
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        price = float(df_slice["Close"].iloc[-1])
        rsi_val = float(df_slice["RSI"].iloc[-1])
        volume_curr = float(df_slice["Volume"].iloc[-1]) if "Volume" in df_slice.columns else 1.0
        vol_ma = float(df_slice["VOL_MA20"].iloc[-1]) if "VOL_MA20" in df_slice.columns else 1.0
        
        ema20_curr = float(df_slice["EMA20"].iloc[-1])
        ema50_curr = float(df_slice["EMA50"].iloc[-1])
        
        ema20_prev = float(df_slice["EMA20"].iloc[-2])
        ema50_prev = float(df_slice["EMA50"].iloc[-2])

        # 🎯 شروط فلترة صارمة جداً لتقليل الصفقات
        golden_cross = (ema20_prev <= ema50_prev) and (ema20_curr > ema50_curr)
        death_cross = (ema20_prev >= ema50_prev) and (ema20_curr < ema50_curr)
        
        not_overextended = price <= (ema50_curr * 1.05)       # السعر غير مبتعد عن المتوسط بأكثر من 5%
        strict_rsi_buy = 38.0 <= rsi_val <= 58.0              # شراء فقط في بداية الزخم
        high_volume = volume_curr >= (vol_ma * 1.3)          # شرط سيولة قوي (أعلى بـ 30%)

        s = states[name]

        profit = 0.0
        if s["entry_price"] > 0:
            profit = ((price - s["entry_price"]) / s["entry_price"]) * 100
            if price > s["peak_price"]:
                s["peak_price"] = price

        # 🟢 دخول بحجم صفقات أقل وبأعلى دقة
        if s["position"] == 0.0 and golden_cross and not_overextended and strict_rsi_buy and high_volume:
            s["position"] = 1.0
            s["entry_price"] = price
            s["peak_price"] = price
            
            trades_history.append({
                "symbol": name,
                "status": "OPEN",
                "entry_date": date_str,
                "entry_price": round(price, 2),
                "exit_date": None,
                "exit_price": None,
                "profit_pct": None
            })

        # 🔴 خروج احترافي (الاستفادة من أقصى ربح + حماية من الارتداد)
        elif s["position"] == 1.0:
            peak_profit = ((s["peak_price"] - s["entry_price"]) / s["entry_price"]) * 100
            
            hard_stop = profit <= -4.5
            trailing_stop = (peak_profit >= 10.0) and ((s["peak_price"] - price) / s["peak_price"] * 100 >= 4.0)
            rsi_target = (rsi_val >= 75.0) and (profit >= 6.0)
            
            exit_triggered = death_cross or hard_stop or trailing_stop or rsi_target

            if exit_triggered:
                active = [t for t in trades_history if t["symbol"] == name and t["status"] == "OPEN"]
                if active:
                    active[-1]["status"] = "CLOSED"
                    active[-1]["exit_date"] = date_str
                    active[-1]["exit_price"] = round(price, 2)
                    active[-1]["profit_pct"] = round(profit, 2)

                total_portfolio_profit += profit
                s["position"] = 0.0
                s["entry_price"] = 0.0
                s["peak_price"] = 0.0

    portfolio_equity_curve.append(total_portfolio_profit)

# ============================================================
# COMPUTE STATISTICS & SAVE TO FILES
# ============================================================

closed_trades = [t for t in trades_history if t["status"] == "CLOSED"]
winning_trades = [t for t in closed_trades if t["profit_pct"] is not None and t["profit_pct"] > 0]
losing_trades = [t for t in closed_trades if t["profit_pct"] is not None and t["profit_pct"] <= 0]

total_count = len(closed_trades)
wins_count = len(winning_trades)
losses_count = len(losing_trades)
win_rate = (wins_count / total_count) * 100 if total_count > 0 else 0.0

avg_win = float(np.mean([t["profit_pct"] for t in winning_trades])) if winning_trades else 0.0
avg_loss = float(np.mean([t["profit_pct"] for t in losing_trades])) if losing_trades else 0.0

equity = np.array(portfolio_equity_curve) + 100.0
peak = np.maximum.accumulate(equity)
drawdown = (peak - equity) / peak * 100.0
max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

results_summary = {
    "statistics": {
        "total_trades": total_count,
        "winning_trades": wins_count,
        "losing_trades": losses_count,
        "win_rate_percent": round(win_rate, 2),
        "total_profit_percent": round(total_portfolio_profit, 2),
        "average_winning_trade_percent": round(avg_win, 2),
        "average_losing_trade_percent": round(avg_loss, 2),
        "maximum_drawdown_percent": round(max_drawdown, 2)
    }
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

with open(TRADES_FILE, "w", encoding="utf-8") as f:
    json.dump(trades_history, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("HIGH-PRECISION CROSSOVER SYSTEM COMPLETE")
print("=" * 60)
print(f"Total Closed Trades: {total_count}")
print(f"Win Rate:            {win_rate:.2f}%")
print(f"Total Profit:        {total_portfolio_profit:.2f}%")
print(f"Max Drawdown:        {max_drawdown:.2f}%")
print("=" * 60)

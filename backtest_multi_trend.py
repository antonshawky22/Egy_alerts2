import json
import os
import pandas as pd
import numpy as np

# ============================================================
# WEEKLY TREND FOLLOWING SYSTEM (EMA10/20 + ANTI-SIDEWAYS FILTER)
# ============================================================

DB_FILE = "egx_weekly_database_v1.json"
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

def calculate_adx(df, period=14):
    """حساب مؤشر ADX لقياس قوة الاتجاه وتجنب النطاق العرضي"""
    high = df["High"] if "High" in df.columns else df["Close"]
    low = df["Low"] if "Low" in df.columns else df["Close"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

# ============================================================
# LOAD DATABASE & PREPARE WEEKLY DATA
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

        if len(df) < 25:
            continue

        close = df["Close"]

        # حساب المتوسطات ومؤشر ADX للفريم الأسبوعي
        df["EMA10"] = close.ewm(span=10, adjust=False).mean()
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        df["ADX"] = calculate_adx(df, period=14)

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
        if idx < 20:
            continue

        df_slice = df.iloc[: idx + 1]
        
        if df_slice[["Close", "EMA10", "EMA20", "ADX"]].iloc[-1].isna().any():
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        price = float(df_slice["Close"].iloc[-1])
        adx_val = float(df_slice["ADX"].iloc[-1])
        
        ema10_curr = float(df_slice["EMA10"].iloc[-1])
        ema20_curr = float(df_slice["EMA20"].iloc[-1])
        
        ema10_prev = float(df_slice["EMA10"].iloc[-2])
        ema20_prev = float(df_slice["EMA20"].iloc[-2])

        # 🎯 إشارات التقاط الفريم الأسبوعي
        golden_cross = (ema10_prev <= ema20_prev) and (ema10_curr > ema20_curr)
        death_cross = (ema10_prev >= ema20_prev) and (ema10_curr < ema20_curr)
        
        # 🚫 شرط منع الاتجاه العرضي: السهم ليس في اتجاه عرضي إذا كان ADX >= 20 ومتوسط EMA10 أسرع صعوداً
        trending_market = adx_val >= 20.0

        s = states[name]

        profit = 0.0
        if s["entry_price"] > 0:
            profit = ((price - s["entry_price"]) / s["entry_price"]) * 100
            if price > s["peak_price"]:
                s["peak_price"] = price

        # 🟢 دخول أسبوعي (تقاطع 10/20 + اتجاه غير عرضي)
        if s["position"] == 0.0 and golden_cross and trending_market:
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

        # 🔴 خروج عند تقاطع الموت الأسبوعي أو وقف الخسارة
        elif s["position"] == 1.0:
            peak_profit = ((s["peak_price"] - s["entry_price"]) / s["entry_price"]) * 100
            
            # وقف خسارة أسبوعي أوسع لحماية الصفقة من التذبذب السعري
            hard_stop = profit <= -6.0
            trailing_stop = (peak_profit >= 12.0) and ((s["peak_price"] - price) / s["peak_price"] * 100 >= 5.0)
            
            exit_triggered = death_cross or hard_stop or trailing_stop

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
print("WEEKLY EMA 10/20 BACKTEST COMPLETE")
print("=" * 60)
print(f"Total Closed Trades: {total_count}")
print(f"Win Rate:            {win_rate:.2f}%")
print(f"Total Profit:        {total_portfolio_profit:.2f}%")
print(f"Max Drawdown:        {max_drawdown:.2f}%")
print("=" * 60)

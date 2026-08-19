import json
import os
import pandas as pd
import numpy as np

# ============================================================
# BACKTEST - MULTI TREND EMA70 (V8 HIGH-CONVICTION SYSTEM)
# ============================================================

DB_FILE = "egx_history_database_v2.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"
STOCK_SUMMARY_FILE = "backtest_summary_by_stock.json"

EGX30_KEY = "EGX30"

# ============================================================
# STRATEGY PARAMETERS (V8 OPTIMIZED)
# ============================================================

# EMA70 TREND
EMA70_UP_MIN_STEP_PERCENT = 0.30
EMA70_SIDE_MAX_DISTANCE_PERCENT = 1.00
EMA70_DOWN_MIN_STEP_PERCENT = 1.00

# RSI THRESHOLDS (V8 HIGH-CONVICTION)
RSI_UP_BUY = 55            # دخول مع الزخم القوي فقط
RSI_SIDE_BUY_2 = 42        # شريحة تجميع ثانية عند التصحيح
RSI_SIDE_BUY_3 = 20        # شريحة أعمق إن وجدت

RSI_PARTIAL_SELL = 72      # رفع هدف البيع الجزئي لتكبير الأرباح
RSI_FINAL_SELL = 78        # هدف البيع الكلي

# STOP LOSS
EMA70_STOP_MIN_DROP_PERCENT = 1.00

# POSITION SIZING (2 TRANCHES SYSTEM - 50% EACH)
TRANCHE_SIZE = 0.50        # دخول بـ 50% لكل شريحة لتعظيم التراكمي
MAX_TRANCHES = 2           # شريحتين كحد أقصى لتكثيف السيولة

# ============================================================
# RSI - WILDER
# ============================================================

def rsi(series, period=14):
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(f"Database file not found: {DB_FILE}")

with open(DB_FILE, "r", encoding="utf-8") as f:
    raw_database = json.load(f)

print("Historical database loaded.")

symbols = list(raw_database.keys())

if EGX30_KEY not in raw_database:
    raise ValueError("EGX30 not found in database.")

# ============================================================
# PREPARE DATA
# ============================================================

prepared_data = {}

for symbol in symbols:
    content = raw_database.get(symbol, {})
    if "data" not in content or "columns" not in content:
        continue
    try:
        df = pd.DataFrame.from_dict(content["data"], orient="index", columns=content["columns"])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if len(df) < 80:
            continue

        df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA70"] = df["Close"].ewm(span=70, adjust=False).mean()
        df["RSI14"] = rsi(df["Close"], 14)

        df["cross_up"] = (df["EMA12"] > df["EMA20"]) & (df["EMA12"].shift(1) <= df["EMA20"].shift(1))
        df["cross_down"] = (df["EMA12"] < df["EMA20"]) & (df["EMA12"].shift(1) >= df["EMA20"].shift(1))

        prepared_data[symbol] = df
    except Exception as e:
        print(f"Failed preparing {symbol}: {e}")

print(f"Prepared {len(prepared_data)} symbols.")

# ============================================================
# EMA70 TREND DETECTION
# ============================================================

def calculate_trend(df, index):
    if index < 12:
        return "🔛"

    ema70_now = float(df.iloc[index]["EMA70"])
    ema70_4 = float(df.iloc[index - 4]["EMA70"])
    ema70_8 = float(df.iloc[index - 8]["EMA70"])
    ema70_12 = float(df.iloc[index - 12]["EMA70"])

    if any(pd.isna(x) for x in [ema70_now, ema70_4, ema70_8, ema70_12]):
        return "🔛"

    step_1 = ((ema70_now - ema70_4) / ema70_4) * 100
    step_2 = ((ema70_4 - ema70_8) / ema70_8) * 100
    step_3 = ((ema70_8 - ema70_12) / ema70_12) * 100

    if step_1 >= EMA70_UP_MIN_STEP_PERCENT and step_2 >= EMA70_UP_MIN_STEP_PERCENT and step_3 >= EMA70_UP_MIN_STEP_PERCENT:
        return "↗️"

    if step_1 <= -EMA70_DOWN_MIN_STEP_PERCENT and step_2 <= -EMA70_DOWN_MIN_STEP_PERCENT and step_3 <= -EMA70_DOWN_MIN_STEP_PERCENT:
        return "🔻"

    max_ema70 = max(ema70_now, ema70_4, ema70_8, ema70_12)
    min_ema70 = min(ema70_now, ema70_4, ema70_8, ema70_12)

    if min_ema70 <= 0:
        return "🔛"

    distance_percent = ((max_ema70 - min_ema70) / min_ema70) * 100

    if distance_percent <= EMA70_SIDE_MAX_DISTANCE_PERCENT:
        return "🔛"

    return "🔛"

# ============================================================
# EMA70 SHARP DECLINE DETECTION
# ============================================================

def ema70_sharp_decline(df, index):
    if index < 4:
        return False

    ema70_now = float(df.iloc[index]["EMA70"])
    ema70_4 = float(df.iloc[index - 4]["EMA70"])

    if pd.isna(ema70_now) or pd.isna(ema70_4) or ema70_4 <= 0:
        return False

    decline_percent = ((ema70_now - ema70_4) / ema70_4) * 100
    return decline_percent <= -EMA70_STOP_MIN_DROP_PERCENT

# ============================================================
# ALL TRADING DATES
# ============================================================

all_dates = set()
for symbol, df in prepared_data.items():
    if symbol == EGX30_KEY:
        continue
    all_dates.update(df.index)

all_dates = sorted(all_dates)

if not all_dates:
    raise ValueError("No trading dates available.")

print(f"Backtest period: {all_dates[0].strftime('%Y-%m-%d')} -> {all_dates[-1].strftime('%Y-%m-%d')}")

# ============================================================
# STATE & TRACKING
# ============================================================

states = {}
for symbol in prepared_data:
    if symbol == EGX30_KEY:
        continue
    states[symbol] = {
        "in_position": False,
        "tranches": [],
        "entry_date": None,
        "entry_trend": None,
        "partial_sell_done": False
    }

trades = []
closed_profit_percent = 0.0
equity_curve = []
peak_equity = 0.0
max_drawdown = 0.0

trend_days = {"↗️": 0, "🔛": 0, "🔻": 0}

# ============================================================
# TRANCHE HELPERS
# ============================================================

def position_units(state):
    return sum(tranche["size"] for tranche in state["tranches"])

def average_entry_price(state):
    total_size = position_units(state)
    if total_size <= 0:
        return None
    weighted_value = sum(tranche["size"] * tranche["price"] for tranche in state["tranches"])
    return weighted_value / total_size

def add_tranche(state, price, date):
    state["tranches"].append({
        "size": TRANCHE_SIZE,
        "price": float(price),
        "date": date.strftime("%Y-%m-%d")
    })
    state["in_position"] = True

def sell_one_tranche(state, close, current_date, reason):
    if not state["tranches"]:
        return 0.0, None

    tranche = state["tranches"].pop(0)
    entry_price = tranche["price"]
    size = tranche["size"]

    profit_percent = ((close - entry_price) / entry_price) * 100
    contribution = profit_percent * size

    sale = {
        "entry_date": tranche["date"],
        "entry_price": round(entry_price, 4),
        "exit_date": current_date.strftime("%Y-%m-%d"),
        "exit_price": round(close, 4),
        "size": round(size, 4),
        "profit_pct_on_tranche": round(profit_percent, 2),
        "portfolio_contribution_pct": round(contribution, 2),
        "reason": reason
    }

    if not state["tranches"]:
        state["in_position"] = False

    return contribution, sale

def sell_all(state, close, current_date, reason):
    total_contribution = 0.0
    sales = []

    while state["tranches"]:
        contribution, sale = sell_one_tranche(state, close, current_date, reason)
        total_contribution += contribution
        if sale:
            sales.append(sale)

    state["in_position"] = False
    return total_contribution, sales

# ============================================================
# MAIN BACKTEST LOOP
# ============================================================

for current_date in all_dates:
    for symbol, df in prepared_data.items():
        if symbol == EGX30_KEY or current_date not in df.index:
            continue

        current_index = df.index.get_loc(current_date)
        if current_index < 80:
            continue

        row = df.iloc[current_index]
        close = float(row["Close"])

        if pd.isna(close) or pd.isna(row["RSI14"]):
            continue

        state = states[symbol]
        units = position_units(state)
        trend = calculate_trend(df, current_index)
        trend_days[trend] += 1

        buy_signal = False
        buy_level = None
        realized_profit = 0.0
        sales = []

        # --- UPTREND LOGIC ---
        if trend == "↗️":
            if units == 0 and row["RSI14"] < RSI_UP_BUY:
                buy_signal = True
                buy_level = 1

            elif units > 0 and row["RSI14"] > RSI_PARTIAL_SELL:
                realized_profit, sale = sell_one_tranche(
                    state, close, current_date, f"RSI_{RSI_PARTIAL_SELL}"
                )
                if sale:
                    sales.append(sale)

        # --- SIDEWAYS LOGIC ---
        elif trend == "🔛":
            # Second Tranche Buy (Max 2 tranches in V8)
            if TRANCHE_SIZE - 0.0001 <= units < (2 * TRANCHE_SIZE - 0.0001) and row["RSI14"] <= RSI_SIDE_BUY_2:
                buy_signal = True
                buy_level = 2

            elif units > 0 and row["RSI14"] > RSI_PARTIAL_SELL:
                realized_profit, sale = sell_one_tranche(
                    state, close, current_date, f"SIDE_RSI_{RSI_PARTIAL_SELL}"
                )
                if sale:
                    sales.append(sale)

        # --- DOWNTREND LOGIC ---
        elif trend == "🔻":
            if units > 0:
                realized_profit, sales = sell_all(state, close, current_date, "EMA70_STRONG_DOWN")

        # --- FINAL RSI TARGET EXIT ---
        if state["in_position"] and row["RSI14"] > RSI_FINAL_SELL and trend != "🔻":
            final_profit, final_sales = sell_all(state, close, current_date, f"RSI_{RSI_FINAL_SELL}")
            realized_profit += final_profit
            sales.extend(final_sales)

        # --- STOP LOSS EXIT ---
        if state["in_position"] and ema70_sharp_decline(df, current_index):
            stop_profit, stop_sales = sell_all(state, close, current_date, "EMA70_SHARP_DECLINE")
            realized_profit += stop_profit
            sales.extend(stop_sales)

        # --- EXECUTE BUYS ---
        if buy_signal and not state["in_position"] and buy_level == 1:
            add_tranche(state, close, current_date)
            state["entry_date"] = current_date.strftime("%Y-%m-%d")
            state["entry_trend"] = trend

        elif buy_signal and state["in_position"] and buy_level in [2, 3] and len(state["tranches"]) < MAX_TRANCHES:
            add_tranche(state, close, current_date)

        # --- RECORD TRADES ---
        if realized_profit != 0.0:
            closed_profit_percent += realized_profit

            if not state["in_position"]:
                total_profit = sum(sale["portfolio_contribution_pct"] for sale in sales)
                trade = {
                    "symbol": symbol,
                    "entry_date": state["entry_date"],
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "entry_trend": state["entry_trend"],
                    "exit_trend": trend,
                    "profit_pct": round(total_profit, 2),
                    "exit_reason": sales[-1]["reason"] if sales else "UNKNOWN",
                    "sales": sales
                }
                trades.append(trade)
                state["entry_date"] = None
                state["entry_trend"] = None
                state["partial_sell_done"] = False
            else:
                state["partial_sell_done"] = True

    equity_curve.append(closed_profit_percent)
    if closed_profit_percent > peak_equity:
        peak_equity = closed_profit_percent
    drawdown = peak_equity - closed_profit_percent
    if drawdown > max_drawdown:
        max_drawdown = drawdown

# ============================================================
# RESULTS GENERATION & STATS
# ============================================================

total_trades = len(trades)
winning_trades = [t for t in trades if t["profit_pct"] > 0]
losing_trades = [t for t in trades if t["profit_pct"] <= 0]

wins = len(winning_trades)
losses = len(losing_trades)
win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
average_profit = np.mean([t["profit_pct"] for t in winning_trades]) if winning_trades else 0
average_loss = np.mean([t["profit_pct"] for t in losing_trades]) if losing_trades else 0

results = {
    "statistics": {
        "total_trades": total_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_percent": round(win_rate, 2),
        "total_profit_percent": round(closed_profit_percent, 2),
        "average_winning_trade_percent": round(float(average_profit), 2),
        "average_losing_trade_percent": round(float(average_loss), 2),
        "maximum_drawdown_percent": round(max_drawdown, 2)
    }
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("V8 BACKTEST COMPLETE")
print("=" * 60)
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total Profit: {closed_profit_percent:.2f}%")
print("=" * 60)

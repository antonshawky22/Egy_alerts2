import json
import os
import pandas as pd
import numpy as np

# ============================================================
# 🚀 EGX LADDER BACKTEST v1.4
# REALISTIC NEXT-DAY EXECUTION
# SIMPLE LADDER ANALYSIS
# ============================================================

DB_FILE = "egx_history_database_v2.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"
SUMMARY_FILE = "backtest_summary_by_stock.json"

# ============================================================
# ⚙️ CAPITAL / PORTFOLIO
# ============================================================

INITIAL_CAPITAL = 100000
MAX_PORTFOLIO_POSITIONS = 8
POSITION_SIZE = 1 / MAX_PORTFOLIO_POSITIONS

# ============================================================
# 📊 INDICATORS
# ============================================================

RSI_PERIOD = 14
EMA_PERIOD = 95

# ============================================================
# 🛡️ BUY FILTERS
# ============================================================

RUNUP_LOOKBACK = 160
MAX_RUNUP_PERCENT = 80

MAX_GAP_DOWN_PERCENT = -5

# ============================================================
# 🪜 LADDER BUY
# ============================================================

BUY1_RSI = 60
BUY2_RSI = 55
BUY3_RSI = 48

# ============================================================
# 💰 LADDER SELL
# ============================================================

SELL1_RSI = 66
SELL1_MIN_PROFIT = 15

SELL2_MIN_POSITION = 0.30
SELL2_MAX_POSITION = 0.70
SELL2_RSI = 84
SELL2_MIN_PROFIT = 25

SELL3_RSI = 86
SELL3_MIN_PROFIT = 25

# ============================================================
# 🛑 STOP LOSS
# ============================================================

STOP_L1 = -18
STOP_L2 = -17
STOP_L3 = -13

# ============================================================
# 📈 TRAILING STOP
# ============================================================

TRAILING_TRIGGER = 36
TRAILING_GIVEBACK = 11

# ============================================================
# DATA
# ============================================================

MIN_BARS = 40

SYMBOLS = [
    "OLFI","EMFD","ETEL","EAST","EFIH","ABUK","OIH","SWDY",
    "ISPH","ATQA","MTIE","HRHO","ORWE","JUFO","DSCW","SUGR",
    "ELSH","RMDA","RAYA","EEII","MPCO","GBCO","TMGH","ORHD",
    "AMOC","FWRY","COMI","ADIB","PHDC","MCQE","SKPC","EGAL"
]

# ============================================================
# 📥 LOAD DATABASE
# ============================================================

def load_database():
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database not found: {DB_FILE}")

    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 📊 CONVERT SYMBOL DATA TO DATAFRAME
# ============================================================

def fetch_local_data(raw_database, symbol):

    if symbol not in raw_database:
        return None

    data = raw_database[symbol]

    if not isinstance(data, dict):
        return None

    rows = []

    for date, values in data.items():

        if not isinstance(values, dict):
            continue

        try:
            row = {
                "Date": pd.to_datetime(date),
                "Open": float(values.get("Open")),
                "High": float(values.get("High")),
                "Low": float(values.get("Low")),
                "Close": float(values.get("Close")),
                "Volume": float(values.get("Volume", 0))
            }

            rows.append(row)

        except:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)

    df = df.sort_values("Date").reset_index(drop=True)

    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    ).reset_index(drop=True)

    if len(df) < MIN_BARS:
        return None

    return df


# ============================================================
# 📈 RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        com=period - 1,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        com=period - 1,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# 📊 ADD INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["EMA95"] = df["Close"].ewm(
        span=EMA_PERIOD,
        adjust=False
    ).mean()

    df["RSI"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )

    return df


# ============================================================
# 🛡️ SAFE TO BUY
# ============================================================

def safe_to_buy(df, i):

    if i < RUNUP_LOOKBACK:
        return True

    window = df.iloc[
        i - RUNUP_LOOKBACK + 1:i + 1
    ]

    highest = window["High"].max()
    lowest = window["Low"].min()

    if lowest <= 0:
        return True

    runup_percent = (
        (highest - lowest) / lowest
    ) * 100

    return runup_percent <= MAX_RUNUP_PERCENT


# ============================================================
# 🛡️ GAP DOWN FILTER
# ============================================================

def no_gap_down(df, i):

    start = max(1, i - 2)

    for j in range(start, i + 1):

        previous_close = df.iloc[j - 1]["Close"]
        current_open = df.iloc[j]["Open"]

        if previous_close <= 0:
            continue

        gap_percent = (
            (current_open - previous_close)
            / previous_close
        ) * 100

        if gap_percent <= MAX_GAP_DOWN_PERCENT:
            return False

    return True


# ============================================================
# 📈 EMA TREND
# ============================================================

def ema_up(df, i):

    if i < 10:
        return False

    ema_now = df.iloc[i]["EMA95"]
    ema_5 = df.iloc[i - 5]["EMA95"]
    ema_10 = df.iloc[i - 10]["EMA95"]

    close = df.iloc[i]["Close"]

    if pd.isna(ema_now) or pd.isna(ema_5) or pd.isna(ema_10):
        return False

    if ema_now <= ema_5:
        return False

    if ema_now <= ema_10 * 1.002:
        return False

    if close > ema_now * 1.07:
        return False

    return True


# ============================================================
# 🪜 CREATE TRADE
# ============================================================

def create_trade(symbol, signal_date, execution_date, price, level):

    return {
        "symbol": symbol,
        "entry_date": execution_date,
        "signal_date": signal_date,
        "position": round(level, 2),
        "avg_price": price,
        "peak_profit": 0,
        "realized_profit": 0,
        "entries": 1,
        "second_entry": None,
        "third_entry": None,
        "exit_date": None,
        "exit_price": None,
        "exit_reason": None
    }


# ============================================================
# 📊 CURRENT PROFIT
# ============================================================

def calculate_current_profit(trade, price):

    avg = trade["avg_price"]

    if avg <= 0:
        return 0

    return (
        (price - avg) / avg
    ) * 100


# ============================================================
# 🪜 ADD LADDER ENTRY
# ============================================================

def add_ladder_entry(
    trade,
    price,
    level,
    execution_date
):

    old_position = trade["position"]
    old_avg = trade["avg_price"]

    new_position = old_position + level

    if new_position > 1:
        new_position = 1

    new_avg = (
        old_avg * old_position +
        price * level
    ) / new_position

    trade["avg_price"] = new_avg
    trade["position"] = round(new_position, 2)
    trade["entries"] += 1

    if level == 0.33:
        if trade["second_entry"] is None:
            trade["second_entry"] = execution_date
        else:
            trade["third_entry"] = execution_date

    # After averaging down, reset peak relative to new cost basis
    trade["peak_profit"] = 0


# ============================================================
# 💰 PARTIAL SELL
# ============================================================

def partial_sell(
    trade,
    price,
    sell_position,
    execution_date,
    reason
):

    if sell_position <= 0:
        return False

    current_position = trade["position"]

    if sell_position > current_position:
        sell_position = current_position

    profit = calculate_current_profit(
        trade,
        price
    )

    realized = (
        profit *
        sell_position
    ) / 100

    trade["realized_profit"] += realized

    trade["position"] = round(
        current_position - sell_position,
        2
    )

    if trade["position"] <= 0.001:

        trade["position"] = 0

        trade["exit_date"] = execution_date
        trade["exit_price"] = price
        trade["exit_reason"] = reason

        return True

    return False


# ============================================================
# 🔴 FULL CLOSE
# ============================================================

def close_trade(
    trade,
    price,
    execution_date,
    reason
):

    current_profit = calculate_current_profit(
        trade,
        price
    )

    remaining_profit = (
        current_profit *
        trade["position"]
    ) / 100

    total_profit = (
        trade["realized_profit"] +
        remaining_profit
    ) * 100

    trade["exit_date"] = execution_date
    trade["exit_price"] = price
    trade["exit_reason"] = reason
    trade["final_profit"] = total_profit

    trade["position"] = 0

    return trade


# ============================================================
# 🧮 FINAL PROFIT FOR OPEN TRADE
# ============================================================

def mark_open_trade(trade, price):

    current_profit = calculate_current_profit(
        trade,
        price
    )

    remaining_profit = (
        current_profit *
        trade["position"]
    ) / 100

    total_profit = (
        trade["realized_profit"] +
        remaining_profit
    ) * 100

    trade["final_profit"] = total_profit

    return trade


# ============================================================
# 📊 SINGLE STOCK BACKTEST
# ============================================================

def run_symbol_backtest(symbol, df):

    trades = []

    trade = None

    start_index = max(
        EMA_PERIOD,
        RUNUP_LOOKBACK,
        15
    )

    for i in range(
        start_index,
        len(df) - 1
    ):

        row = df.iloc[i]

        next_row = df.iloc[i + 1]

        signal_date = str(
            row["Date"].date()
        )

        execution_date = str(
            next_row["Date"].date()
        )

        signal_close = row["Close"]

        signal_rsi = row["RSI"]

        if pd.isna(signal_rsi):
            continue

        # ====================================================
        # 🟢 NO OPEN TRADE → LOOK FOR L1
        # ====================================================

        if trade is None:

            buy_signal = (
                ema_up(df, i)
                and safe_to_buy(df, i)
                and no_gap_down(df, i)
                and signal_rsi <= BUY1_RSI
            )

            if buy_signal:

                execution_price = next_row["Open"]

                trade = create_trade(
                    symbol,
                    signal_date,
                    execution_date,
                    execution_price,
                    0.33
                )

                trade["signal_rsi"] = float(
                    signal_rsi
                )

                continue

        # ====================================================
        # 🟡 OPEN TRADE
        # ====================================================

        if trade is None:
            continue

        current_price = signal_close

        current_profit = calculate_current_profit(
            trade,
            current_price
        )

        # ====================================================
        # 📈 UPDATE PEAK
        # ====================================================

        if current_profit > trade["peak_profit"]:
            trade["peak_profit"] = current_profit

        # ====================================================
        # 🛑 STOP LOSS
        # ====================================================

        position = trade["position"]

        stop_level = STOP_L3

        if position <= 0.33:
            stop_level = STOP_L1

        elif position <= 0.66:
            stop_level = STOP_L2

        if current_profit <= stop_level:

            execution_price = next_row["Open"]

            close_trade(
                trade,
                execution_price,
                execution_date,
                "STOP LOSS"
            )

            trades.append(trade)
            trade = None

            continue

        # ====================================================
        # 📉 TRAILING STOP
        # ====================================================

        if (
            trade["peak_profit"] >= TRAILING_TRIGGER
            and
            trade["peak_profit"] - current_profit
            >= TRAILING_GIVEBACK
        ):

            execution_price = next_row["Open"]

            close_trade(
                trade,
                execution_price,
                execution_date,
                "TRAILING STOP"
            )

            trades.append(trade)
            trade = None

            continue

        # ====================================================
        # 🔴 SELL L3 / FULL EXIT
        # ====================================================

        if (
            signal_rsi >= SELL3_RSI
            and current_profit >= SELL3_MIN_PROFIT
        ):

            execution_price = next_row["Open"]

            close_trade(
                trade,
                execution_price,
                execution_date,
                "SELL L3"
            )

            trades.append(trade)
            trade = None

            continue

        # ====================================================
        # 🟠 SELL L2
        # ====================================================

        if (
            SELL2_MIN_POSITION
            < position
            <= SELL2_MAX_POSITION
            and
            signal_rsi >= SELL2_RSI
            and
            current_profit >= SELL2_MIN_PROFIT
        ):

            execution_price = next_row["Open"]

            closed = partial_sell(
                trade,
                execution_price,
                0.33,
                execution_date,
                "SELL L2"
            )

            if closed:
                trade["final_profit"] = (
                    trade["realized_profit"] * 100
                )

                trades.append(trade)
                trade = None

            continue

        # ====================================================
        # 🟢 SELL L1
        # ====================================================

        if (
            position > 0.70
            and
            signal_rsi >= SELL1_RSI
            and
            current_profit >= SELL1_MIN_PROFIT
        ):

            execution_price = next_row["Open"]

            closed = partial_sell(
                trade,
                execution_price,
                0.33,
                execution_date,
                "SELL L1"
            )

            if closed:

                trade["final_profit"] = (
                    trade["realized_profit"] * 100
                )

                trades.append(trade)
                trade = None

            continue

        # ====================================================
        # 🪜 L2 BUY
        # ====================================================

        if (
            position <= 0.33
            and
            signal_rsi <= BUY2_RSI
            and
            signal_close < trade["avg_price"] * 0.97
            and
            ema_up(df, i)
            and
            no_gap_down(df, i)
        ):

            execution_price = next_row["Open"]

            add_ladder_entry(
                trade,
                execution_price,
                0.33,
                execution_date
            )

            continue

        # ====================================================
        # 🪜 L3 BUY
        # ====================================================

        if (
            0.33 <= position < 1
            and
            signal_rsi <= BUY3_RSI
            and
            signal_close < trade["avg_price"] * 0.94
            and
            ema_up(df, i)
            and
            no_gap_down(df, i)
        ):

            execution_price = next_row["Open"]

            add_ladder_entry(
                trade,
                execution_price,
                0.34,
                execution_date
            )

            continue

    # ========================================================
    # 📌 MARK OPEN TRADE AT LAST CLOSE
    # ========================================================

    if trade is not None:

        last_price = df.iloc[-1]["Close"]

        mark_open_trade(
            trade,
            last_price
        )

        trade["exit_reason"] = "OPEN AT END"

        trades.append(trade)

    return trades


# ============================================================
# 📊 STOCK SUMMARY
# ============================================================

def build_stock_summary(symbol, trades):

    closed = [
        t for t in trades
        if t.get("exit_reason") != "OPEN AT END"
    ]

    if not trades:
        return {
            "symbol": symbol,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_profit": 0,
            "avg_profit": 0,
            "best_trade": 0,
            "worst_trade": 0
        }

    profits = [
        float(t.get("final_profit", 0))
        for t in trades
    ]

    wins = [
        p for p in profits
        if p > 0
    ]

    losses = [
        p for p in profits
        if p <= 0
    ]

    return {
        "symbol": symbol,
        "trades": len(trades),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(
            len(wins) / len(profits) * 100,
            2
        ) if profits else 0,
        "total_profit": round(
            sum(profits),
            2
        ),
        "avg_profit": round(
            np.mean(profits),
            2
        ) if profits else 0,
        "best_trade": round(
            max(profits),
            2
        ) if profits else 0,
        "worst_trade": round(
            min(profits),
            2
        ) if profits else 0
    }


# ============================================================
# 📊 GLOBAL STATISTICS
# ============================================================

def build_global_results(all_trades):

    if not all_trades:

        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_profit": 0,
            "avg_profit": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "ladder_l1": 0,
            "ladder_l2": 0,
            "ladder_l3": 0
        }

    profits = [
        float(t.get("final_profit", 0))
        for t in all_trades
    ]

    wins = [
        p for p in profits
        if p > 0
    ]

    losses = [
        p for p in profits
        if p <= 0
    ]

    gross_profit = sum(wins)

    gross_loss = abs(sum(losses))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 0
    )

    ladder_l1 = sum(
        1 for t in all_trades
        if t.get("entries", 0) >= 1
    )

    ladder_l2 = sum(
        1 for t in all_trades
        if t.get("entries", 0) >= 2
    )

    ladder_l3 = sum(
        1 for t in all_trades
        if t.get("entries", 0) >= 3
    )

    return {
        "total_trades": len(profits),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(
            len(wins) / len(profits) * 100,
            2
        ),
        "total_profit": round(
            sum(profits),
            2
        ),
        "avg_profit": round(
            np.mean(profits),
            2
        ),
        "avg_win": round(
            np.mean(wins),
            2
        ) if wins else 0,
        "avg_loss": round(
            np.mean(losses),
            2
        ) if losses else 0,
        "profit_factor": round(
            profit_factor,
            2
        ),
        "best_trade": round(
            max(profits),
            2
        ),
        "worst_trade": round(
            min(profits),
            2
        ),
        "ladder_l1": ladder_l1,
        "ladder_l2": ladder_l2,
        "ladder_l3": ladder_l3
    }


# ============================================================
# 📊 EXIT STATISTICS
# ============================================================

def build_exit_statistics(all_trades):

    exits = {}

    for trade in all_trades:

        reason = trade.get(
            "exit_reason",
            "UNKNOWN"
        )

        exits[reason] = exits.get(
            reason,
            0
        ) + 1

    return exits


# ============================================================
# 📈 SIMPLE PORTFOLIO SIMULATION
# ============================================================

def portfolio_simulation(all_trades):

    capital = INITIAL_CAPITAL

    ordered = sorted(
        all_trades,
        key=lambda x: x.get(
            "exit_date",
            ""
        )
    )

    for trade in ordered:

        profit = float(
            trade.get("final_profit", 0)
        )

        capital *= (
            1 +
            (profit / 100) *
            POSITION_SIZE
        )

    total_return = (
        (capital - INITIAL_CAPITAL)
        / INITIAL_CAPITAL
    ) * 100

    return {
        "initial_capital": INITIAL_CAPITAL,
        "position_size_percent": round(
            POSITION_SIZE * 100,
            2
        ),
        "final_capital": round(
            capital,
            2
        ),
        "total_return_percent": round(
            total_return,
            2
        )
    }


# ============================================================
# 🚀 MAIN
# ============================================================

def main():

    print("=" * 70)
    print("🚀 EGX LADDER BACKTEST v1.4")
    print("📊 REALISTIC NEXT-DAY EXECUTION")
    print("📊 SIMPLE LADDER ANALYSIS")
    print("=" * 70)

    raw_database = load_database()

    print(
        f"📂 Database loaded: "
        f"{len(raw_database)} symbols"
    )

    all_trades = []
    stock_summaries = {}

    successful = 0
    skipped = 0

    for number, symbol in enumerate(
        SYMBOLS,
        1
    ):

        print(
            f"\n[{number}/{len(SYMBOLS)}] "
            f"{symbol}"
        )

        df = fetch_local_data(
            raw_database,
            symbol
        )

        if df is None:

            print("   ⚠️ No valid data")

            skipped += 1

            continue

        df = add_indicators(df)

        print(
            f"   📊 Bars: {len(df)}"
        )

        trades = run_symbol_backtest(
            symbol,
            df
        )

        print(
            f"   🪜 Trades: {len(trades)}"
        )

        if trades:

            total_profit = sum(
                t.get(
                    "final_profit",
                    0
                )
                for t in trades
            )

            print(
                f"   💰 Total P/L: "
                f"{total_profit:.2f}%"
            )

        stock_summaries[symbol] = (
            build_stock_summary(
                symbol,
                trades
            )
        )

        all_trades.extend(trades)

        successful += 1

    # ========================================================
    # GLOBAL RESULTS
    # ========================================================

    results = build_global_results(
        all_trades
    )

    results["exit_statistics"] = (
        build_exit_statistics(
            all_trades
        )
    )

    results["portfolio_simulation"] = (
        portfolio_simulation(
            all_trades
        )
    )

    results["symbols_tested"] = successful
    results["symbols_skipped"] = skipped

    # ========================================================
    # SAVE TRADES
    # ========================================================

    with open(
        TRADES_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_trades,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SAVE STOCK SUMMARY
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stock_summaries,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("🏁 BACKTEST COMPLETE")
    print("=" * 70)

    print(
        f"📊 Symbols tested : "
        f"{successful}"
    )

    print(
        f"📊 Total trades   : "
        f"{results['total_trades']}"
    )

    print(
        f"✅ Wins           : "
        f"{results['wins']}"
    )

    print(
        f"❌ Losses         : "
        f"{results['losses']}"
    )

    print(
        f"🎯 Win rate       : "
        f"{results['win_rate']}%"
    )

    print(
        f"💰 Total P/L      : "
        f"{results['total_profit']}%"
    )

    print(
        f"📈 Avg trade      : "
        f"{results['avg_profit']}%"
    )

    print(
        f"🏆 Best trade     : "
        f"{results['best_trade']}%"
    )

    print(
        f"💥 Worst trade    : "
        f"{results['worst_trade']}%"
    )

    print(
        f"📊 Profit Factor  : "
        f"{results['profit_factor']}"
    )

    print("\n🪜 LADDER")

    print(
        f"   L1 trades: "
        f"{results['ladder_l1']}"
    )

    print(
        f"   L2 trades: "
        f"{results['ladder_l2']}"
    )

    print(
        f"   L3 trades: "
        f"{results['ladder_l3']}"
    )

    print("\n💰 SIMPLE PORTFOLIO")

    portfolio = results[
        "portfolio_simulation"
    ]

    print(
        f"   Initial capital: "
        f"{portfolio['initial_capital']}"
    )

    print(
        f"   Final capital:   "
        f"{portfolio['final_capital']}"
    )

    print(
        f"   Return:          "
        f"{portfolio['total_return_percent']}%"
    )

    print("\n📤 Files created:")

    print(
        f"   {RESULTS_FILE}"
    )

    print(
        f"   {TRADES_FILE}"
    )

    print(
        f"   {SUMMARY_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

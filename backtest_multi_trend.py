import json
import os
import pandas as pd
import numpy as np

DB_FILE = "egx_weekly_database_v1.json"
OUTPUT_FILE = "backtest_results.json"

RSI_PERIOD = 14
RSI_BUY = 33
RSI_SELL_1 = 60
RSI_SELL_2 = 70

FIRST_SELL = 0.50
SECOND_SELL = 0.50

HARD_STOP = 0.05
BREAK_EVEN_AFTER_FIRST_SELL = True


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def load_database():
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(
            f"Database not found: {DB_FILE}"
        )

    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def database_to_dataframe(raw):
    rows = []

    if not isinstance(raw, dict):
        return pd.DataFrame()

    for symbol, value in raw.items():

        if symbol == "EGX30":
            continue

        if isinstance(value, dict):

            data = value.get("data")

            if isinstance(data, dict):
                for date, prices in data.items():
                    if isinstance(prices, dict):
                        row = {
                            "Date": date,
                            "Open": prices.get("Open"),
                            "High": prices.get("High"),
                            "Low": prices.get("Low"),
                            "Close": prices.get("Close"),
                            "Volume": prices.get("Volume"),
                            "Symbol": symbol
                        }
                        rows.append(row)

            elif isinstance(value.get("rows"), list):
                for item in value["rows"]:
                    if not isinstance(item, dict):
                        continue

                    row = dict(item)
                    row["Symbol"] = symbol
                    rows.append(row)

        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue

                row = dict(item)
                row["Symbol"] = symbol
                rows.append(row)

    return pd.DataFrame(rows)


def prepare_dataframe(df):
    if df.empty:
        return df

    df = df.copy()

    date_col = None

    for col in ["Date", "date", "Datetime", "datetime"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    rename = {}

    for col in df.columns:
        low = str(col).lower()

        if low == "open":
            rename[col] = "Open"
        elif low == "high":
            rename[col] = "High"
        elif low == "low":
            rename[col] = "Low"
        elif low == "close":
            rename[col] = "Close"
        elif low == "volume":
            rename[col] = "Volume"

    df = df.rename(columns=rename)

    required = ["Date", "Close"]

    if any(x not in df.columns for x in required):
        return pd.DataFrame()

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    if "Low" in df.columns:
        df["Low"] = pd.to_numeric(
            df["Low"],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Date", "Close"]
    )

    df = df[df["Close"] > 0]

    df = df.sort_values("Date")

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    df["RSI14"] = calculate_rsi(
        df["Close"],
        RSI_PERIOD
    )

    return df.reset_index(drop=True)


def weighted_profit(sales):
    total = 0.0

    for sale in sales:
        total += (
            sale["profit_pct"] *
            sale["position_sold"]
        )

    return total


def backtest_symbol(symbol, df):

    if len(df) < RSI_PERIOD + 5:
        return []

    trades = []

    in_position = False
    entry_price = 0.0
    entry_date = None

    position = 0.0
    first_sold = False

    sales = []

    for i in range(RSI_PERIOD, len(df)):

        row = df.iloc[i]

        date = row["Date"]
        close = float(row["Close"])
        rsi = float(row["RSI14"])

        if np.isnan(rsi):
            continue

        # =========================
        # ENTRY
        # =========================

        if not in_position:

            if rsi < RSI_BUY:

                in_position = True
                entry_price = close
                entry_date = date
                position = 1.0
                first_sold = False
                sales = []

            continue

        # =========================
        # STOP LOSS
        # =========================

        stop_price = entry_price * (1 - HARD_STOP)

        if not first_sold:

            if close <= stop_price:

                profit = (
                    (close - entry_price) /
                    entry_price
                ) * 100

                sales.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "price": round(close, 4),
                    "position_sold": 1.0,
                    "profit_pct": round(profit, 2),
                    "reason": "HARD_STOP"
                })

                trades.append({
                    "symbol": symbol,
                    "status": "CLOSED",
                    "entry_date": entry_date.strftime(
                        "%Y-%m-%d"
                    ),
                    "entry_price": round(
                        entry_price, 4
                    ),
                    "position": 1.0,
                    "sales": sales,
                    "exit_date": date.strftime(
                        "%Y-%m-%d"
                    ),
                    "exit_price": round(
                        close, 4
                    ),
                    "profit_pct": round(
                        profit, 2
                    ),
                    "exit_reason": "HARD_STOP"
                })

                in_position = False
                position = 0.0
                sales = []

                continue

        # =========================
        # FIRST SELL 50%
        # =========================

        if not first_sold and rsi >= RSI_SELL_1:

            profit = (
                (close - entry_price) /
                entry_price
            ) * 100

            sold = FIRST_SELL

            sales.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(close, 4),
                "position_sold": sold,
                "profit_pct": round(profit, 2),
                "reason": "RSI_60"
            })

            position -= sold
            first_sold = True

            continue

        # =========================
        # BREAK EVEN STOP
        # =========================

        if (
            first_sold
            and BREAK_EVEN_AFTER_FIRST_SELL
            and close <= entry_price
        ):

            sold = position

            profit = 0.0

            sales.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(close, 4),
                "position_sold": sold,
                "profit_pct": round(profit, 2),
                "reason": "BREAK_EVEN_STOP"
            })

            total_profit = weighted_profit(sales)

            trades.append({
                "symbol": symbol,
                "status": "CLOSED",
                "entry_date": entry_date.strftime(
                    "%Y-%m-%d"
                ),
                "entry_price": round(
                    entry_price, 4
                ),
                "position": 1.0,
                "sales": sales,
                "exit_date": date.strftime(
                    "%Y-%m-%d"
                ),
                "exit_price": round(
                    close, 4
                ),
                "profit_pct": round(
                    total_profit, 2
                ),
                "exit_reason": "BREAK_EVEN_STOP"
            })

            in_position = False
            position = 0.0
            sales = []

            continue

        # =========================
        # SECOND SELL 50%
        # =========================

        if first_sold and rsi >= RSI_SELL_2:

            sold = position

            profit = (
                (close - entry_price) /
                entry_price
            ) * 100

            sales.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(close, 4),
                "position_sold": sold,
                "profit_pct": round(profit, 2),
                "reason": "RSI_70"
            })

            total_profit = weighted_profit(sales)

            trades.append({
                "symbol": symbol,
                "status": "CLOSED",
                "entry_date": entry_date.strftime(
                    "%Y-%m-%d"
                ),
                "entry_price": round(
                    entry_price, 4
                ),
                "position": 1.0,
                "sales": sales,
                "exit_date": date.strftime(
                    "%Y-%m-%d"
                ),
                "exit_price": round(
                    close, 4
                ),
                "profit_pct": round(
                    total_profit, 2
                ),
                "exit_reason": "RSI_70"
            })

            in_position = False
            position = 0.0
            sales = []

            continue

    # =========================
    # OPEN POSITION
    # =========================

    if in_position:

        last = df.iloc[-1]

        trades.append({
            "symbol": symbol,
            "status": "OPEN",
            "entry_date": entry_date.strftime(
                "%Y-%m-%d"
            ),
            "entry_price": round(
                entry_price, 4
            ),
            "position": round(
                position, 2
            ),
            "sales": sales,
            "exit_date": None,
            "exit_price": None,
            "profit_pct": None
        })

    return trades


def calculate_statistics(trades):

    closed = [
        t for t in trades
        if t["status"] == "CLOSED"
    ]

    wins = [
        t for t in closed
        if t["profit_pct"] > 0
    ]

    losses = [
        t for t in closed
        if t["profit_pct"] <= 0
    ]

    total = len(closed)

    win_rate = (
        len(wins) / total * 100
        if total else 0
    )

    avg_win = (
        np.mean([
            t["profit_pct"]
            for t in wins
        ])
        if wins else 0
    )

    avg_loss = (
        np.mean([
            t["profit_pct"]
            for t in losses
        ])
        if losses else 0
    )

    # ==================================
    # Compound portfolio return
    # ==================================

    capital = 1.0
    equity_curve = [capital]

    for trade in closed:

        ret = trade["profit_pct"] / 100

        capital *= (1 + ret)

        equity_curve.append(capital)

    compound_return = (
        (capital - 1) * 100
    )

    # ==================================
    # Maximum drawdown
    # ==================================

    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:

        if equity > peak:
            peak = equity

        if peak > 0:

            dd = (
                (peak - equity) /
                peak
            ) * 100

            max_dd = max(
                max_dd,
                dd
            )

    return {
        "total_trades": total,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_percent": round(
            win_rate, 2
        ),
        "compound_portfolio_return_percent":
            round(
                compound_return,
                2
            ),
        "average_winning_trade_percent":
            round(avg_win, 2),
        "average_losing_trade_percent":
            round(avg_loss, 2),
        "maximum_drawdown_percent":
            round(max_dd, 2)
    }


def main():

    print(
        "EGX WEEKLY RSI 33/60/70 "
        "BACKTEST"
    )

    raw = load_database()

    df_all = database_to_dataframe(raw)

    if df_all.empty:
        print("No database data found.")
        return

    all_trades = []

    symbols = sorted(
        df_all["Symbol"]
        .dropna()
        .unique()
    )

    print(
        f"Symbols prepared: {len(symbols)}"
    )

    for symbol in symbols:

        symbol_df = df_all[
            df_all["Symbol"] == symbol
        ].copy()

        symbol_df = prepare_dataframe(
            symbol_df
        )

        if symbol_df.empty:
            continue

        trades = backtest_symbol(
            symbol,
            symbol_df
        )

        all_trades.extend(trades)

    all_trades.sort(
        key=lambda x: (
            x["entry_date"],
            x["symbol"]
        )
    )

    closed = [
        t for t in all_trades
        if t["status"] == "CLOSED"
    ]

    open_positions = [
        t for t in all_trades
        if t["status"] == "OPEN"
    ]

    statistics = calculate_statistics(
        all_trades
    )

    result = {
        "strategy":
            "Weekly RSI 33/60/70 + Break-Even",
        "parameters": {
            "rsi_period": RSI_PERIOD,
            "rsi_buy": RSI_BUY,
            "rsi_sell_1": RSI_SELL_1,
            "rsi_sell_2": RSI_SELL_2,
            "first_sell_percent": 50,
            "second_sell_percent": 50,
            "hard_stop_percent": 5.0,
            "break_even_after_first_sell":
                BREAK_EVEN_AFTER_FIRST_SELL
        },
        "statistics": statistics,
        "trades": all_trades,
        "open_positions": open_positions
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Closed trades: {len(closed)}"
    )

    print(
        f"Open positions: "
        f"{len(open_positions)}"
    )

    print(
        f"Win rate: "
        f"{statistics['win_rate_percent']}%"
    )

    print(
        f"Compound return: "
        f"{statistics['compound_portfolio_return_percent']}%"
    )

    print(
        f"Max drawdown: "
        f"{statistics['maximum_drawdown_percent']}%"
    )

    print(
        f"Results saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

import json
import os
import pandas as pd
import numpy as np

# ============================================================
# BACKTEST - MULTI TREND EMA70
# 3-TRANCHE POSITION SYSTEM
# ============================================================

DB_FILE = "egx_history_database_v2.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"
STOCK_SUMMARY_FILE = "backtest_summary_by_stock.json"

EGX30_KEY = "EGX30"

# ============================================================
# STRATEGY PARAMETERS
# ============================================================

# ------------------------------------------------------------
# EMA70 TREND
# ------------------------------------------------------------

# EMA70 current > EMA70-4 > EMA70-8 > EMA70-12
# with clear positive steps
EMA70_UP_MIN_STEP_PERCENT = 0.30

# EMA70 levels close together = sideways
EMA70_SIDE_MAX_DISTANCE_PERCENT = 1.00

# Very strong downward movement
EMA70_DOWN_MIN_STEP_PERCENT = 1.00

# ------------------------------------------------------------
# RSI
# ------------------------------------------------------------

# First buy while EMA70 is clearly rising
RSI_UP_BUY = 48

# Second buy after EMA70 starts moving sideways
# and price gives a deeper pullback
RSI_SIDE_BUY_2 = 30

# Third buy only at a still lower RSI level
RSI_SIDE_BUY_3 = 25

# Partial profit target
RSI_PARTIAL_SELL = 66

# Strong profit target
RSI_FINAL_SELL = 77

# ------------------------------------------------------------
# STOP LOSS
# ------------------------------------------------------------

STOP_LOSS_PERCENT = 7.0

# ============================================================
# POSITION SIZE
# ============================================================

TRANCHE_SIZE = 1.0 / 3.0

# ============================================================
# RSI - WILDER
# ============================================================

def rsi(series, period=14):

    if len(series) < period:
        return pd.Series(
            np.nan,
            index=series.index
        )

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        com=period - 1,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        com=period - 1,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(
        f"Database file not found: {DB_FILE}"
    )

with open(
    DB_FILE,
    "r",
    encoding="utf-8"
) as f:
    raw_database = json.load(f)

print("Historical database loaded.")

symbols = list(raw_database.keys())

if EGX30_KEY not in raw_database:
    raise ValueError(
        "EGX30 not found in database."
    )


# ============================================================
# PREPARE DATA
# ============================================================

prepared_data = {}

for symbol in symbols:

    content = raw_database.get(
        symbol,
        {}
    )

    if (
        "data" not in content
        or
        "columns" not in content
    ):
        continue

    try:

        df = pd.DataFrame.from_dict(
            content["data"],
            orient="index",
            columns=content["columns"]
        )

        df.index = pd.to_datetime(
            df.index
        )

        df = df.sort_index()

        # EMA70 needs enough history
        if len(df) < 80:
            continue

        # ----------------------------------------------------
        # INDICATORS
        # ----------------------------------------------------

        df["EMA12"] = df["Close"].ewm(
            span=12,
            adjust=False
        ).mean()

        df["EMA20"] = df["Close"].ewm(
            span=20,
            adjust=False
        ).mean()

        df["EMA70"] = df["Close"].ewm(
            span=70,
            adjust=False
        ).mean()

        df["RSI14"] = rsi(
            df["Close"],
            14
        )

        # ----------------------------------------------------
        # EMA12 / EMA20 CROSS
        # ----------------------------------------------------

        df["cross_up"] = (
            (df["EMA12"] > df["EMA20"])
            &
            (
                df["EMA12"].shift(1)
                <=
                df["EMA20"].shift(1)
            )
        )

        df["cross_down"] = (
            (df["EMA12"] < df["EMA20"])
            &
            (
                df["EMA12"].shift(1)
                >=
                df["EMA20"].shift(1)
            )
        )

        prepared_data[symbol] = df

    except Exception as e:

        print(
            f"Failed preparing {symbol}: {e}"
        )


print(
    f"Prepared {len(prepared_data)} symbols."
)


# ============================================================
# EMA70 TREND DETECTION
# ============================================================

def calculate_trend(df, index):

    if index < 12:
        return "🔛"

    ema70_now = float(
        df.iloc[index]["EMA70"]
    )

    ema70_4 = float(
        df.iloc[index - 4]["EMA70"]
    )

    ema70_8 = float(
        df.iloc[index - 8]["EMA70"]
    )

    ema70_12 = float(
        df.iloc[index - 12]["EMA70"]
    )

    if any(
        pd.isna(x)
        for x in [
            ema70_now,
            ema70_4,
            ema70_8,
            ema70_12
        ]
    ):
        return "🔛"

    # --------------------------------------------------------
    # Percentage steps
    # --------------------------------------------------------

    step_1 = (
        (ema70_now - ema70_4)
        /
        ema70_4
    ) * 100

    step_2 = (
        (ema70_4 - ema70_8)
        /
        ema70_8
    ) * 100

    step_3 = (
        (ema70_8 - ema70_12)
        /
        ema70_12
    ) * 100

    # --------------------------------------------------------
    # STRONG UP
    # --------------------------------------------------------

    if (
        step_1 >= EMA70_UP_MIN_STEP_PERCENT
        and
        step_2 >= EMA70_UP_MIN_STEP_PERCENT
        and
        step_3 >= EMA70_UP_MIN_STEP_PERCENT
    ):
        return "↗️"

    # --------------------------------------------------------
    # STRONG DOWN
    # --------------------------------------------------------

    if (
        step_1 <= -EMA70_DOWN_MIN_STEP_PERCENT
        and
        step_2 <= -EMA70_DOWN_MIN_STEP_PERCENT
        and
        step_3 <= -EMA70_DOWN_MIN_STEP_PERCENT
    ):
        return "🔻"

    # --------------------------------------------------------
    # SIDEWAYS
    # --------------------------------------------------------

    max_ema70 = max(
        ema70_now,
        ema70_4,
        ema70_8,
        ema70_12
    )

    min_ema70 = min(
        ema70_now,
        ema70_4,
        ema70_8,
        ema70_12
    )

    if min_ema70 <= 0:
        return "🔛"

    distance_percent = (
        (max_ema70 - min_ema70)
        /
        min_ema70
    ) * 100

    if (
        distance_percent
        <= EMA70_SIDE_MAX_DISTANCE_PERCENT
    ):
        return "🔛"

    # Everything else is neutral
    return "🔛"


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
    raise ValueError(
        "No trading dates available."
    )

print(
    f"Backtest period: "
    f"{all_dates[0].strftime('%Y-%m-%d')} "
    f"-> "
    f"{all_dates[-1].strftime('%Y-%m-%d')}"
)


# ============================================================
# STATE
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

        "partial_sell_done": False,

        # ====================================================
        # IMPORTANT ACCOUNTING FIX
        #
        # Stores ALL sales belonging to the current
        # complete trade cycle.
        # ====================================================

        "cycle_sales": []
    }


# ============================================================
# TRADES
# ============================================================

trades = []


# ============================================================
# EQUITY
# ============================================================

closed_profit_percent = 0.0
equity_curve = []

peak_equity = 0.0
max_drawdown = 0.0


# ============================================================
# TREND STATISTICS
# ============================================================

trend_days = {
    "↗️": 0,
    "🔛": 0,
    "🔻": 0
}


# ============================================================
# TRANCHE HELPERS
# ============================================================

def position_units(state):

    return sum(
        tranche["size"]
        for tranche in state["tranches"]
    )


def average_entry_price(state):

    total_size = position_units(state)

    if total_size <= 0:
        return None

    weighted_value = sum(
        tranche["size"]
        *
        tranche["price"]
        for tranche in state["tranches"]
    )

    return weighted_value / total_size


def add_tranche(
    state,
    price,
    date
):

    state["tranches"].append({

        "size": TRANCHE_SIZE,

        "price": float(price),

        "date": date.strftime(
            "%Y-%m-%d"
        )
    })

    state["in_position"] = True


def sell_one_tranche(
    state,
    close,
    current_date,
    reason
):
    """
    Sell one 1/3 tranche.

    Returns the realized profit contribution
    relative to total portfolio capital.
    """

    if not state["tranches"]:
        return 0.0, None

    # Sell the oldest tranche first.
    tranche = state["tranches"].pop(0)

    entry_price = tranche["price"]

    size = tranche["size"]

    profit_percent = (
        (close - entry_price)
        /
        entry_price
    ) * 100

    # Because size is 1/3 of total capital,
    # only one third of this return contributes
    # to total portfolio return.
    contribution = (
        profit_percent * size
    )

    sale = {

        "entry_date": tranche["date"],

        "entry_price": round(
            entry_price,
            4
        ),

        "exit_date": current_date.strftime(
            "%Y-%m-%d"
        ),

        "exit_price": round(
            close,
            4
        ),

        "size": round(
            size,
            4
        ),

        "profit_pct_on_tranche": round(
            profit_percent,
            2
        ),

        "portfolio_contribution_pct": round(
            contribution,
            2
        ),

        "reason": reason
    }

    if not state["tranches"]:

        state["in_position"] = False

    return contribution, sale


def sell_all(
    state,
    close,
    current_date,
    reason
):
    """
    Close all remaining tranches.
    """

    total_contribution = 0.0

    sales = []

    while state["tranches"]:

        contribution, sale = sell_one_tranche(
            state,
            close,
            current_date,
            reason
        )

        total_contribution += contribution

        if sale:

            sales.append(
                sale
            )

    state["in_position"] = False

    return total_contribution, sales


# ============================================================
# MAIN BACKTEST LOOP
# ============================================================

for current_date in all_dates:

    # --------------------------------------------------------
    # IMPORTANT:
    # EGX30 IS NOT USED AS A MARKET FILTER.
    # --------------------------------------------------------

    for symbol, df in prepared_data.items():

        if symbol == EGX30_KEY:
            continue

        if current_date not in df.index:
            continue

        current_index = df.index.get_loc(
            current_date
        )

        if current_index < 80:
            continue

        row = df.iloc[current_index]

        close = float(
            row["Close"]
        )

        if pd.isna(close):
            continue

        if pd.isna(row["RSI14"]):
            continue

        state = states[symbol]

        in_position = state["in_position"]

        units = position_units(
            state
        )

        trend = calculate_trend(
            df,
            current_index
        )

        trend_days[trend] += 1

        buy_signal = False

        buy_level = None

        realized_profit = 0.0

        # ====================================================
        # STRONG UP TREND
        # ====================================================

        if trend == "↗️":

            # ------------------------------------------------
            # FIRST ENTRY
            #
            # Only the first 1/3 is bought here.
            # We DO NOT chase the price upward.
            # ------------------------------------------------

            if (
                units == 0
                and
                row["RSI14"] < RSI_UP_BUY
            ):

                buy_signal = True

                buy_level = 1

            # ------------------------------------------------
            # PROFIT TAKING AT RSI 66
            #
            # Sell only ONE tranche.
            # ------------------------------------------------

            elif (
                units > 0
                and
                row["RSI14"] > RSI_PARTIAL_SELL
            ):

                realized_profit, sale = (
                    sell_one_tranche(
                        state,
                        close,
                        current_date,
                        "RSI_66"
                    )
                )

                # =================================================
                # ACCOUNTING FIX:
                # Keep this sale inside the complete trade cycle.
                # =================================================

                if sale:

                    state["cycle_sales"].append(
                        sale
                    )

        # ====================================================
        # SIDEWAYS
        # ====================================================

        elif trend == "🔛":

            # ------------------------------------------------
            # SECOND BUY
            #
            # Must already have first tranche.
            # RSI must fall to <= 30.
            #
            # This is a better-price addition.
            # ------------------------------------------------

            if (
                units >= TRANCHE_SIZE - 0.000001
                and
                units < (
                    2 * TRANCHE_SIZE
                    - 0.000001
                )
                and
                row["RSI14"] <= RSI_SIDE_BUY_2
            ):

                buy_signal = True

                buy_level = 2

            # ------------------------------------------------
            # THIRD BUY
            #
            # Only after RSI reaches an even lower level.
            # We NEVER buy the third tranche because
            # the stock is rising.
            # ------------------------------------------------

            elif (
                units >= (
                    2 * TRANCHE_SIZE
                    - 0.000001
                )
                and
                units < (
                    3 * TRANCHE_SIZE
                    - 0.000001
                )
                and
                row["RSI14"] <= RSI_SIDE_BUY_3
            ):

                buy_signal = True

                buy_level = 3

            # ------------------------------------------------
            # SIDEWAYS PROFIT
            #
            # RSI > 66 sells one tranche.
            # ------------------------------------------------

            elif (
                units > 0
                and
                row["RSI14"] > RSI_PARTIAL_SELL
            ):

                realized_profit, sale = (
                    sell_one_tranche(
                        state,
                        close,
                        current_date,
                        "SIDE_RSI_66"
                    )
                )

                # =================================================
                # ACCOUNTING FIX
                # =================================================

                if sale:

                    state["cycle_sales"].append(
                        sale
                    )

        # ====================================================
        # STRONG DOWN TREND
        # ====================================================

        elif trend == "🔻":

            # ------------------------------------------------
            # NO NEW BUY
            #
            # CLOSE ALL OPEN POSITION
            # ------------------------------------------------

            if units > 0:

                realized_profit, stop_sales = sell_all(
                    state,
                    close,
                    current_date,
                    "EMA70_STRONG_DOWN"
                )

                # =================================================
                # ACCOUNTING FIX
                # =================================================

                for sale in stop_sales:

                    state["cycle_sales"].append(
                        sale
                    )

        # ====================================================
        # FINAL RSI TARGET
        #
        # RSI > 77:
        # Sell EVERYTHING that remains.
        #
        # This is checked after trend logic so that
        # strong downtrend always has priority.
        # ====================================================

        if (
            state["in_position"]
            and
            row["RSI14"] > RSI_FINAL_SELL
            and
            trend != "🔻"
        ):

            final_profit, final_sales = sell_all(
                state,
                close,
                current_date,
                "RSI_77"
            )

            realized_profit += final_profit

            # =================================================
            # ACCOUNTING FIX
            # =================================================

            for sale in final_sales:

                state["cycle_sales"].append(
                    sale
                )

        # ====================================================
        # STOP LOSS
        #
        # Applied to the weighted average position.
        #
        # IMPORTANT:
        # It does not use the first entry price only.
        # ====================================================

        if state["in_position"]:

            avg_price = average_entry_price(
                state
            )

            if (
                avg_price is not None
                and
                close
                <
                avg_price
                *
                (
                    1
                    -
                    STOP_LOSS_PERCENT
                    /
                    100
                )
            ):

                stop_profit, stop_sales = sell_all(
                    state,
                    close,
                    current_date,
                    "STOP_LOSS"
                )

                realized_profit += stop_profit

                # =================================================
                # ACCOUNTING FIX
                # =================================================

                for sale in stop_sales:

                    state["cycle_sales"].append(
                        sale
                    )

        # ====================================================
        # BUY EXECUTION
        # ====================================================

        if (
            buy_signal
            and
            not state["in_position"]
            and
            buy_level == 1
        ):

            add_tranche(
                state,
                close,
                current_date
            )

            state["entry_date"] = (
                current_date.strftime(
                    "%Y-%m-%d"
                )
            )

            state["entry_trend"] = trend

            # =================================================
            # New complete trade cycle starts here.
            # =================================================

            state["cycle_sales"] = []

        elif (
            buy_signal
            and
            state["in_position"]
            and
            buy_level in [2, 3]
        ):

            add_tranche(
                state,
                close,
                current_date
            )

        # ====================================================
        # RECORD REALIZED SELL CONTRIBUTION
        # ====================================================

        if realized_profit != 0.0:

            closed_profit_percent += (
                realized_profit
            )

            # =================================================
            # COMPLETE TRADE CYCLE
            # =================================================

            if not state["in_position"]:

                total_profit = 0.0

                # =================================================
                # IMPORTANT ACCOUNTING FIX:
                #
                # Use ALL sales belonging to this complete
                # position cycle.
                # =================================================

                for sale in state["cycle_sales"]:

                    contribution = sale[
                        "portfolio_contribution_pct"
                    ]

                    total_profit += contribution

                trade = {

                    "symbol": symbol,

                    "entry_date": state["entry_date"],

                    "exit_date": current_date.strftime(
                        "%Y-%m-%d"
                    ),

                    "entry_trend": state["entry_trend"],

                    "exit_trend": trend,

                    "profit_pct": round(
                        total_profit,
                        2
                    ),

                    "exit_reason":
                        state["cycle_sales"][-1]["reason"]
                        if state["cycle_sales"]
                        else "UNKNOWN",

                    "sales":
                        state["cycle_sales"].copy()
                }

                trades.append(
                    trade
                )

                # =================================================
                # RESET COMPLETE TRADE STATE
                # =================================================

                state["entry_date"] = None

                state["entry_trend"] = None

                state["partial_sell_done"] = False

                state["cycle_sales"] = []

            else:

                # Partial sale happened.
                # Keep original cycle open.

                state["partial_sell_done"] = True

        # ====================================================
        # EQUITY
        # ====================================================

    equity_curve.append(
        closed_profit_percent
    )

    if (
        closed_profit_percent
        >
        peak_equity
    ):

        peak_equity = (
            closed_profit_percent
        )

    drawdown = (
        peak_equity
        -
        closed_profit_percent
    )

    if drawdown > max_drawdown:

        max_drawdown = drawdown


# ============================================================
# CLOSE REMAINING POSITIONS
# ============================================================

open_positions = []

for symbol, state in states.items():

    if not state["in_position"]:
        continue

    df = prepared_data[symbol]

    last_date = df.index[-1]

    last_price = float(
        df.iloc[-1]["Close"]
    )

    avg_price = average_entry_price(
        state
    )

    open_positions.append({

        "symbol": symbol,

        "entry_date": state["entry_date"],

        "average_entry_price":
            round(
                avg_price,
                4
            )
            if avg_price
            else None,

        "position_size":
            round(
                position_units(state),
                4
            ),

        "last_date":
            last_date.strftime(
                "%Y-%m-%d"
            ),

        "last_price":
            last_price,

        "unrealized_profit_percent":
            round(
                (
                    (
                        last_price
                        -
                        avg_price
                    )
                    /
                    avg_price
                ) * 100,
                2
            )
            if avg_price
            else None,

        "tranches":
            state["tranches"]
    })


# ============================================================
# STATISTICS
# ============================================================

total_trades = len(
    trades
)

winning_trades = [
    t for t in trades
    if t["profit_pct"] > 0
]

losing_trades = [
    t for t in trades
    if t["profit_pct"] <= 0
]

wins = len(
    winning_trades
)

losses = len(
    losing_trades
)

win_rate = (
    (wins / total_trades) * 100
    if total_trades > 0
    else 0
)

average_profit = (
    np.mean([
        t["profit_pct"]
        for t in winning_trades
    ])
    if winning_trades
    else 0
)

average_loss = (
    np.mean([
        t["profit_pct"]
        for t in losing_trades
    ])
    if losing_trades
    else 0
)

best_trade = (
    max(
        trades,
        key=lambda x: x["profit_pct"]
    )
    if trades
    else None
)

worst_trade = (
    min(
        trades,
        key=lambda x: x["profit_pct"]
    )
    if trades
    else None
)


# ============================================================
# RESULTS
# ============================================================

results = {

    "backtest_period": {

        "start":
            all_dates[0].strftime(
                "%Y-%m-%d"
            ),

        "end":
            all_dates[-1].strftime(
                "%Y-%m-%d"
            )
    },

    "strategy": {

        "primary_indicator":
            "EMA70",

        "ema70_levels": [

            "current",

            "4_candles_ago",

            "8_candles_ago",

            "12_candles_ago"
        ],

        "uptrend_rule": (
            "EMA70 current > EMA70-4 > "
            "EMA70-8 > EMA70-12 "
            "with clear positive steps"
        ),

        "sideways_rule": (
            "EMA70 levels are close"
        ),

        "downtrend_rule": (
            "EMA70 levels decrease "
            "with very clear negative steps"
        ),

        "position_model": (
            "3 equal capital tranches"
        ),

        "tranche_size_percent":
            33.33,

        "first_entry": (
            "EMA70 clearly rising + RSI14 < 48"
        ),

        "second_entry": (
            "EMA70 sideways + RSI14 <= 30 "
            "after first tranche"
        ),

        "third_entry": (
            "EMA70 sideways + RSI14 <= 25 "
            "after second tranche"
        ),

        "important_rule": (
            "Third tranche is bought only "
            "at a lower price/RSI level; "
            "never because price is rising"
        ),

        "partial_exit": (
            "RSI14 > 66 -> sell one tranche"
        ),

        "final_exit": (
            "RSI14 > 77 -> sell all remaining"
        ),

        "down_action": (
            "No new buy + close all open tranches"
        ),

        "stop_loss_percent":
            STOP_LOSS_PERCENT,

        "egx30_market_filter":
            "DISABLED"
    },

    "statistics": {

        "total_trades":
            total_trades,

        "winning_trades":
            wins,

        "losing_trades":
            losses,

        "win_rate_percent":
            round(
                win_rate,
                2
            ),

        "total_profit_percent":
            round(
                closed_profit_percent,
                2
            ),

        "average_winning_trade_percent":
            round(
                float(
                    average_profit
                ),
                2
            ),

        "average_losing_trade_percent":
            round(
                float(
                    average_loss
                ),
                2
            ),

        "maximum_drawdown_percent":
            round(
                max_drawdown,
                2
            )
    },

    "trend_days":
        trend_days,

    "best_trade":
        best_trade,

    "worst_trade":
        worst_trade,

    "open_positions":
        open_positions
}


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

with open(
    TRADES_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        trades,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# PER STOCK SUMMARY
# ============================================================

stock_summary = {}

for trade in trades:

    symbol = trade["symbol"]

    profit = trade["profit_pct"]

    if symbol not in stock_summary:

        stock_summary[symbol] = {

            "symbol":
                symbol,

            "total_trades":
                0,

            "winning_trades":
                0,

            "losing_trades":
                0,

            "win_rate_percent":
                0,

            "total_profit_percent":
                0,

            "average_profit_percent":
                0,

            "average_loss_percent":
                0,

            "best_trade_percent":
                None,

            "worst_trade_percent":
                None
        }

    stock_summary[symbol][
        "total_trades"
    ] += 1

    stock_summary[symbol][
        "total_profit_percent"
    ] += profit

    if profit > 0:

        stock_summary[symbol][
            "winning_trades"
        ] += 1

    else:

        stock_summary[symbol][
            "losing_trades"
        ] += 1


# ============================================================
# STOCK STATISTICS
# ============================================================

for symbol, summary in stock_summary.items():

    total = summary[
        "total_trades"
    ]

    wins = summary[
        "winning_trades"
    ]

    symbol_trades = [
        t for t in trades
        if t["symbol"] == symbol
    ]

    winning_profits = [
        t["profit_pct"]
        for t in symbol_trades
        if t["profit_pct"] > 0
    ]

    losing_profits = [
        t["profit_pct"]
        for t in symbol_trades
        if t["profit_pct"] <= 0
    ]

    summary[
        "win_rate_percent"
    ] = round(

        (
            wins / total
        ) * 100

        if total > 0

        else 0,

        2
    )

    summary[
        "total_profit_percent"
    ] = round(

        summary[
            "total_profit_percent"
        ],

        2
    )

    summary[
        "average_profit_percent"
    ] = round(

        float(
            np.mean(
                winning_profits
            )
        )

        if winning_profits

        else 0,

        2
    )

    summary[
        "average_loss_percent"
    ] = round(

        float(
            np.mean(
                losing_profits
            )
        )

        if losing_profits

        else 0,

        2
    )

    if symbol_trades:

        summary[
            "best_trade_percent"
        ] = round(

            max(
                t["profit_pct"]
                for t in symbol_trades
            ),

            2
        )

        summary[
            "worst_trade_percent"
        ] = round(

            min(
                t["profit_pct"]
                for t in symbol_trades
            ),

            2
        )


# ============================================================
# SORT STOCKS
# ============================================================

stock_summary_list = list(
    stock_summary.values()
)

stock_summary_list.sort(
    key=lambda x:
    x["total_profit_percent"],
    reverse=True
)


# ============================================================
# SAVE STOCK SUMMARY
# ============================================================

with open(
    STOCK_SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        stock_summary_list,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# TOP / WORST STOCKS
# ============================================================

print()
print("=" * 60)
print("TOP 10 STOCKS")
print("=" * 60)

for stock in stock_summary_list[:10]:

    print(
        f"{stock['symbol']} | "
        f"Trades: {stock['total_trades']} | "
        f"Win Rate: "
        f"{stock['win_rate_percent']:.2f}% | "
        f"Result: "
        f"{stock['total_profit_percent']:.2f}%"
    )


print()
print("=" * 60)
print("WORST 10 STOCKS")
print("=" * 60)

for stock in stock_summary_list[-10:]:

    print(
        f"{stock['symbol']} | "
        f"Trades: {stock['total_trades']} | "
        f"Win Rate: "
        f"{stock['win_rate_percent']:.2f}% | "
        f"Result: "
        f"{stock['total_profit_percent']:.2f}%"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("MULTI-TREND EMA70 3-TRANCHE BACKTEST")
print("=" * 60)

print(
    f"Period: "
    f"{results['backtest_period']['start']} "
    f"-> "
    f"{results['backtest_period']['end']}"
)

print(
    f"Total Trades: "
    f"{total_trades}"
)

print(
    f"Winning Trades: "
    f"{wins}"
)

print(
    f"Losing Trades: "
    f"{losses}"
)

print(
    f"Win Rate: "
    f"{win_rate:.2f}%"
)

print(
    f"Total Profit: "
    f"{closed_profit_percent:.2f}%"
)

print(
    f"Average Win: "
    f"{average_profit:.2f}%"
)

print(
    f"Average Loss: "
    f"{average_loss:.2f}%"
)

print(
    f"Maximum Drawdown: "
    f"{max_drawdown:.2f}%"
)

print()
print(
    "EMA70 Trend Distribution:"
)

print(
    f"Up Trend Days: "
    f"{trend_days['↗️']}"
)

print(
    f"Sideways Days: "
    f"{trend_days['🔛']}"
)

print(
    f"Down Trend Days: "
    f"{trend_days['🔻']}"
)

if best_trade:

    print(
        f"Best Trade: "
        f"{best_trade['symbol']} "
        f"{best_trade['profit_pct']:.2f}%"
    )

if worst_trade:

    print(
        f"Worst Trade: "
        f"{worst_trade['symbol']} "
        f"{worst_trade['profit_pct']:.2f}%"
    )

print()

print(
    f"Results saved to: "
    f"{RESULTS_FILE}"
)

print(
    f"Trades saved to: "
    f"{TRADES_FILE}"
)

print(
    f"Stock summary saved to: "
    f"{STOCK_SUMMARY_FILE}"
)

print("=" * 60)
print("Backtest Complete.")
print("=" * 60)

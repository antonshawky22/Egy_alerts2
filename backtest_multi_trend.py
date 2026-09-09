print("=" * 80)
print("EGX LADDER CYCLE SYSTEM - FULL BACKTEST v1.2")
print("MATCHED TO LIVE LADDER STRATEGY v3.4")
print("PORTFOLIO CAPACITY CONSTRAINED")
print("=" * 80)

import json
import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DB_FILE = "egx_tradingview_daily_v1.json"
RESULT_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"
STOCK_SUMMARY_FILE = "ladder_backtest_summary_by_stock.json"

INITIAL_CAPITAL = 100000.0

# ============================================================
# REAL PORTFOLIO LIMIT
# ============================================================

MAX_PORTFOLIO_POSITIONS = 8
POSITION_SIZE = 1.0 / MAX_PORTFOLIO_POSITIONS


# ============================================================
# LIVE PARAMETERS
# ============================================================

RSI_PERIOD = 14
EMA100_PERIOD = 95

RUNUP_LOOKBACK = 160
MAX_RUNUP_PERCENT = 80

MAX_GAP_DOWN_PERCENT = -5

BUY1_RSI = 60
BUY2_RSI = 55
BUY3_RSI = 48

SELL1_RSI = 66
SELL1_MIN_PROFIT = 15

SELL2_MIN_POSITION = 0.30
SELL2_MAX_POSITION = 0.70
SELL2_RSI = 84
SELL2_MIN_PROFIT = 25

SELL3_RSI = 86
SELL3_MIN_PROFIT = 25

STOP_L1 = -18
STOP_L2 = -17
STOP_L3 = -13

TRAILING_TRIGGER = 36
TRAILING_GIVEBACK = 11

MIN_BARS = 40


# ============================================================
# SYMBOLS
# ============================================================

SYMBOLS = {
    "OLFI": "OLFI",
    "EMFD": "EMFD",
    "ETEL": "ETEL",
    "EAST": "EAST",
    "ABUK": "ABUK",
    "OIH": "OIH",
    "SWDY": "SWDY",
    "ISPH": "ISPH",
    "ATQA": "ATQA",
    "MTIE": "MTIE",
    "HRHO": "HRHO",
    "ORWE": "ORWE",
    "JUFO": "JUFO",
    "SUGR": "SUGR",
    "RMDA": "RMDA",
    "RAYA": "RAYA",
    "EEII": "EEII",
    "MPCO": "MPCO",
    "GBCO": "GBCO",
    "TMGH": "TMGH",
    "ORHD": "ORHD",
    "AMOC": "AMOC",
    "FWRY": "FWRY",
    "COMI": "COMI",
    "PHDC": "PHDC",
    "MCQE": "MCQE",
    "SKPC": "SKPC",
    "EGAL": "EGAL"
}


# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(DB_FILE)

with open(DB_FILE, "r", encoding="utf-8") as f:
    raw_database = json.load(f)

print(f"Database symbols: {len(raw_database)}")


# ============================================================
# DATA LOADER
# ============================================================

def fetch_local_data(name):
    try:
        if name not in raw_database:
            return None

        content = raw_database[name]

        if "columns" not in content or "data" not in content:
            return None

        df = pd.DataFrame.from_dict(
            content["data"],
            orient="index",
            columns=content["columns"]
        )

        df.index.name = "Date"
        df.index = pd.to_datetime(
            df.index,
            errors="coerce"
        )

        df = df.sort_index()

        required = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for col in required:
            if col not in df.columns:
                return None

        for col in required:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=required
        )

        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]

        return df

    except Exception as e:
        print(
            f"⚠️ Error processing {name}: {e}"
        )
        return None


# ============================================================
# RSI
# ============================================================

def rsi(series, period=14):

    if len(series) < period + 1:
        return pd.Series(
            np.nan,
            index=series.index
        )

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

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
# PREPARE INDICATORS
# ============================================================

def prepare_data(df):

    df = df.copy()

    close = df["Close"]

    df["EMA100"] = close.ewm(
        span=EMA100_PERIOD,
        adjust=False
    ).mean()

    df["RSI"] = rsi(
        close,
        RSI_PERIOD
    )

    return df


# ============================================================
# UPDATE AVERAGE PRICE
# ============================================================

def update_avg(
    old_avg,
    old_pos,
    new_price,
    new_pos
):

    if new_pos == 0:
        return 0.0

    added_pos = (
        new_pos
        - old_pos
    )

    if added_pos <= 0:
        return old_avg

    total_cost = (
        (old_avg * old_pos)
        +
        (new_price * added_pos)
    )

    return total_cost / new_pos


# ============================================================
# FINAL WEIGHTED PNL
# ============================================================

def calculate_final_pnl(
    realized_tracker,
    current_position,
    current_profit
):

    tracker = list(
        realized_tracker
    )

    tracker.append(
        (
            current_position,
            current_profit
        )
    )

    weight_sum = sum(
        weight
        for weight, _ in tracker
    )

    if weight_sum <= 0:
        return current_profit

    return sum(
        profit * weight
        for weight, profit in tracker
    ) / weight_sum


# ============================================================
# CREATE TRADE
# ============================================================

def create_trade(
    symbol,
    cycle,
    date,
    price
):

    return {

        "symbol": symbol,

        "cycle": cycle,

        "status": "OPEN",

        "portfolio_accepted": True,

        "portfolio_rejection_reason": None,

        "first_entry": {
            "date": date,
            "price": round(
                price,
                4
            ),
            "position_added": 0.33
        },

        "second_entry": None,

        "third_entry": None,

        "exits": [],

        "last_average_price": round(
            price,
            4
        ),

        "final_exit_price": None,

        "exit_date": None,

        "profit_pct": None,

        "max_position": 0.33,

        "peak_profit": 0.0,

        "exit_reason": None
    }


# ============================================================
# CLOSE TRADE
# ============================================================

def close_trade(
    trade,
    date,
    price,
    final_profit,
    reason
):

    trade["status"] = "CLOSED"

    trade["final_exit_price"] = round(
        price,
        4
    )

    trade["exit_date"] = date

    trade["profit_pct"] = round(
        final_profit,
        2
    )

    trade["exit_reason"] = reason

    return trade


# ============================================================
# BACKTEST ONE STOCK
# ============================================================

def backtest_stock(
    symbol,
    df
):

    df = prepare_data(df)

    state = {

        "cycle": 1,

        "position": 0.0,

        "avg_price": 0.0,

        "peak_profit": 0.0,

        "realized_pnl_tracker": []
    }

    trades = []

    signals = []

    for i in range(
        MIN_BARS,
        len(df)
    ):

        row = df.iloc[i]

        date = df.index[i].strftime(
            "%Y-%m-%d"
        )

        price = float(
            row["Close"]
        )

        rsi_val = float(
            row["RSI"]
        )

        ema100 = float(
            row["EMA100"]
        )

        if (
            pd.isna(rsi_val)
            or
            pd.isna(ema100)
        ):
            continue


        position = state["position"]

        avg_price = state["avg_price"]


        # ====================================================
        # RUN-UP FILTER
        # ====================================================

        lookback = min(
            len(df.iloc[:i + 1]),
            RUNUP_LOOKBACK
        )

        recent = df.iloc[
            i + 1 - lookback:
            i + 1
        ]

        lowest_price = float(
            recent["Low"].min()
        )

        highest_price = float(
            recent["High"].max()
        )

        if lowest_price > 0:

            run_up_percent = (
                (
                    highest_price
                    - lowest_price
                )
                /
                lowest_price
            ) * 100

        else:

            run_up_percent = 0.0

        safe_to_buy = (
            run_up_percent
            <= MAX_RUNUP_PERCENT
        )


        # ====================================================
        # GAP FILTER
        # ====================================================

        if i < 3:
            continue

        gap1 = (
            (
                df["Open"].iloc[i]
                -
                df["Close"].iloc[i - 1]
            )
            /
            df["Close"].iloc[i - 1]
        ) * 100

        gap2 = (
            (
                df["Open"].iloc[i - 1]
                -
                df["Close"].iloc[i - 2]
            )
            /
            df["Close"].iloc[i - 2]
        ) * 100

        gap3 = (
            (
                df["Open"].iloc[i - 2]
                -
                df["Close"].iloc[i - 3]
            )
            /
            df["Close"].iloc[i - 3]
        ) * 100

        no_gap_down = (
            gap1 > MAX_GAP_DOWN_PERCENT
            and
            gap2 > MAX_GAP_DOWN_PERCENT
            and
            gap3 > MAX_GAP_DOWN_PERCENT
        )


        # ====================================================
        # EMA TREND FILTER
        # ====================================================

        ema_up = (

            df["EMA100"].iloc[i]
            >
            df["EMA100"].iloc[i - 5]

            and

            df["EMA100"].iloc[i - 5]
            >
            df["EMA100"].iloc[i - 10]

            and

            df["EMA100"].iloc[i]
            >
            df["EMA100"].iloc[i - 10]
            * 1.002

            and

            price
            <=
            df["EMA100"].iloc[i]
            * 1.07
        )


        # ====================================================
        # BUY CONDITIONS
        # ====================================================

        buy1 = (

            safe_to_buy

            and
            ema_up

            and
            no_gap_down

            and
            rsi_val <= BUY1_RSI
        )

        buy2 = (

            safe_to_buy

            and
            ema_up

            and
            no_gap_down

            and
            rsi_val <= BUY2_RSI
        )

        buy3 = (

            safe_to_buy

            and
            ema_up

            and
            no_gap_down

            and
            rsi_val <= BUY3_RSI
        )


        # ====================================================
        # CURRENT PROFIT
        # ====================================================

        profit = 0.0

        if avg_price > 0:

            profit = (
                (
                    price
                    - avg_price
                )
                /
                avg_price
            ) * 100


        # ====================================================
        # SELL CONDITIONS
        # ====================================================

        sell1 = (

            position > 0.70

            and
            rsi_val >= SELL1_RSI

            and
            profit > SELL1_MIN_PROFIT
        )

        sell2 = (

            position > SELL2_MIN_POSITION

            and
            position <= SELL2_MAX_POSITION

            and
            rsi_val >= SELL2_RSI

            and
            profit > SELL2_MIN_PROFIT
        )

        sell3 = (

            position > 0

            and
            rsi_val >= SELL3_RSI

            and
            profit > SELL3_MIN_PROFIT
        )

        action = None


        # ====================================================
        # BUY L1
        # ====================================================

        if position == 0 and buy1:

            state["position"] = 0.33

            state["avg_price"] = price

            state["peak_profit"] = 0.0

            state["realized_pnl_tracker"] = []

            profit = 0.0

            action = "BUY L1"

            trade = create_trade(
                symbol,
                state["cycle"],
                date,
                price
            )

            trades.append(
                trade
            )

            signals.append({

                "symbol": symbol,

                "date": date,

                "action": "BUY L1",

                "price": round(
                    price,
                    4
                ),

                "rsi": round(
                    rsi_val,
                    2
                ),

                "position": 0.33
            })


        # ====================================================
        # BUY L2
        # ====================================================

        elif (

            0.32 < position < 0.50

            and
            buy2

            and
            price < avg_price * 0.97
        ):

            old_pos = position

            state["position"] = 0.66

            state["avg_price"] = update_avg(

                avg_price,

                old_pos,

                price,

                state["position"]
            )

            profit = (

                (
                    price
                    - state["avg_price"]
                )
                /
                state["avg_price"]
            ) * 100

            action = "BUY L2"

            trade = trades[-1]

            trade["second_entry"] = {

                "date": date,

                "price": round(
                    price,
                    4
                ),

                "position_added": 0.33
            }

            trade["last_average_price"] = round(

                state["avg_price"],

                4
            )

            trade["max_position"] = max(

                trade["max_position"],

                0.66
            )

            signals.append({

                "symbol": symbol,

                "date": date,

                "action": "BUY L2",

                "price": round(
                    price,
                    4
                ),

                "rsi": round(
                    rsi_val,
                    2
                ),

                "position": 0.66,

                "avg_price": round(
                    state["avg_price"],
                    4
                )
            })


        # ====================================================
        # BUY L3
        # ====================================================

        elif (

            0.65 < position < 1.0

            and
            buy3

            and
            price < avg_price * 0.94
        ):

            old_pos = position

            state["position"] = 1.0

            state["avg_price"] = update_avg(

                avg_price,

                old_pos,

                price,

                state["position"]
            )

            profit = (

                (
                    price
                    - state["avg_price"]
                )
                /
                state["avg_price"]
            ) * 100

            action = "BUY L3"

            trade = trades[-1]

            trade["third_entry"] = {

                "date": date,

                "price": round(
                    price,
                    4
                ),

                "position_added": 0.34
            }

            trade["last_average_price"] = round(

                state["avg_price"],

                4
            )

            trade["max_position"] = 1.0

            signals.append({

                "symbol": symbol,

                "date": date,

                "action": "BUY L3",

                "price": round(
                    price,
                    4
                ),

                "rsi": round(
                    rsi_val,
                    2
                ),

                "position": 1.0,

                "avg_price": round(
                    state["avg_price"],
                    4
                )
            })


        # ====================================================
        # UPDATE PEAK PROFIT
        # ====================================================

        if profit > state["peak_profit"]:

            state["peak_profit"] = profit


        if trades:

            active_trade = trades[-1]

            if active_trade["status"] == "OPEN":

                active_trade["peak_profit"] = max(

                    active_trade["peak_profit"],

                    state["peak_profit"]
                )


        initial_pos = position


        # ====================================================
        # EXIT MANAGEMENT
        # ====================================================

        if (

            initial_pos > 0

            and
            state["position"] > 0
        ):

            stop_loss_triggered = False

            trailing_stop_triggered = False


            # =================================================
            # STOP LOSS
            # =================================================

            if (

                state["position"] <= 0.33

                and
                profit <= STOP_L1
            ):

                stop_loss_triggered = True

            elif (

                state["position"] <= 0.66

                and
                profit <= STOP_L2
            ):

                stop_loss_triggered = True

            elif (

                state["position"] == 1.0

                and
                profit <= STOP_L3
            ):

                stop_loss_triggered = True


            # =================================================
            # TRAILING STOP
            # =================================================

            if (

                state["peak_profit"]
                >
                TRAILING_TRIGGER

                and

                (
                    state["peak_profit"]
                    -
                    profit
                )
                >=
                TRAILING_GIVEBACK
            ):

                trailing_stop_triggered = True


            # =================================================
            # TRAILING STOP EXIT
            # =================================================

            if (

                trailing_stop_triggered

                and
                not stop_loss_triggered
            ):

                action = "TRAILING STOP"

                final_profit = calculate_final_pnl(

                    state[
                        "realized_pnl_tracker"
                    ],

                    state["position"],

                    profit
                )

                trade = trades[-1]

                close_trade(

                    trade,

                    date,

                    price,

                    final_profit,

                    "TRAILING STOP"
                )

                trade[
                    "stop_trigger_profit"
                ] = round(
                    profit,
                    2
                )

                state["position"] = 0.0

                signals.append({

                    "symbol": symbol,

                    "date": date,

                    "action":
                        "TRAILING STOP",

                    "price": round(
                        price,
                        4
                    ),

                    "rsi": round(
                        rsi_val,
                        2
                    ),

                    "profit": round(
                        final_profit,
                        2
                    )
                })


            # =================================================
            # HARD STOP LOSS
            # =================================================

            elif stop_loss_triggered:

                action = "STOP LOSS"

                final_profit = calculate_final_pnl(

                    state[
                        "realized_pnl_tracker"
                    ],

                    state["position"],

                    profit
                )

                trade = trades[-1]

                close_trade(

                    trade,

                    date,

                    price,

                    final_profit,

                    "STOP LOSS"
                )

                trade[
                    "stop_trigger_profit"
                ] = round(
                    profit,
                    2
                )

                state["position"] = 0.0

                signals.append({

                    "symbol": symbol,

                    "date": date,

                    "action":
                        "STOP LOSS",

                    "price": round(
                        price,
                        4
                    ),

                    "rsi": round(
                        rsi_val,
                        2
                    ),

                    "profit": round(
                        final_profit,
                        2
                    )
                })


            # =================================================
            # FULL EXIT
            # =================================================

            elif sell3:

                action = "EXIT FULL"

                final_profit = calculate_final_pnl(

                    state[
                        "realized_pnl_tracker"
                    ],

                    state["position"],

                    profit
                )

                trade = trades[-1]

                close_trade(

                    trade,

                    date,

                    price,

                    final_profit,

                    "EXIT FULL"
                )

                state["position"] = 0.0

                signals.append({

                    "symbol": symbol,

                    "date": date,

                    "action":
                        "EXIT FULL",

                    "price": round(
                        price,
                        4
                    ),

                    "rsi": round(
                        rsi_val,
                        2
                    ),

                    "profit": round(
                        final_profit,
                        2
                    )
                })


            # =================================================
            # SELL L2
            # =================================================

            elif sell2:

                sell_amount = min(

                    0.33,

                    state["position"]
                )

                state[
                    "realized_pnl_tracker"
                ].append(

                    (
                        sell_amount,

                        profit
                    )
                )

                state["position"] = round(

                    state["position"]
                    -
                    sell_amount,

                    2
                )

                action = "SELL L2"

                trade = trades[-1]

                trade["exits"].append({

                    "date": date,

                    "type": "SELL L2",

                    "price": round(
                        price,
                        4
                    ),

                    "position_sold": round(
                        sell_amount,
                        2
                    ),

                    "profit": round(
                        profit,
                        2
                    )
                })

                signals.append({

                    "symbol": symbol,

                    "date": date,

                    "action":
                        "SELL L2",

                    "price": round(
                        price,
                        4
                    ),

                    "rsi": round(
                        rsi_val,
                        2
                    ),

                    "profit": round(
                        profit,
                        2
                    ),

                    "position":
                        state["position"]
                })

                if state["position"] == 0:

                    final_profit = calculate_final_pnl(

                        state[
                            "realized_pnl_tracker"
                        ],

                        0,

                        profit
                    )

                    close_trade(

                        trade,

                        date,

                        price,

                        final_profit,

                        "SELL L2"
                    )


            # =================================================
            # SELL L1
            # =================================================

            elif sell1:

                sell_amount = min(

                    0.33,

                    state["position"]
                )

                state[
                    "realized_pnl_tracker"
                ].append(

                    (
                        sell_amount,

                        profit
                    )
                )

                state["position"] = round(

                    state["position"]
                    -
                    sell_amount,

                    2
                )

                action = "SELL L1"

                trade = trades[-1]

                trade["exits"].append({

                    "date": date,

                    "type": "SELL L1",

                    "price": round(
                        price,
                        4
                    ),

                    "position_sold": round(
                        sell_amount,
                        2
                    ),

                    "profit": round(
                        profit,
                        2
                    )
                })

                signals.append({

                    "symbol": symbol,

                    "date": date,

                    "action":
                        "SELL L1",

                    "price": round(
                        price,
                        4
                    ),

                    "rsi": round(
                        rsi_val,
                        2
                    ),

                    "profit": round(
                        profit,
                        2
                    ),

                    "position":
                        state["position"]
                })

                if state["position"] == 0:

                    final_profit = calculate_final_pnl(

                        state[
                            "realized_pnl_tracker"
                        ],

                        0,

                        profit
                    )

                    close_trade(

                        trade,

                        date,

                        price,

                        final_profit,

                        "SELL L1"
                    )


        # ====================================================
        # RESET CYCLE
        # ====================================================

        if (

            state["position"] == 0

            and

            action in {

                "SELL L1",

                "SELL L2",

                "EXIT FULL",

                "STOP LOSS",

                "TRAILING STOP"
            }
        ):

            state["avg_price"] = 0.0

            state["peak_profit"] = 0.0

            state[
                "realized_pnl_tracker"
            ] = []

            state["cycle"] += 1


    # ========================================================
    # END OF DATA
    # ========================================================

    if (

        state["position"] > 0

        and
        trades
    ):

        trade = trades[-1]

        if trade["status"] == "OPEN":

            last_price = float(
                df["Close"].iloc[-1]
            )

            last_date = df.index[-1].strftime(
                "%Y-%m-%d"
            )

            current_profit = (

                (
                    last_price
                    -
                    state["avg_price"]
                )
                /
                state["avg_price"]
            ) * 100

            final_profit = calculate_final_pnl(

                state[
                    "realized_pnl_tracker"
                ],

                state["position"],

                current_profit
            )

            close_trade(

                trade,

                last_date,

                last_price,

                final_profit,

                "END_OF_DATA"
            )

            trade["status"] = "OPEN"

    return trades, signals


# ============================================================
# RUN ALL STOCKS
# ============================================================

all_trades = []
all_signals = []

print("\nRUNNING BACKTEST...\n")

for symbol in SYMBOLS:

    df = fetch_local_data(
        symbol
    )

    if df is None:

        print(
            f"{symbol:8} | NO DATA"
        )

        continue

    if len(df) < MIN_BARS:

        print(

            f"{symbol:8} | "
            f"INSUFFICIENT DATA "
            f"({len(df)})"
        )

        continue

    trades, signals = backtest_stock(

        symbol,

        df
    )

    all_trades.extend(
        trades
    )

    all_signals.extend(
        signals
    )

    closed_count = sum(

        t["status"] == "CLOSED"

        for t in trades
    )

    print(

        f"{symbol:8} | "
        f"{closed_count:3} closed | "
        f"{len(signals):3} signals"
    )


# ============================================================
# SORT ALL TRADES
# ============================================================

all_trades.sort(

    key=lambda x:
    x["first_entry"]["date"]
)

all_signals.sort(

    key=lambda x:
    x["date"]
)


# ============================================================
# RAW CLOSED / OPEN TRADES
# ============================================================

raw_closed_trades = [

    t

    for t in all_trades

    if (

        t["status"] == "CLOSED"

        and
        t["profit_pct"] is not None

        and
        t["exit_reason"]
        !=
        "END_OF_DATA"
    )
]

raw_open_trades = [

    t

    for t in all_trades

    if t["status"] == "OPEN"
]


# ============================================================
# REAL PORTFOLIO CAPACITY FILTER
# ============================================================
#
# IMPORTANT:
#
# The strategy engine generates trades independently
# for every stock.
#
# This layer applies the actual global portfolio limit:
#
# MAX_PORTFOLIO_POSITIONS = 8
#
# A trade occupies ONE portfolio slot from its first
# entry date until its final exit date.
#
# If 8 slots are already occupied when a new trade starts,
# the new trade is rejected.
#
# Existing strategy logic is NOT changed.
# ============================================================

def apply_portfolio_capacity(
    trades,
    max_positions
):

    candidates = sorted(

        trades,

        key=lambda t: (

            t["first_entry"]["date"],

            t["symbol"],

            t["cycle"]
        )
    )

    accepted = []

    rejected = []

    active = []

    max_simultaneous = 0

    capacity_by_date = {}

    for trade in candidates:

        entry_date = pd.Timestamp(
            trade["first_entry"]["date"]
        )

        exit_date = trade.get(
            "exit_date"
        )

        # ----------------------------------------------------
        # Remove positions that already finished
        # before this entry date.
        # ----------------------------------------------------

        still_active = []

        for active_trade in active:

            active_exit = active_trade.get(
                "exit_date"
            )

            if active_exit is None:

                still_active.append(
                    active_trade
                )

                continue

            active_exit_date = pd.Timestamp(
                active_exit
            )

            if active_exit_date >= entry_date:

                still_active.append(
                    active_trade
                )

        active = still_active

        current_active_count = len(
            active
        )

        max_simultaneous = max(

            max_simultaneous,

            current_active_count
        )

        # ----------------------------------------------------
        # Capacity available
        # ----------------------------------------------------

        if current_active_count < max_positions:

            trade[
                "portfolio_accepted"
            ] = True

            trade[
                "portfolio_rejection_reason"
            ] = None

            accepted.append(
                trade
            )

            active.append(
                trade
            )

        else:

            trade[
                "portfolio_accepted"
            ] = False

            trade[
                "portfolio_rejection_reason"
            ] = (
                "MAX_PORTFOLIO_POSITIONS"
            )

            rejected.append(
                trade
            )

        capacity_by_date[
            trade["first_entry"]["date"]
        ] = current_active_count


    # Final maximum

    max_simultaneous = max(

        max_simultaneous,

        len(active)
    )

    return (
        accepted,
        rejected,
        max_simultaneous,
        capacity_by_date
    )


(
    portfolio_closed_trades,
    rejected_trades,
    max_simultaneous_positions,
    capacity_by_date
) = apply_portfolio_capacity(

    raw_closed_trades,

    MAX_PORTFOLIO_POSITIONS
)


# ============================================================
# PORTFOLIO OPEN TRADES
# ============================================================
#
# Open trades at END OF DATA are not included in the
# closed-trade performance statistics.
#
# We still apply the 8-position capacity filter to them.
# ============================================================

accepted_open_trades = []

rejected_open_trades = []

for trade in sorted(

    raw_open_trades,

    key=lambda t:
    (
        t["first_entry"]["date"],
        t["symbol"],
        t["cycle"]
    )
):

    entry_date = pd.Timestamp(
        trade["first_entry"]["date"]
    )

    active_count = 0

    for accepted_trade in (

        portfolio_closed_trades
        +
        accepted_open_trades
    ):

        accepted_entry = pd.Timestamp(
            accepted_trade[
                "first_entry"
            ]["date"]
        )

        accepted_exit = accepted_trade.get(
            "exit_date"
        )

        if accepted_exit is None:
            accepted_exit_date = pd.Timestamp.max

        else:
            accepted_exit_date = pd.Timestamp(
                accepted_exit
            )

        if (

            accepted_entry <= entry_date

            and
            accepted_exit_date >= entry_date
        ):

            active_count += 1

    if active_count < MAX_PORTFOLIO_POSITIONS:

        trade[
            "portfolio_accepted"
        ] = True

        trade[
            "portfolio_rejection_reason"
        ] = None

        accepted_open_trades.append(
            trade
        )

    else:

        trade[
            "portfolio_accepted"
        ] = False

        trade[
            "portfolio_rejection_reason"
        ] = (
            "MAX_PORTFOLIO_POSITIONS"
        )

        rejected_open_trades.append(
            trade
        )


# ============================================================
# FINAL PORTFOLIO TRADE SET
# ============================================================

closed_trades = portfolio_closed_trades

open_trades = [

    t

    for t in accepted_open_trades
]


# ============================================================
# BASIC STATISTICS
# ============================================================

profits = [

    float(
        t["profit_pct"]
    )

    for t in closed_trades
]


wins = [

    p

    for p in profits

    if p > 0
]


losses = [

    p

    for p in profits

    if p <= 0
]


total_trades = len(
    profits
)

winning_trades = len(
    wins
)

losing_trades = len(
    losses
)


win_rate = (

    winning_trades
    /
    total_trades
    *
    100

    if total_trades

    else 0
)


sum_profit = sum(
    profits
)


average_profit = (

    float(
        np.mean(profits)
    )

    if profits

    else 0
)


average_win = (

    float(
        np.mean(wins)
    )

    if wins

    else 0
)


average_loss = (

    float(
        np.mean(losses)
    )

    if losses

    else 0
)


# ============================================================
# PROFIT FACTOR
# ============================================================

gross_profit = sum(

    p

    for p in profits

    if p > 0
)


gross_loss = abs(

    sum(

        p

        for p in profits

        if p < 0
    )
)


profit_factor = (

    gross_profit
    /
    gross_loss

    if gross_loss > 0

    else 0
)


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================
#
# Each accepted trade represents one portfolio slot.
#
# POSITION_SIZE = 12.5%
#
# The return from every completed trade is applied only
# to its allocated portfolio fraction.
#
# Trades are processed by actual EXIT DATE.
#
# This is now capacity-constrained, unlike v1.1.
# ============================================================

portfolio = INITIAL_CAPITAL

equity_curve = []

peak_equity = portfolio

max_drawdown = 0.0

losing_streak = 0

max_losing_streak = 0

winning_streak = 0

max_winning_streak = 0


portfolio_trade_results = sorted(

    closed_trades,

    key=lambda t: (

        t["exit_date"],

        t["symbol"],

        t["cycle"]
    )
)


for trade in portfolio_trade_results:

    trade_return = (

        trade["profit_pct"]
        /
        100
    ) * POSITION_SIZE


    portfolio_before = portfolio


    portfolio *= (

        1
        +
        trade_return
    )


    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    peak_equity = max(

        peak_equity,

        portfolio
    )


    drawdown = (

        (
            peak_equity
            -
            portfolio
        )
        /
        peak_equity
    ) * 100


    max_drawdown = max(

        max_drawdown,

        drawdown
    )


    # --------------------------------------------------------
    # Losing / winning streak
    # --------------------------------------------------------

    if trade["profit_pct"] < 0:

        losing_streak += 1

        winning_streak = 0

    else:

        winning_streak = (
            winning_streak
            + 1
        )

        losing_streak = 0


    max_losing_streak = max(

        max_losing_streak,

        losing_streak
    )


    max_winning_streak = max(

        max_winning_streak,

        winning_streak
    )


    equity_curve.append({

        "date":
            trade["exit_date"],

        "symbol":
            trade["symbol"],

        "profit_pct":
            trade["profit_pct"],

        "portfolio_return_percent":
            round(
                trade_return * 100,
                4
            ),

        "portfolio_value_before":
            round(
                portfolio_before,
                2
            ),

        "portfolio_value":
            round(
                portfolio,
                2
            ),

        "portfolio_drawdown_percent":
            round(
                drawdown,
                2
            )
    })


compound_return = (

    (
        portfolio
        /
        INITIAL_CAPITAL
    )
    -
    1
) * 100


# ============================================================
# SIMPLE ROBUSTNESS / BAD MARKET CHECK
# ============================================================

sorted_profits = sorted(
    profits
)


worst_10_count = (

    max(
        1,
        int(
            len(sorted_profits)
            * 0.10
        )
    )

    if sorted_profits

    else 0
)


worst_20_count = (

    max(
        1,
        int(
            len(sorted_profits)
            * 0.20
        )
    )

    if sorted_profits

    else 0
)


worst_10 = (

    sorted_profits[
        :worst_10_count
    ]

    if sorted_profits

    else []
)


worst_20 = (

    sorted_profits[
        :worst_20_count
    ]

    if sorted_profits

    else []
)


worst_10_average_loss = (

    float(
        np.mean(worst_10)
    )

    if worst_10

    else 0
)


worst_20_average_loss = (

    float(
        np.mean(worst_20)
    )

    if worst_20

    else 0
)


worst_trade_percent = (

    min(profits)

    if profits

    else 0
)


# ============================================================
# STRESS DRAWDOWN
# ============================================================
#
# Theoretical scenario:
# Worst 8 historical losing trades happen together.
#
# This remains a stress test and is NOT the same as
# chronological portfolio Max Drawdown.
# ============================================================

worst_losses = sorted(

    [
        p

        for p in profits

        if p < 0
    ]

)[:MAX_PORTFOLIO_POSITIONS]


if worst_losses:

    stress_drawdown = (

        abs(
            sum(
                worst_losses
            )
        )
        *
        POSITION_SIZE
    )

else:

    stress_drawdown = 0.0


# ============================================================
# EXIT ANALYSIS
# ============================================================

exit_analysis = {}


for trade in closed_trades:

    reason = trade[
        "exit_reason"
    ]

    exit_analysis[reason] = (

        exit_analysis.get(
            reason,
            0
        )
        +
        1
    )


# ============================================================
# LADDER ANALYSIS
# ============================================================

buy_l1_count = sum(

    1

    for s in all_signals

    if s["action"] == "BUY L1"
)


buy_l2_count = sum(

    1

    for s in all_signals

    if s["action"] == "BUY L2"
)


buy_l3_count = sum(

    1

    for s in all_signals

    if s["action"] == "BUY L3"
)


sell_l1_count = sum(

    1

    for s in all_signals

    if s["action"] == "SELL L1"
)


sell_l2_count = sum(

    1

    for s in all_signals

    if s["action"] == "SELL L2"
)


full_exit_count = sum(

    1

    for s in all_signals

    if s["action"] == "EXIT FULL"
)


stop_count = sum(

    1

    for s in all_signals

    if s["action"] == "STOP LOSS"
)


trailing_stop_count = sum(

    1

    for s in all_signals

    if s["action"] == "TRAILING STOP"
)


l3_completed = sum(

    1

    for trade in closed_trades

    if trade[
        "third_entry"
    ] is not None
)


# ============================================================
# BEST / WORST TRADE
# ============================================================

best_trade = (

    max(

        closed_trades,

        key=lambda x:
        x["profit_pct"]
    )

    if closed_trades

    else None
)


worst_trade = (

    min(

        closed_trades,

        key=lambda x:
        x["profit_pct"]
    )

    if closed_trades

    else None
)


# ============================================================
# STOCK SUMMARY
# ============================================================

stock_summary = {}


for symbol in SYMBOLS:

    stock_trades = [

        t

        for t in closed_trades

        if t["symbol"] == symbol
    ]


    if not stock_trades:
        continue


    stock_profits = [

        float(
            t["profit_pct"]
        )

        for t in stock_trades
    ]


    stock_wins = [

        p

        for p in stock_profits

        if p > 0
    ]


    stock_losses = [

        p

        for p in stock_profits

        if p <= 0
    ]


    stock_summary[symbol] = {

        "trades":
            len(stock_profits),

        "wins":
            len(stock_wins),

        "losses":
            len(stock_losses),

        "win_rate_percent":
            round(

                len(stock_wins)
                /
                len(stock_profits)
                *
                100,

                2
            ),

        "sum_profit_percent":
            round(

                sum(
                    stock_profits
                ),

                2
            ),

        "average_profit_percent":
            round(

                float(
                    np.mean(
                        stock_profits
                    )
                ),

                2
            ),

        "average_win_percent":
            round(

                float(
                    np.mean(
                        stock_wins
                    )
                ),

                2
            )

            if stock_wins

            else 0,

        "average_loss_percent":
            round(

                float(
                    np.mean(
                        stock_losses
                    )
                ),

                2
            )

            if stock_losses

            else 0
    }


# ============================================================
# PORTFOLIO CAPACITY STATISTICS
# ============================================================

rejected_trade_count = len(
    rejected_trades
)

rejected_open_count = len(
    rejected_open_trades
)

total_rejected = (

    rejected_trade_count
    +
    rejected_open_count
)


# ============================================================
# RESULT
# ============================================================

result = {

    "strategy":
        "EGX Ladder Cycle System v3.4 Optimized",

    "backtest_version":
        "Full historical state-machine backtest v1.2 + portfolio capacity constraint",

    "description":
        "Historical simulation matching the live Ladder Strategy v3.4 logic with a global 8-position portfolio capacity layer.",

    "data_file":
        DB_FILE,

    "parameters": {

        "rsi_period":
            RSI_PERIOD,

        "ema100_period":
            EMA100_PERIOD,

        "runup_lookback":
            RUNUP_LOOKBACK,

        "max_runup_percent":
            MAX_RUNUP_PERCENT,

        "max_gap_down_percent":
            MAX_GAP_DOWN_PERCENT,

        "buy1_rsi":
            BUY1_RSI,

        "buy2_rsi":
            BUY2_RSI,

        "buy3_rsi":
            BUY3_RSI,

        "sell1_rsi":
            SELL1_RSI,

        "sell1_min_profit":
            SELL1_MIN_PROFIT,

        "sell2_rsi":
            SELL2_RSI,

        "sell2_min_profit":
            SELL2_MIN_PROFIT,

        "sell3_rsi":
            SELL3_RSI,

        "sell3_min_profit":
            SELL3_MIN_PROFIT,

        "stop_l1":
            STOP_L1,

        "stop_l2":
            STOP_L2,

        "stop_l3":
            STOP_L3,

        "trailing_trigger":
            TRAILING_TRIGGER,

        "trailing_giveback":
            TRAILING_GIVEBACK,

        "max_portfolio_positions":
            MAX_PORTFOLIO_POSITIONS,

        "position_size_percent":
            round(
                POSITION_SIZE * 100,
                2
            )
    },


    "portfolio_constraints": {

        "max_positions":
            MAX_PORTFOLIO_POSITIONS,

        "position_size_percent":
            round(
                POSITION_SIZE * 100,
                2
            ),

        "max_simultaneous_positions":
            max_simultaneous_positions,

        "rejected_closed_trades":
            rejected_trade_count,

        "rejected_open_trades":
            rejected_open_count,

        "total_rejected_trades":
            total_rejected
    },


    "statistics": {

        "total_closed_trades":
            total_trades,

        "winning_trades":
            winning_trades,

        "losing_trades":
            losing_trades,

        "win_rate_percent":
            round(
                win_rate,
                2
            ),

        "sum_trade_profit_percent":
            round(
                sum_profit,
                2
            ),

        "average_trade_profit_percent":
            round(
                average_profit,
                2
            ),

        "average_win_percent":
            round(
                average_win,
                2
            ),

        "average_loss_percent":
            round(
                average_loss,
                2
            ),

        "gross_profit":
            round(
                gross_profit,
                2
            ),

        "gross_loss":
            round(
                gross_loss,
                2
            ),

        "profit_factor":
            round(
                profit_factor,
                2
            ),

        "realistic_compound_return_percent":
            round(
                compound_return,
                2
            ),

        "maximum_drawdown_percent":
            round(
                max_drawdown,
                2
            ),

        "stress_drawdown_percent":
            round(
                stress_drawdown,
                2
            ),

        "worst_10_percent_average_loss":
            round(
                worst_10_average_loss,
                2
            ),

        "worst_20_percent_average_loss":
            round(
                worst_20_average_loss,
                2
            ),

        "worst_trade_percent":
            round(
                worst_trade_percent,
                2
            ),

        "max_losing_streak":
            max_losing_streak,

        "max_winning_streak":
            max_winning_streak,

        "open_positions":
            len(open_trades)
    },


    "ladder_statistics": {

        "buy_l1":
            buy_l1_count,

        "buy_l2":
            buy_l2_count,

        "buy_l3":
            buy_l3_count,

        "sell_l1":
            sell_l1_count,

        "sell_l2":
            sell_l2_count,

        "full_exit":
            full_exit_count,

        "stop_loss":
            stop_count,

        "trailing_stop":
            trailing_stop_count,

        "trades_reaching_l3":
            l3_completed
    },


    "exit_analysis":
        exit_analysis,


    "best_trade":
        best_trade,


    "worst_trade":
        worst_trade,


    "stock_summary":
        stock_summary,


    "open_positions":
        open_trades,


    "rejected_trades":
        rejected_trades,


    "rejected_open_trades":
        rejected_open_trades,


    "portfolio_equity":
        equity_curve,


    "trades":
        all_trades
}


# ============================================================
# SAVE RESULTS
# ============================================================

with open(

    RESULT_FILE,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        result,

        f,

        ensure_ascii=False,

        indent=2
    )


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


with open(

    STOCK_SUMMARY_FILE,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        stock_summary,

        f,

        ensure_ascii=False,

        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("FINAL BACKTEST RESULTS")
print("=" * 80)

print(
    f"Closed Trades       : "
    f"{total_trades}"
)

print(
    f"Winners             : "
    f"{winning_trades}"
)

print(
    f"Losers              : "
    f"{losing_trades}"
)

print(
    f"Win Rate            : "
    f"{win_rate:.2f}%"
)

print(
    f"Sum Profit          : "
    f"{sum_profit:.2f}%"
)

print(
    f"Average Trade       : "
    f"{average_profit:.2f}%"
)

print(
    f"Average Win         : "
    f"{average_win:.2f}%"
)

print(
    f"Average Loss        : "
    f"{average_loss:.2f}%"
)

print(
    f"Profit Factor       : "
    f"{profit_factor:.2f}"
)

print(
    f"Compound Return     : "
    f"{compound_return:.2f}%"
)

print(
    f"Maximum Drawdown    : "
    f"{max_drawdown:.2f}%"
)

print(
    f"Stress Drawdown     : "
    f"{stress_drawdown:.2f}%"
)

print(
    f"Worst 10% Avg Loss  : "
    f"{worst_10_average_loss:.2f}%"
)

print(
    f"Worst 20% Avg Loss  : "
    f"{worst_20_average_loss:.2f}%"
)

print(
    f"Worst Trade         : "
    f"{worst_trade_percent:.2f}%"
)

print(
    f"Max Losing Streak   : "
    f"{max_losing_streak}"
)

print(
    f"Max Winning Streak  : "
    f"{max_winning_streak}"
)

print(
    f"Open Positions      : "
    f"{len(open_trades)}"
)

print("\n" + "-" * 80)
print("PORTFOLIO CAPACITY")
print("-" * 80)

print(
    f"Max Positions       : "
    f"{MAX_PORTFOLIO_POSITIONS}"
)

print(
    f"Position Size       : "
    f"{POSITION_SIZE * 100:.2f}%"
)

print(
    f"Max Simultaneous    : "
    f"{max_simultaneous_positions}"
)

print(
    f"Rejected Closed     : "
    f"{rejected_trade_count}"
)

print(
    f"Rejected Open       : "
    f"{rejected_open_count}"
)

print(
    f"Total Rejected      : "
    f"{total_rejected}"
)

print("\n" + "-" * 80)
print("LADDER STATISTICS")
print("-" * 80)

print(
    f"BUY L1              : "
    f"{buy_l1_count}"
)

print(
    f"BUY L2              : "
    f"{buy_l2_count}"
)

print(
    f"BUY L3              : "
    f"{buy_l3_count}"
)

print(
    f"SELL L1             : "
    f"{sell_l1_count}"
)

print(
    f"SELL L2             : "
    f"{sell_l2_count}"
)

print(
    f"FULL EXIT            : "
    f"{full_exit_count}"
)

print(
    f"STOP LOSS           : "
    f"{stop_count}"
)

print(
    f"TRAILING STOP       : "
    f"{trailing_stop_count}"
)

print(
    f"Trades Reaching L3  : "
    f"{l3_completed}"
)

print("\n" + "=" * 80)
print("BACKTEST COMPLETED")
print("=" * 80)

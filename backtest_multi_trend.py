print("=" * 72)
print("EGX RESISTANCE BREAKOUT + VOLUME STRATEGY v1.0")
print("CONFIRMED SWING RESISTANCE + VOLUME CONFIRMATION")
print("=" * 72)

import json
import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DB_FILE = "egx_history_database_v2.json"

RESULT_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

INITIAL_CAPITAL = 100000.0

MAX_POSITIONS = 8
POSITION_SIZE = 1.0 / MAX_POSITIONS


# ============================================================
# SWING DETECTION
# ============================================================

PIVOT_LEFT = 3
PIVOT_RIGHT = 3


# ============================================================
# HISTORICAL RESISTANCE
# ============================================================

LOOKBACK = 120

# Two resistance pivots within this distance
# become one resistance zone.
ZONE_DISTANCE_PERCENT = 1.0

# Minimum confirmed reaction required
# to consider resistance valid.
MIN_RESISTANCE_REACTIONS = 1


# ============================================================
# RESISTANCE REACTION
# ============================================================

REACTION_WINDOW = 3

# Minimum downward reaction after swing high.
MIN_REACTION_PERCENT = 1.0


# ============================================================
# BREAKOUT
# ============================================================

# Price must close above the resistance zone
# by at least this percentage.
BREAKOUT_PERCENT = 0.20

# Minimum distance from entry to next resistance.
MIN_UPSIDE_PERCENT = 5.0


# ============================================================
# VOLUME CONFIRMATION
# ============================================================

# Number of previous candles used to calculate
# average volume.
VOLUME_LOOKBACK = 20

# Breakout volume must be >= average volume
# multiplied by this factor.
VOLUME_MULTIPLIER = 1.50


# ============================================================
# STOP LOSS
# ============================================================

# Stop is placed below broken resistance.
STOP_BUFFER_PERCENT = 0.50


# ============================================================
# BACKTEST
# ============================================================

MIN_BARS = 60


# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(DB_FILE)

with open(DB_FILE, encoding="utf-8") as f:
    db = json.load(f)

print(f"Database: {len(db)} symbols")


# ============================================================
# DATA CONVERSION
# ============================================================

def to_df(x):

    if isinstance(x, dict) and "data" in x and "columns" in x:

        rows = []

        for d, v in x["data"].items():

            if isinstance(v, dict):
                r = v.copy()
            else:
                r = dict(zip(x["columns"], v))

            r["Date"] = d
            rows.append(r)

        x = rows

    if not isinstance(x, list):
        return None

    df = pd.DataFrame(x)

    if df.empty:
        return None

    # Normalize column names
    df.columns = [
        str(c).strip().capitalize()
        for c in df.columns
    ]

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close"
    ]

    if not all(c in df.columns for c in required):
        return None

    # Volume is required for this strategy.
    if "Volume" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    for c in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = (
        df
        .dropna(
            subset=[
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        )
        .drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# SWING DETECTION
# ============================================================

def detect_pivots(df):
    """
    Detect swing highs/lows.

    IMPORTANT:
    Pivot at index i becomes known only at:

        i + PIVOT_RIGHT

    Therefore it cannot be used before confirmation.
    """

    lows = df["Low"].values
    highs = df["High"].values

    n = len(df)

    pivot_lows = []
    pivot_highs = []

    for i in range(
        PIVOT_LEFT,
        n - PIVOT_RIGHT
    ):

        left_lows = lows[
            i - PIVOT_LEFT:i
        ]

        right_lows = lows[
            i + 1:i + PIVOT_RIGHT + 1
        ]

        left_highs = highs[
            i - PIVOT_LEFT:i
        ]

        right_highs = highs[
            i + 1:i + PIVOT_RIGHT + 1
        ]

        # ----------------------------------------------------
        # Swing Low
        # ----------------------------------------------------

        if (
            lows[i] < left_lows.min()
            and
            lows[i] <= right_lows.min()
        ):

            pivot_lows.append({
                "index": i,
                "confirmed_at": i + PIVOT_RIGHT,
                "price": float(lows[i])
            })

        # ----------------------------------------------------
        # Swing High
        # ----------------------------------------------------

        if (
            highs[i] > left_highs.max()
            and
            highs[i] >= right_highs.max()
        ):

            pivot_highs.append({
                "index": i,
                "confirmed_at": i + PIVOT_RIGHT,
                "price": float(highs[i])
            })

    return pivot_lows, pivot_highs


# ============================================================
# RESISTANCE REACTION
# ============================================================

def pivot_has_resistance_reaction(
    df,
    pivot_index,
    pivot_price
):
    """
    Determines whether a swing high produced
    a meaningful downward reaction.

    Only candles after the pivot are inspected.
    """

    end = min(
        len(df),
        pivot_index + 1 + REACTION_WINDOW
    )

    future = df.iloc[
        pivot_index + 1:end
    ]

    if future.empty:
        return False

    min_close = float(
        future["Close"].min()
    )

    reaction = (
        (pivot_price - min_close)
        / pivot_price
        * 100
    )

    return reaction >= MIN_REACTION_PERCENT


# ============================================================
# BUILD RESISTANCE ZONES
# ============================================================

def build_resistance_zones(
    df,
    confirmed_highs,
    current_index
):
    """
    Build resistance zones using ONLY pivots that were
    already confirmed by current_index.

    No future information is allowed.
    """

    start = max(
        0,
        current_index - LOOKBACK + 1
    )

    highs = [
        p for p in confirmed_highs
        if (
            p["confirmed_at"] <= current_index
            and
            p["index"] >= start
        )
    ]

    # --------------------------------------------------------
    # Cluster pivots
    # --------------------------------------------------------

    clusters = []

    highs = sorted(
        highs,
        key=lambda x: x["index"]
    )

    for p in highs:

        placed = False

        for zone in clusters:

            zone_price = zone["price"]

            distance = (
                abs(
                    p["price"] - zone_price
                )
                / zone_price
                * 100
            )

            if distance <= ZONE_DISTANCE_PERCENT:

                zone["pivots"].append(p)

                prices = [
                    x["price"]
                    for x in zone["pivots"]
                ]

                zone["price"] = float(
                    np.mean(prices)
                )

                placed = True
                break

        if not placed:

            clusters.append({
                "price": float(p["price"]),
                "pivots": [p]
            })

    # --------------------------------------------------------
    # Calculate reactions
    # --------------------------------------------------------

    zones = []

    for zone in clusters:

        reactions = 0

        for p in zone["pivots"]:

            reaction_end = (
                p["index"]
                + 1
                + REACTION_WINDOW
            )

            # Reaction window must already exist.
            if reaction_end > current_index:
                continue

            if pivot_has_resistance_reaction(
                df,
                p["index"],
                p["price"]
            ):

                reactions += 1

        if reactions < MIN_RESISTANCE_REACTIONS:
            continue

        prices = [
            p["price"]
            for p in zone["pivots"]
        ]

        zone_low = min(prices)
        zone_high = max(prices)

        # Expand zone slightly.
        zone_low *= (
            1 - ZONE_DISTANCE_PERCENT / 100
        )

        zone_high *= (
            1 + ZONE_DISTANCE_PERCENT / 100
        )

        zones.append({

            "direction":
                "RESISTANCE",

            "price":
                round(
                    float(np.mean(prices)),
                    4
                ),

            "low":
                round(
                    float(zone_low),
                    4
                ),

            "high":
                round(
                    float(zone_high),
                    4
                ),

            "reactions":
                reactions,

            "first_pivot":
                min(
                    p["index"]
                    for p in zone["pivots"]
                ),

            "last_pivot":
                max(
                    p["index"]
                    for p in zone["pivots"]
                )
        })

    return zones


# ============================================================
# FIND RESISTANCE BELOW CURRENT PRICE
# ============================================================

def find_broken_resistance(
    resistances,
    close
):
    """
    Find the nearest resistance that has just been broken.
    """

    candidates = []

    for resistance in resistances:

        breakout_level = float(
            resistance["high"]
        )

        if close <= (
            breakout_level
            *
            (
                1
                + BREAKOUT_PERCENT / 100
            )
        ):
            continue

        candidates.append(
            resistance
        )

    if not candidates:
        return None

    # Nearest broken resistance.
    candidates.sort(
        key=lambda r:
        r["high"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# FIND NEXT RESISTANCE TARGET
# ============================================================

def find_next_resistance(
    resistances,
    entry_price,
    broken_resistance
):
    """
    Find the first resistance above the actual
    entry price.

    The broken resistance itself cannot be the target.
    """

    candidates = []

    for resistance in resistances:

        resistance_price = float(
            resistance["price"]
        )

        # Must be above entry.
        if resistance_price <= entry_price:
            continue

        # Don't use the same resistance
        # that generated the breakout.
        if (
            resistance["first_pivot"]
            ==
            broken_resistance["first_pivot"]
            and
            resistance["last_pivot"]
            ==
            broken_resistance["last_pivot"]
        ):
            continue

        upside = (
            (resistance_price - entry_price)
            / entry_price
            * 100
        )

        if upside < MIN_UPSIDE_PERCENT:
            continue

        candidates.append(
            resistance
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda r:
        r["price"]
    )

    return candidates[0]


# ============================================================
# VOLUME CONFIRMATION
# ============================================================

def volume_confirmed(
    df,
    index
):
    """
    Breakout candle volume must be at least
    VOLUME_MULTIPLIER times the average volume
    of the previous VOLUME_LOOKBACK candles.

    IMPORTANT:
    Current candle volume is NOT included
    in the average.
    """

    start = max(
        0,
        index - VOLUME_LOOKBACK
    )

    previous_volume = df.iloc[
        start:index
    ]["Volume"]

    if len(previous_volume) < VOLUME_LOOKBACK:
        return False

    average_volume = float(
        previous_volume.mean()
    )

    if average_volume <= 0:
        return False

    current_volume = float(
        df.iloc[index]["Volume"]
    )

    return (
        current_volume
        >=
        average_volume
        * VOLUME_MULTIPLIER
    )


# ============================================================
# BREAKOUT CONFIRMATION
# ============================================================

def valid_breakout(
    row,
    resistance
):
    """
    Confirm that the current candle has genuinely
    broken resistance.

    The close must be above the upper edge
    of the resistance zone.
    """

    close = float(
        row["Close"]
    )

    breakout_level = float(
        resistance["high"]
    )

    required_close = (
        breakout_level
        *
        (
            1
            + BREAKOUT_PERCENT / 100
        )
    )

    return close > required_close


# ============================================================
# CLOSE TRADE
# ============================================================

def close_trade(
    position,
    date,
    price,
    reason
):

    profit = (
        (price - position["entry_price"])
        /
        position["entry_price"]
        * 100
    )

    return {

        "symbol":
            position["symbol"],

        "status":
            "CLOSED",

        "entry_date":
            position["entry_date"],

        "entry_price":
            round(
                position["entry_price"],
                4
            ),

        "exit_date":
            date,

        "exit_price":
            round(
                price,
                4
            ),

        "profit_pct":
            round(
                profit,
                2
            ),

        "exit_reason":
            reason,

        "broken_resistance":
            position[
                "broken_resistance"
            ],

        "resistance_reactions":
            position[
                "resistance_reactions"
            ],

        "target_resistance":
            position[
                "target_resistance"
            ],

        "upside_to_target":
            position[
                "upside_to_target"
            ],

        "breakout_volume_ratio":
            position[
                "breakout_volume_ratio"
            ]
    }


# ============================================================
# BACKTEST ONE SYMBOL
# ============================================================

def backtest(
    sym,
    df
):

    pivot_lows, pivot_highs = detect_pivots(df)

    position = None

    trades = []

    for i in range(
        MIN_BARS,
        len(df)
    ):

        row = df.iloc[i]

        date = row["Date"].strftime(
            "%Y-%m-%d"
        )

        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        # ====================================================
        # BUILD ONLY INFORMATION KNOWN TODAY
        # ====================================================

        resistances = build_resistance_zones(
            df,
            pivot_highs,
            i
        )

        # ====================================================
        # MANAGE OPEN POSITION
        # ====================================================

        if position is not None:

            broken_resistance = position[
                "broken_resistance_zone"
            ]

            target_resistance = position[
                "target_resistance_zone"
            ]

            # ------------------------------------------------
            # STRUCTURAL STOP
            # ------------------------------------------------

            stop_price = (
                broken_resistance["low"]
                *
                (
                    1
                    -
                    STOP_BUFFER_PERCENT / 100
                )
            )

            # ------------------------------------------------
            # STOP HAS PRIORITY
            # ------------------------------------------------

            if close < stop_price:

                trades.append(
                    close_trade(
                        position,
                        date,
                        close,
                        "BREAKOUT_FAILURE"
                    )
                )

                position = None
                continue

            # ------------------------------------------------
            # TARGET
            # ------------------------------------------------

            if target_resistance is not None:

                target = float(
                    target_resistance["price"]
                )

                if high >= target:

                    trades.append(
                        close_trade(
                            position,
                            date,
                            target,
                            "RESISTANCE"
                        )
                    )

                    position = None
                    continue

            continue

        # ====================================================
        # FIND BROKEN RESISTANCE
        # ====================================================

        broken_resistance = find_broken_resistance(
            resistances,
            close
        )

        if broken_resistance is None:
            continue

        # ====================================================
        # CONFIRM BREAKOUT
        # ====================================================

        if not valid_breakout(
            row,
            broken_resistance
        ):
            continue

        # ====================================================
        # VOLUME CONFIRMATION
        # ====================================================

        if not volume_confirmed(
            df,
            i
        ):
            continue

        # ====================================================
        # FIND NEXT RESISTANCE
        # ====================================================

        # At this point the signal is known at
        # today's close.

        # We don't know tomorrow's open yet,
        # therefore first use today's close
        # as a preliminary filter.

        target_resistance = find_next_resistance(
            resistances,
            close,
            broken_resistance
        )

        if target_resistance is None:
            continue

        # ====================================================
        # NEXT DAY ENTRY
        # ====================================================

        if i + 1 >= len(df):
            continue

        next_row = df.iloc[
            i + 1
        ]

        entry_date = next_row[
            "Date"
        ].strftime(
            "%Y-%m-%d"
        )

        entry_price = float(
            next_row["Open"]
        )

        # ----------------------------------------------------
        # Recalculate upside using actual entry.
        # ----------------------------------------------------

        target_price = float(
            target_resistance["price"]
        )

        actual_upside = (
            (target_price - entry_price)
            /
            entry_price
            * 100
        )

        if actual_upside < MIN_UPSIDE_PERCENT:
            continue

        # ----------------------------------------------------
        # Volume ratio for reporting.
        # ----------------------------------------------------

        volume_start = (
            i - VOLUME_LOOKBACK
        )

        average_volume = float(
            df.iloc[
                volume_start:i
            ]["Volume"].mean()
        )

        current_volume = float(
            df.iloc[i]["Volume"]
        )

        volume_ratio = (
            current_volume
            /
            average_volume
            if average_volume > 0
            else 0
        )

        # ====================================================
        # CREATE POSITION
        # ====================================================

        position = {

            "symbol":
                sym,

            "entry_date":
                entry_date,

            "entry_price":
                entry_price,

            "broken_resistance_zone":
                broken_resistance,

            "broken_resistance":
                round(
                    float(
                        broken_resistance[
                            "price"
                        ]
                    ),
                    4
                ),

            "resistance_reactions":
                broken_resistance[
                    "reactions"
                ],

            "target_resistance_zone":
                target_resistance,

            "target_resistance":
                round(
                    target_price,
                    4
                ),

            "upside_to_target":
                round(
                    actual_upside,
                    2
                ),

            "breakout_volume_ratio":
                round(
                    volume_ratio,
                    2
                )
        }

    # ========================================================
    # CLOSE OPEN POSITION AT END OF DATA
    # ========================================================

    if position is not None:

        last = df.iloc[-1]

        last_date = last[
            "Date"
        ].strftime(
            "%Y-%m-%d"
        )

        last_close = float(
            last["Close"]
        )

        trade = close_trade(
            position,
            last_date,
            last_close,
            "END_OF_DATA"
        )

        trade["status"] = "OPEN"

        trades.append(
            trade
        )

    return trades


# ============================================================
# RUN ALL SYMBOLS
# ============================================================

all_trades = []

for sym, data in db.items():

    if sym.upper() in {
        "EGX30",
        "EGX70",
        "EGX100"
    }:
        continue

    df = to_df(data)

    if df is None:

        print(
            f"⚠️ {sym}: invalid data "
            "or Volume missing"
        )

        continue

    if len(df) < MIN_BARS:

        print(
            f"⚠️ {sym}: insufficient data "
            f"({len(df)})"
        )

        continue

    trades = backtest(
        sym,
        df
    )

    all_trades.extend(
        trades
    )

    closed_count = sum(
        t["status"] == "CLOSED"
        for t in trades
    )

    print(
        f"{sym:8} | "
        f"{closed_count:3} closed"
    )


# ============================================================
# SORT TRADES
# ============================================================

all_trades.sort(
    key=lambda x:
    x["entry_date"]
)

closed = [
    t for t in all_trades
    if t["status"] == "CLOSED"
]

opened = [
    t for t in all_trades
    if t["status"] == "OPEN"
]


# ============================================================
# BASIC STATISTICS
# ============================================================

profits = [
    float(
        t["profit_pct"]
    )
    for t in closed
]

wins = [
    p for p in profits
    if p > 0
]

losses = [
    p for p in profits
    if p <= 0
]

n = len(profits)

winrate = (
    len(wins)
    /
    n
    *
    100
    if n
    else 0
)

sumprofit = sum(
    profits
)

avgwin = (
    float(
        np.mean(wins)
    )
    if wins
    else 0
)

avgloss = (
    float(
        np.mean(losses)
    )
    if losses
    else 0
)


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

portfolio = INITIAL_CAPITAL

equity = [
    portfolio
]

portfolio_history = []

for trade in closed:

    trade_return = (
        trade["profit_pct"]
        /
        100
        *
        POSITION_SIZE
    )

    portfolio *= (
        1 + trade_return
    )

    equity.append(
        portfolio
    )

    portfolio_history.append({

        "date":
            trade["exit_date"],

        "symbol":
            trade["symbol"],

        "trade_return_percent":
            trade["profit_pct"],

        "portfolio_return_percent":
            round(
                trade_return * 100,
                4
            ),

        "portfolio_value":
            round(
                portfolio,
                2
            )
    })


compound_return = (
    portfolio
    /
    INITIAL_CAPITAL
    - 1
) * 100


# ============================================================
# MAX DRAWDOWN
# ============================================================

peak = INITIAL_CAPITAL

max_drawdown = 0

for value in equity:

    peak = max(
        peak,
        value
    )

    drawdown = (
        peak - value
    )
    /
    peak
    *
    100

    max_drawdown = max(
        max_drawdown,
        drawdown
    )


# ============================================================
# EXIT ANALYSIS
# ============================================================

exit_analysis = {}

for trade in closed:

    reason = trade.get(
        "exit_reason",
        "UNKNOWN"
    )

    exit_analysis[reason] = (
        exit_analysis.get(
            reason,
            0
        )
        + 1
    )


# ============================================================
# BEST / WORST
# ============================================================

best = (
    max(
        closed,
        key=lambda x:
        x["profit_pct"]
    )
    if closed
    else None
)

worst = (
    min(
        closed,
        key=lambda x:
        x["profit_pct"]
    )
    if closed
    else None
)


# ============================================================
# RESULT JSON
# ============================================================

result = {

    "strategy":
        "EGX Resistance Breakout + Volume Strategy v1.0",

    "description":
        "Confirmed swing resistance breakout with volume confirmation, next resistance target and structural breakout-failure stop.",

    "parameters": {

        "pivot_left":
            PIVOT_LEFT,

        "pivot_right":
            PIVOT_RIGHT,

        "lookback":
            LOOKBACK,

        "zone_distance_percent":
            ZONE_DISTANCE_PERCENT,

        "minimum_resistance_reactions":
            MIN_RESISTANCE_REACTIONS,

        "reaction_window":
            REACTION_WINDOW,

        "minimum_reaction_percent":
            MIN_REACTION_PERCENT,

        "breakout_percent":
            BREAKOUT_PERCENT,

        "volume_lookback":
            VOLUME_LOOKBACK,

        "volume_multiplier":
            VOLUME_MULTIPLIER,

        "minimum_upside_percent":
            MIN_UPSIDE_PERCENT,

        "stop_buffer_percent":
            STOP_BUFFER_PERCENT,

        "max_positions":
            MAX_POSITIONS,

        "position_size_percent":
            POSITION_SIZE * 100
    },

    "statistics": {

        "total_trades":
            n,

        "winning_trades":
            len(wins),

        "losing_trades":
            len(losses),

        "win_rate_percent":
            round(
                winrate,
                2
            ),

        "sum_trade_profit_percent":
            round(
                sumprofit,
                2
            ),

        "realistic_compound_return_percent":
            round(
                compound_return,
                2
            ),

        "average_win_percent":
            round(
                avgwin,
                2
            ),

        "average_loss_percent":
            round(
                avgloss,
                2
            ),

        "maximum_drawdown_percent":
            round(
                max_drawdown,
                2
            ),

        "open_positions":
            len(opened)
    },

    "exit_analysis":
        exit_analysis,

    "best_trade":
        best,

    "worst_trade":
        worst,

    "open_positions":
        opened,

    "portfolio_equity":
        portfolio_history,

    "trades":
        all_trades
}


# ============================================================
# SAVE
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


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "FINAL RESULTS"
)

print(
    "=" * 72
)

print(
    f"Trades              : {n}"
)

print(
    f"Winners             : {len(wins)}"
)

print(
    f"Losers              : {len(losses)}"
)

print(
    f"Win Rate            : {winrate:.2f}%"
)

print(
    f"Sum Profit          : {sumprofit:.2f}%"
)

print(
    f"Compound Return     : "
    f"{compound_return:.2f}%"
)

print(
    f"Average Win         : "
    f"{avgwin:.2f}%"
)

print(
    f"Average Loss        : "
    f"{avgloss:.2f}%"
)

print(
    f"Maximum Drawdown    : "
    f"{max_drawdown:.2f}%"
)

print(
    f"Open Positions      : "
    f"{len(opened)}"
)


print(
    "\nEXIT ANALYSIS"
)

for reason, count in exit_analysis.items():

    print(
        f"{reason:22}: {count}"
    )


if best:

    print(
        f"\nBEST  : "
        f"{best['symbol']} | "
        f"{best['profit_pct']:.2f}% | "
        f"{best['entry_date']} -> "
        f"{best['exit_date']}"
    )


if worst:

    print(
        f"WORST : "
        f"{worst['symbol']} | "
        f"{worst['profit_pct']:.2f}% | "
        f"{worst['entry_date']} -> "
        f"{worst['exit_date']}"
    )


print(
    f"\nSaved: "
    f"{RESULT_FILE}, "
    f"{TRADES_FILE}"
)

print(
    "=" * 72
)

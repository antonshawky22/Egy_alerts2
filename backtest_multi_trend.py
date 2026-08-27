print("=" * 72)
print("EGX SUPPORT & RESISTANCE STRATEGY v1.0")
print("PURE OHLC PRICE ACTION")
print("=" * 72)

import json
import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DB_FILE = "egx_weekly_database_v1.json"

RESULT_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

INITIAL_CAPITAL = 100000.0
MAX_POSITIONS = 8
POSITION_SIZE = 1.0 / MAX_POSITIONS

# ------------------------------------------------------------
# Swing detection
# ------------------------------------------------------------

PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# ------------------------------------------------------------
# Historical zone detection
# ------------------------------------------------------------

LOOKBACK = 120

# Two pivots within this distance become one zone
ZONE_DISTANCE_PERCENT = 1.0

# Minimum confirmed reactions required for a support entry
MIN_SUPPORT_REACTIONS = 2

# Resistance needs only one confirmed reaction
MIN_RESISTANCE_REACTIONS = 1

# ------------------------------------------------------------
# Reaction
# ------------------------------------------------------------

REACTION_WINDOW = 3

# Minimum bounce from the pivot/zone
MIN_REACTION_PERCENT = 1.0

# ------------------------------------------------------------
# Failed Breakdown
# ------------------------------------------------------------

# Price must enter/break the lower part of support
BREAK_TOLERANCE_PERCENT = 0.20

# Close must recover above the support zone
RECOVERY_REQUIRED = True

# ------------------------------------------------------------
# Trade quality
# ------------------------------------------------------------

# Minimum distance from entry to target resistance
MIN_UPSIDE_PERCENT = 5.0

# Small buffer below support for structural stop
STOP_BUFFER_PERCENT = 0.50

# ------------------------------------------------------------
# Backtest
# ------------------------------------------------------------

MIN_BARS = 40


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

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    for c in required[1:]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = (
        df
        .dropna(subset=required)
        .drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# RAW SWING DETECTION
# ============================================================

def detect_pivots(df):
    """
    Creates candidate pivots.

    IMPORTANT:
    A pivot at index i becomes known only at:
        i + PIVOT_RIGHT

    This information is carried in 'confirmed_at'.

    This prevents look-ahead bias.
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

        # -----------------------------
        # Swing Low
        # -----------------------------

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

        # -----------------------------
        # Swing High
        # -----------------------------

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
# REACTION CHECK
# ============================================================

def pivot_has_reaction(
    df,
    pivot_index,
    pivot_price,
    direction
):
    """
    Determines whether a pivot produced a meaningful
    price reaction.

    No indicators are used.

    Support:
        price must move upward after the pivot.

    Resistance:
        price must move downward after the pivot.
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

    if direction == "SUPPORT":

        max_close = float(
            future["Close"].max()
        )

        rebound = (
            (max_close - pivot_price)
            / pivot_price
            * 100
        )

        return rebound >= MIN_REACTION_PERCENT

    if direction == "RESISTANCE":

        min_close = float(
            future["Close"].min()
        )

        reaction = (
            (pivot_price - min_close)
            / pivot_price
            * 100
        )

        return reaction >= MIN_REACTION_PERCENT

    return False


# ============================================================
# BUILD ZONES
# ============================================================

def build_zones(
    df,
    confirmed_lows,
    confirmed_highs,
    current_index
):
    """
    Builds support/resistance zones using ONLY pivots
    that are already confirmed by current_index.

    No future pivot is allowed.

    Zones are rebuilt from the most recent LOOKBACK window.
    """

    start = max(
        0,
        current_index - LOOKBACK + 1
    )

    # --------------------------------------------------------
    # Confirmed pivots only
    # --------------------------------------------------------

    lows = [
        p for p in confirmed_lows
        if (
            p["confirmed_at"] <= current_index
            and
            p["index"] >= start
        )
    ]

    highs = [
        p for p in confirmed_highs
        if (
            p["confirmed_at"] <= current_index
            and
            p["index"] >= start
        )
    ]

    # --------------------------------------------------------
    # Create clusters
    # --------------------------------------------------------

    def cluster_pivots(
        pivots,
        direction
    ):

        clusters = []

        # Process chronologically
        pivots = sorted(
            pivots,
            key=lambda x: x["index"]
        )

        for p in pivots:

            placed = False

            for zone in clusters:

                zone_price = zone["price"]

                distance = (
                    abs(p["price"] - zone_price)
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
                    "pivots": [p],
                    "direction": direction
                })

        # ----------------------------------------------------
        # Calculate reaction count
        # ----------------------------------------------------

        zones = []

        for zone in clusters:

            reactions = 0

            valid_pivots = []

            for p in zone["pivots"]:

                # Reaction can only be evaluated when
                # its reaction window has already happened.
                reaction_end = (
                    p["index"]
                    + 1
                    + REACTION_WINDOW
                )

                if reaction_end > current_index:
                    continue

                if pivot_has_reaction(
                    df,
                    p["index"],
                    p["price"],
                    direction
                ):

                    reactions += 1
                    valid_pivots.append(p)

            if reactions == 0:
                continue

            prices = [
                p["price"]
                for p in zone["pivots"]
            ]

            zone_low = min(prices)
            zone_high = max(prices)

            # Expand the zone slightly using the agreed
            # percentage tolerance.
            zone_low *= (
                1 - ZONE_DISTANCE_PERCENT / 100
            )

            zone_high *= (
                1 + ZONE_DISTANCE_PERCENT / 100
            )

            zones.append({
                "direction": direction,
                "price": round(
                    float(np.mean(prices)),
                    4
                ),
                "low": round(
                    float(zone_low),
                    4
                ),
                "high": round(
                    float(zone_high),
                    4
                ),
                "reactions": reactions,
                "first_pivot": min(
                    p["index"]
                    for p in zone["pivots"]
                ),
                "last_pivot": max(
                    p["index"]
                    for p in zone["pivots"]
                )
            })

        return zones

    supports = cluster_pivots(
        lows,
        "SUPPORT"
    )

    resistances = cluster_pivots(
        highs,
        "RESISTANCE"
    )

    return supports, resistances


# ============================================================
# FIND TARGET RESISTANCE
# ============================================================

def find_target_resistance(
    resistances,
    entry_price
):

    candidates = []

    for zone in resistances:

        if zone["reactions"] < MIN_RESISTANCE_REACTIONS:
            continue

        if zone["price"] <= entry_price:
            continue

        upside = (
            (zone["price"] - entry_price)
            / entry_price
            * 100
        )

        if upside >= MIN_UPSIDE_PERCENT:

            candidates.append(
                (zone["price"], zone)
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0]
    )

    return candidates[0][1]


# ============================================================
# FAILED BREAKDOWN
# ============================================================

def failed_breakdown(
    row,
    support
):
    """
    Core BUY condition.

    Price must test/break the lower part of support,
    but the candle must close back above the zone.

    No indicators.
    """

    low = float(row["Low"])
    close = float(row["Close"])

    zone_low = float(
        support["low"]
    )

    zone_high = float(
        support["high"]
    )

    # Price must reach the support area.
    touched = (
        low <= zone_high
    )

    # A real penetration of the zone.
    broke = (
        low <
        zone_low *
        (1 - BREAK_TOLERANCE_PERCENT / 100)
    )

    # Recovery above zone.
    recovered = (
        close >= zone_high
        if RECOVERY_REQUIRED
        else close >= zone_low
    )

    return (
        touched
        and broke
        and recovered
    )


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
        / position["entry_price"]
        * 100
    )

    return {
        "symbol": position["symbol"],
        "status": "CLOSED",

        "entry_date": position["entry_date"],
        "entry_price": round(
            position["entry_price"],
            4
        ),

        "exit_date": date,
        "exit_price": round(
            price,
            4
        ),

        "profit_pct": round(
            profit,
            2
        ),

        "exit_reason": reason,

        "support": position["support"],
        "support_reactions": position[
            "support_reactions"
        ],

        "resistance": position[
            "resistance"
        ],

        "upside_to_resistance": position[
            "upside_to_resistance"
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

        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        # ====================================================
        # BUILD ONLY WHAT WAS KNOWN BY THIS DATE
        # ====================================================

        supports, resistances = build_zones(
            df,
            pivot_lows,
            pivot_highs,
            i
        )

        # ====================================================
        # MANAGE OPEN POSITION
        # ====================================================

        if position is not None:

            support = position[
                "support_zone"
            ]

            resistance = position[
                "resistance_zone"
            ]

            # ------------------------------------------------
            # Structural stop
            # ------------------------------------------------

            stop_price = (
                support["low"]
                *
                (1 - STOP_BUFFER_PERCENT / 100)
            )

            # ------------------------------------------------
            # Stop has priority
            # ------------------------------------------------

            if close < stop_price:

                trades.append(
                    close_trade(
                        position,
                        date,
                        close,
                        "SUPPORT_BREAK"
                    )
                )

                position = None
                continue

            # ------------------------------------------------
            # Target resistance
            # ------------------------------------------------

            if resistance is not None:

                target = float(
                    resistance["price"]
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
        # FIND BUY SETUP
        # ====================================================

        valid_supports = [
            s for s in supports
            if s["reactions"]
            >= MIN_SUPPORT_REACTIONS
        ]

        if not valid_supports:
            continue

        # ----------------------------------------------------
        # Find supports currently being tested
        # ----------------------------------------------------

        tested_supports = []

        for support in valid_supports:

            if low > support["high"]:
                continue

            if close < support["low"]:
                continue

            if failed_breakdown(
                row,
                support
            ):

                tested_supports.append(
                    support
                )

        if not tested_supports:
            continue

        # ----------------------------------------------------
        # Choose nearest support below/around price
        # ----------------------------------------------------

        tested_supports.sort(
            key=lambda s:
            abs(close - s["price"])
        )

        support = tested_supports[0]

        # ====================================================
        # FIND TARGET RESISTANCE
        # ====================================================

        resistance = find_target_resistance(
            resistances,
            close
        )

        if resistance is None:
            continue

        upside = (
            (resistance["price"] - close)
            / close
            * 100
        )

        if upside < MIN_UPSIDE_PERCENT:
            continue

        # ====================================================
        # ENTRY
        # ====================================================

        # Signal is known only after today's close.
        # Therefore entry occurs at NEXT day's OPEN.

        if i + 1 >= len(df):
            continue

        next_row = df.iloc[i + 1]

        entry_date = next_row[
            "Date"
        ].strftime("%Y-%m-%d")

        entry_price = float(
            next_row["Open"]
        )

        # Recalculate available upside
        # using actual next-day entry.
        actual_upside = (
            (resistance["price"] - entry_price)
            / entry_price
            * 100
        )

        if actual_upside < MIN_UPSIDE_PERCENT:
            continue

        position = {
            "symbol": sym,

            "entry_date": entry_date,
            "entry_price": entry_price,

            "support_zone": support,
            "resistance_zone": resistance,

            "support": round(
                support["price"],
                4
            ),

            "support_reactions":
                support["reactions"],

            "resistance": round(
                resistance["price"],
                4
            ),

            "upside_to_resistance":
                round(
                    actual_upside,
                    2
                )
        }

    # ========================================================
    # CLOSE OPEN POSITION AT LAST CLOSE
    # ========================================================

    if position is not None:

        last = df.iloc[-1]

        last_date = last[
            "Date"
        ].strftime("%Y-%m-%d")

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

        trades.append(trade)

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
            f"⚠️ {sym}: invalid data"
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
# RESULTS
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

profits = [
    float(t["profit_pct"])
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
    len(wins) / n * 100
    if n
    else 0
)

sumprofit = sum(profits)

avgwin = (
    float(np.mean(wins))
    if wins
    else 0
)

avgloss = (
    float(np.mean(losses))
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
        / 100
        * POSITION_SIZE
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
    portfolio / INITIAL_CAPITAL
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
    ) / peak * 100

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
        ) + 1
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
        "EGX Support & Resistance Strategy v1.0",

    "description":
        "Pure OHLC support/resistance strategy using confirmed swing pivots, price zones, failed breakdown entries and structural resistance exits.",

    "parameters": {

        "pivot_left":
            PIVOT_LEFT,

        "pivot_right":
            PIVOT_RIGHT,

        "lookback":
            LOOKBACK,

        "zone_distance_percent":
            ZONE_DISTANCE_PERCENT,

        "minimum_support_reactions":
            MIN_SUPPORT_REACTIONS,

        "minimum_resistance_reactions":
            MIN_RESISTANCE_REACTIONS,

        "reaction_window":
            REACTION_WINDOW,

        "minimum_reaction_percent":
            MIN_REACTION_PERCENT,

        "break_tolerance_percent":
            BREAK_TOLERANCE_PERCENT,

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

print("FINAL RESULTS")

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


print("\nEXIT ANALYSIS")

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

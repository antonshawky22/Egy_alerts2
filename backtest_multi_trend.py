import json
import os
import pandas as pd
import numpy as np

# ============================================================
# WEEKLY EMA10/20 SIDEWAYS BREAKOUT BACKTEST
# ============================================================
#
# الفكرة:
# 1) اكتشاف أن السهم كان في نطاق عرضي خلال آخر 15 شمعة.
# 2) الاعتماد على EMA10 و EMA20 في تحديد العرضي.
# 3) وجود عدة تقاطعات + ميل ضعيف للمتوسطين = Sideways.
# 4) بعد انتهاء العرضي:
#       Golden Cross -> BUY
#       Death Cross  -> SELL
#
# ============================================================


DB_FILE = "egx_weekly_database_v1.json"

RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"


# ============================================================
# SYMBOLS
# ============================================================

symbols = {
    "OLFI": "OLFI",
    "EMFD": "EMFD",
    "ETEL": "ETEL",
    "EAST": "EAST",
    "EFIH": "EFIH",
    "ABUK": "ABUK",
    "OIH": "OIH",
    "SWDY": "SWDY",
    "ISPH": "ISPH",
    "ATQA": "ATQA",
    "MTIE": "MTIE",
    "HRHO": "HRHO",
    "ORWE": "ORWE",
    "JUFO": "JUFO",
    "DSCW": "DSCW",
    "SUGR": "SUGR",
    "ELSH": "ELSH",
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
    "ADIB": "ADIB",
    "PHDC": "PHDC",
    "MCQE": "MCQE",
    "SKPC": "SKPC",
    "EGAL": "EGAL"
}


# ============================================================
# PARAMETERS
# ============================================================

# عدد الشموع المستخدمة لاكتشاف النطاق العرضي
SIDEWAYS_LOOKBACK = 15

# الحد الأقصى لميل EMA10 خلال الفترة
EMA10_SLOPE_LIMIT = 3.0

# الحد الأقصى لميل EMA20 خلال الفترة
EMA20_SLOPE_LIMIT = 3.0

# أقصى مسافة بين EMA10 و EMA20 عند الشمعة الحالية
EMA_GAP_LIMIT = 3.0

# أقل عدد تقاطعات داخل فترة الـ Sideways
MIN_CROSSES = 2

# Stop Loss
HARD_STOP_PERCENT = 5.0

# Trailing Stop
TRAILING_ACTIVATION_PROFIT = 10.0
TRAILING_RETRACEMENT = 4.5


# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(
        f"Database file not found: {DB_FILE}"
    )


with open(DB_FILE, "r", encoding="utf-8") as f:
    raw_database = json.load(f)


# ============================================================
# PREPARE DATA
# ============================================================

prepared_data = {}


for name in symbols:

    if name not in raw_database:
        continue

    content = raw_database[name]

    if "data" not in content or "columns" not in content:
        continue

    try:

        df = pd.DataFrame.from_dict(
            content["data"],
            orient="index",
            columns=content["columns"]
        )

        df.index = pd.to_datetime(df.index)

        df = df.sort_index(ascending=True)

        if len(df) < 30:
            continue

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        for col in ["Open", "High", "Low", "Close", "Volume"]:

            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        df = df.dropna(subset=["Close"])

        close = df["Close"]

        # ----------------------------------------------------
        # EMA10 / EMA20
        # ----------------------------------------------------

        df["EMA10"] = close.ewm(
            span=10,
            adjust=False
        ).mean()

        df["EMA20"] = close.ewm(
            span=20,
            adjust=False
        ).mean()

        prepared_data[name] = df

    except Exception as e:

        print(
            f"⚠️ Error preparing {name}: {e}"
        )


if not prepared_data:
    raise RuntimeError(
        "No valid symbols found in database."
    )


# ============================================================
# ALL DATES
# ============================================================

all_dates = sorted(
    set().union(
        *[
            df.index
            for df in prepared_data.values()
        ]
    )
)


# ============================================================
# STATE TRACKING
# ============================================================

states = {

    name: {

        "position": 0.0,

        "entry_price": 0.0,

        "peak_price": 0.0

    }

    for name in prepared_data
}


# ============================================================
# TRADE STORAGE
# ============================================================

trades_history = []


total_portfolio_profit = 0.0

portfolio_equity_curve = []


# ============================================================
# HELPER FUNCTION
# ============================================================

def detect_sideways(df_slice):
    """
    تحديد هل آخر 15 شمعة تمثل نطاقًا عرضيًا.

    الشروط:

    1) عدد التقاطعات >= MIN_CROSSES
    2) ميل EMA10 ضعيف
    3) ميل EMA20 ضعيف
    4) المسافة الحالية بين EMA10 و EMA20 صغيرة
    """

    if len(df_slice) < SIDEWAYS_LOOKBACK:
        return False

    recent = df_slice.tail(SIDEWAYS_LOOKBACK)

    ema10_start = float(
        recent["EMA10"].iloc[0]
    )

    ema10_end = float(
        recent["EMA10"].iloc[-1]
    )

    ema20_start = float(
        recent["EMA20"].iloc[0]
    )

    ema20_end = float(
        recent["EMA20"].iloc[-1]
    )

    if (
        ema10_start <= 0
        or ema20_start <= 0
    ):
        return False

    # --------------------------------------------------------
    # EMA10 slope
    # --------------------------------------------------------

    ema10_slope_pct = abs(
        (
            ema10_end - ema10_start
        )
        / ema10_start
        * 100
    )

    # --------------------------------------------------------
    # EMA20 slope
    # --------------------------------------------------------

    ema20_slope_pct = abs(
        (
            ema20_end - ema20_start
        )
        / ema20_start
        * 100
    )

    # --------------------------------------------------------
    # Count EMA10 / EMA20 crosses
    # --------------------------------------------------------

    cross_count = 0

    ema10_values = recent["EMA10"].values
    ema20_values = recent["EMA20"].values

    for i in range(1, len(recent)):

        ema10_prev = ema10_values[i - 1]
        ema20_prev = ema20_values[i - 1]

        ema10_curr = ema10_values[i]
        ema20_curr = ema20_values[i]

        golden = (
            ema10_prev <= ema20_prev
            and
            ema10_curr > ema20_curr
        )

        death = (
            ema10_prev >= ema20_prev
            and
            ema10_curr < ema20_curr
        )

        if golden or death:
            cross_count += 1

    # --------------------------------------------------------
    # Current EMA gap
    # --------------------------------------------------------

    current_ema10 = float(
        recent["EMA10"].iloc[-1]
    )

    current_ema20 = float(
        recent["EMA20"].iloc[-1]
    )

    if current_ema20 <= 0:
        return False

    ema_gap_pct = abs(
        (
            current_ema10 - current_ema20
        )
        / current_ema20
        * 100
    )

    # --------------------------------------------------------
    # Final Sideways condition
    # --------------------------------------------------------

    sideways = (

        cross_count >= MIN_CROSSES

        and

        ema10_slope_pct <= EMA10_SLOPE_LIMIT

        and

        ema20_slope_pct <= EMA20_SLOPE_LIMIT

        and

        ema_gap_pct <= EMA_GAP_LIMIT

    )

    return sideways


# ============================================================
# BACKTEST LOOP
# ============================================================

for current_date in all_dates:

    for name, df in prepared_data.items():

        if current_date not in df.index:
            continue

        idx = df.index.get_loc(current_date)

        # نحتاج 20 شمعة لحساب EMA20
        # + 15 شمعة لاكتشاف العرضي
        if idx < max(20, SIDEWAYS_LOOKBACK):
            continue

        df_slice = df.iloc[:idx + 1]

        # ----------------------------------------------------
        # Validate current indicators
        # ----------------------------------------------------

        current_row = df_slice.iloc[-1]

        required_columns = [
            "Close",
            "EMA10",
            "EMA20"
        ]

        if current_row[required_columns].isna().any():
            continue

        # ----------------------------------------------------
        # Current values
        # ----------------------------------------------------

        date_str = current_date.strftime(
            "%Y-%m-%d"
        )

        price = float(
            current_row["Close"]
        )

        ema10_curr = float(
            current_row["EMA10"]
        )

        ema20_curr = float(
            current_row["EMA20"]
        )

        ema10_prev = float(
            df_slice["EMA10"].iloc[-2]
        )

        ema20_prev = float(
            df_slice["EMA20"].iloc[-2]
        )

        # ----------------------------------------------------
        # Cross Detection
        # ----------------------------------------------------

        golden_cross = (

            ema10_prev <= ema20_prev

            and

            ema10_curr > ema20_curr

        )

        death_cross = (

            ema10_prev >= ema20_prev

            and

            ema10_curr < ema20_curr

        )

        # ----------------------------------------------------
        # Sideways Detection
        # ----------------------------------------------------

        is_sideways = detect_sideways(
            df_slice
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        s = states[name]

        # ----------------------------------------------------
        # Current profit
        # ----------------------------------------------------

        profit = 0.0

        if s["entry_price"] > 0:

            profit = (
                (
                    price
                    - s["entry_price"]
                )
                /
                s["entry_price"]
            ) * 100

            if price > s["peak_price"]:

                s["peak_price"] = price

        # ====================================================
        # BUY
        # ====================================================

        if (

            s["position"] == 0.0

            and

            golden_cross

            and

            is_sideways

        ):

            s["position"] = 1.0

            s["entry_price"] = price

            s["peak_price"] = price

            trades_history.append({

                "symbol": name,

                "status": "OPEN",

                "entry_date": date_str,

                "entry_price": round(
                    price,
                    2
                ),

                "exit_date": None,

                "exit_price": None,

                "profit_pct": None

            })

        # ====================================================
        # SELL / STOP
        # ====================================================

        elif s["position"] == 1.0:

            # ------------------------------------------------
            # Peak profit
            # ------------------------------------------------

            peak_profit = (

                (
                    s["peak_price"]
                    - s["entry_price"]
                )
                /
                s["entry_price"]

            ) * 100

            # ------------------------------------------------
            # Hard Stop
            # ------------------------------------------------

            hard_stop = (

                profit
                <=
                -HARD_STOP_PERCENT

            )

            # ------------------------------------------------
            # Trailing Stop
            # ------------------------------------------------

            retracement_pct = (

                (
                    s["peak_price"]
                    - price
                )
                /
                s["peak_price"]

            ) * 100

            trailing_stop = (

                peak_profit
                >=
                TRAILING_ACTIVATION_PROFIT

                and

                retracement_pct
                >=
                TRAILING_RETRACEMENT

            )

            # ------------------------------------------------
            # Death Cross
            # ------------------------------------------------

            cross_exit = death_cross

            # ------------------------------------------------
            # Final Exit Trigger
            # ------------------------------------------------

            exit_triggered = (

                cross_exit

                or

                hard_stop

                or

                trailing_stop

            )

            if exit_triggered:

                active = [

                    t

                    for t in trades_history

                    if (

                        t["symbol"] == name

                        and

                        t["status"] == "OPEN"

                    )

                ]

                if active:

                    active[-1]["status"] = "CLOSED"

                    active[-1]["exit_date"] = date_str

                    active[-1]["exit_price"] = round(
                        price,
                        2
                    )

                    active[-1]["profit_pct"] = round(
                        profit,
                        2
                    )

                    # سبب الخروج
                    if hard_stop:

                        active[-1]["exit_reason"] = (
                            "HARD_STOP"
                        )

                    elif trailing_stop:

                        active[-1]["exit_reason"] = (
                            "TRAILING_STOP"
                        )

                    elif cross_exit:

                        active[-1]["exit_reason"] = (
                            "DEATH_CROSS"
                        )

                total_portfolio_profit += profit

                s["position"] = 0.0

                s["entry_price"] = 0.0

                s["peak_price"] = 0.0

    # --------------------------------------------------------
    # Equity Curve
    # --------------------------------------------------------

    portfolio_equity_curve.append(
        total_portfolio_profit
    )


# ============================================================
# CLOSE REMAINING OPEN POSITIONS
# ============================================================

# لا نحسب الصفقات المفتوحة كصفقات مغلقة.
# فقط نتركها OPEN في ملف النتائج.


# ============================================================
# STATISTICS
# ============================================================

closed_trades = [

    t

    for t in trades_history

    if t["status"] == "CLOSED"

]


winning_trades = [

    t

    for t in closed_trades

    if (

        t["profit_pct"] is not None

        and

        t["profit_pct"] > 0

    )

]


losing_trades = [

    t

    for t in closed_trades

    if (

        t["profit_pct"] is not None

        and

        t["profit_pct"] <= 0

    )

]


total_count = len(
    closed_trades
)

wins_count = len(
    winning_trades
)

losses_count = len(
    losing_trades
)


win_rate = (

    wins_count
    /
    total_count
    *
    100

    if total_count > 0

    else 0.0

)


avg_win = (

    float(
        np.mean(
            [
                t["profit_pct"]
                for t in winning_trades
            ]
        )
    )

    if winning_trades

    else 0.0

)


avg_loss = (

    float(
        np.mean(
            [
                t["profit_pct"]
                for t in losing_trades
            ]
        )
    )

    if losing_trades

    else 0.0

)


# ============================================================
# MAX DRAWDOWN
# ============================================================

equity = (

    np.array(
        portfolio_equity_curve
    )

    + 100.0

)


if len(equity) > 0:

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (

        (peak - equity)
        /
        peak
        *
        100.0

    )

    max_drawdown = float(
        np.max(drawdown)
    )

else:

    max_drawdown = 0.0


# ============================================================
# RESULTS
# ============================================================

results_summary = {

    "strategy": (
        "EMA10/20 Sideways "
        "Breakout Weekly"
    ),

    "parameters": {

        "sideways_lookback":
            SIDEWAYS_LOOKBACK,

        "ema10_slope_limit":
            EMA10_SLOPE_LIMIT,

        "ema20_slope_limit":
            EMA20_SLOPE_LIMIT,

        "ema_gap_limit":
            EMA_GAP_LIMIT,

        "minimum_crosses":
            MIN_CROSSES,

        "hard_stop_percent":
            HARD_STOP_PERCENT,

        "trailing_activation_profit":
            TRAILING_ACTIVATION_PROFIT,

        "trailing_retracement":
            TRAILING_RETRACEMENT

    },

    "statistics": {

        "total_trades":
            total_count,

        "winning_trades":
            wins_count,

        "losing_trades":
            losses_count,

        "win_rate_percent":
            round(
                win_rate,
                2
            ),

        "total_profit_percent":
            round(
                total_portfolio_profit,
                2
            ),

        "average_winning_trade_percent":
            round(
                avg_win,
                2
            ),

        "average_losing_trade_percent":
            round(
                avg_loss,
                2
            ),

        "maximum_drawdown_percent":
            round(
                max_drawdown,
                2
            )

    }

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
        results_summary,
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
        trades_history,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)

print(
    "WEEKLY EMA10/20 SIDEWAYS BREAKOUT "
    "BACKTEST COMPLETE"
)

print("=" * 65)

print(
    f"Sideways Lookback:   "
    f"{SIDEWAYS_LOOKBACK} weeks"
)

print(
    f"Minimum Crosses:     "
    f"{MIN_CROSSES}"
)

print(
    f"EMA10 Slope Limit:   "
    f"{EMA10_SLOPE_LIMIT:.2f}%"
)

print(
    f"EMA20 Slope Limit:   "
    f"{EMA20_SLOPE_LIMIT:.2f}%"
)

print(
    f"EMA Gap Limit:       "
    f"{EMA_GAP_LIMIT:.2f}%"
)

print("-" * 65)

print(
    f"Total Closed Trades: "
    f"{total_count}"
)

print(
    f"Winning Trades:      "
    f"{wins_count}"
)

print(
    f"Losing Trades:       "
    f"{losses_count}"
)

print(
    f"Win Rate:            "
    f"{win_rate:.2f}%"
)

print(
    f"Total Profit:        "
    f"{total_portfolio_profit:.2f}%"
)

print(
    f"Average Win:         "
    f"{avg_win:.2f}%"
)

print(
    f"Average Loss:        "
    f"{avg_loss:.2f}%"
)

print(
    f"Max Drawdown:        "
    f"{max_drawdown:.2f}%"
)

print("=" * 65)
print("Results saved to:")
print(RESULTS_FILE)
print(TRADES_FILE)
print("=" * 65)

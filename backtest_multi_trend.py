import json
import os
import pandas as pd
import numpy as np

# ============================================================
# WEEKLY SIDEWAYS + RSI STRATEGY
# ============================================================

DB_FILE = "egx_weekly_database_v1.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

symbols = {
    "OLFI":"OLFI","EMFD":"EMFD","ETEL":"ETEL","EAST":"EAST",
    "EFIH":"EFIH","ABUK":"ABUK","OIH":"OIH","SWDY":"SWDY",
    "ISPH":"ISPH","ATQA":"ATQA","MTIE":"MTIE","HRHO":"HRHO",
    "ORWE":"ORWE","JUFO":"JUFO","DSCW":"DSCW","SUGR":"SUGR",
    "ELSH":"ELSH","RMDA":"RMDA","RAYA":"RAYA","EEII":"EEII",
    "MPCO":"MPCO","GBCO":"GBCO","TMGH":"TMGH","ORHD":"ORHD",
    "AMOC":"AMOC","FWRY":"FWRY","COMI":"COMI","ADIB":"ADIB",
    "PHDC":"PHDC","MCQE":"MCQE","SKPC":"SKPC","EGAL":"EGAL"
}

# ============================================================
# PARAMETERS
# ============================================================

SIDEWAYS_LOOKBACK = 15

MIN_CROSSES = 3

EMA10_SLOPE_LIMIT = 3.0
EMA20_SLOPE_LIMIT = 3.0

AVG_GAP_LIMIT = 3.0
MAX_GAP_LIMIT = 5.0

MIN_SIDEWAYS_SCORE = 4

RSI_BUY = 33
RSI_SELL = 63

HARD_STOP_PERCENT = 5.0

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

prepared_data = {}

# ============================================================
# PREPARE DATA
# ============================================================

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
        df = df.sort_index()

        if len(df) < 30:
            continue

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]:

            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        df = df.dropna(
            subset=["Close"]
        )

        close = df["Close"]

        # EMA10
        df["EMA10"] = close.ewm(
            span=10,
            adjust=False
        ).mean()

        # EMA20
        df["EMA20"] = close.ewm(
            span=20,
            adjust=False
        ).mean()

        # RSI14
        delta = close.diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(
            0,
            np.nan
        )

        df["RSI14"] = 100 - (
            100 / (1 + rs)
        )

        prepared_data[name] = df

    except Exception as e:

        print(
            f"Error preparing {name}: {e}"
        )

if not prepared_data:
    raise RuntimeError(
        "No valid symbols found."
    )

# ============================================================
# DATES
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
# SIDEWAYS DETECTION
# ============================================================

def detect_sideways(df):

    # مهم:
    # نستخدم الـ15 شمعة السابقة فقط
    # ولا ندخل الشمعة الحالية في القياس.

    if len(df) < SIDEWAYS_LOOKBACK + 1:
        return False

    r = df.iloc[
        -(SIDEWAYS_LOOKBACK + 1):-1
    ]

    e10 = r["EMA10"].values
    e20 = r["EMA20"].values

    if (
        np.isnan(e10).any()
        or
        np.isnan(e20).any()
    ):
        return False

    # --------------------------------------------------------
    # 1. عدد التقاطعات السابقة
    # --------------------------------------------------------

    cross_count = 0

    for i in range(1, len(r)):

        golden = (
            e10[i - 1] <= e20[i - 1]
            and
            e10[i] > e20[i]
        )

        death = (
            e10[i - 1] >= e20[i - 1]
            and
            e10[i] < e20[i]
        )

        if golden or death:
            cross_count += 1

    # --------------------------------------------------------
    # 2. ميل EMA10
    # --------------------------------------------------------

    e10_start = e10[0]
    e10_end = e10[-1]

    if e10_start <= 0:
        return False

    slope10 = abs(
        (e10_end - e10_start)
        / e10_start
        * 100
    )

    # --------------------------------------------------------
    # 3. ميل EMA20
    # --------------------------------------------------------

    e20_start = e20[0]
    e20_end = e20[-1]

    if e20_start <= 0:
        return False

    slope20 = abs(
        (e20_end - e20_start)
        / e20_start
        * 100
    )

    # --------------------------------------------------------
    # 4. المسافة بين EMA10 و EMA20
    # --------------------------------------------------------

    gaps = np.abs(
        (e10 - e20) / e20
    ) * 100

    avg_gap = float(
        np.mean(gaps)
    )

    max_gap = float(
        np.max(gaps)
    )

    # --------------------------------------------------------
    # Sideways Score
    # --------------------------------------------------------

    score = 0

    if cross_count >= MIN_CROSSES:
        score += 1

    if slope10 <= EMA10_SLOPE_LIMIT:
        score += 1

    if slope20 <= EMA20_SLOPE_LIMIT:
        score += 1

    if (
        avg_gap <= AVG_GAP_LIMIT
        and
        max_gap <= MAX_GAP_LIMIT
    ):
        score += 1

    return score >= MIN_SIDEWAYS_SCORE


# ============================================================
# STATES
# ============================================================

states = {

    name: {
        "position": 0.0,
        "entry_price": 0.0,
        "peak_price": 0.0
    }

    for name in prepared_data
}

trades_history = []

total_portfolio_profit = 0.0

portfolio_equity_curve = []

# ============================================================
# BACKTEST
# ============================================================

for current_date in all_dates:

    for name, df in prepared_data.items():

        if current_date not in df.index:
            continue

        idx = df.index.get_loc(
            current_date
        )

        if idx < SIDEWAYS_LOOKBACK + 20:
            continue

        d = df.iloc[
            :idx + 1
        ]

        row = d.iloc[-1]

        required = [
            "Close",
            "EMA10",
            "EMA20",
            "RSI14"
        ]

        if row[required].isna().any():
            continue

        price = float(
            row["Close"]
        )

        rsi = float(
            row["RSI14"]
        )

        date_str = current_date.strftime(
            "%Y-%m-%d"
        )

        # ----------------------------------------------------
        # Sideways
        # ----------------------------------------------------

        is_sideways = detect_sideways(
            d
        )

        s = states[name]

        # ----------------------------------------------------
        # Current Profit
        # ----------------------------------------------------

        profit = 0.0

        if s["entry_price"] > 0:

            profit = (
                (price - s["entry_price"])
                / s["entry_price"]
            ) * 100

            if price > s["peak_price"]:
                s["peak_price"] = price

        # ====================================================
        # BUY
        # ====================================================

        if (
            s["position"] == 0.0
            and is_sideways
            and rsi < RSI_BUY
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
        # EXIT
        # ====================================================

        elif s["position"] == 1.0:

            peak_profit = (
                (
                    s["peak_price"]
                    -
                    s["entry_price"]
                )
                /
                s["entry_price"]
            ) * 100

            # ------------------------------------------------
            # RSI Sell
            # ------------------------------------------------

            rsi_exit = (
                rsi > RSI_SELL
            )

            # ------------------------------------------------
            # Hard Stop
            # ------------------------------------------------

            hard_stop = (
                profit <=
                -HARD_STOP_PERCENT
            )

            # ------------------------------------------------
            # Trailing Stop
            # ------------------------------------------------

            retracement = (
                (
                    s["peak_price"]
                    -
                    price
                )
                /
                s["peak_price"]
            ) * 100

            trailing_stop = (
                peak_profit >=
                TRAILING_ACTIVATION_PROFIT
                and
                retracement >=
                TRAILING_RETRACEMENT
            )

            # ------------------------------------------------
            # Exit Reason
            # ------------------------------------------------

            if hard_stop:

                exit_reason = "HARD_STOP"

            elif trailing_stop:

                exit_reason = "TRAILING_STOP"

            elif rsi_exit:

                exit_reason = "RSI_63"

            else:

                exit_reason = None

            # ------------------------------------------------
            # Execute Exit
            # ------------------------------------------------

            if exit_reason:

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

                    active[-1]["exit_reason"] = (
                        exit_reason
                    )

                total_portfolio_profit += profit

                s["position"] = 0.0

                s["entry_price"] = 0.0

                s["peak_price"] = 0.0

    portfolio_equity_curve.append(
        total_portfolio_profit
    )

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

    if t["profit_pct"] > 0

]

losing_trades = [

    t

    for t in closed_trades

    if t["profit_pct"] <= 0

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
        np.mean([
            t["profit_pct"]
            for t in winning_trades
        ])
    )

    if winning_trades

    else 0.0

)

avg_loss = (

    float(
        np.mean([
            t["profit_pct"]
            for t in losing_trades
        ])
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
    + 100
)

if len(equity) > 0:

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        (peak - equity)
        / peak
    ) * 100

    max_drawdown = float(
        np.max(drawdown)
    )

else:

    max_drawdown = 0.0

# ============================================================
# RESULTS
# ============================================================

results_summary = {

    "parameters": {

        "sideways_lookback":
            SIDEWAYS_LOOKBACK,

        "minimum_crosses":
            MIN_CROSSES,

        "ema10_slope_limit":
            EMA10_SLOPE_LIMIT,

        "ema20_slope_limit":
            EMA20_SLOPE_LIMIT,

  "average_gap_limit":
            AVG_GAP_LIMIT,

        "maximum_gap_limit":
            MAX_GAP_LIMIT,

        "minimum_sideways_scimport json
import os
import pandas as pd
import numpy as np

# ============================================================
# WEEKLY SIDEWAYS + RSI STRATEGY
# ============================================================

DB_FILE = "egx_weekly_database_v1.json"
RESULTS_FILE = "backtest_results.json"
TRADES_FILE = "backtest_trades.json"

symbols = {
    "OLFI":"OLFI","EMFD":"EMFD","ETEL":"ETEL","EAST":"EAST",
    "EFIH":"EFIH","ABUK":"ABUK","OIH":"OIH","SWDY":"SWDY",
    "ISPH":"ISPH","ATQA":"ATQA","MTIE":"MTIE","HRHO":"HRHO",
    "ORWE":"ORWE","JUFO":"JUFO","DSCW":"DSCW","SUGR":"SUGR",
    "ELSH":"ELSH","RMDA":"RMDA","RAYA":"RAYA","EEII":"EEII",
    "MPCO":"MPCO","GBCO":"GBCO","TMGH":"TMGH","ORHD":"ORHD",
    "AMOC":"AMOC","FWRY":"FWRY","COMI":"COMI","ADIB":"ADIB",
    "PHDC":"PHDC","MCQE":"MCQE","SKPC":"SKPC","EGAL":"EGAL"
}

# ============================================================
# PARAMETERS
# ============================================================

SIDEWAYS_LOOKBACK = 15

MIN_CROSSES = 3

EMA10_SLOPE_LIMIT = 3.0
EMA20_SLOPE_LIMIT = 3.0

AVG_GAP_LIMIT = 3.0
MAX_GAP_LIMIT = 5.0

MIN_SIDEWAYS_SCORE = 4

RSI_BUY = 33
RSI_SELL = 63

HARD_STOP_PERCENT = 5.0

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

prepared_data = {}

# ============================================================
# PREPARE DATA
# ============================================================

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
        df = df.sort_index()

        if len(df) < 30:
            continue

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]:

            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        df = df.dropna(
            subset=["Close"]
        )

        close = df["Close"]

        # EMA10
        df["EMA10"] = close.ewm(
            span=10,
            adjust=False
        ).mean()

        # EMA20
        df["EMA20"] = close.ewm(
            span=20,
            adjust=False
        ).mean()

        # RSI14
        delta = close.diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(
            0,
            np.nan
        )

        df["RSI14"] = 100 - (
            100 / (1 + rs)
        )

        prepared_data[name] = df

    except Exception as e:

        print(
            f"Error preparing {name}: {e}"
        )

if not prepared_data:
    raise RuntimeError(
        "No valid symbols found."
    )

# ============================================================
# DATES
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
# SIDEWAYS DETECTION
# ============================================================

def detect_sideways(df):

    # مهم:
    # نستخدم الـ15 شمعة السابقة فقط
    # ولا ندخل الشمعة الحالية في القياس.

    if len(df) < SIDEWAYS_LOOKBACK + 1:
        return False

    r = df.iloc[
        -(SIDEWAYS_LOOKBACK + 1):-1
    ]

    e10 = r["EMA10"].values
    e20 = r["EMA20"].values

    if (
        np.isnan(e10).any()
        or
        np.isnan(e20).any()
    ):
        return False

    # --------------------------------------------------------
    # 1. عدد التقاطعات السابقة
    # --------------------------------------------------------

    cross_count = 0

    for i in range(1, len(r)):

        golden = (
            e10[i - 1] <= e20[i - 1]
            and
            e10[i] > e20[i]
        )

        death = (
            e10[i - 1] >= e20[i - 1]
            and
            e10[i] < e20[i]
        )

        if golden or death:
            cross_count += 1

    # --------------------------------------------------------
    # 2. ميل EMA10
    # --------------------------------------------------------

    e10_start = e10[0]
    e10_end = e10[-1]

    if e10_start <= 0:
        return False

    slope10 = abs(
        (e10_end - e10_start)
        / e10_start
        * 100
    )

    # --------------------------------------------------------
    # 3. ميل EMA20
    # --------------------------------------------------------

    e20_start = e20[0]
    e20_end = e20[-1]

    if e20_start <= 0:
        return False

    slope20 = abs(
        (e20_end - e20_start)
        / e20_start
        * 100
    )

    # --------------------------------------------------------
    # 4. المسافة بين EMA10 و EMA20
    # --------------------------------------------------------

    gaps = np.abs(
        (e10 - e20) / e20
    ) * 100

    avg_gap = float(
        np.mean(gaps)
    )

    max_gap = float(
        np.max(gaps)
    )

    # --------------------------------------------------------
    # Sideways Score
    # --------------------------------------------------------

    score = 0

    if cross_count >= MIN_CROSSES:
        score += 1

    if slope10 <= EMA10_SLOPE_LIMIT:
        score += 1

    if slope20 <= EMA20_SLOPE_LIMIT:
        score += 1

    if (
        avg_gap <= AVG_GAP_LIMIT
        and
        max_gap <= MAX_GAP_LIMIT
    ):
        score += 1

    return score >= MIN_SIDEWAYS_SCORE


# ============================================================
# STATES
# ============================================================

states = {

    name: {
        "position": 0.0,
        "entry_price": 0.0,
        "peak_price": 0.0
    }

    for name in prepared_data
}

trades_history = []

total_portfolio_profit = 0.0

portfolio_equity_curve = []

# ============================================================
# BACKTEST
# ============================================================

for current_date in all_dates:

    for name, df in prepared_data.items():

        if current_date not in df.index:
            continue

        idx = df.index.get_loc(
            current_date
        )

        if idx < SIDEWAYS_LOOKBACK + 20:
            continue

        d = df.iloc[
            :idx + 1
        ]

        row = d.iloc[-1]

        required = [
            "Close",
            "EMA10",
            "EMA20",
            "RSI14"
        ]

        if row[required].isna().any():
            continue

        price = float(
            row["Close"]
        )

        rsi = float(
            row["RSI14"]
        )

        date_str = current_date.strftime(
            "%Y-%m-%d"
        )

        # ----------------------------------------------------
        # Sideways
        # ----------------------------------------------------

        is_sideways = detect_sideways(
            d
        )

        s = states[name]

        # ----------------------------------------------------
        # Current Profit
        # ----------------------------------------------------

        profit = 0.0

        if s["entry_price"] > 0:

            profit = (
                (price - s["entry_price"])
                / s["entry_price"]
            ) * 100

            if price > s["peak_price"]:
                s["peak_price"] = price

        # ====================================================
        # BUY
        # ====================================================

        if (
            s["position"] == 0.0
            and is_sideways
            and rsi < RSI_BUY
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
        # EXIT
        # ====================================================

        elif s["position"] == 1.0:

            peak_profit = (
                (
                    s["peak_price"]
                    -
                    s["entry_price"]
                )
                /
                s["entry_price"]
            ) * 100

            # ------------------------------------------------
            # RSI Sell
            # ------------------------------------------------

            rsi_exit = (
                rsi > RSI_SELL
            )

            # ------------------------------------------------
            # Hard Stop
            # ------------------------------------------------

            hard_stop = (
                profit <=
                -HARD_STOP_PERCENT
            )

            # ------------------------------------------------
            # Trailing Stop
            # ------------------------------------------------

            retracement = (
                (
                    s["peak_price"]
                    -
                    price
                )
                /
                s["peak_price"]
            ) * 100

            trailing_stop = (
                peak_profit >=
                TRAILING_ACTIVATION_PROFIT
                and
                retracement >=
                TRAILING_RETRACEMENT
            )

            # ------------------------------------------------
            # Exit Reason
            # ------------------------------------------------

            if hard_stop:

                exit_reason = "HARD_STOP"

            elif trailing_stop:

                exit_reason = "TRAILING_STOP"

            elif rsi_exit:

                exit_reason = "RSI_63"

            else:

                exit_reason = None

            # ------------------------------------------------
            # Execute Exit
            # ------------------------------------------------

            if exit_reason:

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

                    active[-1]["exit_reason"] = (
                        exit_reason
                    )

                total_portfolio_profit += profit

                s["position"] = 0.0

                s["entry_price"] = 0.0

                s["peak_price"] = 0.0

    portfolio_equity_curve.append(
        total_portfolio_profit
    )

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

    if t["profit_pct"] > 0

]

losing_trades = [

    t

    for t in closed_trades

    if t["profit_pct"] <= 0

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
        np.mean([
            t["profit_pct"]
            for t in winning_trades
        ])
    )

    if winning_trades

    else 0.0

)

avg_loss = (

    float(
        np.mean([
            t["profit_pct"]
            for t in losing_trades
        ])
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
    + 100
)

if len(equity) > 0:

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        (peak - equity)
        / peak
    ) * 100

    max_drawdown = float(
        np.max(drawdown)
    )

else:

    max_drawdown = 0.0

# ============================================================
# RESULTS
# ============================================================

results_summary = {

    "parameters": {

        "sideways_lookback":
            SIDEWAYS_LOOKBACK,

        "minimum_crosses":
            MIN_CROSSES,

        "ema10_slope_limit":
            EMA10_SLOPE_LIMIT,

        "ema20_slope_limit":
            EMA20_SLOPE_LIMIT,

        "average_gap_limit":
            AVG_GAP_LIMIT,

        "maximum_gap_limit":
            MAX_GAP_LIMIT,

        "minimum_sideways_score":
            MIN_SIDEWAYS_SCORE,

        "rsi_buy":
            RSI_BUY,

        "rsi_sell":
            RSI_SELL,

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
# SAVE
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
# PRINT
# ============================================================

print("\n" + "=" * 55)
print("WEEKLY SIDEWAYS + RSI STRATEGY")
print("=" * 55)

print(f"Sideways Lookback : {SIDEWAYS_LOOKBACK}")
print(f"Minimum Crosses   : {MIN_CROSSES}")
print(f"Sideways Score    : {MIN_SIDEWAYS_SCORE}/4")
print(f"RSI Buy           : < {RSI_BUY}")
print(f"RSI Sell          : > {RSI_SELL}")
print("-" * 55)
print(f"Total Trades      : {total_count}")
print(f"Winning Trades    : {wins_count}")
print(f"Losing Trades     : {losses_count}")
print(f"Win Rate          : {win_rate:.2f}%")
print(f"Total Profit      : {total_portfolio_profit:.2f}%")
print(f"Average Win       : {avg_win:.2f}%")
print(f"Average Loss      : {avg_loss:.2f}%")
print(f"Max Drawdown      : {max_drawdown:.2f}%")
print("=" * 55)￼Enterore":
            MIN_SIDEWAYS_SCORE,

        "rsi_buy":
            RSI_BUY,

        "rsi_sell":
            RSI_SELL,

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


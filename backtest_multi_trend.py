print("="*72)
print("EGX RESISTANCE BREAKOUT + VOLUME STRATEGY v1.2")
print("CONFIRMED RESISTANCE + PRESSURE + REAL BREAKOUT + VOLUME")
print("="*72)

import json
import os
import numpy as np
import pandas as pd

DB_FILE="egx_history_database_v2.json"
RESULT_FILE="backtest_results.json"
TRADES_FILE="backtest_trades.json"

INITIAL_CAPITAL=100000.0
MAX_POSITIONS=8
POSITION_SIZE=1.0/MAX_POSITIONS

# Swing
PIVOT_LEFT=3
PIVOT_RIGHT=3

# Resistance
LOOKBACK=120
ZONE_DISTANCE_PERCENT=1.0
MIN_RESISTANCE_REACTIONS=1

# Reaction
REACTION_WINDOW=3
MIN_REACTION_PERCENT=2.0

# Breakout
BREAKOUT_PERCENT=0.30
MIN_UPSIDE_PERCENT=6.0

# Pressure
PRESSURE_LOOKBACK=8
MIN_PRESSURE_CANDLES=4
MAX_PRESSURE_DISTANCE_PERCENT=2.0

# Volume
VOLUME_LOOKBACK=20
VOLUME_MULTIPLIER=2.50

# Stop
STOP_BUFFER_PERCENT=0.50

# Backtest
MIN_BARS=60


# ============================================================
# LOAD DATABASE
# ============================================================

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(DB_FILE)

with open(DB_FILE,encoding="utf-8") as f:
    db=json.load(f)

print(f"Database: {len(db)} symbols")


# ============================================================
# DATA
# ============================================================

def to_df(x):
    if isinstance(x,dict) and "data" in x and "columns" in x:
        rows=[]
        for d,v in x["data"].items():
            r=v.copy() if isinstance(v,dict) else dict(zip(x["columns"],v))
            r["Date"]=d
            rows.append(r)
        x=rows

    if not isinstance(x,list):
        return None

    df=pd.DataFrame(x)
    if df.empty:
        return None

    df.columns=[str(c).strip().capitalize() for c in df.columns]

    required=["Date","Open","High","Low","Close","Volume"]

    if not all(c in df.columns for c in required):
        return None

    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")

    for c in required[1:]:
        df[c]=pd.to_numeric(df[c],errors="coerce")

    df=df.dropna(subset=required)
    df=df.drop_duplicates("Date")
    df=df.sort_values("Date")
    df=df.reset_index(drop=True)

    return df


# ============================================================
# PIVOTS
# ============================================================

def detect_pivots(df):
    lows=df["Low"].values
    highs=df["High"].values
    n=len(df)

    pivot_lows=[]
    pivot_highs=[]

    for i in range(PIVOT_LEFT,n-PIVOT_RIGHT):
        left_lows=lows[i-PIVOT_LEFT:i]
        right_lows=lows[i+1:i+PIVOT_RIGHT+1]
        left_highs=highs[i-PIVOT_LEFT:i]
        right_highs=highs[i+1:i+PIVOT_RIGHT+1]

        if lows[i]<left_lows.min() and lows[i]<=right_lows.min():
            pivot_lows.append({
                "index":i,
                "confirmed_at":i+PIVOT_RIGHT,
                "price":float(lows[i])
            })

        if highs[i]>left_highs.max() and highs[i]>=right_highs.max():
            pivot_highs.append({
                "index":i,
                "confirmed_at":i+PIVOT_RIGHT,
                "price":float(highs[i])
            })

    return pivot_lows,pivot_highs


# ============================================================
# RESISTANCE REACTION
# ============================================================

def pivot_has_resistance_reaction(df,pivot_index,pivot_price):
    end=min(len(df),pivot_index+1+REACTION_WINDOW)
    future=df.iloc[pivot_index+1:end]

    if future.empty:
        return False

    min_close=float(future["Close"].min())

    reaction=(pivot_price-min_close)/pivot_price*100

    return reaction>=MIN_REACTION_PERCENT


# ============================================================
# RESISTANCE ZONES
# ============================================================

def build_resistance_zones(df,confirmed_highs,current_index):
    start=max(0,current_index-LOOKBACK+1)

    highs=[
        p for p in confirmed_highs
        if p["confirmed_at"]<=current_index and p["index"]>=start
    ]

    clusters=[]

    for p in sorted(highs,key=lambda x:x["index"]):
        placed=False

        for zone in clusters:
            distance=abs(p["price"]-zone["price"])/zone["price"]*100

            if distance<=ZONE_DISTANCE_PERCENT:
                zone["pivots"].append(p)
                zone["price"]=float(np.mean([x["price"] for x in zone["pivots"]]))
                placed=True
                break

        if not placed:
            clusters.append({
                "price":float(p["price"]),
                "pivots":[p]
            })

    zones=[]

    for zone in clusters:
        reactions=0

        for p in zone["pivots"]:
            reaction_end=p["index"]+1+REACTION_WINDOW

            if reaction_end>current_index:
                continue

            if pivot_has_resistance_reaction(df,p["index"],p["price"]):
                reactions+=1

        if reactions<MIN_RESISTANCE_REACTIONS:
            continue

        prices=[p["price"] for p in zone["pivots"]]

        zone_low=min(prices)*(1-ZONE_DISTANCE_PERCENT/100)
        zone_high=max(prices)*(1+ZONE_DISTANCE_PERCENT/100)

        zones.append({
            "direction":"RESISTANCE",
            "price":round(float(np.mean(prices)),4),
            "low":round(float(zone_low),4),
            "high":round(float(zone_high),4),
            "reactions":reactions,
            "first_pivot":min(p["index"] for p in zone["pivots"]),
            "last_pivot":max(p["index"] for p in zone["pivots"])
        })

    return zones


# ============================================================
# REAL BREAKOUT
# ============================================================

def find_broken_resistance(resistances,prev_close,close):
    candidates=[]

    for resistance in resistances:
        level=float(resistance["high"])
        required=level*(1+BREAKOUT_PERCENT/100)

        # IMPORTANT:
        # Previous candle must NOT already be above
        # the breakout level.
        if prev_close>required:
            continue

        # Current candle must break it.
        if close<=required:
            continue

        candidates.append(resistance)

    if not candidates:
        return None

    candidates.sort(key=lambda r:r["high"],reverse=True)

    return candidates[0]


# ============================================================
# NEXT RESISTANCE
# ============================================================

def find_next_resistance(resistances,entry_price,broken_resistance):
    candidates=[]

    for resistance in resistances:
        price=float(resistance["price"])

        if price<=entry_price:
            continue

        if (
            resistance["first_pivot"]==broken_resistance["first_pivot"]
            and
            resistance["last_pivot"]==broken_resistance["last_pivot"]
        ):
            continue

        upside=(price-entry_price)/entry_price*100

        if upside<MIN_UPSIDE_PERCENT:
            continue

        candidates.append(resistance)

    if not candidates:
        return None

    return min(candidates,key=lambda r:r["price"])


# ============================================================
# VOLUME
# ============================================================

def volume_confirmed(df,index):
    start=index-VOLUME_LOOKBACK

    if start<0:
        return False

    previous=df.iloc[start:index]["Volume"]

    if len(previous)<VOLUME_LOOKBACK:
        return False

    avg=float(previous.mean())

    if avg<=0:
        return False

    current=float(df.iloc[index]["Volume"])

    return current>=avg*VOLUME_MULTIPLIER


def volume_ratio(df,index):
    start=index-VOLUME_LOOKBACK

    if start<0:
        return 0

    avg=float(df.iloc[start:index]["Volume"].mean())

    if avg<=0:
        return 0

    return float(df.iloc[index]["Volume"])/avg


# ============================================================
# PRESSURE
# ============================================================

def count_pressure_candles(df,index,resistance):
    start=index-PRESSURE_LOOKBACK

    if start<0:
        return 0

    candles=df.iloc[start:index]

    resistance_high=float(resistance["high"])
    resistance_low=float(resistance["low"])

    count=0

    for _,candle in candles.iterrows():
        high=float(candle["High"])
        close=float(candle["Close"])

        distance=(resistance_high-close)/resistance_high*100

        near_resistance=(
            0<=distance<=MAX_PRESSURE_DISTANCE_PERCENT
        )

        touched_zone=high>=resistance_low

        if near_resistance or touched_zone:
            count+=1

    return count


def pressure_confirmed(df,index,resistance):
    return count_pressure_candles(df,index,resistance)>=MIN_PRESSURE_CANDLES


# ============================================================
# BREAKOUT CONFIRMATION
# ============================================================

def valid_breakout(row,prev_row,resistance):
    close=float(row["Close"])
    prev_close=float(prev_row["Close"])

    level=float(resistance["high"])
    required=level*(1+BREAKOUT_PERCENT/100)

    return prev_close<=required and close>required


# ============================================================
# CLOSE TRADE
# ============================================================

def close_trade(position,date,price,reason):
    profit=(price-position["entry_price"])/position["entry_price"]*100

    return {
        "symbol":position["symbol"],
        "status":"CLOSED",
        "entry_date":position["entry_date"],
        "entry_price":round(position["entry_price"],4),
        "exit_date":date,
        "exit_price":round(price,4),
        "profit_pct":round(profit,2),
        "exit_reason":reason,
        "broken_resistance":position["broken_resistance"],
        "resistance_reactions":position["resistance_reactions"],
        "target_resistance":position["target_resistance"],
        "upside_to_target":position["upside_to_target"],
        "breakout_volume_ratio":position["breakout_volume_ratio"],
        "pressure_candles":position["pressure_candles"]
    }


# ============================================================
# BACKTEST
# ============================================================

def backtest(sym,df):
    _,pivot_highs=detect_pivots(df)

    position=None
    trades=[]

    for i in range(MIN_BARS,len(df)):
        row=df.iloc[i]
        prev_row=df.iloc[i-1]

        date=row["Date"].strftime("%Y-%m-%d")
        high=float(row["High"])
        low=float(row["Low"])
        close=float(row["Close"])

        resistances=build_resistance_zones(df,pivot_highs,i)

        # ----------------------------------------------------
        # MANAGE POSITION
        # ----------------------------------------------------

        if position is not None:
            broken=position["broken_resistance_zone"]
            target_zone=position["target_resistance_zone"]

            stop_price=(
                float(broken["low"])
                *(1-STOP_BUFFER_PERCENT/100)
            )

            # Stop has priority.
            if low<=stop_price:
                trades.append(
                    close_trade(
                        position,
                        date,
                        stop_price,
                        "BREAKOUT_FAILURE"
                    )
                )
                position=None
                continue

            if target_zone is not None:
                target=float(target_zone["price"])

                if high>=target:
                    trades.append(
                        close_trade(
                            position,
                            date,
                            target,
                            "RESISTANCE"
                        )
                    )
                    position=None
                    continue

            continue

        # ----------------------------------------------------
        # FIND REAL BREAKOUT
        # ----------------------------------------------------

        broken=find_broken_resistance(
            resistances,
            float(prev_row["Close"]),
            close
        )

        if broken is None:
            continue

        if not valid_breakout(row,prev_row,broken):
            continue

        # ----------------------------------------------------
        # PRESSURE
        # ----------------------------------------------------

        pressure_count=count_pressure_candles(
            df,
            i,
            broken
        )

        if pressure_count<MIN_PRESSURE_CANDLES:
            continue

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        if not volume_confirmed(df,i):
            continue

        vr=volume_ratio(df,i)

        # ----------------------------------------------------
        # NEXT RESISTANCE
        # ----------------------------------------------------

        target_zone=find_next_resistance(
            resistances,
            close,
            broken
        )

        if target_zone is None:
            continue

        # ----------------------------------------------------
        # NEXT DAY ENTRY
        # ----------------------------------------------------

        if i+1>=len(df):
            continue

        next_row=df.iloc[i+1]

        entry_date=next_row["Date"].strftime("%Y-%m-%d")
        entry_price=float(next_row["Open"])
        target_price=float(target_zone["price"])

        actual_upside=(target_price-entry_price)/entry_price*100

        if actual_upside<MIN_UPSIDE_PERCENT:
            continue

        position={
            "symbol":sym,
            "entry_date":entry_date,
            "entry_price":entry_price,
            "broken_resistance_zone":broken,
            "broken_resistance":round(float(broken["price"]),4),
            "resistance_reactions":broken["reactions"],
            "target_resistance_zone":target_zone,
            "target_resistance":round(target_price,4),
            "upside_to_target":round(actual_upside,2),
            "breakout_volume_ratio":round(vr,2),
            "pressure_candles":pressure_count
        }

    # --------------------------------------------------------
    # END OF DATA
    # --------------------------------------------------------

    if position is not None:
        last=df.iloc[-1]

        trade=close_trade(
            position,
            last["Date"].strftime("%Y-%m-%d"),
            float(last["Close"]),
            "END_OF_DATA"
        )

        trade["status"]="OPEN"
        trades.append(trade)

    return trades


# ============================================================
# RUN
# ============================================================

all_trades=[]

for sym,data in db.items():
    if sym.upper() in {"EGX30","EGX70","EGX100"}:
        continue

    df=to_df(data)

    if df is None:
        print(f"⚠️ {sym}: invalid data or Volume missing")
        continue

    if len(df)<MIN_BARS:
        print(f"⚠️ {sym}: insufficient data ({len(df)})")
        continue

    trades=backtest(sym,df)
    all_trades.extend(trades)

    closed_count=sum(t["status"]=="CLOSED" for t in trades)

    print(f"{sym:8} | {closed_count:3} closed")


# ============================================================
# SORT
# ============================================================

all_trades.sort(key=lambda x:x["entry_date"])

closed=[t for t in all_trades if t["status"]=="CLOSED"]
opened=[t for t in all_trades if t["status"]=="OPEN"]


# ============================================================
# STATISTICS
# ============================================================

profits=[float(t["profit_pct"]) for t in closed]
wins=[p for p in profits if p>0]
losses=[p for p in profits if p<=0]

n=len(profits)

winrate=len(wins)/n*100 if n else 0
sumprofit=sum(profits)

avgwin=float(np.mean(wins)) if wins else 0
avgloss=float(np.mean(losses)) if losses else 0


# ============================================================
# PORTFOLIO
# ============================================================

portfolio=INITIAL_CAPITAL
equity=[portfolio]
portfolio_history=[]

for trade in closed:
    trade_return=trade["profit_pct"]/100*POSITION_SIZE
    portfolio*=1+trade_return
    equity.append(portfolio)

    portfolio_history.append({
        "date":trade["exit_date"],
        "symbol":trade["symbol"],
        "trade_return_percent":trade["profit_pct"],
        "portfolio_return_percent":round(trade_return*100,4),
        "portfolio_value":round(portfolio,2)
    })

compound_return=(portfolio/INITIAL_CAPITAL-1)*100


# ============================================================
# DRAWDOWN
# ============================================================

peak=INITIAL_CAPITAL
max_drawdown=0.0

for value in equity:
    peak=max(peak,value)
    drawdown=(peak-value)/peak*100
    max_drawdown=max(max_drawdown,drawdown)


# ============================================================
# EXIT ANALYSIS
# ============================================================

exit_analysis={}

for trade in closed:
    reason=trade.get("exit_reason","UNKNOWN")
    exit_analysis[reason]=exit_analysis.get(reason,0)+1


# ============================================================
# BEST / WORST
# ============================================================

best=max(closed,key=lambda x:x["profit_pct"]) if closed else None
worst=min(closed,key=lambda x:x["profit_pct"]) if closed else None


# ============================================================
# RESULT
# ============================================================

result={
    "strategy":"EGX Resistance Breakout + Volume Strategy v1.2",
    "description":"Confirmed swing resistance breakout with real breakout transition, pre-breakout pressure, strong volume, next resistance target and structural failure stop.",
    "parameters":{
        "pivot_left":PIVOT_LEFT,
        "pivot_right":PIVOT_RIGHT,
        "lookback":LOOKBACK,
        "zone_distance_percent":ZONE_DISTANCE_PERCENT,
        "minimum_resistance_reactions":MIN_RESISTANCE_REACTIONS,
        "reaction_window":REACTION_WINDOW,
        "minimum_reaction_percent":MIN_REACTION_PERCENT,
        "breakout_percent":BREAKOUT_PERCENT,
        "minimum_upside_percent":MIN_UPSIDE_PERCENT,
        "pressure_lookback":PRESSURE_LOOKBACK,
        "minimum_pressure_candles":MIN_PRESSURE_CANDLES,
        "max_pressure_distance_percent":MAX_PRESSURE_DISTANCE_PERCENT,
        "volume_lookback":VOLUME_LOOKBACK,
        "volume_multiplier":VOLUME_MULTIPLIER,
        "stop_buffer_percent":STOP_BUFFER_PERCENT,
        "max_positions":MAX_POSITIONS,
        "position_size_percent":POSITION_SIZE*100
    },
    "statistics":{
        "total_trades":n,
        "winning_trades":len(wins),
        "losing_trades":len(losses),
        "win_rate_percent":round(winrate,2),
        "sum_trade_profit_percent":round(sumprofit,2),
        "realistic_compound_return_percent":round(compound_return,2),
        "average_win_percent":round(avgwin,2),
        "average_loss_percent":round(avgloss,2),
        "maximum_drawdown_percent":round(max_drawdown,2),
        "open_positions":len(opened)
    },
    "exit_analysis":exit_analysis,
    "best_trade":best,
    "worst_trade":worst,
    "open_positions":opened,
    "portfolio_equity":portfolio_history,
    "trades":all_trades
}


# ============================================================
# SAVE
# ============================================================

with open(RESULT_FILE,"w",encoding="utf-8") as f:
    json.dump(result,f,ensure_ascii=False,indent=2)

with open(TRADES_FILE,"w",encoding="utf-8") as f:
    json.dump(all_trades,f,ensure_ascii=False,indent=2)


# ============================================================
# FINAL
# ============================================================

print("\n"+"="*72)
print("FINAL RESULTS")
print("="*72)

print(f"Trades              : {n}")
print(f"Winners             : {len(wins)}")
print(f"Losers              : {len(losses)}")
print(f"Win Rate            : {winrate:.2f}%")
print(f"Sum Profit          : {sumprofit:.2f}%")
print(f"Compound Return     : {compound_return:.2f}%")
print(f"Average Win         : {avgwin:.2f}%")
print(f"Average Loss        : {avgloss:.2f}%")
print(f"Maximum Drawdown    : {max_drawdown:.2f}%")
print(f"Open Positions      : {len(opened)}")

print("\nEXIT ANALYSIS")

for reason,count in exit_analysis.items():
    print(f"{reason:22}: {count}")

if best:
    print(
        f"\nBEST  : {best['symbol']} | "
        f"{best['profit_pct']:.2f}% | "
        f"{best['entry_date']} -> {best['exit_date']}"
    )

if worst:
    print(
        f"WORST : {worst['symbol']} | "
        f"{worst['profit_pct']:.2f}% | "
        f"{worst['entry_date']} -> {worst['exit_date']}"
    )

print(f"\nSaved: {RESULT_FILE}, {TRADES_FILE}")
print("="*72)

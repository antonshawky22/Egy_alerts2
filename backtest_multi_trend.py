import json
import os
import pandas as pd
import numpy as np

DB_FILE="egx_weekly_database_v1.json"
RESULTS_FILE="backtest_results.json"
TRADES_FILE="backtest_trades.json"

symbols={
"OLFI":"OLFI","EMFD":"EMFD","ETEL":"ETEL","EAST":"EAST",
"EFIH":"EFIH","ABUK":"ABUK","OIH":"OIH","SWDY":"SWDY",
"ISPH":"ISPH","ATQA":"ATQA","MTIE":"MTIE","HRHO":"HRHO",
"ORWE":"ORWE","JUFO":"JUFO","DSCW":"DSCW","SUGR":"SUGR",
"ELSH":"ELSH","RMDA":"RMDA","RAYA":"RAYA","EEII":"EEII",
"MPCO":"MPCO","GBCO":"GBCO","TMGH":"TMGH","ORHD":"ORHD",
"AMOC":"AMOC","FWRY":"FWRY","COMI":"COMI","ADIB":"ADIB",
"PHDC":"PHDC","MCQE":"MCQE","SKPC":"SKPC","EGAL":"EGAL"
}

RSI_PERIOD=14
RSI_BUY=33
RSI_SELL_1=60
RSI_SELL_2=70
HARD_STOP_PERCENT=5.0

if not os.path.exists(DB_FILE):
    raise FileNotFoundError(f"Database file not found: {DB_FILE}")

with open(DB_FILE,"r",encoding="utf-8") as f:
    raw_database=json.load(f)

prepared_data={}

for name in symbols:
    if name not in raw_database:
        continue

    content=raw_database[name]

    if "data" not in content or "columns" not in content:
        continue

    try:
        df=pd.DataFrame.from_dict(
            content["data"],
            orient="index",
            columns=content["columns"]
        )

        df.index=pd.to_datetime(df.index)
        df=df.sort_index()

        if len(df)<RSI_PERIOD+2:
            continue

        df["Close"]=pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        df=df.dropna(subset=["Close"])

        delta=df["Close"].diff()

        gain=delta.clip(lower=0)
        loss=-delta.clip(upper=0)

        avg_gain=gain.ewm(
            alpha=1/RSI_PERIOD,
            adjust=False
        ).mean()

        avg_loss=loss.ewm(
            alpha=1/RSI_PERIOD,
            adjust=False
        ).mean()

        rs=avg_gain/avg_loss.replace(
            0,
            np.nan
        )

        df["RSI14"]=100-(100/(1+rs))

        prepared_data[name]=df

    except Exception as e:
        print(f"Error preparing {name}: {e}")

if not prepared_data:
    raise RuntimeError("No valid symbols found.")

all_dates=sorted(
    set().union(
        *[df.index for df in prepared_data.values()]
    )
)

states={
    name:{
        "position":0.0,
        "entry_price":0.0,
        "first_sell_done":False
    }
    for name in prepared_data
}

trades_history=[]
total_portfolio_profit=0.0
portfolio_equity_curve=[]

for current_date in all_dates:

    for name,df in prepared_data.items():

        if current_date not in df.index:
            continue

        idx=df.index.get_loc(current_date)

        if idx<RSI_PERIOD+1:
            continue

        row=df.iloc[idx]

        if pd.isna(row["RSI14"]) or pd.isna(row["Close"]):
            continue

        price=float(row["Close"])
        rsi=float(row["RSI14"])

        date_str=current_date.strftime("%Y-%m-%d")

        s=states[name]

        # ==========================================
        # BUY - RSI < 33
        # ==========================================

        if s["position"]==0.0 and rsi<RSI_BUY:

            s["position"]=1.0
            s["entry_price"]=price
            s["first_sell_done"]=False

            trades_history.append({
                "symbol":name,
                "status":"OPEN",
                "entry_date":date_str,
                "entry_price":round(price,2),
                "position":1.0,
                "sales":[]
            })

        # ==========================================
        # MANAGE OPEN POSITION
        # ==========================================

        elif s["position"]>0.0:

            profit=(
                (price-s["entry_price"])
                /s["entry_price"]
            )*100

            active=[
                t for t in trades_history
                if (
                    t["symbol"]==name
                    and
                    t["status"]=="OPEN"
                )
            ]

            if not active:
                continue

            trade=active[-1]

            # ======================================
            # HARD STOP
            # ======================================

            if profit<=-HARD_STOP_PERCENT:

                sold_position=s["position"]

                trade["sales"].append({
                    "date":date_str,
                    "price":round(price,2),
                    "position_sold":sold_position,
                    "profit_pct":round(profit,2),
                    "reason":"HARD_STOP"
                })

                trade["status"]="CLOSED"
                trade["exit_date"]=date_str
                trade["exit_price"]=round(price,2)
                trade["profit_pct"]=round(profit,2)
                trade["exit_reason"]="HARD_STOP"

                total_portfolio_profit+=profit*sold_position

                s["position"]=0.0
                s["entry_price"]=0.0
                s["first_sell_done"]=False

                continue

            # ======================================
            # FIRST SELL - RSI >= 60
            # SELL 50%
            # ======================================

            if (
                not s["first_sell_done"]
                and
                rsi>=RSI_SELL_1
            ):

                sold=0.5
                sale_profit=profit

                trade["sales"].append({
                    "date":date_str,
                    "price":round(price,2),
                    "position_sold":sold,
                    "profit_pct":round(sale_profit,2),
                    "reason":"RSI_60"
                })

                total_portfolio_profit+=(
                    sale_profit*sold
                )

                s["position"]=0.5
                s["first_sell_done"]=True

            # ======================================
            # SECOND SELL - RSI >= 70
            # SELL REMAINING 50%
            # ======================================

            elif (
                s["first_sell_done"]
                and
                rsi>=RSI_SELL_2
            ):

                sold=0.5
                sale_profit=profit

                trade["sales"].append({
                    "date":date_str,
                    "price":round(price,2),
                    "position_sold":sold,
                    "profit_pct":round(sale_profit,2),
                    "reason":"RSI_70"
                })

                total_portfolio_profit+=(
                    sale_profit*sold
                )

                trade["status"]="CLOSED"
                trade["exit_date"]=date_str
                trade["exit_price"]=round(price,2)
                trade["profit_pct"]=round(
                    (
                        trade["sales"][0]["profit_pct"]*0.5
                        +
                        sale_profit*0.5
                    ),
                    2
                )
                trade["exit_reason"]="RSI_70"

                s["position"]=0.0
                s["entry_price"]=0.0
                s["first_sell_done"]=False

    portfolio_equity_curve.append(
        total_portfolio_profit
    )

# ============================================================
# STATISTICS
# ============================================================

closed_trades=[
    t for t in trades_history
    if t["status"]=="CLOSED"
]

winning_trades=[
    t for t in closed_trades
    if t["profit_pct"]>0
]

losing_trades=[
    t for t in closed_trades
    if t["profit_pct"]<=0
]

total_count=len(closed_trades)
wins_count=len(winning_trades)
losses_count=len(losing_trades)

win_rate=(
    wins_count/total_count*100
    if total_count>0
    else 0.0
)

avg_win=(
    float(np.mean([
        t["profit_pct"]
        for t in winning_trades
    ]))
    if winning_trades
    else 0.0
)

avg_loss=(
    float(np.mean([
        t["profit_pct"]
        for t in losing_trades
    ]))
    if losing_trades
    else 0.0
)

equity=np.array(
    portfolio_equity_curve
)+100

if len(equity)>0:

    peak=np.maximum.accumulate(equity)

    drawdown=(
        (peak-equity)/peak
    )*100

    max_drawdown=float(
        np.max(drawdown)
    )

else:
    max_drawdown=0.0

results_summary={
    "strategy":"Weekly RSI 33/60/70 Partial Exit",
    "parameters":{
        "rsi_period":RSI_PERIOD,
        "rsi_buy":RSI_BUY,
        "rsi_sell_1":RSI_SELL_1,
        "rsi_sell_2":RSI_SELL_2,
        "first_sell_percent":50,
        "second_sell_percent":50,
        "hard_stop_percent":HARD_STOP_PERCENT
    },
    "statistics":{
        "total_trades":total_count,
        "winning_trades":wins_count,
        "losing_trades":losses_count,
        "win_rate_percent":round(win_rate,2),
        "total_profit_percent":round(
            total_portfolio_profit,
            2
        ),
        "average_winning_trade_percent":round(
            avg_win,
            2
        ),
        "average_losing_trade_percent":round(
            avg_loss,
            2
        ),
        "maximum_drawdown_percent":round(
            max_drawdown,
            2
        )
    }
}

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

print("="*55)
print("WEEKLY RSI 33/60/70 BACKTEST")
print("="*55)
print(f"RSI Buy       : < {RSI_BUY}")
print(f"RSI Sell 1    : >= {RSI_SELL_1} (50%)")
print(f"RSI Sell 2    : >= {RSI_SELL_2} (50%)")
print(f"Hard Stop     : {HARD_STOP_PERCENT}%")
print("-"*55)
print(f"Total Trades  : {total_count}")
print(f"Winning       : {wins_count}")
print(f"Losing        : {losses_count}")
print(f"Win Rate      : {win_rate:.2f}%")
print(f"Total Profit  : {total_portfolio_profit:.2f}%")
print(f"Average Win   : {avg_win:.2f}%")
print(f"Average Loss  : {avg_loss:.2f}%")
print(f"Max Drawdown  : {max_drawdown:.2f}%")
print("="*55)

print("="*80)
print("EGX LADDER CYCLE SYSTEM - PERIOD STABILITY BACKTEST v1.2")
print("SAME LIVE PARAMETERS - PERIOD STABILITY TEST")
print("="*80)

import json,os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
DB_FILE="egx_history_database_v2.json"
RESULT_FILE="backtest_results.json"
TRADES_FILE="backtest_trades.json"
STOCK_SUMMARY_FILE="ladder_backtest_summary_by_stock.json"
INITIAL_CAPITAL=100000.0

# ============================================================
# PERIODS - ONE STABILITY TEST
# ============================================================
PERIODS=[
    ("2024-08-01","2024-12-31"),
    ("2025-01-01","2025-06-30"),
    ("2025-07-01","2025-12-31"),
    ("2026-01-01","2099-12-31")
]

# ============================================================
# PARAMETERS - UNCHANGED
# ============================================================
RSI_PERIOD=14
EMA100_PERIOD=80
RUNUP_LOOKBACK=160
MAX_RUNUP_PERCENT=80
MAX_GAP_DOWN_PERCENT=-5
BUY1_RSI=50
BUY2_RSI=45
BUY3_RSI=40
SELL1_RSI=60
SELL1_MIN_PROFIT=15
SELL2_MIN_POSITION=0.30
SELL2_MAX_POSITION=0.70
SELL2_RSI=72
SELL2_MIN_PROFIT=18
SELL3_RSI=82
SELL3_MIN_PROFIT=20
STOP_L1=-18
STOP_L2=-17
STOP_L3=-13
TRAILING_TRIGGER=36
TRAILING_GIVEBACK=11
MIN_BARS=40

MAX_PORTFOLIO_POSITIONS=8
POSITION_SIZE=1.0/MAX_PORTFOLIO_POSITIONS

# ============================================================
# SYMBOLS
# ============================================================
SYMBOLS={
"OLFI":"OLFI","EMFD":"EMFD","ETEL":"ETEL","EAST":"EAST","EFIH":"EFIH",
"ABUK":"ABUK","OIH":"OIH","SWDY":"SWDY","ISPH":"ISPH","ATQA":"ATQA",
"MTIE":"MTIE","HRHO":"HRHO","ORWE":"ORWE","JUFO":"JUFO","DSCW":"DSCW",
"SUGR":"SUGR","ELSH":"ELSH","RMDA":"RMDA","RAYA":"RAYA","EEII":"EEII",
"MPCO":"MPCO","GBCO":"GBCO","TMGH":"TMGH","ORHD":"ORHD","AMOC":"AMOC",
"FWRY":"FWRY","COMI":"COMI","ADIB":"ADIB","PHDC":"PHDC","MCQE":"MCQE",
"SKPC":"SKPC","EGAL":"EGAL"
}

# ============================================================
# LOAD DATABASE
# ============================================================
if not os.path.exists(DB_FILE):
    raise FileNotFoundError(DB_FILE)

with open(DB_FILE,"r",encoding="utf-8") as f:
    raw_database=json.load(f)

print(f"Database symbols: {len(raw_database)}")

# ============================================================
# DATA LOADER
# ============================================================
def fetch_local_data(name):
    try:
        if name not in raw_database:return None
        content=raw_database[name]
        if "columns" not in content or "data" not in content:return None
        df=pd.DataFrame.from_dict(content["data"],orient="index",columns=content["columns"])
        df.index.name="Date"
        df.index=pd.to_datetime(df.index,errors="coerce")
        df=df.sort_index()
        required=["Open","High","Low","Close"]
        if any(c not in df.columns for c in required):return None
        for c in required:df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=required)
        df=df[~df.index.duplicated(keep="last")]
        return df
    except Exception as e:
        print(f"⚠️ Error processing {name}: {e}")
        return None

# ============================================================
# RSI
# ============================================================
def rsi(series,period=14):
    if len(series)<period+1:return pd.Series(np.nan,index=series.index)
    delta=series.diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(com=period-1,adjust=False).mean()
    avg_loss=loss.ewm(com=period-1,adjust=False).mean()
    rs=avg_gain/avg_loss
    return 100-(100/(1+rs))

# ============================================================
# INDICATORS
# ============================================================
def prepare_data(df):
    df=df.copy()
    close=df["Close"]
    df["EMA100"]=close.ewm(span=EMA100_PERIOD,adjust=False).mean()
    df["RSI"]=rsi(close,RSI_PERIOD)
    return df

# ============================================================
# AVG PRICE
# ============================================================
def update_avg(old_avg,old_pos,new_price,new_pos):
    if new_pos==0:return 0.0
    added_pos=new_pos-old_pos
    if added_pos<=0:return old_avg
    return ((old_avg*old_pos)+(new_price*added_pos))/new_pos

# ============================================================
# FINAL PNL
# ============================================================
def calculate_final_pnl(realized_tracker,current_position,current_profit):
    tracker=list(realized_tracker)
    tracker.append((current_position,current_profit))
    weight_sum=sum(w for w,_ in tracker)
    if weight_sum<=0:return current_profit
    return sum(p*w for w,p in tracker)/weight_sum

# ============================================================
# TRADE
# ============================================================
def create_trade(symbol,cycle,date,price):
    return {
        "symbol":symbol,"cycle":cycle,"status":"OPEN",
        "first_entry":{"date":date,"price":round(price,4),"position_added":0.33},
        "second_entry":None,"third_entry":None,"exits":[],
        "last_average_price":round(price,4),"final_exit_price":None,
        "exit_date":None,"profit_pct":None,"max_position":0.33,
        "peak_profit":0.0,"exit_reason":None
    }

def close_trade(trade,date,price,final_profit,reason):
    trade["status"]="CLOSED"
    trade["final_exit_price"]=round(price,4)
    trade["exit_date"]=date
    trade["profit_pct"]=round(final_profit,2)
    trade["exit_reason"]=reason
    trade["days_in_trade"]=(pd.to_datetime(date)-pd.to_datetime(trade["first_entry"]["date"])).days
    return trade

# ============================================================
# BACKTEST ONE STOCK / ONE PERIOD
# ============================================================
def backtest_stock(symbol,df,start_date,end_date):
    df=prepare_data(df)
    start=pd.Timestamp(start_date)
    end=pd.Timestamp(end_date)

    state={"cycle":1,"position":0.0,"avg_price":0.0,"peak_profit":0.0,"realized_pnl_tracker":[]}
    trades=[]
    signals=[]

    # IMPORTANT: indicators use full history, but trading starts at period start
    for i in range(MIN_BARS,len(df)):
        current_date=df.index[i]
        if current_date<start or current_date>end:continue

        row=df.iloc[i]
        date=current_date.strftime("%Y-%m-%d")
        price=float(row["Close"])
        rsi_val=float(row["RSI"])
        ema100=float(row["EMA100"])

        if pd.isna(rsi_val) or pd.isna(ema100):continue

        position=state["position"]
        avg_price=state["avg_price"]

        lookback=min(len(df.iloc[:i+1]),RUNUP_LOOKBACK)
        recent=df.iloc[i+1-lookback:i+1]
        lowest_80=float(recent["Low"].min())
        highest_80=float(recent["High"].max())
        run_up_percent=((highest_80-lowest_80)/lowest_80)*100 if lowest_80>0 else 0.0
        safe_to_buy=run_up_percent<=MAX_RUNUP_PERCENT

        if i<3:continue

        gap1=((df["Open"].iloc[i]-df["Close"].iloc[i-1])/df["Close"].iloc[i-1])*100
        gap2=((df["Open"].iloc[i-1]-df["Close"].iloc[i-2])/df["Close"].iloc[i-2])*100
        gap3=((df["Open"].iloc[i-2]-df["Close"].iloc[i-3])/df["Close"].iloc[i-3])*100
        no_gap_down=gap1>MAX_GAP_DOWN_PERCENT and gap2>MAX_GAP_DOWN_PERCENT and gap3>MAX_GAP_DOWN_PERCENT

        ema_up=(
            df["EMA100"].iloc[i]>df["EMA100"].iloc[i-5] and
            df["EMA100"].iloc[i-5]>df["EMA100"].iloc[i-10] and
            df["EMA100"].iloc[i]>df["EMA100"].iloc[i-10]*1.002 and
            price<=df["EMA100"].iloc[i]*1.07
        )

        buy1=safe_to_buy and ema_up and no_gap_down and rsi_val<=BUY1_RSI
        buy2=safe_to_buy and ema_up and no_gap_down and rsi_val<=BUY2_RSI
        buy3=safe_to_buy and ema_up and no_gap_down and rsi_val<=BUY3_RSI

        profit=((price-avg_price)/avg_price)*100 if avg_price>0 else 0.0

        sell1=position>0.65 and rsi_val>=SELL1_RSI and profit>SELL1_MIN_PROFIT
        sell2=0.30<position<=SELL2_MAX_POSITION and rsi_val>=SELL2_RSI and profit>SELL2_MIN_PROFIT
        sell3=position>0 and rsi_val>=SELL3_RSI and profit>SELL3_MIN_PROFIT

        action=None

        # BUY L1
        if position==0 and buy1:
            state["position"]=0.33
            state["avg_price"]=price
            state["peak_profit"]=0.0
            state["realized_pnl_tracker"]=[]
            profit=0.0
            action="BUY L1"
            trades.append(create_trade(symbol,state["cycle"],date,price))
            signals.append({"symbol":symbol,"date":date,"action":"BUY L1","price":round(price,4),"rsi":round(rsi_val,2),"position":0.33})

        # BUY L2
        elif 0.32<position<0.50 and buy2 and price<avg_price*0.97:
            old_pos=position
            state["position"]=0.66
            state["avg_price"]=update_avg(avg_price,old_pos,price,state["position"])
            profit=((price-state["avg_price"])/state["avg_price"])*100
            action="BUY L2"
            trade=trades[-1]
            trade["second_entry"]={"date":date,"price":round(price,4),"position_added":0.33}
            trade["last_average_price"]=round(state["avg_price"],4)
            trade["max_position"]=max(trade["max_position"],0.66)
            signals.append({"symbol":symbol,"date":date,"action":"BUY L2","price":round(price,4),"rsi":round(rsi_val,2),"position":0.66,"avg_price":round(state["avg_price"],4)})

        # BUY L3
        elif 0.65<position<1.0 and buy3 and price<avg_price*0.94:
            old_pos=position
            state["position"]=1.0
            state["avg_price"]=update_avg(avg_price,old_pos,price,state["position"])
            profit=((price-state["avg_price"])/state["avg_price"])*100
            action="BUY L3"
            trade=trades[-1]
            trade["third_entry"]={"date":date,"price":round(price,4),"position_added":0.34}
            trade["last_average_price"]=round(state["avg_price"],4)
            trade["max_position"]=1.0
            signals.append({"symbol":symbol,"date":date,"action":"BUY L3","price":round(price,4),"rsi":round(rsi_val,2),"position":1.0,"avg_price":round(state["avg_price"],4)})

        if profit>state["peak_profit"]:state["peak_profit"]=profit

        if trades and trades[-1]["status"]=="OPEN":
            trades[-1]["peak_profit"]=max(trades[-1]["peak_profit"],state["peak_profit"])

        initial_pos=position

        if initial_pos>0 and state["position"]>0:
            stop_loss_triggered=False
            trailing_stop_triggered=False

            if state["position"]<=0.33 and profit<=STOP_L1:
                stop_loss_triggered=True
            elif state["position"]<=0.66 and profit<=STOP_L2:
                stop_loss_triggered=True
            elif state["position"]==1.0 and profit<=STOP_L3:
                stop_loss_triggered=True

            if state["peak_profit"]>TRAILING_TRIGGER and state["peak_profit"]-profit>=TRAILING_GIVEBACK:
                trailing_stop_triggered=True

            if trailing_stop_triggered and not stop_loss_triggered:
                action="TRAILING STOP"
                final_profit=calculate_final_pnl(state["realized_pnl_tracker"],state["position"],profit)
                trade=trades[-1]
                close_trade(trade,date,price,final_profit,"TRAILING STOP")
                trade["stop_trigger_profit"]=round(profit,2)
                state["position"]=0.0
                signals.append({"symbol":symbol,"date":date,"action":"TRAILING STOP","price":round(price,4),"rsi":round(rsi_val,2),"profit":round(final_profit,2)})

            elif stop_loss_triggered:
                action="STOP LOSS"
                final_profit=calculate_final_pnl(state["realized_pnl_tracker"],state["position"],profit)
                trade=trades[-1]
                close_trade(trade,date,price,final_profit,"STOP LOSS")
                trade["stop_trigger_profit"]=round(profit,2)
                state["position"]=0.0
                signals.append({"symbol":symbol,"date":date,"action":"STOP LOSS","price":round(price,4),"rsi":round(rsi_val,2),"profit":round(final_profit,2)})

            elif sell3:
                action="EXIT FULL"
                final_profit=calculate_final_pnl(state["realized_pnl_tracker"],state["position"],profit)
                trade=trades[-1]
                close_trade(trade,date,price,final_profit,"EXIT FULL")
                state["position"]=0.0
                signals.append({"symbol":symbol,"date":date,"action":"EXIT FULL","price":round(price,4),"rsi":round(rsi_val,2),"profit":round(final_profit,2)})

            elif sell2:
                sell_amount=min(0.33,state["position"])
                state["realized_pnl_tracker"].append((sell_amount,profit))
                state["position"]=round(state["position"]-sell_amount,2)
                action="SELL L2"
                trade=trades[-1]
                trade["exits"].append({"date":date,"type":"SELL L2","price":round(price,4),"position_sold":round(sell_amount,2),"profit":round(profit,2)})
                signals.append({"symbol":symbol,"date":date,"action":"SELL L2","price":round(price,4),"rsi":round(rsi_val,2),"profit":round(profit,2),"position":state["position"]})
                if state["position"]==0:
                    final_profit=calculate_final_pnl(state["realized_pnl_tracker"],0,profit)
                    close_trade(trade,date,price,final_profit,"SELL L2")

            elif sell1:
                sell_amount=min(0.33,state["position"])
                state["realized_pnl_tracker"].append((sell_amount,profit))
                state["position"]=round(state["position"]-sell_amount,2)
                action="SELL L1"
                trade=trades[-1]
                trade["exits"].append({"date":date,"type":"SELL L1","price":round(price,4),"position_sold":round(sell_amount,2),"profit":round(profit,2)})
                signals.append({"symbol":symbol,"date":date,"action":"SELL L1","price":round(price,4),"rsi":round(rsi_val,2),"profit":round(profit,2),"position":state["position"]})
                if state["position"]==0:
                    final_profit=calculate_final_pnl(state["realized_pnl_tracker"],0,profit)
                    close_trade(trade,date,price,final_profit,"SELL L1")

        if state["position"]==0 and action in {"SELL L1","SELL L2","EXIT FULL","STOP LOSS","TRAILING STOP"}:
            state["avg_price"]=0.0
            state["peak_profit"]=0.0
            state["realized_pnl_tracker"]=[]
            state["cycle"]+=1

    # Open position at end of period is NOT counted as closed
    if state["position"]>0 and trades:
        trade=trades[-1]
        if trade["status"]=="OPEN":
            last_price=float(df.loc[df.index<=end,"Close"].iloc[-1])
            last_date=df.loc[df.index<=end].index[-1].strftime("%Y-%m-%d")
            final_profit=calculate_final_pnl(
                state["realized_pnl_tracker"],state["position"],
                ((last_price-state["avg_price"])/state["avg_price"])*100
            )
            close_trade(trade,last_date,last_price,final_profit,"END_OF_DATA")
            trade["status"]="OPEN"

    return trades,signals

# ============================================================
# PERIOD STATISTICS
# ============================================================
def calculate_statistics(closed_trades):
    profits=[float(t["profit_pct"]) for t in closed_trades]
    wins=[p for p in profits if p>0]
    losses=[p for p in profits if p<=0]
    total=len(profits)
    gross_profit=sum(p for p in profits if p>0)
    gross_loss=abs(sum(p for p in profits if p<0))
    pf=gross_profit/gross_loss if gross_loss>0 else 0

    portfolio=INITIAL_CAPITAL
    peak=portfolio
    max_dd=0.0

    for trade in sorted(closed_trades,key=lambda x:x["exit_date"]):
        portfolio*=1+(trade["profit_pct"]/100)*POSITION_SIZE
        peak=max(peak,portfolio)
        dd=((peak-portfolio)/peak)*100
        max_dd=max(max_dd,dd)

    days=[t["days_in_trade"] for t in closed_trades if "days_in_trade" in t]

    exits={}
    for t in closed_trades:
        exits[t["exit_reason"]]=exits.get(t["exit_reason"],0)+1

    return {
        "total_closed_trades":total,
        "winning_trades":len(wins),
        "losing_trades":len(losses),
        "win_rate_percent":round(len(wins)/total*100,2) if total else 0,
        "sum_trade_profit_percent":round(sum(profits),2),
        "average_trade_profit_percent":round(float(np.mean(profits)),2) if profits else 0,
        "average_win_percent":round(float(np.mean(wins)),2) if wins else 0,
        "average_loss_percent":round(float(np.mean(losses)),2) if losses else 0,
        "gross_profit":round(gross_profit,2),
        "gross_loss":round(gross_loss,2),
        "profit_factor":round(pf,2),
        "realistic_compound_return_percent":round((portfolio/INITIAL_CAPITAL-1)*100,2),
        "maximum_drawdown_percent":round(max_dd,2),
        "average_days":round(float(np.mean(days)),1) if days else 0,
        "trades_60_days_or_more":sum(d>=60 for d in days),
        "trades_90_days_or_more":sum(d>=90 for d in days),
        "trades_180_days_or_more":sum(d>=180 for d in days),
        "exit_analysis":exits
    }

# ============================================================
# RUN THE ONE PERIOD-STABILITY TEST
# ============================================================
all_trades=[]
all_signals=[]
period_results={}

print("\nRUNNING PERIOD STABILITY TEST...\n")

for start_date,end_date in PERIODS:
    label=f"{start_date[:7]} -> {end_date[:7] if end_date[:4]!="2099" else "NOW"}"
    period_trades=[]
    period_signals=[]

    print("="*80)
    print(f"PERIOD: {label}")
    print("="*80)

    for symbol in SYMBOLS:
        df=fetch_local_data(symbol)

        if df is None:
            continue
        if len(df)<MIN_BARS:
            continue

        trades,signals=backtest_stock(symbol,df,start_date,end_date)
        period_trades.extend(trades)
        period_signals.extend(signals)

    closed=[t for t in period_trades if t["status"]=="CLOSED" and t["profit_pct"] is not None and t["exit_reason"]!="END_OF_DATA"]
    open_trades=[t for t in period_trades if t["status"]=="OPEN"]

    stats=calculate_statistics(closed)
    stats["open_positions"]=len(open_trades)

    period_results[label]=stats
    all_trades.extend(period_trades)
    all_signals.extend(period_signals)

    print(f"Closed Trades       : {stats['total_closed_trades']}")
    print(f"Win Rate            : {stats['win_rate_percent']:.2f}%")
    print(f"Average Trade       : {stats['average_trade_profit_percent']:.2f}%")
    print(f"Average Win         : {stats['average_win_percent']:.2f}%")
    print(f"Average Loss        : {stats['average_loss_percent']:.2f}%")
    print(f"Profit Factor       : {stats['profit_factor']:.2f}")
    print(f"Compound Return     : {stats['realistic_compound_return_percent']:.2f}%")
    print(f"Maximum Drawdown    : {stats['maximum_drawdown_percent']:.2f}%")
    print(f"Average Days        : {stats['average_days']:.1f}")
    print(f">=60 Days           : {stats['trades_60_days_or_more']}")
    print(f">=90 Days           : {stats['trades_90_days_or_more']}")
    print(f">=180 Days          : {stats['trades_180_days_or_more']}")
    print(f"Open Positions      : {stats['open_positions']}")

# ============================================================
# OVERALL PERIOD TEST SUMMARY
# ============================================================
all_trades.sort(key=lambda x:x["first_entry"]["date"])
all_signals.sort(key=lambda x:x["date"])

closed_trades=[t for t in all_trades if t["status"]=="CLOSED" and t["profit_pct"] is not None and t["exit_reason"]!="END_OF_DATA"]
open_trades=[t for t in all_trades if t["status"]=="OPEN"]

profits=[float(t["profit_pct"]) for t in closed_trades]
wins=[p for p in profits if p>0]
losses=[p for p in profits if p<=0]

overall={
    "total_closed_trades":len(profits),
    "winning_trades":len(wins),
    "losing_trades":len(losses),
    "win_rate_percent":round(len(wins)/len(profits)*100,2) if profits else 0,
    "average_trade_profit_percent":round(float(np.mean(profits)),2) if profits else 0,
    "profit_factor":round(sum(wins)/abs(sum(losses)),2) if losses else 0,
    "open_positions":len(open_trades)
}

# ============================================================
# SAVE
# ============================================================
result={
    "strategy":"EGX Ladder Cycle System v3.4 Optimized",
    "backtest_version":"Period Stability Test v1.2",
    "purpose":"Test whether the unchanged strategy remains stable across different historical market periods.",
    "data_file":DB_FILE,
    "periods":[{"start":s,"end":e} for s,e in PERIODS],
    "parameters":{
        "rsi_period":RSI_PERIOD,
        "ema100_period":EMA100_PERIOD,
        "runup_lookback":RUNUP_LOOKBACK,
        "max_runup_percent":MAX_RUNUP_PERCENT,
        "max_gap_down_percent":MAX_GAP_DOWN_PERCENT,
        "buy1_rsi":BUY1_RSI,"buy2_rsi":BUY2_RSI,"buy3_rsi":BUY3_RSI,
        "sell1_rsi":SELL1_RSI,"sell1_min_profit":SELL1_MIN_PROFIT,
        "sell2_rsi":SELL2_RSI,"sell2_min_profit":SELL2_MIN_PROFIT,
        "sell3_rsi":SELL3_RSI,"sell3_min_profit":SELL3_MIN_PROFIT,
        "stop_l1":STOP_L1,"stop_l2":STOP_L2,"stop_l3":STOP_L3,
        "trailing_trigger":TRAILING_TRIGGER,
        "trailing_giveback":TRAILING_GIVEBACK
    },
    "period_statistics":period_results,
    "overall_period_test_summary":overall,
    "trades":all_trades
}

with open(RESULT_FILE,"w",encoding="utf-8") as f:
    json.dump(result,f,ensure_ascii=False,indent=2)

with open(TRADES_FILE,"w",encoding="utf-8") as f:
    json.dump(all_trades,f,ensure_ascii=False,indent=2)

# ============================================================
# FINAL DISPLAY
# ============================================================
print("\n")
print("="*80)
print("PERIOD STABILITY TEST - FINAL SUMMARY")
print("="*80)
print(f"{'PERIOD':22} {'TRADES':>7} {'WR':>8} {'PF':>8} {'DD':>8} {'AVG':>8}")
print("-"*80)

for period,stats in period_results.items():
    print(
        f"{period:22} "
        f"{stats['total_closed_trades']:7} "
        f"{stats['win_rate_percent']:7.2f}% "
        f"{stats['profit_factor']:7.2f} "
        f"{stats['maximum_drawdown_percent']:7.2f}% "
        f"{stats['average_trade_profit_percent']:7.2f}%"
    )

print("-"*80)
print(f"TOTAL CLOSED TRADES : {overall['total_closed_trades']}")
print(f"TOTAL WIN RATE       : {overall['win_rate_percent']:.2f}%")
print(f"TOTAL PROFIT FACTOR  : {overall['profit_factor']:.2f}")
print(f"OPEN POSITIONS       : {overall['open_positions']}")
print("="*80)
print("RESULT FILE:",RESULT_FILE)
print("TRADES FILE:",TRADES_FILE)
print("="*80)

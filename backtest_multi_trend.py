import json,os
import pandas as pd
import numpy as np

# ============================================================
# EGX LADDER BACKTEST v1.4
# TradingView Database + Next Day Open Execution
# Correct Cycle Accounting + Real Market Scenarios
# ============================================================
DB_FILE="egx_history_database_v2.json"
RESULTS_FILE="backtest_results.json"
TRADES_FILE="backtest_trades.json"
SUMMARY_FILE="backtest_summary_by_stock.json"

SYMBOLS=["OLFI","EMFD","ETEL","EAST","EFIH","ABUK","OIH","SWDY","ISPH","ATQA","MTIE","HRHO","ORWE","JUFO","DSCW","SUGR","ELSH","RMDA","RAYA","EEII","MPCO","GBCO","TMGH","ORHD","AMOC","FWRY","COMI","ADIB","PHDC","MCQE","SKPC","EGAL"]

# ============================================================
# TEST PERIODS
# ============================================================
TEST_PERIODS={
    "SIDEWAYS_1_2024_2025":("2024-10-01","2025-06-20"),
    "SIDEWAYS_2_2026":("2026-02-20","2026-06-20")
}

RSI_PERIOD=14
EMA_PERIOD=95
RUNUP_LOOKBACK=160
MAX_RUNUP_PERCENT=80
MAX_GAP_DOWN_PERCENT=-5

BUY1_RSI=60
BUY2_RSI=55
BUY3_RSI=48

SELL1_RSI=66
SELL1_MIN_PROFIT=15
SELL2_RSI=84
SELL2_MIN_PROFIT=25
SELL3_RSI=86
SELL3_MIN_PROFIT=25

STOP_L1=-18
STOP_L2=-17
STOP_L3=-13
TRAILING_TRIGGER=36
TRAILING_GIVEBACK=11

MIN_BARS=40

def load_database():
    with open(DB_FILE,"r",encoding="utf-8") as f:
        return json.load(f)

def fetch_local_data(raw_database,symbol):
    candidates=[symbol,symbol+".CA",symbol.replace(".CA","")]
    stock=None
    for key in candidates:
        if key in raw_database:
            stock=raw_database[key]
            break
    if stock is None or not isinstance(stock,dict):
        return None
    data=stock.get("data",stock)
    if not isinstance(data,dict) or len(data)<MIN_BARS:
        return None
    rows=[]
    for date,v in data.items():
        if not isinstance(v,dict):
            continue
        if not all(x in v for x in ["Open","High","Low","Close","Volume"]):
            continue
        try:
            rows.append([date,float(v["Open"]),float(v["High"]),float(v["Low"]),float(v["Close"]),float(v["Volume"])])
        except:
            continue
    if len(rows)<MIN_BARS:
        return None
    df=pd.DataFrame(rows,columns=["Date","Open","High","Low","Close","Volume"])
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    df=df.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date").set_index("Date")
    if len(df)<MIN_BARS:
        return None
    return df

def add_indicators(df):
    df=df.copy()
    df["EMA95"]=df["Close"].ewm(span=EMA_PERIOD,adjust=False).mean()
    delta=df["Close"].diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(com=RSI_PERIOD-1,adjust=False).mean()
    avg_loss=loss.ewm(com=RSI_PERIOD-1,adjust=False).mean()
    rs=avg_gain/avg_loss.replace(0,np.nan)
    df["RSI"]=100-(100/(1+rs))
    df["PrevClose"]=df["Close"].shift(1)
    df["GapDown"]=(df["Open"]-df["PrevClose"])/df["PrevClose"]*100
    return df

def safe_to_buy(df,i):
    start=max(0,i-RUNUP_LOOKBACK+1)
    window=df.iloc[start:i+1]
    if len(window)<20:
        return False
    low=window["Low"].min()
    high=window["High"].max()
    if low<=0:
        return False
    runup=(high-low)/low*100
    return runup<=MAX_RUNUP_PERCENT

def no_gap_down(df,i):
    start=max(1,i-2)
    gaps=df["GapDown"].iloc[start:i+1]
    return not (gaps<=MAX_GAP_DOWN_PERCENT).any()

def ema_up(df,i):
    if i<10:
        return False
    e=df["EMA95"]
    if pd.isna(e.iloc[i]) or pd.isna(e.iloc[i-5]) or pd.isna(e.iloc[i-10]):
        return False
    return e.iloc[i]>e.iloc[i-5] and e.iloc[i-5]>e.iloc[i-10] and e.iloc[i]>e.iloc[i-10]*1.002 and df["Close"].iloc[i]<=e.iloc[i]*1.07

def create_trade(symbol,date,price,cycle_id):
    return {
        "symbol":symbol,
        "cycle_id":cycle_id,
        "entry_date":str(date.date()),
        "entry_price":float(price),
        "position":0.33,
        "avg_price":float(price),
        "peak_profit":0.0,
        "second_entry":None,
        "third_entry":None,
        "last_totally_average_price":float(price),
        "status":"OPEN"
    }

def current_profit(trade,price):
    return (price/trade["avg_price"]-1)*100

def update_avg(trade,price,add_position):
    old=trade["position"]
    new=old+add_position
    trade["avg_price"]=(trade["avg_price"]*old+price*add_position)/new
    trade["position"]=round(new,2)
    trade["last_totally_average_price"]=trade["avg_price"]

def close_trade(trade,date,price,reason,closed_position=None):
    pos=trade["position"] if closed_position is None else closed_position
    profit=(price/trade["avg_price"]-1)*100
    result=trade.copy()
    result.update({
        "exit_date":str(date.date()),
        "exit_price":float(price),
        "exit_position":float(pos),
        "profit_percent":float(profit),
        "reason":reason,
        "status":"CLOSED"
    })
    return result

def partial_sell(trade,date,price,sell_position,reason):
    profit=(price/trade["avg_price"]-1)*100
    result=trade.copy()
    result.update({
        "exit_date":str(date.date()),
        "exit_price":float(price),
        "exit_position":float(sell_position),
        "profit_percent":float(profit),
        "reason":reason,
        "status":"PARTIAL"
    })
    trade["position"]=round(trade["position"]-sell_position,2)
    if trade["position"]<=0.001:
        trade["position"]=0
    return result

def run_symbol_backtest(symbol,df,start_date=None,end_date=None):
    trades=[]
    trade=None
    pending=None
    cycle_id=0

    start_ts=pd.Timestamp(start_date) if start_date else df.index[0]
    end_ts=pd.Timestamp(end_date) if end_date else df.index[-1]

    # نستخدم كل البيانات السابقة لحساب المؤشرات والـlookback،
    # لكن لا نسمح بأي صفقة قبل بداية فترة الاختبار.
    test_start_idx=df.index.searchsorted(start_ts)
    test_end_idx=df.index.searchsorted(end_ts,side="right")-1

    if test_start_idx>=len(df) or test_end_idx<test_start_idx:
        return []

    for i in range(test_start_idx,min(test_end_idx,len(df)-1)):
        row=df.iloc[i]
        next_row=df.iloc[i+1]
        date=df.index[i]
        next_date=df.index[i+1]

        # لا ننفذ صفقة خارج نهاية فترة الاختبار
        if next_date>end_ts:
            break

        next_open=float(next_row["Open"])

        # ====================================================
        # تنفيذ إشارة اليوم السابق عند افتتاح اليوم التالي
        # ====================================================
        if pending is not None:
            action=pending
            pending=None

            if action["type"]=="BUY":
                price=next_open

                if trade is None:
                    cycle_id+=1
                    trade=create_trade(symbol,next_date,price,cycle_id)

                elif action["level"]==2 and trade["position"]<=0.34:
                    update_avg(trade,price,0.33)
                    trade["second_entry"]=str(next_date.date())
                    trade["peak_profit"]=0.0

                elif action["level"]==3 and trade["position"]<1:
                    add=min(0.34,1-trade["position"])
                    update_avg(trade,price,add)
                    trade["third_entry"]=str(next_date.date())
                    trade["peak_profit"]=0.0

            elif action["type"]=="SELL" and trade is not None:
                price=next_open
                result=close_trade(trade,next_date,price,action["reason"])
                trades.append(result)
                trade=None

            elif action["type"]=="PARTIAL" and trade is not None:
                price=next_open
                sell_pos=min(action["position"],trade["position"])
                result=partial_sell(trade,next_date,price,sell_pos,action["reason"])
                trades.append(result)

                if trade["position"]<=0:
                    trade=None

        # ====================================================
        # تحديث أعلى ربح حسب إغلاق اليوم
        # ====================================================
        if trade is not None:
            profit=current_profit(trade,float(row["Close"]))
            if profit>trade["peak_profit"]:
                trade["peak_profit"]=profit

        close=float(row["Close"])
        rsi=float(row["RSI"]) if not pd.isna(row["RSI"]) else np.nan

        # ====================================================
        # L1 ENTRY
        # ====================================================
        if trade is None:
            if not pd.isna(rsi) and ema_up(df,i) and safe_to_buy(df,i) and no_gap_down(df,i) and rsi<=BUY1_RSI:
                pending={"type":"BUY","level":1}

        else:
            profit=current_profit(trade,close)
            pos=trade["position"]

            # =================================================
            # Stops
            # =================================================
            stop=False

            if pos<=0.33 and profit<=STOP_L1:
                stop=True
            elif pos<=0.66 and profit<=STOP_L2:
                stop=True
            elif pos>0.66 and profit<=STOP_L3:
                stop=True

            if stop:
                pending={"type":"SELL","reason":"STOP LOSS"}
                continue

            # =================================================
            # Trailing Stop
            # =================================================
            if trade["peak_profit"]>TRAILING_TRIGGER and trade["peak_profit"]-profit>=TRAILING_GIVEBACK:
                pending={"type":"SELL","reason":"TRAILING STOP"}
                continue

            # =================================================
            # L3
            # =================================================
            if pos>=0.66 and pos<1 and not pd.isna(rsi) and rsi<=BUY3_RSI and close<trade["avg_price"]*0.94 and ema_up(df,i) and no_gap_down(df,i):
                pending={"type":"BUY","level":3}
                continue

            # =================================================
            # L2
            # =================================================
            if pos<=0.34 and not pd.isna(rsi) and rsi<=BUY2_RSI and close<trade["avg_price"]*0.97 and ema_up(df,i) and no_gap_down(df,i):
                pending={"type":"BUY","level":2}
                continue

            # =================================================
            # Full Exit
            # =================================================
            if not pd.isna(rsi) and rsi>=SELL3_RSI and profit>SELL3_MIN_PROFIT:
                pending={"type":"SELL","reason":"SELL L3 / FULL EXIT"}
                continue

            # =================================================
            # Partial L2
            # =================================================
            if 0.30<pos<=0.70 and not pd.isna(rsi) and rsi>=SELL2_RSI and profit>SELL2_MIN_PROFIT:
                pending={"type":"PARTIAL","position":0.33,"reason":"SELL L2"}
                continue

            # =================================================
            # Partial L1
            # =================================================
            if pos>0.70 and not pd.isna(rsi) and rsi>=SELL1_RSI and profit>SELL1_MIN_PROFIT:
                pending={"type":"PARTIAL","position":0.33,"reason":"SELL L1"}
                continue

    # ========================================================
    # نهاية فترة الاختبار
    # ========================================================
    if trade is not None:
        last_idx=min(test_end_idx,len(df)-1)
        last_date=df.index[last_idx]
        last_close=float(df["Close"].iloc[last_idx])
        profit=current_profit(trade,last_close)

        result=trade.copy()
        result.update({
            "exit_date":str(last_date.date()),
            "exit_price":last_close,
            "exit_position":trade["position"],
            "profit_percent":float(profit),
            "reason":"END OF TEST PERIOD" if end_date else "END OF DATA",
            "status":"OPEN_MARKED"
        })
        trades.append(result)

    return trades

def build_cycles(trades):
    cycles={}

    for t in trades:
        key=(t["symbol"],t.get("cycle_id"),t["entry_date"])

        if key not in cycles:
            cycles[key]={
                "symbol":t["symbol"],
                "cycle_id":t.get("cycle_id"),
                "entry_date":t["entry_date"],
                "events":[],
                "open":False
            }

        cycles[key]["events"].append(t)

    output=[]

    for cycle in cycles.values():
        events=cycle["events"]
        closed=any(e["status"]=="CLOSED" for e in events)
        open_marked=any(e["status"]=="OPEN_MARKED" for e in events)

        cycle_return=0.0

        for e in events:
            position_weight=float(e.get("exit_position",0))
            profit=float(e.get("profit_percent",0))
            cycle_return+=profit*position_weight

        last_event=events[-1]

        output.append({
            "symbol":cycle["symbol"],
            "cycle_id":cycle["cycle_id"],
            "entry_date":cycle["entry_date"],
            "last_exit_date":last_event.get("exit_date"),
            "cycle_return":float(cycle_return),
            "status":"CLOSED" if closed and not open_marked else "OPEN",
            "events":len(events)
        })

    return output

def calculate_stats(trades):
    cycles=build_cycles(trades)

    closed_cycles=[c for c in cycles if c["status"]=="CLOSED"]
    open_cycles=[c for c in cycles if c["status"]=="OPEN"]

    profits=[float(c["cycle_return"]) for c in closed_cycles]

    wins=[x for x in profits if x>0]
    losses=[x for x in profits if x<=0]

    gross_win=sum(wins)
    gross_loss=abs(sum(losses))
    pf=gross_win/gross_loss if gross_loss>0 else None

    open_returns=[float(c["cycle_return"]) for c in open_cycles]

    return {
        "cycles":len(closed_cycles),
        "wins":len(wins),
        "losses":len(losses),
        "win_rate":round(len(wins)/len(closed_cycles)*100,2) if closed_cycles else 0,
        "sum_cycle_returns":round(sum(profits),2),
        "avg_cycle_return":round(np.mean(profits),2) if profits else 0,
        "avg_win":round(np.mean(wins),2) if wins else 0,
        "avg_loss":round(np.mean(losses),2) if losses else 0,
        "profit_factor":round(pf,2) if pf is not None else None,
        "best_cycle":round(max(profits),2) if profits else 0,
        "worst_cycle":round(min(profits),2) if profits else 0,
        "open_cycles":len(open_cycles),
        "open_cycle_returns":round(sum(open_returns),2) if open_returns else 0
    }

def calculate_scenario(symbols_data,start_date,end_date):
    all_trades=[]
    summary={}

    for symbol,df in symbols_data.items():
        trades=run_symbol_backtest(symbol,df,start_date,end_date)
        stats=calculate_stats(trades)
        summary[symbol]=stats
        all_trades.extend(trades)

    return all_trades,summary,calculate_stats(all_trades)

def main():
    print("🚀 EGX LADDER BACKTEST v1.4")
    print("📊 TradingView Database | Next-Day Open Execution")
    print("📊 Correct Cycle Accounting | Real Market Scenarios")
    print("="*70)

    raw_database=load_database()
    print(f"📁 Symbols in database: {len(raw_database)}")

    symbols_data={}
    invalid=[]

    for symbol in SYMBOLS:
        df=fetch_local_data(raw_database,symbol)

        if df is None:
            invalid.append(symbol)
            print(f"❌ {symbol}: INVALID / insufficient data")
            continue

        df=add_indicators(df)
        symbols_data[symbol]=df

    valid=len(symbols_data)

    # ========================================================
    # FULL DATABASE BACKTEST
    # ========================================================
    print("\n" + "="*70)
    print("📊 FULL DATABASE BACKTEST")
    print("="*70)

    all_trades_full=[]
    summary_full={}

    for symbol,df in symbols_data.items():
        trades=run_symbol_backtest(symbol,df)
        stats=calculate_stats(trades)

        all_trades_full.extend(trades)
        summary_full[symbol]=stats

        print(
            f"✅ {symbol}: {len(df)} bars | "
            f"Cycles: {stats['cycles']} | "
            f"Win: {stats['win_rate']}% | "
            f"Return: {stats['sum_cycle_returns']}%"
        )

    overall_full=calculate_stats(all_trades_full)

    print("\n📊 FULL PERIOD RESULTS")
    print(f"Valid symbols      : {valid}/{len(SYMBOLS)}")
    print(f"Cycles             : {overall_full['cycles']}")
    print(f"Wins               : {overall_full['wins']}")
    print(f"Losses             : {overall_full['losses']}")
    print(f"Win rate           : {overall_full['win_rate']}%")
    print(f"Sum cycle returns  : {overall_full['sum_cycle_returns']}%")
    print(f"Avg cycle return   : {overall_full['avg_cycle_return']}%")
    print(f"Avg win            : {overall_full['avg_win']}%")
    print(f"Avg loss           : {overall_full['avg_loss']}%")
    print(f"Profit factor      : {overall_full['profit_factor']}")
    print(f"Best cycle         : {overall_full['best_cycle']}%")
    print(f"Worst cycle        : {overall_full['worst_cycle']}%")
    print(f"Open cycles        : {overall_full['open_cycles']}")
    print(f"Open cycle returns : {overall_full['open_cycle_returns']}%")

    # ========================================================
    # REAL MARKET SCENARIO TESTS
    # ========================================================
    scenario_results={}

    for name,(start_date,end_date) in TEST_PERIODS.items():
        print("\n" + "="*70)
        print(f"🟡 SCENARIO: {name}")
        print(f"📅 {start_date} → {end_date}")
        print("="*70)

        trades,summary,stats=calculate_scenario(
            symbols_data,
            start_date,
            end_date
        )

        scenario_results[name]={
            "period":{"start":start_date,"end":end_date},
            "overall":stats,
            "by_stock":summary,
            "trades":trades
        }

        print(f"Cycles             : {stats['cycles']}")
        print(f"Wins               : {stats['wins']}")
        print(f"Losses             : {stats['losses']}")
        print(f"Win rate           : {stats['win_rate']}%")
        print(f"Sum cycle returns  : {stats['sum_cycle_returns']}%")
        print(f"Avg cycle return   : {stats['avg_cycle_return']}%")
        print(f"Avg win            : {stats['avg_win']}%")
        print(f"Avg loss           : {stats['avg_loss']}%")
        print(f"Profit factor      : {stats['profit_factor']}")
        print(f"Best cycle         : {stats['best_cycle']}%")
        print(f"Worst cycle        : {stats['worst_cycle']}%")
        print(f"Open cycles        : {stats['open_cycles']}")
        print(f"Open cycle returns : {stats['open_cycle_returns']}%")

    # ========================================================
    # SAVE RESULTS
    # ========================================================
    results={
        "parameters":{
            "EMA_PERIOD":EMA_PERIOD,
            "RSI_PERIOD":RSI_PERIOD,
            "BUY1_RSI":BUY1_RSI,
            "BUY2_RSI":BUY2_RSI,
            "BUY3_RSI":BUY3_RSI,
            "SELL1_RSI":SELL1_RSI,
            "SELL2_RSI":SELL2_RSI,
            "SELL3_RSI":SELL3_RSI,
            "STOP_L1":STOP_L1,
            "STOP_L2":STOP_L2,
            "STOP_L3":STOP_L3,
            "TRAILING_TRIGGER":TRAILING_TRIGGER,
            "TRAILING_GIVEBACK":TRAILING_GIVEBACK,
            "RUNUP_LOOKBACK":RUNUP_LOOKBACK,
            "MAX_RUNUP_PERCENT":MAX_RUNUP_PERCENT,
            "MAX_GAP_DOWN_PERCENT":MAX_GAP_DOWN_PERCENT,
            "EXECUTION":"NEXT DAY OPEN",
            "ACCOUNTING":"POSITION-WEIGHTED CYCLE RETURN",
            "SCENARIO_WARMUP":"FULL DATABASE BEFORE TEST START"
        },
        "full_period":{
            "overall":overall_full,
            "valid_symbols":valid,
            "invalid_symbols":invalid,
            "by_stock":summary_full
        },
        "scenarios":scenario_results
    }

    with open(RESULTS_FILE,"w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)

    with open(TRADES_FILE,"w",encoding="utf-8") as f:
        json.dump(all_trades_full,f,ensure_ascii=False,indent=2)

    with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
        json.dump(summary_full,f,ensure_ascii=False,indent=2)

    print("\n" + "="*70)
    print("✅ BACKTEST COMPLETE")
    print("="*70)
    print(f"Valid symbols : {valid}/{len(SYMBOLS)}")
    print(f"Full cycles   : {overall_full['cycles']}")
    print(f"Full win rate : {overall_full['win_rate']}%")
    print(f"Full return   : {overall_full['sum_cycle_returns']}%")

    print("\n🟡 SCENARIO SUMMARY")
    for name,data in scenario_results.items():
        s=data["overall"]
        print(
            f"{name}: "
            f"Cycles={s['cycles']} | "
            f"Win={s['win_rate']}% | "
            f"Return={s['sum_cycle_returns']}% | "
            f"PF={s['profit_factor']}"
        )

    if invalid:
        print("\n⚠️ Invalid symbols:",", ".join(invalid))

if __name__=="__main__":
    main()

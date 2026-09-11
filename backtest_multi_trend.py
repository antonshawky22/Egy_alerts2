import json,os
import pandas as pd
import numpy as np

# ============================================================
# EGX LADDER BACKTEST v1.3
# TradingView Database + Next Day Open Execution
# ============================================================
DB_FILE="egx_history_database_v2.json"
RESULTS_FILE="backtest_results.json"
TRADES_FILE="backtest_trades.json"
SUMMARY_FILE="backtest_summary_by_stock.json"

SYMBOLS=["OLFI","EMFD","ETEL","EAST","EFIH","ABUK","OIH","SWDY","ISPH","ATQA","MTIE","HRHO","ORWE","JUFO","DSCW","SUGR","ELSH","RMDA","RAYA","EEII","MPCO","GBCO","TMGH","ORHD","AMOC","FWRY","COMI","ADIB","PHDC","MCQE","SKPC","EGAL"]

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

def create_trade(symbol,date,price):
    return {"symbol":symbol,"entry_date":str(date.date()),"entry_price":float(price),"position":0.33,"avg_price":float(price),"peak_profit":0.0,"second_entry":None,"third_entry":None,"last_totally_average_price":float(price),"status":"OPEN"}

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
    result.update({"exit_date":str(date.date()),"exit_price":float(price),"exit_position":float(pos),"profit_percent":float(profit),"reason":reason,"status":"CLOSED"})
    return result

def partial_sell(trade,date,price,sell_position,reason):
    sold=trade["position"]
    profit=(price/trade["avg_price"]-1)*100
    result=trade.copy()
    result.update({"exit_date":str(date.date()),"exit_price":float(price),"exit_position":float(sell_position),"profit_percent":float(profit),"reason":reason,"status":"PARTIAL"})
    trade["position"]=round(trade["position"]-sell_position,2)
    if trade["position"]<=0.001:
        trade["position"]=0
    return result

def run_symbol_backtest(symbol,df):
    trades=[]
    trade=None
    pending=None
    for i in range(len(df)-1):
        row=df.iloc[i]
        next_row=df.iloc[i+1]
        date=df.index[i]
        next_date=df.index[i+1]
        next_open=float(next_row["Open"])
        # تنفيذ أي إشارة من اليوم السابق عند افتتاح اليوم التالي
        if pending is not None:
            action=pending
            pending=None
            if action["type"]=="BUY":
                price=next_open
                if trade is None:
                    trade=create_trade(symbol,next_date,price)
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
        # تحديث أعلى ربح حسب إغلاق اليوم
        if trade is not None:
            profit=current_profit(trade,float(row["Close"]))
            if profit>trade["peak_profit"]:
                trade["peak_profit"]=profit
        # توليد الإشارة من إغلاق اليوم
        close=float(row["Close"])
        rsi=float(row["RSI"]) if not pd.isna(row["RSI"]) else np.nan
        if trade is None:
            if not pd.isna(rsi) and ema_up(df,i) and safe_to_buy(df,i) and no_gap_down(df,i) and rsi<=BUY1_RSI:
                pending={"type":"BUY","level":1}
        else:
            profit=current_profit(trade,close)
            pos=trade["position"]
            # Stops
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
            # Trailing stop
            if trade["peak_profit"]>TRAILING_TRIGGER and trade["peak_profit"]-profit>=TRAILING_GIVEBACK:
                pending={"type":"SELL","reason":"TRAILING STOP"}
                continue
            # L3
            if pos>=0.66 and pos<1 and not pd.isna(rsi) and rsi<=BUY3_RSI and close<trade["avg_price"]*0.94 and ema_up(df,i) and no_gap_down(df,i):
                pending={"type":"BUY","level":3}
                continue
            # L2
            if pos<=0.34 and not pd.isna(rsi) and rsi<=BUY2_RSI and close<trade["avg_price"]*0.97 and ema_up(df,i) and no_gap_down(df,i):
                pending={"type":"BUY","level":2}
                continue
            # Full exit
            if not pd.isna(rsi) and rsi>=SELL3_RSI and profit>SELL3_MIN_PROFIT:
                pending={"type":"SELL","reason":"SELL L3 / FULL EXIT"}
                continue
            # Partial L2
            if 0.30<pos<=0.70 and not pd.isna(rsi) and rsi>=SELL2_RSI and profit>SELL2_MIN_PROFIT:
                pending={"type":"PARTIAL","position":0.33,"reason":"SELL L2"}
                continue
            # Partial L1
            if pos>0.70 and not pd.isna(rsi) and rsi>=SELL1_RSI and profit>SELL1_MIN_PROFIT:
                pending={"type":"PARTIAL","position":0.33,"reason":"SELL L1"}
                continue
    # لا ننفذ إشارة آخر يوم لأنه لا يوجد يوم تالٍ
    if trade is not None:
        last_date=df.index[-1]
        last_close=float(df["Close"].iloc[-1])
        profit=current_profit(trade,last_close)
        result=trade.copy()
        result.update({"exit_date":str(last_date.date()),"exit_price":last_close,"exit_position":trade["position"],"profit_percent":float(profit),"reason":"END OF DATA","status":"OPEN_MARKED"})
        trades.append(result)
    return trades

def calculate_stats(trades):
    closed=[t for t in trades if t["reason"]!="END OF DATA"]
    profits=[float(t["profit_percent"]) for t in closed]
    wins=[x for x in profits if x>0]
    losses=[x for x in profits if x<=0]
    gross_win=sum(wins)
    gross_loss=abs(sum(losses))
    pf=gross_win/gross_loss if gross_loss>0 else None
    return {"trades":len(closed),"wins":len(wins),"losses":len(losses),"win_rate":round(len(wins)/len(closed)*100,2) if closed else 0,"total_profit":round(sum(profits),2),"avg_profit":round(np.mean(profits),2) if profits else 0,"avg_win":round(np.mean(wins),2) if wins else 0,"avg_loss":round(np.mean(losses),2) if losses else 0,"profit_factor":round(pf,2) if pf is not None else None,"best_trade":round(max(profits),2) if profits else 0,"worst_trade":round(min(profits),2) if profits else 0}

def main():
    print("🚀 EGX LADDER BACKTEST v1.3")
    print("📊 TradingView Database | Next-Day Open Execution")
    print("="*65)
    raw_database=load_database()
    print(f"📁 Symbols in database: {len(raw_database)}")
    all_trades=[]
    summary={}
    valid=0
    invalid=[]
    for symbol in SYMBOLS:
        df=fetch_local_data(raw_database,symbol)
        if df is None:
            invalid.append(symbol)
            print(f"❌ {symbol}: INVALID / insufficient data")
            continue
        df=add_indicators(df)
        trades=run_symbol_backtest(symbol,df)
        stats=calculate_stats(trades)
        all_trades.extend(trades)
        summary[symbol]=stats
        valid+=1
        print(f"✅ {symbol}: {len(df)} bars | Trades: {stats['trades']} | Win: {stats['win_rate']}% | Profit: {stats['total_profit']}%")
    overall=calculate_stats(all_trades)
    results={"parameters":{"EMA_PERIOD":EMA_PERIOD,"RSI_PERIOD":RSI_PERIOD,"BUY1_RSI":BUY1_RSI,"BUY2_RSI":BUY2_RSI,"BUY3_RSI":BUY3_RSI,"SELL1_RSI":SELL1_RSI,"SELL2_RSI":SELL2_RSI,"SELL3_RSI":SELL3_RSI,"STOP_L1":STOP_L1,"STOP_L2":STOP_L2,"STOP_L3":STOP_L3,"TRAILING_TRIGGER":TRAILING_TRIGGER,"TRAILING_GIVEBACK":TRAILING_GIVEBACK,"RUNUP_LOOKBACK":RUNUP_LOOKBACK,"MAX_RUNUP_PERCENT":MAX_RUNUP_PERCENT,"MAX_GAP_DOWN_PERCENT":MAX_GAP_DOWN_PERCENT,"EXECUTION":"NEXT DAY OPEN"},"overall":overall,"valid_symbols":valid,"invalid_symbols":invalid}
    with open(RESULTS_FILE,"w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    with open(TRADES_FILE,"w",encoding="utf-8") as f:
        json.dump(all_trades,f,ensure_ascii=False,indent=2)
    with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    print("="*65)
    print("📊 OVERALL RESULTS")
    print(f"Valid symbols : {valid}/{len(SYMBOLS)}")
    print(f"Trades        : {overall['trades']}")
    print(f"Wins          : {overall['wins']}")
    print(f"Losses        : {overall['losses']}")
    print(f"Win rate      : {overall['win_rate']}%")
    print(f"Total profit  : {overall['total_profit']}%")
    print(f"Avg profit    : {overall['avg_profit']}%")
    print(f"Avg win       : {overall['avg_win']}%")
    print(f"Avg loss      : {overall['avg_loss']}%")
    print(f"Profit factor : {overall['profit_factor']}")
    print(f"Best trade    : {overall['best_trade']}%")
    print(f"Worst trade   : {overall['worst_trade']}%")
    print("="*65)
    if invalid:
        print("⚠️ Invalid symbols:",", ".join(invalid))

if __name__=="__main__":
    main()

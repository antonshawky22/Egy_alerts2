print("="*72);print("EGX WEEKLY PURE TRENDLINE STRATEGY v3.0");print("="*72)
import json,os,numpy as np,pandas as pd

DB_FILE="egx_weekly_database_v1.json"; RESULT_FILE="backtest_results.json"; TRADES_FILE="backtest_trades.json"
INITIAL_CAPITAL=100000.;MAX_POSITIONS=8;POSITION_SIZE=1/MAX_POSITIONS
PIVOT_LEFT=3;PIVOT_RIGHT=3;MAX_LINE_BARS=60
MIN_SLOPE_PERCENT=.10;MIN_PIVOT_GAP=5
MAX_TOUCH_DISTANCE=2.;MIN_REBOUND=.30;MAX_EXTENSION=5.
MIN_TOUCHES=2;MAX_VIOLATIONS=1;TOUCH_TOLERANCE=1.5
STOP_BUFFER=1.;MIN_STOP=3.;MAX_STOP=8.
BREAK_BUFFER=1.;BREAK_BARS=1
TRAIL_START=10.;TRAIL_DISTANCE=5.;MIN_BARS=120
MIN_BODY_PERCENT=.25;MIN_CLOSE_POSITION=.60

if not os.path.exists(DB_FILE): raise FileNotFoundError(DB_FILE)
with open(DB_FILE,encoding="utf-8") as f: db=json.load(f)
print(f"Database: {len(db)} symbols")

def to_df(x):
    if isinstance(x,dict) and "data" in x and "columns" in x:
        rows=[]
        for d,v in x["data"].items():
            r=v.copy() if isinstance(v,dict) else dict(zip(x["columns"],v))
            r["Date"]=d;rows.append(r)
        x=rows
    if not isinstance(x,list): return None
    df=pd.DataFrame(x)
    if df.empty:return None
    df.columns=[str(c).strip().capitalize() for c in df.columns]
    req=["Date","Open","High","Low","Close"]
    if not all(c in df for c in req):return None
    df["Date"]=pd.to_datetime(df.Date,errors="coerce")
    for c in req[1:]:df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna(subset=req).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

def pivots(df):
    lo,hi=df.Low.values,df.High.values;n=len(df);pl=[];ph=[]
    for i in range(PIVOT_LEFT,n-PIVOT_RIGHT):
        if lo[i]<lo[i-PIVOT_LEFT:i].min() and lo[i]<=lo[i+1:i+PIVOT_RIGHT+1].min():
            pl.append((i,i+PIVOT_RIGHT,float(lo[i])))
        if hi[i]>hi[i-PIVOT_LEFT:i].max() and hi[i]>=hi[i+1:i+PIVOT_RIGHT+1].max():
            ph.append((i,i+PIVOT_RIGHT,float(hi[i])))
    return pl,ph

def line(p1,p2):
    x1,y1=p1[0],p1[2];x2,y2=p2[0],p2[2]
    if x2<=x1 or x2-x1<MIN_PIVOT_GAP:return None
    s=(y2-y1)/(x2-x1)
    if s<=0 or s/y1*100<MIN_SLOPE_PERCENT:return None
    return {"x1":x1,"y1":y1,"x2":x2,"y2":y2,"s":s,"b":y1-s*x1}

def lv(L,i):return L["s"]*i+L["b"]

def best_line(df,i):
    ps,_=pivots(df.iloc[:i+1])
    ps=[p for p in ps if p[1]<=i]
    if len(ps)<2:return None
    cand=[]
    for a in range(max(0,len(ps)-10),len(ps)-1):
        for b in range(a+1,len(ps)):
            p1,p2=ps[a],ps[b]
            if p2[0]-p1[0]>MAX_LINE_BARS:continue
            L=line(p1,p2)
            if not L:continue
            touches=viol=0;last_touch=p2[0]
            for p in ps:
                if p[0]<=p1[0]:continue
                d=(p[2]-lv(L,p[0]))/lv(L,p[0])*100
                if abs(d)<=TOUCH_TOLERANCE:touches+=1;last_touch=p[0]
                elif d< -TOUCH_TOLERANCE:viol+=1
            if touches<MIN_TOUCHES or viol>MAX_VIOLATIONS:continue
            score=touches*100+p2[0]*2-(i-last_touch)*.5-(p2[0]-p1[0])*.2-viol*100
            cand.append((score,L,p1,p2,touches,viol))
    return max(cand,key=lambda x:x[0]) if cand else None

def close_trade(pos,date,price,reason):
    p=(price-pos["avg_price"])/pos["avg_price"]*100
    w=pos["weight"]
    pos["sales"].append({"date":date,"price":round(price,4),"weight":w,
                         "profit_pct":round(p,2),"capital_return":round(p*w,4),"reason":reason})
    pos.update(status="CLOSED",exit_date=date,exit_price=round(price,4),
               exit_reason=reason,weight=0.,profit_pct=round(sum(x["capital_return"] for x in pos["sales"]),2))
    return pos

def backtest(sym,df):
    pl,_=pivots(df);pos=None;trades=[];breaks=0
    for i in range(MIN_BARS,len(df)):
        r=df.iloc[i];date=r.Date.strftime("%Y-%m-%d")
        close,high,low,op=map(float,[r.Close,r.High,r.Low,r.Open])
        info=best_line(df,i);L=info[1] if info else None
        support=lv(L,i) if L else None

        if pos is None:
            if not L or support<=0:continue
            dist=(close-support)/support*100
            lowdist=(low-support)/support*100
            rng=high-low
            body=abs(close-op)
            bullish=close>op
            rejection=(bullish and rng>0 and body/rng>=MIN_BODY_PERCENT and
                       (close-low)/rng>=MIN_CLOSE_POSITION)
            entry=(0<=dist<=MAX_EXTENSION and lowdist<=MAX_TOUCH_DISTANCE and
                   close>=support*(1+MIN_REBOUND/100) and rejection)
            if entry:
                stop=support*(1-STOP_BUFFER/100)
                risk=(close-stop)/close*100
                if risk<MIN_STOP:stop=close*(1-MIN_STOP/100)
                if risk>MAX_STOP:stop=close*(1-MAX_STOP/100)
                pos={"symbol":sym,"status":"OPEN","entry_date":date,
                     "entry_price":close,"avg_price":close,"weight":1.,
                     "highest_price":close,"entry_support":round(support,4),
                     "trendline_slope_percent":round(L["s"]/L["y1"]*100,4),
                     "pivot1_date":df.iloc[info[2][0]].Date.strftime("%Y-%m-%d"),
                     "pivot2_date":df.iloc[info[3][0]].Date.strftime("%Y-%m-%d"),
                     "touches":info[4],"violations":info[5],"stop_price":round(stop,4),
                     "trail_active":False,"sales":[]}
            continue

        avg=pos["avg_price"];prev_high=pos["highest_price"]
        stop=max(float(pos["stop_price"]),avg*(1-MAX_STOP/100))
        if (prev_high-avg)/avg*100>=TRAIL_START:
            pos["trail_active"]=True
            stop=max(stop,prev_high*(1-TRAIL_DISTANCE/100))
        if low<=stop:
            trades.append(close_trade(pos,date,stop,"TRAIL_STOP" if pos["trail_active"] else "STOP_LOSS"))
            pos=None;breaks=0;continue

        if L:
            if close<support*(1-BREAK_BUFFER/100):breaks+=1
            else:breaks=0
        else:breaks+=1
        if breaks>=BREAK_BARS:
            trades.append(close_trade(pos,date,close,"TRENDLINE_BREAK"))
            pos=None;breaks=0;continue

        pos["highest_price"]=max(prev_high,high)

    if pos:
        last=float(df.iloc[-1].Close)
        pos["last_price"]=last;pos["unrealized_pct"]=round((last-pos["avg_price"])/pos["avg_price"]*100,2)
        trades.append(pos)
    return trades

all_trades=[]
for sym,data in db.items():
    if sym.upper() in {"EGX30","EGX70","EGX100"}:continue
    df=to_df(data)
    if df is None or len(df)<MIN_BARS:
        print(f"⚠️ {sym}: insufficient/invalid data");continue
    t=backtest(sym,df);all_trades+=t
    print(f"{sym:8} | {sum(x['status']=='CLOSED' for x in t):3} closed")

all_trades.sort(key=lambda x:x["entry_date"])
closed=[x for x in all_trades if x["status"]=="CLOSED"]
opened=[x for x in all_trades if x["status"]=="OPEN"]
profits=[float(x["profit_pct"]) for x in closed]
wins=[x for x in profits if x>0];losses=[x for x in profits if x<=0]
n=len(profits);winrate=len(wins)/n*100 if n else 0
sumprofit=sum(profits);avgwin=np.mean(wins) if wins else 0;avgloss=np.mean(losses) if losses else 0

portfolio=INITIAL_CAPITAL;equity=[portfolio];peq=[]
for t in closed:
    ret=t["profit_pct"]/100*POSITION_SIZE
    portfolio*=1+ret
    equity.append(portfolio)
    peq.append({"date":t["exit_date"],"symbol":t["symbol"],
                "trade_return_percent":t["profit_pct"],
                "portfolio_return_percent":round(ret*100,4),
                "portfolio_value":round(portfolio,2)})
compound=(portfolio/INITIAL_CAPITAL-1)*100
peak=INITIAL_CAPITAL;dd=0
for v in equity:
    peak=max(peak,v);dd=max(dd,(peak-v)/peak*100)

exit_analysis={}
for t in closed:
    r=t.get("exit_reason","UNKNOWN");exit_analysis[r]=exit_analysis.get(r,0)+1
best=max(closed,key=lambda x:x["profit_pct"]) if closed else None
worst=min(closed,key=lambda x:x["profit_pct"]) if closed else None

result={
"strategy":"Weekly Pure Diagonal Trendline Strategy v3.0",
"description":"Confirmed ascending support trendline + quality scoring + pullback touch + bullish rejection + structural stop + trendline break",
"parameters":{
"pivot_left":PIVOT_LEFT,"pivot_right":PIVOT_RIGHT,"max_line_bars":MAX_LINE_BARS,
"min_slope_percent":MIN_SLOPE_PERCENT,"min_pivot_gap":MIN_PIVOT_GAP,
"max_touch_distance_percent":MAX_TOUCH_DISTANCE,"min_rebound_percent":MIN_REBOUND,
"max_extension_percent":MAX_EXTENSION,"min_touches":MIN_TOUCHES,
"max_violations":MAX_VIOLATIONS,"touch_tolerance_percent":TOUCH_TOLERANCE,
"stop_buffer_percent":STOP_BUFFER,"min_stop_percent":MIN_STOP,"max_stop_percent":MAX_STOP,
"break_buffer_percent":BREAK_BUFFER,"break_confirmation_bars":BREAK_BARS,
"trail_start_percent":TRAIL_START,"trail_distance_percent":TRAIL_DISTANCE,
"max_positions":MAX_POSITIONS,"position_size_percent":POSITION_SIZE*100},
"statistics":{"total_trades":n,"winning_trades":len(wins),"losing_trades":len(losses),
"win_rate_percent":round(winrate,2),"sum_trade_profit_percent":round(sumprofit,2),
"realistic_compound_return_percent":round(compound,2),"average_win_percent":round(avgwin,2),
"average_loss_percent":round(avgloss,2),"maximum_drawdown_percent":round(dd,2),
"open_positions":len(opened)},"exit_analysis":exit_analysis,
"best_trade":best,"worst_trade":worst,"open_positions":opened,
"portfolio_equity":peq,"trades":all_trades}

for fn,obj in [(RESULT_FILE,result),(TRADES_FILE,all_trades)]:
    with open(fn,"w",encoding="utf-8") as f:json.dump(obj,f,ensure_ascii=False,indent=2)

print("\n"+"="*72);print("FINAL RESULTS");print("="*72)
print(f"Trades              : {n}")
print(f"Winners             : {len(wins)}")
print(f"Losers              : {len(losses)}")
print(f"Win Rate            : {winrate:.2f}%")
print(f"Sum Profit          : {sumprofit:.2f}%")
print(f"Compound Return     : {compound:.2f}%")
print(f"Average Win         : {avgwin:.2f}%")
print(f"Average Loss        : {avgloss:.2f}%")
print(f"Maximum Drawdown    : {dd:.2f}%")
print(f"Open Positions      : {len(opened)}")
print("\nEXIT ANALYSIS")
for k,v in exit_analysis.items():print(f"{k:22}: {v}")
if best:print(f"\nBEST  : {best['symbol']} | {best['profit_pct']:.2f}% | {best['entry_date']} -> {best['exit_date']}")
if worst:print(f"WORST : {worst['symbol']} | {worst['profit_pct']:.2f}% | {worst['entry_date']} -> {worst['exit_date']}")
print(f"\nSaved: {RESULT_FILE}, {TRADES_FILE}")
print("="*72)

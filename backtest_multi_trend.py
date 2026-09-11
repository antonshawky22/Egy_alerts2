import json,pandas as pd,numpy as np

DB_FILE="egx_history_database_v2.json"

BUY1_RSI,BUY2_RSI,BUY3_RSI=60,55,48
SELL1_RSI,SELL1_MIN_PROFIT=66,15
SELL2_MIN_POSITION,SELL2_MAX_POSITION,SELL2_RSI,SELL2_MIN_PROFIT=.30,.70,84,25
SELL3_RSI,SELL3_MIN_PROFIT=86,25
STOP_L1,STOP_L2,STOP_L3=-18,-17,-13
TRAILING_TRIGGER,TRAILING_GIVEBACK=36,11
EMA100_PERIOD=95
RUNUP_LOOKBACK=160
MAX_RUNUP_PERCENT=80
MAX_GAP_DOWN_PERCENT=-5
MIN_BARS=40

db=json.load(open(DB_FILE))
symbols=[s for s in db if s!="EGX30"]
all_cycles=[]

def calc_rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(com=n-1,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(com=n-1,adjust=False).mean()
    return 100-(100/(1+up/dn.replace(0,np.nan)))

for sym in symbols:
    raw=db[sym].get("data",db[sym])
    df=pd.DataFrame.from_dict(raw,orient="index")
    df.index=pd.to_datetime(df.index)
    df=df.sort_index()
    df.columns=[c.capitalize() for c in df.columns]
    if not all(c in df.columns for c in ["Open","High","Low","Close"]): continue
    df=df[(df.Open>0)&(df.High>0)&(df.Low>0)&(df.Close>0)].copy()
    if len(df)<MIN_BARS: continue

    df["EMA"]=df.Close.ewm(span=EMA100_PERIOD,adjust=False).mean()
    df["RSI"]=calc_rsi(df.Close)

    pos=0.0
    avg=0.0
    peak=0.0
    cycle=None
    cycle_id=0

    for i in range(max(MIN_BARS,RUNUP_LOOKBACK+1),len(df)-1):
        d=df.iloc[i]
        nxt=df.iloc[i+1]

        if pos==0:
            if d.EMA<=df.iloc[i-1].EMA: continue
            if d.Close>d.EMA*1.05: continue
            run=(d.Close/df.Close.iloc[max(0,i-RUNUP_LOOKBACK):i].min()-1)*100
            if run>MAX_RUNUP_PERCENT: continue
            gap=(d.Open/df.iloc[i-1].Close-1)*100
            if gap<=MAX_GAP_DOWN_PERCENT: continue
            if pd.isna(d.RSI) or d.RSI>BUY1_RSI: continue

            buy=nxt.Open
            if buy<=0: continue

            pos=.33
            avg=buy
            peak=0.0
            cycle_id+=1
            cycle={
                "symbol":sym,"id":cycle_id,
                "start":df.index[i+1],
                "avg":avg,"mdd":0.0,
                "max_profit":0.0
            }
            continue

        profit=(d.Close/avg-1)*100
        low_profit=(d.Low/avg-1)*100
        peak=max(peak,profit)
        cycle["max_profit"]=max(cycle["max_profit"],profit)
        cycle["mdd"]=min(cycle["mdd"],low_profit)

        action=None

        if pos<=.33 and profit<=STOP_L1: action="STOP"
        elif pos<=.66 and profit<=STOP_L2: action="STOP"
        elif pos>=.99 and profit<=STOP_L3: action="STOP"
        elif peak>TRAILING_TRIGGER and peak-profit>=TRAILING_GIVEBACK: action="TRAIL"
        elif pos<=.33 and profit>=SELL1_MIN_PROFIT and d.RSI>=SELL1_RSI: action="SELL1"
        elif .30<=pos<=.70 and profit>=SELL2_MIN_PROFIT and d.RSI>=SELL2_RSI: action="SELL2"
        elif pos>=.99 and profit>=SELL3_MIN_PROFIT and d.RSI>=SELL3_RSI: action="SELL3"

        if action is None and pos<=.33 and d.Close<avg*.98 and d.RSI<=BUY2_RSI:
            buy=nxt.Open
            if buy>0:
                avg=(avg*.33+buy*.33)/.66
                pos=.66
                continue

        if action is None and pos<=.66 and d.Close<avg*.97 and d.RSI<=BUY3_RSI:
            buy=nxt.Open
            if buy>0:
                avg=avg*.66+buy*.34
                pos=1.0
                continue

        if action and cycle is not None:
            exit_price=nxt.Open
            if exit_price<=0: continue
            final_profit=(exit_price/avg-1)*100
            cycle["end"]=df.index[i+1]
            cycle["days"]=(cycle["end"]-cycle["start"]).days
            cycle["months"]=cycle["days"]/30.44
            cycle["exit"]=action
            cycle["profit"]=final_profit
            all_cycles.append(cycle)

            pos=0.0
            avg=0.0
            peak=0.0
            cycle=None

print("\n"+"="*70)
print("📊 LADDER LONG / BORING TRADES + MDD ANALYSIS")
print("="*70)

if not all_cycles:
    print("No closed cycles found.")
    raise SystemExit

x=pd.DataFrame(all_cycles)

print(f"Closed cycles    : {len(x)}")
print(f"Average days     : {x.days.mean():.1f}")
print(f"Median days      : {x.days.median():.1f}")
print(f"Average months   : {x.months.mean():.2f}")
print(f"Median months    : {x.months.median():.2f}")
print(f"Average MDD      : {x.mdd.mean():.2f}%")
print(f"Median MDD       : {x.mdd.median():.2f}%")
print(f"Worst MDD        : {x.mdd.min():.2f}%")
print(f"Average MaxProfit: {x.max_profit.mean():.2f}%")

for m in [3,6,12]:
    q=x[x.months>m]
    print(f"\n⏳ > {m} MONTHS: {len(q)} trades")
    if len(q):
        print(f"Avg profit : {q.profit.mean():.2f}%")
        print(f"Win rate   : {(q.profit>0).mean()*100:.2f}%")
        print(f"Avg MDD    : {q.mdd.mean():.2f}%")
        print(f"Worst MDD  : {q.mdd.min():.2f}%")

print("\n"+"-"*70)
print("🐢 TOP 15 LONGEST CLOSED TRADES")
print("-"*70)

for _,r in x.sort_values("days",ascending=False).head(15).iterrows():
    print(f"{r.symbol:6} | {r.start.date()} → {r.end.date()} | {r.days:4}d ({r.months:5.1f}m) | Profit {r.profit:7.2f}% | MaxProfit {r.max_profit:7.2f}% | MDD {r.mdd:7.2f}% | {r.exit}")

print("\n"+"-"*70)
print("💥 TOP 15 WORST MDD")
print("-"*70)

for _,r in x.sort_values("mdd").head(15).iterrows():
    print(f"{r.symbol:6} | {r.start.date()} → {r.end.date()} | {r.days:4}d | Profit {r.profit:7.2f}% | MaxProfit {r.max_profit:7.2f}% | MDD {r.mdd:7.2f}% | {r.exit}")

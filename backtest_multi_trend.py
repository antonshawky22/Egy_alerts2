print("="*78)
print("EGX WEEKLY DIAGONAL TRENDLINE BACKTEST v2.0")
print("PURE ASCENDING TRENDLINE + PULLBACK + REBOUND + BREAK EXIT")
print("="*78)

import json,os
import pandas as pd
import numpy as np

# ================= CONFIG =================
DB_FILE="egx_weekly_database_v1.json"
RESULT_FILE="backtest_results.json"
TRADES_FILE="backtest_trades.json"

INITIAL_CAPITAL=100000.0
MAX_POSITIONS=8
POSITION_SIZE=1/MAX_POSITIONS

PIVOT_LEFT=3
PIVOT_RIGHT=3
MIN_PIVOTS=2
MAX_LINE_BARS=60
RECENT_PIVOTS=8

# الخط يجب أن يصعد على الأقل بهذا المعدل
MIN_SLOPE_PERCENT=.10

# قرب السعر من الخط
MAX_TOUCH_DISTANCE=3.0

# يجب أن يغلق السعر فوق الخط بهذا المقدار
MIN_REBOUND=.30

# لا ندخل إذا كان السعر بعيدًا جدًا عن الخط
MAX_EXTENSION=6.0

# هامش الوقف تحت الخط
STOP_BUFFER=1.0
MIN_STOP=3.0
MAX_STOP=8.0

# كسر الخط
BREAK_BUFFER=1.0

# حماية الأرباح
TRAIL_START=10.0
TRAIL_DISTANCE=5.0

MIN_BARS=120


# ================= DATA =================
if not os.path.exists(DB_FILE): raise FileNotFoundError(DB_FILE)
with open(DB_FILE,encoding="utf-8") as f: db=json.load(f)
print(f"\nDatabase loaded: {len(db)} symbols")


def to_df(x):
    if isinstance(x,dict) and "data" in x and "columns" in x:
        rows=[]
        for d,v in x["data"].items():
            if isinstance(v,dict): r=v.copy()
            elif len(v)==len(x["columns"]): r=dict(zip(x["columns"],v))
            else: continue
            r["Date"]=d; rows.append(r)
        df=pd.DataFrame(rows)
    elif isinstance(x,list): df=pd.DataFrame(x)
    else:return None

    if df.empty:return None
    df.columns=[str(c).strip().capitalize() for c in df.columns]
    req=["Date","Open","High","Low","Close"]
    if not all(c in df.columns for c in req):return None

    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    for c in req[1:]:df[c]=pd.to_numeric(df[c],errors="coerce")

    return (df.dropna(subset=req).sort_values("Date")
              .drop_duplicates("Date",keep="last")
              .reset_index(drop=True))


# ================= CONFIRMED PIVOTS =================
def get_pivots(df):
    x=df["Low"].values
    out=[]
    for i in range(PIVOT_LEFT,len(df)-PIVOT_RIGHT):
        if (x[i]<x[i-PIVOT_LEFT:i].min() and
            x[i]<=x[i+1:i+PIVOT_RIGHT+1].min()):
            out.append({
                "i":i,
                "c":i+PIVOT_RIGHT,
                "p":float(x[i])
            })
    return out


# ================= TRENDLINE =================
def make_line(a,b):
    if b["i"]<=a["i"] or b["p"]<=a["p"]:return None
    s=(b["p"]-a["p"])/(b["i"]-a["i"])
    if s/a["p"]*100<MIN_SLOPE_PERCENT:return None
    return {
        "a":a["i"],"b":b["i"],
        "p1":a["p"],"p2":b["p"],
        "s":s,
        "k":a["p"]-s*a["i"]
    }


def line_price(t,i):
    return t["s"]*i+t["k"]


def valid_line(df,t,a,b):
    # لا نسمح بانتهاك الدعم بأكثر من 1.5%
    for p in get_pivots(df):
        if p["i"]<=a["i"] or p["i"]>len(df)-1:continue
        lp=line_price(t,p["i"])
        if p["p"]<lp*.985:return False
    return True


def active_line(df,i):
    piv=[p for p in get_pivots(df.iloc[:i+1])
         if p["c"]<=i]

    if len(piv)<MIN_PIVOTS:return None

    recent=piv[-RECENT_PIVOTS:]
    candidates=[]

    for n,a in enumerate(recent[:-1]):
        for b in recent[n+1:]:
            if b["i"]<=a["i"]:continue
            if b["i"]-a["i"]>MAX_LINE_BARS:continue

            t=make_line(a,b)
            if not t:continue

            if valid_line(df.iloc[:i+1],t,a,b):
                score=b["i"]*100-(b["i"]-a["i"])
                candidates.append((score,t,a,b))

    if not candidates:return None
    _,t,a,b=max(candidates,key=lambda x:x[0])
    return {"line":t,"p1":a,"p2":b}


# ================= BACKTEST =================
def backtest(symbol,df):
    trades=[]
    pos=None
    break_count=0

    for i in range(MIN_BARS,len(df)):
        r=df.iloc[i]
        date=r.Date.strftime("%Y-%m-%d")
        close,high,low,op=map(float,[r.Close,r.High,r.Low,r.Open])

        info=active_line(df,i)
        t=info["line"] if info else None

        # ================================================================
        # ENTRY
        # ================================================================
        if pos is None and t:

            support=line_price(t,i)
            if support<=0:continue

            close_dist=(close-support)/support*100
            low_dist=(low-support)/support*100

            # الشمعة اختبرت الخط
            touched=low_dist<=MAX_TOUCH_DISTANCE

            # السعر ليس بعيدًا عن الخط
            near=0<=close_dist<=MAX_EXTENSION

            # شمعة ارتداد صاعدة
            bullish=close>op

            # الإغلاق فوق الخط
            rebound=close>=support*(1+MIN_REBOUND/100)

            if touched and near and bullish and rebound:

                pos={
                    "symbol":symbol,
                    "status":"OPEN",
                    "entry_date":date,
                    "entry_price":close,
                    "avg_price":close,
                    "weight":1.0,
                    "highest_price":close,
                    "entry_support":round(support,4),
                    "entry_distance_percent":round(close_dist,2),
                    "trendline_slope_percent":round(t["s"]/t["p1"]*100,4),
                    "pivot1_date":df.iloc[info["p1"]["i"]].Date.strftime("%Y-%m-%d"),
                    "pivot1_price":info["p1"]["p"],
                    "pivot2_date":df.iloc[info["p2"]["i"]].Date.strftime("%Y-%m-%d"),
                    "pivot2_price":info["p2"]["p"],
                    "sales":[],
                    "trail_active":False
                }
                continue

        if pos is None:continue

        avg=pos["avg_price"]
        prev_high=pos["highest_price"]

        # ================================================================
        # PROFIT PROTECTION
        # ================================================================
        profit=(prev_high-avg)/avg*100
        stop=avg*(1-MIN_STOP/100)

        # وقف أولي مرتبط بالخط الحالي
        if t:
            support=line_price(t,i)
            line_stop=support*(1-STOP_BUFFER/100)
            fixed_stop=avg*(1-MAX_STOP/100)
            stop=max(fixed_stop,min(line_stop,avg*(1-MIN_STOP/100)))
        else:
            stop=avg*(1-MIN_STOP/100)

        if profit>=TRAIL_START:
            pos["trail_active"]=True
            stop=max(stop,prev_high*(1-TRAIL_DISTANCE/100))

        # ================================================================
        # TRENDLINE BREAK
        # ================================================================
        if t:
            support=line_price(t,i)
            broken=close<support*(1-BREAK_BUFFER/100)
            break_count=break_count+1 if broken else 0
        else:
            break_count+=1

        trend_break=break_count>=1

        # ================================================================
        # EXIT
        # ================================================================
        if low<=stop:
            exit_price=stop
            reason="TRAIL_STOP" if pos["trail_active"] else "STOP_LOSS"

        elif trend_break:
            exit_price=close
            reason="TRENDLINE_BREAK"

        else:
            pos["highest_price"]=max(prev_high,high)
            continue

        pct=(exit_price-avg)/avg*100

        pos["sales"].append({
            "date":date,
            "price":round(exit_price,4),
            "weight":1.0,
            "profit_pct":round(pct,2),
            "capital_return":round(pct,4),
            "reason":reason
        })

        pos.update({
            "weight":0,
            "status":"CLOSED",
            "exit_date":date,
            "exit_price":round(exit_price,4),
            "exit_reason":reason,
            "profit_pct":round(pct,2)
        })

        trades.append(pos)
        pos=None
        break_count=0

    # ================================================================
    # OPEN
    # ================================================================
    if pos:
        last=float(df.iloc[-1].Close)
        pos["last_price"]=last
        pos["unrealized_pct"]=round((last-pos["avg_price"])/pos["avg_price"]*100,2)
        trades.append(pos)

    return trades


# ================= RUN =================
all_trades=[]
print("\nStarting backtest...\n")

for symbol,data in db.items():

    if symbol.upper() in {"EGX30","EGX70","EGX100"}:continue

    df=to_df(data)

    if df is None:
        print(f"⚠️ {symbol}: invalid data")
        continue

    if len(df)<MIN_BARS:
        print(f"⚠️ {symbol}: {len(df)} bars")
        continue

    trades=backtest(symbol,df)
    all_trades.extend(trades)

    closed=sum(t["status"]=="CLOSED" for t in trades)

    print(f"{symbol:8} | Closed Trades: {closed:3}")


# ================= STATISTICS =================
all_trades.sort(key=lambda x:x["entry_date"])

closed=[t for t in all_trades if t["status"]=="CLOSED"]
open_pos=[t for t in all_trades if t["status"]=="OPEN"]

profits=[float(t["profit_pct"]) for t in closed]
wins=[p for p in profits if p>0]
losses=[p for p in profits if p<=0]

total=len(profits)
winrate=len(wins)/total*100 if total else 0
sum_profit=sum(profits)

avg_win=np.mean(wins) if wins else 0
avg_loss=np.mean(losses) if losses else 0


# ================= PORTFOLIO =================
portfolio=INITIAL_CAPITAL
equity=[portfolio]
portfolio_history=[]

for t in closed:
    ret=t["profit_pct"]/100*POSITION_SIZE
    portfolio*=1+ret
    equity.append(portfolio)

    portfolio_history.append({
        "date":t["exit_date"],
        "symbol":t["symbol"],
        "trade_return_percent":round(t["profit_pct"],2),
        "portfolio_return_percent":round(ret*100,4),
        "portfolio_value":round(portfolio,2)
    })

compound=(portfolio/INITIAL_CAPITAL-1)*100


# ================= DRAWDOWN =================
peak=INITIAL_CAPITAL
max_dd=0

for v in equity:
    peak=max(peak,v)
    max_dd=max(max_dd,(peak-v)/peak*100)


# ================= EXIT ANALYSIS =================
exit_analysis={}

for t in closed:
    reason=t.get("exit_reason","UNKNOWN")
    exit_analysis[reason]=exit_analysis.get(reason,0)+1

best=max(closed,key=lambda x:x["profit_pct"]) if closed else None
worst=min(closed,key=lambda x:x["profit_pct"]) if closed else None


# ================= RESULT =================
result={
    "strategy":"Weekly Pure Diagonal Trendline Strategy v2.0",
    "description":"Confirmed ascending support trendline + pullback touch + bullish rebound + trendline break exit",
    "parameters":{
        "pivot_left":PIVOT_LEFT,
        "pivot_right":PIVOT_RIGHT,
        "max_line_bars":MAX_LINE_BARS,
        "min_slope_percent":MIN_SLOPE_PERCENT,
        "max_touch_distance_percent":MAX_TOUCH_DISTANCE,
        "min_rebound_percent":MIN_REBOUND,
        "max_extension_percent":MAX_EXTENSION,
        "stop_buffer_percent":STOP_BUFFER,
        "min_stop_percent":MIN_STOP,
        "max_stop_percent":MAX_STOP,
        "break_buffer_percent":BREAK_BUFFER,
        "trail_start_percent":TRAIL_START,
        "trail_distance_percent":TRAIL_DISTANCE,
        "max_positions":MAX_POSITIONS,
        "position_size_percent":POSITION_SIZE*100
    },
    "statistics":{
        "total_trades":total,
        "winning_trades":len(wins),
        "losing_trades":len(losses),
        "win_rate_percent":round(winrate,2),
        "sum_trade_profit_percent":round(sum_profit,2),
        "realistic_compound_return_percent":round(compound,2),
        "average_win_percent":round(avg_win,2),
        "average_loss_percent":round(avg_loss,2),
        "maximum_drawdown_percent":round(max_dd,2),
        "open_positions":len(open_pos)
    },
    "exit_analysis":exit_analysis,
    "best_trade":best,
    "worst_trade":worst,
    "open_positions":open_pos,
    "portfolio_equity":portfolio_history,
    "trades":all_trades
}


# ================= SAVE =================
with open(RESULT_FILE,"w",encoding="utf-8") as f:
    json.dump(result,f,ensure_ascii=False,indent=2)

with open(TRADES_FILE,"w",encoding="utf-8") as f:
    json.dump(all_trades,f,ensure_ascii=False,indent=2)


# ================= OUTPUT =================
print("\n"+"="*78)
print("FINAL BACKTEST RESULTS")
print("="*78)

print(f"Total Trades              : {total}")
print(f"Winning Trades            : {len(wins)}")
print(f"Losing Trades             : {len(losses)}")
print(f"Win Rate                  : {winrate:.2f}%")
print(f"Sum Trade Profit          : {sum_profit:.2f}%")
print(f"REALISTIC COMPOUND RETURN : {compound:.2f}%")
print(f"Average Win               : {avg_win:.2f}%")
print(f"Average Loss              : {avg_loss:.2f}%")
print(f"Maximum Drawdown          : {max_dd:.2f}%")
print(f"Open Positions            : {len(open_pos)}")

print("\nEXIT ANALYSIS")
print("-"*78)

for reason,count in exit_analysis.items():
    print(f"{reason:25} : {count}")

if best:
    print("\nBEST TRADE")
    print(f"{best['symbol']} | {best['profit_pct']:.2f}% | {best['entry_date']} -> {best['exit_date']}")

if worst:
    print("\nWORST TRADE")
    print(f"{worst['symbol']} | {worst['profit_pct']:.2f}% | {worst['entry_date']} -> {worst['exit_date']}")

print("\nFILES SAVED")
print(f"  {RESULT_FILE}")
print(f"  {TRADES_FILE}")
print("="*78)
print("BACKTEST COMPLETE")
print("="*78)

import json,os,pandas as pd,numpy as np

# ==========================================
# EGX LADDER CYCLE SYSTEM - FULL BACKTEST v1.2
# MARKET REGIME ANALYSIS + PORTFOLIO ORDER FIX
# ==========================================

DB_FILE="egx_history_database_v2.json"
RESULTS_FILE="backtest_results.json"
TRADES_FILE="backtest_trades.json"
SUMMARY_FILE="ladder_backtest_summary_by_stock.json"

INITIAL_CAPITAL=100000
MAX_PORTFOLIO_POSITIONS=8
POSITION_SIZE=1/MAX_PORTFOLIO_POSITIONS

RSI_PERIOD=14
EMA100_PERIOD=95
RUNUP_LOOKBACK=160
MAX_RUNUP_PERCENT=80
MAX_GAP_DOWN_PERCENT=-5

BUY1_RSI=60
BUY2_RSI=55
BUY3_RSI=48

SELL1_RSI=66
SELL1_MIN_PROFIT=15
SELL2_MIN_POSITION=.30
SELL2_MAX_POSITION=.70
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

SYMBOLS=[
"OLFI","EMFD","ETEL","EAST","EFIH","ABUK","OIH","SWDY","ISPH","ATQA",
"MTIE","HRHO","ORWE","JUFO","DSCW","SUGR","ELSH","RMDA","RAYA","EEII",
"MPCO","GBCO","TMGH","ORHD","AMOC","FWRY","COMI","ADIB","PHDC","MCQE",
"SKPC","EGAL"
]

def load_database():
    with open(DB_FILE,"r",encoding="utf-8") as f:
        return json.load(f)

def fetch_local_data(raw_database,symbol):
    if symbol not in raw_database:
        return pd.DataFrame()
    obj=raw_database[symbol]
    if isinstance(obj,dict) and "columns" in obj and "data" in obj:
        df=pd.DataFrame(obj["data"],columns=obj["columns"])
    elif isinstance(obj,list):
        df=pd.DataFrame(obj)
    else:
        return pd.DataFrame()
    df.columns=[str(c).lower() for c in df.columns]
    date_col="date" if "date" in df.columns else df.columns[0]
    df[date_col]=pd.to_datetime(df[date_col],errors="coerce")
    df=df.dropna(subset=[date_col]).set_index(date_col)
    rename={}
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            rename[c]=c
    df=df.rename(columns=rename)
    needed=["open","high","low","close"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    for c in needed:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=needed).sort_index()
    df=df[~df.index.duplicated(keep="last")]
    return df

def add_indicators(df):
    df=df.copy()
    df["ema95"]=df["close"].ewm(span=EMA100_PERIOD,adjust=False).mean()
    delta=df["close"].diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    avg_gain=gain.ewm(com=RSI_PERIOD-1,adjust=False).mean()
    avg_loss=loss.ewm(com=RSI_PERIOD-1,adjust=False).mean()
    rs=avg_gain/avg_loss.replace(0,np.nan)
    df["rsi"]=100-(100/(1+rs))
    df["ema95_prev5"]=df["ema95"].shift(5)
    return df

def prepare_market_regime(raw_database):
    df=fetch_local_data(raw_database,"EGX30")
    if df.empty:
        return pd.DataFrame()
    df=add_indicators(df)
    df["ema95_slope_5"]=df["ema95"]-df["ema95_prev5"]
    df["above_ema95"]=df["close"]>df["ema95"]
    df["below_ema95"]=df["close"]<df["ema95"]
    conditions=[
        (df["close"]>df["ema95"])&(df["ema95"]>df["ema95_prev5"]),
        (df["close"]<df["ema95"])&(df["ema95"]<df["ema95_prev5"])
    ]
    df["market_regime"]=np.select(
        conditions,
        ["BULL","BEAR"],
        default="DIFFICULT / SIDEWAYS"
    )
    df.loc[df["ema95_prev5"].isna(),"market_regime"]="UNKNOWN"
    return df

def get_market_regime(date,market_df):
    if market_df.empty:
        return "UNKNOWN"
    date=pd.Timestamp(date)
    pos=market_df.index.searchsorted(date,side="right")-1
    if pos<0:
        return "UNKNOWN"
    return str(market_df.iloc[pos]["market_regime"])

def get_market_value(date,market_df,column):
    if market_df.empty:
        return None
    date=pd.Timestamp(date)
    pos=market_df.index.searchsorted(date,side="right")-1
    if pos<0:
        return None
    value=market_df.iloc[pos][column]
    return None if pd.isna(value) else float(value)

def longest_true_streak(mask):
    best=cur=0
    for x in mask:
        if bool(x):
            cur+=1
            best=max(best,cur)
        else:
            cur=0
    return best

def safe_to_buy(df,i):
    start=max(0,i-RUNUP_LOOKBACK+1)
    section=df.iloc[start:i+1]
    if section.empty:
        return False
    low=section["low"].min()
    high=section["high"].max()
    if low<=0:
        return False
    runup=(high-low)/low*100
    return runup<=MAX_RUNUP_PERCENT

def no_gap_down(df,i):
    if i<3:
        return True
    for j in range(i-2,i+1):
        if j<=0:
            continue
        gap=(df.iloc[j]["open"]-df.iloc[j-1]["close"])/df.iloc[j-1]["close"]*100
        if gap<=MAX_GAP_DOWN_PERCENT:
            return False
    return True

def ema_up(df,i):
    if i<10:
        return False
    e=df.iloc[i]["ema95"]
    e5=df.iloc[i-5]["ema95"]
    e10=df.iloc[i-10]["ema95"]
    if pd.isna(e) or pd.isna(e5) or pd.isna(e10):
        return False
    return e>e5>e10 and e>e10*1.002 and df.iloc[i]["close"]<=e*1.07

def update_avg(avg,position,price,add_position):
    if position<=0:
        return price
    return ((avg*position)+(price*add_position))/(position+add_position)

def create_trade(symbol,date,price,market_df):
    return {
        "symbol":symbol,
        "status":"OPEN",
        "cycle":1,
        "first_entry_date":str(date.date()),
        "last_entry_date":str(date.date()),
        "exit_date":None,
        "first_entry_price":round(float(price),6),
        "avg_price":round(float(price),6),
        "last_price":round(float(price),6),
        "position":.33,
        "peak_profit":0,
        "realized_profit":0,
        "second_entry":None,
        "third_entry":None,
        "last_totally_average_price":round(float(price),6),
        "entry_market_regime":get_market_regime(date,market_df),
        "entry_egx30_close":get_market_value(date,market_df,"close"),
        "entry_egx30_ema95":get_market_value(date,market_df,"ema95"),
        "exit_market_regime":None,
        "holding_days":None,
        "exit_reason":None,
        "profit_pct":0
    }

def calculate_current_profit(state,price):
    if state["avg_price"]<=0:
        return 0
    return (price-state["avg_price"])/state["avg_price"]*100

def close_trade(state,date,price,reason,market_df):
    profit=calculate_current_profit(state,price)
    remaining=state["position"]
    if remaining>0:
        state["realized_profit"]+=profit*remaining
    total_profit=state["realized_profit"]
    state["last_price"]=round(float(price),6)
    state["profit_pct"]=round(float(total_profit),4)
    state["exit_date"]=str(pd.Timestamp(date).date())
    state["exit_reason"]=reason
    state["exit_market_regime"]=get_market_regime(date,market_df)
    state["holding_days"]=(pd.Timestamp(date)-pd.Timestamp(state["first_entry_date"])).days
    state["status"]="CLOSED"
    state["position"]=0
    return state

def partial_sell(state,date,price,sell_position,reason):
    profit=calculate_current_profit(state,price)
    state["realized_profit"]+=profit*sell_position
    state["position"]=round(state["position"]-sell_position,4)
    state["last_price"]=round(float(price),6)
    state["profit_pct"]=round(float(state["realized_profit"]),4)
    state["exit_reason"]=reason
    state["last_sell_date"]=str(pd.Timestamp(date).date())
    return state

def run_symbol_backtest(symbol,raw_database,market_df):
    df=fetch_local_data(raw_database,symbol)
    if len(df)<MIN_BARS:
        return [],[]
    df=add_indicators(df)
    trades=[]
    signals=[]
    state=None
    for i in range(len(df)):
        date=df.index[i]
        row=df.iloc[i]
        price=float(row["close"])
        rsi=row["rsi"]
        if pd.isna(rsi) or pd.isna(row["ema95"]):
            continue
        if state is not None:
            state["last_price"]=round(price,6)
            current_profit=calculate_current_profit(state,price)
            state["peak_profit"]=max(state["peak_profit"],current_profit)
        if state is None:
            if (
                ema_up(df,i)
                and safe_to_buy(df,i)
                and no_gap_down(df,i)
                and rsi<=BUY1_RSI
            ):
                state=create_trade(symbol,date,price,market_df)
                signals.append({
                    "symbol":symbol,"date":str(date.date()),"action":"BUY L1",
                    "price":round(price,6),"rsi":round(float(rsi),2),
                    "position":state["position"],
                    "market_regime":state["entry_market_regime"]
                })
            continue
        pos=state["position"]
        avg=state["avg_price"]
        profit=calculate_current_profit(state,price)
        if pos<=.33:
            stop_level=STOP_L1
        elif pos<=.66:
            stop_level=STOP_L2
        else:
            stop_level=STOP_L3
        trailing=state["peak_profit"]>TRAILING_TRIGGER and state["peak_profit"]-profit>=TRAILING_GIVEBACK
        hard_stop=profit<=stop_level
        full_exit=rsi>=SELL3_RSI and profit>SELL3_MIN_PROFIT
        sell2=.30<pos<=SELL2_MAX_POSITION and rsi>=SELL2_RSI and profit>SELL2_MIN_PROFIT
        sell1=pos>.70 and rsi>=SELL1_RSI and profit>SELL1_MIN_PROFIT
        action=None
        if trailing:
            action="TRAILING STOP"
        elif hard_stop:
            action="STOP LOSS"
        elif full_exit:
            action="FULL EXIT"
        elif sell2:
            action="SELL L2"
        elif sell1:
            action="SELL L1"
        if action:
            if action in ["TRAILING STOP","STOP LOSS","FULL EXIT"]:
                closed=close_trade(state,date,price,action,market_df)
                trades.append(closed.copy())
                signals.append({
                    "symbol":symbol,"date":str(date.date()),"action":action,
                    "price":round(price,6),"rsi":round(float(rsi),2),
                    "profit":round(float(closed["profit_pct"]),2)
                })
                state=None
                continue
            if action=="SELL L2":
                partial_sell(state,date,price,.33,action)
                signals.append({
                    "symbol":symbol,"date":str(date.date()),"action":action,
                    "price":round(price,6),"rsi":round(float(rsi),2),
                    "profit":round(float(profit),2)
                })
                continue
            if action=="SELL L1":
                partial_sell(state,date,price,.33,action)
                signals.append({
                    "symbol":symbol,"date":str(date.date()),"action":action,
                    "price":round(price,6),"rsi":round(float(rsi),2),
                    "profit":round(float(profit),2)
                })
                continue
        if state is not None:
            pos=state["position"]
            if pos<=.33 and rsi<=BUY2_RSI and price<avg*.97 and ema_up(df,i) and no_gap_down(df,i):
                old_pos=pos
                state["avg_price"]=update_avg(avg,pos,price,.33)
                state["position"]=round(pos+.33,4)
                state["second_entry"]={
                    "date":str(date.date()),"price":round(price,6)
                }
                state["last_totally_average_price"]=round(state["avg_price"],6)
                signals.append({
                    "symbol":symbol,"date":str(date.date()),"action":"BUY L2",
                    "price":round(price,6),"rsi":round(float(rsi),2),
                    "position":state["position"]
                })
                continue
            if .33<=pos<1 and rsi<=BUY3_RSI and price<avg*.94 and ema_up(df,i) and no_gap_down(df,i):
                state["avg_price"]=update_avg(avg,pos,price,.34)
                state["position"]=1.0
                state["third_entry"]={
                    "date":str(date.date()),"price":round(price,6)
                }
                state["last_totally_average_price"]=round(state["avg_price"],6)
                signals.append({
                    "symbol":symbol,"date":str(date.date()),"action":"BUY L3",
                    "price":round(price,6),"rsi":round(float(rsi),2),
                    "position":state["position"]
                })
                continue
    if state is not None:
        end_date=df.index[-1]
        end_price=float(df.iloc[-1]["close"])
        state["last_price"]=round(end_price,6)
        state["profit_pct"]=round(
            state["realized_profit"]+
            calculate_current_profit(state,end_price)*state["position"],4
        )
        state["exit_market_regime"]=get_market_regime(end_date,market_df)
        state["end_date"]=str(end_date.date())
        state["status"]="OPEN"
        trades.append(state.copy())
    return trades,signals

def calculate_trade_stats(trades):
    closed=[t for t in trades if t.get("status")=="CLOSED"]
    if not closed:
        return {
            "trades":0,"wins":0,"losses":0,"win_rate":0,
            "sum_profit":0,"avg_profit":0,"avg_win":0,"avg_loss":0,
            "gross_profit":0,"gross_loss":0,"profit_factor":0,
            "worst_trade":0,"best_trade":0
        }
    profits=[float(t.get("profit_pct",0)) for t in closed]
    wins=[p for p in profits if p>0]
    losses=[p for p in profits if p<=0]
    gross_profit=sum(wins)
    gross_loss=abs(sum(losses))
    return {
        "trades":len(profits),
        "wins":len(wins),
        "losses":len(losses),
        "win_rate":round(len(wins)/len(profits)*100,2),
        "sum_profit":round(sum(profits),2),
        "avg_profit":round(np.mean(profits),2),
        "avg_win":round(np.mean(wins),2) if wins else 0,
        "avg_loss":round(np.mean(losses),2) if losses else 0,
        "gross_profit":round(gross_profit,2),
        "gross_loss":round(gross_loss,2),
        "profit_factor":round(gross_profit/gross_loss,2) if gross_loss else 999,
        "worst_trade":round(min(profits),2),
        "best_trade":round(max(profits),2)
    }

def regime_trade_stats(trades,regime):
    selected=[
        t for t in trades
        if t.get("status")=="CLOSED" and
        t.get("entry_market_regime")==regime
    ]
    stats=calculate_trade_stats(selected)
    stats["stop_loss_count"]=sum(t.get("exit_reason")=="STOP LOSS" for t in selected)
    stats["trailing_stop_count"]=sum(t.get("exit_reason")=="TRAILING STOP" for t in selected)
    stats["l3_count"]=sum(t.get("third_entry") is not None for t in selected)
    return stats

def regime_transition_stats(trades):
    result={}
    closed=[t for t in trades if t.get("status")=="CLOSED"]
    for t in closed:
        key=f'{t.get("entry_market_regime","UNKNOWN")} -> {t.get("exit_market_regime","UNKNOWN")}'
        if key not in result:
            result[key]={"trades":0,"sum_profit":0,"avg_profit":0}
        result[key]["trades"]+=1
        result[key]["sum_profit"]+=float(t.get("profit_pct",0))
    for v in result.values():
        v["sum_profit"]=round(v["sum_profit"],2)
        v["avg_profit"]=round(v["sum_profit"]/v["trades"],2)
    return result

def consecutive_losses(trades):
    closed=sorted(
        [t for t in trades if t.get("status")=="CLOSED"],
        key=lambda x:x.get("exit_date") or ""
    )
    best=cur=0
    for t in closed:
        if float(t.get("profit_pct",0))<=0:
            cur+=1
            best=max(best,cur)
        else:
            cur=0
    return best

def portfolio_simulation(closed_trades):
    ordered=sorted(
        closed_trades,
        key=lambda t:pd.Timestamp(t["exit_date"])
    )
    capital=INITIAL_CAPITAL
    peak=capital
    max_dd=0
    equity=[]
    for t in ordered:
        ret=float(t.get("profit_pct",0))/100
        capital*=1+(ret*POSITION_SIZE)
        peak=max(peak,capital)
        dd=(peak-capital)/peak*100
        max_dd=max(max_dd,dd)
        equity.append({
            "date":t["exit_date"],
            "capital":round(capital,2),
            "drawdown":round(dd,2)
        })
    compound=(capital/INITIAL_CAPITAL-1)*100
    return {
        "final_capital":round(capital,2),
        "compound_return_percent":round(compound,2),
        "max_drawdown_percent":round(max_dd,2),
        "equity_curve":equity,
        "portfolio_model":"Sequential closed-trade compounding at 12.5% per trade; not a concurrent daily mark-to-market portfolio."
    }

def max_concurrent_trades(trades):
    events=[]
    for t in trades:
        if t.get("status")!="CLOSED":
            continue
        try:
            entry=pd.Timestamp(t["first_entry_date"])
            exit=pd.Timestamp(t["exit_date"])
        except:
            continue
        events.append((entry,1))
        events.append((exit,-1))
    events.sort(key=lambda x:(x[0],x[1]))
    current=best=0
    for _,change in events:
        current+=change
        best=max(best,current)
    return best

def market_summary(market_df):
    if market_df.empty:
        return {}
    total=len(market_df)
    result={}
    for regime in ["BULL","DIFFICULT / SIDEWAYS","BEAR","UNKNOWN"]:
        count=int((market_df["market_regime"]==regime).sum())
        result[regime]={
            "days":count,
            "percent":round(count/total*100,2) if total else 0
        }
    below=int(market_df["below_ema95"].sum())
    above=int(market_df["above_ema95"].sum())
    below_streak=longest_true_streak(market_df["below_ema95"].fillna(False).tolist())
    bear_streak=longest_true_streak((market_df["market_regime"]=="BEAR").tolist())
    return {
        "period_start":str(market_df.index.min().date()),
        "period_end":str(market_df.index.max().date()),
        "total_days":total,
        "regimes":result,
        "simple_above_ema95_days":above,
        "simple_below_ema95_days":below,
        "simple_above_ema95_percent":round(above/total*100,2),
        "simple_below_ema95_percent":round(below/total*100,2),
        "longest_below_ema95_streak":below_streak,
        "longest_bear_regime_streak":bear_streak
    }

# ==========================================
# MAIN
# ==========================================

print("="*70)
print("🚀 EGX LADDER BACKTEST v1.2")
print("📊 MARKET REGIME ANALYSIS + PORTFOLIO ORDER FIX")
print("="*70)

raw_database=load_database()
market_df=prepare_market_regime(raw_database)

if market_df.empty:
    print("❌ EGX30 data not found in database.")
    raise SystemExit

print(f"📅 EGX30 PERIOD: {market_df.index.min().date()} -> {market_df.index.max().date()}")

all_trades=[]
all_signals=[]

for n,symbol in enumerate(SYMBOLS,1):
    print(f"[{n}/{len(SYMBOLS)}] {symbol}")
    trades,signals=run_symbol_backtest(symbol,raw_database,market_df)
    all_trades.extend(trades)
    all_signals.extend(signals)

closed_trades=[t for t in all_trades if t.get("status")=="CLOSED"]
open_trades=[t for t in all_trades if t.get("status")=="OPEN"]

overall=calculate_trade_stats(all_trades)
portfolio=portfolio_simulation(closed_trades)

# ==========================================
# MARKET REGIME ANALYSIS
# ==========================================

mkt=market_summary(market_df)

regimes=["BULL","DIFFICULT / SIDEWAYS","BEAR","UNKNOWN"]
regime_stats={r:regime_trade_stats(all_trades,r) for r in regimes}

open_by_regime={}
for t in open_trades:
    r=t.get("entry_market_regime","UNKNOWN")
    open_by_regime[r]=open_by_regime.get(r,0)+1

transition_stats=regime_transition_stats(all_trades)

# ==========================================
# EXIT COUNTS
# ==========================================

exit_counts={}
for t in closed_trades:
    reason=t.get("exit_reason","UNKNOWN")
    exit_counts[reason]=exit_counts.get(reason,0)+1

# ==========================================
# LADDER COUNTS
# ==========================================

l1_count=sum(1 for s in all_signals if s["action"]=="BUY L1")
l2_count=sum(1 for s in all_signals if s["action"]=="BUY L2")
l3_count=sum(1 for s in all_signals if s["action"]=="BUY L3")
sell1_count=sum(1 for s in all_signals if s["action"]=="SELL L1")
sell2_count=sum(1 for s in all_signals if s["action"]=="SELL L2")

# ==========================================
# BY STOCK
# ==========================================

summary_by_stock={}
for symbol in SYMBOLS:
    st=[t for t in all_trades if t["symbol"]==symbol]
    closed=[t for t in st if t.get("status")=="CLOSED"]
    stats=calculate_trade_stats(st)
    summary_by_stock[symbol]={
        **stats,
        "open_trades":sum(1 for t in st if t.get("status")=="OPEN"),
        "stop_loss_count":sum(t.get("exit_reason")=="STOP LOSS" for t in closed),
        "trailing_stop_count":sum(t.get("exit_reason")=="TRAILING STOP" for t in closed),
        "l3_count":sum(t.get("third_entry") is not None for t in st)
    }

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n"+"="*70)
print("📊 OVERALL RESULTS")
print("="*70)
for k,v in overall.items():
    print(f"{k}: {v}")

print("\n💰 PORTFOLIO MODEL")
print(f"Final Capital       : {portfolio['final_capital']:,.2f}")
print(f"Compound Return     : {portfolio['compound_return_percent']:.2f}%")
print(f"Max Drawdown        : {portfolio['max_drawdown_percent']:.2f}%")
print("⚠️ "+portfolio["portfolio_model"])

print("\n"+"="*70)
print("🏛️ MARKET REGIME ANALYSIS - EGX30 / EMA95")
print("="*70)
print(f"Period              : {mkt['period_start']} -> {mkt['period_end']}")
print(f"Total Days          : {mkt['total_days']}")
print(f"Above EMA95         : {mkt['simple_above_ema95_days']} days ({mkt['simple_above_ema95_percent']}%)")
print(f"Below EMA95         : {mkt['simple_below_ema95_days']} days ({mkt['simple_below_ema95_percent']}%)")
print(f"Longest Below EMA95 : {mkt['longest_below_ema95_streak']} days")
print(f"Longest BEAR        : {mkt['longest_bear_regime_streak']} days")

for r in regimes:
    x=mkt["regimes"][r]
    print(f"{r:22}: {x['days']} days ({x['percent']}%)")

print("\n"+"="*70)
print("📈 TRADE PERFORMANCE BY ENTRY REGIME")
print("="*70)

for r in regimes:
    s=regime_stats[r]
    print(f"\n--- {r} ---")
    print(f"Trades       : {s['trades']}")
    print(f"Wins/Losses  : {s['wins']}/{s['losses']}")
    print(f"Win Rate     : {s['win_rate']}%")
    print(f"Sum Profit   : {s['sum_profit']}%")
    print(f"Avg Trade    : {s['avg_profit']}%")
    print(f"Avg Win      : {s['avg_win']}%")
    print(f"Avg Loss     : {s['avg_loss']}%")
    print(f"Profit Factor: {s['profit_factor']}")
    print(f"Worst Trade  : {s['worst_trade']}%")
    print(f"Stops        : {s['stop_loss_count']}")
    print(f"Trailing     : {s['trailing_stop_count']}")
    print(f"L3 Entries   : {s['l3_count']}")

if regime_stats["BEAR"]["trades"]==0:
    print("\n⚠️ NO CLOSED TRADES ENTERED DURING BEAR REGIME.")
    print("⚠️ THIS BACKTEST CANNOT VALIDATE BEAR-MARKET SURVIVAL.")

print("\n"+"="*70)
print("🔄 ENTRY -> EXIT MARKET REGIME")
print("="*70)
for k,v in sorted(transition_stats.items()):
    print(f"{k}: {v['trades']} trades | Sum {v['sum_profit']}% | Avg {v['avg_profit']}%")

print("\n"+"="*70)
print("📌 OPEN POSITIONS BY ENTRY REGIME")
print("="*70)
for r in regimes:
    print(f"{r}: {open_by_regime.get(r,0)}")

print("\n"+"="*70)
print("📊 TRADE / LADDER SUMMARY")
print("="*70)
print(f"Closed Trades        : {len(closed_trades)}")
print(f"Open Trades          : {len(open_trades)}")
print(f"BUY L1               : {l1_count}")
print(f"BUY L2               : {l2_count}")
print(f"BUY L3               : {l3_count}")
print(f"SELL L1              : {sell1_count}")
print(f"SELL L2              : {sell2_count}")
print(f"Stop Loss            : {exit_counts.get('STOP LOSS',0)}")
print(f"Trailing Stop        : {exit_counts.get('TRAILING STOP',0)}")
print(f"Full Exit            : {exit_counts.get('FULL EXIT',0)}")
print(f"Max Concurrent Trades: {max_concurrent_trades(all_trades)}")
print(f"Max Consecutive Loss : {consecutive_losses(all_trades)}")

holding=[t["holding_days"] for t in closed_trades if t.get("holding_days") is not None]
if holding:
    print(f"Longest Holding Days : {max(holding)}")
    print(f"Average Holding Days : {round(np.mean(holding),1)}")

# ==========================================
# SAVE RESULTS
# ==========================================

results={
    "backtest_version":"v1.2",
    "parameters":{
        "RSI_PERIOD":RSI_PERIOD,
        "EMA95_PERIOD":EMA100_PERIOD,
        "RUNUP_LOOKBACK":RUNUP_LOOKBACK,
        "MAX_RUNUP_PERCENT":MAX_RUNUP_PERCENT,
        "MAX_GAP_DOWN_PERCENT":MAX_GAP_DOWN_PERCENT,
        "BUY1_RSI":BUY1_RSI,
        "BUY2_RSI":BUY2_RSI,
        "BUY3_RSI":BUY3_RSI,
        "SELL1_RSI":SELL1_RSI,
        "SELL1_MIN_PROFIT":SELL1_MIN_PROFIT,
        "SELL2_RSI":SELL2_RSI,
        "SELL2_MIN_PROFIT":SELL2_MIN_PROFIT,
        "SELL3_RSI":SELL3_RSI,
        "SELL3_MIN_PROFIT":SELL3_MIN_PROFIT,
        "STOP_L1":STOP_L1,
        "STOP_L2":STOP_L2,
        "STOP_L3":STOP_L3,
        "TRAILING_TRIGGER":TRAILING_TRIGGER,
        "TRAILING_GIVEBACK":TRAILING_GIVEBACK
    },
    "overall":overall,
    "portfolio":portfolio,
    "market_regime_summary":mkt,
    "trade_performance_by_entry_regime":regime_stats,
    "regime_transition_stats":transition_stats,
    "open_positions_by_entry_regime":open_by_regime,
    "exit_counts":exit_counts,
    "ladder_counts":{
        "BUY_L1":l1_count,
        "BUY_L2":l2_count,
        "BUY_L3":l3_count,
        "SELL_L1":sell1_count,
        "SELL_L2":sell2_count
    },
    "max_concurrent_trades":max_concurrent_trades(all_trades),
    "max_consecutive_losses":consecutive_losses(all_trades),
    "longest_holding_days":max(holding) if holding else 0,
    "average_holding_days":round(float(np.mean(holding)),2) if holding else 0,
    "open_trades":len(open_trades),
    "closed_trades":len(closed_trades),
    "portfolio_model_warning":"Sequential closed-trade compounding at 12.5% per trade; not a true concurrent daily mark-to-market portfolio."
}

with open(RESULTS_FILE,"w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

with open(TRADES_FILE,"w",encoding="utf-8") as f:
    json.dump(all_trades,f,ensure_ascii=False,indent=2)

with open(SUMMARY_FILE,"w",encoding="utf-8") as f:
    json.dump(summary_by_stock,f,ensure_ascii=False,indent=2)

print("\n"+"="*70)
print("✅ BACKTEST FINISHED")
print("="*70)
print(f"📁 {RESULTS_FILE}")
print(f"📁 {TRADES_FILE}")
print(f"📁 {SUMMARY_FILE}")
print("="*70)

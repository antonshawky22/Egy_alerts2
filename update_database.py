import json,os,time,sys,argparse,requests,pandas as pd
from tradingview_ta import Interval as TVInterval,get_multiple_analysis
try:
 from tradingviewApiPython import Client
 HISTORICAL_API_AVAILABLE=True
except ImportError:
 HISTORICAL_API_AVAILABLE=False
TOKEN=os.getenv("TELEGRAM_TOKEN");CHAT_ID=os.getenv("TELEGRAM_CHAT_ID")
DB_FILE="egx_history_database_v2.json";REPORT_FILE="egx_history_database_quality_v2.json";TEMP_DB_FILE=DB_FILE+".tmp"
INITIAL_REFRESH_BARS=500;GAP_REFRESH_BARS=150;MAX_DATABASE_BARS=500;GAP_REFRESH_PERCENT=5.0;MAX_RETRIES=3;RETRY_DELAY=5;REQUEST_DELAY=2
SYMBOLS="EGX30 OLFI EMFD ETEL EAST EFIH ABUK OIH SWDY ISPH ATQA MTIE HRHO ORWE JUFO DSCW SUGR ELSH RMDA RAYA EEII MPCO GBCO TMGH ORHD AMOC FWRY COMI ADIB PHDC MCQE SKPC EGAL".split()
COLUMNS=["Open","High","Low","Close","Volume"]
p=argparse.ArgumentParser();p.add_argument("--refresh-history",action="store_true");args=p.parse_args()
MANUAL_REFRESH_HISTORY=args.refresh_history or os.getenv("REFRESH_HISTORY","").lower() in ["1","true","yes","on"]
def send_telegram(text):
 if not TOKEN or not CHAT_ID:return
 try:requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",data={"chat_id":CHAT_ID,"text":text,"parse_mode":"Markdown"},timeout=15)
 except Exception as e:print(f"⚠️ Telegram error: {e}")
def load_database():
 if not os.path.exists(DB_FILE):return {}
 try:
  with open(DB_FILE,"r",encoding="utf-8") as f:raw=json.load(f)
 except Exception as e:print(f"❌ Database load error: {e}");return {}
 out={}
 for name in SYMBOLS:
  x=raw.get(name)
  if not isinstance(x,dict) or "data" not in x:out[name]=pd.DataFrame(columns=COLUMNS);continue
  try:
   df=pd.DataFrame.from_dict(x["data"],orient="index")
   for c in COLUMNS:
    if c not in df.columns:df[c]=None
   df=df[COLUMNS];df.index=pd.to_datetime(df.index,errors="coerce");df=df[~df.index.isna()]
   for c in COLUMNS:df[c]=pd.to_numeric(df[c],errors="coerce")
   df=df[~df.index.duplicated(keep="last")].sort_index();out[name]=df
  except Exception as e:print(f"⚠️ {name} load error: {e}");out[name]=pd.DataFrame(columns=COLUMNS)
 return out
def tv_bulk():
 for attempt in range(1,MAX_RETRIES+1):
  try:
   print(f"🔌 Bulk attempt {attempt}/{MAX_RETRIES}")
   x=get_multiple_analysis(screener="egypt",interval=TVInterval.INTERVAL_1_DAY,symbols=[f"EGX:{s}" for s in SYMBOLS])
   if x:return x
  except Exception as e:print(f"⚠️ Bulk error: {e}")
  if attempt<MAX_RETRIES:time.sleep(RETRY_DELAY)
 return None
def bulk_row(bulk,name):
 try:
  x=bulk.get(f"EGX:{name}") or bulk.get(name)
  if not x:return None,None
  i=x.indicators
  r={"Open":float(i["open"]),"High":float(i["high"]),"Low":float(i["low"]),"Close":float(i["close"]),"Volume":float(i.get("volume",0))}
  return r,pd.Timestamp(x.time).normalize()
 except Exception as e:
  print(f"⚠️ {name} live data error: {e}");return None,None
def valid_row(r):
 try:
  o,h,l,c,v=[float(r[x]) for x in COLUMNS]
  return o>0 and h>0 and l>0 and c>0 and h>=max(o,c,l) and l<=min(o,c,h) and v>=0
 except:return False
def historical(name,bars):
 if not HISTORICAL_API_AVAILABLE:return pd.DataFrame(columns=COLUMNS)
 try:
  chart=Client().Session.Chart();chart.set_market(f"EGX:{name}",{"timeframe":"D","range":bars})
  rows=[]
  for b in chart.periods:
   try:
    d=pd.to_datetime(int(b["time"]),unit="s").normalize()
    r={"Open":float(b["open"]),"High":float(b["max"]),"Low":float(b["min"]),"Close":float(b["close"]),"Volume":float(b.get("volume",0))}
    if valid_row(r):rows.append((d,r))
   except:continue
  if not rows:return pd.DataFrame(columns=COLUMNS)
  df=pd.DataFrame([r for _,r in rows],index=[d for d,_ in rows]);df=df[~df.index.duplicated(keep="last")].sort_index();return df[COLUMNS]
 except Exception as e:
  print(f"❌ {name} historical error: {e}");return pd.DataFrame(columns=COLUMNS)
def normalize(df):
 if df.empty:return df
 df=df.copy();df.index=pd.to_datetime(df.index).normalize()
 for c in COLUMNS:df[c]=pd.to_numeric(df[c],errors="coerce")
 df=df[COLUMNS];df=df[~df.index.duplicated(keep="last")].sort_index()
 return df.tail(MAX_DATABASE_BARS).round(2)
def save_database(db):
 blocks=[]
 for name in SYMBOLS:
  df=normalize(db.get(name,pd.DataFrame(columns=COLUMNS))).sort_index(ascending=False)
  data={str(i.date()):{c:(None if pd.isna(r[c]) else float(r[c])) for c in COLUMNS} for i,r in df.iterrows()}
  blocks.append(f'  "{name}": {{\n    "columns": {json.dumps(COLUMNS)},\n    "data": {json.dumps(data,separators=(",",":"))}\n  }}')
 text="{\n"+",\n".join(blocks)+"\n}\n"
 with open(TEMP_DB_FILE,"w",encoding="utf-8") as f:f.write(text)
 os.replace(TEMP_DB_FILE,DB_FILE)
def quality(db,gaps,hist_ok,hist_fail,check_date):
 report={}
 for name in SYMBOLS:
  df=normalize(db.get(name,pd.DataFrame(columns=COLUMNS)));bad=0;large=0;date_gaps=0
  if not df.empty:
   bad=int(((df["High"]<df[["Open","Close","Low"]].max(axis=1))|(df["Low"]>df[["Open","Close","High"]].min(axis=1))|(df[COLUMNS]<=0).any(axis=1)).sum())
   large=int((df["Close"].pct_change().abs()>=.10).sum());date_gaps=int((df.index.to_series().diff().dt.days>5).sum())
  report[name]={"bars":len(df),"start":str(df.index.min().date()) if len(df) else None,"end":str(df.index.max().date()) if len(df) else None,"bad_ohlc":bad,"large_moves_10pct":large,"date_gaps_over_5d":date_gaps,"automatic_gap_trigger":name in gaps,"historical_refresh":"success" if name in hist_ok else "failed" if name in hist_fail else "not_requested"}
 with open(REPORT_FILE,"w",encoding="utf-8") as f:json.dump({"check_date":str(check_date),"symbols":report},f,ensure_ascii=False,indent=2)
def main():
 print("⚙️ EGX ENGINE v13.1 - TradingView Production Engine")
 print(f"📊 Final database: {MAX_DATABASE_BARS} bars | Gap refresh: {GAP_REFRESH_BARS} bars | Manual/Initial: {INITIAL_REFRESH_BARS} bars")
 db=load_database();bulk=tv_bulk()
 if not bulk:
  send_telegram("🚨 *CRITICAL: TradingView bulk fetch failed*\n❌ Database was not modified.");print("❌ TradingView bulk fetch failed. Database unchanged.");return
 pulse=False
 for n in ["COMI","SWDY","HRHO"]:
  r,d=bulk_row(bulk,n);df=db.get(n,pd.DataFrame())
  if r and (df.empty or not valid_row(r)):pulse=True;break
  if r and not df.empty:
   old=df.iloc[-1]
   if float(r["Volume"])>0 and any(abs(float(r[c])-float(old[c]))>0.000001 for c in ["Open","High","Low","Close"]):pulse=True;break
 if not pulse and not MANUAL_REFRESH_HISTORY:
  print("ℹ️ No real market activity detected. Safe exit.");return
 updated=[];gaps=[];auto_refresh_tickers=[];live_dates={}
 for name in SYMBOLS:
  r,candle_date=bulk_row(bulk,name)
  if not r or not candle_date or not valid_row(r):
   print(f"⚠️ Invalid live data: {name}");continue
  live_dates[name]=candle_date
  df=db.get(name,pd.DataFrame(columns=COLUMNS)).copy()
  if df.empty:
   auto_refresh_tickers.append(name);print(f"🆕 {name}: new symbol -> {INITIAL_REFRESH_BARS} bars")
  else:
   prev=df[df.index<candle_date]
   if not prev.empty:
    gap=(float(r["Open"])-float(prev.iloc[-1]["Close"]))/float(prev.iloc[-1]["Close"])*100
    if abs(gap)>=GAP_REFRESH_PERCENT:
     auto_refresh_tickers.append(name);gaps.append((name,round(gap,2)));print(f"⚠️ {name}: gap {gap:.2f}% -> {GAP_REFRESH_BARS} bars")
  df.loc[candle_date,COLUMNS]=[r[c] for c in COLUMNS];db[name]=normalize(df);updated.append(name)
 check_date=max(live_dates.values()) if live_dates else pd.Timestamp.now().normalize()
 if MANUAL_REFRESH_HISTORY:targets=SYMBOLS;bars=INITIAL_REFRESH_BARS
 elif auto_refresh_tickers:targets=auto_refresh_tickers;bars=GAP_REFRESH_BARS
 else:targets=[];bars=GAP_REFRESH_BARS
 hist_ok=[];hist_fail=[]
 if targets:
  if not HISTORICAL_API_AVAILABLE:hist_fail=targets.copy();print("❌ Historical API unavailable")
  else:
   print(f"🔄 Historical refresh: {len(targets)} symbols × {bars} bars")
   for name in targets:
    print(f"📥 {name}...");h=historical(name,bars)
    if h.empty:hist_fail.append(name);print(f"❌ {name}: refresh failed");continue
    df=db.get(name,pd.DataFrame(columns=COLUMNS)).copy()
    if df.empty:df=h
    else:df=pd.concat([df,h]);df=df[~df.index.duplicated(keep="last")].sort_index()
    df=normalize(df)
    if name in live_dates and live_dates[name] in df.index:
     r,_=bulk_row(bulk,name)
     if r:df.loc[live_dates[name],COLUMNS]=[r[c] for c in COLUMNS];df=normalize(df)
    db[name]=df;hist_ok.append(name);print(f"✅ {name}: {len(df)} bars");time.sleep(REQUEST_DELAY)
 save_database(db);quality(db,[x[0] for x in gaps],hist_ok,hist_fail,check_date)
 print(f"💾 Database saved: {DB_FILE}");print(f"📊 Updated: {len(updated)}/{len(SYMBOLS)}");print(f"📈 Gap targets: {len(gaps)}");print(f"🔄 Historical success: {len(hist_ok)} | failed: {len(hist_fail)}")
 if MANUAL_REFRESH_HISTORY:msg=f"🔵 *Manual Historical Refresh*\n📅 Date: {check_date.date()}\n🔄 Bars: {INITIAL_REFRESH_BARS}\n✅ Success: {len(hist_ok)}/{len(SYMBOLS)}"
 elif gaps:msg=f"⚠️ *Gap Refresh Triggered*\n📅 Date: {check_date.date()}\n🔍 Threshold: ±{GAP_REFRESH_PERCENT}%\n🔄 Bars: {GAP_REFRESH_BARS}\n🎯 Targets: {', '.join(x[0] for x in gaps)}\n🔵 Historical Success: {len(hist_ok)}"
 else:msg=f"✅ *EGX Price Database Updated*\n📅 Date: {check_date.date()}\n📊 Updated: {len(updated)}/{len(SYMBOLS)}"
 if MANUAL_REFRESH_HISTORY and hist_fail:msg+=f"\n❌ Failed: {', '.join(hist_fail)}"
 elif gaps and hist_fail:msg+=f"\n❌ Failed: {', '.join(hist_fail)}"
 send_telegram(msg);print("📨 Telegram notification sent.")
if __name__=="__main__":
 try:main()
 except Exception as e:
  print(f"🚨 FATAL ERROR: {e}");send_telegram(f"🚨 *EGX ENGINE FATAL ERROR*\n`{e}`")

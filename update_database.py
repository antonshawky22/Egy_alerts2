print("⚙️ EGX ENGINE v12.0 - TradingView Production Engine\n   Daily Live Update + Automatic 5% Gap Historical Refresh\n"+"="*70)

import json,os,pandas as pd,time,sys,requests,argparse
from tradingview_ta import Interval as TVInterval,get_multiple_analysis
try:
    from tradingviewApiPython import Client
    HISTORICAL_API_AVAILABLE=True
except:
    HISTORICAL_API_AVAILABLE=False

TELEGRAM_TOKEN=os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID")
DB_FILE="egx_history_database_v2.json"
REPORT_FILE="egx_history_database_quality_v2.json"
TEMP_DB_FILE=DB_FILE+".tmp"
REFRESH_BARS=200
GAP_REFRESH_PERCENT=5.0
MAX_RETRIES=3
RETRY_DELAY=5
REQUEST_DELAY=2

parser=argparse.ArgumentParser()
parser.add_argument("--refresh-history",action="store_true")
args=parser.parse_args()
env_refresh=os.getenv("REFRESH_HISTORY","").strip().lower()
MANUAL_REFRESH_HISTORY=args.refresh_history or env_refresh in ["1","true","yes","on"]

print("\n🔵 MODE: MANUAL FULL HISTORICAL REFRESH" if MANUAL_REFRESH_HISTORY else "\n🟢 MODE: NORMAL DAILY LIVE UPDATE")
if MANUAL_REFRESH_HISTORY: print(f"📈 Latest {REFRESH_BARS} daily candles will be synchronized for all symbols.")
else: print(f"📈 Latest TradingView candle will be updated.\n🔍 Automatic historical refresh: Gap >= {GAP_REFRESH_PERCENT:.1f}%.")

def send_telegram(message):
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN=="YOUR_BOT_TOKEN_HERE":
        print(f"📱 [Telegram Mock]: {message}"); return
    try:
        r=requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",json={"chat_id":TELEGRAM_CHAT_ID,"text":message,"parse_mode":"Markdown"},timeout=10)
        if not r.ok: print("⚠️ Telegram HTTP",r.status_code)
    except Exception as e: print(f"⚠️ Telegram error: {e}")

symbols={x:x for x in """EGX30 OLFI EMFD ETEL EAST EFIH ABUK OIH SWDY ISPH ATQA MTIE HRHO ORWE JUFO DSCW SUGR ELSH RMDA RAYA EEII MPCO GBCO TMGH ORHD AMOC FWRY COMI ADIB PHDC MCQE SKPC EGAL""".split()}

print("\n"+"="*70+"\n📂 LOADING DATABASE\n"+"="*70)
try:
    with open(DB_FILE,encoding="utf-8") as f: raw_database=json.load(f)
    print("💾 Existing V2 database loaded.")
    print(f"📊 Symbols in file: {len(raw_database)}")
except Exception as e:
    raw_database={}
    print("🆕 Database not found or invalid."); print(f"   Error: {e}")

database={}
cleaning_report={}
for name,content in raw_database.items():
    try:
        if isinstance(content,dict) and "columns" in content and "data" in content:
            df=pd.DataFrame.from_dict(content["data"],orient="index",columns=content["columns"])
            df.index=pd.to_datetime(df.index,errors="coerce")
            df=df[~df.index.isna()]
            df.index=df.index.strftime("%Y-%m-%d"); df.index.name="Date"
            for c in ["Open","High","Low","Close","Volume"]:
                if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
            before=len(df); df=df[~df.index.duplicated(keep="last")].sort_index()
            database[name]=df
            cleaning_report[name]={"duplicate_dates_removed_on_load":before-len(df),"bars_after_load":len(df)}
        else:
            database[name]=pd.DataFrame(); cleaning_report[name]={"status":"INVALID_FORMAT"}
    except Exception as e:
        print(f"⚠️ Failed loading {name}: {e}")
        database[name]=pd.DataFrame(); cleaning_report[name]={"status":"LOAD_ERROR","error":str(e)}

print("\n"+"="*70+"\n🚀 TRADINGVIEW BULK FETCH\n"+"="*70)
tv_symbols=[f"EGX:{x}" for x in symbols.values()]
bulk_analysis=None
for attempt in range(1,MAX_RETRIES+1):
    try:
        print(f"🔌 Bulk attempt {attempt}/{MAX_RETRIES}")
        bulk_analysis=get_multiple_analysis(screener="egypt",interval=TVInterval.INTERVAL_1_DAY,symbols=tv_symbols)
        if bulk_analysis:
            print("✅ Bulk data retrieved successfully from TradingView!"); break
    except Exception as e:
        print(f"⚠️ Attempt {attempt} failed: {e}")
        if attempt<MAX_RETRIES: time.sleep(RETRY_DELAY)

if not bulk_analysis:
    msg="💥 **خطأ حرج في النظام!**\nفشل سحب الأسعار الجماعي من TradingView.\nتم إيقاف البرنامج بدون تعديل قاعدة البيانات."
    print(msg); send_telegram(msg); sys.exit(1)

print("\n"+"="*70+"\n🚨 MARKET PULSE SENSOR\n"+"="*70)
market_active=False
check_date=None
for name in ["COMI","SWDY","HRHO"]:
    a=bulk_analysis.get(f"EGX:{name}")
    if not a: continue
    i=a.indicators
    vol=float(i.get("volume",0)); close=float(i.get("close",0)); op=float(i.get("open",0)); high=float(i.get("high",0)); low=float(i.get("low",0))
    check_date=str(a.time.date())
    df=database.get(name,pd.DataFrame())
    if not df.empty:
        try:
            old=df.iloc[-1]
            if vol>0 and (close!=old["Close"] or op!=old["Open"] or high!=old["High"] or low!=old["Low"]):
                market_active=True
                print(f"🟢 Market activity detected via {name}\n   Volume: {vol:,.0f}\n   Date: {check_date}")
                break
        except Exception as e: print(f"⚠️ Market pulse check failed for {name}: {e}")

if not market_active:
    print("😴 Market is Closed or No New Trading Activity Detected.")
    if not MANUAL_REFRESH_HISTORY:
        print("🛡️ Exiting safely without modifying database."); sys.exit(0)
    print("🔵 Manual historical refresh mode: continuing...")
else: print(f"🟢 Processing market updates for date: {check_date}...")

updated_count=0
has_real_price_changes=False
failed_tickers=[]
auto_refresh_tickers=[]
gap_report={}

print("\n"+"="*70+"\n📈 LIVE DAILY PRICE UPDATE\n"+"="*70)

for name,ticker in symbols.items():
    try:
        df=database.get(name,pd.DataFrame())
        a=bulk_analysis.get(f"EGX:{ticker}")
        if not a:
            print(f"⚠️ {name:6} -> No TradingView analysis returned.")
            failed_tickers.append(name); continue

        i=a.indicators
        date=str(a.time.date())
        close=float(i.get("close",0))
        op=float(i.get("open",close))
        high=float(i.get("high",close))
        low=float(i.get("low",close))
        volume=float(i.get("volume",0))

        if min(op,high,low,close)<=0 or high<max(op,close,low) or low>min(op,close):
            raise Exception("Invalid OHLC values/relationship")

        gap=None
        previous_close=None
        if not df.empty:
            try:
                idx=pd.to_datetime(df.index,errors="coerce")
                temp=df.loc[~idx.isna()].copy()
                temp.index=idx[~idx.isna()]
                dates=temp.index[temp.index<pd.to_datetime(date)]
                if len(dates):
                    previous_close=float(temp.loc[dates.max(),"Close"])
                    if previous_close>0:
                        gap=(op-previous_close)/previous_close*100
                        triggered=abs(gap)>=GAP_REFRESH_PERCENT
                        gap_report[name]={"date":date,"previous_date":str(dates.max().date()),"previous_close":round(previous_close,2),"today_open":round(op,2),"gap_percent":round(gap,2),"threshold_percent":GAP_REFRESH_PERCENT,"historical_refresh_triggered":triggered}
                        if triggered:
                            auto_refresh_tickers.append(name)
                            print(f"⚠️ {name:6} -> GAP {gap:+.2f}% | Previous close: {previous_close:.2f} | Open: {op:.2f} | 🔄 {REFRESH_BARS}-bar refresh")
                        else: print(f"📊 {name:6} -> Gap: {gap:+.2f}% → Normal.")
            except Exception as e: print(f"⚠️ Gap calculation failed for {name}: {e}")

        if df.empty or date not in df.index: has_real_price_changes=True
        else:
            old=df.loc[date]
            if any(old[c]!=v for c,v in zip(["Open","High","Low","Close","Volume"],[op,high,low,close,volume])): has_real_price_changes=True

        df.loc[date,["Open","High","Low","Close","Volume"]]=[op,high,low,close,volume]
        database[name]=df[~df.index.duplicated(keep="last")].sort_index().round(2)
        updated_count+=1
        print(f"✅ {name:6} -> {date}. Volume: {volume:,.0f}.")
    except Exception as e:
        print(f"💥 Failed to update {name}: {e}")
        failed_tickers.append(name)

auto_refresh_tickers=list(dict.fromkeys(auto_refresh_tickers))

refresh_targets=list(symbols) if MANUAL_REFRESH_HISTORY else auto_refresh_tickers
historical_success=[]
historical_failed=[]
historical_changed=[]

if refresh_targets:
    print("\n"+"="*70)
    print(f"🔵 {'MANUAL FULL REFRESH' if MANUAL_REFRESH_HISTORY else 'AUTOMATIC GAP-TRIGGERED REFRESH'} - LAST {REFRESH_BARS} BARS")
    print("="*70+f"\n📊 Targets: {len(refresh_targets)}\n📋 {', '.join(refresh_targets)}")

    if not HISTORICAL_API_AVAILABLE:
        print("💥 Historical TradingView API package is not available.\n⚠️ Historical refresh skipped.")
        historical_failed=refresh_targets.copy()
    else:
        for number,name in enumerate(refresh_targets,1):
            print(f"\n🔄 Historical [{number}/{len(refresh_targets)}] {name}")
            success=False
            for attempt in range(1,MAX_RETRIES+1):
                try:
                    print(f"🔌 Historical attempt {attempt}/{MAX_RETRIES}")
                    client=Client(); chart=client.Session.Chart()
                    chart.set_market(f"EGX:{symbols[name]}",{"timeframe":"D","range":REFRESH_BARS})
                    time.sleep(5)
                    periods=chart.periods
                    if not periods: raise Exception("TradingView returned no historical bars")

                    rows=[]; bad_bars=0
                    for bar in periods:
                        try:
                            row={"Date":pd.to_datetime(float(bar["time"]),unit="s").strftime("%Y-%m-%d"),"Open":float(bar["open"]),"High":float(bar["max"]),"Low":float(bar["min"]),"Close":float(bar["close"]),"Volume":float(bar.get("volume",0))}
                            if min(row["Open"],row["High"],row["Low"],row["Close"])<=0 or row["High"]<max(row["Open"],row["Close"],row["Low"]) or row["Low"]>min(row["Open"],row["Close"]): bad_bars+=1
                            else: rows.append(row)
                        except: bad_bars+=1

                    refresh_df=pd.DataFrame(rows)
                    if refresh_df.empty: raise Exception("No valid historical bars after parsing")
                    refresh_df=refresh_df.sort_values("Date").drop_duplicates("Date",keep="last").reset_index(drop=True)
                    date_series=pd.to_datetime(refresh_df["Date"])
                    gaps=date_series.diff().dt.days
                    suspicious=refresh_df.loc[gaps>5,"Date"].astype(str).tolist()
                    max_gap=int(gaps.max()) if not gaps.dropna().empty else 0
                    old_df=database.get(name,pd.DataFrame()).copy()

                    if old_df.empty:
                        merged_df=refresh_df.set_index("Date"); changed_rows=len(merged_df)
                    else:
                        old_df.index=pd.to_datetime(old_df.index,errors="coerce")
                        old_df=old_df[~old_df.index.isna()]
                        old_df.index=old_df.index.strftime("%Y-%m-%d")
                        new_df=refresh_df.set_index("Date")
                        changed_rows=0
                        for d in new_df.index:
                            if d not in old_df.index: changed_rows+=1
                            else:
                                if any(float(old_df.loc[d,c])!=float(new_df.loc[d,c]) for c in ["Open","High","Low","Close","Volume"]): changed_rows+=1
                            old_df.loc[d,["Open","High","Low","Close","Volume"]]=new_df.loc[d,["Open","High","Low","Close","Volume"]]
                        merged_df=old_df

                    merged_df=merged_df[["Open","High","Low","Close","Volume"]]
                    database[name]=merged_df[~merged_df.index.duplicated(keep="last")].sort_index().round(2)
                    historical_success.append(name)
                    if changed_rows>0: historical_changed.append(name)
                    print(f"✅ SUCCESS | Received: {len(periods)} | Valid: {len(refresh_df)} | {refresh_df['Date'].iloc[0]} → {refresh_df['Date'].iloc[-1]} | Changed: {changed_rows} | Bad: {bad_bars} | Max gap: {max_gap}")
                    if suspicious: print("   ⚠️ Suspicious gaps:",", ".join(suspicious))
                    else: print("   ✅ No suspicious large gaps.")
                    success=True; break
                except Exception as e:
                    print(f"⚠️ Historical attempt {attempt} failed: {e}")
                    if attempt<MAX_RETRIES: time.sleep(RETRY_DELAY)
            if not success:
                historical_failed.append(name)
                print(f"❌ HISTORICAL REFRESH FAILED FOR {name} | 🛡️ Existing data preserved.")
            if number<len(refresh_targets): time.sleep(REQUEST_DELAY)
else:
    print("\n"+"="*70+f"\nℹ️ NO HISTORICAL REFRESH REQUIRED\n"+"="*70+f"\n✅ No stock had a Gap of {GAP_REFRESH_PERCENT:.1f}% or more.")

print("\n"+"="*70+"\n🔍 FINAL DATABASE QUALITY CHECK\n"+"="*70)
final_quality_report={}
total_bad_ohlc=total_duplicate_dates=total_suspicious_gaps=0

for name in symbols:
    df=database.get(name,pd.DataFrame())
    if df.empty:
        final_quality_report[name]={"status":"EMPTY","bars":0}; continue

    before=len(df)
    df=df[~df.index.duplicated(keep="last")].sort_index()
    duplicate_count=before-len(df)
    total_duplicate_dates+=duplicate_count

    bad=df[(df["Open"]<=0)|(df["High"]<=0)|(df["Low"]<=0)|(df["Close"]<=0)|(df["High"]<df["Open"])|(df["High"]<df["Close"])|(df["High"]<df["Low"])|(df["Low"]>df["Open"])|(df["Low"]>df["Close"])]
    bad_count=len(bad); total_bad_ohlc+=bad_count
    moves=df["Close"].pct_change()*100
    large_moves=moves[moves.abs()>=10]
    dates=pd.to_datetime(df.index,errors="coerce")
    diffs=dates.to_series().diff().dt.days
    suspicious=diffs[diffs>5].dropna()
    total_suspicious_gaps+=len(suspicious)
    database[name]=df.round(2)

    final_quality_report[name]={
        "status":"OK","bars":len(df),"start_date":str(df.index[0]),"end_date":str(df.index[-1]),
        "bad_ohlc_rows":bad_count,"duplicate_dates_removed":duplicate_count,
        "large_moves_10_percent_or_more":len(large_moves),
        "suspicious_gaps_over_5_days":len(suspicious),
        "automatic_gap_percent":GAP_REFRESH_PERCENT,
        "automatic_gap_refresh_triggered":name in auto_refresh_tickers,
        "historical_refresh_success":name in historical_success,
        "historical_refresh_failed":name in historical_failed,
        "historical_data_changed":name in historical_changed
    }

print("\n"+"="*70+"\n💾 PREPARING FINAL DATABASE\n"+"="*70)
required=["Open","High","Low","Close","Volume"]
final_blocks=[]

for name in symbols:
    df=database.get(name,pd.DataFrame())
    if df.empty:
        print(f"⚠️ {name:6} -> No data. Skipped."); continue
    df=df.sort_index(ascending=False)
    missing=[c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️ {name:6} -> Missing columns: {missing}. Skipped."); continue
    final_blocks.append(f'  "{name}": {{\n    "columns": {json.dumps(required)},\n    "data": {json.dumps(df[required].to_dict(orient="index"),ensure_ascii=False)}\n  }}')

if not final_blocks:
    print("\n💥 CRITICAL: Final database is empty.\n🛡️ Original database will NOT be modified.")
    send_telegram("⚠️ **تحذير خطير:**\nمحاولة حفظ قاعدة بيانات فارغة.\nتم إيقاف الحفظ لحماية قاعدة البيانات القديمة.")
    sys.exit(1)

print("\n"+"="*70+"\n🛡️ SAFE ATOMIC SAVE\n"+"="*70)
try:
    with open(TEMP_DB_FILE,"w",encoding="utf-8") as f:
        f.write("{\n"+",\n".join(final_blocks)+"\n}"); f.flush(); os.fsync(f.fileno())
    with open(TEMP_DB_FILE,encoding="utf-8") as f: json.load(f)
    os.replace(TEMP_DB_FILE,DB_FILE)
    print(f"✅ Database safely saved: {DB_FILE}")
except Exception as e:
    print(f"💥 DATABASE SAVE FAILED: {e}")
    try:
        if os.path.exists(TEMP_DB_FILE): os.remove(TEMP_DB_FILE)
    except: pass
    send_telegram("⚠️ **خطأ في حفظ قاعدة البيانات!**\nتم الحفاظ على الملف الأصلي.")
    sys.exit(1)

report={
    "engine_version":"v12.0","data_source":"TradingView","database_file":DB_FILE,
    "manual_full_refresh_enabled":MANUAL_REFRESH_HISTORY,
    "automatic_gap_refresh_enabled":True,
    "automatic_gap_refresh_percent":GAP_REFRESH_PERCENT,
    "refresh_bars":REFRESH_BARS,"symbols_total":len(symbols),
    "live_updated":updated_count,"live_failed":len(failed_tickers),
    "automatic_gap_refresh_triggered":len(auto_refresh_tickers),
    "automatic_gap_refresh_symbols":auto_refresh_tickers,
    "historical_refresh_targets":refresh_targets,
    "historical_success":len(historical_success),
    "historical_failed":len(historical_failed),
    "historical_changed":len(historical_changed),
    "total_bad_ohlc":total_bad_ohlc,
    "total_duplicate_dates":total_duplicate_dates,
    "total_suspicious_gaps":total_suspicious_gaps,
    "failed_live_symbols":failed_tickers,
    "failed_historical_symbols":historical_failed,
    "historical_changed_symbols":historical_changed,
    "gap_report":gap_report,"symbols":final_quality_report
}

try:
    with open(REPORT_FILE,"w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"📄 Quality report saved: {REPORT_FILE}")
except Exception as e: print(f"⚠️ Failed to save quality report: {e}")

print("\n"+"="*70+"\n🏁 EGX DATABASE ENGINE FINISHED\n"+"="*70)
print(f"\n📊 Total symbols       : {len(symbols)}")
print(f"📈 Live updated        : {updated_count}")
print(f"⚠️ Live failed         : {len(failed_tickers)}")
print(f"\n🔍 Gap threshold       : {GAP_REFRESH_PERCENT:.1f}%")
print(f"🔄 Auto refresh targets: {len(auto_refresh_tickers)}")
if auto_refresh_tickers: print("   "+", ".join(auto_refresh_tickers))
if MANUAL_REFRESH_HISTORY or auto_refresh_tickers:
    print(f"\n🔵 Historical success  : {len(historical_success)}")
    print(f"🔴 Historical failed   : {len(historical_failed)}")
    print(f"🔄 Historical changed  : {len(historical_changed)}")
print(f"\n❗ Bad OHLC rows       : {total_bad_ohlc}")
print(f"❗ Duplicate dates     : {total_duplicate_dates}")
print(f"⚠️ Suspicious gaps     : {total_suspicious_gaps}")
if failed_tickers: print("\n❌ LIVE FAILED SYMBOLS:\n"+", ".join(failed_tickers))
if historical_failed: print("\n❌ HISTORICAL FAILED SYMBOLS:\n"+", ".join(historical_failed))

if auto_refresh_tickers:
    print("\n"+"="*70+"\n⚠️ AUTOMATIC GAP REFRESH EVENTS\n"+"="*70)
    for name in auto_refresh_tickers:
        x=gap_report.get(name,{})
        print(f"{name:<8} Gap: {x.get('gap_percent','N/A')}% | Previous close: {x.get('previous_close','N/A')} | Open: {x.get('today_open','N/A')}")

print("\n"+"="*70+"\n📊 DATABASE OVERVIEW\n"+"="*70")
for name in symbols:
    df=database.get(name,pd.DataFrame())
    print(f"{name:<8} {len(df):>5} bars   {df.index[0]} → {df.index[-1]}" if not df.empty else f"{name:<8} ❌ NOT AVAILABLE")

if updated_count>0 and has_real_price_changes:
    message=f"✅ *EGX Price Database Updated*\n📅 Date: {check_date}\n📊 Updated: {updated_count}/{len(symbols)}"
    if auto_refresh_tickers:
        message+=f"\n⚠️ *Gap Refresh Triggered*\n🔍 Threshold: {GAP_REFRESH_PERCENT:.1f}%\n🔄 Targets: {', '.join(auto_refresh_tickers)}\n🔵 Historical Success: {len(historical_success)}"
        if historical_failed: message+=f"\n⚠️ Historical Failed: {len(historical_failed)}"
    elif MANUAL_REFRESH_HISTORY:
        message+=f"\n🔵 *Manual Historical Refresh*\n🔄 Success: {len(historical_success)}/{len(symbols)}"
        if historical_failed: message+=f"\n⚠️ Failed: {len(historical_failed)}"
    send_telegram(message)
else:
    print("\nℹ️ No new price changes detected.\n📱 Telegram notification skipped to avoid noise.")

print("\n"+"="*70+"\n🎯 FINAL STATUS\n"+"="*70)
print("✅ TradingView only | ✅ No Yahoo | ✅ No manual adjustment | ✅ No artificial correction")
print(f"✅ Automatic Gap detection: {GAP_REFRESH_PERCENT:.1f}%")
print(f"✅ Gap >= {GAP_REFRESH_PERCENT:.1f}% triggers {REFRESH_BARS}-bar historical refresh.")
print("✅ Only affected symbols are automatically refreshed.")
print("✅ Normal price movements update daily candle only.")
print("✅ Duplicate dates removed | ✅ OHLC checked | ✅ Atomic save")
print("✅ Older historical data preserved.")
print("✅ Failed historical refreshes do not overwrite old data.")
print("\n🏁 Production database update completed.")

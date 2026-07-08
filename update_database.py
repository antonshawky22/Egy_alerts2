print("⚙️ EGX ENGINE v9.8 - Batching Strategy & Ultimate Anti-429 Edition")

import yfinance as yf
from tradingview_ta import TA_Handler, Interval as TVInterval
import json
import os
import pandas as pd
import time
import sys

DB_FILE = "egx_history_database_v2.json"

symbols = {
    "EGX30": "EGX30",
    "OLFI": "OLFI", "EMFD": "EMFD", "ETEL": "ETEL", "EAST": "EAST",
    "EFIH": "EFIH", "ABUK": "ABUK", "OIH": "OIH", "SWDY": "SWDY", "ISPH": "ISPH",
    "ATQA": "ATQA", "MTIE": "MTIE", "HRHO": "HRHO", "ORWE": "ORWE",
    "JUFO": "JUFO", "DSCW": "DSCW", "SUGR": "SUGR", "ELSH": "ELSH", "RMDA": "RMDA",
    "RAYA": "RAYA", "EEII": "EEII", "MPCO": "MPCO", "GBCO": "GBCO", "TMGH": "TMGH",
    "ORHD": "ORHD", "AMOC": "AMOC", "FWRY": "FWRY", "COMI": "COMI", "ADIB": "ADIB",
    "PHDC": "PHDC", "MCQE": "MCQE", "SKPC": "SKPC", "EGAL": "EGAL"
}

# 1. تحميل وقراءة البيانات مع التنظيف الفوري
try:
    with open(DB_FILE, "r") as f:
        raw_database = json.load(f)
    print("💾 Existing V2 database loaded. Starting Auto-Clean check...")
except:
    raw_database = {}
    print("🆕 Database not found. Creating a fresh stable database...")

database = {}
for name, content in raw_database.items():
    try:
        if isinstance(content, dict) and "columns" in content and "data" in content:
            df_temp = pd.DataFrame.from_dict(content["data"], orient="index", columns=content["columns"])
            df_temp.index = pd.to_datetime(df_temp.index).strftime('%Y-%m-%d')
            df_temp.index.name = "Date"
            df_temp = df_temp.sort_index(ascending=True) # الترتيب التصاعدي الصحيح للحسابات الفنية
            df_temp = df_temp[~df_temp.index.duplicated(keep='last')]
            database[name] = df_temp
        else:
            database[name] = pd.DataFrame()
    except:
        database[name] = pd.DataFrame()

# ==========================================
# 🚨 2. رادار فحص نبض السوق (Market Pulse Sensor)
# ==========================================
print("\n🔍 Scanning market pulse using leaders (COMI, FWRY, TMGH, EGX30)...")
market_is_open = False
leaders_to_check = ["COMI", "FWRY", "TMGH", "EGX30"]
check_date = None

for leader in leaders_to_check:
    try:
        handler = TA_Handler(symbol=symbols[leader], screener="egypt", exchange="EGX", interval=TVInterval.INTERVAL_1_DAY)
        
        try:
            analysis = handler.get_analysis()
        except Exception as pulse_err:
            if "429" in str(pulse_err):
                print(f"⚠️ Pulse hit 429 for {leader}. Pre-cooldown 15s...")
                time.sleep(15)
                analysis = handler.get_analysis()
            else:
                raise pulse_err

        indicators = analysis.indicators
        check_date = str(analysis.time.date())
        
        new_close = float(indicators.get("close", 0))
        new_open = float(indicators.get("open", new_close))
        new_high = float(indicators.get("high", new_close))
        new_low = float(indicators.get("low", new_close))
        
        # قراءة آخر إغلاق مسجل في الـ JSON لهذا السهم القيادي
        df_leader = database.get(leader, pd.DataFrame())
        if not df_leader.empty and check_date in df_leader.index:
            old_data = df_leader.loc[check_date].values # مقارنة بنفس اليوم لو الكود اشتغل تاني أثناء الجلسة
            if new_close != old_data[3] or new_open != old_data[0]:
                market_is_open = True
                break
        elif not df_leader.empty:
            last_recorded_date = df_leader.index[-1]
            old_data = df_leader.iloc[-1].values # مقارنة بآخر شمعة مسجلة تاريخياً
            # لو الأسعار تغيرت ولو بمليم واحد عن آخر إغلاق، إذن السوق شغال وضخ أسعار جديدة
            if (new_close != old_data[3] or new_open != old_data[0] or 
                new_high != old_data[1] or new_low != old_data[2]):
                market_is_open = True
                break
        else:
            market_is_open = True
            break
    except Exception as e:
        print(f"⚠️ Pulse check skipped for {leader}: {e}")

# قرار الفرملة الحاسم: لو الأسعار متطابقة بالملي ومفيش أي نبض تغيير
if not market_is_open and check_date is not None:
    print(f"🛑 Market is CLOSED/HOLIDAY for {check_date} (No price movement detected). Script stopped to prevent duplication.")
    sys.exit(0) # الخروج الآمن الفوري دون لمس الـ JSON
else:
    print("🟢 Market is OPEN and active. Processing updates...")

# ==========================================
# 3. تحديث البيانات وضخ الأسعار الجديدة (نظام المجموعات الذكي)
# ==========================================
symbols_items = list(symbols.items())
total_symbols = len(symbols_items)
batch_size = 8  # تحديث كل 8 أسهم كحد أقصى ثم تبريد السيرفر

for index, (name, ticker) in enumerate(symbols_items):
    try:
        # ترييحة قصيرة بين الأسهم داخل نفس المجموعة
        time.sleep(2) 
        df = database.get(name, pd.DataFrame())

        if df.empty or len(df) < 20:
            print(f"📥 Downloading base history for {name} from Yahoo...")
            yf_ticker = "^CASE30" if name == "EGX30" else f"{ticker}.CA"
            yf_df = yf.download(yf_ticker, period="1mo", interval="1d", auto_adjust=False, progress=False)
            
            if not yf_df.empty:
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                df = yf_df.dropna(subset=["Close"])[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
                df.index.name = "Date"

        # جلب شمعة اليوم من TradingView مع صمام الحماية الديناميكي
        handler = TA_Handler(symbol=ticker, screener="egypt", exchange="EGX", interval=TVInterval.INTERVAL_1_DAY)
        
        try:
            analysis = handler.get_analysis()
        except Exception as tv_err:
            if "429" in str(tv_err):
                print(f"⚠️ Rate limit hit (429) for {name}. Sleeping for 20 seconds to cooldown...")
                time.sleep(20)
                analysis = handler.get_analysis()
            else:
                raise tv_err
                
        tv_indicators = analysis.indicators
        last_candle_date = str(analysis.time.date())
        
        tv_close = float(tv_indicators.get("close"))
        tv_open = float(tv_indicators.get("open", tv_close))
        tv_high = float(tv_indicators.get("high", tv_close))
        tv_low = float(tv_indicators.get("low", tv_close))
        
        # جلب فوليوم EGX30 الحقيقي سحب مباشر من التحديث اللحظي
        tv_volume = float(tv_indicators.get("volume", 0))

        # تحديث أو إضافة اليوم الحالي
        df.loc[last_candle_date] = [tv_open, tv_high, tv_low, tv_close, tv_volume]
        
        # التأكيد النهائي لمسح أي تكرار مع الحفاظ التام على الترتيب التصاعدي للحسابات الفنية
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index(ascending=True).round(2)
        database[name] = df
        print(f"✅ {name:6} -> Cleanly updated for {last_candle_date}. Volume: {tv_volume:,.0f}. Total days: {len(df)}")

        # 🔥 كمين تبريد السيرفر (Batch Cooldown)
        # لو خلصنا 8 أسهم ولسه مصلناش لآخر سهم في اللستة، ريح دقيقة كاملة لتصفير عداد تريدنج فيو
        if (index + 1) % batch_size == 0 and (index + 1) < total_symbols:
            print(f"☕ Batch of {batch_size} completed ({index + 1}/{total_symbols}). Sleeping for 60 seconds to completely reset TradingView rate limit...")
            time.sleep(60)

    except Exception as e:
        print(f"💥 Failed to update {name}: {e}")

# ==========================================
# 4. حفظ قاعدة البيانات بالترتيب المريح للموبايل (الأحدث فوق)
# ==========================================
final_blocks = []
for name, df in database.items():
    if not df.empty:
        df_for_mobile = df.sort_index(ascending=False)
        columns_line = json.dumps(list(df_for_mobile.columns))
        data_line = json.dumps(df_for_mobile.to_dict(orient="index"))
        
        block = f'  "{name}": {{\n    "columns": {columns_line},\n    "data": {data_line}\n  }}'
        final_blocks.append(block)

with open(DB_FILE, "w") as f:
    f.write("{\n" + ",\n".join(final_blocks) + "\n}")

print(f"\n🏁 Complete! File '{DB_FILE}' repaired, optimized, and saved successfully.")

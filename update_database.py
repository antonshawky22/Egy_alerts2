print("⚙️ EGX ENGINE v9.5 - GitHub Auto-Clean & Repair Production Engine")

import yfinance as yf
from tradingview_ta import TA_Handler, Interval as TVInterval
import json
import os
import pandas as pd
import time

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

# 1. تحميل وقراءة البيانات مع ميزة التنظيف الفوري من المكرر القديم
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
            
            # تنظيف صيغة التاريخ وإزالة أي مسافات أو ساعات مسببة للتكرار
            df_temp.index = pd.to_datetime(df_temp.index).strftime('%Y-%m-%d')
            df_temp.index.name = "Date"
            
            # ترتيب تصاعدي صارم عشان الحسابات الفنية والمؤشرات
            df_temp = df_temp.sort_index(ascending=True)
            
            # 🔥 سحق وحذف أي سطور مكررة قديمة مبهدلة الملف فوراً والاحتفاظ بآخر سعر صحيح
            df_temp = df_temp[~df_temp.index.duplicated(keep='last')]
            database[name] = df_temp
        else:
            database[name] = pd.DataFrame()
    except:
        database[name] = pd.DataFrame()

# 2. تحديث البيانات وضخ شمعة اليوم من TradingView (بما فيهم EGX30)
for name, ticker in symbols.items():
    try:
        time.sleep(1.5) # ترييحة أمان لمنع حظر السيرفر على جيت هب
        df = database.get(name, pd.DataFrame())

        # بناء الأساس التاريخي لو الملف فاضي خالص أو متدمر
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

        # جلب شمعة اليوم الحية من TradingView لـ كـل الأصول (المؤشر + الأسهم)
        handler = TA_Handler(symbol=ticker, screener="egypt", exchange="EGX", interval=TVInterval.INTERVAL_1_DAY)
        analysis = handler.get_analysis()
        tv_indicators = analysis.indicators
        
        last_candle_date = str(analysis.time.date())
        
        tv_close = float(tv_indicators.get("close"))
        tv_open = float(tv_indicators.get("open", tv_close))
        tv_high = float(tv_indicators.get("high", tv_close))
        tv_low = float(tv_indicators.get("low", tv_close))
        tv_volume = 0.0 if name == "EGX30" else float(tv_indicators.get("volume", 0))

        # الفحص الصارم ضد التكرار: تحديث اليوم الحالي لو موجود أو إضافته لو يوم جديد
        df.loc[last_candle_date] = [tv_open, tv_high, tv_low, tv_close, tv_volume]
        
        # التأكيد النهائي لمسح أي تكرار قبل الحفظ
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index(ascending=True).round(2)
        database[name] = df
        print(f"✅ {name:6} -> Cleanly updated for {last_candle_date}. Total days: {len(df)}")

    except Exception as e:
        print(f"💥 Failed to update {name}: {e}")

# 3. حفظ قاعدة البيانات بالترتيب المريح للموبايل (الأحدث فوق)
final_blocks = []
for name, df in database.items():
    if not df.empty:
        # قلب الترتيب للأحدث فوق فقط وقت الحفظ النهائي لراحة عينك في ملف الـ JSON
        df_for_mobile = df.sort_index(ascending=False)
        columns_line = json.dumps(list(df_for_mobile.columns))
        data_line = json.dumps(df_for_mobile.to_dict(orient="index"))
        
        block = f'  "{name}": {{\n    "columns": {columns_line},\n    "data": {data_line}\n  }}'
        final_blocks.append(block)

with open(DB_FILE, "w") as f:
    f.write("{\n" + ",\n".join(final_blocks) + "\n}")

print(f"\n🏁 Complete! File '{DB_FILE}' repaired, optimized, and saved successfully.")

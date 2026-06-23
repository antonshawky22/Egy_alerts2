print("EGX DATABASE - Advanced Compact Sourcing Engine")

import yfinance as yf
from tradingview_ta import TA_Handler, Interval as TVInterval
import json
import os
import pandas as pd

# =====================
# Symbols
# =====================
symbols = {
    "OFH": "OFH", "OLFI": "OLFI", "EMFD": "EMFD", "ETEL": "ETEL", "EAST": "EAST",
    "EFIH": "EFIH", "ABUK": "ABUK", "OIH": "OIH", "SWDY": "SWDY", "ISPH": "ISPH",
    "ATQA": "ATQA", "MTIE": "MTIE", "ELEC": "ELEC", "HRHO": "HRHO", "ORWE": "ORWE",
    "JUFO": "JUFO", "DSCW": "DSCW", "SUGR": "SUGR", "ELSH": "ELSH", "RMDA": "RMDA",
    "RAYA": "RAYA", "EEII": "EEII", "MPCO": "MPCO", "GBCO": "GBCO", "TMGH": "TMGH",
    "ORHD": "ORHD", "AMOC": "AMOC", "FWRY": "FWRY", "COMI": "COMI", "ADIB": "ADIB",
    "PHDC": "PHDC", "MCQE": "MCQE", "SKPC": "SKPC", "EGAL": "EGAL",
    "EGX30": "EGX30"  # 🟢 تم إضافة المؤشر بنجاح
}

DB_FILE = "egx_history_database.json"

# تحميل قاعدة البيانات بالهيكل الجديد
try:
    with open(DB_FILE, "r") as f:
        raw_database = json.load(f)
    print("💾 Existing compact database loaded successfully.")
except:
    raw_database = {}
    print("🆕 No database found. Initializing a new one...")

database = {}
# تحويل الهيكل الجديد لـ DataFrames للتعامل معها داخل الكود
for name, content in raw_database.items():
    if "columns" in content and "data" in content:
        df_temp = pd.DataFrame.from_dict(content["data"], orient="index", columns=content["columns"])
        df_temp.index.name = "Date"
        database[name] = df_temp
    else:
        database[name] = pd.DataFrame()

# تحديث الأسعار
for name, ticker in symbols.items():
    try:
        df = database.get(name, pd.DataFrame())

        # إذا كانت الداتا فارغة، اسحب من ياهو لأول مرة
        if df.empty or len(df) < 100:
            print(f"📥 Downloading historical baseline for {name} from Yahoo...")
            
            # 🟢 تعديل جراحي: توجيه البوت لرمز ياهو الصحيح للمؤشر العام
            yf_ticker = "^CASE30" if name == "EGX30" else f"{ticker}.CA"
            
            yf_df = yf.download(yf_ticker, period="7mo", interval="1d", auto_adjust=False, progress=False, timeout=15)
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df = yf_df.dropna(subset=["Close"])
            
            df = yf_df[["Open", "High", "Low", "Close"]].copy()
            df.index = df.index.astype(str)

        # جلب شمعة اليوم من TradingView
        handler = TA_Handler(symbol=ticker, screener="egypt", exchange="EGX", interval=TVInterval.INTERVAL_1_DAY)
        analysis = handler.get_analysis()
        tv_indicators = analysis.indicators
        
        last_candle_date = str(analysis.time.date())
        
        tv_close = float(tv_indicators.get("close"))
        tv_open = float(tv_indicators.get("open", tv_close))
        tv_high = float(tv_indicators.get("high", tv_close))
        tv_low = float(tv_indicators.get("low", tv_close))
        
        # إضافة أو تحديث السطر
        df.loc[last_candle_date] = [tv_open, tv_high, tv_low, tv_close]
        
        # تقريب كل الأسعار لرقمتين عشريتين فوراً لمنع الكسور الطويلة
        df = df.round(2)
        
        database[name] = df
        print(f"✅ {name} updated for date {last_candle_date}. Total days: {len(df)}")

    except Exception as e:
        print(f"💥 Failed to update {name}: {e}")

# حفظ قاعدة البيانات بالهيكل المضغوط الاحترافي الجديد
final_json = {}
for name, df in database.items():
    if not df.empty:
        # ترتيب التواريخ من الأقدم للأحدث قبل الحفظ
        df = df.sort_index()
        final_json[name] = {
            "columns": list(df.columns),
            "data": df.to_dict(orient="index")
        }

with open(DB_FILE, "w") as f:
    json.dump(final_json, f, indent=2)

print(f"🏁 Compact Database update complete. File '{DB_FILE}' optimized.")

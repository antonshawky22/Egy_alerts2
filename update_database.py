print("⚙️ EGX DATABASE - GitHub Production Sourcing Engine (With Volume v2)")

import yfinance as yf
from tradingview_ta import TA_Handler, Interval as TVInterval
import json
import os
import pandas as pd

# =====================
# Symbols
# =====================
symbols = {
    "EGX30": "EGX30",  # المؤشر الرئيسي
    "OLFI": "OLFI", "EMFD": "EMFD", "ETEL": "ETEL", "EAST": "EAST",
    "EFIH": "EFIH", "ABUK": "ABUK", "OIH": "OIH", "SWDY": "SWDY", "ISPH": "ISPH",
    "ATQA": "ATQA", "MTIE": "MTIE", "HRHO": "HRHO", "ORWE": "ORWE",
    "JUFO": "JUFO", "DSCW": "DSCW", "SUGR": "SUGR", "ELSH": "ELSH", "RMDA": "RMDA",
    "RAYA": "RAYA", "EEII": "EEII", "MPCO": "MPCO", "GBCO": "GBCO", "TMGH": "TMGH",
    "ORHD": "ORHD", "AMOC": "AMOC", "FWRY": "FWRY", "COMI": "COMI", "ADIB": "ADIB",
    "PHDC": "PHDC", "MCQE": "MCQE", "SKPC": "SKPC", "EGAL": "EGAL"
}

# 🟢 اسم الملف الجديد الخماسي
DB_FILE = "egx_history_database_v2.json"

# تحميل قاعدة البيانات بالهيكل الجديد المنسق
try:
    with open(DB_FILE, "r") as f:
        raw_database = json.load(f)
    print("💾 Existing V2 compact database loaded successfully.")
except:
    raw_database = {}
    print("🆕 No V2 database found. Initializing a clean 5-column database...")

database = {}
for name, content in raw_database.items():
    try:
        if isinstance(content, dict) and "columns" in content and "data" in content:
            df_temp = pd.DataFrame.from_dict(content["data"], orient="index", columns=content["columns"])
            df_temp.index.name = "Date"
            database[name] = df_temp
        else:
            database[name] = pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Re-initializing structural breakdown for {name} due to format mismatch.")
        database[name] = pd.DataFrame()

# تحديث الأسعار وضخ البيانات
for name, ticker in symbols.items():
    try:
        df = database.get(name, pd.DataFrame())

        # إذا كانت الداتا فارغة أو تحتاج بناء الأساس التاريخي
        if df.empty or len(df) < 100:
            if name == "EGX30":
                print(f"📥 Downloading Hourly historical baseline for {name} from Yahoo (Auto-Handling)...")
                
                # 🟢 تم إزالة معامل session وترك المكتبة تدير الاتصال داخلياً لحل خطأ curl_cffi
                yf_hourly = yf.download("^CASE30", period="6mo", interval="1h", auto_adjust=False, progress=False, timeout=20)
                
                if not yf_hourly.empty:
                    if isinstance(yf_hourly.columns, pd.MultiIndex):
                        yf_hourly.columns = yf_hourly.columns.get_level_values(0)
                    yf_hourly.columns = [str(col).strip() for col in yf_hourly.columns]
                    
                    yf_hourly.index = pd.to_datetime(yf_hourly.index).tz_localize(None)
                    
                    # تجميع الساعات ليومي مع تجميع الفوليوم
                    df_daily = yf_hourly.resample('D').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                    
                    df = df_daily[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.index = df.index.astype(str).str[:10]
                else:
                    print("❌ Yahoo hourly returned empty dataframe.")
            else:
                # آلية السحب اليومية الطبيعية للأسهم العادية (تم إزالة السيسشن المكررة)
                print(f"📥 Downloading historical baseline for {name} from Yahoo (Auto-Handling)...")
                yf_df = yf.download(f"{ticker}.CA", period="7mo", interval="1d", auto_adjust=False, progress=False, timeout=15)
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                yf_df = yf_df.dropna(subset=["Close"])
                
                df = yf_df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = df.index.astype(str).str[:10]

        # 🟢 الالتفاف الذكي للمؤشر: يكتفي بداتا ياهو بالفوليوم والأسعار ويمر لمنع تعنت TradingView مع المؤشرات
        if name == "EGX30":
            df = df.round(2)
            database[name] = df
            print(f"✅ {name} stabilized with Volume via Yahoo. Total days: {len(df)}")
            continue

        # جلب شمعة اليوم المباشرة والمحدثة من TradingView (للأسهم فقط)
        handler = TA_Handler(symbol=ticker, screener="egypt", exchange="EGX", interval=TVInterval.INTERVAL_1_DAY)
        analysis = handler.get_analysis()
        tv_indicators = analysis.indicators
        
        last_candle_date = str(analysis.time.date())
        
        tv_close = float(tv_indicators.get("close"))
        tv_open = float(tv_indicators.get("open", tv_close))
        tv_high = float(tv_indicators.get("high", tv_close))
        tv_low = float(tv_indicators.get("low", tv_close))
        tv_volume = float(tv_indicators.get("volume", 0)) # قنص الفوليوم الدقيق لليوم من تريدنج فيو
        
        # دمج أو تحديث السطر بالخمس قيم كاملة
        df.loc[last_candle_date] = [tv_open, tv_high, tv_low, tv_close, tv_volume]
        df = df.round(2)
        
        database[name] = df
        print(f"✅ {name} updated for date {last_candle_date}. Total days: {len(df)}")

    except Exception as e:
        print(f"💥 Failed to update {name}: {e}")

# حفظ قاعدة البيانات بالهيكل المضغوط والمُنظم
final_json = {}
for name, df in database.items():
    if not df.empty:
        df = df.sort_index()
        final_json[name] = {
            "columns": list(df.columns),
            "data": df.to_dict(orient="index")
        }

with open(DB_FILE, "w") as f:
    json.dump(final_json, f, indent=2)

print(f"\n🏁 Compact Database update complete. File '{DB_FILE}' optimized and saved successfully.")

print("⚙️ EGX DATABASE - GitHub Production Sourcing Engine")

import yfinance as yf
from tradingview_ta import TA_Handler, Interval as TVInterval
import json
import os
import pandas as pd
import requests

# =====================
# Symbols
# =====================
symbols = {
    "EGX30": "EGX30",  # المؤشر الرئيسي
    "OFH": "OFH", "OLFI": "OLFI", "EMFD": "EMFD", "ETEL": "ETEL", "EAST": "EAST",
    "EFIH": "EFIH", "ABUK": "ABUK", "OIH": "OIH", "SWDY": "SWDY", "ISPH": "ISPH",
    "ATQA": "ATQA", "MTIE": "MTIE", "ELEC": "ELEC", "HRHO": "HRHO", "ORWE": "ORWE",
    "JUFO": "JUFO", "DSCW": "DSCW", "SUGR": "SUGR", "ELSH": "ELSH", "RMDA": "RMDA",
    "RAYA": "RAYA", "EEII": "EEII", "MPCO": "MPCO", "GBCO": "GBCO", "TMGH": "TMGH",
    "ORHD": "ORHD", "AMOC": "AMOC", "FWRY": "FWRY", "COMI": "COMI", "ADIB": "ADIB",
    "PHDC": "PHDC", "MCQE": "MCQE", "SKPC": "SKPC", "EGAL": "EGAL"
}

DB_FILE = "egx_history_database.json"

# تجهيز الـ Session وتخطي حظر جيت هب (Custom User-Agent للبيئة المؤتمتة)
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# تحميل قاعدة البيانات بالهيكل الجديد المنسق
try:
    with open(DB_FILE, "r") as f:
        raw_database = json.load(f)
    print("💾 Existing compact database loaded successfully.")
except:
    raw_database = {}
    print("🆕 No database found. Initializing a new one...")

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
                print(f"📥 Downloading Hourly historical baseline for {name} from Yahoo (GitHub Session Mode)...")
                
                # استخدام الـ session لتفادي حظر الـ IP في جيت هب، وتوسيع المدة لـ 6 أشهر لضمان تدفق البيانات
                yf_hourly = yf.download("^CASE30", period="6mo", interval="1h", auto_adjust=False, progress=False, timeout=20, session=session)
                
                if not yf_hourly.empty:
                    if isinstance(yf_hourly.columns, pd.MultiIndex):
                        yf_hourly.columns = yf_hourly.columns.get_level_values(0)
                    yf_hourly.columns = [str(col).strip() for col in yf_hourly.columns]
                    
                    yf_hourly.index = pd.to_datetime(yf_hourly.index).tz_localize(None)
                    df_daily = yf_hourly.resample('D').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                    }).dropna()
                    
                    df = df_daily[["Open", "High", "Low", "Close"]].copy()
                    df.index = df.index.astype(str).str[:10]
                else:
                    print("❌ Yahoo hourly returned empty dataframe on GitHub server.")
            else:
                # آلية السحب اليومية الطبيعية للأسهم العادية مع الـ session
                print(f"📥 Downloading historical baseline for {name} from Yahoo...")
                yf_df = yf.download(f"{ticker}.CA", period="7mo", interval="1d", auto_adjust=False, progress=False, timeout=15, session=session)
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                yf_df = yf_df.dropna(subset=["Close"])
                
                df = yf_df[["Open", "High", "Low", "Close"]].copy()
                df.index = df.index.astype(str).str[:10]

        # جلب شمعة اليوم المباشرة والمحدثة من TradingView (تعمل دائماً بكفاءة على جيت هب)
        handler = TA_Handler(symbol=ticker, screener="egypt", exchange="EGX", interval=TVInterval.INTERVAL_1_DAY)
        analysis = handler.get_analysis()
        tv_indicators = analysis.indicators
        
        last_candle_date = str(analysis.time.date())
        
        tv_close = float(tv_indicators.get("close"))
        tv_open = float(tv_indicators.get("open", tv_close))
        tv_high = float(tv_indicators.get("high", tv_close))
        tv_low = float(tv_indicators.get("low", tv_close))
        
        # دمج أو تحديث السطر الحصري لجلسة اليوم
        df.loc[last_candle_date] = [tv_open, tv_high, tv_low, tv_close]
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

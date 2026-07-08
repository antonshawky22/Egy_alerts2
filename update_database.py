print("⚙️ EGX ENGINE v10.0 - TradingView Bulk Fetch Production Engine")

from tradingview_ta import TA_Handler, Interval as TVInterval, get_multiple_analysis
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
            df_temp = df_temp.sort_index(ascending=True)
            df_temp = df_temp[~df_temp.index.duplicated(keep='last')]
            database[name] = df_temp
        else:
            database[name] = pd.DataFrame()
    except:
        database[name] = pd.DataFrame()

# ==========================================
# 🚀 2. السحب الجماعي الصاروخي من TradingView (طلب واحد لكل السوق)
# ==========================================
print("\n⚡ Fetching live data for all tickers from TradingView in ONE BULK REQUEST...")

# تحضير التيكرات بالصيغة المطلوبة للسحب الجماعي: "EXCHANGE:SYMBOL"
tv_symbols_list = [f"EGX:{ticker}" for ticker in symbols.values()]

try:
    # أمر السحب الجماعي السحري في سطر واحد
    bulk_analysis = get_multiple_analysis(
        screener="egypt",
        interval=TVInterval.INTERVAL_1_DAY,
        symbols=tv_symbols_list
    )
    print("✅ Bulk data retrieved successfully from TradingView!")
except Exception as e:
    print(f"💥 Failed bulk fetch from TradingView: {e}")
    sys.exit(1)

# ==========================================
# 🚨 3. رادار فحص نبض السوق الفعلي (Market Pulse Sensor)
# ==========================================
print("🔍 Scanning market pulse using leaders...")
market_is_open = False
check_date = None

# فحص النبض من سهم القيادي التجاري الدولي المسحوب جماعياً
try:
    comi_analysis = bulk_analysis.get("EGX:COMI")
    if comi_analysis:
        indicators = comi_analysis.indicators
        check_date = str(comi_analysis.time.date())
        new_close = float(indicators.get("close", 0))
        new_open = float(indicators.get("open", new_close))
        
        df_comi = database.get("COMI", pd.DataFrame())
        if not df_comi.empty:
            old_data = df_comi.iloc[-1].values
            if new_close != old_data[3] or new_open != old_data[0]:
                market_is_open = True
except Exception as e:
    print(f"⚠️ Pulse check error: {e}")
    market_is_open = True # أمان

if not market_is_open and check_date is not None:
    print(f"🛑 Market is CLOSED/HOLIDAY for {check_date} (No price movement detected). Script stopped.")
    sys.exit(0)
else:
    print("🟢 Market is OPEN and active. Processing updates...")

# ==========================================
# 4. تفكيك البيانات المستلمة وضخها في الجداول
# ==========================================
for name, ticker in symbols.items():
    try:
        df = database.get(name, pd.DataFrame())
        
        # لو السهم جديد خالص، هيروح لياهو يسحب الهيستوري القديم (مرة واحدة بس في العمر)
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

        # استخراج تحليل السهم الحالي من القاموس الجماعي
        stock_analysis = bulk_analysis.get(f"EGX:{ticker}")
        if not stock_analysis:
            continue
            
        tv_indicators = stock_analysis.indicators
        last_candle_date = str(stock_analysis.time.date())
        
        tv_close = float(tv_indicators.get("close"))
        tv_open = float(tv_indicators.get("open", tv_close))
        tv_high = float(tv_indicators.get("high", tv_close))
        tv_low = float(tv_indicators.get("low", tv_close))
        
        # ⭐ الفوليوم الحقيقي لـ EGX30 وباقي الأسهم لايف وبدون تصفير
        tv_volume = float(tv_indicators.get("volume", 0))

        # تحديث اليوم الحالي
        df.loc[last_candle_date] = [tv_open, tv_high, tv_low, tv_close, tv_volume]
        
        # مسح التكرار والترتيب التصاعدي للحسابات الفنية الاستراتيجية
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index(ascending=True).round(2)
        database[name] = df
        print(f"✅ {name:6} -> Cleanly updated for {last_candle_date}. Volume: {tv_volume:,.0f}. Total days: {len(df)}")

    except Exception as e:
        print(f"💥 Failed to update {name}: {e}")

# ==========================================
# 5. حفظ قاعدة البيانات بالترتيب المريح للموبايل (الأحدث فوق)
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

print(f"\n🏁 Complete! File '{DB_FILE}' updated and saved successfully via TV Bulk Fetch.")

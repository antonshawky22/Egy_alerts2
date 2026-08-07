print("⚙️ EGX ENGINE v10.0 - TradingView Bulk Fetch Weekly Production Engine")

import json
import os
import pandas as pd
import time
import sys
import requests
import yfinance as yf
from tradingview_ta import TA_Handler, Interval as TVInterval, get_multiple_analysis

# ==========================================
# ⚙️ إعدادات المتغيرات وقاعدة البيانات الأسبوعية
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# تم تغيير اسم الملف ليكون ملف أسبوعي مستقل عن البيانات اليومية
DB_FILE = "egx_weekly_database_v1.json"

def send_telegram(message):
    """دالة إرسال رسائل التليجرام مع التأكد من عدم انهيار السكربت لو فشل الإرسال"""
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE" or not TELEGRAM_TOKEN:
        print(f"📱 [Telegram Mock]: {message}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Failed to send Telegram message: {e}")
      
symbols = {
"EGX30": "EGX30", "OLFI": "OLFI", "EMFD": "EMFD", "ETEL": "ETEL", "EAST": "EAST",
"EFIH": "EFIH", "ABUK": "ABUK", "OIH": "OIH", "SWDY": "SWDY", "ISPH": "ISPH",
"ATQA": "ATQA", "MTIE": "MTIE", "HRHO": "HRHO", "ORWE": "ORWE",
"JUFO": "JUFO", "DSCW": "DSCW", "SUGR": "SUGR", "ELSH": "ELSH", "RMDA": "RMDA",
"RAYA": "RAYA", "EEII": "EEII", "MPCO": "MPCO", "GBCO": "GBCO", "TMGH": "TMGH",
"ORHD": "ORHD", "AMOC": "AMOC", "FWRY": "FWRY", "COMI": "COMI", "ADIB": "ADIB",
"PHDC": "PHDC", "MCQE": "MCQE", "SKPC": "SKPC", "EGAL": "EGAL", "HELI": "HELI",
"QNBA": "QNBA", "HDBK": "HDBK", "FAIT": "FAIT", "SAUD": "SAUD",
"CCAP": "CCAP", "EKHO": "EKHO", "EGTS": "EGTS", "SDTI": "SDTI", "ARAB": "ARAB",
"KABO": "KABO", "SPIN": "SPIN", "MBSC": "MBSC", "EFIC": "EFIC",
"UBEE": "UBEE", "DAPH": "DAPH", "ACGC": "ACGC", "ASCM": "ASCM",
"BTFH": "BTFH", "CNFN": "CNFN", "MOIN": "MOIN", "INFI": "INFI",
"POUL": "POUL", "PRMH": "PRMH", "EPPK": "EPPK", "VERT": "VERT",
"MEPA": "MEPA", "NEDA": "NEDA", "OCDI": "OCDI", "GDWA": "GDWA",
"EGCH": "EGCH", "NDRL": "NDRL", "AJWA": "AJWA", "RAKT": "RAKT",
"NCCW": "NCCW", "EGSA": "EGSA", "EGAS": "EGAS", "BIOC": "BIOC",
"MEDI": "MEDI", "CLHO": "CLHO", "ICFC": "ICFC", "MHOT": "MHOT",
"ROTO": "ROTO", "PACH": "PACH", "UPMS": "UPMS", "UNIT": "UNIT"
}

# ==========================================
# 1. تحميل وقراءة البيانات الأسبوعية مع التنظيف
# ==========================================
try:
    with open(DB_FILE, "r") as f:
        raw_database = json.load(f)
    print("💾 Existing Weekly database loaded. Starting Auto-Clean check...")
except Exception as e:
    raw_database = {}
    print(f"🆕 Database not found or invalid ({e}). Creating a fresh stable weekly database...")

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
    except Exception:
        database[name] = pd.DataFrame()

# ==========================================
# 🚀 2. السحب الجماعي الأسبوعي (Weekly Bulk Fetch)
# ==========================================
print("\n⚡ Fetching live WEEKLY data for all tickers from TradingView in ONE BULK REQUEST...")
tv_symbols_list = [f"EGX:{ticker}" for ticker in symbols.values()]

bulk_analysis = None
max_retries = 3

for attempt in range(1, max_retries + 1):
    try:
        # استخدام Interval أسبوعي بدلاً من اليومي
        bulk_analysis = get_multiple_analysis(
            screener="egypt",
            interval=TVInterval.INTERVAL_1_WEEK,
            symbols=tv_symbols_list
        )
        if bulk_analysis:
            print("✅ Bulk weekly data retrieved successfully from TradingView!")
            break
    except Exception as e:
        print(f"⚠️ Attempt {attempt} failed: {e}")
        if attempt < max_retries:
            time.sleep(5)

if not bulk_analysis:
    error_msg = "💥 **خطأ حرج في النظام!**\nفشل سحب الأسعار الأسبوعية الجماعي من TradingView بعد 3 محاولات."
    print(error_msg)
    send_telegram(error_msg)
    sys.exit(1)

# ==========================================
# 🚨 3. رادار فحص نبض السوق
# ==========================================
print("🔍 Scanning weekly market pulse using leaders...")
check_date = None

try:
    comi_analysis = bulk_analysis.get("EGX:COMI")
    if comi_analysis:
        check_date = str(comi_analysis.time.date())
except Exception as e:
    print(f"⚠️ Pulse check error: {e}")

print(f"🟢 Processing weekly market updates for date: {check_date}...")

# ==========================================
# 4. تفكيك البيانات وضخها في الجداول الأسبوعية
# ==========================================
updated_count = 0
has_real_price_changes = False
failed_tickers = []

for name, ticker in symbols.items():
    try:
        df = database.get(name, pd.DataFrame())
        
        # لو السهم جديد، يتم سحب هيستوري أسبوعي قديم من Yahoo Finance
        if df.empty or len(df) < 10:
            print(f"📥 Downloading base weekly history for {name} from Yahoo...")
            yf_ticker = "^CASE30" if name == "EGX30" else f"{ticker}.CA"
            yf_df = yf.download(yf_ticker, period="2y", interval="1wk", auto_adjust=False, progress=False)
            if not yf_df.empty:
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                df = yf_df.dropna(subset=["Close"])[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
                df.index.name = "Date"

        # استخراج تحليل شمعة الأسبوع الحالية للسهم
        stock_analysis = bulk_analysis.get(f"EGX:{ticker}")
        if not stock_analysis:
            failed_tickers.append(name)
            continue
            
        tv_indicators = stock_analysis.indicators
        last_candle_date = str(stock_analysis.time.date())
        
        tv_close = float(tv_indicators.get("close"))
        tv_open = float(tv_indicators.get("open", tv_close))
        tv_high = float(tv_indicators.get("high", tv_close))
        tv_low = float(tv_indicators.get("low", tv_close))
        tv_volume = float(tv_indicators.get("volume", 0))

        # فحص وجود تغيرات في شمعة الأسبوع
        if not df.empty and last_candle_date in df.index:
            old_row = df.loc[last_candle_date]
            if (old_row["Close"] != tv_close or 
                old_row["Open"] != tv_open or 
                old_row["High"] != tv_high or 
                old_row["Low"] != tv_low):
                has_real_price_changes = True
        else:
            has_real_price_changes = True

        # تحديث بيانات الشمعة الأسبوعية الحالية
        df.loc[last_candle_date] = [tv_open, tv_high, tv_low, tv_close, tv_volume]
        
        # الترتيب والتنظيف
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index(ascending=True).round(2)
        database[name] = df
        updated_count += 1
        print(f"✅ {name:6} -> Cleanly updated weekly for {last_candle_date}. Volume: {tv_volume:,.0f}.")

    except Exception as e:
        print(f"💥 Failed to update weekly data for {name}: {e}")
        failed_tickers.append(name)

# ==========================================
# 5. حفظ قاعدة البيانات الأسبوعية
# ==========================================
if database:
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

    print(f"\n🏁 Complete! Weekly File '{DB_FILE}' updated.")

    if updated_count > 0 and has_real_price_changes:
        success_msg = f"✅ **تم تحديث أسعار البورصة الأسبوعية بنجاح!**\n📅 الأسبوع: `{check_date}`\n📊 الأسهم المحدثة: {updated_count}/{len(symbols)}"
        if failed_tickers:
            success_msg += f"\n⚠️ أسهم لم يتم تحديثها: {', '.join(failed_tickers)}"
        send_telegram(success_msg)
    else:
        print("ℹ️ No new weekly price changes detected. Telegram notification skipped.")

else:
    send_telegram("⚠️ **تحذير:** محاولة حفظ قاعدة بيانات أسبوعية فارغة! تم إيقاف الحفظ لحماية البيانات القديمة.")

print("⚙️ EGX ENGINE v10.1 - TradingView Bulk Fetch Production Engine (Pulse Sensor Integrated)")

import json
import os
import pandas as pd
import time
import sys
import requests
import yfinance as yf
from tradingview_ta import TA_Handler, Interval as TVInterval, get_multiple_analysis

# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DB_FILE = "egx_history_database_v2.json"

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
except Exception as e:
    raw_database = {}
    print(f"🆕 Database not found or invalid ({e}). Creating a fresh stable database...")

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
# 🚀 2. السحب الجماعي مع آلية إعادات المحاولة (Retry Logic)
# ==========================================
print("\n⚡ Fetching live data for all tickers from TradingView in ONE BULK REQUEST...")
tv_symbols_list = [f"EGX:{ticker}" for ticker in symbols.values()]

bulk_analysis = None
max_retries = 3

for attempt in range(1, max_retries + 1):
    try:
        bulk_analysis = get_multiple_analysis(
            screener="egypt",
            interval=TVInterval.INTERVAL_1_DAY,
            symbols=tv_symbols_list
        )
        if bulk_analysis:
            print("✅ Bulk data retrieved successfully from TradingView!")
            break
    except Exception as e:
        print(f"⚠️ Attempt {attempt} failed: {e}")
        if attempt < max_retries:
            time.sleep(5) # الانتظار 5 ثوانٍ قبل إعادة المحاولة

if not bulk_analysis:
    error_msg = "💥 **خطأ حرج في النظام!**\nفشل سحب الأسعار الجماعي من TradingView بعد 3 محاولات (احتمال حظر أو انقطاع الخدمة)."
    print(error_msg)
    send_telegram(error_msg)
    sys.exit(1)

# ==========================================
# 🚨 3. رادار فحص نبض السوق (Market Pulse Sensor)
# ==========================================
print("🔍 Scanning market pulse using market leaders (COMI, SWDY, HRHO)...")

market_active = False
leader_tickers = ["COMI", "SWDY", "HRHO"]
check_date = None

for l_name in leader_tickers:
    l_analysis = bulk_analysis.get(f"EGX:{symbols[l_name]}")
    if not l_analysis:
        continue

    l_indicators = l_analysis.indicators
    l_volume = float(l_indicators.get("volume", 0))
    l_close = float(l_indicators.get("close", 0))
    l_open = float(l_indicators.get("open", 0))
    l_high = float(l_indicators.get("high", 0))
    l_low = float(l_indicators.get("low", 0))

    check_date = str(l_analysis.time.date())

    # قراءة آخر شمعة مسجلة في الداتا للرائد
    df_leader = database.get(l_name, pd.DataFrame())

    if not df_leader.empty:
        last_recorded = df_leader.iloc[-1]
        # فحص هل توجد حركة فوليوم حقيقية + اختلاف في الأسعار عن آخر شمعة مسجلة
        if l_volume > 0 and (
            l_close != last_recorded["Close"]
            or l_open != last_recorded["Open"]
            or l_high != last_recorded["High"]
            or l_low != last_recorded["Low"]
        ):
            market_active = True
            print(f"🟢 Market activity detected via {l_name} (Vol: {l_volume:,.0f}) for date: {check_date}")
            break

if not market_active:
    print("😴 Market is Closed or No New Trading Activity Detected (Weekend/Holiday). Exiting safely without modifying database...")
    sys.exit(0)

print(f"🟢 Processing market updates for date: {check_date}...")

# ==========================================
# 4. تفكيك البيانات المستلمة وضخها في الجداول
# ==========================================
updated_count = 0
has_real_price_changes = False # تتبع وجود تغييرات حقيقية في الأسعار
failed_tickers = []

for name, ticker in symbols.items():
    try:
        df = database.get(name, pd.DataFrame())
        
        # لو السهم جديد، هيروح لياهو يسحب الهيستوري القديم
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

        # استخراج تحليل السهم الحالي
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

        # فحص هل توجد قيمة جديدة بالفعل تختلف عن آخر سعر مسجل في الداتا
        if not df.empty and last_candle_date in df.index:
            old_row = df.loc[last_candle_date]
            if (old_row["Close"] != tv_close or 
                old_row["Open"] != tv_open or 
                old_row["High"] != tv_high or 
                old_row["Low"] != tv_low):
                has_real_price_changes = True
        else:
            has_real_price_changes = True

        # تحديث اليوم الحالي
        df.loc[last_candle_date] = [tv_open, tv_high, tv_low, tv_close, tv_volume]
        
        # الترتيب والتنظيف
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index(ascending=True).round(2)
        database[name] = df
        updated_count += 1
        print(f"✅ {name:6} -> Cleanly updated for {last_candle_date}. Volume: {tv_volume:,.0f}.")

    except Exception as e:
        print(f"💥 Failed to update {name}: {e}")
        failed_tickers.append(name)

# ==========================================
# 5. حفظ قاعدة البيانات والحماية من الملفات الفارغة
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

    print(f"\n🏁 Complete! File '{DB_FILE}' updated.")

    # 🎯 الإرسال الذكي: تليجرام يصلك فقط لو كانت هناك أسعار حديثة وتغيرت بالفعل
    if updated_count > 0 and has_real_price_changes:
        print(
        f"✅ Price database updated successfully. "
        f"Date: {check_date}, "
        f"Updated: {updated_count}/{len(symbols)}")
    else:
        print("ℹ️ No new price changes detected. Telegram notification skipped to avoid noise.")
else:
    send_telegram("⚠️ **تحذير:** محاولة حفظ قاعدة بيانات فارغة! تم إيقاف الحفظ لحماية البيانات القديمة.")

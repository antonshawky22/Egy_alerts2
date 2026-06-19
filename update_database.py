print("EGX DATABASE - Dedicated Data Sourcing Engine")

import yfinance as yf
from tradingview_ta import TA_Handler, Interval as TVInterval
import json
import os
import pandas as pd

# =====================
# Symbols
# =====================
symbols = {
    "OFH": "OFH",
    "OLFI": "OLFI",
    "EMFD": "EMFD",
    "ETEL": "ETEL",
    "EAST": "EAST",
    "EFIH": "EFIH",
    "ABUK": "ABUK",
    "OIH": "OIH",
    "SWDY": "SWDY",
    "ISPH": "ISPH",
    "ATQA": "ATQA",
    "MTIE": "MTIE",
    "ELEC": "ELEC",
    "HRHO": "HRHO",
    "ORWE": "ORWE",
    "JUFO": "JUFO",
    "DSCW": "DSCW",
    "SUGR": "SUGR",
    "ELSH": "ELSH",
    "RMDA": "RMDA",
    "RAYA": "RAYA",
    "EEII": "EEII",
    "MPCO": "MPCO",
    "GBCO": "GBCO",
    "TMGH": "TMGH",
    "ORHD": "ORHD",
    "AMOC": "AMOC",
    "FWRY": "FWRY",
    "COMI": "COMI",
    "ADIB": "ADIB",
    "PHDC": "PHDC",
    "MCQE": "MCQE",
    "SKPC": "SKPC",
    "EGAL": "EGAL",
}

DB_FILE = "egx_history_database.json"

# 1. محاولة تحميل قاعدة البيانات إذا كانت موجودة، أو إنشائها لو أول مرة
try:
    with open(DB_FILE, "r") as f:
        database = json.load(f)
    print("💾 Existing database loaded successfully.")
except:
    database = {}
    print("🆕 No database found. Initializing a new one...")

# 2. المرور على الأسهم لتحديث الداتا التاريخية
for name, ticker in symbols.items():
    try:
        stock_data = database.get(name, [])

        # إذا كان السهم غير مخزن أو الداتا فارغة، نسحب 7 شهور من ياهو لأول مرة فقط كأرضية ثابتة
        if not stock_data or len(stock_data) < 100:
            print(
                f"📥 Downloading historical baseline for {name} from Yahoo..."
            )
            yf_df = yf.download(
                f"{ticker}.CA",
                period="7mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                timeout=15,
            )
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df = yf_df.dropna(subset=["Close"])

            stock_data = []
            for date, row in yf_df.iterrows():
                stock_data.append(
                    {
                        "Date": str(date.date()),
                        "Open": float(row["Open"]),
                        "High": float(row["High"]),
                        "Low": float(row["Low"]),
                        "Close": float(row["Close"]),
                    }
                )

        # تحويل البيانات إلى DataFrame لإضافة الجلسة الجديدة بدقة
        df = pd.DataFrame(stock_data)
        df.set_index("Date", inplace=True)

        # جلب أسعار وجدول الجلسة الأخيرة الفورية من TradingView
        handler = TA_Handler(
            symbol=ticker,
            screener="egypt",
            exchange="EGX",
            interval=TVInterval.INTERVAL_1_DAY,
        )
        analysis = handler.get_analysis()
        tv_indicators = analysis.indicators

        # قراءة تاريخ الشمعة مباشرة من تريدنج فيو وتحويله لصيغة YYYY-MM-DD
        tv_time = analysis.time
        last_candle_date = str(tv_time.date())

        tv_close = float(tv_indicators.get("close"))
        tv_open = float(tv_indicators.get("open", tv_close))
        tv_high = float(tv_indicators.get("high", tv_close))
        tv_low = float(tv_indicators.get("low", tv_close))

        # دمج أو تحديث سطر الجلسة الحالية بناءً على تاريخ تريدنج فيو الفعلي
        df.loc[last_candle_date] = [tv_open, tv_high, tv_low, tv_close]

        # إعادة حفظ البيانات بصيغة القائمة داخل القاموس للـ JSON
        df_reset = df.reset_index()
        database[name] = df_reset.to_dict(orient="records")
        print(f"✅ {name} updated for date {last_candle_date}. Total days: {len(df)}")

    except Exception as e:
        print(f"💥 Failed to update {name}: {e}")

# 3. حفظ قاعدة البيانات النهائية في الملف المخزن
with open(DB_FILE, "w") as f:
    json.dump(database, f, indent=2)

print(f"🏁 Database update complete. File '{DB_FILE}' is updated dynamically.")

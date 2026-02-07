import yfinance as yf
import pandas as pd

# =====================
# EGX symbols
# =====================
symbols = {
    "OFH": "OFH.CA","OLFI": "OLFI.CA","EMFD": "EMFD.CA","ETEL": "ETEL.CA",
    "EAST": "EAST.CA","EFIH": "EFIH.CA","ABUK": "ABUK.CA","OIH": "OIH.CA",
    "SWDY": "SWDY.CA","ISPH": "ISPH.CA","ATQA": "ATQA.CA","MTIE": "MTIE.CA",
    "ELEC": "ELEC.CA","HRHO": "HRHO.CA","ORWE": "ORWE.CA","JUFO": "JUFO.CA",
    "DSCW": "DSCW.CA","SUGR": "SUGR.CA","ELSH": "ELSH.CA","RMDA": "RMDA.CA",
    "RAYA": "RAYA.CA","EEII": "EEII.CA","MPCO": "MPCO.CA","GBCO": "GBCO.CA",
    "TMGH": "TMGH.CA","ORHD": "ORHD.CA","AMOC": "AMOC.CA","FWRY": "FWRY.CA",
    "COMI": "COMI.CA","ADIB": "ADIB.CA","PHDC": "PHDC.CA",
    "EGTS": "EGTS.CA","MCQE": "MCQE.CA","SKPC": "SKPC.CA",
    "EGAL": "EGAL.CA"
}

# =====================
# Helpers
# =====================
def fetch_data(ticker):
    try:
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

def add_emas(df):
    df["EMA25"] = df["Close"].ewm(span=25, adjust=False).mean()
    df["EMA35"] = df["Close"].ewm(span=35, adjust=False).mean()
    df["EMA45"] = df["Close"].ewm(span=45, adjust=False).mean()
    df["EMA55"] = df["Close"].ewm(span=55, adjust=False).mean()
    df["EMA4"]  = df["Close"].ewm(span=4, adjust=False).mean()
    df["EMA9"]  = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA3"]  = df["Close"].ewm(span=3, adjust=False).mean()
    df["EMA5"]  = df["Close"].ewm(span=5, adjust=False).mean()
    return df

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# =====================
# Trend determination
# =====================
def determine_trend(df):
    last = df.iloc[-1]
    price = last["Close"]
    
    ema25 = last["EMA25"]
    ema35 = last["EMA35"]
    ema45 = last["EMA45"]
    ema55 = last["EMA55"]
    
    # هابط
    if ema55 > ema45 > ema35 and price < ema25 and price < ema35:
        return "هابط"
    
    # صاعد
    prev_price = df["Close"].iloc[-2]
    if ema25 > ema35 and price > ema35 and prev_price > ema35:
        return "صاعد"
    
    # عرضي
    return "عرضي"

# =====================
# Main loop
# =====================
for name, ticker in symbols.items():
    df = fetch_data(ticker)
    if df is None or len(df) < 60:
        print(f"{name}: فشل جلب البيانات أو بيانات قليلة")
        continue
    
    df = add_emas(df)
    df["RSI14"] = rsi(df["Close"], 14)
    last = df.iloc[-1]
    
    trend = determine_trend(df)
    print(f"{name} | آخر سعر: {last['Close']:.2f} | آخر شمعة: {df.index[-1].date()} | الاتجاه: {trend}")

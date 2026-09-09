print("⚙️ EGX ENGINE v11.0 - TradingView Production Engine")
print("   Daily Live Update + Weekly 200-Bar Historical Refresh")
print("=" * 70)

import json, os, pandas as pd, time, sys, requests, argparse
from tradingview_ta import TA_Handler, Interval as TVInterval, get_multiple_analysis

try:
    from tradingviewApiPython import Client
    HISTORICAL_API_AVAILABLE = True
except Exception:
    HISTORICAL_API_AVAILABLE = False

# ===================== SETTINGS =====================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DB_FILE = "egx_history_database_v2.json"
REPORT_FILE = "egx_history_database_quality_v2.json"
TEMP_DB_FILE = DB_FILE + ".tmp"

REFRESH_BARS = 200
MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_DELAY = 2

# ===================== COMMAND LINE =====================

parser = argparse.ArgumentParser(description="EGX TradingView Production Database Engine")
parser.add_argument("--refresh-history", action="store_true",
                    help="Refresh the latest 200 daily candles for every symbol")
args = parser.parse_args()

env_refresh = os.getenv("REFRESH_HISTORY", "").strip().lower()
REFRESH_HISTORY = args.refresh_history or env_refresh in ["1", "true", "yes", "on"]

if REFRESH_HISTORY:
    print("\n🔵 MODE: WEEKLY HISTORICAL REFRESH")
    print(f"📈 Latest {REFRESH_BARS} daily candles will be synchronized from TradingView.")
else:
    print("\n🟢 MODE: NORMAL DAILY LIVE UPDATE")
    print("📈 Only the latest TradingView candle will be updated.")

# ===================== TELEGRAM =====================

def send_telegram(message):
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE" or not TELEGRAM_TOKEN:
        print(f"📱 [Telegram Mock]: {message}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=10)
        if not response.ok:
            print("⚠️ Telegram returned HTTP", response.status_code)
    except Exception as e:
        print(f"⚠️ Failed to send Telegram message: {e}")

# ===================== SYMBOLS =====================

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

# ===================== LOAD DATABASE =====================

print("\n" + "=" * 70)
print("📂 LOADING DATABASE")
print("=" * 70)

try:
    with open(DB_FILE, "r", encoding="utf-8") as f:
        raw_database = json.load(f)
    print("💾 Existing V2 database loaded.")
    print(f"📊 Symbols in file: {len(raw_database)}")
except Exception as e:
    raw_database = {}
    print("🆕 Database not found or invalid.")
    print(f"   Error: {e}")
    print("⚠️ A fresh database will be created only from successfully retrieved data.")

# ===================== JSON → DATAFRAMES =====================

database = {}
cleaning_report = {}

for name, content in raw_database.items():
    try:
        if isinstance(content, dict) and "columns" in content and "data" in content:
            df_temp = pd.DataFrame.from_dict(
                content["data"], orient="index", columns=content["columns"]
            )
            df_temp.index = pd.to_datetime(df_temp.index, errors="coerce")
            df_temp = df_temp[~df_temp.index.isna()]
            df_temp.index = df_temp.index.strftime("%Y-%m-%d")
            df_temp.index.name = "Date"

            for column in ["Open", "High", "Low", "Close", "Volume"]:
                if column in df_temp.columns:
                    df_temp[column] = pd.to_numeric(df_temp[column], errors="coerce")

            df_temp = df_temp.sort_index(ascending=True)

            before_duplicates = len(df_temp)
            df_temp = df_temp[~df_temp.index.duplicated(keep="last")]
            duplicate_count = before_duplicates - len(df_temp)

            database[name] = df_temp
            cleaning_report[name] = {
                "duplicate_dates_removed_on_load": duplicate_count,
                "bars_after_load": len(df_temp)
            }
        else:
            database[name] = pd.DataFrame()
            cleaning_report[name] = {"status": "INVALID_FORMAT"}
    except Exception as e:
        print(f"⚠️ Failed loading {name}: {e}")
        database[name] = pd.DataFrame()
        cleaning_report[name] = {"status": "LOAD_ERROR", "error": str(e)}

# ===================== TRADINGVIEW BULK FETCH =====================

print("\n" + "=" * 70)
print("🚀 TRADINGVIEW BULK FETCH")
print("=" * 70)
print("⚡ Fetching live data for all tickers from TradingView in ONE BULK REQUEST...")

tv_symbols_list = [f"EGX:{ticker}" for ticker in symbols.values()]
bulk_analysis = None

for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"🔌 Bulk attempt {attempt}/{MAX_RETRIES}")
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
        if attempt < MAX_RETRIES:
            print(f"⏳ Waiting {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

if not bulk_analysis:
    error_msg = (
        "💥 **خطأ حرج في النظام!**\n"
        "فشل سحب الأسعار الجماعي من TradingView "
        f"بعد {MAX_RETRIES} محاولات.\n"
        "تم إيقاف البرنامج بدون تعديل قاعدة البيانات."
    )
    print(error_msg)
    send_telegram(error_msg)
    sys.exit(1)

# ===================== MARKET PULSE SENSOR =====================

print("\n" + "=" * 70)
print("🚨 MARKET PULSE SENSOR")
print("=" * 70)
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

    df_leader = database.get(l_name, pd.DataFrame())

    if not df_leader.empty:
        last_recorded = df_leader.iloc[-1]
        if (
            l_volume > 0 and
            (
                l_close != last_recorded["Close"] or
                l_open != last_recorded["Open"] or
                l_high != last_recorded["High"] or
                l_low != last_recorded["Low"]
            )
        ):
            market_active = True
            print(f"🟢 Market activity detected via {l_name}")
            print(f"   Volume: {l_volume:,.0f}")
            print(f"   Date: {check_date}")
            break

# ===================== MARKET ACTIVITY DECISION =====================

if not market_active:
    print("😴 Market is Closed or No New Trading Activity Detected.")
    if not REFRESH_HISTORY:
        print("🛡️ Normal daily mode:")
        print("   Exiting safely without modifying database...")
        sys.exit(0)
    else:
        print("🔵 Weekly historical refresh mode:")
        print("   Continuing with historical synchronization...")
else:
    print(f"🟢 Processing market updates for date: {check_date}...")

# ===================== DAILY LIVE UPDATE =====================

updated_count = 0
has_real_price_changes = False
failed_tickers = []

print("\n" + "=" * 70)
print("📈 LIVE DAILY PRICE UPDATE")
print("=" * 70)

for name, ticker in symbols.items():
    try:
        df = database.get(name, pd.DataFrame())
        stock_analysis = bulk_analysis.get(f"EGX:{ticker}")

        if not stock_analysis:
            print(f"⚠️ {name:6} -> No TradingView analysis returned.")
            failed_tickers.append(name)
            continue

        tv_indicators = stock_analysis.indicators
        last_candle_date = str(stock_analysis.time.date())

        tv_close_raw = tv_indicators.get("close")
        if tv_close_raw is None:
            raise Exception("TradingView close is missing")

        tv_close = float(tv_close_raw)
        tv_open = float(tv_indicators.get("open", tv_close))
        tv_high = float(tv_indicators.get("high", tv_close))
        tv_low = float(tv_indicators.get("low", tv_close))
        tv_volume = float(tv_indicators.get("volume", 0))

        if tv_open <= 0 or tv_high <= 0 or tv_low <= 0 or tv_close <= 0:
            raise Exception("Invalid OHLC values")

        if (
            tv_high < tv_open or
            tv_high < tv_close or
            tv_high < tv_low or
            tv_low > tv_open or
            tv_low > tv_close
        ):
            raise Exception("Invalid OHLC relationship")

        if not df.empty and last_candle_date in df.index:
            old_row = df.loc[last_candle_date]
            if (
                old_row["Close"] != tv_close or
                old_row["Open"] != tv_open or
                old_row["High"] != tv_high or
                old_row["Low"] != tv_low or
                old_row["Volume"] != tv_volume
            ):
                has_real_price_changes = True
        else:
            has_real_price_changes = True

        df.loc[
            last_candle_date,
            ["Open", "High", "Low", "Close", "Volume"]
        ] = [tv_open, tv_high, tv_low, tv_close, tv_volume]

        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index(ascending=True).round(2)
        database[name] = df
        updated_count += 1

        print(
            f"✅ {name:6} -> Cleanly updated for "
            f"{last_candle_date}. Volume: {tv_volume:,.0f}."
        )

    except Exception as e:
        print(f"💥 Failed to update {name}: {e}")
        failed_tickers.append(name)

# ===================== WEEKLY HISTORICAL REFRESH =====================

historical_success = []
historical_failed = []
historical_changed = []
gap_report = {}

if REFRESH_HISTORY:
    print("\n" + "=" * 70)
    print(f"🔵 WEEKLY HISTORICAL REFRESH - LAST {REFRESH_BARS} BARS")
    print("=" * 70)

    if not HISTORICAL_API_AVAILABLE:
        print("\n💥 Historical TradingView API package is not available.")
        print("⚠️ Daily update remains valid.")
        print("⚠️ Historical refresh will be skipped.")
    else:
        for number, (name, ticker) in enumerate(symbols.items(), start=1):
            print("\n" + "-" * 70)
            print(f"🔄 Historical [{number}/{len(symbols)}] {name}")

            full_symbol = f"EGX:{ticker}"
            refresh_success = False

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"🔌 Historical attempt {attempt}/{MAX_RETRIES}")

                    client = Client()
                    chart = client.Session.Chart()
                    chart.set_market(
                        full_symbol,
                        {"timeframe": "D", "range": REFRESH_BARS}
                    )

                    time.sleep(5)
                    periods = chart.periods

                    if periods is None or len(periods) == 0:
                        raise Exception("TradingView returned no historical bars")

                    print(f"📥 Received {len(periods)} historical bars")

                    rows = []
                    bad_bars = 0

                    for bar in periods:
                        try:
                            timestamp = float(bar["time"])
                            date = pd.to_datetime(
                                timestamp, unit="s"
                            ).strftime("%Y-%m-%d")

                            open_price = float(bar["open"])
                            high_price = float(bar["max"])
                            low_price = float(bar["min"])
                            close_price = float(bar["close"])
                            volume = float(bar.get("volume", 0))

                            if (
                                open_price <= 0 or
                                high_price <= 0 or
                                low_price <= 0 or
                                close_price <= 0
                            ):
                                bad_bars += 1
                                continue

                            if (
                                high_price < open_price or
                                high_price < close_price or
                                high_price < low_price or
                                low_price > open_price or
                                low_price > close_price
                            ):
                                bad_bars += 1
                                continue

                            rows.append({
                                "Date": date,
                                "Open": open_price,
                                "High": high_price,
                                "Low": low_price,
                                "Close": close_price,
                                "Volume": volume
                            })
                        except Exception:
                            bad_bars += 1

                    refresh_df = pd.DataFrame(rows)

                    if refresh_df.empty:
                        raise Exception("No valid historical bars after parsing")

                    refresh_df = (
                        refresh_df.sort_values("Date")
                        .reset_index(drop=True)
                    )

                    before_duplicates = len(refresh_df)
                    refresh_df = (
                        refresh_df.drop_duplicates(
                            subset=["Date"], keep="last"
                        ).reset_index(drop=True)
                    )
                    duplicate_count = before_duplicates - len(refresh_df)

                    refresh_start = refresh_df["Date"].iloc[0]
                    refresh_end = refresh_df["Date"].iloc[-1]

                    date_series = pd.to_datetime(refresh_df["Date"])
                    date_diffs = date_series.diff().dt.days

                    suspicious_gaps = (
                        refresh_df.loc[
                            date_diffs > 5, "Date"
                        ].astype(str).tolist()
                    )

                    max_gap_days = (
                        int(date_diffs.max())
                        if not date_diffs.dropna().empty else 0
                    )

                    gap_report[name] = {
                        "refresh_start": refresh_start,
                        "refresh_end": refresh_end,
                        "max_calendar_gap_days": max_gap_days,
                        "suspicious_gaps_over_5_days": suspicious_gaps
                    }

                    old_df = database.get(name, pd.DataFrame()).copy()

                    if old_df.empty:
                        merged_df = refresh_df.set_index("Date")
                        changed_rows = len(merged_df)
                    else:
                        old_df.index = pd.to_datetime(
                            old_df.index, errors="coerce"
                        )
                        old_df = old_df[~old_df.index.isna()]
                        old_df.index = old_df.index.strftime("%Y-%m-%d")

                        new_df = refresh_df.set_index("Date")
                        changed_rows = 0

                        for date in new_df.index:
                            if date not in old_df.index:
                                changed_rows += 1
                            else:
                                old_row = old_df.loc[date]
                                new_row = new_df.loc[date]

                                if (
                                    float(old_row["Open"]) != float(new_row["Open"]) or
                                    float(old_row["High"]) != float(new_row["High"]) or
                                    float(old_row["Low"]) != float(new_row["Low"]) or
                                    float(old_row["Close"]) != float(new_row["Close"]) or
                                    float(old_row["Volume"]) != float(new_row["Volume"])
                                ):
                                    changed_rows += 1

                        merged_df = old_df.copy()

                        for date in new_df.index:
                            merged_df.loc[
                                date,
                                ["Open", "High", "Low", "Close", "Volume"]
                            ] = [
                                new_df.loc[date, "Open"],
                                new_df.loc[date, "High"],
                                new_df.loc[date, "Low"],
                                new_df.loc[date, "Close"],
                                new_df.loc[date, "Volume"]
                            ]

                    merged_df = merged_df[
                        ["Open", "High", "Low", "Close", "Volume"]
                    ]
                    merged_df = merged_df[
                        ~merged_df.index.duplicated(keep="last")
                    ]
                    merged_df = (
                        merged_df.sort_index(ascending=True).round(2)
                    )

                    database[name] = merged_df
                    historical_success.append(name)

                    if changed_rows > 0:
                        historical_changed.append(name)

                    print("\n✅ HISTORICAL SUCCESS")
                    print(f"   Received       : {len(periods)}")
                    print(f"   Valid          : {len(refresh_df)}")
                    print(f"   From           : {refresh_start}")
                    print(f"   To             : {refresh_end}")
                    print(f"   Changed rows   : {changed_rows}")
                    print(f"   Bad bars       : {bad_bars}")
                    print(f"   Duplicates     : {duplicate_count}")
                    print(f"   Max gap        : {max_gap_days} days")

                    if suspicious_gaps:
                        print("   ⚠️ Suspicious gaps:")
                        for gap in suspicious_gaps:
                            print(f"      {gap}")
                    else:
                        print("   ✅ No suspicious large gaps detected.")

                    refresh_success = True
                    break

                except Exception as e:
                    print(f"\n⚠️ Historical attempt {attempt} failed:")
                    print(f"   {e}")

                    if attempt < MAX_RETRIES:
                        print(f"   ⏳ Waiting {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)

            if not refresh_success:
                historical_failed.append(name)
                print(f"\n❌ HISTORICAL REFRESH FAILED FOR {name}")
                print("🛡️ Existing historical data was preserved.")

            if number < len(symbols):
                print(
                    f"⏳ Waiting {REQUEST_DELAY} seconds before next symbol..."
                )
                time.sleep(REQUEST_DELAY)

# ===================== FINAL QUALITY CHECK =====================

print("\n" + "=" * 70)
print("🔍 FINAL DATABASE QUALITY CHECK")
print("=" * 70)

final_quality_report = {}
total_bad_ohlc = 0
total_duplicate_dates = 0
total_suspicious_gaps = 0

for name in symbols:
    df = database.get(name, pd.DataFrame())

    if df.empty:
        final_quality_report[name] = {"status": "EMPTY", "bars": 0}
        continue

    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    duplicate_count = before - len(df)
    total_duplicate_dates += duplicate_count

    df = df.sort_index(ascending=True)

    bad_ohlc = df[
        (df["Open"] <= 0) |
        (df["High"] <= 0) |
        (df["Low"] <= 0) |
        (df["Close"] <= 0) |
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"]) |
        (df["High"] < df["Low"]) |
        (df["Low"] > df["Open"]) |
        (df["Low"] > df["Close"])
    ]

    bad_ohlc_count = len(bad_ohlc)
    total_bad_ohlc += bad_ohlc_count

    temp_change = df["Close"].pct_change() * 100
    large_moves = temp_change[temp_change.abs() >= 10]

    dates = pd.to_datetime(df.index, errors="coerce")
    diffs = dates.to_series().diff().dt.days

    suspicious_gaps = diffs[diffs > 5].dropna().tolist()
    total_suspicious_gaps += len(suspicious_gaps)

    database[name] = df.round(2)

    final_quality_report[name] = {
        "status": "OK",
        "bars": len(df),
        "start_date": str(df.index[0]),
        "end_date": str(df.index[-1]),
        "bad_ohlc_rows": bad_ohlc_count,
        "duplicate_dates_removed": duplicate_count,
        "large_moves_10_percent_or_more": len(large_moves),
        "suspicious_gaps_over_5_days": len(suspicious_gaps),
        "historical_refresh_success": name in historical_success,
        "historical_refresh_failed": name in historical_failed,
        "historical_data_changed": name in historical_changed
    }

# ===================== BUILD FINAL JSON =====================

print("\n" + "=" * 70)
print("💾 PREPARING FINAL DATABASE")
print("=" * 70)

final_blocks = []
required_columns = ["Open", "High", "Low", "Close", "Volume"]

for name in symbols:
    df = database.get(name, pd.DataFrame())

    if df.empty:
        print(f"⚠️ {name:6} -> No data. Skipped.")
        continue

    df_for_mobile = df.sort_index(ascending=False)

    missing_columns = [
        col for col in required_columns
        if col not in df_for_mobile.columns
    ]

    if missing_columns:
        print(
            f"⚠️ {name:6} -> Missing columns: "
            f"{missing_columns}. Skipped."
        )
        continue

    df_for_mobile = df_for_mobile[required_columns]

    columns_line = json.dumps(required_columns, ensure_ascii=False)
    data_line = json.dumps(
        df_for_mobile.to_dict(orient="index"),
        ensure_ascii=False
    )

    final_blocks.append(
        f'  "{name}": {{\n'
        f'    "columns": {columns_line},\n'
        f'    "data": {data_line}\n'
        f'  }}'
    )

# ===================== EMPTY DATABASE PROTECTION =====================

if not final_blocks:
    print("\n💥 CRITICAL:")
    print("   Final database is empty.")
    print("🛡️ Original database will NOT be modified.")

    send_telegram(
        "⚠️ **تحذير خطير:**\n"
        "محاولة حفظ قاعدة بيانات فارغة.\n"
        "تم إيقاف الحفظ لحماية قاعدة البيانات القديمة."
    )
    sys.exit(1)

# ===================== ATOMIC SAVE =====================

print("\n" + "=" * 70)
print("🛡️ SAFE ATOMIC SAVE")
print("=" * 70)

final_json = "{\n" + ",\n".join(final_blocks) + "\n}"

try:
    with open(TEMP_DB_FILE, "w", encoding="utf-8") as f:
        f.write(final_json)
        f.flush()
        os.fsync(f.fileno())

    with open(TEMP_DB_FILE, "r", encoding="utf-8") as f:
        json.load(f)

    os.replace(TEMP_DB_FILE, DB_FILE)

    print("✅ Database safely saved:")
    print(f"   {DB_FILE}")

except Exception as e:
    print(f"\n💥 DATABASE SAVE FAILED: {e}")
    print("🛡️ Original database was protected.")

    try:
        if os.path.exists(TEMP_DB_FILE):
            os.remove(TEMP_DB_FILE)
    except Exception:
        pass

    send_telegram(
        "⚠️ **خطأ في حفظ قاعدة البيانات!**\n"
        "تم الحفاظ على الملف الأصلي."
    )
    sys.exit(1)

# ===================== QUALITY REPORT =====================

report = {
    "engine_version": "v11.0",
    "data_source": "TradingView",
    "database_file": DB_FILE,
    "refresh_history_enabled": REFRESH_HISTORY,
    "refresh_bars": REFRESH_BARS,
    "symbols_total": len(symbols),
    "live_updated": updated_count,
    "live_failed": len(failed_tickers),
    "historical_success": len(historical_success),
    "historical_failed": len(historical_failed),
    "historical_changed": len(historical_changed),
    "total_bad_ohlc": total_bad_ohlc,
    "total_duplicate_dates": total_duplicate_dates,
    "total_suspicious_gaps": total_suspicious_gaps,
    "failed_live_symbols": failed_tickers,
    "failed_historical_symbols": historical_failed,
    "historical_changed_symbols": historical_changed,
    "gap_report": gap_report,
    "symbols": final_quality_report
}

try:
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("📄 Quality report saved:")
    print(f"   {REPORT_FILE}")

except Exception as e:
    print(f"⚠️ Failed to save quality report: {e}")

# ===================== FINAL SUMMARY =====================

print("\n" + "=" * 70)
print("🏁 EGX DATABASE ENGINE FINISHED")
print("=" * 70)

print(f"\n📊 Total symbols       : {len(symbols)}")
print(f"📈 Live updated        : {updated_count}")
print(f"⚠️ Live failed         : {len(failed_tickers)}")

if REFRESH_HISTORY:
    print(f"🔵 Historical success  : {len(historical_success)}")
    print(f"🔴 Historical failed   : {len(historical_failed)}")
    print(f"🔄 Historical changed  : {len(historical_changed)}")

print(f"\n❗ Bad OHLC rows       : {total_bad_ohlc}")
print(f"❗ Duplicate dates     : {total_duplicate_dates}")
print(f"⚠️ Suspicious gaps     : {total_suspicious_gaps}")

# ===================== FAILED SYMBOLS =====================

if failed_tickers:
    print("\n❌ LIVE FAILED SYMBOLS:")
    print(", ".join(failed_tickers))

if REFRESH_HISTORY and historical_failed:
    print("\n❌ HISTORICAL FAILED SYMBOLS:")
    print(", ".join(historical_failed))

# ===================== DATABASE OVERVIEW =====================

print("\n" + "=" * 70)
print("📊 DATABASE OVERVIEW")
print("=" * 70)

for name in symbols:
    df = database.get(name, pd.DataFrame())

    if not df.empty:
        print(
            f"{name:<8} {len(df):>5} bars   "
            f"{df.index[0]} → {df.index[-1]}"
        )
    else:
        print(f"{name:<8} ❌ NOT AVAILABLE")

# ===================== TELEGRAM =====================

if updated_count > 0 and has_real_price_changes:
    message = (
        "✅ *EGX Price Database Updated*\n"
        f"📅 Date: {check_date}\n"
        f"📊 Updated: {updated_count}/{len(symbols)}"
    )

    if REFRESH_HISTORY:
        message += (
            f"\n🔄 Historical Refresh: "
            f"{len(historical_success)}/{len(symbols)}"
        )

        if historical_failed:
            message += (
                f"\n⚠️ Historical Failed: "
                f"{len(historical_failed)}"
            )

    send_telegram(message)

else:
    print("\nℹ️ No new price changes detected.")
    print("📱 Telegram notification skipped to avoid noise.")

# ===================== FINAL STATUS =====================

print("\n" + "=" * 70)
print("🎯 FINAL STATUS")
print("=" * 70)

print("✅ TradingView is the only price source.")
print("✅ No Yahoo Finance used.")
print("✅ No manual corporate-action adjustment.")
print("✅ No artificial price correction.")
print("✅ Large movements are reported only.")
print("✅ Duplicate dates are removed.")
print("✅ OHLC integrity is checked.")
print("✅ Database is saved atomically.")

if REFRESH_HISTORY:
    print(f"✅ Latest {REFRESH_BARS} candles were synchronized where possible.")
    print("✅ Older historical data is preserved.")
    print("✅ Failed historical refreshes do not overwrite old data.")

print("\n🏁 Production database update completed.")

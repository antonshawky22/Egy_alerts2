import json
import os
import torch
import numpy as np
import pandas as pd

print("Initializing Kronos Multi-Timeframe (Daily + Weekly) Engine...")

# 1. تحميل قاعدتي البيانات (اليومي والأسبوعي)
daily_db_path = "egx_history_database_v2.json"
weekly_db_path = "egx_weekly_database_v2.json"

if not os.path.exists(daily_db_path) or not os.path.exists(weekly_db_path):
    raise FileNotFoundError("One or both database files (Daily/Weekly) are missing in root directory!")

print(f"Loading Daily database from {daily_db_path}...")
with open(daily_db_path, "r", encoding="utf-8") as f:
    daily_db = json.load(f)

print(f"Loading Weekly database from {weekly_db_path}...")
with open(weekly_db_path, "r", encoding="utf-8") as f:
    weekly_db = json.load(f)

symbols = list(weekly_db.keys())
print(f"Available symbols to analyze across dual timeframes: {symbols}")

# 2. إعداد جهاز التشغيل (CPU)
device = torch.device("cpu")
print(f"Using device: {device}")

# دوال المؤشرات الفنية المساعدة
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    diff = np.diff(prices[-period-1:])
    gain = np.mean(np.maximum(diff, 0))
    loss = np.mean(np.maximum(-diff, 0))
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_ema(prices, period=10):
    if len(prices) < period:
        return prices[-1]
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return np.dot(prices[-period:], weights)

# 3. دالة المحاكاة المدمجة (متعددة الأطر الزمنية)
def run_multi_timeframe_backtest(symbol_name, w_data, d_data, horizon=10):
    w_prices = []
    w_dates = []
    
    if isinstance(w_data, dict) and "data" in w_data:
        w_dict = w_data["data"]
        for d in sorted(w_dict.keys()):
            c_val = w_dict[d].get("Close", w_dict[d].get("close", w_dict[d].get("CLOSE", None)))
            if c_val is not None:
                w_prices.append(float(c_val))
                w_dates.append(d)
                
    d_prices = []
    d_dates = []
    d_lookup = {}
    
    if isinstance(d_data, dict) and "data" in d_data:
        d_dict = d_data["data"]
        for d in sorted(d_dict.keys()):
            c_val = d_dict[d].get("Close", d_dict[d].get("close", d_dict[d].get("CLOSE", None)))
            if c_val is not None:
                val = float(c_val)
                d_prices.append(val)
                d_dates.append(d)
                d_lookup[d] = val

    print(f"[{symbol_name}] Loaded {len(w_prices)} weekly candles and {len(d_prices)} daily candles.")
    
    if len(w_prices) < 20 or len(d_prices) < 50:
        return []

    results = []
    start_idx = 15
    end_idx = len(w_prices) - horizon
    
    for i in range(start_idx, end_idx, 4):
        w_slice = w_prices[:i]
        actual_future_w = w_prices[i : i + horizon]
        
        current_w_price = w_slice[-1]
        test_date = w_dates[i] if i < len(w_dates) else f"Index_{i}"
        
        try:
            # 1. فلتر الأسبوعي (البحث عن الاختراق والاتجاه الصاعد القوي)
            lookback = min(12, len(w_slice) - 1)
            recent_resistance = max(w_slice[-lookback - 1 : -1])
            is_weekly_breakout = current_w_price >= recent_resistance * 1.005
            
            if not is_weekly_breakout:
                continue
                
            w_ema = calculate_ema(w_slice, 10)
            if current_w_price < w_ema:
                continue
                
            # 2. فلتر اليومي (التأكد من أن السعر اليومي غير متضخم أو في تصحيح سيء)
            # سنبحث في الأسعار اليومية المتاحة حتى تاريخ الأسبوع الحالي
            valid_daily_prices = [d_lookup[d] for d in sorted(d_lookup.keys()) if d <= test_date]
            if len(valid_daily_prices) < 20:
                continue
                
            daily_rsi = calculate_rsi(valid_daily_prices, 14)
            daily_ema = calculate_ema(valid_daily_prices, 10)
            current_daily_price = valid_daily_prices[-1]
            
            # شروط يومية إضافية للدقة
            if current_daily_price < daily_ema or daily_rsi > 78 or daily_rsi < 45:
                continue
                
            predicted_return = 0.055
            predicted_price = current_w_price * (1.0 + predicted_return)
            
        except Exception as e:
            continue
        
        # محاكاة الحركة باستخدام الأسعار المستقبلية الأسبوعية بنظام التخارج بالموجة
        actual_final_price = actual_future_w[-1]
        
        success_flag = False
        achieved_exit_return = 0.0
        
        initial_stop_loss = 0.02
        trailing_drop_pct = 0.015
        
        peak_price = float(current_w_price)
        
        for future_p in actual_future_w:
            f_price = float(future_p)
            curr_ret = (f_price - float(current_w_price)) / float(current_w_price)
            
            if f_price > peak_price:
                peak_price = f_price
            
            if curr_ret <= -initial_stop_loss:
                achieved_exit_return = curr_ret
                success_flag = False
                break
            
            current_drop_from_peak = (peak_price - f_price) / peak_price
            peak_gain_from_entry = (peak_price - float(current_w_price)) / float(current_w_price)
            
            if peak_gain_from_entry >= 0.03 and current_drop_from_peak >= trailing_drop_pct:
                achieved_exit_return = (peak_price * (1.0 - trailing_drop_pct) - float(current_w_price)) / float(current_w_price)
                success_flag = True
                break
            
            achieved_exit_return = curr_ret
            success_flag = curr_ret > 0.01

        actual_return = achieved_exit_return

        results.append({
            "Symbol": symbol_name,
            "Date": str(test_date),
            "Entry_Price": round(float(current_w_price), 2),
            "Predicted_Price": round(float(predicted_price), 2),
            "Actual_Final_Price": round(float(actual_final_price), 2),
            "Actual_Return_%": round(float(actual_return * 100), 2),
            "Success": success_flag
        })
        
    return results

# 4. تشغيل التحليل المدمج على كل الأسهم
all_simulation_results = []

for symbol in symbols:
    if symbol in daily_db and symbol in weekly_db:
        print(f"\n--- Running Multi-Timeframe Analysis on: {symbol} ---")
        sym_results = run_multi_timeframe_backtest(symbol, weekly_db[symbol], daily_db[symbol])
        all_simulation_results.extend(sym_results)

report_filename = "kronos_backtest_report.csv"

if all_simulation_results:
    df_report = pd.DataFrame(all_simulation_results)
    overall_win_rate = (df_report["Success"].sum() / len(df_report)) * 100
    avg_return = df_report["Actual_Return_%"].mean()
    total_cum_return = df_report["Actual_Return_%"].sum()
    
    print("\n========================================")
    print(f"MULTI-TIMEFRAME (DAILY + WEEKLY) SUMMARY:")
    print(f"Total Filtered Checkpoints: {len(df_report)}")
    print(f"Overall Win Rate: {overall_win_rate:.2f}%")
    print(f"Mean Return per Trade: {avg_return:.2f}%")
    print(f"Total Cumulative Return: {total_cum_return:.2f}%")
    print("========================================")
else:
    print("\n[WARNING] No simulation results generated.")
    df_report = pd.DataFrame([{
        "Symbol": "N/A",
        "Date": "N/A",
        "Entry_Price": 0,
        "Predicted_Price": 0,
        "Actual_Final_Price": 0,
        "Actual_Return_%": 0,
        "Success": False
    }])

df_report.to_csv(report_filename, index=False)
print(f"Report saved successfully to: {report_filename} (Rows: {len(df_report)})")

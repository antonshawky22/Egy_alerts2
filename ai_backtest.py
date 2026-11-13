import json
import os
import torch
import numpy as np
import pandas as pd

print("Initializing Kronos Advanced Backtest Engine with Trend Filter...")

# 1. تحميل قاعدة بيانات الأسعار التاريخية
db_path = "egx_history_database_v2.json"
if not os.path.exists(db_path):
    raise FileNotFoundError(f"Database file {db_path} not found in root directory!")

print(f"Loading database from {db_path}...")
with open(db_path, "r", encoding="utf-8") as f:
    db = json.load(f)

symbols = list(db.keys())
print(f"Available symbols in database to test: {symbols}")

# 2. إعداد جهاز التشغيل (CPU)
device = torch.device("cpu")
print(f"Using device: {device}")

# 3. دالة محاكاة التنبؤ مع إضافة فلتر الاتجاه (Trend Filter)
def run_kronos_backtest_with_filter(symbol_name, symbol_data, horizon=20):
    prices = []
    dates = []
    
    if isinstance(symbol_data, dict) and "data" in symbol_data:
        data_dict = symbol_data["data"]
        # ترتيب التواريخ تصاعدياً من الأقدم للأحدث
        sorted_dates = sorted(data_dict.keys())
        
        for d in sorted_dates:
            candle = data_dict[d]
            if isinstance(candle, dict):
                close_val = candle.get("Close", candle.get("close", candle.get("CLOSE", None)))
                if close_val is not None:
                    prices.append(float(close_val))
                    dates.append(d)
    
    print(f"[{symbol_name}] Extracted {len(prices)} chronological price candles.")
    
    if len(prices) < horizon + 50:
        print(f"[{symbol_name}] Not enough data for trend filter. Skipping.")
        return []

    results = []
    start_idx = 50  # نحتاج على الأقل 50 شمعة لحساب المتوسط المتحرك للاتجاه
    end_idx = len(prices) - horizon
    
    for i in range(start_idx, end_idx, 15):  # عينة كل 15 جلسة
        historical_slice = prices[:i]
        actual_future_prices = prices[i : i + horizon]
        
        current_price = historical_slice[-1]
        test_date = dates[i] if i < len(dates) else f"Index_{i}"
        
        try:
            # 1. حساب المتوسط المتحرك للاتجاة (مثلاً آخر 50 شمعة)
            sma_50 = np.mean(historical_slice[-50:])
            
            # 2. حساب التوقع الذكي
            recent_change = (historical_slice[-1] - historical_slice[-10]) / historical_slice[-10]
            predicted_return = recent_change * (horizon / 10)
            predicted_price = float(current_price) * (1.0 + float(predicted_return))
            
            # 3. تطبيق فلتر الاتجاه (Trend Filter)
            # إذا كان النموذج يتوقع صعوداً، لكن السعر تحت المتوسط المتحرك (تريند هابط)، نتحفظ ونعدل التوقع أو نتجنبه
            is_predicted_up = predicted_price > current_price
            is_above_trend = current_price >= sma_50
            
            # إذا كانت الإشارة صعودية ولكن السهم تحت المتوسط العام، نخفض التوقع لتجنب الصفقات الخاطئة
            if is_predicted_up and not is_above_trend:
                # فلتر الاتجاه: إلغاء إشارة الشراء الخاطئة في التريند الهابط
                predicted_price = current_price * 0.99  # تعديل طفيف ليعكس الحذر
                
        except Exception as e:
            continue
        
        actual_final_price = actual_future_prices[-1]
        
        actual_return = (float(actual_final_price) - float(current_price)) / float(current_price)
        pred_return_calc = (float(predicted_price) - float(current_price)) / float(current_price)
        
        direction_match = (pred_return_calc * actual_return) > 0 or (abs(actual_return) < 0.005)
        
        results.append({
            "Symbol": symbol_name,
            "Date": str(test_date),
            "Entry_Price": round(float(current_price), 2),
            "Predicted_Price": round(float(predicted_price), 2),
            "Actual_Final_Price": round(float(actual_final_price), 2),
            "Actual_Return_%": round(float(actual_return * 100), 2),
            "Success": direction_match
        })
        
    return results

# 4. تشغيل الـ Backtest المطور لكل الأسهم
all_simulation_results = []

for symbol in symbols:
    print(f"\n--- Running Filtered Backtest on: {symbol} ---")
    sym_results = run_kronos_backtest_with_filter(symbol, db[symbol])
    all_simulation_results.extend(sym_results)

report_filename = "kronos_backtest_report.csv"

if all_simulation_results:
    df_report = pd.DataFrame(all_simulation_results)
    overall_win_rate = (df_report["Success"].sum() / len(df_report)) * 100
    
    print("\n========================================")
    print(f"FILTERED GLOBAL BACKTEST SUMMARY:")
    print(f"Total Test Checkpoints: {len(df_report)}")
    print(f"Overall Directional Win Rate (With Trend Filter): {overall_win_rate:.2f}%")
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
print(f"Enhanced report saved successfully to: {report_filename} (Rows: {len(df_report)})")

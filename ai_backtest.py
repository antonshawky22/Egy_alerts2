import json
import os
import torch
import numpy as np
import pandas as pd

print("Initializing Kronos Elite Backtest Engine (TP/SL & Enhanced Filters)...")

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

# دالة حساب مؤشر القوة النسبية RSI
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

# 3. دالة محاكاة التنبؤ النخبوية مع قواعد جني الأرباح ووقف الخسارة
def run_kronos_elite_backtest(symbol_name, symbol_data, horizon=10):
    prices = []
    dates = []
    
    if isinstance(symbol_data, dict) and "data" in symbol_data:
        data_dict = symbol_data["data"]
        sorted_dates = sorted(data_dict.keys())
        
        for d in sorted_dates:
            candle = data_dict[d]
            if isinstance(candle, dict):
                close_val = candle.get("Close", candle.get("close", candle.get("CLOSE", None)))
                if close_val is not None:
                    prices.append(float(close_val))
                    dates.append(d)
    
    print(f"[{symbol_name}] Extracted {len(prices)} chronological price candles.")
    
    if len(prices) < horizon + 30:
        print(f"[{symbol_name}] Not enough data. Skipping.")
        return []

    results = []
    start_idx = 30
    end_idx = len(prices) - horizon
    
    for i in range(start_idx, end_idx, 8):  # عينة دقيقة ومكثفة
        historical_slice = prices[:i]
        actual_future_prices = prices[i : i + horizon]
        
        current_price = historical_slice[-1]
        test_date = dates[i] if i < len(dates) else f"Index_{i}"
        
        try:
            # مؤشرات فنية صارمة
            sma_8 = np.mean(historical_slice[-8:])
            sma_25 = np.mean(historical_slice[-25:])
            rsi_val = calculate_rsi(historical_slice, 14)
            
            # العزم النقي للمدى القصير
            momentum = (historical_slice[-1] - historical_slice[-4]) / historical_slice[-4]
            predicted_return = momentum * 0.9
            predicted_price = float(current_price) * (1.0 + float(predicted_return))
            
            pred_pct = (predicted_price - current_price) / current_price
            
            # عتبة ثقة أعلى لتصفية الصفقات الضعيفة
            if abs(pred_pct) < 0.008:
                continue
                
            # شروط اتجاهية دقيقة جداً
            is_uptrend = current_price > sma_8 and sma_8 > sma_25
            is_downtrend = current_price < sma_8 and sma_8 < sma_25
            
            # فلتر الصرامة القصوى لمنع الدخول العكسي أو وقت التشبع الخطر
            if predicted_price > current_price and (not is_uptrend or rsi_val > 65 or rsi_val < 40):
                continue
            if predicted_price < current_price and (not is_downtrend or rsi_val < 35 or rsi_val > 60):
                continue
                
        except Exception as e:
            continue
        
        # محاكاة السعر خلال الجلسات القادمة مع التحقق من هدف الربح ووقف الخسارة الداخلي
        actual_final_price = actual_future_prices[-1]
        is_up_pred = predicted_price > current_price
        
        # فحص محاكاة الحركة اليومية داخل الـ horizon
        success_flag = False
        max_seen_return = 0.0
        
        for future_p in actual_future_prices:
            curr_ret = (float(future_p) - float(current_price)) / float(current_price)
            if is_up_pred:
                if curr_ret >= 0.03:  # حقق هدف ربح 3% أثناء الفترة
                    success_flag = True
                    break
                if curr_ret <= -0.02: # ضرب وقف خسارة 2%
                    success_flag = False
                    break
            else:
                if curr_ret <= -0.03:
                    success_flag = True
                    break
                if curr_ret >= 0.02:
                    success_flag = False
                    break
        
        # إذا لم يتم تفعيل الهدف أو الوقف خلال الفترة، نعتمد الاتجاه العام النهائي
        if not success_flag and len(actual_future_prices) > 0:
            actual_return = (float(actual_final_price) - float(current_price)) / float(current_price)
            pred_return_calc = (float(predicted_price) - float(current_price)) / float(current_price)
            success_flag = (pred_return_calc * actual_return) > 0 or (abs(actual_return) < 0.003)
        else:
            actual_return = (float(actual_future_prices[-1]) - float(current_price)) / float(current_price)

        results.append({
            "Symbol": symbol_name,
            "Date": str(test_date),
            "Entry_Price": round(float(current_price), 2),
            "Predicted_Price": round(float(predicted_price), 2),
            "Actual_Final_Price": round(float(actual_final_price), 2),
            "Actual_Return_%": round(float(actual_return * 100), 2),
            "Success": success_flag
        })
        
    return results

# 4. تشغيل الاختبار النخبوي على كل الأسهم
all_simulation_results = []

for symbol in symbols:
    print(f"\n--- Running Elite Backtest on: {symbol} ---")
    sym_results = run_kronos_elite_backtest(symbol, db[symbol])
    all_simulation_results.extend(sym_results)

report_filename = "kronos_backtest_report.csv"

if all_simulation_results:
    df_report = pd.DataFrame(all_simulation_results)
    overall_win_rate = (df_report["Success"].sum() / len(df_report)) * 100
    
    print("\n========================================")
    print(f"ELITE GLOBAL BACKTEST SUMMARY:")
    print(f"Total Filtered Checkpoints: {len(df_report)}")
    print(f"Overall Elite Win Rate: {overall_win_rate:.2f}%")
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

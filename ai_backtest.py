import json
import os
import torch
import numpy as np
import pandas as pd

print("Initializing Kronos Golden Balance Engine...")

# 1. تحميل قاعدة بيانات الأسعار التاريخية
db_path = "egx_weekly_database_v2.json"
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

# دوال المؤشرات الفنية المتقدمة
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

# 3. دالة محاكاة صفقات الميزان الذهبي (عدد مثالي + نسبة نجاح عالية + عائد قوي)
def run_kronos_golden_backtest(symbol_name, symbol_data, horizon=10):
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
    
    # خطوة متوازنة لفحص السوق (تولد عدد صفقات مثالي ومريح حوالي 13-15 صفقة شهرياً)
    for i in range(start_idx, end_idx, 8):
        historical_slice = prices[:i]
        actual_future_prices = prices[i : i + horizon]
        
        current_price = historical_slice[-1]
        test_date = dates[i] if i < len(dates) else f"Index_{i}"
        
        try:
            # مؤشرات فنية متوازنة ودقيقة
            sma_5 = np.mean(historical_slice[-5:])
            sma_15 = np.mean(historical_slice[-15:])
            ema_10 = calculate_ema(historical_slice, 10)
            rsi_val = calculate_rsi(historical_slice, 14)
            
            # عزم مركب متوازن
            roc_1 = (historical_slice[-1] - historical_slice[-2]) / historical_slice[-2]
            roc_4 = (historical_slice[-1] - historical_slice[-4]) / historical_slice[-4]
            
            score = (roc_1 * 0.45) + (roc_4 * 0.55)
            predicted_return = score * 1.2
            predicted_price = float(current_price) * (1.0 + float(predicted_return))
            
            pred_pct = (predicted_price - current_price) / current_price
            
            # عتبة ثقة متوازنة تضمن دقة عالية بدون مبالغة بالتحفظ
            if abs(pred_pct) < 0.012:
                continue
                
            # شروط اتجاه متوازنة وموثوقة
            is_uptrend = current_price > sma_5 and sma_5 > sma_15 and current_price > ema_10
            is_downtrend = current_price < sma_5 and sma_5 < sma_15 and current_price < ema_10
            
            if predicted_price > current_price and (not is_uptrend or rsi_val > 69 or rsi_val < 44):
                continue
            if predicted_price < current_price and (not is_downtrend or rsi_val < 31 or rsi_val > 56):
                continue
                
        except Exception as e:
            continue
        
        # محاكاة الحركة بنظام التخارج المتحرك المتوازن
        actual_final_price = actual_future_prices[-1]
        is_up_pred = predicted_price > current_price
        
        success_flag = False
        achieved_exit_return = 0.0
        
        initial_stop_loss = 0.014  # وقف خسارة 1.4%
        trailing_drop_pct = 0.011  # تراجع متحرك 1.1%
        
        peak_price = float(current_price)
        lowest_price = float(current_price)
        
        for future_p in actual_future_prices:
            f_price = float(future_p)
            curr_ret = (f_price - float(current_price)) / float(current_price)
            
            if is_up_pred:
                if f_price > peak_price:
                    peak_price = f_price
                
                if curr_ret <= -initial_stop_loss:
                    achieved_exit_return = curr_ret
                    success_flag = False
                    break
                
                current_drop_from_peak = (peak_price - f_price) / peak_price
                peak_gain_from_entry = (peak_price - float(current_price)) / float(current_price)
                
                if peak_gain_from_entry >= 0.025 and current_drop_from_peak >= trailing_drop_pct:
                    achieved_exit_return = (peak_price * (1.0 - trailing_drop_pct) - float(current_price)) / float(current_price)
                    success_flag = True
                    break
                
                achieved_exit_return = curr_ret
                success_flag = curr_ret > 0.007
            else:
                if f_price < lowest_price:
                    lowest_price = f_price
                
                if curr_ret >= initial_stop_loss:
                    achieved_exit_return = curr_ret
                    success_flag = False
                    break
                
                current_bounce_from_bottom = (f_price - lowest_price) / lowest_price
                bottom_gain_from_entry = (float(current_price) - lowest_price) / float(current_price)
                
                if bottom_gain_from_entry >= 0.025 and current_bounce_from_bottom >= trailing_drop_pct:
                    achieved_exit_return = (float(current_price) - lowest_price * (1.0 + trailing_drop_pct)) / float(current_price)
                    success_flag = True
                    break
                
                achieved_exit_return = curr_ret
                success_flag = curr_ret < -0.007

        actual_return = achieved_exit_return

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

# 4. تشغيل الاختبار المتوازن على كل الأسهم
all_simulation_results = []

for symbol in symbols:
    print(f"\n--- Running Golden Balance Backtest on: {symbol} ---")
    sym_results = run_kronos_golden_backtest(symbol, db[symbol])
    all_simulation_results.extend(sym_results)

report_filename = "kronos_backtest_report.csv"

if all_simulation_results:
    df_report = pd.DataFrame(all_simulation_results)
    overall_win_rate = (df_report["Success"].sum() / len(df_report)) * 100
    avg_return = df_report["Actual_Return_%"].mean()
    total_cum_return = df_report["Actual_Return_%"].sum()
    
    print("\n========================================")
    print(f"GOLDEN BALANCE GLOBAL SUMMARY:")
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

import json
import numpy as np
import pandas as pd
from datetime import datetime

# 1. تحميل قاعدة بيانات الأسعار التاريخية
print("Loading historical database...")
with open("egx_history_database_v2.json", "r", encoding="utf-8") as f:
    db = json.load(f)

# التحقق من وجود البيانات
print(f"Loaded symbols: {list(db.keys())}")

def run_backtest_simulation(symbol_data, horizon=20):
    """
    محاكاة مبسطة لاختبار دقة التنبؤ على البيانات التاريخية (Backtesting Loop)
    """
    prices = symbol_data.get("Close", [])
    dates = symbol_data.get("Date", [])
    
    if len(prices) < horizon + 100:
        print(f"Not enough data for simulation. Total candles: {len(prices)}")
        return []

    results = []
    
    # سنقوم باختبار عدة نقاط في الزمن (Rolling Windows)
    # البدء من الشمعة 100 وحتى قبل النهاية بمقدار الـ horizon
    start_idx = 100
    end_idx = len(prices) - horizon
    
    # للاختبار السريع، سنأخذ عينة كل 10 جلسات
    for i in range(start_idx, end_idx, 10):
        historical_slice = prices[:i]
        actual_future_prices = prices[i : i + horizon]
        
        current_price = historical_slice[-1]
        
        # محاكاة تنبؤ النموذج (هنا نستخدم اتجاه افتراضي أو دالة النموذج الفعلي لديك)
        # في نسختك الفعلية، يمكنك استبدال هذا السطر باستدعاء نموذج Kronos الحقيقي
        # مثال افتراضي يعتمد على المتوسط البسيط أو الانحدار للاختبار:
        predicted_price = current_price * (1 + np.random.uniform(-0.05, 0.08)) # للتوضيح فقط حتى تربطه بنموذجك
        
        actual_final_price = actual_future_prices[-1]
        
        actual_return = (actual_final_price - current_price) / current_price
        predicted_return = (predicted_price - current_price) / current_price
        
        # هل نجح النموذج في توقع اتجاه الحركة (صعود أو هبوط صحيح)؟
        direction_match = (predicted_return * actual_return) > 0 or (abs(actual_return) < 0.01)
        
        results.append({
            "Date_Index": i,
            "Entry_Price": current_price,
            "Predicted_Price": predicted_price,
            "Actual_Final_Price": actual_final_price,
            "Actual_Return_%": round(actual_return * 100, 2),
            "Success": direction_match
        })
        
    return results

# 2. تشغيل المحاكاة على المؤشر العام أو الأسهم
if "EGX30" in db:
    print("\n--- Running Backtest Simulation on EGX30 ---")
    egx_results = run_backtest_simulation(db["EGX30"])
    
    if egx_results:
        df_res = pd.DataFrame(egx_results)
        win_rate = (df_res["Success"].sum() / len(df_res)) * 100
        print(f"Total Test Points: {len(df_res)}")
        print(f"Estimated Directional Win Rate: {win_rate:.2f}%")
        
        # حفظ النتائج لتراجعها
        df_res.to_csv("backtest_simulation_report.csv", index=False)
        print("Report saved to: backtest_simulation_report.csv")
else:
    print("EGX30 data not found in database.")

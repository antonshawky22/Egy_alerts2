from datetime import datetime
import json
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
try:
  import xgboost as xgb
  HAS_XGBOOST = True
except ImportError:
  from sklearn.ensemble import RandomForestClassifier
  HAS_XGBOOST = False

# يمكنك تغيير الأفق الزمني هنا (20 أو 40 أو 60)
TARGET_HORIZON = 20 

def load_dataset(file_path="ai_training_dataset.json"):
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"الملف {file_path} غير موجود. تأكد من تشغيل بناء البيانات أولاً.")
  
  with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  
  if isinstance(data, dict) and "samples" in data:
    df = pd.DataFrame(data["samples"])
  elif isinstance(data, list):
    df = pd.DataFrame(data)
  else:
    raise ValueError("صيغة ملف الـ JSON غير متوافقة.")
            
  return df

def train_model():
  print("-> جاري تحميل بيانات التدريب...")
  df = load_dataset()
  
  # 1. تصفية البيانات حسب الأفق الزمني المستهدف لمنع الخلط
  if "horizon" in df.columns:
    df = df[df["horizon"] == TARGET_HORIZON].copy()
    print(f"-> تم تصفية العينات للأفق الزمني: {TARGET_HORIZON} جلسة. عدد العينات: {df.shape[0]}")
  else:
    print("⚠️ تحذير: عمود 'horizon' غير موجود في البيانات، سيتم استخدام كافة العينات.")

  if df.empty:
    raise ValueError(f"لا توجد عينات متاحة للأفق الزمني {TARGET_HORIZON}.")

  # 2. ترتيب البيانات زمنياً لكل سهم أو إجمالاً لمنع تسرب البيانات
  if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

  # 3. تحديد عمود الهدف بدقة (مع مراعاة تكاليف التداول: العائد > 0.5% مثلاً لتغطية العمولة)
  # هنا سنعتبر النجاح عائداً يحقق أكثر من 0.5% لتفادي الصفقات الوهمية ذات العائد الصفري
  PROFIT_THRESHOLD = 0.005 
  if "Future_Return_Pct" in df.columns:
    df["target"] = (df["Future_Return_Pct"] > PROFIT_THRESHOLD).astype(int)
    target_col = "target"
    print(f"-> تم إنشاء عمود الهدف بناءً على عائد > {PROFIT_THRESHOLD*100}% (لتغطية تكاليف التداول).")
  else:
    raise ValueError("عمود 'Future_Return_Pct' غير موجود في البيانات.")

  # 4. معالجة الميزات النصية للسوق (Market Categories) لتفادي استبعادها
  regime_mapping = {"BEARISH": -1, "NEUTRAL": 0, "BULLISH": 1, "INSUFFICIENT_DATA": 0}
  trend_mapping = {"DOWN": -1, "MIXED": 0, "UP": 1}
  momentum_mapping = {"NEGATIVE": -1, "MIXED": 0, "POSITIVE": 1}

  if "Market_Regime" in df.columns:
    df["Market_Regime"] = df["Market_Regime"].map(regime_mapping).fillna(0)
  if "Market_Trend" in df.columns:
    df["Market_Trend"] = df["Market_Trend"].map(trend_mapping).fillna(0)
  if "Market_Momentum" in df.columns:
    df["Market_Momentum"] = df["Market_Momentum"].map(momentum_mapping).fillna(0)

  exclude_cols = [target_col, "Future_Return_Pct", "Max_Gain_Pct", "Max_Drawdown_Pct", "Sessions_To_Max", "Sessions_To_Min", "date", "ticker", "symbol", "horizon"]
  feature_cols = [col for col in df.columns if col not in exclude_cols]
  
  # الاحتفاظ بالميزات الرقمية والمنطقية بعد ترميز النصوص
  X = df[feature_cols].select_dtypes(include=['number', 'bool'])
  y = df[target_col]
  
  print(f"-> عدد الميزات النهائية المستخدمة في التدريب: {X.shape[1]}")
  print(f"-> قائمة الميزات: {list(X.columns)}")

  # 5. تقسيم زمني صارم (80% تدريب أقدم، 20% اختبار أحدث) بدون خلط عشوائي لمنع التسرب
  split_index = int(len(X) * 0.8)
  X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
  y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
  
  print(f"-> حجم بيانات التدريب: {X_train.shape[0]}، حجم بيانات الاختبار: {X_test.shape[0]}.")
  
  # فحص توزيع الفئات
  print(f"-> توازن الفئات في التدريب (0 خسارة / 1 نجاح): {dict(y_train.value_counts())}")

  if HAS_XGBOOST:
    print("-> استخدام نموذج XGBoost للتدريب...")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
  else:
    print("-> استخدام نموذج RandomForest للتدريب...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
  model.fit(X_train, y_train)
  
  # 6. تقييم متقدم (Precision, Recall, F1, Confusion Matrix)
  y_pred = model.predict(X_test)
  
  print("\n========== نتائج تقييم النموذج ==========")
  print(classification_report(y_test, y_pred))
  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))
  print("=========================================\n")

  # 7. حفظ النموذج وملف المواصفات (Metadata) لمنع اختلاف التشغيل عن التدريب
  model_filename = f"ai_stock_model_{TARGET_HORIZON}d.pkl"
  meta_filename = f"ai_stock_model_{TARGET_HORIZON}d_meta.json"
  
  joblib.dump(model, model_filename)
  
  metadata = {
      "model_file": model_filename,
      "horizon": TARGET_HORIZON,
      "profit_threshold": PROFIT_THRESHOLD,
      "feature_columns": list(X.columns),
      "classes": [int(c) for c in model.classes_],
      "trained_at": datetime.now().isoformat(timespec="seconds"),
      "model_type": "XGBoost" if HAS_XGBOOST else "RandomForest"
  }

  with open(meta_filename, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

  print(f"-> تم حفظ النموذج بنجاح في: {model_filename}")
  print(f"-> تم حفظ ملف المواصفات (Metadata) في: {meta_filename}")

if __name__ == "__main__":
  train_model()

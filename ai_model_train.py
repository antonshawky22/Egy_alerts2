import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False

def load_dataset(file_path="ai_training_dataset.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"الملف {file_path} غير موجود. تأكد من تشغيل خطوات بناء البيانات أولاً.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # معالجة مرنة للـ JSON بغض النظر عن طريقة حفظه لضمان عدم حدوث خطأ الأطوال
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        try:
            df = pd.DataFrame(data)
        except ValueError:
            print("-> تنبيه: صيغة الـ JSON تحتاج معالجة خاصة، جاري ضبط البيانات...")
            df = pd.DataFrame.from_dict(data, orient='index').T
    else:
        raise ValueError("صيغة ملف الـ JSON غير مدعومة.")
        
    return df

def train_model():
    print("-> جاري تحميل بيانات التدريب...")
    df = load_dataset()
    
    if "target" not in df.columns:
        raise ValueError("عمود الهدف (Target/Label) غير موجود في ملف البيانات.")
    
    exclude_cols = ["target", "date", "ticker", "symbol"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    print(f"-> حجم بيانات التدريب: {X_train.shape[0]} صف، حجم بيانات الاختبار: {X_test.shape[0]} صف.")
    
    if HAS_XGBOOST:
        print("-> استخدام نموذج XGBoost للتدريب...")
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    else:
        print("-> استخدام نموذج RandomForest للتدريب...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"-> دقة النموذج على بيانات الاختبار: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    model_filename = "ai_stock_model.pkl"
    joblib.dump(model, model_filename)
    print(f"-> تم حفظ النموذج بنجاح في ملف: {model_filename}")

if __name__ == "__main__":
    train_model()

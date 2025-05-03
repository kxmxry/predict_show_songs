import pandas as pd
import joblib
import os

# טעינת המודל
model_path = 'predict/eden_rf_model_balanced.pkl'
future_path = 'dataset/future_show_features.csv'
output_path = 'predict/future_predictions.csv'

# בדיקה שהקובץ קיים
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ המודל לא נמצא: {model_path}")
if not os.path.exists(future_path):
    raise FileNotFoundError(f"❌ קובץ הפציפות העתידה לא נמצא: {future_path}")

# טעינה
model = joblib.load(model_path)
future_data = pd.read_csv(future_path)

# שמירת שם השירים וצפיות
song_names = future_data['שם השיר']
views = future_data['צפיות ביוטיוב']

# הכנת הפיצ'רים לחיזוי
features = [
    'בוצע בהופעה "Live קיסריה 2022"',
    'בוצע בהופעה "מנורה 2023"',
    'צפיות ביוטיוב',
    'האם סינגל'
]
X_future = future_data[features]

# קבלת סיכוי על בסיס סף ספק חזה
probabilities = model.predict_proba(X_future)[:, 1]  # סיכוי שהשיר יבוצע
predictions = (probabilities >= 0.1).astype(int)     #   סף חצו 0.25

# יצירת DataFrame עם תוצאות
results = pd.DataFrame({
    'שם השיר': song_names,
    'צפיות ביוטיוב': views,
    'סיכוי לביצוע': probabilities.round(3),
    'תחזית ביצוע': predictions
})


top = results.sort_values(by='סיכוי לביצוע', ascending=False).head(5)
print("🎯 שירים עם הסיכוי הגבוה ביותר לביצוע:")
print(top[['שם השיר', 'סיכוי לביצוע']])


# שמירה
results.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ התחזיות נשמרו ל: {output_path}\n")

# הצגת התוצאות
print(results)

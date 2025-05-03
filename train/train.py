import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


os.makedirs('predict', exist_ok=True)

# איתוז הדאטה סט 
X_train = pd.read_csv('dataset/X_train_median.csv')
X_test = pd.read_csv('dataset/X_test_median.csv')
y_train = pd.read_csv('dataset/y_train_median.csv').squeeze()
y_test = pd.read_csv('dataset/y_test_median.csv').squeeze()

# המרת y לערכי בינאריים
y_train = y_train.round().astype(int)
y_test = y_test.round().astype(int)

# התפלגות של היעד
print("\U0001f50d התפלגות ב-y_train:")
print(y_train.value_counts(normalize=True), "\n")

# class_weight='balanced' אימון המודל עם
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# חיזוי והערכה
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# פלט ביצועים
print("\U0001f4ca דוח ביצועים:")
print(report)

print("\U0001f4c9 מטריצת בלבול:")
print(conf_matrix)

# שמירת המודל
joblib.dump(model, 'predict/eden_rf_model_balanced.pkl')
print("\n\u2705 המודל נשמר בהצלחה: predict/eden_rf_model_balanced.pkl")

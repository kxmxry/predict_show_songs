import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from matplotlib import rcParams

# וודא: עדף שהמודל שומר קיים
model_path = 'predict/eden_rf_model_balanced.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"\u274c המודל לא נמצא במקום: {model_path}")

# טעינת המודל
model = joblib.load(model_path)
print("\u2705 המודל טעון בהצלחה\n")

# הצגת חשיבות תכונות
feature_names = [
    'בוצע בהופעה "Live קיסריה 2022"',
    'בוצע בהופעה "מנורה 2023"',
    'צפיות ביוטיוב',
    'האם סינגל'
]

importances = model.feature_importances_
rcParams['font.family'] = 'Arial'

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.xlabel('חשיבות')
plt.title('חשיבות תכונות במודל RandomForest')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# דוגמה לחיזוי (בעוד שיש פרוז)
# X_new = pd.read_csv('dataset/future_show_features.csv')
# predictions = model.predict(X_new)
# print("\nחיזוים:", predictions)

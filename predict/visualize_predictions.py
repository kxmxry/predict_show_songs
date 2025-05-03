import pandas as pd
import matplotlib.pyplot as plt
import os

# נתיב לקובץ התחזיות
predictions_path = 'predict/future_predictions.csv'

# בדיקה שהקובץ קיים
if not os.path.exists(predictions_path):
    raise FileNotFoundError(f"\u274c קובץ התחזיות לא נמצא: {predictions_path}")

# קריאה לקובץ התחזיות
df = pd.read_csv(predictions_path)

# הצגת רשימת שירים שהמודל חוזה שיבוצעו
predicted = df[df['תחזית ביצוע'] == 1]
print("\n\U0001f3a4 שירים שהמודל חוזה שיבוצעו:")
print(predicted['שם השיר'].tolist())

# אם קיימת עמודת 'צפיות ביוטיוב' – צייר גרף
if 'צפיות ביוטיוב' in df.columns:
    top = predicted.sort_values(by='צפיות ביוטיוב', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(top['שם השיר'], top['צפיות ביוטיוב'])
    plt.xlabel('צפיות ביוטיוב')
    plt.title('שירים שהמודל חוזה שיבוצעו')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print("\n\u26a0\ufe0f אין עמודת 'צפיות ביוטיוב' בקובץ התחזיות – לא ניתן להציג גרף.")

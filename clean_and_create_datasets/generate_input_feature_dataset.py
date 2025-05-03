import pandas as pd

# טען את הדאטהסט המקורי
df = pd.read_csv('dataset/clean_dataset.csv')

# נגדיר את העמודות שהמודל צריך
features = [
    'בוצע בהופעה "Live קיסריה 2022"',
    'בוצע בהופעה "מנורה 2023"',
    'צפיות ביוטיוב',
    'האם סינגל',
    'שם השיר'  # להצגה בלבד
]

# סנן שירים שעדיין **לא** בוצעו ב"מופע העשור 2024"
df_future = df[df['בוצע ב"מופע העשור 2024"'] != 1][features]

# הסר שורות עם ערכים חסרים
df_future = df_future.dropna()


# ניקוי שמות עמודות ממרכאות כפולות
df_future.columns = [col.replace('"', '') for col in df_future.columns]

# שמור כקובץ לקלט עתידי
df_future.to_csv('dataset/future_show_features.csv', index=False, encoding='utf-8-sig')
print("✅ קובץ future_show_features.csv נוצר בהצלחה עם שמות שירים אמיתיים")

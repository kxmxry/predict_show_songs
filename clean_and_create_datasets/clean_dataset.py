#המרת ערכי [יש למלא] ל־NaN

#המרת תאריכים בעמודת "תאריך הוצאה"

#המרת ערכים בינאריים לעמודות מספריות:

#"כן" = 1

#"לא" = 0

#"סביר" = 0.5

#המרת עמודת "צפיות ביוטיוב" למספרים



import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_excel(r"C:\projects\predict_show_songs\dataset\eden_full_songs excel.xlsx")

# Convert missing values like [יש למלא] to NaN
df = df.replace('[יש למלא]', np.nan)

# Convert date column to datetime
df['תאריך הוצאה'] = pd.to_datetime(df['תאריך הוצאה'], errors='coerce')

# Convert binary columns to numeric (1/0)
binary_columns = [
    'בוצע בהופעה "Live קיסריה 2022"',
    'בוצע בהופעה "מנורה 2023"',
    'בוצע ב"מופע העשור 2024"',
    'האם סינגל'
]

for col in binary_columns:
    df[col] = df[col].map({'כן': 1, 'לא': 0, 'סביר': 0.5})

# Clean text columns (remove special characters and convert to numeric where possible)
# Clean YouTube views
df['צפיות ביוטיוב'] = df['צפיות ביוטיוב'].astype(str).str.replace(',', '').str.replace('K', '000').str.replace('M', '000000').astype(float)

# Clean Spotify streams
df['זרמים בספוטיפיי'] = df['זרמים בספוטיפיי'].astype(str).str.replace(',', '').astype(float)

# Clean chart positions (extract numbers only)
df['מיקום במצעדים'] = df['מיקום במצעדים'].astype(str).str.extract('(\d+)').astype(float)

# Save the cleaned dataset
df.to_csv("dataset/clean_dataset.csv", index=False, encoding='utf-8-sig')

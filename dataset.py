import pandas as pd
import os
from collections import Counter
import whisper
import yt_dlp

eden_2023_df = pd.read_csv('eden_ben_zaken_songs.csv')

# הורדת אודיו מהיוטיוב (קישור יש לעדכן לפי ההופעה הרצויה)
url = "https://www.youtube.com/watch?v=X83PNKZLwYU"
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': 'eden_live.%(ext)s'
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
print("✅ שלב 1: הורדת האודיו הסתיימה")

# תמלול האודיו
model = whisper.load_model("medium")
result = model.transcribe("eden_live.mp3", language="he" , fp16 = False)
print("✅ שלב 2: תמלול הסתיים")

with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
print("📄 התמלול נשמר לקובץ transcript.txt")

# זיהוי שירים מתוך תמלול
known_songs = [
    "מלכת השושנים", "יאסו", "חיים שלי", "אגרוף", "תל אביב בלילה",
    "עמנואלה", "לזאת שניצחה", "פילטרים יפים", "משחק קלפים",
    "תזיזו", "מה קרה", "רציתי", "שיכורים מאהבה", "כינורות"
]

found_songs = []
with open("transcript.txt", encoding="utf-8") as f:
    text = f.read()
for song in known_songs:
    if song in text:
        found_songs.append(song)

if not found_songs:
    print("⚠️ לא זוהו שירים מהתמלול. ייתכן שאין התאמות או שהתמלול לא הצליח.")
else:
    print(f"🔍 שירים שזוהו בתמלול: {found_songs}")

# יצירת טבלה חדשה מהשירים שזוהו
df_transcribed = pd.DataFrame(Counter(found_songs).items(), columns=["שם שיר", "כמות מופעים"])
df_transcribed["בתוך מחרוזת"] = "לא נבדק"
df_transcribed["דואט עם"] = ""
df_transcribed["מופע"] = "הופעת העשור"
df_transcribed["תאריך"] = "2023-11-05"
df_transcribed["מיקום"] = "היכל מנורה, תל אביב"
df_transcribed["מזהה הופעה"] = "heichal_menora_2023"

# הדפסה
print(df_transcribed)
print("✅ סיום: הטבלה שנוצרה מהשירים שזוהו:")
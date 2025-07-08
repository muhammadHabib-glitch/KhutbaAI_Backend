# import requests
# import urllib.parse
#
# # ── Test Quran API ───────────────────────────────────────
# try:
#     # Use a known exact phrase from Surah Al-Fatiha
#     phrase = "بسم الله الرحمن الرحيم"
#     q = urllib.parse.quote(phrase)
#     url = f"https://api.alquran.cloud/v1/search/{q}/all/quran-uthmani"
#     r = requests.get(url, timeout=5)
#     print("Quran API status:", r.status_code)
#     print("Quran API response:", r.json())
# except Exception as e:
#     print("Quran API error:", e)
#
# # ── Test Hadith API ──────────────────────────────────────
# try:
#     # Use a short English snippet that might appear in a hadith record
#     snippet = "Actions are judged by intentions"
#     url = "https://hadithapi.com/api/hadiths"
#     params = {
#         "apiKey": "YOUR_REAL_HADITH_API_KEY",  # replace with your key
#         "hadithEnglish": snippet
#     }
#     r2 = requests.get(url, params=params, timeout=5)
#     print("Hadith API status:", r2.status_code)
#     print("Hadith API response:", r2.json())
# except Exception as e:
#     print("Hadith API error:", e)

# import os
# print(os.environ.get("HADITH_API_KEY"))


from dotenv import load_dotenv
import os

load_dotenv()  # This loads the .env file

print("HADITH_API_KEY =", os.getenv("HADITH_API_KEY"))

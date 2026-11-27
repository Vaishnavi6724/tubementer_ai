import os
from dotenv import load_dotenv

load_dotenv()

# Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# YouTube Data API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Judge0
JUDGE0_API_URL = os.getenv(
    "JUDGE0_API_URL",
    "https://ce.judge0.com"
)

# LibreTranslate
LIBRETRANSLATE_URL = os.getenv(
    "LIBRETRANSLATE_URL",
    "https://translate.fedilab.app/translate"
)
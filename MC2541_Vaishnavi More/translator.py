import requests

from config import LIBRETRANSLATE_URL


def translate_text(text, target_language="mr"):

    if not text or not text.strip():
        raise ValueError("Text to translate is empty")

    data = {
        "q": text,
        "source": "auto",
        "target": target_language,
        "format": "text"
    }

    response = requests.post(
        LIBRETRANSLATE_URL,
        data=data,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if "translatedText" not in result:
        raise ValueError(
            f"Translation failed: {result}"
        )

    return result["translatedText"]
import re
import requests
from config import YOUTUBE_API_KEY


def extract_video_id(url):
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


def get_video_info(url):

    video_id = extract_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY is missing in .env")

    api_url = "https://www.googleapis.com/youtube/v3/videos"

    params = {
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": YOUTUBE_API_KEY
    }

    response = requests.get(
        api_url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("items"):
        raise ValueError("YouTube video not found")

    item = data["items"][0]

    snippet = item["snippet"]
    statistics = item.get("statistics", {})

    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "thumbnail": snippet["thumbnails"]["high"]["url"],
        "views": statistics.get("viewCount", "0"),
        "likes": statistics.get("likeCount", "0")
    }
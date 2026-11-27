import time
import pandas as pd
import requests

from config import YOUTUBE_API_KEY


INPUT_FILE = "data/cleaned_transcripts.csv"
OUTPUT_FILE = "data/youtube_api_data.csv"

YOUTUBE_API_URL = (
    "https://www.googleapis.com/youtube/v3/videos"
)


def get_video_details(video_ids):

    params = {
        "part": "snippet,statistics,contentDetails",
        "id": ",".join(video_ids),
        "key": YOUTUBE_API_KEY
    }

    response = requests.get(
        YOUTUBE_API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def main():

    print("Loading existing dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Total videos:", len(df))

    video_ids = (
        df["video_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print("Unique video IDs:", len(video_ids))

    all_data = []

    # YouTube API allows up to 50 video IDs per request
    for i in range(0, len(video_ids), 50):

        batch = video_ids[i:i + 50]

        print(
            f"Collecting videos "
            f"{i + 1} to {min(i + 50, len(video_ids))}..."
        )

        try:

            result = get_video_details(batch)

            for item in result.get("items", []):

                video_id = item["id"]

                snippet = item.get(
                    "snippet",
                    {}
                )

                statistics = item.get(
                    "statistics",
                    {}
                )

                content_details = item.get(
                    "contentDetails",
                    {}
                )

                record = {
                    "video_id": video_id,

                    "api_title": snippet.get(
                        "title"
                    ),

                    "api_channel_name": snippet.get(
                        "channelTitle"
                    ),

                    "published_at": snippet.get(
                        "publishedAt"
                    ),

                    "category_id": snippet.get(
                        "categoryId"
                    ),

                    "view_count": statistics.get(
                        "viewCount"
                    ),

                    "like_count": statistics.get(
                        "likeCount"
                    ),

                    "comment_count": statistics.get(
                        "commentCount"
                    ),

                    "api_duration": content_details.get(
                        "duration"
                    )
                }

                all_data.append(record)

        except Exception as e:

            print(
                "Error while collecting batch:",
                e
            )

        time.sleep(0.5)

    api_df = pd.DataFrame(all_data)

    api_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("API data collection completed!")
    print("Records collected:", len(api_df))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
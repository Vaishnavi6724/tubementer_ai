import pandas as pd
import ast
import re

INPUT_FILE = "data/youtube_transcripts.csv"
OUTPUT_FILE = "data/cleaned_transcripts.csv"


def extract_transcript(transcript):
    """Convert transcript list/dictionary data into plain text."""

    try:
        # Convert string representation to Python object
        data = ast.literal_eval(transcript) if isinstance(transcript, str) else transcript

        if isinstance(data, list):
            texts = []

            for item in data:
                if isinstance(item, dict) and "text" in item:
                    texts.append(str(item["text"]))

            return " ".join(texts)

        return str(data)

    except Exception:
        return ""


print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))

# Extract clean transcript text
print("Cleaning transcripts...")

df["clean_text"] = df["transcript"].apply(extract_transcript)

# Remove extra spaces
df["clean_text"] = df["clean_text"].apply(
    lambda x: re.sub(r"\s+", " ", x).strip()
)

# Remove records with empty transcripts
df = df[df["clean_text"].str.len() > 50]

# Remove duplicate videos
df = df.drop_duplicates(subset=["video_id"])

# Keep useful columns
df = df[
    [
        "video_id",
        "title",
        "channel_name",
        "url",
        "language",
        "duration_seconds",
        "clean_text",
    ]
]

df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaning completed!")
print("Remaining records:", len(df))
print("Saved to:", OUTPUT_FILE)

print("\nExample cleaned transcript:")
print(df.iloc[0]["clean_text"][:1000])
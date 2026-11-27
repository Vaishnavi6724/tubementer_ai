import pandas as pd
import json
import os

INPUT_FILE = "data/cleaned_transcripts.csv"
OUTPUT_FILE = "data/chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def create_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Videos:", len(df))

all_chunks = []

print("\nCreating chunks...")

for index, row in df.iterrows():

    transcript = str(row["clean_text"])

    chunks = create_chunks(
        transcript,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    for chunk_number, chunk in enumerate(chunks):

        all_chunks.append({
            "video_id": row["video_id"],
            "title": row["title"],
            "channel_name": row["channel_name"],
            "url": row["url"],
            "language": row["language"],
            "chunk_id": chunk_number,
            "text": chunk
        })

print("\nTotal chunks created:", len(all_chunks))

os.makedirs("data", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        all_chunks,
        f,
        ensure_ascii=False,
        indent=2
    )

print("Chunks saved to:", OUTPUT_FILE)

print("\nExample chunk:")
print(all_chunks[0]["text"][:1000])
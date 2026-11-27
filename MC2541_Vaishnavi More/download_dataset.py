from datasets import load_dataset
import pandas as pd

print("Loading YouTube transcript dataset...")

dataset = load_dataset(
    "arrmlet/data-universe-youtube-1",
    split="train"
)

print("\nDataset loaded successfully!")
print("Total records:", len(dataset))
print("Columns:", dataset.column_names)

# Use only 500 records for our project
sample_size = min(500, len(dataset))
sample = dataset.select(range(sample_size))

df = sample.to_pandas()

# Save dataset
df.to_csv("data/youtube_transcripts.csv", index=False)

print("\nDataset saved successfully!")
print("Records saved:", len(df))
print("File: data/youtube_transcripts.csv")

print("\nFirst record:")
print(df.iloc[0])
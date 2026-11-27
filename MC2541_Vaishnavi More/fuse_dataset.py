import pandas as pd


# Input files
ORIGINAL_FILE = "data/cleaned_transcripts.csv"
API_FILE = "data/youtube_api_data.csv"

# Output file
OUTPUT_FILE = "data/fused_dataset.csv"


print("Loading original dataset...")
original_df = pd.read_csv(ORIGINAL_FILE)

print("Loading API dataset...")
api_df = pd.read_csv(API_FILE)


print("\nOriginal dataset shape:", original_df.shape)
print("API dataset shape:", api_df.shape)


# Remove accidental duplicate API records
api_df = api_df.drop_duplicates(
    subset=["video_id"]
)


# Merge datasets using video_id
fused_df = original_df.merge(
    api_df,
    on="video_id",
    how="left"
)


print("\nFused dataset shape:", fused_df.shape)


# Save fused dataset
fused_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFused dataset created successfully!")
print("Saved to:", OUTPUT_FILE)


# Check API enrichment
matched = fused_df["api_title"].notna().sum()
unmatched = fused_df["api_title"].isna().sum()

print("\nAPI records matched:", matched)
print("API records not matched:", unmatched)


# Display missing values
print("\nMissing values in fused dataset:")
print(fused_df.isnull().sum())


print("\nFirst 5 rows of fused dataset:")
print(fused_df.head())
import pandas as pd

print("=" * 50)
print("FUSED DATASET INSIGHTS")
print("=" * 50)

# Load fused dataset
df = pd.read_csv("data/fused_dataset.csv")

print("\n1. DATASET INFORMATION")
print("-" * 50)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
for col in df.columns:
    print("-", col)


# --------------------------------------------------
# API MATCHING
# --------------------------------------------------

print("\n2. API ENRICHMENT")
print("-" * 50)

matched = df["api_title"].notna().sum()
unmatched = df["api_title"].isna().sum()
total = len(df)

print("API matched:", matched)
print("API unmatched:", unmatched)

print("API match percentage:",
      round((matched / total) * 100, 2), "%")

print("API unmatched percentage:",
      round((unmatched / total) * 100, 2), "%")


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print("\n3. MISSING VALUES")
print("-" * 50)

missing = df.isnull().sum()

print(missing[missing > 0])


# --------------------------------------------------
# VIDEO STATISTICS
# --------------------------------------------------

print("\n4. VIDEO STATISTICS")
print("-" * 50)

for col in ["view_count", "like_count", "comment_count"]:

    if col in df.columns:

        print("\n", col)

        print("Count:", df[col].count())
        print("Mean:", round(df[col].mean(), 2))
        print("Median:", round(df[col].median(), 2))
        print("Maximum:", df[col].max())


# --------------------------------------------------
# TOP VIDEOS
# --------------------------------------------------

print("\n5. TOP 10 MOST VIEWED VIDEOS")
print("-" * 50)

top_views = df.nlargest(
    10,
    "view_count"
)[
    ["video_id", "title", "view_count"]
]

print(top_views.to_string(index=False))


print("\n6. TOP 10 MOST LIKED VIDEOS")
print("-" * 50)

top_likes = df.nlargest(
    10,
    "like_count"
)[
    ["video_id", "title", "like_count"]
]
print(top_likes.to_string(index=False))


print("\n7. TOP 10 MOST COMMENTED VIDEOS")
print("-" * 50)

top_comments = df.nlargest(
    10,
    "comment_count"
)[
    ["video_id", "title", "comment_count"]
]

print(top_comments.to_string(index=False))


# --------------------------------------------------
# CHANNEL ANALYSIS
# --------------------------------------------------

print("\n8. TOP CHANNELS BY VIDEO COUNT")
print("-" * 50)

channels = (
    df["channel_name"]
    .value_counts()
    .head(10)
)

print(channels)


# --------------------------------------------------
# CORRELATION
# --------------------------------------------------

print("\n9. CORRELATION ANALYSIS")
print("-" * 50)

numeric_columns = [
    "view_count",
    "like_count",
    "comment_count"
]

print(
    df[numeric_columns]
    .corr()
    .round(3)
)


# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------

print("\n10. DUPLICATE CHECK")
print("-" * 50)

print(
    "Duplicate video IDs:",
    df["video_id"].duplicated().sum()
)


print("\n" + "=" * 50)
print("INSIGHTS ANALYSIS COMPLETED")
print("=" * 50)
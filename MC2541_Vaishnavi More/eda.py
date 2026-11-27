import pandas as pd
import matplotlib.pyplot as plt


FILE = "data/fused_dataset.csv"

df = pd.read_csv(FILE)

print("======================================")
print("EXPLORATORY DATA ANALYSIS")
print("======================================")

print("\nDataset shape:")
print(df.shape)


# ======================================
# 1. Language Distribution
# ======================================

print("\nLanguage distribution:")
print(df["language"].value_counts())


plt.figure(figsize=(8, 5))

df["language"].value_counts().plot(
    kind="bar"
)

plt.title("Language Distribution")
plt.xlabel("Language")
plt.ylabel("Number of Videos")
plt.tight_layout()

plt.savefig(
    "data/language_distribution.png"
)

plt.close()


# ======================================
# 2. Video Duration Distribution
# ======================================

plt.figure(figsize=(8, 5))

df["duration_seconds"].plot(
    kind="hist",
    bins=30
)

plt.title("Video Duration Distribution")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of Videos")
plt.tight_layout()

plt.savefig(
    "data/video_duration_distribution.png"
)

plt.close()


# ======================================
# 3. Views Distribution
# ======================================

views = df["view_count"].dropna()

plt.figure(figsize=(8, 5))

views.plot(
    kind="hist",
    bins=30
)

plt.title("Video Views Distribution")
plt.xlabel("View Count")
plt.ylabel("Number of Videos")
plt.tight_layout()

plt.savefig(
    "data/view_distribution.png"
)

plt.close()


# ======================================
# 4. Likes vs Views
# ======================================

engagement = df[
    ["view_count", "like_count"]
].dropna()

plt.figure(figsize=(8, 5))

plt.scatter(
    engagement["view_count"],
    engagement["like_count"]
)

plt.title("Likes vs Views")
plt.xlabel("Views")
plt.ylabel("Likes")
plt.tight_layout()

plt.savefig(
    "data/likes_vs_views.png"
)

plt.close()


# ======================================
# 5. Comments vs Views
# ======================================

engagement = df[
    ["view_count", "comment_count"]
].dropna()

plt.figure(figsize=(8, 5))

plt.scatter(
    engagement["view_count"],
    engagement["comment_count"]
)

plt.title("Comments vs Views")
plt.xlabel("Views")
plt.ylabel("Comments")
plt.tight_layout()

plt.savefig(
    "data/comments_vs_views.png"
)

plt.close()


# ======================================
# 6. API Enrichment
# ======================================

api_available = df["api_title"].notna()

enrichment = pd.Series(
    {
        "API Matched": api_available.sum(),
        "API Not Matched": (~api_available).sum()
    }
)

print("\nAPI enrichment:")
print(enrichment)


plt.figure(figsize=(7, 5))

enrichment.plot(
    kind="bar"
)

plt.title("API Data Enrichment")
plt.xlabel("Status")
plt.ylabel("Number of Videos")
plt.tight_layout()

plt.savefig(
    "data/api_enrichment.png"
)

plt.close()


print("\n======================================")
print("EDA COMPLETED")
print("======================================")

print("\nGraphs saved in data/ folder.")
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the fused dataset
df = pd.read_csv("data/fused_dataset.csv")

# Create a folder for graphs
os.makedirs("outputs/eda", exist_ok=True)

sns.set_theme(style="whitegrid")

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# 1. Missing values
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]

if not missing.empty:
    plt.figure(figsize=(10, 5))
    missing.plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/eda/missing_values.png")
    plt.close()

# 2. Video language distribution
plt.figure(figsize=(10, 5))
df["language"].fillna("Unknown").value_counts().head(15).plot(kind="bar")
plt.title("Top 15 Video Languages")
plt.xlabel("Language")
plt.ylabel("Number of Videos")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/eda/language_distribution.png")
plt.close()

# 3. Video duration distribution
duration = pd.to_numeric(df["duration_seconds"], errors="coerce").dropna()
duration = duration[duration >= 0]

plt.figure(figsize=(8, 5))
plt.hist(duration, bins=30, edgecolor="black")
plt.title("Video Duration Distribution")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of Videos")
plt.tight_layout()
plt.savefig("outputs/eda/duration_distribution.png")
plt.close()

# 4. View count distribution
views = pd.to_numeric(df["view_count"], errors="coerce").dropna()
views = views[views >= 0]

plt.figure(figsize=(8, 5))
plt.hist(views, bins=30, edgecolor="black")
plt.xscale("log")
plt.title("Video View Count Distribution (Log Scale)")
plt.xlabel("Views (log scale)")
plt.ylabel("Number of Videos")
plt.tight_layout()
plt.savefig("outputs/eda/views_distribution.png")
plt.close()

# 5. Views vs likes
plot_df = df[["view_count", "like_count"]].apply(
    pd.to_numeric, errors="coerce"
).dropna()

plot_df = plot_df[
    (plot_df["view_count"] >= 0) &
    (plot_df["like_count"] >= 0)
]

plt.figure(figsize=(8, 5))
plt.scatter(
    plot_df["view_count"],
    plot_df["like_count"],
    alpha=0.6
)
plt.xscale("symlog")
plt.yscale("symlog")
plt.title("Views vs Likes")
plt.xlabel("Views")
plt.ylabel("Likes")
plt.tight_layout()
plt.savefig("outputs/eda/views_vs_likes.png")
plt.close()

# Save missing-value report
missing.to_csv("outputs/eda/missing_values_report.csv", header=["missing_count"])

print("\nEDA completed successfully!")
print("Graphs and report saved in outputs/eda/")

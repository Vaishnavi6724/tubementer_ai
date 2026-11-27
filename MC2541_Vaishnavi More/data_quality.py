import pandas as pd


FILE = "data/fused_dataset.csv"

df = pd.read_csv(FILE)

print("======================================")
print("DATA QUALITY ANALYSIS")
print("======================================")

print("\nDataset shape:")
print(df.shape)


# 1. Duplicate check
duplicates = df["video_id"].duplicated().sum()

print("\nDuplicate video IDs:")
print(duplicates)


# 2. Missing values
print("\nMissing values:")
print(df.isnull().sum())


# 3. Missing percentage
print("\nMissing percentage:")
missing_percentage = (
    df.isnull().mean() * 100
).round(2)

print(missing_percentage)


# 4. API enrichment
api_matched = df["api_title"].notna().sum()
api_missing = df["api_title"].isna().sum()

print("\nAPI enrichment:")
print("Matched records:", api_matched)
print("Unmatched records:", api_missing)


# 5. Numeric columns
print("\nNumeric column statistics:")

numeric_columns = [
    "view_count",
    "like_count",
    "comment_count"
]

print(df[numeric_columns].describe())


print("\n======================================")
print("DATA QUALITY ANALYSIS COMPLETED")
print("======================================")
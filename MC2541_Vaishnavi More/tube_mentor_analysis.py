import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_FILE = "data/fused_dataset.csv"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 2. LOAD FUSED CSV DATASET
# ==================================================

df = pd.read_csv(DATA_FILE)

print("\n===== TUBEMENTOR AI DATASET =====")
print("Dataset loaded successfully!")
print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

# ==================================================
# 3. DATA QUALITY AND FUSION ANALYSIS
# ==================================================

print("\n===== DATA QUALITY =====")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("\nMissing values per column:")
print(df.isnull().sum())

# Save data quality report
quality_report = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(dtype) for dtype in df.dtypes],
    "Missing_Values": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().mean().values * 100
    ).round(2),
    "Unique_Values": df.nunique().values
})

quality_report.to_csv(
    f"{OUTPUT_DIR}/data_quality_report.csv",
    index=False
)

# Remove exact duplicate records
df = df.drop_duplicates().copy()

# Save the cleaned fused dataset
df.to_csv(
    f"{OUTPUT_DIR}/cleaned_fused_dataset.csv",
    index=False
)

# Graph 1: Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    plt.figure(figsize=(10, 6))
    missing.plot(kind="bar")
    plt.title("TubeMentor AI - Missing Values")
    plt.xlabel("Columns")
    plt.ylabel("Missing Value Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/missing_values.png")
    plt.show()

# Graph 2: Dataset information
plt.figure(figsize=(8, 5))
plt.bar(
    ["Rows", "Columns", "Duplicates Removed"],
    [
        len(df),
        len(df.columns),
        0  # Cleaned dataset has no exact duplicate rows
    ]
)
plt.title("TubeMentor AI Dataset Overview")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dataset_overview.png")
plt.show()

# Graph 3: Numeric feature distributions
numeric_df = df.select_dtypes(include=np.number)

if not numeric_df.empty:
    numeric_df.hist(figsize=(12, 8), bins=15)
    plt.suptitle("TubeMentor AI - Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/numeric_distributions.png")
    plt.show()

# Graph 4: Correlation matrix
if numeric_df.shape[1] >= 2:
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=45,
        ha="right"
    )
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("TubeMentor AI - Feature Correlation")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
    plt.show()

# ==================================================
# 4. CLASSIFICATION TARGET
# ==================================================

# IMPORTANT:
# Change TARGET_COLUMN to a real column in your dataset.
# It must represent the category you want to predict.
# Example: video_category, topic, or difficulty_level.

TARGET_COLUMN = "video_category"

if TARGET_COLUMN not in df.columns:
    print("\n===== ML MODELS NOT RUN =====")
    print(f"Target column '{TARGET_COLUMN}' was not found.")
    print("Available columns:", df.columns.tolist())
    print(
        "Change TARGET_COLUMN in this script to a suitable "
        "categorical column from your dataset."
    )
    raise SystemExit(0)

# Remove rows without a target value
model_df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Exclude identifiers and free-form text from basic
# classification to avoid learning from IDs or long transcripts.
# Add other ID/target-leakage columns here if necessary.
exclude_columns = [
    TARGET_COLUMN,
    "video_id",
    "channel_id",
    "video_url",
    "transcript",
    "description",
    "title"
]

X = model_df.drop(
    columns=[c for c in exclude_columns if c in model_df.columns]
)

y_raw = model_df[TARGET_COLUMN].astype(str)

# Encode class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

if len(np.unique(y)) < 2:
    print("ML cannot run: target has fewer than two classes.")
    raise SystemExit(0)

# Very small classes cannot be split reliably
class_counts = pd.Series(y).value_counts()
if class_counts.min() < 2:
    print(
        "ML cannot run safely: at least one target class "
        "has fewer than two records."
    )
    raise SystemExit(0)

# Remove columns with no useful variation
X = X.loc[:, X.nunique(dropna=False) > 1]

# Exclude free-form text columns from this simple ML baseline
text_columns = [
    c for c in X.columns
    if X[c].dtype == "object"
    and X[c].nunique() > min(50, max(10, len(X) // 2))
]
X = X.drop(columns=text_columns)

if X.empty:
    print("No usable feature columns remain for ML.")
    raise SystemExit(0)

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

transformers = []

if numeric_features:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)

# Stratification preserves class proportions where possible
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ==================================================
# 5. TRAIN THREE CLASSIFICATION ALGORITHMS
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
}

results = []
trained_models = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append({
            "Algorithm": name,
            "Accuracy (%)": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
            "Precision (weighted)": round(
                precision_score(
                    y_test, predictions,
                    average="weighted", zero_division=0
                ), 3
            ),
            "Recall (weighted)": round(
                recall_score(
                    y_test, predictions,
                    average="weighted", zero_division=0
                ), 3
            ),
            "F1 Score (weighted)": round(
                f1_score(
                    y_test, predictions,
                    average="weighted", zero_division=0
                ), 3
            )
        })

        trained_models[name] = pipeline

        print(f"\n{name}")
        print("Accuracy:", round(
            accuracy_score(y_test, predictions) * 100, 2
        ), "%")

        # Confusion matrix for each model
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=label_encoder.classes_,
            xticks_rotation=45
        )
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(
            f"{OUTPUT_DIR}/{safe_name}_confusion_matrix.png"
        )
        plt.show()

    except Exception as error:
        print(f"\n{name} could not run: {error}")

# ==================================================
# 6. ALGORITHM COMPARISON GRAPH
# ==================================================

if results:
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        f"{OUTPUT_DIR}/algorithm_comparison.csv",
        index=False
    )

    print("\n===== ALGORITHM COMPARISON =====")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.bar(
        results_df["Algorithm"],
        results_df["Accuracy (%)"]
    )
    plt.title("TubeMentor AI - Classification Accuracy")
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/classification_accuracy.png"
    )
    plt.show()

# ==================================================
# 7. DECISION TREE VISUALIZATION
# ==================================================

if "Decision Tree" in trained_models:
    tree_pipeline = trained_models["Decision Tree"]
    tree_model = tree_pipeline.named_steps["classifier"]
    transformed_X = tree_pipeline.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    plt.figure(figsize=(18, 10))
    plot_tree(
        tree_model,
        feature_names=transformed_X,
        class_names=[str(c) for c in label_encoder.classes_],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7
    )
    plt.title("TubeMentor AI - Decision Tree")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/decision_tree.png")
    plt.show()

# ==================================================
# 8. FINISH
# ==================================================

print("\n===== TUBEMENTOR AI ANALYSIS COMPLETED =====")
print(f"Check the '{OUTPUT_DIR}' folder for reports and graphs.")




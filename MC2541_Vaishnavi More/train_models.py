
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load dataset
df = pd.read_csv("data/fused_dataset.csv")

# 2. Keep rows with known view counts
df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce")
df = df.dropna(subset=["view_count"]).copy()

# 3. Create a popularity label using the median view count
median_views = df["view_count"].median()
df["popularity"] = (df["view_count"] > median_views).astype(int)

# 4. Combine title and transcript as text features
df["model_text"] = (
    df["title"].fillna("") + " " +
    df["clean_text"].fillna("")
).str.strip()

df = df[df["model_text"].str.len() > 0].copy()

X = df["model_text"]
y = df["popularity"]

print("Videos used:", len(df))
print("Median view count:", median_views)
print("Class counts:\n", y.map({
    0: "Low/Median Popularity",
    1: "High Popularity"
}).value_counts())

# Ensure both classes can be stratified
if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError("Not enough examples in each popularity class.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF converts text into numerical features
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, stop_words=None)),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, stop_words=None)),
        ("model", MultinomialNB())
    ]),
    "Decision Tree": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words=None)),
        ("model", DecisionTreeClassifier(
            max_depth=15,
            random_state=42,
            class_weight="balanced"
        ))
    ])
}

os.makedirs("outputs/models", exist_ok=True)
results = []

# 5. Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Low/Median", "High"],
        zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Low/Median", "High"],
        values_format="d"
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_")
    plt.savefig(f"outputs/models/{filename}_confusion_matrix.png")
    plt.close()

# 6. Compare model accuracy
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/models/model_results.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.ylim(0, 1)
plt.ylabel("Test Accuracy")
plt.title("TubeMentor AI - Model Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/models/model_comparison.png")
plt.close()

print("\nResults:")
print(results_df.to_string(index=False))
print("\nSaved model metrics and graphs in outputs/models/")
